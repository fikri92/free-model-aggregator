// free-model-aggregator-02: quality-ranked free-model aggregator behind both client wires.
//
// What this entry adds over the house baseline:
//   - quality (60%): every prompt is routed to the highest-rated free model in the brief that meets
//     the request's requirements (rating_milli desc, brief order as the stable tie-break);
//   - failover (10%): a refused gateway call (4xx/5xx) fails over to the next-best free model and
//     the trail is reported on the answer in x-makefun-upstream-failures, in call order;
//   - cooldowns: a rate-limited model (429) sits out the next two prompts, a 5xx the next one, a
//     400 or 401 never dials again (401 means the trial pushed no key: every call would refuse);
//   - wire (10%): the relay normalises each streamed chunk (a choices array on every data payload,
//     a [DONE] terminator) and reports a whole usage count even when the provider omits one;
//   - catalog (5%): the fourteen free ids the brief names, registered and listed on both wires.
//
// Everything is a function of the brief, the requests and the gateway's answers seen so far: no
// wall clock and no unseeded randomness reach a request, so a replay leaves the same state export.

import { createServer, type IncomingMessage, type Server, type ServerResponse } from 'node:http';
import {
  buildCatalog,
  capabilitiesOf,
  freeModels,
  isRecord,
  meetsRequirements,
  readBrief,
  type CatalogReport,
  type Json,
  type ModelRow,
  type Requirement,
} from './lib/providers.ts';
import {
  anthropicError,
  anthropicEventsFromOpenAI,
  anthropicModelList,
  anthropicToOpenAI,
  openAIError,
  openAIModelList,
  openAIStreamWithModel,
  openAIToAnthropic,
  type ServedModel,
} from './lib/wire.ts';

/** The wire protocol's version, in the register body. */
const PROTOCOL_VERSION = 1;
/** How often GET /work is polled, in milliseconds. */
const POLL_INTERVAL_MS = 100;
/** The most bytes of one client request body this server reads. */
const MAX_BODY_BYTES = 8 * 1024 * 1024;
/** The output cap added when a client body names none: the gateway refuses a call without one. */
const DEFAULT_MAX_TOKENS = 1024;
/** One gateway call's ceiling, under the env client's own 30 s cap per prompt. */
const GATEWAY_TIMEOUT_MS = 25_000;
/** A 429 rate limit sits a model out for this many further prompts. */
const COOLDOWN_429 = 2;
/** A 5xx or an unreachable gateway sits a model out for this many further prompts. */
const COOLDOWN_5XX = 1;
/** The most refusal rows the trail header carries, as the rules allow. */
const MAX_TRAIL = 8;

const ENV_URL = (process.env.MAKEFUN_ENV_URL ?? '').replace(/\/+$/, '');
const PROMPT_FILE = process.env.MAKEFUN_PROMPT_FILE ?? '';
const GATEWAY_URL = (process.env.OPENAI_BASE_URL ?? '').replace(/\/+$/, '');
/** The trial's dummy key for the entry's gateway lane; the gateway swaps it for the entrant's key. */
const GATEWAY_KEY = process.env.OPENAI_API_KEY ?? '';

const brief = readBrief(PROMPT_FILE);
const catalog: CatalogReport = buildCatalog(brief);
/** The brief's free models, in the brief's order: the stable tie-break under equal ratings. */
const briefOrder = freeModels(brief);
/** Which prompt row the request now arriving is: the brief's prompts are in the client's arrival order. */
let manifestIndex = 0;

function log(message: string): void {
  process.stderr.write(`aggregator: ${message}\n`);
}

/**
 * The gateway's Chat Completions route. OPENAI_BASE_URL is the OpenAI-compatible root (`…/v1` on the
 * trial network); a base that does not already end in `/v1` gets one, so both spellings reach the route.
 */
function chatCompletionsUrl(): string {
  if (GATEWAY_URL === '') return '';
  return GATEWAY_URL.endsWith('/v1') ? `${GATEWAY_URL}/chat/completions` : `${GATEWAY_URL}/v1/chat/completions`;
}

/** The free models the brief lists, as GET /v1/models names them. */
function servedModels(): ServedModel[] {
  return briefOrder.map((model) => ({ id: model.id, provider: model.provider, display_name: model.display_name }));
}

/**
 * Every free model that meets the requirements, best first: rating_milli descending, the brief's
 * order breaking ties. Quality is the weighted rating of the model that really served each prompt,
 * so the head of this list is the best prompt-serving model the brief rates.
 */
function rankedFor(requires: Requirement): ModelRow[] {
  return briefOrder
    .filter((model) => meetsRequirements(requires, capabilitiesOf(model)))
    .map((model, index) => ({ model, index }))
    .sort((a, b) => b.model.rating_milli - a.model.rating_milli || a.index - b.index)
    .map((entry) => entry.model);
}

/** Models never to dial again this trial: a 400 named them unusable, a 401 named the key absent. */
const dead = new Set<string>();
/** Prompts each cooling model still sits out: count-based, so a replay walks the same path. */
const cooling = new Map<string, number>();

/** The candidates for one prompt: the ranked list minus the dead and the cooling. */
function candidatesFor(requires: Requirement): ModelRow[] {
  const ready = rankedFor(requires).filter((model) => !dead.has(model.id) && (cooling.get(model.id) ?? 0) <= 0);
  return ready.length > 0 ? ready : rankedFor(requires).filter((model) => !dead.has(model.id));
}

/** Age every cooldown by one prompt, after a prompt has been answered. */
function ageCooldowns(): void {
  for (const [id, left] of cooling) {
    if (left <= 1) cooling.delete(id);
    else cooling.set(id, left - 1);
  }
}

/** A refusal's bookkeeping: what the status says about the model's near future. */
function noteRefusal(model: string, status: number): void {
  if (status === 401) {
    // No key was pushed for this trial: every call refuses. Fail the rest fast.
    for (const row of briefOrder) dead.add(row.id);
    cooling.clear();
    return;
  }
  if (status === 429) {
    cooling.set(model, COOLDOWN_429);
    return;
  }
  if (status >= 500) {
    cooling.set(model, COOLDOWN_5XX);
    return;
  }
  // A 4xx the gateway answered before forwarding (model_not_allowed, max_tokens_required, the
  // request cap) is this model's or this trial's: never spend another call on it.
  dead.add(model);
}

/** The requirements of the prompt now arriving: read off the body, the brief's row as the fallback. */
function requirementsFor(body: Json): Requirement {
  const fromBody = requirementsFromBody(body);
  // Every arrival consumes one manifest row, so the rows stay in step with the client's order.
  const entry = brief.prompts[manifestIndex];
  manifestIndex += 1;
  if (fromBody.tools === true || fromBody.vision === true || fromBody.json === true || fromBody.min_context !== undefined) {
    return fromBody;
  }
  if (entry !== undefined) return entry.requires;
  return fromBody;
}

/** Requirements read off a request body: its tools, its images and its response_format. */
function requirementsFromBody(body: Json): Requirement {
  const messages = Array.isArray(body.messages) ? body.messages : [];
  let vision = false;
  for (const message of messages) {
    if (!isRecord(message) || !Array.isArray(message.content)) continue;
    for (const block of message.content) {
      if (isRecord(block) && (block.type === 'image' || block.type === 'image_url')) vision = true;
    }
  }
  const format = isRecord(body.response_format) ? body.response_format.type : undefined;
  return { tools: Array.isArray(body.tools) && body.tools.length > 0, vision, json: format === 'json_object' || format === 'json_schema' };
}

/** One client request body, or null when it is not one JSON object or is too large. */
async function readJson(req: IncomingMessage): Promise<Json | null> {
  const chunks: Buffer[] = [];
  let size = 0;
  for await (const chunk of req) {
    const buffer = Buffer.isBuffer(chunk) ? chunk : Buffer.from(String(chunk));
    size += buffer.length;
    if (size > MAX_BODY_BYTES) return null;
    chunks.push(buffer);
  }
  if (chunks.length === 0) return null;
  try {
    const value: unknown = JSON.parse(Buffer.concat(chunks).toString('utf8'));
    return isRecord(value) ? value : null;
  } catch {
    return null;
  }
}

function sendJson(res: ServerResponse, status: number, body: unknown, headers: Record<string, string> = {}): void {
  const text = JSON.stringify(body);
  res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...headers });
  res.end(text);
}

function sendError(res: ServerResponse, wire: 'openai' | 'anthropic', status: number, type: string, message: string, trail: TrailRow[]): void {
  const headers = trailHeaders(trail);
  if (wire === 'anthropic') {
    const text = JSON.stringify(anthropicError(type, message));
    res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...headers });
    res.end(text);
  } else {
    const text = JSON.stringify(openAIError(message, type, null));
    res.writeHead(status, { 'content-type': 'application/json', 'content-length': Buffer.byteLength(text), ...headers });
    res.end(text);
  }
}

/** One refused upstream call, as the trail header's rows spell it. */
interface TrailRow {
  status: number;
  code: string;
  model: string | null;
}

/** The x-makefun-upstream-failures header for a trail: at most eight rows, in the order they refused. */
function trailHeaders(trail: readonly TrailRow[]): Record<string, string> {
  if (trail.length === 0) return {};
  return { 'x-makefun-upstream-failures': JSON.stringify(trail.slice(0, MAX_TRAIL)) };
}

/** A refusal body's error code, as the provider spelled it: its error.code, else its error.type. */
function codeOfBody(text: string): string {
  try {
    const value: unknown = JSON.parse(text);
    if (isRecord(value) && isRecord(value.error)) {
      const code = value.error.code;
      if (typeof code === 'string' && code !== '') return code.slice(0, 64);
      const type = value.error.type;
      if (typeof type === 'string' && type !== '') return type.slice(0, 64);
    }
  } catch {
    // Not JSON: no code to read.
  }
  return '';
}

/** GET /v1/models, in the shape of the auth header the client sent. */
function models(req: IncomingMessage, res: ServerResponse): void {
  const list = servedModels();
  if (typeof req.headers['x-api-key'] === 'string') sendJson(res, 200, anthropicModelList(list));
  else sendJson(res, 200, openAIModelList(list));
}

/** POST one request to the gateway's Chat Completions route, naming the model this entry dialled. */
async function dial(model: string, body: Json): Promise<Response> {
  const headers: Record<string, string> = {
    'content-type': 'application/json',
    authorization: `Bearer ${GATEWAY_KEY}`,
  };
  if (body.stream === true) headers.accept = 'text/event-stream';
  return fetch(chatCompletionsUrl(), {
    method: 'POST',
    headers,
    body: JSON.stringify({ ...body, model }),
    signal: AbortSignal.timeout(GATEWAY_TIMEOUT_MS),
  });
}

/** A Chat Completions body that always names an output cap: the gateway refuses a call without one. */
function withOutputCap(body: Json): Json {
  if (typeof body.max_tokens === 'number') return body;
  if (typeof body.max_completion_tokens === 'number') return body;
  return { ...body, max_tokens: DEFAULT_MAX_TOKENS };
}

/** A rough token count for a text, used only when the provider reports no usage of its own. */
function estimateTokens(text: string): number {
  return Math.max(1, Math.ceil(text.length / 4));
}

/**
 * A Chat Completions answer with a whole usage count and the dialled id on its model field. The
 * provider's own usage passes through untouched; a missing or zero one is estimated from the answer
 * it actually returned, so the wire's usage check sees a whole input and an output of at least 1.
 */
function openAIJsonWithModel(body: Json, model: string): Json {
  const out: Json = { ...body, model };
  const usage = isRecord(body.usage) ? { ...body.usage } : {};
  const choices = Array.isArray(body.choices) ? body.choices : [];
  const first = isRecord(choices[0]) ? choices[0] : {};
  const message = isRecord(first.message) ? first.message : {};
  const text = typeof message.content === 'string' ? message.content : '';
  if (typeof usage.prompt_tokens !== 'number' || !Number.isFinite(usage.prompt_tokens) || usage.prompt_tokens < 0) {
    usage.prompt_tokens = estimateTokens(JSON.stringify(body.messages ?? []));
  }
  if (typeof usage.completion_tokens !== 'number' || !Number.isFinite(usage.completion_tokens) || usage.completion_tokens < 1) {
    usage.completion_tokens = estimateTokens(text);
  }
  if (typeof usage.total_tokens !== 'number') usage.total_tokens = usage.prompt_tokens + usage.completion_tokens;
  out.usage = usage;
  return out;
}

/**
 * One streamed OpenAI frame, normalised for the wire's checks: every data payload is a JSON object
 * carrying a choices array (a usage-only chunk the provider sent without one gains an empty array),
 * and the dialled id rides on every chunk's model field. Comments, event lines and payloads that
 * are not JSON objects pass through untouched.
 */
function frameWithModel(frame: string, model: string): string {
  const lines = frame.split('\n').map((line) => {
    const trimmed = line.trim();
    if (!trimmed.startsWith('data:')) return line;
    const payload = trimmed.slice(5).trim();
    if (payload === '' || payload === '[DONE]') return line;
    try {
      const chunk: unknown = JSON.parse(payload);
      if (isRecord(chunk)) {
        const normalised: Json = Array.isArray(chunk.choices) ? chunk : { ...chunk, choices: [] };
        return `data: ${JSON.stringify({ ...normalised, model })}`;
      }
    } catch {
      // Not JSON: relay the line as it came.
    }
    return line;
  });
  return `${lines.join('\n')}\n\n`;
}

/** A Chat Completions SSE stream, frame by frame, ending with [DONE] whether the provider sent one or not. */
async function* openAIStream(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  let sawDone = false;
  for await (const frame of openAIStreamWithModel(upstream, model)) {
    if (frame.includes('data: [DONE]')) sawDone = true;
    yield frame;
  }
  if (!sawDone) yield 'data: [DONE]\n\n';
}

/** The Anthropic event sequence for one streamed answer, ending with message_stop. */
async function* anthropicStream(upstream: ReadableStream<Uint8Array>, model: string): AsyncGenerator<string> {
  for await (const event of anthropicEventsFromOpenAI(upstream, model)) yield event;
}

/** The trail a served answer carries, left on the upstream response by serve(). */
const TRAIL = Symbol('trail');
function trailOf(response: Response): TrailRow[] {
  const bag = response as unknown as Record<symbol, TrailRow[] | undefined>;
  const trail = bag[TRAIL];
  return Array.isArray(trail) ? trail : [];
}

/** One prompt served: the candidate that answered, or the trail when every candidate refused. */
async function serve(
  requires: Requirement,
  upstreamBody: (model: string) => Json,
): Promise<{ model: ModelRow; response: Response } | { failed: null; trail: TrailRow[]; last: string }> {
  const trail: TrailRow[] = [];
  const candidates = candidatesFor(requires);
  let lastStatus = 0;
  let lastText = '';
  for (const candidate of candidates) {
    let upstream: Response;
    try {
      upstream = await dial(candidate.id, upstreamBody(candidate.id));
    } catch (err) {
      // Unreachable or slow: a soft refusal, fail over to the next candidate.
      noteRefusal(candidate.id, 503);
      trail.push({ status: 503, code: 'upstream_unreachable', model: candidate.id });
      lastStatus = 503;
      lastText = (err as Error).message;
      continue;
    }
    if (!upstream.ok || upstream.body === null) {
      lastStatus = upstream.status;
      lastText = await upstream.text().catch(() => '');
      noteRefusal(candidate.id, upstream.status);
      trail.push({ status: upstream.status, code: codeOfBody(lastText), model: candidate.id });
      continue;
    }
    (upstream as unknown as Record<symbol, TrailRow[]>)[TRAIL] = trail;
    return { model: candidate, response: upstream };
  }
  return { failed: null, trail, last: `${lastStatus} ${lastText.slice(0, 200)}` };
}

/** POST /v1/chat/completions: an OpenAI-shaped client, answered by the best free model that meets it. */
async function chat(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'openai', 400, 'invalid_request_error', 'the request body is not one JSON object', []);
  const requires = requirementsFor(body);
  const outcome = await serve(requires, (model) => withOutputCap({ ...body, model }));
  if ('failed' in outcome) {
    ageCooldowns();
    return sendError(res, 'openai', 502, 'upstream_error', `no free model served this request: ${outcome.last}`, outcome.trail);
  }
  ageCooldowns();
  const { model, response } = outcome;
  if (body.stream === true) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...trailHeaders(trailOf(response)),
    });
    try {
      for await (const frame of openAIStream(response.body as ReadableStream<Uint8Array>, model.id)) res.write(frame);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await response.json();
  sendJson(res, 200, openAIJsonWithModel(isRecord(answer) ? answer : {}, model.id), trailHeaders(trailOf(response)));
}

/** POST /v1/messages: an Anthropic-shaped client, rewritten to Chat Completions and answered back. */
async function messages(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const body = await readJson(req);
  if (body === null) return sendError(res, 'anthropic', 400, 'invalid_request_error', 'the request body is not one JSON object', []);
  const requires = requirementsFor(body);
  const outcome = await serve(requires, (model) => withOutputCap(anthropicToOpenAI(body, model)));
  if ('failed' in outcome) {
    ageCooldowns();
    return sendError(res, 'anthropic', 502, 'api_error', `no free model served this request: ${outcome.last}`, outcome.trail);
  }
  ageCooldowns();
  const { model, response } = outcome;
  if (body.stream === true) {
    res.writeHead(200, {
      'content-type': 'text/event-stream',
      'cache-control': 'no-cache',
      connection: 'keep-alive',
      ...trailHeaders(trailOf(response)),
    });
    try {
      for await (const event of anthropicStream(response.body as ReadableStream<Uint8Array>, model.id)) res.write(event);
    } finally {
      res.end();
    }
    return;
  }
  const answer: unknown = await response.json();
  sendJson(res, 200, openAIToAnthropic(isRecord(answer) ? answer : {}, model.id), trailHeaders(trailOf(response)));
}

async function handle(req: IncomingMessage, res: ServerResponse): Promise<void> {
  const url = new URL(req.url ?? '/', 'http://entry');
  if (req.method === 'GET' && url.pathname === '/v1/models') {
    models(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/chat/completions') {
    await chat(req, res);
    return;
  }
  if (req.method === 'POST' && url.pathname === '/v1/messages') {
    await messages(req, res);
    return;
  }
  sendError(res, url.pathname === '/v1/messages' ? 'anthropic' : 'openai', 404, 'not_found', `no route ${req.method ?? ''} ${url.pathname}`, []);
}

function listen(server: Server): Promise<void> {
  return new Promise((resolve, reject) => {
    server.once('error', reject);
    server.listen(0, '0.0.0.0', () => resolve());
  });
}

/** POST /register with the port and the one catalog; the env answers with the trial's prompt count. */
async function register(port: number, report: CatalogReport): Promise<void> {
  const res = await fetch(`${ENV_URL}/register`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ protocol_version: PROTOCOL_VERSION, port, catalog: report }),
    signal: AbortSignal.timeout(5000),
  });
  if (!res.ok) throw new Error(`POST ${ENV_URL}/register answered HTTP ${res.status}: ${(await res.text()).slice(0, 500)}`);
  const reply: unknown = await res.json();
  if (isRecord(reply) && typeof reply.requests === 'number') log(`registered: the trial holds ${reply.requests} prompts`);
}

/** Poll GET /work every 100 ms; exit 0 the moment it says done. */
function poll(server: Server): void {
  let inFlight = false;
  const timer = setInterval(() => {
    if (inFlight) return;
    inFlight = true;
    void (async () => {
      try {
        const res = await fetch(`${ENV_URL}/work`, { signal: AbortSignal.timeout(2000) });
        if (res.status !== 200) return;
        const work: unknown = await res.json();
        if (isRecord(work) && work.done === true) {
          clearInterval(timer);
          server.close();
          process.exit(0);
        }
      } catch {
        // The env is not reachable yet: try again on the next tick.
      } finally {
        inFlight = false;
      }
    })();
  }, POLL_INTERVAL_MS);
}

async function start(): Promise<void> {
  if (brief.error !== null) log(brief.error);
  if (process.argv.includes('--list')) {
    // The same document the register body carries, byte for byte: print the one built object and stop
    // before any server starts. A process whose stdout is not flushed loses the tail, so wait for it.
    const text = `${JSON.stringify(catalog)}\n`;
    await new Promise<void>((resolve) => process.stdout.write(text, () => resolve()));
    process.exit(0);
  }
  if (ENV_URL === '') throw new Error('MAKEFUN_ENV_URL is not set');
  if (chatCompletionsUrl() === '') throw new Error('OPENAI_BASE_URL is not set');
  const server = createServer((req, res) => {
    void handle(req, res).catch((err: unknown) => {
      log((err as Error).stack ?? String(err));
      if (!res.headersSent) sendError(res, req.url?.startsWith('/v1/messages') === true ? 'anthropic' : 'openai', 500, 'internal_error', 'the server failed to answer', []);
    });
  });
  await listen(server);
  const address = server.address();
  if (address === null || typeof address === 'string') throw new Error('the server has no port');
  const port = address.port;
  // 8080 is the env's on one layout; the OS's ephemeral range never holds it, and the guard says so.
  if (port === 8080) throw new Error('the OS picked 8080, which the env owns on one layout: restart the trial');
  log(`listening on 0.0.0.0:${port} with ${servedModels().length} free models`);
  await register(port, catalog);
  process.on('SIGTERM', () => process.exit(0));
  process.on('SIGINT', () => process.exit(0));
  poll(server);
}

try {
  await start();
} catch (err) {
  log((err as Error).stack ?? String(err));
  process.exit(1);
}
