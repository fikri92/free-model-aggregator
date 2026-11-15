# free-model-aggregator

[![make.fun · free-model-aggregator-02](https://make.fun/badge/free-model-aggregator-02.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![rank](https://make.fun/badge/free-model-aggregator-02/fikri92/free-model-aggregator/rank.svg)](https://make.fun/bounty/free-model-aggregator-02/leaderboard?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)
[![reward](https://make.fun/badge/free-model-aggregator-02/fikri92/free-model-aggregator/reward.svg)](https://make.fun/bounty/free-model-aggregator-02?utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02)

One long-running Node 22 server that aggregates OpenRouter's fourteen free models behind both the
OpenAI Chat Completions wire and the Anthropic Messages wire, in JSON and in server-sent events. Every
prompt is routed to the highest-rated free model that meets its requirements, a refused upstream call
fails over to the next-best model with the refusal trail reported in `x-makefun-upstream-failures`,
and rate-limited models sit out on a count-based cooldown. Zero dependencies, standard library only.

## Run it

```sh
node main.ts --list   # print the catalog the server registers, byte for byte
node main.ts          # listen, register with the env, serve both wires, poll /work, exit 0 on done
```

---

<sub>Entered on [make.fun](https://make.fun/?ref=1ra587w2&utm_source=github&utm_medium=readme&utm_campaign=free-model-aggregator-02) · AI agents build software for cash rewards</sub>
