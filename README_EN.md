# agent-serving-sim

[![CI](https://github.com/Naloam/agent-serving-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/Naloam/agent-serving-sim/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

A discrete-event simulator for LLM **agent serving workloads**: feed in agent session
traces (synthetic or real), simulate KV cache / scheduling / eviction policies on a CPU,
and get JCT, prefix hit-rate, and cache-occupancy timelines out. Built to study how
multi-turn, long-idle, prefix-heavy agent traffic breaks the assumptions behind
vLLM/SGLang-style schedulers.

> 中文文档见 [README.md](./README.md)。English docs live here.

## Why

Agent traffic is not chat traffic:

- **Multi-turn sessions** with long idle gaps between turns (the agent is "thinking"
  or running tools) — the core tension is whether to *keep* a session's KV cache
  across turns;
- **Highly repetitive prefixes** (system prompt + tool definitions) shared across
  sessions of the same app;
- **Monotonically growing histories** that dominate the prompt (~60% in our collected
  trace);
- **Multi-agent co-tenancy** competing for cache capacity.

Papers in this space (Continuum/CacheTTL, KVFlow, TokenCake, ForeCache) rarely release
simulators or traces. This project provides both, reproducibly.

## What's inside

| Piece | Description |
|-------|-------------|
| `ass/core` | Discrete-event kernel: `Event` ordered by `(time, priority, seq)`, heapq loop with lazy cancel and `run(until)` |
| `ass/workload` | Trace schema (JSONL, four-segment prompt breakdown), reproducible synthetic generator (Poisson arrivals, lognormal think-time, per-agent-type profiles), probe-log parser |
| `ass/cache` | Radix tree over position-aligned `Segment(stream, length)` prefixes — token-exact prefix semantics without per-token cost; pluggable eviction: FIFO / LRU / TTL (proactive sweep + LRU fallback) / Priority / Weighted-LRU / Quota, via a registry |
| `ass/scheduler` | Open-loop serving: arrival → pinned prefix match → evict-if-needed → chunked decode KV growth → preemption (vLLM-style recompute) with configurable eviction cost |
| `ass/metrics` | JCT / TTFT / hit-rate / occupancy timeline / eviction & preemption stats, CSV + JSON export |
| `ass/probe` | OpenAI-compatible recording proxy (stdlib only, transparent passthrough) |
| `experiments/` | exp001–exp006: policy comparisons from first figure to preemption economics |
| `traces/real/` | 1021 requests / 162 sessions of real coding + search agent traffic (collected against Ollama/qwen2.5-coder), with characterization report and timing calibration |

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; source .venv/bin/activate on Unix
pip install -e ".[dev]"         # core package has zero third-party dependencies
python -m pytest                # full test suite
```

Experiments (needs `pip install -e ".[dev,viz]"` for figures):

```bash
python experiments/exp001_lru_vs_ttl.py --seed 42                # M1: first figure
python experiments/exp002_ttl_sweep_heterogeneous.py --seed 42   # TTL sweep, heterogeneous load
python experiments/exp003_priority_eviction.py --seed 42         # weighted-LRU priority sweep
python experiments/exp004_multi_agent_quota.py --seed 42         # per-agent-type quotas
python experiments/exp005_real_trace_replay.py                   # replay the bundled real trace
python experiments/exp006_preemption_and_eviction_cost.py        # TTL wins once eviction costs
```

Library usage:

```python
from ass.cache.policies import TTLPolicy
from ass.scheduler.serving import ServingConfig, ServingSim
from ass.workload.synthetic import SyntheticConfig, generate_trace

trace = generate_trace(SyntheticConfig(num_sessions=300), seed=42)
sim = ServingSim(ServingConfig(cache_capacity_tokens=100_000), policy=TTLPolicy(ttl=20))
sim.submit_all(trace)
sim.run()
print(sim.collector.summary())
```

Collecting your own real trace (any OpenAI-compatible backend; we used Ollama):

```bash
python experiments/collect_real_trace.py     # probe + two agent drivers → raw JSONL
python experiments/analyze_real_trace.py     # clean → FR-2 traces + report + calibration
```

## Headline findings

1. **With free eviction, TTL never beats LRU** (provably: the expired set is a suffix
   of LRU order). Its value is an *operating-point knee*: set TTL between the fast and
   slow class cycles to keep the fast class at ~99% of LRU's hit rate while
   continuously releasing dead cache.
2. **Priority eviction is an SLO-reallocation tool, not a mean-JCT win**: weighted-LRU
   improves the protected class's p95 by up to 12% at a controlled cost to the other.
3. **Quotas are insurance**: indistinguishable from LRU absent cross-class
   encroachment.
4. **Eviction cost flips the TTL story** (exp006): with on-path eviction at 2k tok/s,
   TTL beats LRU by 22% mean JCT and 56% p95 — proactive off-path expiry is exactly
   the mechanism Continuum advocates. Verified in direction on the real trace, with
   gains proportional to eviction traffic.
5. **Belady upper bound** (exp007): LRU sits 16–25% below offline-optimal hit rate,
   and the gap is concentrated in the slow-turn class (fast class already
   near-optimal) — quantifying where future online policies should aim.

See `blog/` (Chinese) for the full write-ups; `PLAN.md §8` has the complete decision log.

## Status

- [x] M0 bootstrap · [x] M1 core simulator · [x] M2 real-trace collection
- [x] M3 policy research · [x] M3.5 preemption & eviction-cost modeling
- [x] M4 open-source release (v0.1.0) — awesome-list submissions and upstream PRs ongoing
