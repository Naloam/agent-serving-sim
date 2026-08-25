# agent-serving-sim

[![CI](https://github.com/Naloam/agent-serving-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/Naloam/agent-serving-sim/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

A discrete-event simulator for LLM agent serving workloads: feed in agent session traces (synthetic or real), simulate KV cache, scheduling and eviction policies on a CPU, and get JCT, prefix hit-rate, and cache-occupancy timelines out.

The starting point is how agent traffic differs from chat: long multi-turn sessions, long idle gaps between turns (the agent is "thinking" or running tools), highly repetitive system prompts and tool definitions, monotonically growing histories, and multi-agent co-tenancy competing for cache — assumptions that today's serving schedulers were not designed for. The simulator reproduces this structure in pure Python on a CPU (radix-tree prefix cache, pluggable eviction, vLLM-style preemption, eviction cost); 100k requests simulate end to end in about 19 seconds. 中文文档：[README.md](./README.md).

## Quick start

```bash
python -m venv .venv            # venv lives in the project root
.venv\Scripts\activate          # Windows; use `source .venv/bin/activate` on Unix
pip install -e ".[dev]"         # core package has zero third-party dependencies; dev = pytest
python -m pytest                # full test suite
```

## Running experiments

Figures and CSV/JSON summaries land in `experiments/results/`. Install `pip install -e ".[dev,viz]"` for plotting (numpy / pandas / matplotlib).

```bash
python experiments/exp001_lru_vs_ttl.py --seed 42                # LRU vs TTL baseline
python experiments/exp002_ttl_sweep_heterogeneous.py --seed 42   # TTL sweep on mixed fast/slow agents
python experiments/exp003_priority_eviction.py --seed 42         # weighted-LRU priority sweep (SLO tail trade-offs)
python experiments/exp004_multi_agent_quota.py --seed 42         # per-agent-type cache quotas
python experiments/exp005_real_trace_replay.py                   # replay the bundled real trace
python experiments/exp006_preemption_and_eviction_cost.py        # preemption + when eviction cost flips TTL vs LRU
python experiments/exp007_belady_upper_bound.py --seed 42        # Belady bound: how far LRU is from optimal
python experiments/exp008_predictive_policy.py --seed 42         # online predictive eviction closing that gap
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

To add your own eviction policy: subclass `ass.cache.policies.EvictionPolicy`, decorate it with `register_policy`, and instantiate via `create_policy(name, **kwargs)` — no kernel changes needed.

## Real traces

The repo bundles 1,360 requests / 224 sessions of real agent traffic (coding + search agents, collected against Ollama/qwen2.5-coder), with a workload characterization report and timing calibration under `traces/real/`. The collection pipeline is open and works against any OpenAI-compatible backend:

```bash
python experiments/collect_real_trace.py    # probe + two agent drivers -> raw JSONL
python experiments/analyze_real_trace.py    # clean into traces + characterize + calibrate
```

## License

MIT
