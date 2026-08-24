# agent-serving-sim

[![CI](https://github.com/Naloam/agent-serving-sim/actions/workflows/ci.yml/badge.svg)](https://github.com/Naloam/agent-serving-sim/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)

Agent 负载推理服务模拟器：输入 LLM agent 会话 trace，模拟 KV cache、调度与淘汰策略，输出 JCT / 缓存命中率 / 显存占用，用于研究 agent 负载下的推理服务优化。

Agent 负载与传统 chat 负载的差异是本项目的出发点：多轮长会话、轮间长空闲（agent 在"思考"或执行工具）、system prompt 与工具定义高度重复、会话历史逐轮增长、多 agent 并发互相挤占 cache——现有推理服务的调度策略并非为此设计。模拟器在 CPU 上以纯 Python 复现这些结构（radix tree 前缀缓存、可插拔驱逐、vLLM 式抢占、驱逐成本），10 万请求端到端约 19 秒。English docs: [README_EN.md](./README_EN.md)。

## 快速开始

```bash
python -m venv .venv            # venv 落在项目根
.venv\Scripts\activate          # Windows；Unix 用 source .venv/bin/activate
pip install -e ".[dev]"         # 主包零第三方依赖；dev = pytest
python -m pytest                # 全量单测
```

## 运行实验

需要可视化则安装 `pip install -e ".[dev,viz]"`（numpy / pandas / matplotlib）。产出（汇总表 CSV/JSON、CDF、扫描曲线、显存时间线）写入 `experiments/results/`。

```bash
python experiments/exp001_lru_vs_ttl.py --seed 42                # LRU vs TTL 基线
python experiments/exp002_ttl_sweep_heterogeneous.py --seed 42   # 异构负载 TTL 扫描
python experiments/exp003_priority_eviction.py --seed 42         # 带权 LRU 优先级扫描
python experiments/exp004_multi_agent_quota.py --seed 42         # 多 agent 配额
python experiments/exp005_real_trace_replay.py                   # 真实 trace 重放
python experiments/exp006_preemption_and_eviction_cost.py        # 抢占与驱逐成本
python experiments/exp007_belady_upper_bound.py --seed 42        # Belady 理论上限
python experiments/exp008_predictive_policy.py --seed 42         # 预测型在线驱逐
```

库方式使用：

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

自定义驱逐策略：继承 `ass.cache.policies.EvictionPolicy` 并用 `register_policy` 装饰，`create_policy(name, **kwargs)` 实例化，不需要改动内核。

## 真实 trace

仓库附带 1360 个请求 / 224 个会话的真实 agent trace（coding + search 两类，基于 Ollama/qwen2.5-coder 采集），含负载刻画报告与计时标定（`traces/real/`）。采集管线开放：任何 OpenAI 兼容后端都可复用。

```bash
python experiments/collect_real_trace.py    # 探针 + 两类 agent 驱动 → 原始 JSONL
python experiments/analyze_real_trace.py    # 清洗入库 + 负载刻画 + 计时标定
```

## License

MIT
