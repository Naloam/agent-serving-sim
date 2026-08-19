# agent-serving-sim

Agent 负载推理服务模拟器：输入 LLM agent 会话 trace，模拟 KV cache、调度与淘汰策略，输出 JCT / 缓存命中率 / 显存占用，用于研究 agent 负载下的推理服务优化。

## 文档

| 文档 | 内容 |
|------|------|
| [PLAN.md](./PLAN.md) | 战略与架构：背景调研、设计决策、里程碑、风险 |
| [PRD.md](./PRD.md) | 需求与验收：功能需求（FR）、非功能需求、各里程碑 DoD |
| [AGENTS.md](./AGENTS.md) | 协作规范：环境、命令、代码/测试规范、工作流（AI 助手必读） |

## 快速开始

```bash
python -m venv .venv            # venv 落在项目根（D 盘）
.venv\Scripts\activate
pip install -e ".[dev]"         # 主包零第三方依赖；dev = pytest
python -m pytest                # 全量单测
```

运行实验（需要可视化则装 `pip install -e ".[dev,viz]"`）：

```bash
python experiments/exp001_lru_vs_ttl.py --seed 42                # M1: LRU vs TTL
python experiments/exp002_ttl_sweep_heterogeneous.py --seed 42   # M3: 异构 TTL 扫描
python experiments/exp003_priority_eviction.py --seed 42         # M3: 带权 LRU 权重扫描
python experiments/exp004_multi_agent_quota.py --seed 42         # M3: 多 agent 配额
python experiments/exp005_real_trace_replay.py                   # M3: 真实 trace 重放
# 产出：experiments/results/ 下的汇总表（CSV/JSON）、JCT CDF、扫描曲线、显存时间线图
```

真实 trace 采集管线（需本机 Ollama，模型 qwen2.5-coder-16k）：

```bash
python experiments/collect_real_trace.py          # 探针 + 两类 agent 驱动 → traces/real/raw/
python experiments/analyze_real_trace.py          # 清洗入库 + 负载刻画 + 计时标定
```

已有数据：`traces/real/`（1021 请求 / 162 会话，coding+search 两类，含刻画报告与标定）。

以库方式使用：

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

## 项目阶段

- [x] M0 起步：脚手架 + 论文阅读
- [x] M1 核心模拟器：事件循环 + radix cache + LRU/TTL + 第一个实验
- [x] M2 真实 trace：本地推理服务 + agent 采集（Ollama 方案）
- [x] M3 策略研究：TTL 扫描 / 优先级驱逐 / 多 agent 配额
- [ ] M4 开源与上游贡献
