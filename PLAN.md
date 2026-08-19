# agent-serving-sim 项目计划

> Agent 负载推理服务模拟器 —— 面向 LLM Agent Workload 的离散事件模拟与调度策略研究平台
>
> 创建日期：2026-08-19
> 状态：规划完成，待启动 M0

---

## 1. 项目定位

**一句话**：输入 LLM agent 会话 trace，在 CPU 上模拟推理服务的 KV cache、调度与淘汰策略，输出 JCT / 缓存命中率 / 显存占用曲线，用于研究 agent 负载下的服务优化。

**要解决的问题**：agent 负载与传统 chat 负载差异巨大——多轮调用、轮次间存在长空闲（agent 在"思考"或执行工具）、system prompt 与工具定义高度重复、会话历史逐轮增长、多 agent 并发会互相挤占 cache。现有推理服务系统（vLLM/SGLang）的调度策略并非为此设计。学术界 2025-2026 已出现一批针对性工作（见 §2），但**模拟器与 trace 大多不开源**。

**价值定位**：
1. 一个开源、可复现的 agent 负载模拟器（现有论文的通用短板，社区有真实需求）
2. 一份本地采集的真实 agent trace（这个领域最稀缺的资源之一）
3. 在其上做策略研究（TTL / 优先级驱逐 / 多 agent 配额），产出实验报告与技术 blog
4. 个人目标：以此项目切入 AI infra，建立"推理服务系统"方向的完整认知与可展示作品

---

## 2. 背景调研：当前格局（2025-2026）

核心洞察（所有近期工作的共识）：**agent 请求两轮之间的 KV cache"留还是不留"是核心矛盾**——留着可复用省重算，但占显存；不留则下轮全量重算拉高 JCT。

### 学术工作（学习清单，按阅读顺序）

| 论文 | 会议 | 核心思想 |
|------|------|----------|
| **Efficient Memory Management for LLM Serving with PagedAttention**（vLLM） | SOSP'23 | 分页 KV cache，一切的基础，必读第一篇 |
| **SGLang: Efficient Execution of Structured Language Model Programs**（RadixAttention） | NeurIPS'24 | radix tree 组织前缀 cache，agent 复用的关键机制，必读第二篇 |
| **Continuum / CacheTTL** | ICLR'26 | 给 agent cache 加 TTL 机制，优化多轮 JCT，本项目的直接对标 |
| **KVFlow** | NeurIPS'25 | agent workflow 建模为 DAG，调度 cache 迁移 |
| **TokenCake** | arXiv 2510.18586 | 多 agent 共存时的 cache 空间争抢与配额 |
| **ForeCache** | MLSys'26 | coding agent 负载刻画与 cache 优化 |

### 工业系统（对标与提 PR 的目标）

- **llm-d**：KV-cache 感知路由（Red Hat 开源）
- **TensorRT-LLM**：负载感知的优先级淘汰 API
- **vLLM / SGLang**：源码阅读目标（vLLM 的 scheduler 模块、SGLang 的 radix_cache 实现）

### 调研结论

方向足够新（核心论文集中在近 12 个月）、足够热（ICLR/NeurIPS/MLSys 持续收）、且**标准研究方法论不依赖 GPU**：trace 分析 + 离散事件模拟 + 分析模型。本机恰好有一张 4060（见 §3），真实 trace 采集与后期验证也能本地完成。

---

## 3. 本机环境与资源（已探明）

| 项目 | 状态 | 说明 |
|------|------|------|
| GPU | RTX 4060 Laptop, 8GB | 可本地跑 vLLM/SGLang + Qwen2.5 0.5B~3B，trace 采集与实验验证**无需租卡** |
| 驱动 / CUDA | 596.49 / CUDA 13.3 工具链 | nvcc 可用，torch 2.7.1 已装 |
| Python | 3.13.4（全局） | numpy/scipy/pandas/matplotlib 齐备；无 conda |
| 磁盘 | D 盘剩余 134GB | **C 盘仅剩 17GB，一切大体积安装（模型、venv）必须指向 D 盘** |
| Git / Rust / Node | 2.55 / 1.94 / 22.14 | 齐备 |

**已知风险**：
- vLLM 对 Python 3.13 的支持待验证 → M2 时用 `uv` 建 3.11/3.12 独立 venv（`uv venv --python 3.12`），模型缓存目录设到 D 盘
- 8GB 显存 → 优先 Qwen2.5-0.5B/1.5B，3B 可尝试，7B 需量化且体验差

---

## 4. 架构设计

### 4.1 目录结构

```
agent-serving-sim/
├── PLAN.md                 # 本文件
├── README.md
├── pyproject.toml          # M0 创建，零第三方强依赖（numpy/pandas/matplotlib 可选增强）
├── ass/                    # 主包（Agent-Serving-Sim 缩写）
│   ├── core/               # 离散事件模拟内核
│   │   ├── event.py        #   Event（时间、类型、回调）+ 优先队列调度
│   │   └── sim.py          #   Simulation 主循环（heapq 实现）
│   ├── workload/           # 负载建模
│   │   ├── schema.py       #   trace 格式定义（dataclass + JSONL 读写）
│   │   ├── synthetic.py    #   合成负载生成器（泊松到达、多轮会话、前缀结构）
│   │   └── loaders.py      #   真实 trace 解析器（M2 产出后的接入点）
│   ├── cache/              # KV cache 模型
│   │   ├── radix.py        #   radix tree（节点=前缀段，记录 token 数与引用计数）
│   │   └── policies.py     #   驱逐策略：FIFO / LRU / TTL / Priority（可插拔接口）
│   ├── scheduler/          # 调度模型
│   │   └── serving.py      #   到达→匹配前缀→分配 cache→（简化 batching）→完成
│   ├── metrics/            # 指标采集
│   │   └── collector.py    #   JCT、TTFT、逐会话统计、命中率、显存时间线
│   └── viz/                # 可视化
│       └── plots.py        #   CDF 图、时间线图、参数扫描曲线
├── traces/
│   ├── synthetic/          #   合成 trace
│   └── real/               #   真实采集 trace（M2）
├── experiments/
│   ├── exp001_lru_vs_ttl.py    # M1 结尾的第一个实验
│   └── results/                #   图与 CSV 产出
└── tests/                  # pytest 单测（radix tree 与策略必须有单测）
```

### 4.2 关键设计决策

| 决策 | 选择 | 理由 |
|------|------|------|
| 事件循环 | **自研**（heapq 优先队列，约 300 行） | 核心 infra 学习价值；零依赖（simpy 未装，作为备选方案保留） |
| 计时模型 | **解析式**：token 吞吐参数（如 prefill X tok/s、decode Y tok/s）折算耗时，不模拟真实计算 | 模拟器本意是对比策略相对收益；后期用本机 vLLM 实测标定参数 |
| cache 模型 | radix tree + 按 token 数计费显存 | 与 SGLang RadixAttention 同构，学到的知识直接对上真实系统 |
| trace 格式 | JSONL，每行一个请求 | 简单、可 diff、可被 pandas 直接读 |
| batching | 简化为"容量约束的并发上限" | 第一版不做 token 级 batching；M3 视需要加 continuous batching 模型 |

### 4.3 Trace 格式（JSONL，每行一个请求）

```json
{
  "session_id": "sess_0001",
  "turn_id": 3,
  "arrival_time": 42.7,
  "prompt": {"system": 812, "tools": 1043, "history": 2210, "new": 156},
  "output_tokens": 388,
  "think_time": 18.3,
  "agent_type": "coding",
  "priority": 1
}
```

- `prompt` 四段分解让模拟器能精确计算前缀复用（system+tools 跨会话共享、history 会话内逐轮增长）
- `think_time` 是轮次间空闲（上一轮结束到下一轮到达的间隔），agent 负载区别于 chat 的本质特征，TTL 策略的核心变量
- `agent_type` / `priority` 支撑 M3 的多 agent 配额与优先级驱逐实验

### 4.4 核心模拟流程

```
事件驱动主循环：
  on_request_arrival(req):
      prefix_len = radix_tree.match(req.prompt_prefix)   # 命中 → 省 prefill
      if cache 空间不足:
          eviction_policy.evict(until 腾出足够空间)        # LRU/TTL/Priority 在此插拔
      radix_tree.insert(req)
      prefill_time = (req.total_prompt - prefix_len) / prefill_tps
      decode_time  = req.output_tokens / decode_tps
      schedule(on_complete, now + prefill_time + decode_time)
  on_complete(req):
      metrics.record(JCT 分量、命中情况)
      下轮到达时间 = now + think_time（由 workload 生成器预排或此处触发）
```

---

## 5. 里程碑（预计 3-4 个月业余时间）

### M0 — 起步（第 1 周）
- [ ] git init、pyproject.toml、venv（指向 D 盘）、装 pytest
- [ ] 读 PagedAttention 论文（vLLM）
- [ ] 读 SGLang 论文（RadixAttention 部分）
- [ ] 搭好 `ass/` 包骨架与 CI 式自测（`pytest` 通过空测试）
- **产出**：可安装的空项目 + 两篇论文笔记

### M1 — 核心模拟器（第 2-4 周）
- [ ] 离散事件内核（event/sim）+ 单测
- [ ] radix tree + LRU/TTL 驱逐策略 + 单测（重点：引用计数下的正确淘汰）
- [ ] 合成负载生成器（泊松到达、N 个会话、每会话 T 轮、前缀结构参数化）
- [ ] 指标采集与基础可视化
- [ ] **exp001：合成 trace 上 LRU vs TTL 的 JCT/命中率对比，出第一张图**
- **产出**：可 `pip install -e .` 并跑通端到端实验的模拟器 + 第一篇技术 blog 素材

### M2 — 真实 trace 采集（第 5-7 周，依赖 4060）
- [ ] `uv` 建 Python 3.12 venv，装 vLLM（或 SGLang），本地起 Qwen2.5-0.5B/1.5B 服务，模型与缓存全指向 D 盘
- [ ] 写采集探针：拦截 OpenAI 兼容请求（时间戳、token 分解、间隔）
- [ ] 驱动 1-2 个真实 agent 采样：coding agent（如简化版 mini-code-agent）+ 搜索问答 agent
- [ ] 真实 trace 解析入 `traces/real/`，做负载特征分析（think_time 分布、前缀共享率、轮次长度增长）
- **产出**：一份真实 agent 负载刻画报告（这本身就是有传播价值的内容）

### M3 — 策略研究（第 8-12 周）
- [ ] TTL 参数扫描：命中率/JCT 随 TTL 变化的曲线，验证存在最优点（复现 Continuum 核心结论）
- [ ] 优先级驱逐：coding vs 搜索 agent 混部时按 agent_type 定权重
- [ ] 多 agent cache 配额实验（对标 TokenCake 思路）
- [ ] 用真实 trace 重跑上述实验
- [ ] 尝试用 vLLM 实测数据标定计时模型，缩小模拟与真实的误差
- **产出**：完整实验报告；挑选最有价值的结果写第二篇 blog

### M4 — 开源与贡献（持续）
- [ ] 英文 README、文档、示例 notebook
- [ ] GitHub 开源，挂到相关 awesome 列表
- [ ] 从 vLLM/SGLang 的 good first issue 入手提 1-2 个 PR（长期目标）

---

## 6. 风险与对策

| 风险 | 对策 |
|------|------|
| vLLM 不支持 Python 3.13 | uv 建 3.12 独立 venv；或退回 SGLang / llama.cpp + 采集代理的方案 |
| 8GB 显存跑不动目标模型 | 0.5B/1.5B 优先；量化版兜底；极端情况用 API + 代理采集（只损失计时标定） |
| 真实 agent 采集工程量大 | 模拟器（M1）不依赖它；先合成 trace 保进度，采集并行推进 |
| 计时模型失真 | 定位是"策略相对收益对比"而非绝对性能预测；M3 用实测数据校准 |
| 独自做难坚持 | 每个里程碑都有可展示产出（图/报告/blog）；M1 结束即发第一篇 |
| 范围蔓延（想做 batching/GPU 细节） | 严格守住"cache 与调度策略"主线，其余记入 Future Work |

### Future Work（明确不做，防止蔓延）
token 级 continuous batching 仿真、分布式多实例路由、CUDA 微架构建模、真实权重前向

---

## 7. 学习资料清单

**论文（按序）**
1. PagedAttention（vLLM, SOSP'23）
2. SGLang / RadixAttention（NeurIPS'24）
3. Continuum / CacheTTL（arXiv 2511.02230, ICLR'26）
4. KVFlow（NeurIPS'25） / TokenCake（arXiv 2510.18586） / ForeCache（MLSys'26）

**源码阅读入口**
- vLLM：`vllm/core/scheduler.py`（调度主循环）、`vllm/core/block_manager.py`（block 分配）
- SGLang：`python/sglang/srt/mem_cache/radix_cache.py`（radix tree 实现，本模拟器 cache 模块的对标）

**实践参考**
- Awesome-KV-Cache-Optimization（GitHub 综述仓库）
- llm-d 的 KV-cache 感知路由博客（工业视角）

---

## 8. 进度记录

| 日期 | 事项 |
|------|------|
| 2026-08-19 | 完成环境探查、方向调研与本计划；项目立项 |
| 2026-08-19 | 补充文档体系：PRD.md（需求与验收）、AGENTS.md（协作规范）、CLAUDE.md（指针） |
