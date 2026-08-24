# agent-serving-sim 项目计划

> Agent 负载推理服务模拟器 —— 面向 LLM Agent Workload 的离散事件模拟与调度策略研究平台
>
> 创建日期：2026-08-19
> 状态：M0~M3 完成（待确认后进入 M4 开源整理）

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
│   ├── probe/              # 采集探针（M2）
│   │   └── proxy.py        #   OpenAI 兼容流量记录代理（透传 + JSONL 原始日志）
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
│   ├── exp002_ttl_sweep_heterogeneous.py  # M3：异构负载 TTL 扫描
│   ├── exp003_priority_eviction.py        # M3：优先级驱逐（带权 LRU 扫描）
│   ├── exp004_multi_agent_quota.py        # M3：多 agent 配额
│   ├── collect_real_trace.py    # M2：真实 trace 采集驱动器（探针 + Ollama）
│   ├── analyze_real_trace.py    # M2：负载刻画 + 计时标定
│   └── results/                #   图与 CSV 产出
├── blog/                   # 技术博客（中文，配图在 blog/assets/，M2 起落地）
├── examples/               # 示例 notebook（M4）
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
| M2 服务栈 | Ollama + 本地 qwen2.5-coder(-16k) | vLLM/SGLang 无 Windows 原生支持；WSL2 需把 vhdx 从 C 盘迁 D（红线级，挂起待用户决策）；Ollama 已装且模型在 D 盘，§6 风险表已授权 llama.cpp 级回退 |
| token 记账 | 前导按类型一次定型 + 对话流累计 | 逐请求按字符比例独立估算会让同一 system prompt 逐轮抖动，破坏前缀连续性（重放命中率 0.42→0.99 的教训） |

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
- [x] git init、pyproject.toml、venv（指向 D 盘）、装 pytest
- [x] 读 PagedAttention 论文（vLLM）
- [x] 读 SGLang 论文（RadixAttention 部分）
- [x] 搭好 `ass/` 包骨架与 CI 式自测（`pytest` 通过空测试）
- **产出**：可安装的空项目 + 两篇论文笔记

### M1 — 核心模拟器（第 2-4 周）
- [x] 离散事件内核（event/sim）+ 单测
- [x] radix tree + LRU/TTL 驱逐策略 + 单测（重点：引用计数下的正确淘汰）
- [x] 合成负载生成器（泊松到达、N 个会话、每会话 T 轮、前缀结构参数化）
- [x] 指标采集与基础可视化
- [x] **exp001：合成 trace 上 LRU vs TTL 的 JCT/命中率对比，出第一张图**
- **产出**：可 `pip install -e .` 并跑通端到端实验的模拟器 + 第一篇技术 blog 素材

### M2 — 真实 trace 采集（第 5-7 周，依赖 4060）
- [x] 服务栈：~~uv 建 3.12 venv + vLLM~~ → **Ollama**（vLLM 无 Windows 原生支持；WSL2 需迁 C 盘 vhdx，挂起；从本地 blob 派生 qwen2.5-coder-16k 变体防截断）
- [x] 写采集探针：拦截 OpenAI 兼容请求（时间戳、token 分解、间隔），不阻塞请求
- [x] 驱动 1-2 个真实 agent 采样：coding agent（工具调用回路）+ 搜索问答 agent
- [x] 真实 trace 解析入 `traces/real/`，负载特征分析（think_time 分布、前缀共享率、轮次长度增长）
- **产出**：一份真实 agent 负载刻画报告（`traces/real/REPORT.md` + Blog #2）

### M3 — 策略研究（第 8-12 周）
- [x] TTL 参数扫描：命中率/JCT 随 TTL 变化的曲线。**结论修正**：一阶模型中 TTL ≤ LRU（可证），价值在拐点工作点（TTL 设于类间回转周期之间，快回转类命中保持 LRU 的 ~99% 且持续释放死缓存）
- [x] 优先级驱逐：coding vs search 混部按 agent_type 定权重（WeightedLRUPolicy 平滑扫描：均值最优在 LRU，高价值类 p95 −12% = SLO 再分配工具）
- [x] 多 agent cache 配额实验（对标 TokenCake 思路）：中段与 LRU 重合（保险性质），配额低于工作集时反伤
- [x] 用真实 trace 重跑上述实验（exp005，标定参数，结论复现）
- [x] 计时模型标定：prefill 3912 tok/s / decode 51.8 tok/s，R²=0.32（非流式总时延含排队，局限已记录在案；流式 TTFT 留待下轮）
- **产出**：完整实验报告（Blog #3）；Future Work 新增：运行中请求抢占建模（Continuum 类收益的主战场）

### M3.5 — 抢占建模（2026-08-20 用户确认新增）

动机：M3 的统一图景表明，TTL/优先级类策略的文献收益依赖二阶效应，其中最重要的是
**运行中请求的抢占**（vLLM 在容量耗尽时以 recompute 方式抢占 decode 中的序列）。
TTL 主动释放 idle KV 的真正卖点是"让增长中的序列永远抢不到别人也抢不到它"——
该命题需要抢占语义才可测。

- [x] decode 期 KV 分段增长：prompt 部分在准入时插入 pin，输出部分按块增长（`decode_chunks`）
- [x] 增长遇容量耗尽：先驱逐 idle 叶子；无 idle 可逐时**抢占**在途请求（受害者按
  最新准入选择，其 KV 全部丢弃、回队重算，JCT 记入重算成本；每请求被抢上限 3 次后转不缓存保活）
- [x] 指标：抢占次数、被抢 token、浪费计算时间（手算核对单测覆盖）
- [x] 驱逐成本建模（`evict_tps`）：按需驱逐的 token/s 计入触发请求关键路径；TTL 的 sweep 保持免费
- [x] exp006：**驱逐免费时 LRU 仍胜（结构性结论复现）；驱逐计费 2k tok/s 时 TTL-15 翻盘——
  JCT −22%、p95 −56%（LRR 关键路径驱逐 2.7M token vs TTL 0.18M）；20K 容量 thrash 区
  全策略塌缩（抢占风暴 74 次、JCT 124s）**。Blog #4 完成
- [x] 流式 TTFT 采集轮（339 请求）：decode 拟合 48.6 tok/s **R²=0.9994**（解析式模型验证）；
  TTFT 与 prompt 规模零相关（R²=0.0095，~3.1s 固定+排队主导）——旧总时延拟合的
  prefill 估计系伪影，方法论教训：非流式回归把排队摊进 prefill 系数

### M3.6 — 预测型在线驱逐策略（2026-08-20 用户确认新增）

动机：exp007 表明 LRU 距 Belady 上限 16~25%，缺口集中在慢回转类的
"不知道哪些会话还会回来"。方向：用负载特征（think_time 分布、逐轮存活率）
**在线预测**会话回归，替代按原始空闲时长排序。

- [x] `ClassTTLPolicy`：按 agent_type 的静态 TTL（低压档收窄 10.7%，高压档失效——静态阈值不自适应）
- [x] `PredictivePolicy`：在线学习（每类 log-think 对数正态 + `P(窗口内回归)` 排序，无未来信息）——**收窄 LRU→Belady 缺口 30.5%（合成 40K）/ 20%（真实 4K）**，且低压档无害
- [x] 可选 `on_admit` 观测钩子（默认 no-op，在线策略的唯一观测入口）
- [x] exp008 四方对比完成；**负结果存档**：①逐轮存活率有删失偏差（重叠到达下度量的是到达进度而非终止，混入后全局劣化 −111%~−229%）；②对数正态 MRL 排序理论优雅但方向错误（U 型风险率下驱逐"间隔中必回归"会话，低压 −124%）——教训：在线最优序的目标应是"窗口内回归概率"而非"期望回归距离"

### M4 — 开源与贡献（持续）
- [x] 英文 README、文档、示例 notebook（2026-08-20 用户确认启动，先私有整理，
      发布动作另行确认；License 选择待定）——已完成 README_EN.md、
      examples/quickstart.ipynb、.github/workflows/ci.yml（pytest + exp001 冒烟）
- [ ] GitHub 开源，挂到相关 awesome 列表
  - 2026-08-20 已发布 **<https://github.com/Naloam/agent-serving-sim>**（public，MIT，
    v0.1.0，CI 已挂）
  - 2026-08-20 已提交：PDZZXL/Awesome-LLM-Serving PR#3（LLM Program/Agent 分区）；
    jjiantong/Awesome-KV-Cache-Optimization issue#9（提议 Tools 分区，先问后投）——待对方维护者处理
- [x] 从 vLLM/SGLang 的 good first issue 入手提 1-2 个 PR
  - 2026-08-20 提交 **sgl-project/sglang PR#35621**：`[core] Root the JIT kernel cache
    under SGLANG_CACHE_DIR`（#19612 残留缺口：自家 JIT 缓存硬编码回退不跟随根目录；
    含单测×2 + 用户文档同步；black/isort 过、提交信息按 [area] 规范）
  - 经验记录：vLLM/SGLang 的 good-first-issue 板块严重"抢票"（多个候选 issue 经评论区
    核实已有人提 PR 或已修复）；有效路径是读 issue 全部评论找维护者明确留白的残留工作
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
| 2026-08-19 | M0 工程部分完成：git init（main 分支）、pyproject.toml（零运行时依赖 + dev/viz extras）、项目根 venv（D 盘）、`ass/` 六子包骨架 + tests 冒烟测试，`python -m pytest` 通过；剩余：两篇论文笔记 |
| 2026-08-19 | M0 关闭：用户确认完成 PagedAttention 与 SGLang/RadixAttention 两篇论文阅读，M0 全部验收项达成，进入 M1 |
| 2026-08-19 | M1 完成：FR-1~FR-3、FR-5~FR-10 全部实现并配套 81 个单测；exp001 出图（LRU 命中率 0.813，TTL-5 降至 0.549，TTL-80 收敛回 LRU）；10 万请求端到端 19s（NFR-2 < 1min 达标）。设计要点：radix tree 以位置对齐的 Segment 段为元素（免 token 级展开保性能）；TTL 采用事件驱动的主动清除 + LRU 兜底 |
| 2026-08-19 | 用户确认 M1 通过；授权过夜自主推进 M2→M3 并撰写各里程碑 blog。M2 服务栈决策：**Ollama**（qwen2.5-coder:7b，本机已装、模型在 D 盘；vLLM 不支持 Windows 原生，WSL2 方案需迁移 C 盘 vhdx，挂起待用户决策）；采集目标 ≥1000 请求（coding+search 两类）。结构变更：新增 `blog/` 顶层目录（§4.1 已同步）；Blog #1（M1 篇）完成 |
| 2026-08-19 | M2 工程链路完成：FR-11 采集探针（ass/probe/，stdlib 透传代理 + JSONL 原始日志）、FR-4 解析器（loaders.py：到达序对齐、四段按字符占比配准 usage、坏行跳过、ProbeTiming 留档）、采集驱动器（collect_real_trace.py，两类 agent 工具调用循环 + 合成工具结果 + 泊松错峰并发）；从本地 blob 派生 qwen2.5-coder-16k 变体（num_ctx 16384，防截断）。正式采集后台运行中（目标 1050 请求），冒烟预检：think 中位数 coding 6.4s / search 21.1s |
| 2026-08-19 | M3 策略研究（合成负载部分）完成：SyntheticConfig 支持 AgentProfile 按类型差异化参数；新增 QuotaPolicy（软配额）与 WeightedLRUPolicy（有效年龄=空闲/权重，LRU↔严格优先的平滑插值）。exp002 异构 TTL 扫描：**结构性结论——零成本驱逐+开环模型下 TTL ≤ LRU（过期集⊆LRU 序尾部）**，价值在工作点拐点（ttl 介于两类回转周期之间时快回转类命中率保持 LRU 的 ~99% 且持续释放死缓存）；exp003 带权 LRU 扫描：均值加权 JCT 最优仍在 w=1（LRU），但高价值类 p95 单调改善至 −12%——优先级驱逐是 SLO 再分配工具而非均值优化工具；exp004 配额扫描：无跨类侵占时与 LRU 重合（保险性质），配额低于工作集时反伤。负载刻画与标定脚本（analyze_real_trace.py）、真实 trace 重放脚本（exp005）就绪 |
| 2026-08-20 | M2 完成：两批采集 **1021 请求**（coding 702 / search 319，162 会话，~70 min）达标 PRD ≥1000；刻画入库（think 中位 5.8s / 19.3s 双峰对数正态、前导 29%/37%、历史 ~60% 逐轮线性增长）；修复两处数据管线缺陷（逐请求比例估算致前缀断裂 → 累计记账；匿名预热请求污染类型前导定型）；计时标定 prefill 3912 / decode 51.8 tok/s（R²=0.32，排队混入总时延，流式 TTFT 留待下轮）。Blog #2 完成 |
| 2026-08-20 | M3 完成：exp005 真实 trace 重放复现结构性结论（LRU 0.9855 贴近理论上限；TTL 拐点在 2×~4× 全局 think 中位数）。统一图景：一阶模型中 LRU 均值不可战胜，TTL=显存确定性、优先级=SLO 再分配（p95 −12%）、配额=隔离保险；Continuum 类收益依赖二阶效应（驱逐锁开销、运行中请求抢占、多实例路由）——Future Work 新增抢占建模。Blog #3 完成。挂起待用户确认：① WSL2+vLLM 补采方案（需迁 vhdf 到 D 盘）；② 流式 TTFT 采集轮；③ M4 开源整理启动 |

### 待用户确认（2026-08-20 晨间检查清单）

**2026-08-20 已全部决策**：

1. ~~WSL2 + vLLM 补采~~ → **不做**（既有结论不依赖，vhdf 迁移风险/收益比不佳）
2. ~~流式 TTFT 采集轮~~ → **跑一轮**（升级计时标定为 TTFT/排队分解）
3. ~~M4 开源整理~~ → **启动，先私有整理**（英文 README/notebook/CI；对外发布另行确认）
4. ~~抢占建模~~ → **列入计划并开工**（新增 M3.5 里程碑与 PRD FR-13）

| 2026-08-20 | 晨间四项决策执行完毕：M3.5 抢占+驱逐成本建模完成（FR-13 验收：手算单测 + exp006）；exp006 核心结论——**驱逐免费时 LRU 仍胜；驱逐计费时 TTL-15 翻盘（JCT −22%、p95 −56%）**；流式 TTFT 轮（339 请求）标定分解：decode 48.6 tok/s（R²=0.9994）、TTFT 与 prompt 规模零相关（~3.1s 排队主导）；M4 私有整理完成（README_EN、quickstart notebook、CI workflow），对外发布与 License 待用户决策。Blog #4 完成 |
| 2026-08-20 | 用户授权发布：**GitHub 开源 <https://github.com/Naloam/agent-serving-sim>**（public、MIT、CI 徽章已挂、blog #1 链接回填）；`fixed_overhead_s`（TTFT 截距参数）入模。M4 剩余：awesome 列表提交（需向第三方仓库提 PR，属对外动作，留待用户确认目标列表）、vLLM/SGLang 上游 PR（长期）。v0.1.0 tag 发布 |
| 2026-08-20 | 技术收口（用户选定 ①+②）：真实 trace 双重验证 exp006——驱逐计费下 TTL-29.6 JCT 3.137 vs LRU 3.158（方向复现，幅度 0.7% vs 合成 22%，与驱逐流量 354K→295K 的降幅成正比，自洽）；新增 **BeladyPolicy**（离线最优，按流内位置 + 未来访问 oracle）与 exp007——**LRU 距理论上限 16.3%（80K）/24.9%（40K），缺口集中在慢回转类**（search 0.757 vs 0.492；快类近最优），为后续在线策略给出量化靶点（预测慢类回归）。Blog #4 增后记。挂起：SGLang PR#35621、PDZZXL PR#3、jjiantong#9 等待对方响应 |
| 2026-08-20 | **M3.6 预测型在线驱逐完成**（用户确认新增，FR-14）：`on_admit` 观测钩子 + `ClassTTLPolicy` + `PredictivePolicy`（在线对数正态拟合，窗口回归概率排序，117 单测）。exp008 核心结果：**在线预测策略收窄 LRU→Belady 缺口 30.5%（合成 40K）/ 20%（真实 4K）**，低压档与 class-ttl 持平无害；两个方法论负结果存档（轮次存活率删失偏差、MRL 排序方向错误）——"窗口内回归概率"是在线排序的正确目标 |
| 2026-08-20 | Blog #5（预测策略两次翻车与反转）完成；应用 writing-beats/edit-article skill 对 #1~#4 全文去 AI 味重写：叙事化段落（≤240 字符）、砍路标语与加粗、删模板结尾、数据表保留 |
| 2026-08-20 | jjiantong 列表维护者回复 issue#9 称仓库链接 404——排查发现 **agent-serving-sim 仓库一度变为 private**（创建时为 public；原因待查，疑似 GitHub 自动化误标记或误触），已改回 public 并经匿名 API 验证可访问；已回复维护者（附正确链接 + 工具定位说明 + 按其偏好格式提 PR 的邀约）。**待用户自查**：GitHub 邮件/仓库设置有无 Trust&Safety 标记通知，若再次自动转私需联系 GitHub 支持 |
| 2026-08-20 | **用户要求 blog 仅本地保留**：`git rm --cached blog` + `.git/info/exclude`，README_EN 与 v0.1.0 release 备注中的 blog 引用已清理，远端已验证不含 blog/。**注意：任何后续会话不要把 blog/ 重新加入版本库**（本地排除已配置，此处备忘防止误"修复"） |

### 待用户确认（后续可选方向）

- Blog 对外发布平台（知乎/掘金/个人站/Medium）未定
- 第二个上游 PR（llm-d 文档或 SGLang 下一个残留）
- 基于 exp006/007 + 模拟器 + trace 写 arXiv 预印本（超出 PLAN 范围，需用户决策）
- 预测型在线策略研究（exp007 指出的靶点：按 think_time 分布预测慢类回归）
