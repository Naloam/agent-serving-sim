# PRD — agent-serving-sim 产品需求文档

> 版本：v1.0（2026-08-19）　|　战略与架构见 [PLAN.md](./PLAN.md)　|　执行规范见 [AGENTS.md](./AGENTS.md)
>
> 本文档定义"做什么、做到什么程度算完成"。功能范围变更需先改本文档。

---

## 1. 问题与目标

**问题**：LLM agent 负载（多轮、轮间长空闲、前缀高度重复、多 agent 并发）与现有推理服务系统的调度/缓存假设不匹配；研究者缺少公开可复现的模拟器与真实 trace。

**产品**：一个 Python 离散事件模拟器（`ass` 包）+ 配套 trace 工具 + 实验脚本。

**目标用户**（按优先级）：
1. **本人**：学习推理服务系统、产出可展示的研究成果
2. **研究者/工程师**：复现 agent 负载实验、验证缓存与调度策略
3. **开源社区**：作为论文复现与策略对比的公共基线

**成功指标**：
- M1 结束：模拟 10 万请求级别合成 trace，端到端出图（LRU vs TTL 对比）
- M2 结束：≥ 2 类真实 agent、≥ 1000 次请求的真实 trace + 负载刻画报告
- M3 结束：复现 Continuum 的"TTL 存在最优值"核心结论，误差可解释

---

## 2. 用户故事

- US-1 作为研究者，我读入一份 trace（合成或真实），选择缓存策略与参数，运行后得到 JCT 分布、命中率、显存曲线，用于对比策略收益。
- US-2 作为研究者，我扫描某参数（如 TTL、并发上限、缓存容量），自动得到参数-指标曲线。
- US-3 作为贡献者，我实现一个新的驱逐策略（一个类、一个注册点），不需要改动模拟器内核。
- US-4 作为作者，我用真实 trace 与合成 trace 分别运行同一实验，对比负载特征对结论的影响。

---

## 3. 功能需求

> 实现状态（2026-08-20）：FR-1~FR-12 全部实现并通过验收测试（`python -m pytest` 99 passed）。FR-11/FR-12 基于 Ollama 服务栈（vLLM 无 Windows 原生支持，见 PLAN §4.2 决策表）。

### 模拟内核（core）
- **FR-1** 事件与主循环：`Event`（时间、优先级、回调）、基于 heapq 的 `Simulation`，支持 `schedule / cancel / run(until)`。到达、完成等全部行为事件化。
  - 验收：时间乱序插入事件按时间序执行；cancel 后不触发；有单测。

### 负载（workload）
- **FR-2** Trace schema：与 [PLAN.md §4.3](./PLAN.md) 一致的 JSONL 格式，dataclass 定义 + 读写 + 字段校验（prompt 四段非负、think_time ≥ 0 等）。
  - 验收：非法行报明确错误；合法文件可往返（读→写→读）不变。
- **FR-3** 合成生成器：可配置会话数、每会话轮数、泊松到达率、think_time 分布（对数正态等）、prompt 四段长度分布、agent_type / priority 比例；固定随机种子可复现。
  - 验收：固定 seed 两次生成结果逐字节一致；统计量（到达间隔、轮长增长）与配置吻合。
- **FR-4** 真实 trace 解析器（M2）：读入采集探针输出的原始日志，清洗、对齐为 FR-2 格式。
  - 验收：至少兼容 coding 与搜索两类 agent 的日志；解析失败行单独记录不中断。

### 缓存（cache）
- **FR-5** Radix tree：节点为前缀段，记录 token 数；支持 `match(prefix) → 命中 token 数`、`insert`、按 token 数计费容量；被在途请求引用的节点不可淘汰（引用计数）。
  - 验收：命中计算、共享前缀分裂、引用保护均有单测。
- **FR-6** 驱逐策略：可插拔接口（统一 `select_victims(need_tokens) → 节点列表`）；实现 FIFO、LRU、TTL（超时未命中即淘汰）、Priority（按 agent_type/priority 排序）。
  - 验收：四种策略单测覆盖；新策略接入无需改内核（US-3）。

### 调度（scheduler）
- **FR-7** 请求生命周期：到达 → 前缀匹配 → 容量不足则驱逐 → 计入 cache → 按解析式模型计时（prefill = 未命中 token/吞吐，decode = 输出 token/吞吐）→ 完成 → 触发指标与下轮到达。并发上限约束在途请求数。
  - 验收：单请求路径手工计算核对一致；容量不足且无可淘汰时的行为有明确定义（排队）。

### 指标与可视化（metrics / viz）
- **FR-8** 指标：JCT（逐会话聚合与逐请求）、TTFT、前缀命中率（命中 token / 总 prompt token）、显存占用时间线、驱逐统计；导出 CSV + JSON。
- **FR-9** 可视化：JCT 的 CDF 图、显存时间线图、参数扫描曲线；matplotlib 出图存 `experiments/results/`。

### 实验（experiments）
- **FR-10** exp001（M1 验收实验）：合成 trace 上 LRU vs TTL，至少含命中率与 JCT 两个维度。
- **FR-11**（M2）采集探针：拦截 OpenAI 兼容 API 流量，记录时间戳、token 分解、间隔，不阻塞请求。
- **FR-12**（M3）策略实验套件：TTL 扫描、优先级驱逐、多 agent 配额，均可一键运行并输出图。
- **FR-13**（M3.5，2026-08-20 用户确认新增）**抢占建模**：decode 期 KV 按
  块分段增长（prompt 部分准入即插入 pin，输出部分逐块追加）；增长遇容量耗尽
  时先驱逐 idle 叶子，无 idle 可逐则按策略抢占在途请求——受害者 KV 全部
  丢弃、回队重算，抢占次数/被抢 token/浪费计算时间计入指标。
  - 验收：抢占路径手算核对单测（受害者回队后 JCT 含重算成本）；并发容量
    不足场景下抢占可复现；exp006 在抢占语义下重跑 TTL 扫描出图。
- **FR-14**（M3.6，2026-08-20 用户确认新增）**预测型在线驱逐**：
  `ClassTTLPolicy`（按类型静态 TTL）与 `PredictivePolicy`（在线对数正态
  think_time 拟合 + 逐轮存活率，按预测回归概率排序驱逐），配合可选
  `on_admit` 观测钩子；exp008 度量其相对 LRU→Belady 缺口的收窄。
  - 验收：策略无未来信息（在线因果）；单测覆盖存活概率计算与排序；
    exp008 一键运行出图（合成 + 真实 trace）。

---

## 4. 非功能需求

- **NFR-1 依赖**：core/workload/cache/scheduler 仅用标准库；metrics/viz 允许 numpy/pandas/matplotlib（本机已装）。
- **NFR-2 性能**：10 万请求模拟 Wall time < 1 分钟（普通笔记本 CPU）。
- **NFR-3 可复现**：所有实验脚本暴露 `--seed`，默认固定值。
- **NFR-4 可测性**：core/cache 必须 pytest 单测；整体测试 `python -m pytest` 一键运行。
- **NFR-5 环境**：Windows 优先（本机），不引入 Unix-only 依赖。

## 5. 明确不做（与 PLAN.md §6 Future Work 对齐）

token 级 continuous batching 仿真、分布式路由、GPU 微架构建模、真实权重前向、Web 界面。

（抢占建模原为 Future Work 候选，2026-08-20 经用户确认升级为 FR-13 纳入范围；
token 级 continuous batching 仍不做——FR-13 的 decode 增长是按块的粗粒度建模。）

## 6. 里程碑验收（DoD）

| 里程碑 | 完成定义 |
|--------|----------|
| M0 ✅ | git 仓库建立；`pip install -e .` 成功；pytest 空跑通过；论文笔记两篇 |
| M1 ✅ | FR-1~FR-10 全部验收通过；exp001 出图；README 更新用法 |
| M2 ✅ | FR-11 验收（探针 + 驱动器 + 不阻塞透传）；真实 trace 入库（1021 请求 ≥1000、两类 agent ≥2）；负载刻画报告（traces/real/REPORT.md + Blog #2）；计时标定误差记录（R²=0.32 及成因） |
| M3 ✅ | FR-12 验收（exp002~004 一键运行出图）；实验报告（Blog #3，含真实 trace 重放 exp005）；"TTL 存在最优点"修正为结构性结论 + 拐点工作点（论证记录于 Blog #3） |
| M4 | 英文 README；开源发布；≥1 个上游 PR 提交 |
