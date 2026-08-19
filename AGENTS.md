# AGENTS.md — agent 工作规范

> 本文件约束 AI 助手（以及任何协作者）在本仓库中的工作方式。开始任何任务前，先完整阅读本文件。
>
> 文档地图：**[PLAN.md](./PLAN.md)** = 战略/架构/里程碑　|　**[PRD.md](./PRD.md)** = 需求与验收标准　|　**本文件** = 工作规范

---

## 1. 项目一句话

面向 LLM agent 负载的推理服务离散事件模拟器：trace 进 → 模拟 KV cache/调度/淘汰策略 → JCT、命中率、显存曲线出。

## 2. 本机环境（Windows 笔记本）

- **Python 3.13.4**：命令用 `python`（**`python3` 不存在**）；无 conda
- **GPU**：RTX 4060 Laptop 8GB，CUDA 13.3；M2 起 vLLM/SGLang 若不兼容 3.13，用 `uv venv --python 3.12` 建独立环境
- **磁盘红线**：C 盘仅剩 ~17GB。**任何大体积安装（venv、模型、缓存）必须落在 D 盘**；pip 缓存与 HF 缓存目录需显式指向 D 盘
- 已装：numpy/scipy/pandas/matplotlib、torch 2.7.1、git 2.55

## 3. 常用命令

```bash
# 环境（首次）
python -m venv .venv           # 在项目根目录执行，天然落在 D 盘
.venv\Scripts\activate
pip install -e ".[dev]"

# 测试（每次提交前必须通过）
python -m pytest

# 运行实验
python experiments/exp001_lru_vs_ttl.py --seed 42
```

## 4. 代码规范

- 包结构严格遵循 [PLAN.md §4.1](./PLAN.md)，不新增顶层目录
- 标识符、日志、异常信息用英文；**注释与 docstring 用中文**；文档（md）用中文
- 全量 type hints；数据结构优先 dataclass；不用全局可变状态
- 依赖边界（PRD NFR-1）：`core/workload/cache/scheduler` 仅标准库；numpy/pandas/matplotlib 只允许出现在 `metrics/viz/experiments`
- 策略类（驱逐、生成器分布）一律可插拔：新实现 = 新类 + 注册点，不改内核
- 提交信息用 Conventional Commits（`feat: ...` / `fix: ...` / `docs: ...`），一次提交一个主题，小步提交

## 5. 测试规范

- `core/`、`cache/`、`scheduler/` 的每个模块配对应用例，先写测试再实现（TDD 优先，复杂逻辑强制）
- 随机性一律走注入的 `random.Random(seed)`，禁止模块级 `random.*` 直接调用
- 涉及时间的测试不 sleep，用事件循环注入的虚拟时钟

## 6. 工作流（每个任务的标准动作）

1. **先读文档**：动手前确认任务对应 PLAN.md 的哪个里程碑、PRD.md 的哪些 FR
2. **小步实现**：按 FR 粒度提交，每步 `python -m pytest` 通过
3. **验收对照**：完成 FR 后逐条核对 PRD.md 中的验收标准，不满足不算完成
4. **更新文档**（同一提交内完成）：
   - PLAN.md §8 进度记录表加一行（日期 + 事项）
   - 涉及 FR 状态变化的，更新 PRD.md 对应验收标记
   - 设计决策变化的，追加到 PLAN.md §4.2 决策表
5. **报告格式**：结束时说明完成了什么、测试结果、下一步建议，不虚构未验证的结论

## 7. 边界（不要做）

- 不引入 PRD §5 明确排除的功能（batching 仿真、分布式、GPU 建模等）——想加先改 PRD 并征得用户同意
- 不改 trace 格式（PLAN.md §4.3）；确需变更须同步修改 PLAN/PRD 并在进度记录中注明
- 不安装新的重量级依赖（>100MB）前先询问
- 不向 C 盘写入任何大文件；不动 `.venv/` 以外的系统环境
- 不在未经用户确认时跨里程碑推进；每个里程碑完成即停，等确认
