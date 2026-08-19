# agent-serving-sim

Agent 负载推理服务模拟器：输入 LLM agent 会话 trace，模拟 KV cache、调度与淘汰策略，输出 JCT / 缓存命中率 / 显存占用，用于研究 agent 负载下的推理服务优化。

## 文档

| 文档 | 内容 |
|------|------|
| [PLAN.md](./PLAN.md) | 战略与架构：背景调研、设计决策、里程碑、风险 |
| [PRD.md](./PRD.md) | 需求与验收：功能需求（FR）、非功能需求、各里程碑 DoD |
| [AGENTS.md](./AGENTS.md) | 协作规范：环境、命令、代码/测试规范、工作流（AI 助手必读） |

## 快速开始（M1 完成后可用）

```bash
pip install -e .
# 待实现
```

## 项目阶段

- [ ] M0 起步：脚手架 + 论文阅读
- [ ] M1 核心模拟器：事件循环 + radix cache + LRU/TTL + 第一个实验
- [ ] M2 真实 trace：本地 vLLM/SGLang + agent 采集
- [ ] M3 策略研究：TTL 扫描 / 优先级驱逐 / 多 agent 配额
- [ ] M4 开源与上游贡献
