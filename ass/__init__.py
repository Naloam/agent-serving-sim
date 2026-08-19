"""ass —— Agent-Serving-Sim 主包。

面向 LLM agent 负载的推理服务离散事件模拟器：
trace 进 → 模拟 KV cache / 调度 / 淘汰策略 → JCT、命中率、显存曲线出。

子模块划分（详见 PLAN.md §4.1）：
- ``core``      离散事件模拟内核
- ``workload``  负载建模（schema / 合成生成 / 真实 trace 解析）
- ``cache``     KV cache 模型（radix tree / 驱逐策略）
- ``scheduler`` 调度模型
- ``metrics``   指标采集
- ``viz``       可视化
"""

__version__ = "0.1.0"
