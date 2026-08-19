"""合成负载生成器（M1 实现，对应 PRD FR-3）。

规划内容：泊松到达、多轮会话、think_time 分布（对数正态等）、prompt 四段
长度分布、agent_type / priority 比例；随机性全部通过注入的
``random.Random(seed)``，固定种子可复现。
"""
