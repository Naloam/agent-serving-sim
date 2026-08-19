"""可插拔驱逐策略（M1 实现，对应 PRD FR-6）。

规划内容：统一接口 ``select_victims(need_tokens) -> 节点列表``；
实现 FIFO / LRU / TTL / Priority 四种策略。新策略 = 新类 + 注册点，不改内核。
"""
