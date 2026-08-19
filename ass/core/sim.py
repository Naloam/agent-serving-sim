"""离散事件模拟主循环（M1 实现，对应 PRD FR-1）。

规划内容：基于 heapq 的 ``Simulation``，支持 ``schedule / cancel / run(until)``；
到达、完成等全部行为事件化，测试用虚拟时钟注入。
"""
