"""离散事件：模拟中一切行为的最小单元（对应 PRD FR-1）。

事件按 ``(time, priority, seq)`` 升序执行：时间相同时优先级小者先，
仍相同则按入队顺序（``seq`` 由 :class:`~ass.core.sim.Simulation` 分配），
保证同一时刻行为的确定性。
"""

from collections.abc import Callable
from dataclasses import dataclass, field

# 回调约定：无参数、无返回值；副作用通过闭包或模拟器句柄携带
EventCallback = Callable[[], None]


@dataclass(order=True)
class Event:
    """携带执行时间、类型与回调的事件对象。"""

    time: float
    priority: int = field(default=0)
    seq: int = field(default=0)
    # 类型标签仅用于调试与统计，不参与排序
    kind: str = field(default="", compare=False)
    callback: EventCallback | None = field(default=None, compare=False)
    cancelled: bool = field(default=False, compare=False)

    def cancel(self) -> None:
        """将事件标记为取消：到达队首时被跳过，回调不触发。"""
        self.cancelled = True
