"""基于 heapq 的离散事件模拟主循环（对应 PRD FR-1）。

用法::

    sim = Simulation()
    sim.schedule(2.5, callback=hello, kind="arrival")
    sim.run(until=10.0)

时间语义：

- ``schedule`` 以**绝对时间**入队，且必须不早于当前 ``now``（保证因果）；
- ``run(until)`` 执行所有 ``time <= until`` 的事件后返回，更晚的事件留在
  队列中，可被下一次 ``run`` 继续消费；
- 取消采用惰性标记：被取消的事件仍留在堆中，出队时直接跳过。

时钟是纯虚拟量，测试不依赖任何真实 sleep。
"""

import heapq
from collections.abc import Iterator
from itertools import count

from ass.core.event import Event, EventCallback


class Simulation:
    """时间有序的事件循环。"""

    def __init__(self) -> None:
        self._heap: list[Event] = []
        self._seq: Iterator[int] = count()
        self._now: float = 0.0
        self._active: int = 0

    @property
    def now(self) -> float:
        """当前模拟时间（最后一个已执行事件的时间）。"""
        return self._now

    @property
    def pending(self) -> int:
        """尚未执行的未取消事件数。"""
        return self._active

    def schedule(
        self,
        time: float,
        callback: EventCallback | None = None,
        *,
        priority: int = 0,
        kind: str = "",
    ) -> Event:
        """在绝对时间 ``time`` 入队一个事件并返回其句柄。"""
        if time < self._now:
            raise ValueError(
                f"cannot schedule event in the past: time={time}, now={self._now}"
            )
        event = Event(
            time=time,
            priority=priority,
            seq=next(self._seq),
            kind=kind,
            callback=callback,
        )
        heapq.heappush(self._heap, event)
        self._active += 1
        return event

    def cancel(self, event: Event) -> None:
        """取消事件（幂等）：已取消的事件不会触发回调。"""
        if not event.cancelled:
            event.cancelled = True
            self._active -= 1

    def run(self, until: float | None = None) -> float:
        """执行事件直到队列耗尽或下一事件时间晚于 ``until``。

        返回执行结束时的模拟时间（停在最后一个已执行事件的时间上，
        不会凭空快进到 ``until``）。
        """
        while self._heap:
            if until is not None and self._heap[0].time > until:
                break
            event = heapq.heappop(self._heap)
            if event.cancelled:
                continue
            self._active -= 1
            self._now = event.time
            if event.callback is not None:
                event.callback()
        return self._now
