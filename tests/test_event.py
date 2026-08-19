"""Event 排序语义与取消标记的单元测试（FR-1）。"""

from ass.core.event import Event


def test_ordering_by_time() -> None:
    """时间不同时按时间升序。"""
    assert Event(time=1.0, seq=0) < Event(time=2.0, seq=1)
    assert Event(time=2.0, seq=5) > Event(time=1.0, seq=9)


def test_ordering_tie_broken_by_priority_then_seq() -> None:
    """时间相同时：优先级小者先；再相同按入队序号。"""
    assert Event(time=1.0, priority=0, seq=9) < Event(time=1.0, priority=1, seq=1)
    assert Event(time=1.0, priority=2, seq=3) < Event(time=1.0, priority=2, seq=4)


def test_cancel_sets_flag() -> None:
    """cancel() 仅设置标记，事件对象本身保持可比较。"""
    event = Event(time=0.0)
    assert not event.cancelled
    event.cancel()
    assert event.cancelled
