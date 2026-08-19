"""Simulation 主循环的单元测试（FR-1）：乱序插入、取消、run(until)。"""

import pytest

from ass.core.sim import Simulation


def test_empty_run_returns_zero() -> None:
    sim = Simulation()
    assert sim.run() == 0.0
    assert sim.pending == 0


def test_out_of_order_inserts_execute_in_time_order() -> None:
    """乱序插入的事件按时间序执行（PRD 验收项）。"""
    sim = Simulation()
    order: list[str] = []
    sim.schedule(3.0, lambda: order.append("c"), kind="c")
    sim.schedule(1.0, lambda: order.append("a"), kind="a")
    sim.schedule(2.0, lambda: order.append("b"), kind="b")
    sim.run()
    assert order == ["a", "b", "c"]
    assert sim.now == 3.0


def test_same_time_uses_priority_then_fifo() -> None:
    sim = Simulation()
    order: list[str] = []
    sim.schedule(1.0, lambda: order.append("low"))
    sim.schedule(1.0, lambda: order.append("high"), priority=-1)
    sim.schedule(1.0, lambda: order.append("late-same"))
    sim.run()
    assert order == ["high", "low", "late-same"]


def test_cancelled_event_does_not_fire() -> None:
    """cancel 后不触发回调（PRD 验收项）。"""
    sim = Simulation()
    calls: list[int] = []
    event = sim.schedule(1.0, lambda: calls.append(1))
    sim.schedule(2.0, lambda: calls.append(2))
    sim.cancel(event)
    sim.run()
    assert calls == [2]
    assert sim.pending == 0
    assert sim.now == 2.0


def test_run_until_leaves_later_events_pending() -> None:
    sim = Simulation()
    calls: list[str] = []
    sim.schedule(1.0, lambda: calls.append("a"))
    sim.schedule(5.0, lambda: calls.append("b"))
    sim.schedule(5.5, lambda: calls.append("c"))
    sim.run(until=5.0)
    assert calls == ["a", "b"]  # time == until 会被执行
    assert sim.pending == 1
    assert sim.now == 5.0
    sim.run()
    assert calls == ["a", "b", "c"]
    assert sim.now == 5.5


def test_callback_can_schedule_within_same_run() -> None:
    sim = Simulation()
    order: list[int] = []

    def first() -> None:
        order.append(1)
        sim.schedule(2.0, second)

    def second() -> None:
        order.append(2)

    sim.schedule(1.0, first)
    sim.run()
    assert order == [1, 2]


def test_schedule_in_past_raises() -> None:
    sim = Simulation()
    sim.schedule(5.0, lambda: None)
    sim.run(until=5.0)
    with pytest.raises(ValueError, match="past"):
        sim.schedule(4.0, lambda: None)


def test_schedule_at_now_is_allowed() -> None:
    sim = Simulation()
    sim.schedule(3.0, lambda: None)
    sim.run(until=3.0)
    fired: list[int] = []
    sim.schedule(3.0, lambda: fired.append(1))
    sim.run()
    assert fired == [1]
