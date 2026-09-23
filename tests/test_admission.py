"""M6/M7 准入策略的单元测试：FIFO 等价性、优先级、SJF、跳过语义、续链优先。"""

from ass.scheduler.admission import (
    FIFOAdmission,
    PriorityAdmission,
    SessionChainAdmission,
    ShortestJobAdmission,
)
from ass.scheduler.serving import ServingConfig, ServingSim
from ass.workload.schema import PromptBreakdown, TraceRequest


def make_request(session: str, arrival: float, system: int, new: int,
                 output: int, agent: str = "coding", turn: int = 1,
                 history: int = 0) -> TraceRequest:
    return TraceRequest(
        session_id=session, turn_id=turn, arrival_time=arrival,
        prompt=PromptBreakdown(system=system, tools=0, history=history, new=new),
        output_tokens=output, think_time=0.0, agent_type=agent, priority=1,
    )


# ---- 策略排序语义 ----

def test_fifo_returns_head_only() -> None:
    queue = [make_request("a", 0.0, 10, 5, 3), make_request("b", 1.0, 10, 5, 3)]
    assert FIFOAdmission().order(queue, 0.0) == [queue[0]]


def test_priority_orders_by_class_weight_then_arrival() -> None:
    queue = [
        make_request("low1", 0.0, 10, 5, 3, agent="search"),
        make_request("low2", 1.0, 10, 5, 3, agent="search"),
        make_request("high2", 2.0, 10, 5, 3, agent="coding"),
        make_request("high1", 3.0, 10, 5, 3, agent="coding"),
    ]
    ordered = PriorityAdmission(weights={"coding": 2.0, "search": 1.0}).order(queue, 0.0)
    assert [r.session_id for r in ordered] == ["high2", "high1", "low1", "low2"]


def test_sjf_orders_by_known_work() -> None:
    queue = [
        make_request("big", 0.0, system=1000, new=500, output=300),
        make_request("small", 1.0, system=100, new=50, output=30),
    ]
    ordered = ShortestJobAdmission().order(queue, 0.0)
    assert [r.session_id for r in ordered] == ["small", "big"]


def test_priority_rejects_empty_weights() -> None:
    try:
        PriorityAdmission(weights={})
        raised = False
    except ValueError:
        raised = True
    assert raised


# ---- ServingSim 集成 ----

def test_default_admission_matches_legacy_behavior() -> None:
    """默认 FIFO 与旧行为等价：并发 1 下严格按到达序服务。"""
    requests = [
        make_request("a", 0.0, 100, 50, 30),
        make_request("b", 0.0, 100, 50, 20),
        make_request("c", 0.0, 100, 50, 10),
    ]
    sim = ServingSim(ServingConfig(cache_capacity_tokens=10_000, max_concurrent=1))
    sim.submit_all(requests)
    sim.run()
    order = [record.session_id for record in sim.collector.records]
    assert order == ["a", "b", "c"]  # 同刻到达按提交序


def test_priority_admission_reorders_waiting_queue() -> None:
    """单槽被占位请求占据时，等待中的高权重类先被准入。"""
    requests = [
        make_request("blocker", 0.0, 200, 100, 100),               # 服务 4.0s
        make_request("low", 0.1, 100, 50, 30, agent="search"),     # 等待
        make_request("high", 0.2, 100, 50, 30, agent="coding"),    # 等待
    ]
    config = ServingConfig(cache_capacity_tokens=10_000, max_concurrent=1,
                           prefill_tps=100.0, decode_tps=100.0)
    sim = ServingSim(config, admission=PriorityAdmission(weights={"coding": 2.0, "search": 1.0}))
    sim.submit_all(requests)
    sim.run()
    order = [record.session_id for record in sim.collector.records]
    assert order == ["blocker", "high", "low"]
    records = {r.session_id: r for r in sim.collector.records}
    assert records["low"].queue_delay > records["high"].queue_delay


def test_sjf_admission_reduces_mean_wait_under_saturation() -> None:
    """饱和排队下 SJF 的平均排队时延优于 FIFO（排队论经典结论）。"""
    requests = [
        make_request("blocker", 0.0, system=100, new=50, output=30),   # 1.8s，占位
        make_request("huge", 0.1, system=2000, new=1000, output=800),  # 38s
        make_request("t1", 0.2, system=100, new=50, output=40),
        make_request("t2", 0.3, system=100, new=50, output=40),
        make_request("t3", 0.4, system=100, new=50, output=40),
    ]
    config = ServingConfig(cache_capacity_tokens=10_000, max_concurrent=1,
                           prefill_tps=100.0, decode_tps=100.0)

    def mean_wait(sim: ServingSim) -> float:
        return sum(r.queue_delay for r in sim.collector.records) / len(sim.collector.records)

    fifo = ServingSim(config)
    fifo.submit_all(requests)
    fifo.run()
    sjf = ServingSim(config, admission=ShortestJobAdmission())
    sjf.submit_all(requests)
    sjf.run()
    assert mean_wait(sjf) < mean_wait(fifo)


def test_non_fifo_policy_skips_cache_blocked_head() -> None:
    """非 FIFO 准入可跳过被缓存容量阻塞的队头，准入后续候选。"""
    # 容量只够 small（prefill 150+40）；huge 阻塞队头但 small 可入
    requests = [
        make_request("huge", 0.0, system=5000, new=4000, output=100),
        make_request("small", 0.1, system=100, new=50, output=40),
    ]
    config = ServingConfig(cache_capacity_tokens=500, max_concurrent=4,
                           prefill_tps=100.0, decode_tps=100.0)
    sim = ServingSim(config, admission=ShortestJobAdmission())
    sim.submit_all(requests)
    sim.run()
    assert len(sim.collector.records) == 2  # huge 走不缓存路径最终也完成
    records = {r.session_id: r for r in sim.collector.records}
    assert records["huge"].uncached  # 队头容量不可满足时以不缓存方式服务


# ---- M7：观察钩子与续链优先（FR-18）----

def test_admission_hooks_observe_admit_and_complete() -> None:
    """内核在准入成功与完成时回调准入策略的观察钩子。"""
    from ass.scheduler.admission import AdmissionPolicy

    class Recorder(AdmissionPolicy):
        name = "recorder"

        def __init__(self) -> None:
            self.admitted: list[tuple[str, float]] = []
            self.completed: list[str] = []

        def order(self, queue, now):
            return [queue[0]] if queue else []

        def on_admit(self, request, now):
            self.admitted.append((request.session_id, now))

        def on_complete(self, request, now):
            self.completed.append((request.session_id, now))

    recorder = Recorder()
    requests = [make_request("a", 0.0, 100, 50, 30), make_request("b", 5.0, 100, 50, 20)]
    sim = ServingSim(ServingConfig(cache_capacity_tokens=10_000, max_concurrent=1),
                     admission=recorder)
    sim.submit_all(requests)
    sim.run()
    assert [session for session, _ in recorder.admitted] == ["a", "b"]
    assert [session for session, _ in recorder.completed] == ["a", "b"]
    # 钩子时序：完成时刻 = 准入时刻 + 服务时长（a: 0 + 150/5000 + 30/200 = 0.18）
    assert recorder.admitted[0][1] == 0.0
    assert recorder.completed[0][1] == 0.18


def test_session_chain_orders_continuations_first() -> None:
    """已开头的会话（前缀已建）排在开新会话之前；续链内部最近完成者优先。"""
    policy = SessionChainAdmission()
    policy.on_admit(make_request("cold", 0.0, 100, 50, 30), 0.0)
    policy.on_complete(make_request("cold", 0.0, 100, 50, 30), 1.0)   # 1s 前完成
    policy.on_admit(make_request("hot", 0.0, 100, 50, 30), 2.0)
    policy.on_complete(make_request("hot", 0.0, 100, 50, 30), 5.0)    # 刚完成，前缀最热
    queue = [
        make_request("new1", 1.0, 100, 50, 30),
        make_request("cold", 2.0, 100, 50, 30, turn=2, history=80),
        make_request("new2", 3.0, 100, 50, 30),
        make_request("hot", 4.0, 100, 50, 30, turn=2, history=80),
    ]
    ordered = policy.order(queue, 5.0)
    # hot（最近完成）先于 cold（较早完成），二者都先于新会话
    assert [r.session_id for r in ordered] == ["hot", "cold", "new1", "new2"]


def test_session_chain_seen_but_unfinished_precedes_new_sessions() -> None:
    """已准入但尚未完成的会话（在途）仍先于新会话，内部按到达序。"""
    policy = SessionChainAdmission()
    policy.on_admit(make_request("old", 0.0, 100, 50, 30), 0.0)  # 在途，未完成
    queue = [
        make_request("new1", 1.0, 100, 50, 30),
        make_request("old", 2.0, 100, 50, 30, turn=2, history=80),
        make_request("new2", 3.0, 100, 50, 30),
    ]
    ordered = policy.order(queue, 3.0)
    assert [r.session_id for r in ordered] == ["old", "new1", "new2"]


def test_session_chain_degrades_to_fifo_without_seen_sessions() -> None:
    """无任何已见会话（如单轮无结构负载）时退化为纯 FIFO 序。"""
    policy = SessionChainAdmission()
    queue = [
        make_request("x", 2.0, 100, 50, 30),
        make_request("y", 1.0, 100, 50, 30),
        make_request("z", 0.0, 100, 50, 30),
    ]
    ordered = policy.order(queue, 5.0)
    assert [r.session_id for r in ordered] == ["z", "y", "x"]


def test_session_chain_admits_continuation_before_earlier_first_turn() -> None:
    """饱和排队下：后到达的续链请求先于更早到达的新会话首轮被准入。"""
    requests = [
        # old 的前两轮快速完成（策略由此看到 old 会话已开头）
        make_request("old", 0.0, system=100, new=50, output=30, turn=1),
        make_request("old", 0.5, system=100, new=40, output=20, turn=2, history=80),
        # blocker 占住唯一槽位至 ~2.25s，new 与 old 第三轮在队列中等待
        make_request("blocker", 1.0, system=100, new=50, output=30),
        make_request("new", 1.1, system=100, new=50, output=30),
        make_request("old", 1.15, system=100, new=50, output=30, turn=3, history=200),
    ]
    config = ServingConfig(cache_capacity_tokens=10_000, max_concurrent=1,
                           prefill_tps=1000.0, decode_tps=100.0)
    sim = ServingSim(config, admission=SessionChainAdmission())
    sim.submit_all(requests)
    sim.run()
    order = [record.session_id for record in sim.collector.records]
    assert order.index("old") < order.index("new")  # 续链先于新会话
    records = {r.session_id: r for r in sim.collector.records}
    assert records["new"].queue_delay > records["old"].queue_delay
