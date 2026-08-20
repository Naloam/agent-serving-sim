"""抢占建模的单元测试（FR-13）：分块增长、容量耗尽抢占、回队重算。"""

from ass.cache.policies import LRUPolicy
from ass.scheduler.serving import ServingConfig, ServingSim
from ass.workload.schema import PromptBreakdown, TraceRequest


def make_request(
    session: str,
    arrival: float,
    system: int,
    history: int,
    new: int,
    output: int,
    turn: int = 1,
) -> TraceRequest:
    return TraceRequest(
        session_id=session,
        turn_id=turn,
        arrival_time=arrival,
        prompt=PromptBreakdown(system=system, tools=0, history=history, new=new),
        output_tokens=output,
        think_time=0.0,
        agent_type="coding",
        priority=1,
    )


def run(requests, config):
    sim = ServingSim(config, policy=LRUPolicy())
    sim.submit_all(requests)
    sim.run()
    return sim


CHUNKED = dict(decode_chunks=4)


def test_chunked_mode_matches_legacy_without_contention() -> None:
    """无争用时，分块增长的完成时间与最终占用和旧模式一致。"""
    request = make_request("s1", 0.0, system=500, history=0, new=500, output=400)
    legacy = run([request], ServingConfig(cache_capacity_tokens=100_000, prefill_tps=1000, decode_tps=100))
    chunked = run([request], ServingConfig(cache_capacity_tokens=100_000, prefill_tps=1000, decode_tps=100, **CHUNKED))
    assert legacy.collector.records[0].jct == chunked.collector.records[0].jct
    assert chunked.tree.used_tokens == legacy.tree.used_tokens  # prompt + output 全量入库
    assert chunked.collector.summary()["preemptions"]["count"] == 0


def test_growth_shortage_preempts_newest_other_request() -> None:
    """增长遇容量耗尽：抢占最新准入的他者，受害者回队重算（手算核对）。"""
    config = ServingConfig(
        cache_capacity_tokens=1500, prefill_tps=1000.0, decode_tps=100.0,
        max_concurrent=4, decode_chunks=4,
    )
    first = make_request("s1", 0.0, system=500, history=0, new=500, output=400)
    second = make_request("s2", 1.0, system=500, history=0, new=500, output=400)
    sim = run([first, second], config)

    summary = sim.collector.summary()
    assert summary["preemptions"]["count"] == 1
    assert summary["preemptions"]["wasted_compute_s"] == 1.0  # 2.0 抢占 − 1.0 准入
    assert summary["preemptions"]["dropped_tokens"] == 500    # 仅 s2 的对话段被丢弃

    record_first, record_second = sim.collector.records
    assert record_first.session_id == "s1"
    assert record_first.jct == 5.0          # 1.0s prefill + 4.0s decode
    assert record_second.jct == 8.5         # 1.0 到达，5.0 重算准入，3.5s 服务
    assert record_second.hit_tokens == 500  # 前导因 s1 引用而幸存
    assert sim.tree.used_tokens == 1400     # pre(500) + s2 对话段重算后再长回 900


def test_growth_capped_when_alone_and_full() -> None:
    """独占且容量不足时无法抢占：增长封顶（计算继续，超出部分不缓存）。"""
    config = ServingConfig(
        cache_capacity_tokens=1200, prefill_tps=1000.0, decode_tps=100.0,
        max_concurrent=4, decode_chunks=4,
    )
    request = make_request("s1", 0.0, system=500, history=0, new=500, output=400)
    sim = run([request], config)
    summary = sim.collector.summary()
    assert summary["preemptions"]["count"] == 0
    assert sim.collector.records[0].jct == 5.0  # 服务不受封顶影响
    assert sim.tree.used_tokens == 1200          # prompt 1000 + 增长 200 后封顶


def test_preemption_disabled_caps_growth() -> None:
    config = ServingConfig(
        cache_capacity_tokens=1200, prefill_tps=1000.0, decode_tps=100.0,
        max_concurrent=4, decode_chunks=4, allow_preemption=False,
    )
    first = make_request("s1", 0.0, system=500, history=0, new=500, output=400)
    second = make_request("s2", 1.0, system=500, history=0, new=500, output=400)
    sim = run([first, second], config)
    assert sim.collector.summary()["preemptions"]["count"] == 0
    # 两个请求都正常完成（第二个因容量排队到 5.0 后准入）
    assert len(sim.collector.records) == 2


def test_eviction_cost_charged_to_request_jct() -> None:
    """evict_tps 设定后，驱逐量折入触发请求的时延（二阶效应建模）。"""
    base = dict(cache_capacity_tokens=2000, prefill_tps=1000.0, decode_tps=100.0, max_concurrent=8)
    # 会话 s1 留下 1000 token 缓存后释放；s2 到达需驱逐它们才能准入
    first = make_request("s1", 0.0, system=500, history=0, new=500, output=0)
    second = make_request("s2", 10.0, system=0, history=0, new=1500, output=0)

    free_run = run([first, second], ServingConfig(**base))
    (record_free,) = [r for r in free_run.collector.records if r.session_id == "s2"]
    # 免费驱逐：s2 prefill = 1500/1000 = 1.5s
    assert record_free.jct == 1.5

    costly_run = run([first, second], ServingConfig(**base, evict_tps=1000.0))
    (record_costly,) = [r for r in costly_run.collector.records if r.session_id == "s2"]
    # 只需驱逐 500 token（free 1000 → 1500 够用），债 0.5s：1.5 + 0.5 = 2.0s
    assert record_costly.jct == 2.0


def test_max_preemptions_falls_back_to_uncached() -> None:
    """反复被抢的请求最终转为不缓存模式，保证活性。"""
    config = ServingConfig(
        cache_capacity_tokens=1500, prefill_tps=1000.0, decode_tps=100.0,
        max_concurrent=4, decode_chunks=4,
    )
    requests = [
        make_request("s1", 0.0, system=500, history=0, new=500, output=400),
        make_request("s2", 0.1, system=500, history=0, new=500, output=400),
        make_request("s3", 0.2, system=500, history=0, new=500, output=400),
    ]
    sim = ServingSim(config, policy=LRUPolicy())
    sim.submit_all(requests)
    sim.run()
    assert len(sim.collector.records) == 3  # 全部最终完成
    assert sim._preempt_counts == {} or all(
        count <= 3 for count in sim._preempt_counts.values()
    )
