"""抢占建模的单元测试（FR-13）：分块增长、容量耗尽抢占、回队重算。"""

from ass.cache.policies import LRUPolicy
from ass.cache.radix import RadixTree, Segment
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


def test_fixed_overhead_shifts_ttft_and_jct() -> None:
    """fixed_overhead_s（TTFT 截距）计入 TTFT 与完成时间，命中不受影响。"""
    base = dict(cache_capacity_tokens=100_000, prefill_tps=1000.0, decode_tps=100.0)
    request = make_request("s1", 0.0, system=500, history=0, new=500, output=200)

    plain = run([request], ServingConfig(**base))
    overhead = run([request], ServingConfig(**base, fixed_overhead_s=3.1))
    record_plain, record_overhead = plain.collector.records[0], overhead.collector.records[0]
    assert record_plain.ttft == 1.0
    assert record_overhead.ttft == 4.1   # 3.1s 固定开销 + 1.0s prefill
    assert record_plain.jct == 3.0
    assert record_overhead.jct == 6.1
    assert record_overhead.hit_tokens == record_plain.hit_tokens
    assert overhead.tree.used_tokens == plain.tree.used_tokens


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


def test_grow_does_not_overwrite_foreign_chain_child() -> None:
    """同会话轮次重叠：后续轮插入占据链式位置后，旧轮的 decode 增长
    不得顶掉该链节点（否则孤儿节点使 evict 抛 KeyError、树账目损坏）。

    场景来自 exp012 真实到达回放：think_time≈0 时同会话相邻轮次可在
    服务中重叠，轮 t+1 的插入与轮 t 的分块增长在同一 sess 流上交错。
    """
    tree = RadixTree(capacity_tokens=100_000)
    agent = Segment("agent:coding", 100)
    first_pins = tree.insert([agent, Segment("sess:s1", 200)], now=0.0, pin=True)
    leaf = first_pins[-1]

    leaf = tree.grow(leaf, 50)  # 就地延伸 → sess:s1(250)，仍是叶子
    assert leaf.segment.length == 250

    # 轮 t+1 插入更长的同 stream 键：在 leaf 下链式生成子节点并 pin
    second_pins = tree.insert([agent, Segment("sess:s1", 400)], now=1.0, pin=True)
    chain = second_pins[-1]
    assert chain.parent is leaf and chain.segment.length == 150

    # 修复点：leaf 已非叶子且同 stream 槽位被占，grow 必须放弃而非覆盖
    assert tree.grow(leaf, 50) is None
    assert leaf.children.get("sess:s1") is chain  # 链节点仍在树中

    # 释放后逐层驱逐不崩溃，token 账目守恒
    used_before = tree.used_tokens
    tree.release(first_pins)
    tree.release(second_pins)
    freed = sum(tree.evict(node) for node in tree.evictable_leaves())
    while freed < used_before:  # 驱逐腾出空间后上层节点变叶，循环清空
        more = sum(tree.evict(node) for node in tree.evictable_leaves())
        if not more:
            break
        freed += more
    assert tree.used_tokens == used_before - freed
    assert tree.used_tokens >= 0


def test_overlapping_same_session_turns_complete() -> None:
    """同会话相邻轮次在服务中重叠（零思考时间）+ 抢占：整体跑通不崩溃。"""
    config = ServingConfig(
        cache_capacity_tokens=1200, prefill_tps=1000.0, decode_tps=100.0,
        max_concurrent=2, decode_chunks=4,
    )
    requests = [
        make_request("s1", 0.0, system=500, history=0, new=100, output=400),
        # 轮 2：覆盖轮 1 的 prompt + 生成，插入将越过轮 1 当前的叶位置
        make_request("s1", 0.5, system=500, history=500, new=200, output=400, turn=2),
    ]
    sim = ServingSim(config, policy=LRUPolicy())
    sim.submit_all(requests)
    sim.run()
    assert len(sim.collector.records) == 2
    assert sim.tree.used_tokens >= 0
