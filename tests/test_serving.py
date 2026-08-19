"""请求生命周期调度的单元测试（FR-7）：手工核对单请求路径、排队、驱逐。"""

import pytest

from ass.cache.policies import LRUPolicy, TTLPolicy
from ass.scheduler.serving import ServingConfig, ServingSim
from ass.workload.schema import PromptBreakdown, TraceRequest

CONFIG = ServingConfig(
    cache_capacity_tokens=100_000,
    prefill_tps=1000.0,
    decode_tps=100.0,
    max_concurrent=8,
)


def make_request(
    session: str,
    turn: int,
    arrival: float,
    system: int,
    history: int,
    new: int,
    output: int,
    agent: str = "coding",
) -> TraceRequest:
    return TraceRequest(
        session_id=session,
        turn_id=turn,
        arrival_time=arrival,
        prompt=PromptBreakdown(system=system, tools=0, history=history, new=new),
        output_tokens=output,
        think_time=0.0,
        agent_type=agent,
        priority=1,
    )


def run(requests, config=CONFIG, policy=None):
    sim = ServingSim(config, policy=policy)
    sim.submit_all(requests)
    sim.run()
    return sim


def test_single_request_hand_computed() -> None:
    """单请求路径手工计算核对（PRD 验收项）。"""
    request = make_request("s1", 1, 0.0, system=500, history=0, new=500, output=200)
    sim = run([request])
    (record,) = sim.collector.records
    assert record.hit_tokens == 0
    assert record.ttft == pytest.approx(1.0)   # 1000 未命中 token / 1000 tps
    assert record.jct == pytest.approx(3.0)    # + 200 output / 100 tps
    assert record.completion_time == pytest.approx(3.0)
    assert sim.tree.used_tokens == 1200        # prompt 1000 + output 200
    assert sim.collector.summary()["hit_rate"] == 0.0


def test_second_turn_reuses_prefix() -> None:
    """同会话下一轮命中前导 + 历史前缀（跨轮 KV 复用）。"""
    first = make_request("s1", 1, 0.0, system=500, history=0, new=500, output=200)
    # history = 前轮 new + output = 700
    second = make_request("s1", 2, 10.0, system=500, history=700, new=300, output=100)
    sim = run([first, second])
    record = sim.collector.records[1]
    assert record.hit_tokens == 1200           # 500 前导 + 700 对话前缀
    assert record.ttft == pytest.approx(0.3)   # (1500 - 1200) / 1000
    assert record.jct == pytest.approx(1.3)    # + 100 / 100
    assert record.completion_time == pytest.approx(11.3)
    summary = sim.collector.summary()
    assert summary["hit_rate"] == pytest.approx(1200 / 2500)


def test_concurrency_limit_queues_requests() -> None:
    config = ServingConfig(
        cache_capacity_tokens=100_000, prefill_tps=1000.0, decode_tps=100.0, max_concurrent=1
    )
    first = make_request("s1", 1, 0.0, system=500, history=0, new=500, output=200)
    second = make_request("s2", 1, 0.0, system=500, history=0, new=500, output=100)
    sim = run([first, second], config=config)
    record_second = sim.collector.records[1]
    # 第二个请求等到 3.0 才被准入；同类型前导 500 已缓存 → prefill 仅 0.5
    assert record_second.admit_time == pytest.approx(3.0)
    assert record_second.hit_tokens == 500
    assert record_second.ttft == pytest.approx(3.0 + 0.5)
    assert record_second.jct == pytest.approx(3.0 + 0.5 + 1.0)
    assert sim.collector.summary()["queue_delay_mean"] == pytest.approx(1.5)


def test_capacity_shortage_queues_until_release() -> None:
    """容量被在途引用占满且无可淘汰 → 排队，完成后重试（PRD 验收项）。"""
    config = ServingConfig(
        cache_capacity_tokens=1500, prefill_tps=1000.0, decode_tps=100.0, max_concurrent=8
    )
    first = make_request("s1", 1, 0.0, system=500, history=0, new=500, output=200)
    # 同类型另一会话：共享 500 前导，另需 700 新 token；空闲仅 300
    other = make_request("s2", 1, 1.0, system=500, history=0, new=600, output=100)
    sim = run([first, other], config=config)
    record = sim.collector.records[1]
    assert record.admit_time == pytest.approx(3.0)      # 等首个请求 3.0 完成释放
    assert record.hit_tokens == 500                       # 前导仍被保留
    assert record.ttft == pytest.approx(2.0 + 0.6)       # 队列 2.0 + prefill 600/1000
    assert record.jct == pytest.approx(2.0 + 0.6 + 1.0)  # + decode 100/100
    assert sim.collector.summary()["evictions"]["count"] >= 1


def test_oversized_request_served_uncached() -> None:
    """请求超出总容量：不缓存直接服务，不阻塞后续。"""
    config = ServingConfig(
        cache_capacity_tokens=100, prefill_tps=1000.0, decode_tps=100.0, max_concurrent=8
    )
    big = make_request("s1", 1, 0.0, system=500, history=0, new=500, output=200)
    sim = run([big], config=config)
    (record,) = sim.collector.records
    assert record.uncached
    assert record.jct == pytest.approx(3.0)
    assert sim.tree.used_tokens == 0
    assert sim.collector.summary()["uncached_requests"] == 1


def test_ttl_expiry_turns_hit_into_miss() -> None:
    """TTL 过期后同会话下一轮不再命中（对照 LRU 的命中）。"""
    first = make_request("s1", 1, 0.0, system=500, history=0, new=500, output=200)
    second = make_request("s1", 2, 20.0, system=500, history=700, new=300, output=100)

    lru = run([first, second], policy=LRUPolicy())
    assert lru.collector.records[1].hit_tokens == 1200

    ttl = run([first, second], policy=TTLPolicy(ttl=5.0))
    assert ttl.collector.records[1].hit_tokens == 0
    assert ttl.collector.summary()["ttl_expired_tokens"] == 1200


def test_shared_preamble_across_sessions() -> None:
    """同 agent 类型的不同会话共享前导前缀。"""
    first = make_request("s1", 1, 0.0, system=500, history=0, new=500, output=200)
    other = make_request("s2", 1, 10.0, system=500, history=0, new=400, output=100)
    sim = run([first, other])
    assert sim.collector.records[1].hit_tokens == 500


def test_invalid_config_rejected() -> None:
    with pytest.raises(ValueError):
        ServingConfig(cache_capacity_tokens=0)
    with pytest.raises(ValueError):
        ServingConfig(cache_capacity_tokens=100, max_concurrent=0)
