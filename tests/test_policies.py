"""驱逐策略的单元测试（FR-6）：四种策略排序语义 + 注册点（US-3）。"""

import pytest

from ass.cache.policies import (
    FIFOPolicy,
    LRUPolicy,
    PriorityPolicy,
    TTLPolicy,
    create_policy,
    register_policy,
)
from ass.cache.radix import NodeMeta, RadixTree, Segment


def build_tree() -> RadixTree:
    """三段独立缓存：a(10) coding / b(20) search / c(30) coding。"""
    tree = RadixTree(capacity_tokens=1000)
    tree.insert([Segment("a", 10)], now=1.0, meta=NodeMeta(agent_type="coding"))
    tree.insert([Segment("b", 20)], now=2.0, meta=NodeMeta(agent_type="search"))
    tree.insert([Segment("c", 30)], now=3.0, meta=NodeMeta(agent_type="coding"))
    return tree


def streams(nodes: list) -> list[str]:
    return [node.segment.stream for node in nodes]


def test_fifo_evicts_oldest_first() -> None:
    tree = build_tree()
    victims = FIFOPolicy().select_victims(tree, need_tokens=25, now=4.0)
    assert streams(victims) == ["a", "b", "c"]


def test_lru_evicts_least_recently_touched_first() -> None:
    tree = build_tree()
    tree.match([Segment("c", 30)], now=9.0)  # touch c
    victims = LRUPolicy().select_victims(tree, need_tokens=25, now=10.0)
    assert streams(victims) == ["a", "b", "c"]


def test_ttl_prefers_expired_then_lru() -> None:
    tree = build_tree()
    policy = TTLPolicy(ttl=1.5)
    # now=3.6: a(1.0) 与 b(2.0) 已过期（间隔 > 1.5），c(3.0) 未过期
    victims = policy.select_victims(tree, need_tokens=60, now=3.6)
    assert streams(victims) == ["a", "b", "c"]
    # TTL 极大时退化为 LRU
    victims = TTLPolicy(ttl=1e9).select_victims(tree, need_tokens=60, now=3.6)
    assert streams(victims) == ["a", "b", "c"]


def test_ttl_boundary_is_strict() -> None:
    """间隔恰好等于 ttl 不算过期（严格大于才过期）。"""
    tree = build_tree()
    policy = TTLPolicy(ttl=1.0)
    # now=3.0：a 间隔 2.0（过期）、b 间隔 1.0（恰等于 ttl，未过期）、c 间隔 0
    victims = policy.select_victims(tree, need_tokens=60, now=3.0)
    assert victims[0].segment.stream == "a"
    assert victims[1].segment.stream == "b"  # LRU 兜底顺序仍在 c 前
    # sweep 语义：b 在间隔恰为 ttl 时不被清除
    tree2 = build_tree()
    assert tree2.sweep_expired(now=3.0, ttl=1.0) == 10  # 仅 a 过期


def test_priority_evicts_low_weight_first() -> None:
    tree = build_tree()
    policy = PriorityPolicy(agent_weights={"coding": 2.0, "search": 1.0})
    victims = policy.select_victims(tree, need_tokens=60, now=4.0)
    assert streams(victims) == ["b", "a", "c"]


def test_priority_unweighted_defaults_keep_order() -> None:
    tree = build_tree()
    victims = PriorityPolicy().select_victims(tree, need_tokens=60, now=4.0)
    assert streams(victims) == ["a", "b", "c"]


def test_pinned_leaves_never_selected() -> None:
    tree = build_tree()
    tree.match([Segment("a", 10)], now=1.5, pin=True)
    victims = LRUPolicy().select_victims(tree, need_tokens=60, now=4.0)
    assert "a" not in streams(victims)


def test_create_policy_registry_roundtrip() -> None:
    assert create_policy("lru").name == "lru"
    ttl = create_policy("ttl", ttl=7.5)
    assert isinstance(ttl, TTLPolicy)
    assert ttl.ttl == 7.5
    with pytest.raises(KeyError, match="unknown policy"):
        create_policy("nope")


def test_custom_policy_registers_without_kernel_change() -> None:
    """US-3：新策略 = 新类 + 注册点，不改内核。"""

    @register_policy
    class NewestFirst(FIFOPolicy):
        name = "newest-first-test"

        def select_victims(self, tree, need_tokens, now):
            return sorted(tree.evictable_leaves(), key=lambda node: -node.created_seq)

    tree = build_tree()
    policy = create_policy("newest-first-test")
    victims = policy.select_victims(tree, need_tokens=60, now=4.0)
    assert streams(victims) == ["c", "b", "a"]


def test_duplicate_registration_rejected() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @register_policy
        class Duplicate(LRUPolicy):
            name = "lru"


def test_quota_policy_prefers_over_quota_types() -> None:
    """配额策略：超额类型先驱逐，未超额类型受保护。"""
    from ass.cache.policies import QuotaPolicy

    tree = RadixTree(capacity_tokens=10000)
    # coding 用量 600（配额 1000，未超）；search 用量 2600（配额 800，超 1800）
    tree.insert([Segment("c1", 300)], now=1.0, meta=NodeMeta(agent_type="coding"))
    tree.insert([Segment("c2", 300)], now=2.0, meta=NodeMeta(agent_type="coding"))
    tree.insert([Segment("s1", 1300)], now=3.0, meta=NodeMeta(agent_type="search"))
    tree.insert([Segment("s2", 1300)], now=4.0, meta=NodeMeta(agent_type="search"))

    policy = QuotaPolicy(quotas={"coding": 1000, "search": 800})
    victims = policy.select_victims(tree, need_tokens=3000, now=5.0)
    # 两个 search 叶子排最前（超配额）；coding 按 LRU 兜底在后
    assert streams(victims[:2]) == ["s1", "s2"]
    assert sorted(streams(victims[2:])) == ["c1", "c2"]


def test_quota_unknown_type_treated_as_unconstrained() -> None:
    from ass.cache.policies import QuotaPolicy

    tree = RadixTree(capacity_tokens=10000)
    tree.insert([Segment("x1", 500)], now=1.0, meta=NodeMeta(agent_type="other"))
    tree.insert([Segment("s1", 500)], now=2.0, meta=NodeMeta(agent_type="search"))
    victims = QuotaPolicy(quotas={"search": 100}).select_victims(tree, 1000, now=3.0)
    assert victims[0].segment.stream == "s1"  # search 超额先走


def test_quota_rejects_non_positive_quota() -> None:
    from ass.cache.policies import QuotaPolicy

    with pytest.raises(ValueError, match="must be positive"):
        QuotaPolicy(quotas={"coding": 0})


def test_quota_registered_in_registry() -> None:
    from ass.cache.policies import QuotaPolicy

    policy = create_policy("quota", quotas={"coding": 100})
    assert isinstance(policy, QuotaPolicy)


def test_belady_evicts_farthest_reuse_first() -> None:
    """Belady：不再复用的叶子最先驱逐，其次是最晚复用的。"""
    from ass.cache.policies import BeladyPolicy
    from ass.workload.schema import PromptBreakdown, TraceRequest

    def request(session: str, arrival: float, dialogue: int) -> TraceRequest:
        return TraceRequest(
            session_id=session, turn_id=1, arrival_time=arrival,
            prompt=PromptBreakdown(system=0, tools=0, history=0, new=dialogue),
            output_tokens=0, think_time=0.0, agent_type="coding", priority=1,
        )

    tree = RadixTree(capacity_tokens=10000)
    tree.insert([Segment("sess:a", 100)], now=1.0, meta=NodeMeta(agent_type="coding"))
    tree.insert([Segment("sess:b", 100)], now=2.0, meta=NodeMeta(agent_type="coding"))
    tree.insert([Segment("sess:c", 100)], now=3.0, meta=NodeMeta(agent_type="coding"))
    # 未来：a 在 t=100 复用，b 永不复用，c 在 t=50 复用
    oracle = [request("a", 100.0, 150), request("c", 50.0, 120)]
    victims = BeladyPolicy(oracle).select_victims(tree, need_tokens=300, now=4.0)
    assert [node.segment.stream for node in victims] == ["sess:b", "sess:a", "sess:c"]


def test_belady_handles_chained_same_stream_positions() -> None:
    """分裂产生的同流链：流内起点按同流祖先段求和。"""
    from ass.cache.policies import BeladyPolicy
    from ass.workload.schema import PromptBreakdown, TraceRequest

    tree = RadixTree(capacity_tokens=10000)
    tree.insert([Segment("sess:s", 100)], now=1.0, meta=NodeMeta(agent_type="coding"))
    tree.insert([Segment("sess:s", 60), Segment("t", 5)], now=2.0, meta=NodeMeta(agent_type="coding"))
    # 链：sess:s(60) -> sess:s(40 尾段)；未来仅有一个长度 80 的同流访问
    oracle = [
        TraceRequest(
            session_id="s", turn_id=1, arrival_time=10.0,
            prompt=PromptBreakdown(system=0, tools=0, history=80, new=0),
            output_tokens=0, think_time=0.0, agent_type="coding", priority=1,
        )
    ]
    policy = BeladyPolicy(oracle)
    # 尾段 s(40) 的流内起点是 60：未来长度 80 覆盖它 → 有限复用时间
    tail = tree._root.children["sess:s"].children["sess:s"]
    assert policy._in_stream_start(tail) == 60
    assert policy._next_reuse(tail, now=3.0) == 10.0


def test_belady_registered_offline() -> None:
    from ass.cache.policies import BeladyPolicy, create_policy

    policy = create_policy("belady", trace=[])
    assert isinstance(policy, BeladyPolicy)


def test_class_ttl_per_type_expiry() -> None:
    """类级 TTL：慢类未过期而快类已过期时，快类先走。"""
    from ass.cache.policies import ClassTTLPolicy

    tree = RadixTree(capacity_tokens=10000)
    tree.insert([Segment("f1", 100)], now=1.0, meta=NodeMeta(agent_type="fast"))
    tree.insert([Segment("s1", 100)], now=1.0, meta=NodeMeta(agent_type="slow"))
    # now=6: fast 间隔 5 > ttl 4（过期），slow 间隔 5 < ttl 60（未过期）
    victims = ClassTTLPolicy(ttls={"fast": 4.0, "slow": 60.0}).select_victims(tree, 200, 6.0)
    assert victims[0].segment.stream == "f1"
    # 未指定的类型不过期，仅按 LRU 兜底
    tree.insert([Segment("x1", 100)], now=0.5, meta=NodeMeta(agent_type="other"))
    victims = ClassTTLPolicy(ttls={"fast": 4.0}).select_victims(tree, 300, 6.0)
    assert victims[0].segment.stream == "f1"


def test_class_ttl_rejects_nonpositive() -> None:
    from ass.cache.policies import ClassTTLPolicy

    with pytest.raises(ValueError, match="must be positive"):
        ClassTTLPolicy(ttls={"fast": 0})


def test_on_admit_hook_called_by_serving() -> None:
    """serving 在每次准入时回调 on_admit（在线策略的唯一观测入口）。"""
    from ass.scheduler.serving import ServingConfig, ServingSim
    from ass.workload.schema import PromptBreakdown, TraceRequest

    class Spy(LRUPolicy):
        def __init__(self):
            super().__init__()
            self.calls: list[tuple[str, int, float]] = []

        def on_admit(self, request, now):
            self.calls.append((request.session_id, request.turn_id, now))

    request = TraceRequest(
        session_id="s1", turn_id=1, arrival_time=0.0,
        prompt=PromptBreakdown(system=100, tools=0, history=0, new=100),
        output_tokens=10, think_time=0.0, agent_type="coding", priority=1,
    )
    spy = Spy()
    sim = ServingSim(ServingConfig(cache_capacity_tokens=10_000), policy=spy)
    sim.submit_all([request])
    sim.run()
    assert spy.calls == [("s1", 1, 0.0)]


def test_predictive_policy_mrl_orders_slow_class_first() -> None:
    """MRL 排序：同空闲下，类条件期望回归更远的（慢类）先驱逐。"""
    from ass.cache.policies import PredictivePolicy
    from ass.workload.schema import PromptBreakdown, TraceRequest

    def admit_request(session, turn, arrival, think, agent):
        return TraceRequest(
            session_id=session, turn_id=turn, arrival_time=arrival,
            prompt=PromptBreakdown(system=0, tools=0, history=0, new=50),
            output_tokens=10, think_time=think, agent_type=agent, priority=1,
        )

    policy = PredictivePolicy(warmup=5, rank_by="residual")
    for i in range(10):
        policy.on_admit(admit_request(f"f{i:02d}", 2, 0.0, 5.0, "fast"), 5.0)
        policy.on_admit(admit_request(f"s{i:02d}", 2, 0.0, 30.0, "slow"), 30.0)

    tree = RadixTree(capacity_tokens=100000)
    tree.insert([Segment("sess:fast1", 50)], now=10.0, meta=NodeMeta(agent_type="fast"))
    tree.insert([Segment("sess:slow1", 50)], now=10.0, meta=NodeMeta(agent_type="slow"))
    # 叶子对应的会话必须经 on_admit 注册，策略才有其类归属与空闲信息
    policy.on_admit(admit_request("fast1", 2, 10.0, 5.0, "fast"), 10.0)
    policy.on_admit(admit_request("slow1", 2, 10.0, 30.0, "slow"), 10.0)
    victims = policy.select_victims(tree, 100, now=16.0)  # 同空闲 6s
    # 慢类 think≈30（期望剩余大）先逐，快类 think≈5（3s 内回归）保留
    assert victims[0].segment.stream == "sess:slow1"


def test_predictive_policy_cold_start_is_lru_like() -> None:
    """冷启动（样本 < warmup）不崩溃，退化为接近 LRU 的排序。"""
    from ass.cache.policies import PredictivePolicy

    policy = PredictivePolicy(warmup=20)
    tree = RadixTree(capacity_tokens=10000)
    tree.insert([Segment("sess:old", 50)], now=1.0, meta=NodeMeta(agent_type="coding"))
    tree.insert([Segment("sess:new", 50)], now=9.0, meta=NodeMeta(agent_type="coding"))
    victims = policy.select_victims(tree, 100, now=10.0)
    assert victims[0].segment.stream == "sess:old"  # 同分按 LRU 兜底


def test_predictive_policy_preamble_kept() -> None:
    from ass.cache.policies import PredictivePolicy

    policy = PredictivePolicy(warmup=1)
    tree = RadixTree(capacity_tokens=10000)
    tree.insert([Segment("agent:code", 500)], now=1.0, meta=NodeMeta(agent_type="coding"))
    tree.insert([Segment("sess:x", 50)], now=1.0, meta=NodeMeta(agent_type="coding"))
    victims = policy.select_victims(tree, 550, now=2.0)
    assert victims[0].segment.stream == "sess:x"


def test_predictive_policy_rank_by_validation() -> None:
    from ass.cache.policies import PredictivePolicy

    with pytest.raises(ValueError, match="rank_by"):
        PredictivePolicy(rank_by="bogus")
    assert PredictivePolicy(rank_by="return").rank_by == "return"


def test_mean_residual_life_u_shaped() -> None:
    """对数正态风险率先增后减：MRL 先降后升（U 型，尾段单调上升）。"""
    import math
    import random

    from ass.cache.policies import PredictivePolicy, _LognormalOnline

    stats = _LognormalOnline()
    rng = random.Random(7)
    for _ in range(200):
        stats.add(math.log(rng.lognormvariate(1.61, 0.5)))
    values = [PredictivePolicy._mean_residual(x, stats) for x in (0.5, 1.0, 5.0, 20.0, 60.0)]
    assert values[0] > values[1] >= values[2], values                        # 初段下降
    assert all(a <= b + 1e-6 for a, b in zip(values[2:], values[3:])), values  # 尾段上升


def test_weighted_lru_interpolates_between_lru_and_strict_priority() -> None:
    """带权 LRU：权重 ->1 等于 LRU；大权重保护慢回转类。"""
    from ass.cache.policies import WeightedLRUPolicy

    tree = RadixTree(capacity_tokens=10000)
    # coding 空闲 30s（last=0，now=30）、search 空闲 5s（last=25）
    tree.insert([Segment("c1", 500)], now=0.0, meta=NodeMeta(agent_type="coding"))
    tree.insert([Segment("s1", 500)], now=25.0, meta=NodeMeta(agent_type="search"))

    # 权重 1：纯 LRU，coding（更久未用）先走
    victims = WeightedLRUPolicy(agent_weights={"coding": 1.0, "search": 1.0}).select_victims(tree, 1000, 30.0)
    assert victims[0].segment.stream == "c1"
    # 权重 8：coding 有效年龄 30/8=3.75 < search 5 → search 先走
    victims = WeightedLRUPolicy(agent_weights={"coding": 8.0, "search": 1.0}).select_victims(tree, 1000, 30.0)
    assert victims[0].segment.stream == "s1"
