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
