"""Radix tree 的单元测试（FR-5）：命中、分裂、引用保护、容量计费。"""

import pytest

from ass.cache.radix import NodeMeta, RadixTree, Segment


def segs(*pairs: tuple[str, int]) -> list[Segment]:
    return [Segment(stream, length) for stream, length in pairs]


def test_empty_tree_match_returns_zero() -> None:
    tree = RadixTree(capacity_tokens=1000)
    result = tree.match(segs(("a", 10)))
    assert result.hit_tokens == 0
    assert result.path == []


def test_insert_then_full_match() -> None:
    tree = RadixTree(capacity_tokens=1000)
    tree.insert(segs(("a", 5), ("b", 7)), now=1.0)
    result = tree.match(segs(("a", 5), ("b", 7)), now=2.0)
    assert result.hit_tokens == 12
    assert tree.used_tokens == 12


def test_extending_key_partial_hit() -> None:
    """会话对话流延伸：长 key 命中已缓存的短前缀（PRD 验收：命中计算）。"""
    tree = RadixTree(capacity_tokens=1000)
    tree.insert(segs(("a", 5), ("s", 7)), now=1.0)
    result = tree.match(segs(("a", 5), ("s", 11)), now=2.0)
    assert result.hit_tokens == 12
    assert tree.used_tokens == 12


def test_divergent_stream_stops_matching() -> None:
    tree = RadixTree(capacity_tokens=1000)
    tree.insert(segs(("a", 5), ("b", 7)), now=1.0)
    result = tree.match(segs(("a", 5), ("c", 3)), now=2.0)
    assert result.hit_tokens == 5


def test_shared_prefix_split() -> None:
    """共享前缀分裂（PRD 验收项）：短 key 插入后两个 key 都可完整命中。"""
    tree = RadixTree(capacity_tokens=1000)
    tree.insert(segs(("a", 10)), now=1.0)
    tree.insert(segs(("a", 4), ("b", 6)), now=2.0)
    assert tree.used_tokens == 16  # a(4) + a(6 尾段) + b(6)
    assert tree.match(segs(("a", 10)), now=3.0).hit_tokens == 10
    assert tree.match(segs(("a", 4), ("b", 6)), now=3.0).hit_tokens == 10


def test_chained_same_stream_after_split() -> None:
    """分裂后同 stream 链式续接的 key 仍能正确命中。"""
    tree = RadixTree(capacity_tokens=1000)
    tree.insert(segs(("s", 100)), now=1.0)
    tree.insert(segs(("s", 60), ("t", 5)), now=2.0)
    assert tree.match(segs(("s", 80)), now=3.0).hit_tokens == 80
    assert tree.match(segs(("s", 60), ("t", 5)), now=3.0).hit_tokens == 65


def test_pin_protects_node_from_eviction() -> None:
    """引用保护（PRD 验收项）：pin 的节点不可淘汰。"""
    tree = RadixTree(capacity_tokens=1000)
    tree.insert(segs(("a", 10)), now=1.0, pin=True)
    assert tree.evictable_leaves() == []
    with pytest.raises(ValueError, match="not evictable"):
        tree.evict(tree._root.children["a"])
    tree.release(tree._root.children.values())
    assert len(tree.evictable_leaves()) == 1


def test_evict_non_leaf_raises() -> None:
    tree = RadixTree(capacity_tokens=1000)
    tree.insert(segs(("a", 5), ("b", 7)), now=1.0)
    with pytest.raises(ValueError, match="not evictable"):
        tree.evict(tree._root.children["a"])


def test_evict_frees_tokens() -> None:
    tree = RadixTree(capacity_tokens=1000)
    tree.insert(segs(("a", 5), ("b", 7)), now=1.0)
    leaf = tree._root.children["a"].children["b"]
    freed = tree.evict(leaf)
    assert freed == 7
    assert tree.used_tokens == 5
    # 父节点成为新的可淘汰叶子
    assert tree.evictable_leaves() == [tree._root.children["a"]]


def test_release_without_pin_raises() -> None:
    tree = RadixTree(capacity_tokens=1000)
    tree.insert(segs(("a", 5)), now=1.0)
    with pytest.raises(ValueError, match="not pinned"):
        tree.release([tree._root.children["a"]])


def test_insert_beyond_capacity_raises() -> None:
    tree = RadixTree(capacity_tokens=10)
    tree.insert(segs(("a", 8)), now=1.0)
    with pytest.raises(ValueError, match="only"):
        tree.insert(segs(("b", 5)), now=2.0)


def test_match_updates_last_access() -> None:
    tree = RadixTree(capacity_tokens=1000)
    tree.insert(segs(("a", 5)), now=1.0)
    assert tree._root.children["a"].last_access == 1.0
    tree.match(segs(("a", 5)), now=9.0)
    assert tree._root.children["a"].last_access == 9.0


def test_meta_recorded_on_nodes() -> None:
    tree = RadixTree(capacity_tokens=1000)
    tree.insert(segs(("a", 5)), now=1.0, meta=NodeMeta(priority=3, agent_type="coding"))
    node = tree._root.children["a"]
    assert node.priority == 3
    assert node.agent_type == "coding"


def test_zero_length_segments_ignored() -> None:
    tree = RadixTree(capacity_tokens=1000)
    tree.insert(segs(("a", 5), ("b", 0)), now=1.0)
    assert tree.used_tokens == 5
    assert tree.match(segs(("a", 0)), now=2.0).hit_tokens == 0
