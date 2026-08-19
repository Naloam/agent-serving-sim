"""Radix tree KV cache 模型（对应 PRD FR-5）。

设计要点：

- **前缀段元素**：树以 :class:`Segment`（stream, length）为最小单位。同一
  ``stream`` 的 token 按位置对齐——任意短前缀是长前缀的前缀。模拟器无需
  真实 token 内容即可精确表达前缀复用结构（agent 前导流、会话对话流），
  同时保住 10 万请求级的性能（每请求仅常数个段）。
- **节点 = 前缀段**：按 token 数计费容量；部分命中时分裂节点（共享前缀
  分裂）。
- **引用计数**：在途请求 pin 的节点不可淘汰；释放时逐节点归还。
  节点分裂时尾段保留旧身份（引用计数、子树、FIFO 序），上半段视为新节点。
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass
from itertools import count
from typing import NamedTuple


class Segment(NamedTuple):
    """一段同源前缀 token 流（同 stream 按位置对齐）。"""

    stream: str
    length: int


@dataclass(frozen=True)
class NodeMeta:
    """insert 时附带在节点上的请求元数据（供优先级驱逐策略使用）。"""

    priority: int = 1
    agent_type: str = "default"


class MatchResult(NamedTuple):
    """前缀匹配结果：命中 token 数与覆盖路径上的节点。"""

    hit_tokens: int
    path: list["RadixNode"]


class RadixNode:
    """radix tree 节点：一个前缀段及其缓存记账信息。"""

    __slots__ = (
        "segment",
        "parent",
        "children",
        "refcount",
        "last_access",
        "created_seq",
        "priority",
        "agent_type",
    )

    def __init__(
        self,
        segment: Segment,
        parent: "RadixNode | None",
        now: float,
        seq: int,
        priority: int,
        agent_type: str,
    ) -> None:
        self.segment = segment
        self.parent = parent
        self.children: dict[str, RadixNode] = {}
        self.refcount = 0
        self.last_access = now
        self.created_seq = seq
        self.priority = priority
        self.agent_type = agent_type

    @property
    def token_count(self) -> int:
        return self.segment.length

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def evictable(self) -> bool:
        """仅根以外的、无引用的叶子节点可淘汰。"""
        return self.parent is not None and self.refcount == 0 and self.is_leaf


class RadixTree:
    """按 token 数计费容量的 radix tree。"""

    def __init__(self, capacity_tokens: int) -> None:
        if capacity_tokens <= 0:
            raise ValueError(f"capacity_tokens must be positive, got {capacity_tokens}")
        self.capacity_tokens = capacity_tokens
        self._root = RadixNode(Segment("", 0), None, 0.0, -1, 1, "root")
        self._used = 0
        self._seq = count()

    @property
    def used_tokens(self) -> int:
        return self._used

    @property
    def free_tokens(self) -> int:
        return self.capacity_tokens - self._used

    def match(self, key: Sequence[Segment], now: float = 0.0, pin: bool = False) -> MatchResult:
        """返回 key 在树中的最长前缀命中；``pin`` 时对覆盖路径加引用计数。"""
        covered, path = self._walk([seg for seg in key if seg.length > 0])
        for node in path:
            node.last_access = now
        if pin:
            for node in path:
                node.refcount += 1
        return MatchResult(covered, path)

    def insert(
        self,
        key: Sequence[Segment],
        now: float = 0.0,
        meta: NodeMeta | None = None,
        pin: bool = False,
    ) -> list[RadixNode]:
        """把 key 计入缓存（复用/延伸/分裂已有路径）。

        返回本次新物化的节点（新建节点与分裂出的上半段）；若 ``pin`` 为真
        则对这些节点加引用计数。与 ``match(pin=True)`` 搭配时，调用方应
        释放 match.path + 返回值的并集。
        """
        segments = [seg for seg in key if seg.length > 0]
        total = sum(seg.length for seg in segments)
        covered, _ = self._walk(list(segments))
        if self._used + (total - covered) > self.capacity_tokens:
            raise ValueError(
                f"insert needs {total - covered} tokens but only "
                f"{self.free_tokens} free (capacity={self.capacity_tokens})"
            )
        node = self._root
        materialized: list[RadixNode] = []
        idx = 0
        while idx < len(segments):
            seg = segments[idx]
            child = node.children.get(seg.stream)
            if child is None:
                child = self._spawn(seg, node, now, meta)
                materialized.append(child)
                node = child
                idx += 1
                continue
            clen = child.segment.length
            if seg.length == clen:
                self._touch(child, now, meta)
                node = child
                idx += 1
            elif seg.length > clen:
                self._touch(child, now, meta)
                node = child
                segments[idx] = Segment(seg.stream, seg.length - clen)
            else:
                # key 段短于已有节点：分裂出上半段，key 沿上半段继续
                top = self._split(child, seg.length)
                self._touch(top, now, meta)
                materialized.append(top)
                node = top
                idx += 1
        if pin:
            for fresh in materialized:
                fresh.refcount += 1
        return materialized

    def evict(self, node: RadixNode) -> int:
        """摘除一个可淘汰叶子节点，返回释放的 token 数。"""
        if not node.evictable:
            raise ValueError("node is not evictable (must be an unreferenced leaf)")
        del node.parent.children[node.segment.stream]
        self._used -= node.segment.length
        return node.segment.length

    def release(self, nodes: Iterable[RadixNode]) -> None:
        """归还一批 pin 的节点（引用计数减一）。"""
        for node in nodes:
            if node.refcount <= 0:
                raise ValueError("release called on a node that is not pinned")
            node.refcount -= 1

    def evictable_leaves(self) -> list[RadixNode]:
        """按深度优先（插入序）列出所有可淘汰叶子。"""
        return list(self._iter_leaves(self._root))

    def sweep_expired(self, now: float, ttl: float) -> int:
        """TTL 主动清除：摘除所有"距上次访问超过 ttl"的无引用叶子。

        后序遍历保证父节点在子树清空后若同样过期也一并回收；
        返回释放的 token 数。被在途请求引用的节点不受影响。
        """
        freed = 0

        def visit(node: RadixNode) -> None:
            nonlocal freed
            for child in list(node.children.values()):
                visit(child)
                if (
                    not child.children
                    and child.refcount == 0
                    and now - child.last_access > ttl
                ):
                    del node.children[child.segment.stream]
                    freed += child.segment.length
                    self._used -= child.segment.length

        visit(self._root)
        return freed

    # ---- 内部实现 ----

    def _iter_leaves(self, node: RadixNode) -> Iterator[RadixNode]:
        for child in node.children.values():
            if child.is_leaf:
                if child.refcount == 0:
                    yield child
            else:
                yield from self._iter_leaves(child)

    def _walk(self, segments: list[Segment]) -> tuple[int, list[RadixNode]]:
        """无副作用地计算最长前缀覆盖（注意：会消费传入的 segments 副本语义）。"""
        node = self._root
        covered = 0
        path: list[RadixNode] = []
        idx = 0
        while idx < len(segments):
            seg = segments[idx]
            child = node.children.get(seg.stream)
            if child is None:
                break
            clen = child.segment.length
            if seg.length > clen:
                covered += clen
                path.append(child)
                node = child
                segments[idx] = Segment(seg.stream, seg.length - clen)
            elif seg.length == clen:
                covered += clen
                path.append(child)
                node = child
                idx += 1
            else:
                covered += seg.length
                path.append(child)
                break
        return covered, path

    def _spawn(
        self, segment: Segment, parent: RadixNode, now: float, meta: NodeMeta | None
    ) -> RadixNode:
        node = RadixNode(
            segment=segment,
            parent=parent,
            now=now,
            seq=next(self._seq),
            priority=meta.priority if meta else 1,
            agent_type=meta.agent_type if meta else "default",
        )
        parent.children[segment.stream] = node
        self._used += segment.length
        return node

    def _split(self, node: RadixNode, keep: int) -> RadixNode:
        """把 node 拆成 keep 长度的上半段 + 尾段；尾段保留旧身份。

        返回新创建的上半段（已挂到原父节点下），token 总量不变。
        """
        stream, length = node.segment
        top = RadixNode(
            segment=Segment(stream, keep),
            parent=node.parent,
            now=node.last_access,
            seq=next(self._seq),
            priority=node.priority,
            agent_type=node.agent_type,
        )
        node.parent.children[stream] = top
        node.parent = top
        node.segment = Segment(stream, length - keep)
        top.children = {node.segment.stream: node}
        return top

    @staticmethod
    def _touch(node: RadixNode, now: float, meta: NodeMeta | None) -> None:
        node.last_access = now
        if meta is not None:
            node.priority = meta.priority
            node.agent_type = meta.agent_type
