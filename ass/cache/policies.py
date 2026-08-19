"""可插拔驱逐策略（对应 PRD FR-6）。

统一接口：``select_victims(tree, need_tokens, now) -> list[RadixNode]``，
返回按驱逐偏好排序的候选叶子（调用方按序驱逐、够了即停，避免过度驱逐）。

新策略接入（US-3）：继承 :class:`EvictionPolicy` 并用
:class:`register_policy` 装饰即可，通过 ``create_policy(name, **kwargs)``
实例化，无需改动模拟器内核。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping

from ass.cache.radix import RadixNode, RadixTree


class EvictionPolicy(ABC):
    """驱逐策略基类：在可淘汰叶子中按偏好排序。"""

    name: str = "abstract"

    @abstractmethod
    def select_victims(
        self, tree: RadixTree, need_tokens: int, now: float
    ) -> list[RadixNode]:
        """返回候选叶子列表，排序即驱逐优先级。"""


class FIFOPolicy(EvictionPolicy):
    """先入先出：最早创建的段先驱逐。"""

    name = "fifo"

    def select_victims(
        self, tree: RadixTree, need_tokens: int, now: float
    ) -> list[RadixNode]:
        return sorted(tree.evictable_leaves(), key=lambda node: node.created_seq)


class LRUPolicy(EvictionPolicy):
    """最近最少使用：最久未被触碰的段先驱逐。"""

    name = "lru"

    def select_victims(
        self, tree: RadixTree, need_tokens: int, now: float
    ) -> list[RadixNode]:
        return sorted(tree.evictable_leaves(), key=lambda node: node.last_access)


class TTLPolicy(EvictionPolicy):
    """TTL：条目距上次访问超过 ``ttl`` 即视为过期。

    语义由两部分构成（对标 Continuum/CacheTTL）：

    1. **主动清除**：serving 层在每个事件上调用
       ``tree.sweep_expired(now, policy.ttl)``，过期条目到点即被移除，
       之后到达的请求不能再命中它们；
    2. **压力兜底**：容量不足且已无过期条目时，``select_victims`` 按
       LRU 顺序补足victim（TTL 极大时整体退化为 LRU）。
    """

    name = "ttl"

    def __init__(self, ttl: float) -> None:
        if ttl <= 0:
            raise ValueError(f"ttl must be positive, got {ttl}")
        self.ttl = ttl

    def select_victims(
        self, tree: RadixTree, need_tokens: int, now: float
    ) -> list[RadixNode]:
        def order(node: RadixNode) -> tuple[int, float]:
            # 过期条目排最前（0），未过期按 LRU 兜底
            expired = 0 if (now - node.last_access) > self.ttl else 1
            return expired, node.last_access

        return sorted(tree.evictable_leaves(), key=order)


class PriorityPolicy(EvictionPolicy):
    """按 agent 权重驱逐：低权重（不重要）的段先走，同权重按 LRU。

    ``agent_weights`` 为 agent_type -> 权重（越大越保值），未列出的类型
    取 ``default_weight``。
    """

    name = "priority"

    def __init__(
        self,
        agent_weights: Mapping[str, float] | None = None,
        default_weight: float = 1.0,
    ) -> None:
        self.agent_weights = dict(agent_weights or {})
        self.default_weight = default_weight

    def weight_of(self, node: RadixNode) -> float:
        return self.agent_weights.get(node.agent_type, self.default_weight)

    def select_victims(
        self, tree: RadixTree, need_tokens: int, now: float
    ) -> list[RadixNode]:
        return sorted(
            tree.evictable_leaves(),
            key=lambda node: (self.weight_of(node), node.last_access),
        )


_POLICY_REGISTRY: dict[str, type[EvictionPolicy]] = {}


def register_policy(cls: type[EvictionPolicy]) -> type[EvictionPolicy]:
    """类装饰器：把策略类登记到全局注册点。"""
    name = getattr(cls, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError(f"policy class {cls.__name__} must define a non-empty 'name'")
    if name in _POLICY_REGISTRY:
        raise ValueError(f"policy name already registered: {name}")
    _POLICY_REGISTRY[name] = cls
    return cls


def create_policy(name: str, **kwargs: object) -> EvictionPolicy:
    """按注册名实例化策略（新策略接入的唯一入口）。"""
    if name not in _POLICY_REGISTRY:
        known = sorted(_POLICY_REGISTRY)
        raise KeyError(f"unknown policy {name!r}, registered: {known}")
    policy = _POLICY_REGISTRY[name](**kwargs)  # type: ignore[call-arg]
    if not isinstance(policy, EvictionPolicy):
        raise TypeError(f"{name!r} did not produce an EvictionPolicy")
    return policy


register_policy(FIFOPolicy)
register_policy(LRUPolicy)
register_policy(TTLPolicy)
register_policy(PriorityPolicy)
