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


class QuotaPolicy(EvictionPolicy):
    """多 agent cache 配额（对标 TokenCake 思路）：按类型的软容量份额。

    ``quotas`` 为 agent_type -> 配额（token 数）。驱逐偏好：当前用量超出
    配额比例最大的类型优先（超额部分先走），同比例内按 ``fallback``
    策略排序（默认 LRU）。配额是软约束——只改变驱逐顺序，不做准入拒绝，
    未设配额的类型视为不超额。

    用量按**可淘汰叶子**统计（被在途请求 pin 的部分不可见），是总用量的
    下界近似；serving 的逐个驱逐 + 重查询机制使比例每轮自动更新。
    """

    name = "quota"

    def __init__(
        self, quotas: Mapping[str, int], fallback: EvictionPolicy | None = None
    ) -> None:
        for agent_type, quota in quotas.items():
            if quota <= 0:
                raise ValueError(f"quota for {agent_type!r} must be positive, got {quota}")
        self.quotas = dict(quotas)
        self.fallback = fallback if fallback is not None else LRUPolicy()

    def select_victims(
        self, tree: RadixTree, need_tokens: int, now: float
    ) -> list[RadixNode]:
        leaves = tree.evictable_leaves()
        usage: dict[str, int] = {}
        for leaf in leaves:
            usage[leaf.agent_type] = usage.get(leaf.agent_type, 0) + leaf.token_count
        fallback_order = {
            id(node): rank
            for rank, node in enumerate(self.fallback.select_victims(tree, need_tokens, now))
        }

        def over_ratio(node: RadixNode) -> float:
            quota = self.quotas.get(node.agent_type)
            if not quota:
                return 0.0
            return max(0.0, usage.get(node.agent_type, 0) - quota) / quota

        return sorted(
            leaves,
            key=lambda node: (-over_ratio(node), fallback_order.get(id(node), len(leaves))),
        )


class WeightedLRUPolicy(EvictionPolicy):
    """带权 LRU：有效年龄 = 空闲时长 / 类权重（高权重类"老化慢"）。

    权重 → 1 时退化为纯 LRU，权重 → ∞ 时趋近严格类优先（高权重类
    几乎不被驱逐），中间值给出两者之间的**平滑插值**——不同于
    :class:`PriorityPolicy` 的严格类序（其权重只有序数意义）。
    """

    name = "wlru"

    def __init__(
        self,
        agent_weights: Mapping[str, float] | None = None,
        default_weight: float = 1.0,
    ) -> None:
        self.agent_weights = dict(agent_weights or {})
        self.default_weight = default_weight

    def weight_of(self, node: RadixNode) -> float:
        return max(self.agent_weights.get(node.agent_type, self.default_weight), 1e-9)

    def select_victims(
        self, tree: RadixTree, need_tokens: int, now: float
    ) -> list[RadixNode]:
        # 有效年龄 = 空闲时长 / 权重；驱逐最"老"的（降序）
        return sorted(
            tree.evictable_leaves(),
            key=lambda node: (node.last_access - now) / self.weight_of(node),
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
register_policy(QuotaPolicy)
register_policy(WeightedLRUPolicy)
