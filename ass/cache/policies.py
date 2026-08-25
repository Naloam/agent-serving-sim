"""可插拔驱逐策略。

统一接口：``select_victims(tree, need_tokens, now) -> list[RadixNode]``，
返回按驱逐偏好排序的候选叶子（调用方按序驱逐、够了即停，避免过度驱逐）。

新策略接入（US-3）：继承 :class:`EvictionPolicy` 并用
:class:`register_policy` 装饰即可，通过 ``create_policy(name, **kwargs)``
实例化，无需改动模拟器内核。
"""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Mapping

from ass.cache.radix import RadixNode, RadixTree

if TYPE_CHECKING:  # 仅为类型标注引入，避免运行期循环依赖
    from ass.workload.schema import TraceRequest


class EvictionPolicy(ABC):
    """驱逐策略基类：在可淘汰叶子中按偏好排序。

    需要在线学习的策略可覆盖 :meth:`on_admit`——serving 在每次请求
    准入时回调（默认 no-op），是策略获取负载特征的唯一观测入口。
    """

    name: str = "abstract"

    def on_admit(self, request: "TraceRequest", now: float) -> None:
        """观测钩子：请求准入时被调用（在线学习策略的数据入口）。"""

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


class BeladyPolicy(EvictionPolicy):
    """离线最优驱逐（Belady/MIN）：驱逐"未来最晚才被复用"的叶子。

    作为实验上限基线：构造时传入完整 trace，为每条前缀流建立
    ``(到达时间, 覆盖长度)`` 列表。叶子的下次复用时间 = 同流中首个
    "到达晚于当前时刻且覆盖长度越过该叶子流内起点"的请求；不再被
    复用的叶子视为无穷远、最先驱逐。真实系统无法在线获知未来，
    本策略只用于给出可达命中率/JCT 的理论上限参照。
    """

    name = "belady"

    def __init__(self, trace: "Sequence[TraceRequest]") -> None:  # type: ignore[name-defined]
        self._future: dict[str, list[tuple[float, int]]] = {}
        for request in trace:
            preamble = request.prompt.system + request.prompt.tools
            dialogue = request.prompt.history + request.prompt.new + request.output_tokens
            if preamble > 0:
                self._future.setdefault(f"agent:{request.agent_type}", []).append(
                    (request.arrival_time, preamble)
                )
            if dialogue > 0:
                self._future.setdefault(f"sess:{request.session_id}", []).append(
                    (request.arrival_time, dialogue)
                )
        for accesses in self._future.values():
            accesses.sort()

    def _in_stream_start(self, leaf: RadixNode) -> int:
        """叶子在其所属流内的起点位置（同流祖先段长度之和）。"""
        start = 0
        node = leaf.parent
        while node is not None:
            if node.segment.stream == leaf.segment.stream:
                start += node.segment.length
            node = node.parent
        return start

    def _next_reuse(self, leaf: RadixNode, now: float) -> float:
        accesses = self._future.get(leaf.segment.stream)
        if not accesses:
            return math.inf
        start = self._in_stream_start(leaf)
        for arrival, length in accesses:
            if arrival > now and length > start:
                return arrival
        return math.inf

    def select_victims(
        self, tree: RadixTree, need_tokens: int, now: float
    ) -> list[RadixNode]:
        return sorted(
            tree.evictable_leaves(),
            key=lambda leaf: -self._next_reuse(leaf, now),
        )


class ClassTTLPolicy(EvictionPolicy):
    """按 agent_type 的静态 TTL（预测的"类级"下界基线）。

    每类一个 ttl：该类叶子距上次访问超过类 TTL 即视为过期（优先驱逐），
    未过期按 LRU 兜底。未指定的类型按无穷大处理（不过期）。
    相比全局 TTL，它至少把"类间回转周期差异"利用了起来。
    """

    name = "class-ttl"

    def __init__(self, ttls: Mapping[str, float]) -> None:
        for agent_type, ttl in ttls.items():
            if ttl <= 0:
                raise ValueError(f"ttl for {agent_type!r} must be positive, got {ttl}")
        self.ttls = dict(ttls)

    def select_victims(
        self, tree: RadixTree, need_tokens: int, now: float
    ) -> list[RadixNode]:
        def order(node: RadixNode) -> tuple[int, float]:
            ttl = self.ttls.get(node.agent_type, math.inf)
            expired = 0 if (now - node.last_access) > ttl else 1
            return expired, node.last_access

        return sorted(tree.evictable_leaves(), key=order)


class _LognormalOnline:
    """对 log(x) 的在线均值/方差（Welford），用于 think_time 的对数正态拟合。"""

    __slots__ = ("n", "mean", "m2")

    def __init__(self) -> None:
        self.n = 0
        self.mean = 0.0
        self.m2 = 0.0

    def add(self, log_value: float) -> None:
        self.n += 1
        delta = log_value - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (log_value - self.mean)

    @property
    def std(self) -> float:
        return math.sqrt(self.m2 / self.n) if self.n > 1 else 0.0

    @property
    def p90(self) -> float:
        return math.exp(self.mean + 1.2816 * max(self.std, 1e-6))


class _SessionState:
    __slots__ = ("agent_type", "turn", "last_seen")

    def __init__(self, agent_type: str, turn: int, last_seen: float) -> None:
        self.agent_type = agent_type
        self.turn = turn
        self.last_seen = last_seen


def _lognormal_cdf(x: float, mu: float, sigma: float) -> float:
    z = (math.log(max(x, 1e-9)) - mu) / (sigma * math.sqrt(2.0))
    return 0.5 * (1.0 + math.erf(z))


class PredictivePolicy(EvictionPolicy):
    """在线预测型驱逐（M3.6，FR-14）。

    动机（exp007）：LRU 距 Belady 上限的缺口集中在慢回转类。本策略只用
    **在线因果信息**（经 ``on_admit`` 观测）：每类对 ``log(think_time)``
    做增量统计拟合对数正态，然后按类条件排序：

    - ``rank_by="return"``（默认）：按 ``P(H 窗口内回归 | 已空闲 x)`` 升序
      驱逐（H 缺省取该类 think 的 p90）。exp008 实测（合成 40K/真实 4K
      受压档）该排序收窄 LRU→Belady 缺口 20~30%，为在线策略中最优；
    - ``rank_by="residual"``：按平均剩余寿命 ``E[T - x | T > x]``（对数
      正态 MRL 闭式解）降序驱逐。理论动机是"类内无可区分信息时的分布
      最优序"，但实测仅在高压档小幅有效、低压档反噬——对数正态风险率
      先增后减（MRL 呈 U 型），该排序会驱逐"仍在间隔中但必然回归"的
      会话；保留作负结果存档。

    已知不建模：会话终止预测。重叠到达下的逐轮"存活率"存在删失偏差
    （reached[t+1]/reached[t] 度量的是到达进度而非终止），且 iid 间隔
    负载中"会话是否结束"与空闲时长不可区分，故不引入该项。
    冷启动（类样本 < ``warmup``）退化为按空闲时长排序（≈LRU）。
    """

    name = "predict"

    def __init__(
        self,
        horizon_s: float | None = None,
        warmup: int = 20,
        rank_by: str = "return",
    ) -> None:
        if rank_by not in ("residual", "return"):
            raise ValueError(f"rank_by must be 'residual' or 'return', got {rank_by!r}")
        self.horizon_s = horizon_s
        self.warmup = warmup
        self.rank_by = rank_by
        self._gap: dict[str, _LognormalOnline] = {}
        self._sessions: dict[str, _SessionState] = {}

    # ---- 观测（在线因果） ----

    def on_admit(self, request: "TraceRequest", now: float) -> None:
        if request.turn_id > 1 and request.think_time > 0:
            self._gap.setdefault(request.agent_type, _LognormalOnline()).add(
                math.log(request.think_time)
            )
        self._sessions[f"sess:{request.session_id}"] = _SessionState(
            request.agent_type, request.turn_id, now
        )

    # ---- 打分 ----

    def _class_ready(self, agent_type: str) -> _LognormalOnline | None:
        stats = self._gap.get(agent_type)
        return stats if stats is not None and stats.n >= self.warmup else None

    @staticmethod
    def _mean_residual(idle: float, stats: _LognormalOnline) -> float:
        """对数正态平均剩余寿命 E[T - idle | T > idle]（闭式解）。"""
        mu, sigma = stats.mean, max(stats.std, 1e-6)
        z = (math.log(max(idle, 1e-9)) - mu) / sigma
        survival = 1.0 - 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))
        if survival <= 1e-9:
            return math.exp(mu + sigma * sigma / 2.0)  # 尾部近似：无条件均值
        truncated_mean = math.exp(mu + sigma * sigma / 2.0) * 0.5 * (
            1.0 + math.erf((mu + sigma * sigma - math.log(max(idle, 1e-9))) / (sigma * math.sqrt(2.0)))
        )
        return max(truncated_mean / survival - idle, 0.0)

    def _score(self, stream: str, now: float) -> float:
        """驱逐分：越大越先驱逐。"""
        if stream.startswith("agent:"):
            return -1.0  # 前导流共享且高复用：几乎最后驱逐
        state = self._sessions.get(stream)
        if state is None:
            return -0.5  # 未知会话：保守保留
        stats = self._class_ready(state.agent_type)
        if stats is None:
            return -0.25 + (now - state.last_seen) * 1e-9  # 冷启动：轻微按空闲偏置
        idle = max(now - state.last_seen, 0.0)
        if self.rank_by == "residual":
            return self._mean_residual(idle, stats)
        horizon = self.horizon_s or stats.p90
        survival = 1.0 - _lognormal_cdf(idle, stats.mean, max(stats.std, 1e-6))
        if survival <= 1e-9:
            return float("inf")
        window = _lognormal_cdf(idle + horizon, stats.mean, max(stats.std, 1e-6)) - (1.0 - survival)
        p_time = min(max(window / survival, 0.0), 1.0)
        return -p_time  # 回归概率低 → 驱逐分高

    def select_victims(
        self, tree: RadixTree, need_tokens: int, now: float
    ) -> list[RadixNode]:
        # 驱逐分降序；同分按 LRU 兜底
        return sorted(
            tree.evictable_leaves(),
            key=lambda node: (-self._score(node.segment.stream, now), node.last_access),
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


class TransitionPolicy(EvictionPolicy):
    """agent 执行转移预测驱逐（M5；CacheScout 思路的开源模拟器版）。

    在线观测 ``parent_session`` 派生关系，学习一阶马尔可夫转移计数
    ``P(下一执行者 | 当前执行者)``；驱逐时以"最近 ``active_window_s``
    秒内有请求到达的 agent 类型集合"为前沿做多源 BFS，会话离前沿的
    跳距越大（预计越晚执行）越先驱逐。共享前导流（``agent:``/``flow:``）
    最后驱逐。无工作流字段可学时退化为 LRU。
    """

    name = "transition"

    PREAMBLE_SCORE = -1.0  # 共享前导几乎最后驱逐

    def __init__(self, active_window_s: float = 30.0, unknown_hop: float = 2.5):
        if active_window_s <= 0:
            raise ValueError("active_window_s must be positive")
        self.active_window_s = active_window_s
        self.unknown_hop = unknown_hop
        self._transitions: dict[str, dict[str, int]] = {}
        self._sessions: dict[str, _SessionState] = {}

    def on_admit(self, request: "TraceRequest", now: float) -> None:
        if request.parent_session:
            parent = self._sessions.get(f"sess:{request.parent_session}")
            if parent is not None:
                counts = self._transitions.setdefault(parent.agent_type, {})
                counts[request.agent_type] = counts.get(request.agent_type, 0) + 1
        self._sessions[f"sess:{request.session_id}"] = _SessionState(
            request.agent_type, request.turn_id, now
        )

    def _hop_map(self, now: float) -> dict[str, int]:
        """从活跃前沿出发的多源 BFS 跳距（邻接 = 已学习的转移边）。"""
        frontier = {
            state.agent_type
            for state in self._sessions.values()
            if now - state.last_seen <= self.active_window_s
        }
        hops: dict[str, int] = {agent_type: 0 for agent_type in frontier}
        queue = list(frontier)
        while queue:
            current = queue.pop(0)
            for successor in self._transitions.get(current, {}):
                if successor not in hops:
                    hops[successor] = hops[current] + 1
                    queue.append(successor)
        return hops

    def select_victims(
        self, tree: RadixTree, need_tokens: int, now: float
    ) -> list[RadixNode]:
        hops = self._hop_map(now)

        def score(node: RadixNode) -> float:
            stream = node.segment.stream
            if stream.startswith("agent:") or stream.startswith("flow:"):
                return self.PREAMBLE_SCORE
            state = self._sessions.get(stream)
            if state is None:
                return self.unknown_hop
            return float(hops.get(state.agent_type, self.unknown_hop))

        return sorted(
            tree.evictable_leaves(),
            key=lambda node: (-score(node), node.last_access),
        )


register_policy(FIFOPolicy)
register_policy(LRUPolicy)
register_policy(TTLPolicy)
register_policy(PriorityPolicy)
register_policy(QuotaPolicy)
register_policy(WeightedLRUPolicy)
register_policy(BeladyPolicy)
register_policy(ClassTTLPolicy)
register_policy(PredictivePolicy)
register_policy(TransitionPolicy)
