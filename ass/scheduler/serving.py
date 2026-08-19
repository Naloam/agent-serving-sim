"""请求生命周期调度（对应 PRD FR-7）。

流程：到达 → 前缀匹配（pin）→ 容量不足则驱逐 → 计入 cache（pin）→
解析式计时（prefill = 未命中 token / 吞吐，decode = 输出 token / 吞吐）→
完成（释放引用、记账、重试排队请求）。

行为约定：

- **开环到达**：请求按 trace 的 ``arrival_time`` 到达，不因系统状态改变；
- **并发上限**：在途请求数受 ``max_concurrent`` 约束，超出者 FIFO 排队；
- **容量不足且无可淘汰**（缓存被在途引用占满）：请求排队，完成事件
  释放引用后自动重试；
- **请求自身超出总容量**（缓存已空仍放不下）：不缓存直接服务，计入
  ``uncached_requests`` 指标；
- **前缀 key 结构**：agent 前导流（system+tools，跨会话共享）+ 会话
  对话流（history+new+output，会话内逐轮延伸），与 radix tree 的
  位置对齐 Segment 语义一致。

同一时刻完成事件先于到达事件执行（priority=-1），使释放的容量可被
同刻到达的请求立即使用。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import partial
from typing import Iterable

from ass.cache.policies import EvictionPolicy, LRUPolicy
from ass.cache.radix import MatchResult, NodeMeta, RadixNode, RadixTree, Segment
from ass.core.sim import Simulation
from ass.metrics.collector import MetricsCollector
from ass.workload.schema import TraceRequest


@dataclass(frozen=True)
class ServingConfig:
    """服务侧参数（解析式计时模型）。"""

    cache_capacity_tokens: int
    prefill_tps: float = 5000.0
    decode_tps: float = 200.0
    max_concurrent: int = 8

    def __post_init__(self) -> None:
        if self.cache_capacity_tokens <= 0:
            raise ValueError("cache_capacity_tokens must be positive")
        if self.prefill_tps <= 0 or self.decode_tps <= 0:
            raise ValueError("prefill_tps and decode_tps must be positive")
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")


def request_key(request: TraceRequest) -> tuple[Segment, ...]:
    """把请求映射为 radix tree 的前缀段 key。"""
    preamble = request.prompt.system + request.prompt.tools
    dialogue = request.prompt.history + request.prompt.new + request.output_tokens
    segments: list[Segment] = []
    if preamble > 0:
        segments.append(Segment(f"agent:{request.agent_type}", preamble))
    if dialogue > 0:
        segments.append(Segment(f"sess:{request.session_id}", dialogue))
    return tuple(segments)


@dataclass
class _ActiveRequest:
    """已获准服务的在途请求。"""

    request: TraceRequest
    admit_time: float
    hit_tokens: int
    ttft: float
    uncached: bool
    pinned: list[RadixNode] = field(default_factory=list)


class ServingSim:
    """把 trace 喂进事件循环并驱动 cache/策略/指标的仿真器门面。"""

    def __init__(
        self,
        config: ServingConfig,
        policy: EvictionPolicy | None = None,
        collector: MetricsCollector | None = None,
        sim: Simulation | None = None,
    ) -> None:
        self.config = config
        self.sim = sim if sim is not None else Simulation()
        self.tree = RadixTree(config.cache_capacity_tokens)
        self.policy = policy if policy is not None else LRUPolicy()
        self.collector = collector if collector is not None else MetricsCollector()
        self._ttl: float | None = getattr(self.policy, "ttl", None)
        self._waiting: deque[TraceRequest] = deque()
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    @property
    def waiting(self) -> int:
        return len(self._waiting)

    def submit_all(self, requests: Iterable[TraceRequest]) -> None:
        """按 arrival_time 预排全部请求的到达事件。"""
        for request in requests:
            self.sim.schedule(
                request.arrival_time,
                partial(self._on_arrival, request),
                kind="arrival",
            )

    def run(self, until: float | None = None) -> float:
        return self.sim.run(until)

    # ---- 内部：生命周期 ----

    def _on_arrival(self, request: TraceRequest) -> None:
        self._sweep_ttl()
        self.collector.record_arrival(request)
        self._waiting.append(request)
        self._try_admit()

    def _try_admit(self) -> None:
        while self._in_flight < self.config.max_concurrent and self._waiting:
            candidate = self._waiting[0]
            if self._admit(candidate):
                self._waiting.popleft()
            else:
                break  # 队头被容量阻塞，等待完成事件释放后重试

    def _admit(self, request: TraceRequest) -> bool:
        now = self.sim.now
        key = request_key(request)
        total = request.prompt.total + request.output_tokens
        match = self.tree.match(key, now=now, pin=True)
        need = total - match.hit_tokens
        if need > 0 and self.tree.free_tokens < need:
            self._evict_for(need, now)
        if need > 0 and self.tree.free_tokens < need:
            if self.tree.used_tokens == 0:
                # 请求自身超出总容量：不缓存直接服务
                self.tree.release(match.path)
                match = MatchResult(0, [])
                uncached = True
            else:
                # 可淘汰空间不足：排队等待（PRD FR-7 定义行为）
                self.tree.release(match.path)
                return False
        else:
            uncached = False
        if uncached:
            pinned: list[RadixNode] = []
        else:
            pinned = match.path + self.tree.insert(
                key,
                now=now,
                meta=NodeMeta(priority=request.priority, agent_type=request.agent_type),
                pin=True,
            )
            self.collector.record_cache_usage(now, self.tree.used_tokens)
        hit_tokens = match.hit_tokens
        prefill_unmatched = max(0, request.prompt.total - hit_tokens)
        prefill_time = prefill_unmatched / self.config.prefill_tps
        decode_time = request.output_tokens / self.config.decode_tps
        ttft = (now - request.arrival_time) + prefill_time
        active = _ActiveRequest(
            request=request,
            admit_time=now,
            hit_tokens=hit_tokens,
            ttft=ttft,
            uncached=uncached,
            pinned=pinned,
        )
        self._in_flight += 1
        self.sim.schedule(
            now + prefill_time + decode_time,
            partial(self._on_complete, active),
            kind="complete",
            priority=-1,
        )
        return True

    def _evict_for(self, need: int, now: float) -> None:
        """逐个驱逐直到腾够空间；父节点暴露为新叶子时自动纳入下一轮。"""
        used_before = self.tree.used_tokens
        count = 0
        while self.tree.free_tokens < need:
            victims = self.policy.select_victims(self.tree, need, now)
            if not victims:
                break
            self.tree.evict(victims[0])
            count += 1
        if count:
            self.collector.record_evictions(used_before - self.tree.used_tokens, count)
            self.collector.record_cache_usage(now, self.tree.used_tokens)

    def _on_complete(self, active: _ActiveRequest) -> None:
        now = self.sim.now
        self.tree.release(active.pinned)
        self._in_flight -= 1
        self._sweep_ttl()
        self.collector.record_completion(
            active.request,
            admit_time=active.admit_time,
            completion_time=now,
            hit_tokens=active.hit_tokens,
            ttft=active.ttft,
            uncached=active.uncached,
        )
        self._try_admit()

    def _sweep_ttl(self) -> None:
        """TTL 策略下，事件时刻先做过期清除（过期条目不再可命中）。"""
        if self._ttl is None:
            return
        freed = self.tree.sweep_expired(self.sim.now, self._ttl)
        if freed:
            self.collector.record_expiry(freed)
            self.collector.record_cache_usage(self.sim.now, self.tree.used_tokens)
