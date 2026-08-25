"""请求生命周期调度。

流程：到达 → 前缀匹配（pin）→ 容量不足则驱逐 → 计入 cache（pin）→
解析式计时（prefill = 未命中 token / 吞吐，decode = 输出 token / 吞吐）→
完成（释放引用、记账、重试排队请求）。

行为约定：

- **开环到达**：请求按 trace 的 ``arrival_time`` 到达，不因系统状态改变；
- **并发上限**：在途请求数受 ``max_concurrent`` 约束，超出者 FIFO 排队；
- **准入时容量不足且无可淘汰**（缓存被在途引用占满）：请求排队，完成事件
  释放引用后自动重试；
- **请求自身超出总容量**（缓存已空仍放不下）：不缓存直接服务，计入
  ``uncached_requests`` 指标；
- **decode 分块增长**（``decode_chunks > 1`` 时启用，=1 保持旧行为）：
  prompt 部分在准入时插入 pin，输出按 ``decode_chunks`` 块增长——前
  ``chunks-1`` 块由增长事件驱动，末块在完成时插入；
- **增长遇容量耗尽**：先驱逐 idle 叶子；仍不足且 ``allow_preemption``
  时**抢占**在途请求——受害者取"最新准入的他者"，其全部事件被取消、
  KV 丢弃（共享前缀因他人引用而幸存）、队首回队重算（重算成本计入其
  JCT）；同一请求累计被抢 ``MAX_PREEMPTIONS`` 次后转为不缓存模式保证
  活性。抢占统计进指标；
- **前缀 key 结构**：agent 前导流（system+tools，跨会话共享）+ 会话
  对话流（history+new[+output]，会话内逐轮延伸），与 radix tree 的
  位置对齐 Segment 语义一致。

同一时刻完成事件先于到达事件执行（priority=-1），使释放的容量可被
同刻到达的请求立即使用。
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from functools import partial
from typing import Iterable

from ass.scheduler.admission import AdmissionPolicy, FIFOAdmission

from ass.cache.policies import EvictionPolicy, LRUPolicy
from ass.cache.radix import MatchResult, NodeMeta, RadixNode, RadixTree, Segment
from ass.core.event import Event
from ass.core.sim import Simulation
from ass.metrics.collector import MetricsCollector
from ass.workload.schema import TraceRequest

MAX_PREEMPTIONS = 3


@dataclass(frozen=True)
class ServingConfig:
    """服务侧参数（解析式计时模型）。"""

    cache_capacity_tokens: int
    prefill_tps: float = 5000.0
    decode_tps: float = 200.0
    max_concurrent: int = 8
    # >1 时启用 decode 分块增长与抢占语义（FR-13）；=1 为旧的整体插入行为
    decode_chunks: int = 1
    allow_preemption: bool = True
    # 驱逐吞吐（token/s）：None = 驱逐免费零时延（一阶模型）；设值后
    # 驱逐成本计入触发请求的关键路径（锁竞争/块回收的二阶效应）
    evict_tps: float | None = None
    # 每请求固定服务开销（秒）：标定为流式 TTFT 拟合的截距（排队/调度
    # 固定项），计入 TTFT 与完成时间；默认 0 保持旧行为
    fixed_overhead_s: float = 0.0

    def __post_init__(self) -> None:
        if self.cache_capacity_tokens <= 0:
            raise ValueError("cache_capacity_tokens must be positive")
        if self.prefill_tps <= 0 or self.decode_tps <= 0:
            raise ValueError("prefill_tps and decode_tps must be positive")
        if self.max_concurrent <= 0:
            raise ValueError("max_concurrent must be positive")
        if self.decode_chunks < 1:
            raise ValueError("decode_chunks must be >= 1")
        if self.evict_tps is not None and self.evict_tps <= 0:
            raise ValueError("evict_tps must be positive when set")
        if self.fixed_overhead_s < 0:
            raise ValueError("fixed_overhead_s must be non-negative")


def request_key(request: TraceRequest) -> tuple[Segment, ...]:
    """把请求映射为 radix tree 的前缀段 key（prompt + output 全量）。"""
    return _segments(request, include_output=True)


def _segments(request: TraceRequest, include_output: bool) -> tuple[Segment, ...]:
    preamble = request.prompt.system + request.prompt.tools
    dialogue = request.prompt.history + request.prompt.new
    if include_output:
        dialogue += request.output_tokens
    # 工作流负载：同流会话共享前导（前缀流用 flow 标识，跨 agent 类型复用）
    preamble_stream = (
        f"flow:{request.flow_id}" if request.flow_id else f"agent:{request.agent_type}"
    )
    segments: list[Segment] = []
    if preamble > 0:
        segments.append(Segment(preamble_stream, preamble))
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
    # 分块增长模式（FR-13）
    leaf: RadixNode | None = None
    growth_events: list[Event] = field(default_factory=list)
    completion_event: Event | None = None
    final_chunk: int = 0
    growth_capped: bool = False
    finished: bool = False
    preempted: bool = False


class ServingSim:
    """把 trace 喂进事件循环并驱动 cache/策略/指标的仿真器门面。"""

    def __init__(
        self,
        config: ServingConfig,
        policy: EvictionPolicy | None = None,
        collector: MetricsCollector | None = None,
        sim: Simulation | None = None,
        admission: "AdmissionPolicy | None" = None,
    ) -> None:
        self.config = config
        self.sim = sim if sim is not None else Simulation()
        self.tree = RadixTree(config.cache_capacity_tokens)
        self.policy = policy if policy is not None else LRUPolicy()
        self.collector = collector if collector is not None else MetricsCollector()
        self.admission = admission if admission is not None else FIFOAdmission()
        self._ttl: float | None = getattr(self.policy, "ttl", None)
        self._waiting: deque[TraceRequest] = deque()
        self._active: list[_ActiveRequest] = []
        self._in_flight = 0
        self._preempt_counts: dict[int, int] = {}

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
            # 准入策略给出偏好序；FIFO 仅提供队头（保持队头阻塞基线语义），
            # 其他策略会跳过被缓存容量阻塞的候选
            for candidate in self.admission.order(list(self._waiting), self.sim.now):
                if self._admit(candidate):
                    self._waiting.remove(candidate)
                    break
            else:
                break  # 偏好序内均无法准入：等待完成事件释放后重试

    def _admit(self, request: TraceRequest) -> bool:
        now = self.sim.now
        self.policy.on_admit(request, now)  # 在线学习策略的观测钩子（默认 no-op）
        chunked = self.config.decode_chunks > 1
        forced_uncached = self._preempt_counts.get(id(request), 0) >= MAX_PREEMPTIONS
        key = _segments(request, include_output=not chunked)
        total = sum(seg.length for seg in key)
        match = self.tree.match(key, now=now, pin=True)
        need = total - match.hit_tokens
        evict_freed = 0
        if need > 0 and self.tree.free_tokens < need:
            evict_freed = self._evict_for(need, now)
        if need > 0 and self.tree.free_tokens < need:
            if self.tree.used_tokens == 0 or forced_uncached:
                # 请求自身超出总容量 / 被抢次数用尽：不缓存直接服务
                self.tree.release(match.path)
                match = MatchResult(0, [])
                uncached = True
            else:
                # 可淘汰空间不足：排队等待
                self.tree.release(match.path)
                return False
        else:
            uncached = forced_uncached and need == 0
        leaf: RadixNode | None = None
        if uncached:
            pinned: list[RadixNode] = []
        else:
            materialized = self.tree.insert(
                key,
                now=now,
                meta=NodeMeta(priority=request.priority, agent_type=request.agent_type),
                pin=True,
            )
            pinned = match.path + materialized
            if materialized:
                leaf = materialized[-1]
            elif match.path:
                leaf = match.path[-1]
            self.collector.record_cache_usage(now, self.tree.used_tokens)
        hit_tokens = match.hit_tokens
        prefill_unmatched = max(0, request.prompt.total - hit_tokens)
        evict_debt = (
            evict_freed / self.config.evict_tps
            if self.config.evict_tps is not None and evict_freed > 0
            else 0.0
        )
        prefill_time = (
            prefill_unmatched / self.config.prefill_tps
            + evict_debt
            + self.config.fixed_overhead_s
        )
        decode_time = request.output_tokens / self.config.decode_tps
        ttft = (now - request.arrival_time) + prefill_time
        active = _ActiveRequest(
            request=request,
            admit_time=now,
            hit_tokens=hit_tokens,
            ttft=ttft,
            uncached=uncached,
            pinned=pinned,
            leaf=leaf if chunked else None,
        )
        self._in_flight += 1
        self._active.append(active)
        active.completion_event = self.sim.schedule(
            now + prefill_time + decode_time,
            partial(self._on_complete, active),
            kind="complete",
            priority=-1,
        )
        if chunked and not uncached and request.output_tokens > 0:
            self._schedule_growth(active, prefill_time, decode_time)
        return True

    def _schedule_growth(self, active: _ActiveRequest, prefill_time: float, decode_time: float) -> None:
        """前 chunks-1 块由事件驱动（每块 output//chunks），末块在完成时插入。"""
        chunks = self.config.decode_chunks
        output = active.request.output_tokens
        base = active.admit_time + prefill_time
        per_chunk = output // chunks
        active.final_chunk = output - per_chunk * (chunks - 1)
        for index in range(1, chunks):
            event = self.sim.schedule(
                base + decode_time * index / chunks,
                partial(self._on_growth, active, per_chunk),
                kind="growth",
            )
            active.growth_events.append(event)

    def _on_growth(self, active: _ActiveRequest, add_tokens: int) -> None:
        if active.finished or active.preempted:
            return
        self._grow_insert(active, add_tokens, self.sim.now)

    def _grow_insert(self, active: _ActiveRequest, add_tokens: int, now: float) -> None:
        """为 active 增长 add_tokens；容量不足时驱逐→抢占→封顶不缓存。"""
        if add_tokens <= 0 or active.growth_capped or active.leaf is None:
            return
        capacity = self.tree.capacity_tokens
        evict_freed = 0
        if self.tree.used_tokens + add_tokens > capacity:
            evict_freed = self._evict_for(add_tokens, now)
        while self.tree.used_tokens + add_tokens > capacity:
            victim = (
                self._pick_preempt_victim(exclude=active)
                if self.config.allow_preemption and self._in_flight > 1
                else None
            )
            if victim is None:
                break
            self._preempt(victim, now)
        if self.tree.used_tokens + add_tokens > capacity:
            active.growth_capped = True  # 计算继续，但增长部分不缓存
            self.collector.record_cache_usage(now, self.tree.used_tokens)
            return
        if evict_freed > 0 and self.config.evict_tps is not None:
            # 增长期驱逐的成本折入完成时间（关键路径）
            self._delay_completion(active, evict_freed / self.config.evict_tps)
        before = active.leaf
        active.leaf = self.tree.grow(active.leaf, add_tokens)
        if active.leaf is not before:
            active.leaf.refcount += 1  # 链式追加的新尾节点计入本请求引用
            active.pinned.append(active.leaf)
        self.collector.record_cache_usage(now, self.tree.used_tokens)

    def _delay_completion(self, active: _ActiveRequest, extra_s: float) -> None:
        """把完成事件顺延 extra_s（驱逐成本计入该请求的 JCT）。"""
        if extra_s <= 0 or active.finished or active.preempted:
            return
        event = active.completion_event
        if event is None or event.cancelled:
            return
        self.sim.cancel(event)
        active.completion_event = self.sim.schedule(
            event.time + extra_s, event.callback, kind="complete", priority=-1
        )

    def _pick_preempt_victim(self, exclude: _ActiveRequest) -> _ActiveRequest | None:
        candidates = [a for a in self._active if a is not exclude and not a.finished]
        if not candidates:
            return None
        return max(candidates, key=lambda a: a.admit_time)  # 最新准入者先被抢

    def _preempt(self, victim: _ActiveRequest, now: float) -> None:
        """抢占：取消事件、丢弃 KV（共享前缀幸存）、队首回队重算。"""
        victim.preempted = True
        for event in victim.growth_events:
            self.sim.cancel(event)
        if victim.completion_event is not None:
            self.sim.cancel(victim.completion_event)
        self.tree.release(victim.pinned)
        dropped = 0
        for node in victim.pinned:
            if node.evictable:
                dropped += self.tree.evict(node)
        self._active.remove(victim)
        self._in_flight -= 1
        self._preempt_counts[id(victim.request)] = (
            self._preempt_counts.get(id(victim.request), 0) + 1
        )
        self.collector.record_preemption(
            wasted_s=now - victim.admit_time, dropped_tokens=dropped
        )
        self.collector.record_cache_usage(now, self.tree.used_tokens)
        self._waiting.appendleft(victim.request)

    def _evict_for(self, need: int, now: float) -> int:
        """逐个驱逐直到腾够空间；返回释放的 token 数。"""
        used_before = self.tree.used_tokens
        count = 0
        while self.tree.free_tokens < need:
            victims = self.policy.select_victims(self.tree, need, now)
            if not victims:
                break
            self.tree.evict(victims[0])
            count += 1
        if count:
            freed = used_before - self.tree.used_tokens
            self.collector.record_evictions(freed, count)
            self.collector.record_cache_usage(now, self.tree.used_tokens)
            return freed
        return 0

    def _on_complete(self, active: _ActiveRequest) -> None:
        now = self.sim.now
        active.finished = True
        if self.config.decode_chunks > 1 and not active.uncached:
            self._grow_insert(active, active.final_chunk, now)
        self.tree.release(active.pinned)
        self._in_flight -= 1
        self._active.remove(active)
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
