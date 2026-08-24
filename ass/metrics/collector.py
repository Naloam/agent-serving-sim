"""指标采集器。

serving 层在每个生命周期节点调用对应 ``record_*``，仿真结束后通过
``summary()`` 得到聚合指标，或导出 ``write_csv``（逐请求）与
``write_json``（汇总）。仅用标准库。
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any

from ass.workload.schema import TraceRequest


@dataclass(frozen=True)
class RequestRecord:
    """单个请求的完整生命周期记账。"""

    session_id: str
    turn_id: int
    agent_type: str
    priority: int
    arrival_time: float
    admit_time: float
    completion_time: float
    prompt_tokens: int
    hit_tokens: int
    output_tokens: int
    ttft: float
    jct: float
    uncached: bool

    @property
    def queue_delay(self) -> float:
        return self.admit_time - self.arrival_time


class MetricsCollector:
    """聚合 JCT / TTFT / 命中率 / 显存时间线 / 驱逐统计。"""

    def __init__(self) -> None:
        self.records: list[RequestRecord] = []
        self.arrival_count = 0
        self.eviction_count = 0
        self.evicted_tokens = 0
        self.expired_tokens = 0
        self.preemption_count = 0
        self.preempted_wasted_s = 0.0
        self.preempted_dropped_tokens = 0
        self.cache_timeline: list[tuple[float, int]] = []

    # ---- 采集接口（由 serving 调用） ----

    def record_arrival(self, request: TraceRequest) -> None:
        self.arrival_count += 1

    def record_completion(
        self,
        request: TraceRequest,
        *,
        admit_time: float,
        completion_time: float,
        hit_tokens: int,
        ttft: float,
        uncached: bool = False,
    ) -> None:
        self.records.append(
            RequestRecord(
                session_id=request.session_id,
                turn_id=request.turn_id,
                agent_type=request.agent_type,
                priority=request.priority,
                arrival_time=request.arrival_time,
                admit_time=admit_time,
                completion_time=completion_time,
                prompt_tokens=request.prompt.total,
                hit_tokens=hit_tokens,
                output_tokens=request.output_tokens,
                ttft=ttft,
                jct=completion_time - request.arrival_time,
                uncached=uncached,
            )
        )

    def record_evictions(self, tokens: int, count: int) -> None:
        self.eviction_count += count
        self.evicted_tokens += tokens

    def record_expiry(self, tokens: int) -> None:
        """TTL 主动清除释放的 token 数。"""
        self.expired_tokens += tokens

    def record_preemption(self, wasted_s: float, dropped_tokens: int) -> None:
        """一次运行中请求被抢占：丢弃的 KV token 与浪费的计算时间。"""
        self.preemption_count += 1
        self.preempted_wasted_s += wasted_s
        self.preempted_dropped_tokens += dropped_tokens

    def record_cache_usage(self, now: float, used_tokens: int) -> None:
        self.cache_timeline.append((now, used_tokens))

    # ---- 查询与导出 ----

    def jct_values(self, agent_type: str | None = None) -> list[float]:
        if agent_type is None:
            return [record.jct for record in self.records]
        return [record.jct for record in self.records if record.agent_type == agent_type]

    def summary(self) -> dict[str, Any]:
        records = self.records
        total_prompt = sum(r.prompt_tokens for r in records)
        total_hit = sum(r.hit_tokens for r in records)
        jcts = sorted(r.jct for r in records)
        ttfts = sorted(r.ttft for r in records)
        delays = sorted(r.queue_delay for r in records)
        by_agent: dict[str, dict[str, float]] = {}
        for record in records:
            bucket = by_agent.setdefault(
                record.agent_type,
                {"requests": 0, "prompt_tokens": 0, "hit_tokens": 0, "jct_sum": 0.0, "ttft_sum": 0.0},
            )
            bucket["requests"] += 1
            bucket["prompt_tokens"] += record.prompt_tokens
            bucket["hit_tokens"] += record.hit_tokens
            bucket["jct_sum"] += record.jct
            bucket["ttft_sum"] += record.ttft
        agent_stats = {
            name: {
                "requests": bucket["requests"],
                "hit_rate": _safe_div(bucket["hit_tokens"], bucket["prompt_tokens"]),
                "jct_mean": _safe_div(bucket["jct_sum"], bucket["requests"]),
                "ttft_mean": _safe_div(bucket["ttft_sum"], bucket["requests"]),
            }
            for name, bucket in by_agent.items()
        }
        by_session: dict[str, dict[str, float]] = {}
        for record in records:
            bucket = by_session.setdefault(
                record.session_id, {"turns": 0, "jct_sum": 0.0, "first_arrival": record.arrival_time, "last_completion": record.completion_time}
            )
            bucket["turns"] += 1
            bucket["jct_sum"] += record.jct
            bucket["first_arrival"] = min(bucket["first_arrival"], record.arrival_time)
            bucket["last_completion"] = max(bucket["last_completion"], record.completion_time)
        session_jct_sums = [bucket["jct_sum"] for bucket in by_session.values()]
        peak_usage = max((used for _, used in self.cache_timeline), default=0)
        return {
            "arrivals": self.arrival_count,
            "completed": len(records),
            "prompt_tokens": total_prompt,
            "hit_tokens": total_hit,
            "hit_rate": _safe_div(total_hit, total_prompt),
            "uncached_requests": sum(1 for r in records if r.uncached),
            "jct_mean": _mean(jcts),
            "jct_p50": _percentile(jcts, 50),
            "jct_p95": _percentile(jcts, 95),
            "jct_max": jcts[-1] if jcts else 0.0,
            "ttft_mean": _mean(ttfts),
            "ttft_p95": _percentile(ttfts, 95),
            "queue_delay_mean": _mean(delays),
            "sessions": len(by_session),
            "session_jct_sum_mean": _mean(session_jct_sums),
            "by_agent_type": agent_stats,
            "evictions": {"count": self.eviction_count, "tokens": self.evicted_tokens},
            "ttl_expired_tokens": self.expired_tokens,
            "preemptions": {
                "count": self.preemption_count,
                "wasted_compute_s": round(self.preempted_wasted_s, 3),
                "dropped_tokens": self.preempted_dropped_tokens,
            },
            "cache_peak_tokens": peak_usage,
        }

    def write_csv(self, path: str | Path) -> None:
        """逐请求记录导出为 CSV。"""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        columns = [field.name for field in fields(RequestRecord)]
        with file_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for record in self.records:
                writer.writerow([getattr(record, name) for name in columns])

    def write_json(self, path: str | Path) -> None:
        """汇总指标导出为 JSON。"""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as handle:
            json.dump(self.summary(), handle, indent=2)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def _safe_div(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _percentile(sorted_values: list[float], pct: float) -> float:
    """就近秩百分位（输入须已升序排序）。"""
    if not sorted_values:
        return 0.0
    index = min(int(round(pct / 100 * (len(sorted_values) - 1))), len(sorted_values) - 1)
    return sorted_values[index]
