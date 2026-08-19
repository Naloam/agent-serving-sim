"""指标采集器的单元测试（FR-8）：聚合正确性与 CSV/JSON 导出。"""

import csv
import json

from ass.metrics.collector import MetricsCollector
from ass.workload.schema import PromptBreakdown, TraceRequest


def make_request(session: str, turn: int, agent: str, arrival: float, prompt_total: int, output: int) -> TraceRequest:
    return TraceRequest(
        session_id=session,
        turn_id=turn,
        arrival_time=arrival,
        prompt=PromptBreakdown(system=prompt_total, tools=0, history=0, new=0),
        output_tokens=output,
        think_time=0.0,
        agent_type=agent,
        priority=1,
    )


def build_collector() -> MetricsCollector:
    collector = MetricsCollector()
    # 两条请求：coding 全命中，search 半命中
    collector.record_arrival(make_request("s1", 1, "coding", 0.0, 1000, 100))
    collector.record_arrival(make_request("s2", 1, "search", 1.0, 1000, 100))
    collector.record_completion(
        make_request("s1", 1, "coding", 0.0, 1000, 100),
        admit_time=0.5,
        completion_time=3.0,
        hit_tokens=1000,
        ttft=0.5,
    )
    collector.record_completion(
        make_request("s2", 1, "search", 1.0, 1000, 100),
        admit_time=1.0,
        completion_time=5.0,
        hit_tokens=500,
        ttft=1.0,
    )
    collector.record_evictions(tokens=300, count=2)
    collector.record_expiry(tokens=120)
    collector.record_cache_usage(0.5, 1000)
    collector.record_cache_usage(1.0, 1600)
    collector.record_cache_usage(4.0, 1300)
    return collector


def test_summary_aggregates() -> None:
    summary = build_collector().summary()
    assert summary["arrivals"] == 2
    assert summary["completed"] == 2
    assert summary["prompt_tokens"] == 2000
    assert summary["hit_tokens"] == 1500
    assert summary["hit_rate"] == 0.75
    assert summary["jct_mean"] == (3.0 + 4.0) / 2
    assert summary["jct_p50"] == 3.0
    assert summary["jct_p95"] == 4.0
    assert summary["jct_max"] == 4.0
    assert summary["queue_delay_mean"] == 0.25
    assert summary["sessions"] == 2
    assert summary["session_jct_sum_mean"] == 3.5
    assert summary["evictions"] == {"count": 2, "tokens": 300}
    assert summary["ttl_expired_tokens"] == 120
    assert summary["cache_peak_tokens"] == 1600


def test_summary_by_agent_type() -> None:
    summary = build_collector().summary()
    by_agent = summary["by_agent_type"]
    assert by_agent["coding"]["hit_rate"] == 1.0
    assert by_agent["search"]["hit_rate"] == 0.5
    assert by_agent["coding"]["jct_mean"] == 3.0
    assert by_agent["search"]["jct_mean"] == 4.0


def test_empty_summary_is_zeroed() -> None:
    summary = MetricsCollector().summary()
    assert summary["completed"] == 0
    assert summary["hit_rate"] == 0.0
    assert summary["jct_mean"] == 0.0
    assert summary["cache_peak_tokens"] == 0


def test_csv_export(tmp_path) -> None:
    collector = build_collector()
    path = tmp_path / "requests.csv"
    collector.write_csv(path)
    rows = list(csv.DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 2
    assert rows[0]["session_id"] == "s1"
    assert float(rows[0]["jct"]) == 3.0
    assert rows[1]["hit_tokens"] == "500"


def test_json_export_round_trip(tmp_path) -> None:
    collector = build_collector()
    path = tmp_path / "summary.json"
    collector.write_json(path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded == collector.summary()
    assert loaded["hit_rate"] == 0.75


def test_jct_values_filter_by_agent() -> None:
    collector = build_collector()
    assert collector.jct_values() == [3.0, 4.0]
    assert collector.jct_values(agent_type="coding") == [3.0]
    assert collector.jct_values(agent_type="none") == []
