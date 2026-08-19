"""Trace schema 与 JSONL 读写的单元测试（FR-2）。"""

import json

import pytest

from ass.workload.schema import (
    PromptBreakdown,
    TraceRequest,
    TraceValidationError,
    read_trace,
    request_from_dict,
    request_to_dict,
    write_trace,
)


def make_request(**overrides: object) -> TraceRequest:
    base: dict[str, object] = dict(
        session_id="sess_0001",
        turn_id=3,
        arrival_time=42.7,
        prompt=PromptBreakdown(system=812, tools=1043, history=2210, new=156),
        output_tokens=388,
        think_time=18.3,
        agent_type="coding",
        priority=1,
    )
    base.update(overrides)
    return TraceRequest(**base)  # type: ignore[arg-type]


def test_prompt_total() -> None:
    prompt = PromptBreakdown(system=100, tools=200, history=300, new=400)
    assert prompt.total == 1000


def test_invalid_prompt_segment_rejected() -> None:
    with pytest.raises(TraceValidationError, match="prompt.system"):
        PromptBreakdown(system=-1, tools=0, history=0, new=0)


@pytest.mark.parametrize(
    "field, value, pattern",
    [
        ("session_id", "", "session_id"),
        ("turn_id", 0, "turn_id"),
        ("arrival_time", -1.0, "arrival_time"),
        ("output_tokens", -5, "output_tokens"),
        ("think_time", -0.1, "think_time"),
        ("agent_type", "", "agent_type"),
        ("priority", "high", "priority"),
    ],
)
def test_invalid_request_fields(field: str, value: object, pattern: str) -> None:
    with pytest.raises(TraceValidationError, match=pattern):
        make_request(**{field: value})


def test_jsonl_round_trip(tmp_path) -> None:
    """读 → 写 → 读 不变（PRD 验收项）。"""
    requests = [make_request(), make_request(session_id="sess_0002", turn_id=1)]
    path = tmp_path / "trace.jsonl"
    write_trace(requests, path)
    reloaded = read_trace(path)
    assert reloaded == requests


def test_read_trace_reports_line_number(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    good = request_to_dict(make_request())
    path.write_text(
        json.dumps(good) + "\n"
        + json.dumps({**good, "think_time": -1.0}) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(TraceValidationError, match=r"bad\.jsonl:2"):
        read_trace(path)


def test_read_trace_invalid_json_line(tmp_path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text("{not json}\n", encoding="utf-8")
    with pytest.raises(TraceValidationError, match="invalid JSON"):
        read_trace(path)


def test_read_trace_skips_blank_lines(tmp_path) -> None:
    path = tmp_path / "blank.jsonl"
    line = json.dumps(request_to_dict(make_request()))
    path.write_text("\n" + line + "\n\n", encoding="utf-8")
    assert len(read_trace(path)) == 1


def test_from_dict_rejects_unknown_and_missing() -> None:
    good = request_to_dict(make_request())
    with pytest.raises(TraceValidationError, match="unknown fields"):
        request_from_dict({**good, "extra": 1})
    with pytest.raises(TraceValidationError, match="missing fields"):
        request_from_dict({k: v for k, v in good.items() if k != "priority"})
    with pytest.raises(TraceValidationError, match="prompt"):
        request_from_dict({**good, "prompt": {"system": 1}})
