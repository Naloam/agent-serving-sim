"""真实 trace 解析器的单元测试（FR-4）：轮次对齐、四段拆分、坏行跳过。"""

import json

from ass.workload.loaders import parse_probe_log


def chat_entry(
    session: str,
    agent: str,
    ts_request: str,
    ts_complete: str,
    messages: list[dict],
    prompt_tokens: int,
    completion_tokens: int,
    tools: list | None = None,
) -> dict:
    return {
        "ts_request": ts_request,
        "ts_first_byte": ts_request,
        "ts_complete": ts_complete,
        "session_id": session,
        "agent_type": agent,
        "method": "POST",
        "path": "/v1/chat/completions",
        "stream": False,
        "status": 200,
        "error": None,
        "request": {"model": "qwen", "messages": messages, "tools": tools or []},
        "usage": {"prompt_tokens": prompt_tokens, "completion_tokens": completion_tokens},
        "elapsed_s": 1.0,
    }


def write_log(tmp_path, entries: list[dict], raw_lines: list[str] | None = None) -> object:
    lines = [json.dumps(entry, ensure_ascii=False) for entry in entries]
    lines.extend(raw_lines or [])
    path = tmp_path / "probe.jsonl"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def test_two_agent_types_parse_to_trace(tmp_path) -> None:
    """coding 与 search 两类 agent 日志都能解析（PRD 验收项）。"""
    entries = [
        chat_entry(
            "sess_a", "coding", "2026-08-19T12:00:00.000Z", "2026-08-19T12:00:02.000Z",
            [
                {"role": "system", "content": "You are a coding agent. " + "x" * 300},
                {"role": "user", "content": "fix the bug"},
            ],
            prompt_tokens=1000, completion_tokens=200,
            tools=[{"type": "function", "name": "run_tests"}],
        ),
        chat_entry(
            "sess_a", "coding", "2026-08-19T12:00:12.000Z", "2026-08-19T12:00:14.000Z",
            [
                {"role": "system", "content": "You are a coding agent. " + "x" * 300},
                {"role": "user", "content": "fix the bug"},
                {"role": "assistant", "content": "running tests"},
                {"role": "user", "content": "tests passed, continue"},
            ],
            prompt_tokens=2000, completion_tokens=150,
            tools=[{"type": "function", "name": "run_tests"}],
        ),
        chat_entry(
            "sess_b", "search", "2026-08-19T12:00:05.000Z", "2026-08-19T12:00:06.000Z",
            [
                {"role": "system", "content": "Answer with searches."},
                {"role": "user", "content": "what is paged attention"},
            ],
            prompt_tokens=500, completion_tokens=80,
        ),
    ]
    path = write_log(tmp_path, entries)
    report = parse_probe_log(path)
    assert report.skipped == []
    assert len(report.requests) == 3

    # 按到达时间排序：sess_a(t1) -> sess_b(t1) -> sess_a(t2)
    first, second, third = report.requests
    assert first.session_id == "sess_a" and first.turn_id == 1
    assert first.agent_type == "coding"
    assert first.arrival_time == 0.0
    assert first.think_time == 0.0
    assert first.prompt.total == 1000  # 四段之和精确等于 usage.prompt_tokens
    assert first.output_tokens == 200
    assert first.prompt.system > first.prompt.new  # system 占大头

    assert second.session_id == "sess_b" and second.turn_id == 1
    assert second.agent_type == "search"
    assert second.prompt.total == 500
    assert second.arrival_time == 5.0

    assert third.session_id == "sess_a" and third.turn_id == 2
    assert third.think_time == 10.0  # 12:00:12 到达 − 12:00:02 完成
    assert third.prompt.history > 0  # 前轮对话进入 history
    assert third.arrival_time == 12.0

    # 所有请求满足 FR-2 schema（含非负、turn_id >= 1 等校验）
    from ass.workload.schema import request_to_dict
    assert all(request_to_dict(r) for r in report.requests)

    # 计时事实保留
    assert len(report.timings) == 3
    assert report.timings[0].total_seconds == 2.0
    assert report.timings[0].completion_tokens == 200


def test_entries_sorted_by_arrival_before_turn_assignment(tmp_path) -> None:
    """完成顺序与到达顺序不一致时，轮次按到达顺序编号。"""
    entries = [
        chat_entry(
            "sess_a", "coding", "2026-08-19T12:00:10.000Z", "2026-08-19T12:00:11.000Z",
            [{"role": "user", "content": "second turn"}], 100, 10,
        ),
        chat_entry(
            "sess_a", "coding", "2026-08-19T12:00:01.000Z", "2026-08-19T12:00:09.500Z",
            [{"role": "user", "content": "first turn"}], 100, 10,
        ),
    ]
    path = write_log(tmp_path, entries)
    report = parse_probe_log(path)
    first, second = report.requests
    assert first.arrival_time == 0.0 and first.turn_id == 1
    assert second.arrival_time == 9.0 and second.turn_id == 2
    assert second.think_time == 0.5


def test_bad_lines_skipped_not_fatal(tmp_path) -> None:
    """解析失败行单独记录不中断（PRD 验收项）。"""
    entries = [
        chat_entry("sess_a", "coding", "2026-08-19T12:00:00.000Z", "2026-08-19T12:00:01.000Z",
                   [{"role": "user", "content": "hi"}], 100, 10),
    ]
    raw_lines = [
        "{broken json",
        json.dumps({"method": "GET", "path": "/v1/models", "ts_request": "2026-08-19T12:00:00.000Z"}),
        json.dumps({"method": "POST", "path": "/v1/chat/completions", "error": "upstream http error: 500",
                    "ts_request": "2026-08-19T12:00:02.000Z", "ts_complete": "2026-08-19T12:00:02.000Z"}),
        json.dumps({"method": "POST", "path": "/v1/chat/completions", "ts_request": "2026-08-19T12:00:03.000Z",
                    "request": {"messages": [{"role": "user", "content": "x"}]}}),
    ]
    path = write_log(tmp_path, entries, raw_lines)
    report = parse_probe_log(path)
    assert len(report.requests) == 1
    reasons = [reason for _, reason in report.skipped]
    assert any("invalid json" in reason for reason in reasons)
    assert any("not a chat completion" in reason for reason in reasons)
    assert any("request error" in reason for reason in reasons)
    assert any("missing usage" in reason for reason in reasons)


def test_defaults_and_multimodal_content(tmp_path) -> None:
    entries = [
        chat_entry(
            None, None, "2026-08-19T12:00:00.000Z", "2026-08-19T12:00:01.000Z",
            [
                {"role": "user", "content": [{"type": "text", "text": "look"}, {"type": "image_url"}]},
            ],
            300, 20,
        ),
    ]
    path = write_log(tmp_path, entries)
    report = parse_probe_log(path, default_agent_type="search")
    (request,) = report.requests
    assert request.agent_type == "search"
    assert request.session_id.startswith("sess_anon_")
    assert request.prompt.new > 0
    assert request.prompt.total == 300


def test_token_apportion_exact(tmp_path) -> None:
    entries = [
        chat_entry(
            "sess_a", "coding", "2026-08-19T12:00:00.000Z", "2026-08-19T12:00:01.000Z",
            [
                {"role": "system", "content": "a" * 100},
                {"role": "user", "content": "b" * 50},
            ],
            prompt_tokens=101, completion_tokens=5,
        ),
    ]
    path = write_log(tmp_path, entries)
    report = parse_probe_log(path)
    (request,) = report.requests
    assert request.prompt.system + request.prompt.new == 101
    assert request.prompt.system == 67  # floor(101*2/3) + 余数给最大小数部分
