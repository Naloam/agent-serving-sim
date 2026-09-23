"""真实 trace 解析器的单元测试（FR-4）：轮次对齐、四段拆分、坏行跳过。"""

import json

import pytest

from ass.workload.loaders import parse_probe_log, real_tokens_trace_from_csv


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
    assert first.prompt.history == 0

    assert second.session_id == "sess_b" and second.turn_id == 1
    assert second.agent_type == "search"
    assert second.prompt.total == 500
    assert second.arrival_time == 5.0

    assert third.session_id == "sess_a" and third.turn_id == 2
    assert third.think_time == 10.0  # 12:00:12 到达 − 12:00:02 完成
    assert third.arrival_time == 12.0
    # 累计一致记账：同类型前导逐轮稳定，history 按前轮 new+output 推进
    assert third.prompt.system == first.prompt.system
    assert third.prompt.tools == first.prompt.tools
    assert third.prompt.history == first.prompt.new + first.output_tokens
    assert third.prompt.new == 2000 - (third.prompt.system + third.prompt.tools) - third.prompt.history

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


def test_anonymous_requests_do_not_poison_preamble(tmp_path) -> None:
    """匿名请求（无会话头，如预热）不定型前导，自身退回比例拆分。"""
    entries = [
        chat_entry(
            None, None, "2026-08-19T12:00:00.000Z", "2026-08-19T12:00:00.500Z",
            [{"role": "user", "content": "ready?"}], 20, 8,
        ),
        chat_entry(
            "sess_a", "coding", "2026-08-19T12:00:01.000Z", "2026-08-19T12:00:02.000Z",
            [
                {"role": "system", "content": "s" * 300},
                {"role": "user", "content": "hi"},
            ],
            800, 100,
        ),
    ]
    path = write_log(tmp_path, entries)
    report = parse_probe_log(path, default_agent_type="coding")
    anon, named = report.requests
    assert anon.prompt.system == 0  # 匿名请求按自身内容拆分
    assert named.prompt.system > 600  # coding 前导由带会话头的请求定型，未被污染
    assert named.prompt.tools + named.prompt.new + named.prompt.history == 800 - named.prompt.system


def test_token_accounting_cumulative_consistent(tmp_path) -> None:
    """前导定型 + 对话流累计：轮间前缀严格延伸（模拟器可命中的前提）。"""
    entries = [
        chat_entry(
            "sess_a", "coding", "2026-08-19T12:00:00.000Z", "2026-08-19T12:00:01.000Z",
            [
                {"role": "system", "content": "a" * 300},
                {"role": "user", "content": "b" * 100},
            ],
            prompt_tokens=800, completion_tokens=120,
        ),
        chat_entry(
            "sess_a", "coding", "2026-08-19T12:00:05.000Z", "2026-08-19T12:00:06.000Z",
            [
                {"role": "system", "content": "a" * 300},
                {"role": "user", "content": "b" * 100},
                {"role": "assistant", "content": "c" * 110},
                {"role": "user", "content": "d" * 90},
            ],
            prompt_tokens=1200, completion_tokens=60,
        ),
        chat_entry(
            "sess_a", "coding", "2026-08-19T12:00:20.000Z", "2026-08-19T12:00:21.000Z",
            [
                {"role": "system", "content": "a" * 300},
                {"role": "user", "content": "b" * 100},
                {"role": "assistant", "content": "c" * 110},
                {"role": "user", "content": "d" * 90},
                {"role": "assistant", "content": "e" * 55},
                {"role": "user", "content": "f" * 80},
            ],
            prompt_tokens=1500, completion_tokens=30,
        ),
    ]
    path = write_log(tmp_path, entries)
    report = parse_probe_log(path)
    t1, t2, t3 = report.requests
    # 前导在三轮间完全一致
    assert (t1.prompt.system, t1.prompt.tools) == (t2.prompt.system, t2.prompt.tools) == (t3.prompt.system, t3.prompt.tools)
    # 对话流累计推进：history(t+1) = history(t) + new(t) + output(t)
    assert t2.prompt.history == t1.prompt.new + t1.output_tokens
    assert t3.prompt.history == t2.prompt.history + t2.prompt.new + t2.output_tokens
    # 残差守恒：四段之和恰为 usage.prompt_tokens
    assert t1.prompt.total == 800
    assert t2.prompt.total == 1200
    assert t3.prompt.total == 1500
    assert all(t.prompt.new >= 0 for t in report.requests)


AZURE_FIXTURE = "\n".join(
    [
        "TIMESTAMP,ContextTokens,GeneratedTokens",
        "2024-05-10 00:00:00.010000+00:00,2162,5",
        "2024-05-10 00:00:02.500000+00:00,76,15",
        "2024-05-10 00:00:01.000000+00:00,900,3",  # 乱序行：应按时间排序
        "not,a,timestamp",
        "2024-05-10 00:00:03.000000+00:00,abc,7",  # token 非整数：跳过
        "",
    ]
)


def test_real_tokens_trace_from_csv_parses_and_sorts(tmp_path) -> None:
    """三列生产 trace（时间戳/上下文/生成）→ 单轮请求，真实 token 保留。"""
    path = tmp_path / "trace.csv"
    path.write_text(AZURE_FIXTURE, encoding="utf-8")
    requests = real_tokens_trace_from_csv(path, preamble_tokens=1024)

    assert len(requests) == 3
    first, second, third = requests  # 0.99s / 1.99s 间隔（乱序行已归位）
    assert [r.arrival_time for r in requests] == pytest.approx([0.0, 0.99, 2.49])
    assert all(r.turn_id == 1 and r.think_time == 0.0 and r.agent_type == "chat" for r in requests)

    # system 段承载共享前缀，new = max(0, ctx - 前缀)，生成 token 原样保留
    assert (first.prompt.system, first.prompt.new, first.output_tokens) == (1024, 2162 - 1024, 5)
    assert second.prompt.new == 0  # ctx=900 < 前缀：new 钳到 0，总 prompt 为前缀
    assert second.prompt.total == 1024
    assert second.output_tokens == 3
    assert third.output_tokens == 15
    assert len({r.session_id for r in requests}) == 3


def test_real_tokens_trace_from_csv_epoch_and_cap(tmp_path) -> None:
    """epoch 秒时间戳可解析；``max_requests`` 截断排序后的前 N 个到达。"""
    path = tmp_path / "epoch.csv"
    path.write_text(
        "\n".join(
            [
                "1000.0,500,10",
                "1002.5,600,20",
                "1001.0,700,30",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    requests = real_tokens_trace_from_csv(path, preamble_tokens=256, max_requests=2)
    assert [r.arrival_time for r in requests] == [0.0, 1.0]
    assert requests[1].prompt.new == 700 - 256  # 截断保留的是最早到达


def test_real_tokens_trace_from_csv_tolerates_float_tokens(tmp_path) -> None:
    """token 列的浮点写法（BurstGPT 风格）按截断整数接受；负值行跳过。"""
    path = tmp_path / "floats.csv"
    path.write_text(
        "\n".join(
            [
                "TIMESTAMP,ContextTokens,GeneratedTokens",
                "2024-05-10 00:00:00+00:00,2162.0,5.0",
                "2024-05-10 00:00:01+00:00,-3,7",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    requests = real_tokens_trace_from_csv(path, preamble_tokens=64)
    assert len(requests) == 1  # 负 token 行被跳过
    assert (requests[0].prompt.new, requests[0].output_tokens) == (2162 - 64, 5)
