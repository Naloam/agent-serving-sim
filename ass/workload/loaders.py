"""真实 trace 解析器（对应 PRD FR-4）。

把采集探针（ass.probe.proxy）的原始 JSONL 日志清洗、对齐为 FR-2 的
TraceRequest 格式，并顺带保留每请求的计时事实（供 M3 标定解析式计时模型）。

清洗规则：

- 只接受 ``POST /v1/chat/completions`` 且带 usage 的条目；其余（模型列表、
  坏行、流式无 usage 等）跳过并记录原因，不中断整体流程；
- 条目按 ``ts_request`` 排序后再编号轮次（多会话并发时完成顺序与到达
  顺序不一致）；
- ``turn_id`` 为会话内到达序号（从 1 起）；``think_time`` = 本轮到达 −
  同会话上一轮完成（首轮为 0，负值截断为 0）；
- **token 记账采用累计一致方案**（保证轮间前缀严格延伸，模拟器可复用）：

  - 每个 agent_type 的前导（system+tools）token 数在首个请求按字符
    比例估出后**固定**——同一应用的 system prompt 逐字节相同，逐请求
    独立估算的抖动会破坏前缀匹配；
  - 会话对话流按 ``history(t+1) = history(t) + new(t) + output(t)`` 累计，
    其中 ``output`` 取上游真实 ``completion_tokens``，
    ``new(t) = prompt_tokens(t) − 前导 − history(t)``（残差，吸收估算噪声）；
- ``arrival_time`` 归一化为相对日志内首个请求的秒数。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ass.workload.schema import PromptBreakdown, TraceRequest

CHAT_PATH = "/v1/chat/completions"
TIMESTAMP_FORMAT = "%Y-%m-%dT%H:%M:%S.%fZ"


@dataclass(frozen=True)
class ProbeTiming:
    """单请求的真实计时事实（标定用）。"""

    session_id: str
    turn_id: int
    arrival_time: float
    prompt_tokens: int
    completion_tokens: int
    total_seconds: float
    first_byte_seconds: float


@dataclass
class ParseReport:
    """解析结果：trace、计时事实与被跳过的行。"""

    requests: list[TraceRequest] = field(default_factory=list)
    timings: list[ProbeTiming] = field(default_factory=list)
    skipped: list[tuple[int, str]] = field(default_factory=list)


def parse_probe_log(
    path: str | Path,
    *,
    chars_per_token: float = 3.5,
    default_agent_type: str = "coding",
) -> ParseReport:
    """解析探针原始日志；解析失败行单独记录，不中断。"""
    file_path = Path(path)
    entries: list[tuple[float, dict[str, Any]]] = []
    report = ParseReport()
    for lineno, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            entry = json.loads(stripped)
        except json.JSONDecodeError as exc:
            report.skipped.append((lineno, f"invalid json: {exc.msg}"))
            continue
        reason = _reject_reason(entry)
        if reason:
            report.skipped.append((lineno, reason))
            continue
        try:
            ts_request = _parse_timestamp(entry["ts_request"])
        except (KeyError, ValueError) as exc:
            report.skipped.append((lineno, f"bad ts_request: {exc}"))
            continue
        entries.append((ts_request, entry))

    if not entries:
        return report
    entries.sort(key=lambda item: item[0])
    t0 = entries[0][0]

    last_completion: dict[str, float] = {}
    turn_counter: dict[str, int] = {}
    running_dlg: dict[str, int] = {}  # 会话对话流累计 token（累计一致记账）
    preamble_split: dict[str, tuple[int, int]] = {}  # agent_type -> (system, tools) 固定估计
    fallback_session = 0
    for ts_request, entry in entries:
        session_id = entry.get("session_id") or f"sess_anon_{fallback_session:04d}"
        if not entry.get("session_id"):
            fallback_session += 1
        agent_type = entry.get("agent_type") or default_agent_type
        turn_counter[session_id] = turn_counter.get(session_id, 0) + 1
        turn_id = turn_counter[session_id]

        messages = entry["request"]["messages"]
        tools = entry["request"].get("tools") or []
        usage = entry["usage"]
        prompt_total = usage["prompt_tokens"]
        output_tokens = usage["completion_tokens"]

        parts = _split_sections(messages, tools)
        weights = {name: len(text) for name, text in parts.items()}
        if sum(weights.values()) == 0:
            report.skipped.append((-1, f"session {session_id} turn {turn_id}: empty messages"))
            turn_counter[session_id] -= 1
            continue
        if agent_type not in preamble_split and entry.get("session_id"):
            # 只有带会话标识的真实流量才能定型该类型的前导
            # （匿名请求如健康检查/预热可能没有 system 段，会污染估计）
            first = _apportion(prompt_total, weights)
            preamble_split[agent_type] = (first["system"], first["tools"])
        if agent_type in preamble_split and entry.get("session_id"):
            system_tokens, tools_tokens = preamble_split[agent_type]
            preamble = system_tokens + tools_tokens
            history = running_dlg.get(session_id, 0)
            new_tokens = max(0, prompt_total - preamble - history)
        else:
            tokens = _apportion(prompt_total, weights)
            system_tokens, tools_tokens = tokens["system"], tokens["tools"]
            history = tokens["history"]
            new_tokens = tokens["new"]
        running_dlg[session_id] = history + new_tokens + output_tokens

        prev_complete = last_completion.get(session_id)
        think_time = 0.0 if prev_complete is None else max(0.0, ts_request - prev_complete)
        request = TraceRequest(
            session_id=session_id,
            turn_id=turn_id,
            arrival_time=ts_request - t0,
            prompt=PromptBreakdown(
                system=system_tokens if weights["system"] else 0,
                tools=tools_tokens if weights["tools"] else 0,
                history=history,
                new=new_tokens,
            ),
            output_tokens=output_tokens,
            think_time=think_time,
            agent_type=agent_type,
            priority=1,
        )
        report.requests.append(request)
        ts_complete = _parse_timestamp(entry["ts_complete"])
        ts_first = entry.get("ts_first_byte")
        report.timings.append(
            ProbeTiming(
                session_id=session_id,
                turn_id=turn_id,
                arrival_time=request.arrival_time,
                prompt_tokens=prompt_total,
                completion_tokens=output_tokens,
                total_seconds=ts_complete - ts_request,
                first_byte_seconds=(
                    _parse_timestamp(ts_first) - ts_request if ts_first else ts_complete - ts_request
                ),
            )
        )
        last_completion[session_id] = ts_complete
    return report


def _reject_reason(entry: dict[str, Any]) -> str | None:
    if entry.get("method") != "POST" or entry.get("path") != CHAT_PATH:
        return f"not a chat completion: {entry.get('method')} {entry.get('path')}"
    if entry.get("error"):
        return f"request error: {entry['error']}"
    request = entry.get("request")
    if not isinstance(request, dict) or not request.get("messages"):
        return "missing request messages"
    usage = entry.get("usage")
    if not isinstance(usage, dict):
        return "missing usage (stream without usage?)"
    if "prompt_tokens" not in usage or "completion_tokens" not in usage:
        return "incomplete usage fields"
    return None


def _split_sections(messages: list[dict[str, Any]], tools: list[Any]) -> dict[str, str]:
    """按角色把消息流切为四段文本。"""
    system_parts: list[str] = []
    history_parts: list[str] = []
    for message in messages[:-1]:
        text = _message_text(message)
        if message.get("role") == "system":
            system_parts.append(text)
        else:
            history_parts.append(text)
    new_text = _message_text(messages[-1]) if messages else ""
    return {
        "system": "\n".join(system_parts),
        "tools": json.dumps(tools, ensure_ascii=False) if tools else "",
        "history": "\n".join(history_parts),
        "new": new_text,
    }


def _message_text(message: dict[str, Any]) -> str:
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # 多段 content（text parts 等）
        parts = [
            part.get("text", "")
            for part in content
            if isinstance(part, dict) and part.get("type") == "text"
        ]
        return "\n".join(parts)
    return ""


def _apportion(total: int, weights: dict[str, int]) -> dict[str, int]:
    """按权重把 total 精确分摊为整数（最大余数法）。"""
    weight_sum = sum(weights.values())
    raw = {name: total * weight / weight_sum for name, weight in weights.items()}
    result = {name: int(value) for name, value in raw.items()}
    remainder = total - sum(result.values())
    order = sorted(raw, key=lambda name: raw[name] - result[name], reverse=True)
    for name in order[:remainder]:
        result[name] += 1
    return result


def _parse_timestamp(value: str) -> float:
    return datetime.strptime(value, TIMESTAMP_FORMAT).replace(tzinfo=timezone.utc).timestamp()
