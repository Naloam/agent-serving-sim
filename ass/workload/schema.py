"""Trace 格式定义与 JSONL 读写。

每行一个请求；``prompt`` 四段分解（system / tools / history / new）让
模拟器无需 token 内容即可推算前缀复用结构。
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROMPT_FIELDS: tuple[str, ...] = ("system", "tools", "history", "new")
REQUEST_FIELDS: tuple[str, ...] = (
    "session_id",
    "turn_id",
    "arrival_time",
    "prompt",
    "output_tokens",
    "think_time",
    "agent_type",
    "priority",
)


class TraceValidationError(ValueError):
    """trace 数据不合法时抛出，message 说明字段与原因。"""


@dataclass(frozen=True)
class PromptBreakdown:
    """prompt 的四段 token 数分解，均须非负。"""

    system: int
    tools: int
    history: int
    new: int

    def __post_init__(self) -> None:
        for name in PROMPT_FIELDS:
            value = getattr(self, name)
            if not _is_int(value) or value < 0:
                raise TraceValidationError(
                    f"prompt.{name} must be a non-negative int, got {value!r}"
                )

    @property
    def total(self) -> int:
        """prompt 总 token 数。"""
        return self.system + self.tools + self.history + self.new


@dataclass(frozen=True)
class TraceRequest:
    """单个请求（trace 中的一行）。"""

    session_id: str
    turn_id: int
    arrival_time: float
    prompt: PromptBreakdown
    output_tokens: int
    think_time: float
    agent_type: str
    priority: int

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise TraceValidationError("session_id must be a non-empty string")
        if not _is_int(self.turn_id) or self.turn_id < 1:
            raise TraceValidationError(f"turn_id must be an int >= 1, got {self.turn_id!r}")
        if not _is_number(self.arrival_time) or self.arrival_time < 0:
            raise TraceValidationError(
                f"arrival_time must be a non-negative number, got {self.arrival_time!r}"
            )
        if not _is_int(self.output_tokens) or self.output_tokens < 0:
            raise TraceValidationError(
                f"output_tokens must be a non-negative int, got {self.output_tokens!r}"
            )
        if not _is_number(self.think_time) or self.think_time < 0:
            raise TraceValidationError(
                f"think_time must be a non-negative number, got {self.think_time!r}"
            )
        if not isinstance(self.agent_type, str) or not self.agent_type:
            raise TraceValidationError("agent_type must be a non-empty string")
        if not _is_int(self.priority):
            raise TraceValidationError(f"priority must be an int, got {self.priority!r}")


def _is_int(value: Any) -> bool:
    # bool 是 int 子类，显式排除
    return isinstance(value, int) and not isinstance(value, bool)


def _is_number(value: Any) -> bool:
    return _is_int(value) or isinstance(value, float)


def request_to_dict(request: TraceRequest) -> dict[str, Any]:
    """转换为与 JSONL 行一致的 dict。"""
    return asdict(request)


def request_from_dict(data: dict[str, Any]) -> TraceRequest:
    """从 dict 构造请求；多余/缺失字段与类型错误均报明确异常。"""
    if not isinstance(data, dict):
        raise TraceValidationError(f"request line must be a JSON object, got {type(data).__name__}")
    unknown = sorted(set(data) - set(REQUEST_FIELDS))
    if unknown:
        raise TraceValidationError(f"unknown fields: {unknown}")
    missing = [name for name in REQUEST_FIELDS if name not in data]
    if missing:
        raise TraceValidationError(f"missing fields: {missing}")
    prompt_raw = data["prompt"]
    if not isinstance(prompt_raw, dict):
        raise TraceValidationError("prompt must be a JSON object with four segments")
    prompt_unknown = sorted(set(prompt_raw) - set(PROMPT_FIELDS))
    if prompt_unknown:
        raise TraceValidationError(f"unknown prompt segments: {prompt_unknown}")
    prompt_missing = [name for name in PROMPT_FIELDS if name not in prompt_raw]
    if prompt_missing:
        raise TraceValidationError(f"missing prompt segments: {prompt_missing}")
    prompt = PromptBreakdown(**{name: prompt_raw[name] for name in PROMPT_FIELDS})
    return TraceRequest(
        session_id=data["session_id"],
        turn_id=data["turn_id"],
        arrival_time=data["arrival_time"],
        prompt=prompt,
        output_tokens=data["output_tokens"],
        think_time=data["think_time"],
        agent_type=data["agent_type"],
        priority=data["priority"],
    )


def read_trace(path: str | Path) -> list[TraceRequest]:
    """读取 JSONL trace；非法行抛出携带 文件:行号 的 TraceValidationError。"""
    file_path = Path(path)
    requests: list[TraceRequest] = []
    for lineno, line in enumerate(file_path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        try:
            data = json.loads(stripped)
        except json.JSONDecodeError as exc:
            raise TraceValidationError(f"{file_path}:{lineno}: invalid JSON: {exc.msg}") from None
        try:
            requests.append(request_from_dict(data))
        except TraceValidationError as exc:
            raise TraceValidationError(f"{file_path}:{lineno}: {exc}") from None
    return requests


def write_trace(requests: list[TraceRequest], path: str | Path) -> None:
    """写出 JSONL trace（每行一个请求，保持输入顺序）。"""
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as handle:
        for request in requests:
            handle.write(json.dumps(request_to_dict(request), ensure_ascii=False) + "\n")
