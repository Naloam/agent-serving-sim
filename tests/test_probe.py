"""采集探针的单元测试（FR-11）：透传不变形、记录完整、错误路径。"""

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest
import urllib.request

from ass.probe.proxy import RecordingProxy


def read_log_lines(path, expected: int = 1, timeout: float = 5.0) -> list[dict]:
    """轮询读取原始日志（代理在响应返回后才写日志，存在毫秒级竞态）。"""
    deadline = time.monotonic() + timeout
    while True:
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except FileNotFoundError:
            lines = []
        if len(lines) >= expected:
            return [json.loads(line) for line in lines[:expected]]
        if time.monotonic() > deadline:
            raise AssertionError(f"log still has {len(lines)} lines after {timeout}s")
        time.sleep(0.05)

UPSTREAM_RESPONSE = {
    "id": "chatcmpl-1",
    "model": "qwen2.5-coder:7b",
    "choices": [
        {"index": 0, "message": {"role": "assistant", "content": "hello"}, "finish_reason": "stop"}
    ],
    "usage": {"prompt_tokens": 1234, "completion_tokens": 56},
}


class _UpstreamHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass

    def _respond(self, status: int, payload: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:
        if self.path == "/v1/models":
            self._respond(200, b'{"data": []}', "application/json")
        else:
            self._respond(404, b"{}", "application/json")

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if self.path == "/v1/chat/completions":
            body = json.dumps(UPSTREAM_RESPONSE).encode()
            self._respond(200, body, "application/json")
        else:
            self._respond(404, b"{}", "application/json")


class _StreamUpstreamHandler(_UpstreamHandler):
    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Connection", "close")
        self.end_headers()
        chunks = [
            b'data: {"choices": [{"delta": {"content": "he"}}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "llo"}}]}\n\n',
            b'data: {"choices": [], "usage": {"prompt_tokens": 99, "completion_tokens": 3}}\n\n',
            b"data: [DONE]\n\n",
        ]
        for chunk in chunks:
            self.wfile.write(chunk)
            self.wfile.flush()


@pytest.fixture()
def upstream():
    server = ThreadingHTTPServer(("127.0.0.1", 0), _UpstreamHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}"
    server.shutdown()
    server.server_close()


@pytest.fixture()
def proxy(upstream, tmp_path):
    probe = RecordingProxy(upstream, port=0, log_path=tmp_path / "probe.jsonl")
    probe.start_background()
    yield probe
    probe.stop()


def _post_chat(proxy: RecordingProxy, extra_headers: dict | None = None) -> dict:
    payload = json.dumps(
        {
            "model": "qwen2.5-coder:7b",
            "messages": [
                {"role": "system", "content": "You are a coding agent."},
                {"role": "user", "content": "hi"},
            ],
        }
    ).encode()
    request = urllib.request.Request(
        f"{proxy.url}/v1/chat/completions", data=payload, method="POST"
    )
    request.add_header("Content-Type", "application/json")
    for key, value in (extra_headers or {}).items():
        request.add_header(key, value)
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read())


def test_chat_request_passed_through_and_logged(proxy, tmp_path) -> None:
    """请求透传不变形，且记录含会话头、usage 与三个时间戳。"""
    result = _post_chat(
        proxy,
        {"x-ass-session-id": "sess_0001", "x-ass-agent-type": "coding"},
    )
    assert result == UPSTREAM_RESPONSE  # 客户端拿到的与上游完全一致
    (entry,) = read_log_lines(tmp_path / "probe.jsonl")
    assert entry["session_id"] == "sess_0001"
    assert entry["agent_type"] == "coding"
    assert entry["status"] == 200
    assert entry["usage"]["prompt_tokens"] == 1234
    assert entry["usage"]["completion_tokens"] == 56
    assert entry["ts_request"] and entry["ts_first_byte"] and entry["ts_complete"]
    assert entry["request"]["messages"][0]["role"] == "system"
    assert entry["error"] is None


def test_unreachable_upstream_returns_502(tmp_path) -> None:
    probe = RecordingProxy(
        "http://127.0.0.1:1", port=0, log_path=tmp_path / "probe.jsonl"
    )
    probe.start_background()
    try:
        payload = json.dumps({"model": "m", "messages": []}).encode()
        request = urllib.request.Request(
            f"{probe.url}/v1/chat/completions", data=payload, method="POST"
        )
        request.add_header("Content-Type", "application/json")
        try:
            urllib.request.urlopen(request, timeout=10)
            raised = False
        except urllib.error.HTTPError as exc:
            assert exc.code == 502
            raised = True
        assert raised
        (entry,) = read_log_lines(tmp_path / "probe.jsonl")
        assert entry["error"] and "unreachable" in entry["error"]
    finally:
        probe.stop()


def test_get_models_passed_through(proxy, tmp_path) -> None:
    with urllib.request.urlopen(f"{proxy.url}/v1/models", timeout=10) as response:
        assert response.status == 200
        assert json.loads(response.read()) == {"data": []}
    # 原始日志记录全部流量；非会话请求无 usage/request 字段，由 loader 过滤
    (entry,) = read_log_lines(tmp_path / "probe.jsonl")
    assert entry["path"] == "/v1/models"
    assert entry["request"] is None and entry["usage"] is None


def test_stream_response_forwarded_and_usage_extracted(upstream, tmp_path) -> None:
    stream_server = ThreadingHTTPServer(("127.0.0.1", 0), _StreamUpstreamHandler)
    threading.Thread(target=stream_server.serve_forever, daemon=True).start()
    try:
        upstream_url = f"http://127.0.0.1:{stream_server.server_address[1]}"
        probe = RecordingProxy(upstream_url, port=0, log_path=tmp_path / "s.jsonl")
        probe.start_background()
        try:
            payload = json.dumps(
                {"model": "m", "messages": [], "stream": True}
            ).encode()
            request = urllib.request.Request(
                f"{probe.url}/v1/chat/completions", data=payload, method="POST"
            )
            request.add_header("Content-Type", "application/json")
            with urllib.request.urlopen(request, timeout=10) as response:
                body = response.read()
            assert b'"he"' in body and b"[DONE]" in body
            (entry,) = read_log_lines(tmp_path / "s.jsonl")
            assert entry["stream"] is True
            assert entry["usage"]["prompt_tokens"] == 99
        finally:
            probe.stop()
    finally:
        stream_server.shutdown()
        stream_server.server_close()
