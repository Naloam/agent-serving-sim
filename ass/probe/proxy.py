"""OpenAI 兼容流量的记录代理。

部署形态：客户端（agent 驱动器）把 base_url 指向本代理，代理原样转发到
上游推理服务（Ollama / vLLM / 任何 OpenAI 兼容端点），并把每个请求的
原始事实追加写入 JSONL 日志：

- ``ts_request`` / ``ts_first_byte`` / ``ts_complete``：墙钟时间戳；
- ``session_id`` / ``agent_type``：取自定义头 ``x-ass-session-id`` /
  ``x-ass-agent-type``（缺失记 null，由 loader 兜底）；
- ``request``：原始 messages / tools（loader 负责四段拆分与 token 估算）；
- ``usage``：上游返回的 prompt/completion token 数（非流式才有精确值）。

设计约束：仅标准库；透传优先——日志写在响应完成之后，绝不阻塞请求；
上游不可达时向客户端回 502 并记录错误行。
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

SESSION_HEADER = "x-ass-session-id"
AGENT_HEADER = "x-ass-agent-type"
# 不向上游转发的头：逐跳头与压缩（保证代理可解析 JSON 响应）
HOP_BY_HOP = {
    "host",
    "connection",
    "content-length",
    "keep-alive",
    "transfer-encoding",
    "upgrade",
    "accept-encoding",
}
UPSTREAM_TIMEOUT_SECONDS = 600.0


class RecordingProxy:
    """把 OpenAI 兼容流量透传到上游并逐请求落盘。"""

    def __init__(
        self,
        upstream_base: str,
        host: str = "127.0.0.1",
        port: int = 8001,
        log_path: str | Path = "traces/real/raw/probe.jsonl",
    ) -> None:
        self.upstream_base = upstream_base.rstrip("/")
        self.host = host
        self.port = port
        self.log_path = Path(log_path)
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        proxy = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                pass  # 静默默认访问日志，避免干扰采集

            def do_GET(self) -> None:
                proxy._forward(self, method="GET", body=None)

            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length") or 0)
                body = self.rfile.read(length) if length else None
                proxy._forward(self, method="POST", body=body)

            def do_DELETE(self) -> None:
                proxy._forward(self, method="DELETE", body=None)

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = self._server.server_address[1]  # port=0 时回读实际绑定端口
        self._server.daemon_threads = True
        self._thread: threading.Thread | None = None

    # ---- 生命周期 ----

    @property
    def url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def start_background(self) -> None:
        """在守护线程中启动（供测试与库内使用）。"""
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True
        )
        self._thread.start()

    def stop(self) -> None:
        self._server.shutdown()
        self._server.server_close()

    # ---- 转发与记录 ----

    def _forward(self, handler: BaseHTTPRequestHandler, method: str, body: bytes | None) -> None:
        ts_request = _now_iso()
        t0 = time.perf_counter()
        path = handler.path
        headers = {
            key: value
            for key, value in handler.headers.items()
            if key.lower() not in HOP_BY_HOP
        }
        entry = {
            "ts_request": ts_request,
            "ts_first_byte": None,
            "ts_complete": None,
            "session_id": handler.headers.get(SESSION_HEADER),
            "agent_type": handler.headers.get(AGENT_HEADER),
            "method": method,
            "path": path,
            "stream": bool(json.loads(body).get("stream", False))
            if method == "POST" and body
            else False,
            "status": None,
            "error": None,
            "request": None,
            "usage": None,
        }
        if method == "POST" and body:
            try:
                payload = json.loads(body)
                entry["request"] = {
                    "model": payload.get("model"),
                    "messages": payload.get("messages", []),
                    "tools": payload.get("tools", []),
                }
            except json.JSONDecodeError as exc:
                entry["error"] = f"unparsable request body: {exc.msg}"

        url = f"{self.upstream_base}{path}"
        request = urllib.request.Request(url, data=body, method=method)
        for key, value in headers.items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=UPSTREAM_TIMEOUT_SECONDS) as upstream:
                status = upstream.status
                content_type = upstream.headers.get("Content-Type", "application/json")
                is_stream = content_type.startswith("text/event-stream")
                handler.send_response(status)
                handler.send_header("Content-Type", content_type)
                if is_stream:
                    handler.send_header("Connection", "close")
                    handler.end_headers()
                    first = True
                    received = bytearray()
                    while True:
                        chunk = upstream.read(4096)
                        if not chunk:
                            break
                        if first:
                            entry["ts_first_byte"] = _now_iso()
                            first = False
                        received.extend(chunk)
                        handler.wfile.write(chunk)
                        handler.wfile.flush()
                    entry["status"] = status
                    entry["ts_complete"] = _now_iso()
                    entry["usage"] = _extract_stream_usage(bytes(received))
                else:
                    payload_bytes = upstream.read()
                    handler.send_header("Content-Length", str(len(payload_bytes)))
                    handler.end_headers()
                    handler.wfile.write(payload_bytes)
                    entry["ts_first_byte"] = entry["ts_complete"] = _now_iso()
                    entry["status"] = status
                    try:
                        parsed = json.loads(payload_bytes)
                        entry["usage"] = parsed.get("usage")
                    except json.JSONDecodeError:
                        pass
        except urllib.error.HTTPError as exc:
            payload_bytes = exc.read()
            handler.send_response(exc.code)
            handler.send_header("Content-Type", exc.headers.get("Content-Type", "application/json"))
            handler.send_header("Content-Length", str(len(payload_bytes)))
            handler.end_headers()
            handler.wfile.write(payload_bytes)
            entry["status"] = exc.code
            entry["ts_first_byte"] = entry["ts_complete"] = _now_iso()
            entry["error"] = f"upstream http error: {exc.code}"
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            message = f"upstream unreachable: {exc}"
            entry["error"] = message
            entry["ts_complete"] = _now_iso()
            body_out = json.dumps({"error": {"message": message}}).encode()
            handler.send_response(502)
            handler.send_header("Content-Type", "application/json")
            handler.send_header("Content-Length", str(len(body_out)))
            handler.end_headers()
            handler.wfile.write(body_out)
        entry["elapsed_s"] = round(time.perf_counter() - t0, 6)
        self._write_entry(entry)

    def _write_entry(self, entry: dict) -> None:
        line = json.dumps(entry, ensure_ascii=False)
        with self._lock:
            with self.log_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")


def _now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()) + f".{int(time.time() * 1000) % 1000:03d}Z"


def _extract_stream_usage(raw: bytes) -> dict | None:
    """从 SSE 流中提取最后一个 usage 字段（OpenAI 流式规范在末块）。"""
    usage = None
    for line in raw.split(b"\n"):
        stripped = line.strip()
        if not stripped.startswith(b"data:"):
            continue
        data = stripped[5:].strip()
        if data in (b"", b"[DONE]"):
            continue
        try:
            parsed = json.loads(data)
        except json.JSONDecodeError:
            continue
        candidate = parsed.get("usage")
        if candidate:
            usage = candidate
    return usage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="OpenAI-compatible recording proxy")
    parser.add_argument("--upstream", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--log", type=str, default="traces/real/raw/probe.jsonl")
    args = parser.parse_args(argv)
    proxy = RecordingProxy(args.upstream, args.host, args.port, args.log)
    print(f"probe listening on {proxy.url} -> {args.upstream}, log: {args.log}", flush=True)
    try:
        proxy._server.serve_forever()
    except KeyboardInterrupt:
        proxy.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
