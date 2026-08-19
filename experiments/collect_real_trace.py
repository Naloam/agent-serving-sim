"""真实 agent trace 采集驱动器（M2，配合 ass.probe 与 Ollama 使用）。

在同一进程内启动记录代理（RecordingProxy），把两类模拟 agent 的
OpenAI 兼容流量导向代理：

- **coding agent**：任务池 + 工具（run_tests / read_file / write_file），
  轮间思考时间短（对数正态，中位数 ~5s）；
- **search agent**：问题池 + web_search 工具，SERP 摘要进入历史，
  轮间思考时间长（中位数 ~20s）。

工具结果为程序合成（本采集只关心负载结构与服务计时，不执行真实工具）。
会话按泊松间隔错峰启动、多会话并发争抢同一 GPU，形成真实的多租户负载。

停止条件（先到为准）：请求总数达到目标、墙钟预算耗尽（不再发起新一轮，
在途请求自然完成）。

用法::

    python experiments/collect_real_trace.py --coding-sessions 64 --search-sessions 40
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from ass.probe.proxy import RecordingProxy

CODING_SYSTEM = (
    "You are a focused coding agent working in a Python repository (agent-serving-sim). "
    "The repo contains a discrete-event simulator with modules: core/event.py, core/sim.py, "
    "cache/radix.py (radix tree KV cache), cache/policies.py (FIFO/LRU/TTL eviction), "
    "scheduler/serving.py, metrics/collector.py. Tests live in tests/. "
    "Work step by step: read relevant files, reason, propose or write patches, and run tests "
    "with the run_tests tool before concluding. Keep answers concise and technical."
)

SEARCH_SYSTEM = (
    "You are a diligent research assistant. Answer questions using the web_search tool. "
    "Search first, then synthesize a short answer citing what you found. "
    "If the results are insufficient, search again with refined keywords before answering. "
    "Keep answers under 150 words unless asked otherwise."
)

CODING_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_tests",
            "description": "Run the pytest suite (optionally for one file) and return the output.",
            "parameters": {
                "type": "object",
                "properties": {
                    "target": {"type": "string", "description": "test file or empty for all"}
                },
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a source file from the repository.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write a patch to a file in the repository.",
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"],
            },
        },
    },
]

SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "web_search",
            "description": "Search the web and return ranked snippets.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    }
]

CODING_TASKS = [
    "The radix tree insert path duplicates logic between full-match and partial-match branches; refactor it and add a regression test.",
    "TTLPolicy sweeps expired leaves at event time; investigate whether sweeping at arrival time only skews think_time stats.",
    "Add a FIFO-with-second-chance eviction policy behind the existing registry.",
    "MetricsCollector.summary() recomputes percentiles per call; cache them and measure the win.",
    "Simulation.run(until) raises on past scheduling; the serving loop relies on it — document or relax the invariant.",
    "The synthetic generator draws preamble lengths per agent type; make them per-session configurable.",
    "Investigate why uncached requests bypass the waiting queue in serving.py and whether that biases queue_delay stats.",
    "Write a benchmark script that times 100k-request simulations and asserts NFR-2.",
    "RequestRecord lacks prefill/decode split; add fields and keep CSV backward compatible.",
    "The probe writes its JSONL log after responding; assess the risk of losing tail entries on crash.",
    "Add type-checked loading of experiment configs from TOML.",
    "Trace validation rejects unknown fields; soften to warnings for forward compatibility.",
    "plot_cdf recomputes sorted values per series; hoist the sort out of the loop.",
    "Investigate LRU thrashing when cache capacity equals the preamble working set.",
    "Add session-level quota accounting to the metrics collector.",
    "Document the Segment(stream, length) abstraction for new contributors.",
]

SEARCH_QUESTIONS = [
    "What is PagedAttention and why does it matter for LLM serving throughput?",
    "How does SGLang's RadixAttention reuse KV cache across requests?",
    "What causes head-of-line blocking in continuous batching schedulers?",
    "Compare vLLM and TensorRT-LLM prefix caching strategies.",
    "What is speculative decoding and when does it hurt latency?",
    "How do TTL-based cache policies behave under heavy-tailed inter-arrival times?",
    "What KV cache quantization schemes exist for multi-tenant inference?",
    "Summarize the debate around disaggregated prefill/decode serving.",
    "What are the memory bandwidth implications of grouped-query attention?",
    "How do agent frameworks like LangGraph affect inference load patterns?",
    "What techniques reduce time-to-first-token in conversational LLM serving?",
    "Explain chunked prefill and its effect on tail latency.",
    "What is a radix tree and where else is it used in systems?",
    "How does llama.cpp handle prompt caching across sequential calls?",
    "What metrics matter when evaluating LLM inference schedulers?",
    "How do prefix-aware routers like llm-d place requests?",
]

FOLLOW_UPS = [
    "Continue; run the tests again and fix any regressions.",
    "Good. Now handle the edge case we discussed and add a unit test.",
    "Refactor that to reuse the existing helper instead.",
    "Update the docs comment to match the new behavior.",
    "Profile the change and report the delta.",
    "Now apply the same treatment to the sibling module.",
    "Re-check the failure mode under concurrency and summarize.",
    "Tighten the error messages and finalize.",
]

SEARCH_FOLLOW_UPS = [
    "Follow up: what are the practical deployment caveats?",
    "Follow up: how does this interact with quantized models?",
    "Follow up: summarize the benchmark evidence pro and contra.",
    "Follow up: what changed in the last two years?",
    "Follow up: which open-source implementations should I read?",
    "Follow up: give a concrete numeric example.",
]

FAKE_PY_FILE = "\n".join(
    [
        "import heapq",
        "from dataclasses import dataclass",
        "",
        "",
        "@dataclass(order=True)",
        "class Event:",
        "    time: float",
        "    priority: int = 0",
        "    seq: int = 0",
        "",
        "",
        "class Simulation:",
        "    def __init__(self):",
        "        self._heap = []",
        "        self._now = 0.0",
        "",
        "    def schedule(self, time, callback, priority=0):",
        "        if time < self._now:",
        "            raise ValueError('past')",
        "        heapq.heappush(self._heap, Event(time, priority, callback))",
        "",
        "    def run(self, until=None):",
        "        while self._heap and (until is None or self._heap[0].time <= until):",
        "            event = heapq.heappop(self._heap)",
        "            self._now = event.time",
        "            event.callback()",
        "        return self._now",
    ]
)


class Collector:
    """共享状态：请求计数、停止条件、进度与心跳。"""

    def __init__(self, target_requests: int, wall_budget_s: float, heartbeat: Path | None) -> None:
        self.target_requests = target_requests
        self.wall_budget_s = wall_budget_s
        self.deadline = time.monotonic() + wall_budget_s
        self.completed = 0
        self.lock = threading.Lock()
        self.heartbeat = heartbeat
        self.started_at = datetime.now(timezone.utc)

    def should_stop(self) -> bool:
        return self.completed >= self.target_requests or time.monotonic() > self.deadline

    def record_completion(self) -> None:
        with self.lock:
            self.completed += 1
            done = self.completed
        if done % 10 == 0:
            elapsed = time.monotonic() - (self.deadline - self.wall_budget_s)
            rate = done / max(elapsed, 1e-9)
            remaining = max(self.target_requests - done, 0)
            print(
                f"[progress] {done} requests done, {rate * 60:.1f}/min, "
                f"~{remaining / max(rate, 1e-9) / 60:.0f}min remaining",
                flush=True,
            )
        if self.heartbeat is not None and done % 5 == 0:
            self.heartbeat.write_text(
                datetime.now(timezone.utc).isoformat(), encoding="utf-8"
            )


WALL_BUDGET_DEFAULT = 3.2 * 3600.0


def chat(base_url: str, model: str, messages: list[dict], tools: list[dict] | None,
         max_tokens: int, headers: dict[str, str]) -> dict:
    payload: dict = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "stream": False,
    }
    if tools:
        payload["tools"] = tools
    last_error: Exception | None = None
    body = json.dumps(payload).encode()
    for attempt in range(3):
        request = urllib.request.Request(
            f"{base_url}/v1/chat/completions", data=body, method="POST"
        )
        request.add_header("Content-Type", "application/json")
        for key, value in headers.items():
            request.add_header(key, value)
        try:
            with urllib.request.urlopen(request, timeout=600) as response:
                return json.loads(response.read())
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            last_error = exc
            time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"chat request failed after retries: {last_error}")


def synthesize_tool_result(name: str, arguments: dict, rng: random.Random) -> str:
    if name == "run_tests":
        failures = rng.choice([0, 0, 0, 1, 2])
        total = rng.randint(18, 64)
        duration = rng.uniform(0.8, 9.0)
        if failures == 0:
            body = f"Ran {total} tests in {duration:.1f}s\n\nOK"
        else:
            names = "\n".join(
                f"FAILED tests/test_mod{rng.randint(1, 9)}.py::test_case{rng.randint(1, 40)} - "
                f"AssertionError: expected {rng.randint(0, 9)} got {rng.randint(0, 9)}"
                for _ in range(failures)
            )
            body = f"Ran {total} tests in {duration:.1f}s\n\nFAILED (failures={failures})\n{names}"
        return body
    if name == "read_file":
        return f"# {arguments.get('path', 'ass/core/sim.py')}\n{FAKE_PY_FILE}"
    if name == "write_file":
        lines = arguments.get("content", "").count("\n") + 1
        return f"wrote {max(lines, 1)} lines to {arguments.get('path', 'patched.py')}; lint OK"
    if name == "web_search":
        query = arguments.get("query", "llm serving")
        results = []
        for index in range(3):
            domain = rng.choice(["arxiv.org", "blog.vllm.ai", "github.com", "dl.acm.org", "lmsys.org"])
            year = rng.randint(2023, 2026)
            blurb = " ".join(
                rng.choice([
                    "prefix caching", "KV cache eviction", "paged attention",
                    "continuous batching", "disaggregated serving", "agent workloads",
                    "time-to-first-token", "radix tree reuse", "speculative decoding",
                ])
                for _ in range(rng.randint(12, 26))
            )
            results.append(f"[{index + 1}] ({domain}, {year}) {query}: {blurb}...")
        return "\n\n".join(results)
    return "ok"


def run_coding_session(index: int, base_url: str, model: str, collector: Collector,
                        rng: random.Random) -> int:
    session_id = f"coding-{index:03d}"
    headers = {"x-ass-session-id": session_id, "x-ass-agent-type": "coding"}
    messages: list[dict] = [
        {"role": "system", "content": CODING_SYSTEM},
        {"role": "user", "content": f"Task: {CODING_TASKS[index % len(CODING_TASKS)]}"},
    ]
    calls = 0
    max_calls = rng.randint(8, 14)
    while calls < max_calls and not collector.should_stop():
        response = chat(base_url, model, messages, CODING_TOOLS, 256, headers)
        calls += 1
        collector.record_completion()
        message = response["choices"][0]["message"]
        finish = response["choices"][0].get("finish_reason")
        messages.append(message)
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            for call in tool_calls:
                function = call.get("function", {})
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": synthesize_tool_result(function.get("name", ""), arguments, rng),
                    }
                )
            time.sleep(rng.lognormvariate(1.61, 0.55))  # 中位数 ~5s 的真实轮间空闲
        elif finish == "stop" or not message.get("content"):
            if calls >= max_calls or rng.random() < 0.18:
                break
            messages.append(
                {"role": "user", "content": rng.choice(FOLLOW_UPS)}
            )
            time.sleep(rng.lognormvariate(1.79, 0.6))  # 中位数 ~6s
        else:
            time.sleep(rng.lognormvariate(1.61, 0.55))
    return calls


def run_search_session(index: int, base_url: str, model: str, collector: Collector,
                       rng: random.Random) -> int:
    session_id = f"search-{index:03d}"
    headers = {"x-ass-session-id": session_id, "x-ass-agent-type": "search"}
    messages: list[dict] = [
        {"role": "system", "content": SEARCH_SYSTEM},
        {"role": "user", "content": SEARCH_QUESTIONS[index % len(SEARCH_QUESTIONS)]},
    ]
    calls = 0
    max_calls = rng.randint(6, 10)
    while calls < max_calls and not collector.should_stop():
        response = chat(base_url, model, messages, SEARCH_TOOLS, 192, headers)
        calls += 1
        collector.record_completion()
        message = response["choices"][0]["message"]
        finish = response["choices"][0].get("finish_reason")
        messages.append(message)
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            for call in tool_calls:
                function = call.get("function", {})
                try:
                    arguments = json.loads(function.get("arguments") or "{}")
                except json.JSONDecodeError:
                    arguments = {}
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.get("id"),
                        "content": synthesize_tool_result(function.get("name", ""), arguments, rng),
                    }
                )
            time.sleep(rng.lognormvariate(2.5, 0.7))  # 中位数 ~12s：阅读搜索结果
        elif finish == "stop" or not message.get("content"):
            if calls >= max_calls or rng.random() < 0.15:
                break
            messages.append({"role": "user", "content": rng.choice(SEARCH_FOLLOW_UPS)})
            time.sleep(rng.lognormvariate(3.0, 0.7))  # 中位数 ~20s：下一个问题
        else:
            time.sleep(rng.lognormvariate(2.5, 0.7))
    return calls


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Collect real agent traces via probe + Ollama")
    parser.add_argument("--upstream", type=str, default="http://127.0.0.1:11434")
    parser.add_argument("--model", type=str, default="qwen2.5-coder-16k:latest")
    parser.add_argument("--port", type=int, default=8001)
    parser.add_argument("--coding-sessions", type=int, default=64)
    parser.add_argument("--search-sessions", type=int, default=40)
    parser.add_argument("--target-requests", type=int, default=1050)
    parser.add_argument("--wall-budget-hours", type=float, default=3.2)
    parser.add_argument("--session-rate", type=float, default=0.09, help="会话启动速率（个/秒）")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--raw-log", type=str, default="traces/real/raw/probe.jsonl")
    args = parser.parse_args(argv)

    heartbeat = Path(".agent-heartbeat")
    raw_log = Path(args.raw_log)
    probe = RecordingProxy(args.upstream, port=args.port, log_path=raw_log)
    probe.start_background()
    print(f"probe at {probe.url} -> {args.upstream}, log {raw_log}", flush=True)

    # 健康检查 + 预热（把 16k 变体载入显存，避免首个会话承担冷启动）
    with urllib.request.urlopen(f"{probe.url}/v1/models", timeout=30) as response:
        models = [m.get("id") for m in json.loads(response.read()).get("data", [])]
    if args.model not in models:
        print(f"model {args.model} not available: {models}", file=sys.stderr)
        return 2
    print("warming up model (cold load may take ~1min)...", flush=True)
    chat(probe.url, args.model, [{"role": "user", "content": "ready?"}], None, 8, {})
    print("warmup done", flush=True)

    collector = Collector(
        target_requests=args.target_requests,
        wall_budget_s=args.wall_budget_hours * 3600.0,
        heartbeat=heartbeat,
    )
    rng = random.Random(args.seed)
    sessions: list[tuple[str, int]] = [("coding", i) for i in range(args.coding_sessions)]
    sessions += [("search", i) for i in range(args.search_sessions)]
    rng.shuffle(sessions)

    t_start = time.monotonic()
    results: dict[str, int] = {}
    results_lock = threading.Lock()

    def launch(kind: str, index: int) -> None:
        session_rng = random.Random(args.seed * 1009 + index * 31 + (0 if kind == "coding" else 1))
        base = probe.url
        if kind == "coding":
            calls = run_coding_session(index, base, args.model, collector, session_rng)
        else:
            calls = run_search_session(index, base, args.model, collector, session_rng)
        with results_lock:
            results[f"{kind}-{index:03d}"] = calls

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for kind, index in sessions:
            if collector.should_stop():
                break
            futures.append(executor.submit(launch, kind, index))
            time.sleep(rng.expovariate(args.session_rate))
        for future in futures:
            future.result()

    duration = time.monotonic() - t_start
    meta = {
        "started_at": collector.started_at.isoformat(),
        "duration_s": round(duration, 1),
        "completed_requests": collector.completed,
        "target_requests": args.target_requests,
        "sessions": results,
        "model": args.model,
        "config": {
            "coding_sessions": args.coding_sessions,
            "search_sessions": args.search_sessions,
            "session_rate": args.session_rate,
        },
    }
    meta_path = raw_log.parent / "driver_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("duration_s", "completed_requests")}, indent=2), flush=True)
    print(f"meta written to {meta_path}", flush=True)
    time.sleep(1.0)  # 给最后几条日志的落盘留缓冲
    probe.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
