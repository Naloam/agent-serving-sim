"""exp012：真实到达过程回放——用 Azure 生产 trace 验证 exp009-011 三结论。

背景：exp009（结构压过时间）与 exp010/011（突发把瓶颈从缓存层搬到调度层、
准入是第一杠杆、杠杆替代不叠加）全部基于合成 MMPP 到达。本实验把到达过程
换成 **AzureLLMInferenceTrace（code 服务，一周 1680 万请求）** 的两个确定性
连续窗口，验证结论在真实到达过程上是否复现：

- **burst 窗口**：全周每分钟请求数的峰值分钟起 120 秒（~11k 到达）；
- **calm 窗口**：速率最接近 p25 的分钟起 600 秒（~6k 到达）；
  同一真实 trace 内的两种 regime，服务配置完全相同。

负载组合（两个真实源）：**Azure 提供到达过程，traces/real/coding.jsonl
提供会话/前缀结构**。生产 trace 无会话结构，若直接单轮回放，请求间除共享
前缀外无可复用前缀，驱逐杠杆会结构性归零；与 coding trace 循环组合后，
前缀复用结构与 token 规模都来自真实采集，到达相关性（突发/平静、日内
周期）来自 Azure。跨循环会话改名（不同用户），think_time 置 0（Azure 到达
间隔本身携带时间结构）。

臂设计：准入 {fifo, sjf} × 驱逐 {fifo, lru}。驱逐不用 TTL：压缩到达下
会话内间隔 ~11ms-1s，TTL 任何合理档位都不触发主动清除（≈LRU），而会话
轮次在时间上聚集使 LRU 的 recency 信号仍然有效——fifo-vs-lru 是该场景下
有意义的驱逐对比。另设 ``--mode tokens`` 辅助臂：仅用 Azure 自身 token
单轮回放 burst 窗口（``real_tokens_trace_from_csv``），演示无会话结构时
驱逐杠杆的结构性塌缩——前缀结构（而非 token 真实性）才是驱逐策略成为
杠杆的前提。

三个验证点（对照 exp009-011 的结论）：

- C1 瓶颈迁移：驱逐杠杆（lru vs fifo）在 calm 可见、在 burst 相对份额崩塌；
- C2 准入第一杠杆：burst 下准入杠杆（sjf）>> 驱逐杠杆；
- C3 杠杆替代：sjf+lru 不优于两个单杠杆（不叠加）。

用法::

    python experiments/exp012_real_arrival_replay.py \
        --azure traces/real/external/azure_code_1week.csv --seed 42
    python experiments/exp012_real_arrival_replay.py --mode tokens ...  # 辅助臂
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ass.cache.policies import FIFOPolicy, LRUPolicy
from ass.scheduler.admission import FIFOAdmission, ShortestJobAdmission
from ass.scheduler.serving import ServingConfig, ServingSim
from ass.viz.plots import plot_sweep
from ass.workload.loaders import (
    arrival_times_from_csv,
    real_tokens_trace_from_csv,
)
from ass.workload.schema import TraceRequest, read_trace

# 窗口定义（相对全周每分钟速率分布，确定性选取）
BURST_SECONDS = 120.0
CALM_SECONDS = 600.0
CALM_TARGET_QUANTILE = 0.25

# 服务能力标定：计时模型中 prefill/decode_tps 为每请求解析速率，共享
# 资源是 max_concurrent（批槽位）。coding trace prompt 均值 ~1182 /
# output 均值 ~182：decode 800 tok/s → 每请求 ~0.23s，calm ~10 req/s
# 折算在途 ~2.4（8 槽位的 ~30%），burst ~92 req/s 折算在途 ~21（饱和
# ~2.6 倍）→ 排队主导。容量与 exp011 一致保持可比。
ADMISSIONS = (
    ("fifo", lambda: FIFOAdmission()),
    ("sjf", lambda: ShortestJobAdmission()),
)
EVITIONS = (
    ("fifo", lambda: FIFOPolicy()),
    ("lru", lambda: LRUPolicy()),
)
TABLE_COLUMNS = (
    "regime", "mode", "admission", "eviction", "requests", "jct_mean",
    "jct_p95", "ttft_mean", "queue_delay_mean", "hit_rate",
    "eviction_gain_pct", "admission_gain_pct", "preemptions",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-arrival replay (Azure x real session trace)")
    parser.add_argument("--azure", type=str,
                        default="traces/real/external/azure_code_1week.csv")
    parser.add_argument("--session-trace", type=str, default="traces/real/coding.jsonl")
    parser.add_argument("--window-dir", type=str, default="traces/real/azure_windows")
    parser.add_argument("--mode", choices=("sessions", "tokens"), default="sessions")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity", type=int, default=80_000)
    parser.add_argument("--prefill-tps", type=float, default=40_000.0)
    parser.add_argument("--decode-tps", type=float, default=800.0)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--smoke", action="store_true", help="截断到 800 到达做快速自检")
    parser.add_argument("--out-dir", type=str, default="experiments/results")
    return parser.parse_args(argv)


# ----------------------------------------------------------------------------
# 窗口提取：纯标准库一遍扫描全周 per-minute 速率，确定性选窗口后缓存为 txt
# ----------------------------------------------------------------------------

def derive_windows(azure_csv: Path, window_dir: Path) -> dict[str, list[float]]:
    """从 Azure trace 选出 burst/calm 两个窗口的相对到达秒，缓存到 window_dir。"""
    per_minute: dict[int, int] = {}
    with azure_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            raw = row[0].strip()
            if not raw or not raw[0].isdigit():
                continue
            try:
                ts = float(raw)
            except ValueError:
                from datetime import datetime

                try:
                    ts = datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    continue
            per_minute[int(ts // 60)] = per_minute.get(int(ts // 60), 0) + 1

    if not per_minute:
        raise SystemExit(f"no parsable rows in {azure_csv}")

    counts = sorted(per_minute.values())
    idx = int(CALM_TARGET_QUANTILE * (len(counts) - 1))
    calm_target = counts[idx]
    burst_minute = max(per_minute, key=lambda m: per_minute[m])
    calm_minute = min(per_minute, key=lambda m: abs(per_minute[m] - calm_target))
    print(f"per-minute rate: peak={per_minute[burst_minute]} calm(p25)~{calm_target}; "
          f"burst@minute {burst_minute}, calm@minute {calm_minute}")

    windows: dict[str, list[float]] = {}
    token_rows: dict[str, list[tuple[float, int, int]]] = {}
    spans = {"burst": (burst_minute * 60.0, BURST_SECONDS),
             "calm": (calm_minute * 60.0, CALM_SECONDS)}
    with azure_csv.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.reader(handle):
            if not row:
                continue
            raw = row[0].strip()
            if not raw or not raw[0].isdigit():
                continue
            try:
                ts = float(raw)
            except ValueError:
                from datetime import datetime

                try:
                    ts = datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
                except ValueError:
                    continue
            for name, (start, duration) in spans.items():
                if start <= ts < start + duration:
                    windows.setdefault(name, []).append(ts - start)
                    try:
                        token_rows.setdefault(name, []).append(
                            (ts - start, int(row[1]), int(row[2])))
                    except (ValueError, IndexError):
                        pass  # token 列缺失/非整数的行只计入到达

    window_dir.mkdir(parents=True, exist_ok=True)
    for name, arrivals in windows.items():
        arrivals.sort()
        (window_dir / f"{name}.txt").write_text(
            "\n".join(f"{t:.6f}" for t in arrivals) + "\n", encoding="utf-8"
        )
        rows_sorted = sorted(token_rows.get(name, []))
        with (window_dir / f"{name}_tokens.csv").open("w", encoding="utf-8", newline="") as h:
            writer = csv.writer(h)
            writer.writerow(["TIMESTAMP", "ContextTokens", "GeneratedTokens"])
            writer.writerows(rows_sorted)
        print(f"window {name}: {len(arrivals)} arrivals over {spans[name][1]:.0f}s "
              f"({len(arrivals) / spans[name][1]:.1f} req/s)")
    return windows


def load_windows(azure_csv: Path, window_dir: Path) -> dict[str, list[float]]:
    """优先读缓存的窗口 txt；缺失且源 CSV 存在时重新推导。"""
    windows: dict[str, list[float]] = {}
    for name in ("burst", "calm"):
        cache = window_dir / f"{name}.txt"
        if cache.exists():
            windows[name] = [float(line) for line in
                             cache.read_text(encoding="utf-8").split() if line]
    if windows:
        return windows
    if not azure_csv.exists():
        raise SystemExit(f"window cache missing under {window_dir} and azure csv not "
                         f"found at {azure_csv}; download it first (see README)")
    return derive_windows(azure_csv, window_dir)


# ----------------------------------------------------------------------------
# 负载组合：Azure 到达 × coding.jsonl 会话结构（循环改名，think_time 归零）
# ----------------------------------------------------------------------------

def compose_session_replay(
    arrivals: list[float], session_trace: list[TraceRequest]
) -> list[TraceRequest]:
    """把会话 trace 顺序映射到外部到达时间上，循环填充并给会话改名。"""
    composed: list[TraceRequest] = []
    cycle = 0
    while len(composed) < len(arrivals):
        suffix = f"_c{cycle}"
        for request in session_trace:
            if len(composed) >= len(arrivals):
                break
            composed.append(
                TraceRequest(
                    session_id=request.session_id + suffix,
                    turn_id=request.turn_id,
                    arrival_time=arrivals[len(composed)],
                    prompt=request.prompt,
                    output_tokens=request.output_tokens,
                    think_time=0.0,
                    agent_type=request.agent_type,
                    priority=request.priority,
                )
            )
        cycle += 1
    return composed


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------

def run_arm(trace: list[TraceRequest], args: argparse.Namespace,
            admission_factory, eviction_factory) -> dict:
    serving = ServingConfig(
        cache_capacity_tokens=args.capacity,
        prefill_tps=args.prefill_tps,
        decode_tps=args.decode_tps,
        max_concurrent=args.max_concurrent,
        decode_chunks=4,
        evict_tps=2000.0,
    )
    sim = ServingSim(serving, policy=eviction_factory(), admission=admission_factory())
    sim.submit_all(trace)
    sim.run()
    return sim.collector.summary()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = load_windows(Path(args.azure), Path(args.window_dir))
    if args.smoke:
        windows = {name: arr[:800] for name, arr in windows.items()}
        if args.mode == "sessions":
            print("[smoke] windows truncated to 800 arrivals")

    rows: list[dict] = []
    regimes = ("burst",) if args.mode == "tokens" else ("calm", "burst")
    session_trace = read_trace(args.session_trace) if args.mode == "sessions" else []

    for regime in regimes:
        arrivals = windows[regime]
        if args.mode == "sessions":
            trace = compose_session_replay(arrivals, session_trace)
        else:
            # 辅助臂：仅 Azure 自身 token 单轮回放（无会话结构）
            tokens_csv = Path(args.window_dir) / f"{regime}_tokens.csv"
            if not tokens_csv.exists():
                print(f"[skip] tokens-mode needs {tokens_csv} (extract alongside "
                      f"{regime}.txt); run in sessions mode instead")
                continue
            trace = real_tokens_trace_from_csv(
                tokens_csv, preamble_tokens=1024, max_requests=len(arrivals))

        for adm_name, adm_factory in ADMISSIONS:
            for ev_name, ev_factory in EVITIONS:
                summary = run_arm(trace, args, adm_factory, ev_factory)
                rows.append({
                    "regime": regime,
                    "mode": args.mode,
                    "admission": adm_name,
                    "eviction": ev_name,
                    "requests": summary["completed"],
                    "jct_mean": round(summary["jct_mean"], 3),
                    "jct_p95": round(summary["jct_p95"], 3),
                    "ttft_mean": round(summary["ttft_mean"], 3),
                    "queue_delay_mean": round(summary["queue_delay_mean"], 3),
                    "hit_rate": round(summary["hit_rate"], 4),
                    "eviction_gain_pct": 0.0,
                    "admission_gain_pct": 0.0,
                    "preemptions": summary["preemptions"]["count"],
                })

        # 以 fifo+fifo 为基线计算两个杠杆的相对收益
        group = [r for r in rows if r["regime"] == regime and r["mode"] == args.mode]
        baseline = next(r for r in group
                        if r["admission"] == "fifo" and r["eviction"] == "fifo")
        for row in group:
            if row is baseline:
                continue
            gain = (baseline["jct_mean"] - row["jct_mean"]) / baseline["jct_mean"] * 100
            if row["admission"] != "fifo" and row["eviction"] == "fifo":
                row["admission_gain_pct"] = round(gain, 1)
            elif row["admission"] == "fifo" and row["eviction"] != "fifo":
                row["eviction_gain_pct"] = round(gain, 1)

        print(f"=== {regime} / {args.mode} (baseline fifo+fifo "
              f"jct={baseline['jct_mean']}, hit={baseline['hit_rate']}) ===")
        for row in group:
            print(f"  {row['admission']:>4} + {row['eviction']:>4}: "
                  f"jct={row['jct_mean']:>9} p95={row['jct_p95']:>9} "
                  f"queue={row['queue_delay_mean']:>8} hit={row['hit_rate']:.3f} "
                  f"preempt={row['preemptions']}")

    if not rows:
        return 1

    with (out_dir / f"exp012_summary_{args.mode}.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / f"exp012_summary_{args.mode}.json").write_text(
        json.dumps({"seed": args.seed, "mode": args.mode,
                    "capacity": args.capacity, "rows": rows},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 主模式输出结论对比图（calm vs burst 的杠杆收益）
    if args.mode == "sessions" and len([r for r in rows if r["regime"] == "burst"]) >= 4:
        labels = ["calm:\nevict lever", "calm:\nadmit lever", "burst:\nevict lever",
                  "burst:\nadmit lever", "burst:\nboth levers"]

        def _gain(regime: str, adm: str, ev: str) -> float:
            row = next(r for r in rows if r["regime"] == regime
                       and r["admission"] == adm and r["eviction"] == ev)
            base = next(r for r in rows if r["regime"] == regime
                        and r["admission"] == "fifo" and r["eviction"] == "fifo")
            return (base["jct_mean"] - row["jct_mean"]) / base["jct_mean"] * 100

        series = {"jct gain %": [
            _gain("calm", "fifo", "lru"), _gain("calm", "sjf", "fifo"),
            _gain("burst", "fifo", "lru"), _gain("burst", "sjf", "fifo"),
            _gain("burst", "sjf", "lru"),
        ]}
        plot_sweep("regime x lever (real Azure arrivals)", list(range(len(labels))),
                   series, out_dir / "exp012_real_arrival_levers.png",
                   title="real arrivals: lever gains by regime (mean JCT reduction %)",
                   ylabel="JCT reduction (%)")
        print(f"outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
