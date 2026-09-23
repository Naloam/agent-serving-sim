"""exp013：续链优先准入 vs FIFO/SJF——修复 exp012 发现的"前缀盲"问题（M7 / FR-18）。

exp012 的负结果：真实 Azure 到达的突发窗口上，朴素 SJF 恶化 72%
（命中率 0.876→0.337）——它偏好短 prompt，等于偏好各会话首轮，
长会话的后续轮在队列中挨饿至前缀冷死。FIFO 靠到达顺序"意外"保住
轮次局部性，但没有任何主动设计。

``SessionChainAdmission`` 把这个局部性变成显式策略：已准入过的会话的
后续轮（前缀已建、还热）排在新会话首轮之前，类内按到达序——会话级
最短剩余工作 + 前缀驻留保持。

臂设计：准入 {fifo, sjf, session-chain}，驱逐固定 LRU（exp012 已证
该负载上驱逐杠杆 ≈0），regime = calm / burst（sessions 模式）；
另跑 tokens 模式 burst（无会话结构对照：session-chain 应退化为 FIFO，
不劣于基线）。

验证点：

- V1 burst（会话结构）：session-chain 的 JCT/命中率不劣于 FIFO 且
  显著优于 SJF；
- V2 无会话结构（tokens 对照）：session-chain ≈ FIFO（优雅退化）；
- V3 calm：三臂相近（无饱和则无杠杆）。

用法::

    python experiments/exp013_prefix_aware_admission.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ass.cache.policies import LRUPolicy
from ass.scheduler.admission import FIFOAdmission, SessionChainAdmission, ShortestJobAdmission
from ass.scheduler.serving import ServingConfig, ServingSim
from ass.viz.plots import plot_sweep
from ass.workload.loaders import real_tokens_trace_from_csv
from ass.workload.schema import read_trace
from exp012_real_arrival_replay import compose_session_replay, load_windows

ADMISSIONS = (
    ("fifo", lambda: FIFOAdmission()),
    ("sjf", lambda: ShortestJobAdmission()),
    ("session-chain", lambda: SessionChainAdmission()),
)
TABLE_COLUMNS = ("regime", "mode", "admission", "requests", "jct_mean",
                 "jct_p95", "queue_delay_mean", "hit_rate", "preemptions")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prefix-aware admission under real arrivals")
    parser.add_argument("--azure", type=str,
                        default="traces/real/external/azure_code_1week.csv")
    parser.add_argument("--session-trace", type=str, default="traces/real/coding.jsonl")
    parser.add_argument("--extra-session-trace", type=str, default="traces/real/search.jsonl",
                        help="混合类臂的第二份会话 trace（存在则并入 burst 混合场景）")
    parser.add_argument("--window-dir", type=str, default="traces/real/azure_windows")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity", type=int, default=80_000)
    parser.add_argument("--prefill-tps", type=float, default=40_000.0)
    parser.add_argument("--decode-tps", type=float, default=800.0)
    parser.add_argument("--max-concurrent", type=int, default=8)
    parser.add_argument("--out-dir", type=str, default="experiments/results")
    return parser.parse_args(argv)


def interleave(primary: list, secondary: list) -> list:
    """按 2:1 交错合并两份会话 trace（各自内部保序，模拟并发混合负载）。"""
    merged = []
    pi = si = 0
    while pi < len(primary) or si < len(secondary):
        for _ in range(2):
            if pi < len(primary):
                merged.append(primary[pi])
                pi += 1
        if si < len(secondary):
            merged.append(secondary[si])
            si += 1
    return merged


def run_arm(trace, args, admission_factory) -> dict:
    serving = ServingConfig(
        cache_capacity_tokens=args.capacity,
        prefill_tps=args.prefill_tps,
        decode_tps=args.decode_tps,
        max_concurrent=args.max_concurrent,
        decode_chunks=4,
        evict_tps=2000.0,
    )
    sim = ServingSim(serving, policy=LRUPolicy(), admission=admission_factory())
    sim.submit_all(trace)
    sim.run()
    return sim.collector.summary()


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    windows = load_windows(Path(args.azure), Path(args.window_dir))
    session_trace = read_trace(args.session_trace)
    tokens_csv = Path(args.window_dir) / "burst_tokens.csv"

    scenarios: list[tuple[str, str, list]] = []
    for regime in ("calm", "burst"):
        scenarios.append((regime, "sessions",
                          compose_session_replay(windows[regime], session_trace)))
    if tokens_csv.exists():
        scenarios.append(("burst", "tokens",
                          real_tokens_trace_from_csv(tokens_csv, preamble_tokens=1024)))
    extra_path = Path(args.extra_session_trace)
    if extra_path.exists():
        # 混合类臂：异质性（coding ~2.2x search 工作量）× 会话结构共存，
        # 探测排序空间是否存在（理论：守恒律下仍无均值赢面，验证之）
        mixed = interleave(session_trace, read_trace(extra_path))
        scenarios.append(("burst", "mixed",
                          compose_session_replay(windows["burst"], mixed)))

    rows: list[dict] = []
    for regime, mode, trace in scenarios:
        for adm_name, adm_factory in ADMISSIONS:
            summary = run_arm(trace, args, adm_factory)
            rows.append({
                "regime": regime,
                "mode": mode,
                "admission": adm_name,
                "requests": summary["completed"],
                "jct_mean": round(summary["jct_mean"], 3),
                "jct_p95": round(summary["jct_p95"], 3),
                "queue_delay_mean": round(summary["queue_delay_mean"], 3),
                "hit_rate": round(summary["hit_rate"], 4),
                "preemptions": summary["preemptions"]["count"],
            })
        group = [r for r in rows if r["regime"] == regime and r["mode"] == mode]
        baseline = next(r for r in group if r["admission"] == "fifo")
        print(f"=== {regime} / {mode} (fifo jct={baseline['jct_mean']}, "
              f"hit={baseline['hit_rate']}) ===")
        for row in group:
            gain = (baseline["jct_mean"] - row["jct_mean"]) / baseline["jct_mean"] * 100
            print(f"  {row['admission']:>13}: jct={row['jct_mean']:>9} ({gain:+.1f}%) "
                  f"p95={row['jct_p95']:>9} hit={row['hit_rate']:.3f} "
                  f"preempt={row['preemptions']}")

    with (out_dir / "exp013_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "exp013_summary.json").write_text(
        json.dumps({"seed": args.seed, "capacity": args.capacity, "rows": rows},
                   indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # 突发段对比图：按实际参与的场景动态构建标签（场景或臂缺失时跳过）
    labels: list[str] = []
    values: list[float] = []
    for mode in ("sessions", "tokens", "mixed"):
        for adm_name, _ in ADMISSIONS:
            row = next((r for r in rows if r["regime"] == "burst"
                        and r["mode"] == mode and r["admission"] == adm_name), None)
            if row is not None:
                labels.append(f"{mode}:\n{adm_name}")
                values.append(row["jct_mean"])
    if values:
        plot_sweep("admission policy (real Azure burst)", list(range(len(labels))),
                   {"jct_mean": values},
                   out_dir / "exp013_prefix_aware_admission.png",
                   title="real burst arrivals: admission policies x workload structure (mean JCT)",
                   ylabel="JCT (s)")
    print(f"outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
