"""exp001：合成 trace 上 LRU vs TTL 的命中率与 JCT 对比（对应 PRD FR-10）。

同一份合成 trace（固定 seed 生成）分别以 LRU 与多档 TTL 策略仿真，
输出：汇总表（stdout + CSV + JSON）、JCT CDF、TTL 扫描曲线、显存时间线。

用法::

    python experiments/exp001_lru_vs_ttl.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ass.cache.policies import LRUPolicy, TTLPolicy
from ass.metrics.collector import MetricsCollector
from ass.scheduler.serving import ServingConfig, ServingSim
from ass.viz.plots import plot_cdf, plot_sweep, plot_timeline
from ass.workload.synthetic import SyntheticConfig, generate_trace

DEFAULT_TTLS = "5,10,20,40,80"

TABLE_COLUMNS = (
    "policy",
    "hit_rate",
    "jct_mean",
    "jct_p50",
    "jct_p95",
    "ttft_mean",
    "queue_delay_mean",
    "eviction_count",
    "ttl_expired_tokens",
    "cache_peak_tokens",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LRU vs TTL on synthetic agent trace")
    parser.add_argument("--seed", type=int, default=42, help="合成 trace 随机种子")
    parser.add_argument("--sessions", type=int, default=300, help="会话数")
    parser.add_argument("--turns", type=int, default=8, help="每会话轮数")
    parser.add_argument("--rate", type=float, default=0.5, help="会话到达率（次/秒）")
    parser.add_argument("--capacity", type=int, default=100_000, help="缓存容量（token 数）")
    parser.add_argument("--ttls", type=str, default=DEFAULT_TTLS, help="TTL 扫描档位（逗号分隔，秒）")
    parser.add_argument("--out-dir", type=str, default="experiments/results", help="产出目录")
    return parser.parse_args(argv)


def run_one(trace, serving_config: ServingConfig, policy) -> MetricsCollector:
    sim = ServingSim(serving_config, policy=policy)
    sim.submit_all(trace)
    sim.run()
    return sim.collector


def table_row(label: str, summary: dict) -> dict:
    return {
        "policy": label,
        "hit_rate": round(summary["hit_rate"], 4),
        "jct_mean": round(summary["jct_mean"], 3),
        "jct_p50": round(summary["jct_p50"], 3),
        "jct_p95": round(summary["jct_p95"], 3),
        "ttft_mean": round(summary["ttft_mean"], 3),
        "queue_delay_mean": round(summary["queue_delay_mean"], 3),
        "eviction_count": summary["evictions"]["count"],
        "ttl_expired_tokens": summary["ttl_expired_tokens"],
        "cache_peak_tokens": summary["cache_peak_tokens"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ttls = sorted({float(value) for value in args.ttls.split(",")})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    synth_config = SyntheticConfig(
        num_sessions=args.sessions,
        turns_per_session=args.turns,
        session_arrival_rate=args.rate,
    )
    trace = generate_trace(synth_config, args.seed)
    serving_config = ServingConfig(cache_capacity_tokens=args.capacity)
    print(
        f"trace: {len(trace)} requests ({args.sessions} sessions x {args.turns} turns, "
        f"rate={args.rate}/s), cache capacity={args.capacity} tokens"
    )

    runs: dict[str, MetricsCollector] = {"lru": run_one(trace, serving_config, LRUPolicy())}
    for ttl in ttls:
        runs[f"ttl-{ttl:g}"] = run_one(trace, serving_config, TTLPolicy(ttl=ttl))

    rows = [table_row(label, collector.summary()) for label, collector in runs.items()]
    widths = [max(len(str(row[column])) for row in rows + [dict(zip(TABLE_COLUMNS, TABLE_COLUMNS))]) for column in TABLE_COLUMNS]
    header = "  ".join(column.ljust(width) for column, width in zip(TABLE_COLUMNS, widths))
    print(header)
    for row in rows:
        print("  ".join(str(row[column]).ljust(width) for column, width in zip(TABLE_COLUMNS, widths)))

    csv_path = out_dir / "exp001_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    json_path = out_dir / "exp001_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "synthetic_config": {
                    "sessions": args.sessions,
                    "turns": args.turns,
                    "rate": args.rate,
                },
                "cache_capacity_tokens": args.capacity,
                "ttls": ttls,
                "results": {row["policy"]: row for row in rows},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    # 图 1：JCT CDF（LRU + 首/中/末三档 TTL）
    ttl_labels = [f"ttl-{ttl:g}" for ttl in ttls]
    picked = {ttl_labels[i] for i in (0, len(ttl_labels) // 2, len(ttl_labels) - 1)}
    cdf_series = {"lru": runs["lru"].jct_values()}
    cdf_series.update({label: runs[label].jct_values() for label in ttl_labels if label in picked})
    plot_cdf(cdf_series, out_dir / "exp001_jct_cdf.png", title="JCT CDF: LRU vs TTL", xlabel="JCT (s)")

    # 图 2/3：TTL 扫描（带 LRU 水平参考线）
    lru_summary = runs["lru"].summary()
    plot_sweep(
        "ttl (s)",
        ttls,
        {
            "ttl": [runs[label].summary()["hit_rate"] for label in ttl_labels],
            "lru (reference)": [lru_summary["hit_rate"]] * len(ttls),
        },
        out_dir / "exp001_ttl_sweep_hit_rate.png",
        title="prefix hit rate vs TTL",
        ylabel="hit rate",
    )
    plot_sweep(
        "ttl (s)",
        ttls,
        {
            "ttl": [runs[label].summary()["jct_mean"] for label in ttl_labels],
            "lru (reference)": [lru_summary["jct_mean"]] * len(ttls),
        },
        out_dir / "exp001_ttl_sweep_jct.png",
        title="mean JCT vs TTL",
        ylabel="JCT (s)",
    )

    # 图 4/5：显存时间线（LRU 与中位 TTL 各一张）
    mid_label = ttl_labels[len(ttl_labels) // 2]
    for label in ("lru", mid_label):
        timeline = runs[label].cache_timeline
        if timeline:
            times, used = zip(*timeline)
            plot_timeline(
                times,
                used,
                out_dir / f"exp001_cache_timeline_{label}.png",
                title=f"cache usage: {label}",
                ylabel="tokens",
            )

    print(f"outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
