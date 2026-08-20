"""exp005：真实 trace 重放实验（M3 收尾）。

把 M2 采集、清洗后的真实 trace（traces/real/<type>.jsonl）合并重放进
ServingSim，计时参数用 analyze_real_trace.py 的标定结果
（traces/real/calibration.json），对比 LRU / TTL 扫描 / 带权 LRU / 配额。

TTL 档位按真实 think_time 中位数的分档生成（1/4×、1/2×、1×、2×、4×）。

用法::

    python experiments/exp005_real_trace_replay.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ass.cache.policies import LRUPolicy, QuotaPolicy, TTLPolicy, WeightedLRUPolicy
from ass.scheduler.serving import ServingConfig, ServingSim
from ass.viz.plots import plot_sweep
from ass.workload.schema import read_trace

TABLE_COLUMNS = (
    "policy",
    "hit_total",
    "hit_coding",
    "hit_search",
    "jct_total",
    "jct_coding",
    "jct_search",
    "jct_p95",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Replay real traces under M3 policies")
    parser.add_argument("--trace-dir", type=str, default="traces/real")
    parser.add_argument("--calibration", type=str, default="traces/real/calibration.json")
    parser.add_argument("--capacity", type=int, default=200_000)
    parser.add_argument("--max-concurrent", type=int, default=4)
    parser.add_argument("--decode-chunks", type=int, default=1,
                        help=">1 时启用 decode 分块增长与抢占语义（FR-13）")
    parser.add_argument("--evict-tps", type=float, default=None,
                        help="驱逐吞吐（token/s）；缺省为免费驱逐")
    parser.add_argument("--out-dir", type=str, default="experiments/results")
    return parser.parse_args(argv)


def load_trace(trace_dir: Path) -> list:
    requests = []
    for path in sorted(trace_dir.glob("*.jsonl")):
        requests.extend(read_trace(path))
    requests.sort(key=lambda request: request.arrival_time)
    return requests


def class_stats(summary: dict, agent_type: str) -> tuple[float, float]:
    bucket = summary["by_agent_type"].get(agent_type)
    if not bucket:
        return 0.0, 0.0
    return bucket["hit_rate"], bucket["jct_mean"]


def table_row(label: str, summary: dict) -> dict:
    hit_coding, jct_coding = class_stats(summary, "coding")
    hit_search, jct_search = class_stats(summary, "search")
    return {
        "policy": label,
        "hit_total": round(summary["hit_rate"], 4),
        "hit_coding": round(hit_coding, 4),
        "hit_search": round(hit_search, 4),
        "jct_total": round(summary["jct_mean"], 3),
        "jct_coding": round(jct_coding, 3),
        "jct_search": round(jct_search, 3),
        "jct_p95": round(summary["jct_p95"], 3),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    trace = load_trace(Path(args.trace_dir))
    if not trace:
        print(f"no traces found under {args.trace_dir}", file=sys.stderr)
        return 2

    calibration = {}
    calibration_path = Path(args.calibration)
    if calibration_path.exists():
        calibration = json.loads(calibration_path.read_text(encoding="utf-8"))["calibration"]
    serving_config = ServingConfig(
        cache_capacity_tokens=args.capacity,
        prefill_tps=calibration.get("prefill_tps") or 5000.0,
        decode_tps=calibration.get("decode_tps") or 200.0,
        max_concurrent=args.max_concurrent,
        decode_chunks=args.decode_chunks,
        evict_tps=args.evict_tps,
    )

    thinks = [request.think_time for request in trace if request.turn_id > 1]
    thinks.sort()
    median_think = thinks[len(thinks) // 2] if thinks else 10.0
    ttl_grid = sorted({round(median_think * factor, 2) for factor in (0.25, 0.5, 1.0, 2.0, 4.0)})
    print(
        f"trace: {len(trace)} requests, capacity={args.capacity}, "
        f"prefill_tps={serving_config.prefill_tps}, decode_tps={serving_config.decode_tps}, "
        f"median think={median_think:.1f}s, ttls={ttl_grid}"
    )

    def _run(policy):
        sim = ServingSim(serving_config, policy=policy)
        sim.submit_all(trace)
        sim.run()
        return sim.collector

    collectors = {"lru": _run(LRUPolicy())}
    for ttl in ttl_grid:
        collectors[f"ttl-{ttl:g}"] = _run(TTLPolicy(ttl=ttl))
    collectors["wlru-w4"] = _run(
        WeightedLRUPolicy(agent_weights={"coding": 4.0, "search": 1.0})
    )
    collectors["quota-0.5"] = _run(
        QuotaPolicy(
            quotas={"coding": args.capacity // 2, "search": args.capacity // 2}
        )
    )

    rows = [table_row(label, collector.summary()) for label, collector in collectors.items()]
    widths = [
        max(len(str(row[column])) for row in rows + [dict(zip(TABLE_COLUMNS, TABLE_COLUMNS))])
        for column in TABLE_COLUMNS
    ]
    header = "  ".join(column.ljust(width) for column, width in zip(TABLE_COLUMNS, widths))
    print(header)
    for row in rows:
        print("  ".join(str(row[column]).ljust(width) for column, width in zip(TABLE_COLUMNS, widths)))

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "exp005_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "exp005_summary.json").write_text(
        json.dumps(
            {
                "capacity": args.capacity,
                "calibration": calibration,
                "median_think": median_think,
                "results": {row["policy"]: row for row in rows},
            },
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    ttl_labels = [f"ttl-{ttl:g}" for ttl in ttl_grid]
    lru_row = rows[0]
    plot_sweep(
        "ttl (s)",
        ttl_grid,
        {
            "hit_total": [collectors[label].summary()["hit_rate"] for label in ttl_labels],
            "hit_coding": [class_stats(collectors[label].summary(), "coding")[0] for label in ttl_labels],
            "hit_search": [class_stats(collectors[label].summary(), "search")[0] for label in ttl_labels],
            "lru total": [lru_row["hit_total"]] * len(ttl_labels),
        },
        out_dir / "exp005_real_ttl_sweep_hit_rate.png",
        title="real-trace replay: hit rate vs TTL",
        ylabel="hit rate",
    )
    plot_sweep(
        "ttl (s)",
        ttl_grid,
        {
            "jct_total": [collectors[label].summary()["jct_mean"] for label in ttl_labels],
            "lru total": [lru_row["jct_total"]] * len(ttl_labels),
        },
        out_dir / "exp005_real_ttl_sweep_jct.png",
        title="real-trace replay: mean JCT vs TTL",
        ylabel="JCT (s)",
    )
    print(f"outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
