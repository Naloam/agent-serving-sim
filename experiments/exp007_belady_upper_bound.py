"""exp007：Belady 离线最优上限对比（M3 收口）。

问题：LRU（我们的强基线）离理论最优还有多远？TTL 的拐点工作点
收窄了这个差距吗？

设计：异构负载（与 exp002 相同：coding 快回转 5s + search 长思考 30s），
在两个容量档（80K 常规 / 40K 受压）下对比 LRU、TTL 扫描与
BeladyPolicy（Belady/MIN：驱逐未来最晚复用的叶子，需要完整 trace，
仅作上限参照）。

用法::

    python experiments/exp007_belady_upper_bound.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ass.cache.policies import BeladyPolicy, LRUPolicy, TTLPolicy
from ass.scheduler.serving import ServingConfig, ServingSim
from ass.viz.plots import plot_sweep
from ass.workload.synthetic import generate_trace
from exp002_ttl_sweep_heterogeneous import build_config as build_exp002_config

TTLS = (15.0, 30.0, 60.0)
CAPACITIES = (80_000, 40_000)
TABLE_COLUMNS = ("capacity", "policy", "hit_total", "hit_coding", "hit_search",
                 "jct_total", "eviction_count")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LRU/TTL vs Belady upper bound")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="experiments/results")
    return parser.parse_args(argv)


def class_stats(summary: dict, agent_type: str) -> float:
    bucket = summary["by_agent_type"].get(agent_type)
    return bucket["hit_rate"] if bucket else 0.0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_ns = argparse.Namespace(coding_sessions=140, search_sessions=70, turns=10, rate=0.12)
    trace = generate_trace(build_exp002_config(trace_ns), args.seed)
    print(f"trace: {len(trace)} requests, capacities={CAPACITIES}, ttls={TTLS}")

    rows: list[dict] = []
    belady_hit: dict[int, float] = {}
    belady_jct: dict[int, float] = {}
    for capacity in CAPACITIES:
        config = ServingConfig(cache_capacity_tokens=capacity)
        policies: list[tuple[str, object]] = [("lru", LRUPolicy()), ("belady", BeladyPolicy(trace))]
        policies += [(f"ttl-{ttl:g}", TTLPolicy(ttl=ttl)) for ttl in TTLS]
        for label, policy in policies:
            sim = ServingSim(config, policy=policy)
            sim.submit_all(trace)
            sim.run()
            summary = sim.collector.summary()
            if label == "belady":
                belady_hit[capacity] = summary["hit_rate"]
                belady_jct[capacity] = summary["jct_mean"]
            rows.append({
                "capacity": capacity,
                "policy": label,
                "hit_total": round(summary["hit_rate"], 4),
                "hit_coding": round(class_stats(summary, "coding"), 4),
                "hit_search": round(class_stats(summary, "search"), 4),
                "jct_total": round(summary["jct_mean"], 3),
                "eviction_count": summary["evictions"]["count"],
            })

    widths = [max(len(str(row[c])) for row in rows + [dict(zip(TABLE_COLUMNS, TABLE_COLUMNS))])
              for c in TABLE_COLUMNS]
    header = "  ".join(c.ljust(w) for c, w in zip(TABLE_COLUMNS, widths))
    print(header)
    for row in rows:
        print("  ".join(str(row[c]).ljust(w) for c, w in zip(TABLE_COLUMNS, widths)))

    for capacity in CAPACITIES:
        cap_rows = [r for r in rows if r["capacity"] == capacity]
        lru = next(r for r in cap_rows if r["policy"] == "lru")
        best_ttl = min((r for r in cap_rows if r["policy"].startswith("ttl")),
                       key=lambda r: r["jct_total"])
        gap = (belady_hit[capacity] - lru["hit_total"]) / belady_hit[capacity] * 100
        print(
            f"[cap {capacity}] belady hit={belady_hit[capacity]:.4f} vs "
            f"lru {lru['hit_total']:.4f} (LRU 距上限 {gap:.1f}%), "
            f"belady jct={belady_jct[capacity]:.3f} vs best-ttl {best_ttl['jct_total']:.3f}"
        )

    with (out_dir / "exp007_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "exp007_summary.json").write_text(
        json.dumps({"seed": args.seed, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_sweep(
        "ttl (s)",
        list(TTLS),
        {
            f"hit @ {cap}": [
                next(r["hit_total"] for r in rows if r["capacity"] == cap and r["policy"] == f"ttl-{ttl:g}")
                for ttl in TTLS
            ]
            for cap in CAPACITIES
        },
        out_dir / "exp007_belady_upper_bound.png",
        title="TTL sweep vs Belady upper bound (see table for belady/lru values)",
        ylabel="hit rate",
    )
    print(f"outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
