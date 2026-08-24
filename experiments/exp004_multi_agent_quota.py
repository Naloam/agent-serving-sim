"""exp004：多 agent cache 配额实验。

负载与 exp003 相同（价值倒置：coding 慢回转高价值、search 快回转）。
扫描 QuotaPolicy 的容量分配比例 f = coding 配额 / 总容量 ∈ {0.2 .. 0.8}，
与 LRU、WeightedLRUPolicy 参考点对比：

- 配额是**软隔离**：search 空闲时可用 coding 的剩余配额，超额时先被逐；
- 考察类间命中率/JCT 的隔离性与加权 JCT 的形态。

用法::

    python experiments/exp004_multi_agent_quota.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ass.cache.policies import LRUPolicy, QuotaPolicy, WeightedLRUPolicy
from ass.scheduler.serving import ServingConfig, ServingSim
from ass.viz.plots import plot_sweep
from ass.workload.synthetic import generate_trace
from exp003_priority_eviction import (
    CLASS_WEIGHTS,
    _p95,
    build_config,
    class_stats,
    weighted_jct,
)

FRACTIONS = (0.2, 0.35, 0.5, 0.65, 0.8)
TABLE_COLUMNS = (
    "policy",
    "hit_total",
    "hit_coding",
    "hit_search",
    "jct_coding",
    "jct_search",
    "jct_coding_p95",
    "jct_search_p95",
    "weighted_jct",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-agent cache quota sweep")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity", type=int, default=80_000)
    parser.add_argument("--out-dir", type=str, default="experiments/results")
    return parser.parse_args(argv)


def table_row(label: str, collector) -> dict:
    summary = collector.summary()
    hit_coding, jct_coding = class_stats(summary, "coding")
    hit_search, jct_search = class_stats(summary, "search")
    return {
        "policy": label,
        "hit_total": round(summary["hit_rate"], 4),
        "hit_coding": round(hit_coding, 4),
        "hit_search": round(hit_search, 4),
        "jct_coding": round(jct_coding, 3),
        "jct_search": round(jct_search, 3),
        "jct_coding_p95": round(_p95(collector.jct_values(agent_type="coding")), 3),
        "jct_search_p95": round(_p95(collector.jct_values(agent_type="search")), 3),
        "weighted_jct": round(weighted_jct(collector), 3),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # 与 exp003 相同的参数形状（build_config 只吃 exp003 的命名空间字段）
    trace_ns = argparse.Namespace(coding_sessions=70, search_sessions=140, turns=10, rate=0.12)
    trace = generate_trace(build_config(trace_ns), args.seed)
    serving_config = ServingConfig(cache_capacity_tokens=args.capacity)
    print(f"trace: {len(trace)} requests, capacity={args.capacity} tokens")

    def _run(policy):
        sim = ServingSim(serving_config, policy=policy)
        sim.submit_all(trace)
        sim.run()
        return sim.collector

    runs: dict[str, object] = {}
    runs["lru"] = _run(LRUPolicy())
    for fraction in FRACTIONS:
        runs[f"quota-{fraction:.2f}"] = _run(
            QuotaPolicy(
                quotas={
                    "coding": int(args.capacity * fraction),
                    "search": int(args.capacity * (1.0 - fraction)),
                }
            )
        )
    runs["wlru-w4"] = _run(WeightedLRUPolicy(agent_weights={"coding": 4.0, "search": 1.0}))
    runs["wlru-w16"] = _run(WeightedLRUPolicy(agent_weights={"coding": 16.0, "search": 1.0}))

    rows = [table_row(label, collector) for label, collector in runs.items()]
    widths = [
        max(len(str(row[column])) for row in rows + [dict(zip(TABLE_COLUMNS, TABLE_COLUMNS))])
        for column in TABLE_COLUMNS
    ]
    header = "  ".join(column.ljust(width) for column, width in zip(TABLE_COLUMNS, widths))
    print(header)
    for row in rows:
        print("  ".join(str(row[column]).ljust(width) for column, width in zip(TABLE_COLUMNS, widths)))

    with (out_dir / "exp004_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "exp004_summary.json").write_text(
        json.dumps(
            {"seed": args.seed, "capacity": args.capacity, "fractions": list(FRACTIONS),
             "class_weights": CLASS_WEIGHTS, "results": {row["policy"]: row for row in rows}},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    quota_rows = rows[1:1 + len(FRACTIONS)]
    lru_row = rows[0]
    plot_sweep(
        "coding quota fraction",
        list(FRACTIONS),
        {
            "hit_coding": [row["hit_coding"] for row in quota_rows],
            "hit_search": [row["hit_search"] for row in quota_rows],
            "hit_total": [row["hit_total"] for row in quota_rows],
            "lru coding": [lru_row["hit_coding"]] * len(FRACTIONS),
            "lru search": [lru_row["hit_search"]] * len(FRACTIONS),
        },
        out_dir / "exp004_quota_sweep_hit_rate.png",
        title="class hit rate vs quota split",
        ylabel="hit rate",
    )
    plot_sweep(
        "coding quota fraction",
        list(FRACTIONS),
        {
            "jct_coding": [row["jct_coding"] for row in quota_rows],
            "jct_search": [row["jct_search"] for row in quota_rows],
            "weighted_jct": [row["weighted_jct"] for row in quota_rows],
            "lru weighted": [lru_row["weighted_jct"]] * len(FRACTIONS),
        },
        out_dir / "exp004_quota_sweep_jct.png",
        title="JCT vs quota split",
        ylabel="JCT (s)",
    )
    print(f"outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
