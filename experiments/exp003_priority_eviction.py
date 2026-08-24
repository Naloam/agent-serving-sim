"""exp003：优先级驱逐实验。

负载设计与 exp002 镜像：高价值类 coding 的回转**慢**（think 中位数
~30s、prefill 重：长前导 + 长历史，命中一次省得多），低价值类 search
回转**快**（~5s、decode 为主）。LRU 的 recency 排序天然偏袒 search，
与价值方向相反。

实验形式：扫描 PriorityPolicy 的 coding 权重 w ∈ {1, 1.5, 2, 3, 5}
（search 固定 1），以 LRU 为基线，考察：

- 类间权衡（Pareto 前沿）：w 升高 → coding JCT 降、search JCT 升；
- 按类权重（2:1）加权的请求级 JCT 是否存在优于 LRU 的 w*。

用法::

    python experiments/exp003_priority_eviction.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ass.cache.policies import LRUPolicy, PriorityPolicy, WeightedLRUPolicy
from ass.metrics.collector import MetricsCollector
from ass.scheduler.serving import ServingConfig, ServingSim
from ass.viz.plots import plot_cdf, plot_sweep
from ass.workload.synthetic import AgentProfile, SyntheticConfig, generate_trace

CLASS_WEIGHTS = {"coding": 2.0, "search": 1.0}
WEIGHT_SWEEP = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0)

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
    parser = argparse.ArgumentParser(description="Priority eviction on value-inverted load")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coding-sessions", type=int, default=70)
    parser.add_argument("--search-sessions", type=int, default=140)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--rate", type=float, default=0.12)
    parser.add_argument("--capacity", type=int, default=80_000)
    parser.add_argument("--out-dir", type=str, default="experiments/results")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> SyntheticConfig:
    return SyntheticConfig(
        num_sessions=args.coding_sessions + args.search_sessions,
        turns_per_session=args.turns,
        session_arrival_rate=args.rate,
        agent_mix={"coding": args.coding_sessions, "search": args.search_sessions},
        agent_profiles={
            "coding": AgentProfile(
                think_time_mu=3.40,  # 中位数 ~30s：高价值但回转慢
                think_time_sigma=0.7,
                system_tokens_mean=1500.0,
                tools_tokens_mean=800.0,
                new_tokens_mean=450.0,
                new_tokens_std=150.0,
                output_tokens_mean=220.0,
                output_tokens_std=80.0,
            ),
            "search": AgentProfile(
                think_time_mu=1.61,  # 中位数 ~5s：低价值但回转快
                think_time_sigma=0.5,
                system_tokens_mean=400.0,
                tools_tokens_mean=200.0,
                new_tokens_mean=700.0,
                new_tokens_std=250.0,
                output_tokens_mean=180.0,
                output_tokens_std=70.0,
            ),
        },
    )


def weighted_jct(collector: MetricsCollector) -> float:
    total = 0.0
    weight_sum = 0.0
    for record in collector.records:
        weight = CLASS_WEIGHTS.get(record.agent_type, 1.0)
        total += weight * record.jct
        weight_sum += weight
    return total / weight_sum if weight_sum else 0.0


def class_stats(summary: dict, agent_type: str) -> tuple[float, float]:
    bucket = summary["by_agent_type"].get(agent_type)
    if not bucket:
        return 0.0, 0.0
    return bucket["hit_rate"], bucket["jct_mean"]


def _p95(values: list[float]) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(int(round(0.95 * (len(ordered) - 1))), len(ordered) - 1)]


def table_row(label: str, collector: MetricsCollector) -> dict:
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
    trace = generate_trace(build_config(args), args.seed)
    serving_config = ServingConfig(cache_capacity_tokens=args.capacity)
    print(f"trace: {len(trace)} requests, capacity={args.capacity} tokens")

    runs: dict[str, MetricsCollector] = {}
    runs["lru"] = _run(trace, serving_config, LRUPolicy())
    for weight in WEIGHT_SWEEP:
        runs[f"wlru-w{weight:g}"] = _run(
            trace,
            serving_config,
            WeightedLRUPolicy(agent_weights={"coding": weight, "search": 1.0}),
        )
    runs["strict-prio"] = _run(
        trace,
        serving_config,
        PriorityPolicy(agent_weights={"coding": 2.0, "search": 1.0}),
    )

    rows = [table_row(label, collector) for label, collector in runs.items()]
    widths = [
        max(len(str(row[column])) for row in rows + [dict(zip(TABLE_COLUMNS, TABLE_COLUMNS))])
        for column in TABLE_COLUMNS
    ]
    header = "  ".join(column.ljust(width) for column, width in zip(TABLE_COLUMNS, widths))
    print(header)
    for row in rows:
        print("  ".join(str(row[column]).ljust(width) for column, width in zip(TABLE_COLUMNS, widths)))
    best = min(rows[1:-1], key=lambda row: row["weighted_jct"])
    lru_row = rows[0]
    print(
        f"best weighted-lru weight: {best['policy']} "
        f"(weighted_jct {best['weighted_jct']} vs lru {lru_row['weighted_jct']})"
    )

    with (out_dir / "exp003_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "exp003_summary.json").write_text(
        json.dumps(
            {"seed": args.seed, "capacity": args.capacity, "class_weights": CLASS_WEIGHTS,
             "results": {row["policy"]: row for row in rows}},
            indent=2, ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    plot_cdf(
        {
            "lru": runs["lru"].jct_values(agent_type="coding"),
            "wlru-w4": runs["wlru-w4"].jct_values(agent_type="coding"),
            "wlru-w16": runs["wlru-w16"].jct_values(agent_type="coding"),
            "strict-prio": runs["strict-prio"].jct_values(agent_type="coding"),
        },
        out_dir / "exp003_coding_jct_cdf.png",
        title="coding-agent JCT CDF (value weight 2)",
        xlabel="JCT (s)",
    )
    plot_cdf(
        {
            "lru": runs["lru"].jct_values(agent_type="search"),
            "wlru-w4": runs["wlru-w4"].jct_values(agent_type="search"),
            "wlru-w16": runs["wlru-w16"].jct_values(agent_type="search"),
            "strict-prio": runs["strict-prio"].jct_values(agent_type="search"),
        },
        out_dir / "exp003_search_jct_cdf.png",
        title="search-agent JCT CDF (value weight 1)",
        xlabel="JCT (s)",
    )
    plot_sweep(
        "coding weight (search = 1)",
        list(WEIGHT_SWEEP),
        {
            "jct_coding": [row["jct_coding"] for row in rows[1:1 + len(WEIGHT_SWEEP)]],
            "jct_search": [row["jct_search"] for row in rows[1:1 + len(WEIGHT_SWEEP)]],
            "weighted_jct": [row["weighted_jct"] for row in rows[1:1 + len(WEIGHT_SWEEP)]],
            "lru weighted": [lru_row["weighted_jct"]] * len(WEIGHT_SWEEP),
        },
        out_dir / "exp003_weight_sweep.png",
        title="weighted-LRU weight sweep vs LRU baseline",
        ylabel="JCT (s)",
    )
    print(f"outputs written to {out_dir}")
    return 0


def _run(trace, serving_config, policy) -> MetricsCollector:
    sim = ServingSim(serving_config, policy=policy)
    sim.submit_all(trace)
    sim.run()
    return sim.collector


if __name__ == "__main__":
    sys.exit(main())
