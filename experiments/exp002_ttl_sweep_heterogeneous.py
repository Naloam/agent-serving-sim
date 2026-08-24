"""exp002：异构 agent 负载下的 TTL 参数扫描。

负载：coding（快回转，think 中位数 ~5s，小 new 段）+ search（长思考，
think 中位数 ~30s，大 new 段模拟 SERP 摘要）混部，容量受压。

**预期与解读（已在 seed=42 验证）**：本模拟器中 TTL 的命中率/JCT 全程
不超过 LRU 且随 TTL 单调趋近——这不是 bug，而是结构性结论：过期集
（距上次访问超过 ttl）恰好是按 last_access 排序的尾部子集，TTL 主动
清除 + LRU 兜底的 victim 集合不可能比 LRU 更优。TTL 的真实价值体现在
**工作点拐点**：TTL 取在快回转类周期之上、慢回转类周期之下时
（本负载 15~30s），快回转类命中率保持在 LRU 的 ~99%，同时把慢回转类
的死缓存持续释放（显存占用峰值更低、驱逐压力更小）。TTL 优于 LRU
需要模拟器未建模的二阶效应（驱逐/锁开销、运行中请求抢占、跨实例
路由），详见 blog 系列第（三）篇的讨论。

用法::

    python experiments/exp002_ttl_sweep_heterogeneous.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ass.cache.policies import LRUPolicy, TTLPolicy
from ass.scheduler.serving import ServingConfig, ServingSim
from ass.viz.plots import plot_cdf, plot_sweep
from ass.workload.synthetic import AgentProfile, SyntheticConfig, generate_trace

TABLE_COLUMNS = (
    "policy",
    "hit_total",
    "hit_coding",
    "hit_search",
    "jct_total",
    "jct_coding",
    "jct_search",
    "jct_p95",
    "eviction_count",
    "ttl_expired_tokens",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTL sweep on heterogeneous agent load")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--coding-sessions", type=int, default=140)
    parser.add_argument("--search-sessions", type=int, default=70)
    parser.add_argument("--turns", type=int, default=10)
    parser.add_argument("--rate", type=float, default=0.12)
    parser.add_argument("--capacity", type=int, default=80_000)
    parser.add_argument("--ttls", type=str, default="2,4,8,15,30,60,120,300")
    parser.add_argument("--out-dir", type=str, default="experiments/results")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> SyntheticConfig:
    return SyntheticConfig(
        num_sessions=args.coding_sessions + args.search_sessions,
        turns_per_session=args.turns,
        session_arrival_rate=args.rate,
        agent_mix={
            "coding": args.coding_sessions,
            "search": args.search_sessions,
        },
        agent_profiles={
            "coding": AgentProfile(
                think_time_mu=1.61,  # 中位数 ~5s
                think_time_sigma=0.5,
                system_tokens_mean=700.0,
                tools_tokens_mean=300.0,
                new_tokens_mean=120.0,
                new_tokens_std=40.0,
                output_tokens_mean=280.0,
                output_tokens_std=90.0,
            ),
            "search": AgentProfile(
                think_time_mu=3.40,  # 中位数 ~30s
                think_time_sigma=0.7,
                system_tokens_mean=400.0,
                tools_tokens_mean=200.0,
                new_tokens_mean=700.0,
                new_tokens_std=250.0,
                output_tokens_mean=180.0,
                output_tokens_std=70.0,
            ),
        },
    )


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
        "eviction_count": summary["evictions"]["count"],
        "ttl_expired_tokens": summary["ttl_expired_tokens"],
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ttls = sorted({float(value) for value in args.ttls.split(",")})
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    trace = generate_trace(build_config(args), args.seed)
    serving_config = ServingConfig(cache_capacity_tokens=args.capacity)
    print(
        f"trace: {len(trace)} requests, capacity={args.capacity} tokens, "
        f"ttls={ttls}"
    )

    runs: dict[str, object] = {}
    collectors = {}
    collectors["lru"] = _run(trace, serving_config, LRUPolicy())
    for ttl in ttls:
        collectors[f"ttl-{ttl:g}"] = _run(trace, serving_config, TTLPolicy(ttl=ttl))

    rows = [table_row(label, collector.summary()) for label, collector in collectors.items()]
    _print_table(rows)

    csv_path = out_dir / "exp002_summary.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    lru_summary = collectors["lru"].summary()
    json_path = out_dir / "exp002_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "seed": args.seed,
                "capacity": args.capacity,
                "ttls": ttls,
                "lru_reference": table_row("lru", lru_summary),
                "results": {row["policy"]: row for row in rows},
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    ttl_labels = [f"ttl-{ttl:g}" for ttl in ttls]
    plot_sweep(
        "ttl (s)",
        ttls,
        {
            "total": [collectors[label].summary()["hit_rate"] for label in ttl_labels],
            "coding": [class_stats(collectors[label].summary(), "coding")[0] for label in ttl_labels],
            "search": [class_stats(collectors[label].summary(), "search")[0] for label in ttl_labels],
            "lru total": [lru_summary["hit_rate"]] * len(ttls),
        },
        out_dir / "exp002_ttl_sweep_hit_rate.png",
        title="hit rate vs TTL (heterogeneous load)",
        ylabel="hit rate",
    )
    plot_sweep(
        "ttl (s)",
        ttls,
        {
            "total": [collectors[label].summary()["jct_mean"] for label in ttl_labels],
            "coding": [class_stats(collectors[label].summary(), "coding")[1] for label in ttl_labels],
            "search": [class_stats(collectors[label].summary(), "search")[1] for label in ttl_labels],
            "lru total": [lru_summary["jct_mean"]] * len(ttls),
        },
        out_dir / "exp002_ttl_sweep_jct.png",
        title="mean JCT vs TTL (heterogeneous load)",
        ylabel="JCT (s)",
    )
    best_label = max(
        (label for label in ttl_labels),
        key=lambda label: collectors[label].summary()["hit_rate"],
    )
    # 拐点：coding 类命中率最接近 LRU 的最小 TTL（工作点推荐值）
    lru_coding = class_stats(lru_summary, "coding")[0]
    knee_label = next(
        (
            label
            for label in ttl_labels
            if class_stats(collectors[label].summary(), "coding")[0] >= 0.98 * lru_coding
        ),
        ttl_labels[-1],
    )
    print(f"knee (coding hit >= 98% of LRU at smallest ttl): {knee_label}")
    plot_cdf(
        {
            "lru": collectors["lru"].jct_values(agent_type="coding"),
            f"{best_label} (best)": collectors[best_label].jct_values(agent_type="coding"),
            f"{knee_label} (knee)": collectors[knee_label].jct_values(agent_type="coding"),
        },
        out_dir / "exp002_coding_jct_cdf.png",
        title="coding-agent JCT CDF",
        xlabel="JCT (s)",
    )
    print(f"outputs written to {out_dir}")
    return 0


def _run(trace, serving_config, policy):
    sim = ServingSim(serving_config, policy=policy)
    sim.submit_all(trace)
    sim.run()
    return sim.collector


def _print_table(rows: list[dict]) -> None:
    widths = [
        max(len(str(row[column])) for row in rows + [dict(zip(TABLE_COLUMNS, TABLE_COLUMNS))])
        for column in TABLE_COLUMNS
    ]
    header = "  ".join(column.ljust(width) for column, width in zip(TABLE_COLUMNS, widths))
    print(header)
    for row in rows:
        print("  ".join(str(row[column]).ljust(width) for column, width in zip(TABLE_COLUMNS, widths)))


if __name__ == "__main__":
    sys.exit(main())
