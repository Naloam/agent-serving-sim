"""exp008：预测型在线策略 vs LRU/Belady 缺口收窄。

exp007 的靶点：LRU 距 Belady 上限 16~25%，缺口集中在慢回转类。本实验
对比四方：

- ``lru``：基线；
- ``class-ttl``：按类型静态 TTL（预测的"类级"下界，3× 类 think 中位数）；
- ``predict``：在线学习（log-think 对数正态 + 逐轮存活率，见
  :class:`~ass.cache.policies.PredictivePolicy`）——只用因果信息；
- ``belady``：离线最优上限（需要完整 trace，仅作参照）。

核心指标：**缺口收窄比例** = (hit_policy − hit_lru) / (hit_belady − hit_lru)。
负载：合成异构（exp002 配置，80K/40K 两档）+ 真实 trace（4K 受压档）。

用法::

    python experiments/exp008_predictive_policy.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ass.cache.policies import BeladyPolicy, ClassTTLPolicy, LRUPolicy, PredictivePolicy
from ass.scheduler.serving import ServingConfig, ServingSim
from ass.viz.plots import plot_sweep
from ass.workload.schema import read_trace
from ass.workload.synthetic import generate_trace
from exp002_ttl_sweep_heterogeneous import build_config as build_exp002_config

TABLE_COLUMNS = ("workload", "capacity", "policy", "hit_total", "hit_coding",
                 "hit_search", "jct_total", "gap_closed_pct")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predictive policy vs LRU/Belady gap")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--real-capacity", type=int, default=4000)
    parser.add_argument("--out-dir", type=str, default="experiments/results")
    return parser.parse_args(argv)


def class_hit(summary: dict, agent_type: str) -> float:
    bucket = summary["by_agent_type"].get(agent_type)
    return bucket["hit_rate"] if bucket else 0.0


def build_policies(trace, class_ttls: dict[str, float]):
    return [
        ("lru", LRUPolicy()),
        ("class-ttl", ClassTTLPolicy(ttls=class_ttls)),
        ("predict", PredictivePolicy()),  # 窗口回归概率排序（实测最优）
        ("predict-mrl", PredictivePolicy(rank_by="residual")),  # MRL 对照（负结果存档）
        ("belady", BeladyPolicy(trace)),
    ]


def run_all(label: str, trace, capacity: int, class_ttls: dict[str, float], rows: list):
    config = ServingConfig(cache_capacity_tokens=capacity)
    summaries = {}
    for name, policy in build_policies(trace, class_ttls):
        sim = ServingSim(config, policy=policy)
        sim.submit_all(trace)
        sim.run()
        summaries[name] = sim.collector.summary()
    lru_hit = summaries["lru"]["hit_rate"]
    belady_hit = summaries["belady"]["hit_rate"]
    for name in ("lru", "class-ttl", "predict", "predict-mrl", "belady"):
        summary = summaries[name]
        gap = belady_hit - lru_hit
        closed = (
            (summary["hit_rate"] - lru_hit) / gap * 100.0
            if name != "lru" and gap > 1e-9 else
            (100.0 if name == "belady" else 0.0)
        )
        rows.append({
            "workload": label,
            "capacity": capacity,
            "policy": name,
            "hit_total": round(summary["hit_rate"], 4),
            "hit_coding": round(class_hit(summary, "coding"), 4),
            "hit_search": round(class_hit(summary, "search"), 4),
            "jct_total": round(summary["jct_mean"], 3),
            "gap_closed_pct": round(closed, 1),
        })


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    # 合成异构：coding think 中位 ~5s、search ~30s → 类 TTL = 3×中位
    trace_ns = argparse.Namespace(coding_sessions=140, search_sessions=70, turns=10, rate=0.12)
    synthetic = generate_trace(build_exp002_config(trace_ns), args.seed)
    for capacity in (80_000, 40_000):
        run_all("synthetic", synthetic, capacity, {"coding": 15.0, "search": 90.0}, rows)

    # 真实 trace（刻画：coding 5.8s / search 19.3s 中位）
    real_dir = Path("traces/real")
    real = []
    for path in sorted(real_dir.glob("*.jsonl")):
        real.extend(read_trace(path))
    real.sort(key=lambda r: r.arrival_time)
    if real:
        run_all("real", real, args.real_capacity, {"coding": 18.0, "search": 58.0}, rows)

    widths = [max(len(str(row[c])) for row in rows + [dict(zip(TABLE_COLUMNS, TABLE_COLUMNS))])
              for c in TABLE_COLUMNS]
    header = "  ".join(c.ljust(w) for c, w in zip(TABLE_COLUMNS, widths))
    print(header)
    for row in rows:
        print("  ".join(str(row[c]).ljust(w) for c, w in zip(TABLE_COLUMNS, widths)))

    with (out_dir / "exp008_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "exp008_summary.json").write_text(
        json.dumps({"seed": args.seed, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    series: dict[str, list[float]] = {}
    for name in ("lru", "class-ttl", "predict", "belady"):        series[name] = [
            next(r["hit_total"] for r in rows
                 if r["workload"] == w and r["capacity"] == c and r["policy"] == name)
            for w, c in (("synthetic", 80000), ("synthetic", 40000), ("real", args.real_capacity))
        ]
    plot_sweep(
        "workload x capacity (80k / 40k / real-4k)",
        [1, 2, 3],
        series,
        out_dir / "exp008_gap_closure.png",
        title="predictive policy vs LRU and Belady bound",
        ylabel="hit rate",
    )
    print(f"outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
