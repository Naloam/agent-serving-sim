"""exp011：突发负载下准入策略 × 驱逐策略因子实验（M6 / FR-17）。

exp010 的"瓶颈迁移"发现：极端突发下排队取代缓存管理成为 JCT 主导。
本实验把瓶颈搬回自己手里量：在驱逐计费（2k tok/s）的突发负载上做
{准入} × {驱逐} 因子对比——

- 准入：fifo（基线）/ priority（coding:search = 2:1 类权重）/ sjf（短作业优先）
- 驱逐：lru / ttl-15（exp006 起的最优 TTL 档）

核心问题：瓶颈在准入侧时，准入策略能拿回多少 JCT？与驱逐策略的收益
可否叠加？每类尾部（饥饿代价）如何变化？另含泊松对照臂（无饱和 regime）。

用法::

    python experiments/exp011_admission_x_eviction.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ass.cache.policies import LRUPolicy, TTLPolicy
from ass.scheduler.admission import FIFOAdmission, PriorityAdmission, ShortestJobAdmission
from ass.scheduler.serving import ServingConfig, ServingSim
from ass.viz.plots import plot_sweep
from ass.workload.synthetic import MMPPConfig, SyntheticConfig, generate_trace
from exp002_ttl_sweep_heterogeneous import build_config as build_exp002_config

ADMISSIONS = (
    ("fifo", lambda: FIFOAdmission()),
    ("priority", lambda: PriorityAdmission(weights={"coding": 2.0, "search": 1.0})),
    ("sjf", lambda: ShortestJobAdmission()),
)
EVITIONS = (("lru", lambda: LRUPolicy()), ("ttl-15", lambda: TTLPolicy(ttl=15.0)))
TABLE_COLUMNS = ("arrival", "admission", "eviction", "jct_total", "jct_coding",
                 "jct_search", "coding_p95", "search_p95", "queue_delay_mean",
                 "preemptions")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Admission x eviction factorial under bursts")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity", type=int, default=80_000)
    parser.add_argument("--out-dir", type=str, default="experiments/results")
    return parser.parse_args(argv)


def _p95(collector, agent_type: str) -> float:
    values = sorted(collector.jct_values(agent_type=agent_type))
    if not values:
        return 0.0
    return values[min(int(round(0.95 * (len(values) - 1))), len(values) - 1)]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = [
        ("burst", MMPPConfig(background_rate=0.08, burst_rate=1.6,
                              mean_background_s=20.0, mean_burst_s=5.0)),
        ("poisson", None),
    ]
    rows: list[dict] = []
    for arm_name, mmpp in arms:
        base = build_exp002_config(argparse.Namespace(
            coding_sessions=140, search_sessions=70, turns=10, rate=0.12))
        config = SyntheticConfig(**{**base.__dict__, "mmpp": mmpp})
        trace = generate_trace(config, args.seed)
        serving = ServingConfig(cache_capacity_tokens=args.capacity, decode_chunks=4,
                                evict_tps=2000.0)
        for adm_name, adm_factory in ADMISSIONS:
            for ev_name, ev_factory in EVITIONS:
                sim = ServingSim(serving, policy=ev_factory(),
                                 admission=adm_factory())
                sim.submit_all(trace)
                sim.run()
                summary = sim.collector.summary()
                by_class = summary["by_agent_type"]
                rows.append({
                    "arrival": arm_name,
                    "admission": adm_name,
                    "eviction": ev_name,
                    "jct_total": round(summary["jct_mean"], 3),
                    "jct_coding": round(by_class.get("coding", {}).get("jct_mean", 0.0), 3),
                    "jct_search": round(by_class.get("search", {}).get("jct_mean", 0.0), 3),
                    "coding_p95": round(_p95(sim.collector, "coding"), 3),
                    "search_p95": round(_p95(sim.collector, "search"), 3),
                    "queue_delay_mean": round(summary["queue_delay_mean"], 3),
                    "preemptions": summary["preemptions"]["count"],
                })
        burst_rows = [r for r in rows if r["arrival"] == arm_name]
        baseline = next(r for r in burst_rows
                        if r["admission"] == "fifo" and r["eviction"] == "lru")
        print(f"=== {arm_name} (baseline fifo+lru jct={baseline['jct_total']}) ===")
        for row in burst_rows:
            gain = (baseline["jct_total"] - row["jct_total"]) / baseline["jct_total"] * 100
            print(f"  {row['admission']:>8} x {row['eviction']:>6}: "
                  f"jct={row['jct_total']:>8} ({gain:+.1f}%) queue={row['queue_delay_mean']:>7} "
                  f"coding_p95={row['coding_p95']:>7} search_p95={row['search_p95']:>7} "
                  f"preempt={row['preemptions']}")

    with (out_dir / "exp011_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "exp011_summary.json").write_text(
        json.dumps({"seed": args.seed, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    burst = [r for r in rows if r["arrival"] == "burst"]
    labels = [f"{a}+{e}" for a, _ in ADMISSIONS for e, _ in EVITIONS]
    by_key = {(r["admission"], r["eviction"]): r["jct_total"] for r in burst}
    plot_sweep(
        "admission + eviction (burst, costly)",
        list(range(len(labels))),
        {"jct_total": [by_key[(a, e)] for a, _ in ADMISSIONS for e, _ in EVITIONS]},
        out_dir / "exp011_admission_x_eviction.png",
        title="bursty load: admission x eviction factorial (mean JCT)",
        ylabel="JCT (s)",
    )
    print(f"outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
