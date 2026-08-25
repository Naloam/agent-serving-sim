"""exp010：驱逐成本翻转 × 到达突发性（M5-B / FR-16）。

背景：exp006 发现"驱逐计费时 TTL 翻盘 LRU"，但那时到达是泊松
（CV≈1）；ServeGen（NSDI'26）表明生产到达 CV>1。本实验扫突发强度
（泊松 → 两档 MMPP），量化**翻转幅度与抢占风暴随突发强度的变化**——
驱逐成本 × 突发性的交叉此前无人研究。

负载：exp002 的异构配置（coding 快回转 + search 长思考），容量 80K，
decode 分块增长（chunks=4）。每组到达模式下对比免费驱逐与计费驱逐
（2k tok/s）下的 LRU 与 TTL。

用法::

    python experiments/exp010_burstiness_flip.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

from ass.cache.policies import LRUPolicy, TTLPolicy
from ass.scheduler.serving import ServingConfig, ServingSim
from ass.viz.plots import plot_sweep
from ass.workload.synthetic import MMPPConfig, SyntheticConfig, generate_trace
from exp002_ttl_sweep_heterogeneous import build_config as build_exp002_config

TTLS = (15.0, 30.0, 60.0)
TABLE_COLUMNS = ("arrival", "req_cv", "evict_mode", "lru_jct", "best_ttl",
                 "flip_pct", "preemptions", "onpath_evicted_tokens")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Eviction-cost flip x arrival burstiness")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity", type=int, default=80_000)
    parser.add_argument("--out-dir", type=str, default="experiments/results")
    return parser.parse_args(argv)


def request_cv(trace) -> float:
    arrivals = sorted(r.arrival_time for r in trace)
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:]) if b > a]
    if not gaps:
        return 0.0
    mean = sum(gaps) / len(gaps)
    variance = sum((g - mean) ** 2 for g in gaps) / len(gaps)
    return math.sqrt(variance) / mean if mean > 0 else 0.0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    arms = [
        ("poisson", None),
        ("mmpp-x4", MMPPConfig(background_rate=0.08, burst_rate=1.6,
                                mean_background_s=20.0, mean_burst_s=5.0)),
        ("mmpp-x12", MMPPConfig(background_rate=0.05, burst_rate=3.0,
                                 mean_background_s=25.0, mean_burst_s=5.0)),
    ]
    rows: list[dict] = []
    for arm_name, mmpp in arms:
        trace_ns = argparse.Namespace(coding_sessions=140, search_sessions=70,
                                      turns=10, rate=0.12)
        base = build_exp002_config(trace_ns)
        config = SyntheticConfig(**{**base.__dict__, "mmpp": mmpp})
        trace = generate_trace(config, args.seed)
        cv = request_cv(trace)
        for evict_mode, evict_tps in (("free", None), ("2k-tok/s", 2000.0)):
            serving = ServingConfig(cache_capacity_tokens=args.capacity,
                                    decode_chunks=4, evict_tps=evict_tps)
            summaries = {}
            for label, policy in [("lru", LRUPolicy())] + [
                (f"ttl-{t:g}", TTLPolicy(ttl=t)) for t in TTLS
            ]:
                sim = ServingSim(serving, policy=policy)
                sim.submit_all(trace)
                sim.run()
                summaries[label] = sim.collector.summary()
            lru_jct = summaries["lru"]["jct_mean"]
            best_label = min((l for l in summaries if l != "lru"),
                             key=lambda l: summaries[l]["jct_mean"])
            best_jct = summaries[best_label]["jct_mean"]
            flip = (lru_jct - best_jct) / lru_jct * 100.0 if lru_jct > 0 else 0.0
            rows.append({
                "arrival": arm_name,
                "req_cv": round(cv, 2),
                "evict_mode": evict_mode,
                "lru_jct": round(lru_jct, 3),
                "best_ttl": f"{best_label}({best_jct:.3f})",
                "flip_pct": round(flip, 1),
                "preemptions": summaries["lru"]["preemptions"]["count"],
                "onpath_evicted_tokens": summaries["lru"]["evictions"]["tokens"],
            })
            print(f"[{arm_name} cv={cv:.2f} {evict_mode}] lru={lru_jct:.3f} "
                  f"best={best_label} {best_jct:.3f} flip={flip:.1f}% "
                  f"preempt={rows[-1]['preemptions']}")

    widths = [max(len(str(row[c])) for row in rows + [dict(zip(TABLE_COLUMNS, TABLE_COLUMNS))])
              for c in TABLE_COLUMNS]
    print("  ".join(c.ljust(w) for c, w in zip(TABLE_COLUMNS, widths)))
    for row in rows:
        print("  ".join(str(row[c]).ljust(w) for c, w in zip(TABLE_COLUMNS, widths)))

    with (out_dir / "exp010_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "exp010_summary.json").write_text(
        json.dumps({"seed": args.seed, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    costly = [r for r in rows if r["evict_mode"] == "2k-tok/s"]
    plot_sweep(
        "measured request-arrival CV",
        [r["req_cv"] for r in costly],
        {
            "flip_pct (costly eviction)": [r["flip_pct"] for r in costly],
            "preemptions": [r["preemptions"] for r in costly],
        },
        out_dir / "exp010_flip_vs_burstiness.png",
        title="TTL-vs-LRU flip magnitude and preemptions vs arrival burstiness",
        ylabel="flip % / preemption count",
    )
    print(f"outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
