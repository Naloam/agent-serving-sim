"""exp006：抢占与驱逐成本语义下的 TTL vs LRU。

M3 的结构性结论：驱逐免费时 TTL ≤ LRU（过期集 ⊆ LRU 序尾部）。本实验
打开两个二阶效应，检验文献（Continuum/CacheTTL）主张的收益条件：

1. **decode 分块增长 + 抢占**（``decode_chunks=4``）：增长遇容量耗尽先驱逐
   idle、无 idle 可逐则抢占最新在途请求（重算成本入 JCT）；
2. **驱逐成本**（``evict_tps=2000``）：按需驱逐的 token 折入触发请求的
   关键路径；TTL 的到点清除（sweep）在模型中仍是免费的——对应真实系统
   "免锁定时释放 vs 持锁按需回收"的差异。

负载与 exp002 相同（coding 快回转 5s + search 长思考 30s，容量受压）。

用法::

    python experiments/exp006_preemption_and_eviction_cost.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ass.cache.policies import LRUPolicy, TTLPolicy
from ass.scheduler.serving import ServingConfig, ServingSim
from ass.viz.plots import plot_sweep
from ass.workload.synthetic import generate_trace
from exp002_ttl_sweep_heterogeneous import build_config as build_exp002_config

TTLS = (15.0, 30.0, 60.0)
EVICT_MODES = {"free": None, "2k-tok/s": 2000.0}
TABLE_COLUMNS = (
    "evict_mode",
    "policy",
    "hit_total",
    "jct_total",
    "jct_p95",
    "preemptions",
    "wasted_compute_s",
    "dropped_tokens",
    "evicted_tokens",
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TTL vs LRU under preemption and eviction cost")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--capacity", type=int, default=80_000)
    parser.add_argument("--out-dir", type=str, default="experiments/results")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace_ns = argparse.Namespace(
        coding_sessions=140, search_sessions=70, turns=10, rate=0.12
    )
    trace = generate_trace(build_exp002_config(trace_ns), args.seed)
    print(f"trace: {len(trace)} requests, capacity={args.capacity}, decode_chunks=4")

    rows: list[dict] = []
    results: dict[tuple[str, str], dict] = {}
    for evict_mode, evict_tps in EVICT_MODES.items():
        config = ServingConfig(
            cache_capacity_tokens=args.capacity,
            decode_chunks=4,
            allow_preemption=True,
            evict_tps=evict_tps,
        )
        for label, policy in [("lru", LRUPolicy())] + [
            (f"ttl-{ttl:g}", TTLPolicy(ttl=ttl)) for ttl in TTLS
        ]:
            sim = ServingSim(config, policy=policy)
            sim.submit_all(trace)
            sim.run()
            summary = sim.collector.summary()
            results[(evict_mode, label)] = summary
            rows.append(
                {
                    "evict_mode": evict_mode,
                    "policy": label,
                    "hit_total": round(summary["hit_rate"], 4),
                    "jct_total": round(summary["jct_mean"], 3),
                    "jct_p95": round(summary["jct_p95"], 3),
                    "preemptions": summary["preemptions"]["count"],
                    "wasted_compute_s": summary["preemptions"]["wasted_compute_s"],
                    "dropped_tokens": summary["preemptions"]["dropped_tokens"],
                    "evicted_tokens": summary["evictions"]["tokens"],
                }
            )

    widths = [
        max(len(str(row[column])) for row in rows + [dict(zip(TABLE_COLUMNS, TABLE_COLUMNS))])
        for column in TABLE_COLUMNS
    ]
    header = "  ".join(column.ljust(width) for column, width in zip(TABLE_COLUMNS, widths))
    print(header)
    for row in rows:
        print("  ".join(str(row[column]).ljust(width) for column, width in zip(TABLE_COLUMNS, widths)))

    # 读数：驱逐免费时 TTL <= LRU；驱逐计费后 TTL 是否翻盘？
    for evict_mode in EVICT_MODES:
        lru_jct = results[(evict_mode, "lru")]["jct_mean"]
        best_ttl = min(
            (f"ttl-{ttl:g}" for ttl in TTLS),
            key=lambda label: results[(evict_mode, label)]["jct_mean"],
        )
        best_jct = results[(evict_mode, best_ttl)]["jct_mean"]
        verdict = "TTL WINS" if best_jct < lru_jct else "LRU wins/ties"
        print(f"[{evict_mode}] lru jct={lru_jct:.3f}, best {best_ttl} jct={best_jct:.3f} -> {verdict}")

    with (out_dir / "exp006_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "exp006_summary.json").write_text(
        json.dumps({"seed": args.seed, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    labels = ["lru"] + [f"ttl-{ttl:g}" for ttl in TTLS]
    plot_sweep(
        "policy (categorical index)",
        list(range(len(labels))),
        {
            f"jct @ evict={mode}": [results[(mode, label)]["jct_mean"] for label in labels]
            for mode in EVICT_MODES
        },
        out_dir / "exp006_jct_by_evict_mode.png",
        title="mean JCT: eviction free vs costly",
        ylabel="JCT (s)",
    )
    print(f"outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
