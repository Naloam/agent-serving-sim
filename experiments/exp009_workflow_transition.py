"""exp009：多智能体工作流负载下的策略对比（M5-A / FR-15）。

负载：orchestrator 根会话派生 coder / searcher，coder 再派生 critic
（马尔可夫转移）；流内共享前导（``flow:`` 前缀流，跨 agent 类型复用）。

科学问题：在结构性负载上，**工作流转移知识**（TransitionPolicy：在线学
派生图 + 活跃前沿 BFS 跳距）能否胜过**纯时间维度预测**（PredictivePolicy：
类内 think_time 对数正态 + 窗口回归概率）？两者距 Belady 上限各多远？

用法::

    python experiments/exp009_workflow_transition.py --seed 42
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from ass.cache.policies import BeladyPolicy, LRUPolicy, PredictivePolicy, TransitionPolicy
from ass.scheduler.serving import ServingConfig, ServingSim
from ass.viz.plots import plot_sweep
from ass.workload.synthetic import AgentProfile, SyntheticConfig, WorkflowConfig, generate_trace

TABLE_COLUMNS = ("policy", "hit_total", "hit_orchestrator", "hit_coder",
                 "hit_searcher", "hit_critic", "jct_total", "gap_closed_pct")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Workflow load: transition vs time prediction")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--flows", type=int, default=60)
    parser.add_argument("--capacity", type=int, default=50_000)
    parser.add_argument("--out-dir", type=str, default="experiments/results")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> SyntheticConfig:
    return SyntheticConfig(
        num_sessions=args.flows,
        turns_per_session=6,
        session_arrival_rate=0.25,
        system_tokens_mean=1000.0,
        tools_tokens_mean=400.0,
        agent_mix={"orchestrator": 1.0},
        agent_profiles={
            "orchestrator": AgentProfile(
                think_time_mu=1.61, think_time_sigma=0.5,
                new_tokens_mean=150.0, new_tokens_std=50.0,
                output_tokens_mean=250.0, output_tokens_std=80.0,
            ),
            "coder": AgentProfile(
                think_time_mu=1.61, think_time_sigma=0.5,
                new_tokens_mean=250.0, new_tokens_std=80.0,
                output_tokens_mean=350.0, output_tokens_std=100.0,
            ),
            "searcher": AgentProfile(
                think_time_mu=3.0, think_time_sigma=0.7,
                new_tokens_mean=600.0, new_tokens_std=200.0,
                output_tokens_mean=180.0, output_tokens_std=60.0,
            ),
            "critic": AgentProfile(
                think_time_mu=2.0, think_time_sigma=0.5,
                new_tokens_mean=200.0, new_tokens_std=70.0,
                output_tokens_mean=220.0, output_tokens_std=80.0,
            ),
        },
        workflow=WorkflowConfig(
            transitions={
                "orchestrator": {"coder": 0.6, "searcher": 0.4},
                "coder": {"critic": 1.0},
            },
            children_per_flow=3,
            child_turns=4,
            grandchild_prob=0.5,
        ),
    )


def class_hit(summary: dict, agent_type: str) -> float:
    bucket = summary["by_agent_type"].get(agent_type)
    return bucket["hit_rate"] if bucket else 0.0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = generate_trace(build_config(args), args.seed)
    config = ServingConfig(cache_capacity_tokens=args.capacity)
    print(f"trace: {len(trace)} requests, {args.flows} flows, capacity={args.capacity}")

    runs = {}
    for label, policy in (
        ("lru", LRUPolicy()),
        ("predict", PredictivePolicy()),
        ("transition", TransitionPolicy(active_window_s=20.0)),
        ("belady", BeladyPolicy(trace)),
    ):
        sim = ServingSim(config, policy=policy)
        sim.submit_all(trace)
        sim.run()
        runs[label] = sim.collector.summary()

    lru_hit = runs["lru"]["hit_rate"]
    belady_hit = runs["belady"]["hit_rate"]
    rows = []
    for label in ("lru", "predict", "transition", "belady"):
        summary = runs[label]
        gap = belady_hit - lru_hit
        closed = (
            100.0 if label == "belady" else
            (summary["hit_rate"] - lru_hit) / gap * 100.0 if gap > 1e-9 else 0.0
        )
        rows.append({
            "policy": label,
            "hit_total": round(summary["hit_rate"], 4),
            "hit_orchestrator": round(class_hit(summary, "orchestrator"), 4),
            "hit_coder": round(class_hit(summary, "coder"), 4),
            "hit_searcher": round(class_hit(summary, "searcher"), 4),
            "hit_critic": round(class_hit(summary, "critic"), 4),
            "jct_total": round(summary["jct_mean"], 3),
            "gap_closed_pct": round(closed, 1),
        })

    widths = [max(len(str(row[c])) for row in rows + [dict(zip(TABLE_COLUMNS, TABLE_COLUMNS))])
              for c in TABLE_COLUMNS]
    print("  ".join(c.ljust(w) for c, w in zip(TABLE_COLUMNS, widths)))
    for row in rows:
        print("  ".join(str(row[c]).ljust(w) for c, w in zip(TABLE_COLUMNS, widths)))

    with (out_dir / "exp009_summary.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=TABLE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "exp009_summary.json").write_text(
        json.dumps({"seed": args.seed, "rows": rows}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    plot_sweep(
        "policy",
        [1, 2, 3, 4],
        {
            "hit_total": [row["hit_total"] for row in rows],
            "gap_closed_pct": [row["gap_closed_pct"] for row in rows],
        },
        out_dir / "exp009_workflow_policies.png",
        title="workflow load: lru / predict / transition / belady",
        ylabel="hit rate (line), gap closed % (line)",
    )
    print(f"outputs written to {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
