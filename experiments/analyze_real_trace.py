"""真实 trace 负载刻画与计时标定（M2 收尾，M3 标定输入）。

输入：探针原始 JSONL 日志。产出：

1. ``traces/real/<agent_type>.jsonl``：清洗对齐后的 FR-2 格式 trace 入库；
2. 负载特征统计（think_time 分布与对数正态拟合、prompt 四段占比、
   轮长增长、会话规模）+ 特征图；
3. 计时标定：用 ProbeTiming 做线性回归
   ``total_s = a + prompt_tokens / prefill_tps + completion_tokens / decode_tps``，
   输出 ``calibration.json`` 供 ServingConfig 使用。

用法::

    python experiments/analyze_real_trace.py --raw-log traces/real/raw/probe.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

from ass.viz.plots import plot_cdf, plot_timeline
from ass.workload.loaders import ProbeTiming, parse_probe_log
from ass.workload.schema import write_trace


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Real-trace characterization")
    parser.add_argument("--raw-log", type=str, default="traces/real/raw/probe.jsonl")
    parser.add_argument("--out-dir", type=str, default="traces/real")
    parser.add_argument("--fig-dir", type=str, default="experiments/results")
    return parser.parse_args(argv)


def fit_lognormal(values: list[float]) -> tuple[float, float]:
    logs = [math.log(value) for value in values if value > 0]
    if not logs:
        return 0.0, 0.0
    mean = sum(logs) / len(logs)
    variance = sum((x - mean) ** 2 for x in logs) / len(logs)
    return mean, math.sqrt(variance)


def percentile(sorted_values: list[float], pct: float) -> float:
    if not sorted_values:
        return 0.0
    index = min(int(round(pct / 100 * (len(sorted_values) - 1))), len(sorted_values) - 1)
    return sorted_values[index]


def _lstsq_fit(matrix: "np.ndarray", targets: "np.ndarray") -> tuple["np.ndarray", float]:
    coefficients, *_ = np.linalg.lstsq(matrix, targets, rcond=None)
    predicted = matrix @ coefficients
    ss_res = float(np.sum((targets - predicted) ** 2))
    ss_tot = float(np.sum((targets - targets.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
    return coefficients, r2


def calibrate(timings: list[ProbeTiming]) -> dict:
    """计时标定：总时延拟合 + 流式数据的 TTFT/decode 分解拟合。

    - 总时延：``total = a + prompt/prefill_tps + completion/decode_tps``
      （非流式采集下排队混入，R² 有限，如实报告）；
    - TTFT 分解（首字节时间真实可得时）：``first_byte = a1 + prompt × b1``，
      给出更干净的 prefill 估计；decode 由 ``total − first_byte`` 对
      completion_tokens 回归。
    """
    result: dict = {"samples": len(timings)}
    if len(timings) < 10:
        result["note"] = "insufficient samples"
        return result
    prompts = np.array([t.prompt_tokens for t in timings], dtype=float)
    completions = np.array([t.completion_tokens for t in timings], dtype=float)
    totals = np.array([t.total_seconds for t in timings], dtype=float)
    firsts = np.array([t.first_byte_seconds for t in timings], dtype=float)

    matrix = np.column_stack([np.ones_like(prompts), prompts, completions])
    coefficients, r2 = _lstsq_fit(matrix, totals)
    intercept, prompt_coeff, completion_coeff = (float(x) for x in coefficients)
    result.update(
        {
            "intercept_s": round(intercept, 4),
            "prefill_tps": round(1.0 / prompt_coeff, 1) if prompt_coeff > 1e-9 else None,
            "decode_tps": round(1.0 / completion_coeff, 1) if completion_coeff > 1e-9 else None,
            "r2": round(r2, 4),
            "prompt_coeff_s_per_token": round(prompt_coeff, 6),
            "completion_coeff_s_per_token": round(completion_coeff, 6),
        }
    )

    streaming = bool(np.any(firsts < totals - 1e-9))
    result["streaming"] = streaming
    if streaming:
        ttft_matrix = np.column_stack([np.ones_like(prompts), prompts])
        ttft_coeff, ttft_r2 = _lstsq_fit(ttft_matrix, firsts)
        ttft_intercept, ttft_prompt_coeff = (float(x) for x in ttft_coeff)
        decode_times = np.maximum(totals - firsts, 0.0)
        decode_matrix = np.column_stack([np.ones_like(completions), completions])
        decode_coeff, decode_r2 = _lstsq_fit(decode_matrix, decode_times)
        decode_intercept, decode_completion_coeff = (float(x) for x in decode_coeff)
        result["ttft_fit"] = {
            "intercept_s": round(ttft_intercept, 4),
            "prefill_tps": round(1.0 / ttft_prompt_coeff, 1) if ttft_prompt_coeff > 1e-9 else None,
            "r2": round(ttft_r2, 4),
        }
        result["decode_fit"] = {
            "intercept_s": round(decode_intercept, 4),
            "decode_tps": round(1.0 / decode_completion_coeff, 1) if decode_completion_coeff > 1e-9 else None,
            "r2": round(decode_r2, 4),
        }
    return result


def characterize(report, out_dir: Path, fig_dir: Path) -> dict:
    by_type: dict[str, list] = defaultdict(list)
    for request in report.requests:
        by_type[request.agent_type].append(request)
    for agent_type, requests in sorted(by_type.items()):
        write_trace(requests, out_dir / f"{agent_type}.jsonl")

    stats: dict[str, dict] = {}
    think_cdf_series: dict[str, list[float]] = {}
    growth_series: dict[str, tuple[list[float], list[float]]] = {}
    for agent_type, requests in sorted(by_type.items()):
        sessions = {request.session_id for request in requests}
        turns = [request.turn_id for request in requests]
        thinks = [request.think_time for request in requests if request.turn_id > 1]
        thinks_sorted = sorted(thinks)
        mu, sigma = fit_lognormal(thinks)
        segments = {
            "system": sum(r.prompt.system for r in requests) / len(requests),
            "tools": sum(r.prompt.tools for r in requests) / len(requests),
            "history": sum(r.prompt.history for r in requests) / len(requests),
            "new": sum(r.prompt.new for r in requests) / len(requests),
        }
        total_prompt = sum(r.prompt.total for r in requests) / len(requests)
        outputs = sorted(r.output_tokens for r in requests)
        by_turn: dict[int, list[int]] = defaultdict(list)
        for request in requests:
            by_turn[request.turn_id].append(request.prompt.total)
        turn_ids = sorted(by_turn)
        turn_means = [sum(by_turn[t]) / len(by_turn[t]) for t in turn_ids]
        stats[agent_type] = {
            "requests": len(requests),
            "sessions": len(sessions),
            "turns_per_session_mean": len(requests) / len(sessions),
            "max_turns": max(turns),
            "think_time": {
                "mean": round(sum(thinks) / len(thinks), 2) if thinks else 0.0,
                "median": round(percentile(thinks_sorted, 50), 2),
                "p95": round(percentile(thinks_sorted, 95), 2),
                "lognormal_mu": round(mu, 3),
                "lognormal_sigma": round(sigma, 3),
                "lognormal_median": round(math.exp(mu), 2) if mu else 0.0,
            },
            "prompt_mean_tokens": round(total_prompt, 1),
            "segment_share": {
                name: round(value / total_prompt, 3) for name, value in segments.items()
            },
            "output_tokens": {
                "mean": round(sum(outputs) / len(outputs), 1),
                "p95": round(percentile(outputs, 95), 1),
            },
            "preamble_share": round(
                (segments["system"] + segments["tools"]) / total_prompt, 3
            ),
            "turn_growth": {
                str(turn): round(mean, 1) for turn, mean in zip(turn_ids, turn_means)
            },
        }
        think_cdf_series[agent_type] = thinks
        growth_series[agent_type] = (turn_ids, turn_means)  # type: ignore[assignment]

    fig_dir.mkdir(parents=True, exist_ok=True)
    non_empty_thinks = {
        agent_type: thinks
        for agent_type, thinks in think_cdf_series.items()
        if thinks
    }
    if non_empty_thinks:
        plot_cdf(
            non_empty_thinks,
            fig_dir / "real_think_time_cdf.png",
            title="think_time CDF by agent type (real trace)",
            xlabel="think_time (s)",
        )
    for agent_type, (turn_ids, turn_means) in growth_series.items():
        plot_timeline(
            turn_ids,
            turn_means,
            fig_dir / f"real_prompt_growth_{agent_type}.png",
            title=f"mean prompt tokens per turn ({agent_type})",
            xlabel="turn",
            ylabel="tokens",
        )
    return stats


def write_report(path: Path, stats: dict, calibration: dict, skipped_count: int) -> None:
    lines = [
        "# 真实 agent 负载刻画报告（M2）",
        "",
        f"- 解析跳过行数：{skipped_count}",
        f"- 计时标定（总时延拟合）：prefill_tps={calibration.get('prefill_tps')}, "
        f"decode_tps={calibration.get('decode_tps')}, R²={calibration.get('r2')}, "
        f"截距={calibration.get('intercept_s')}s（样本 {calibration.get('samples')}）",
    ]
    if calibration.get("ttft_fit"):
        ttft = calibration["ttft_fit"]
        decode = calibration.get("decode_fit", {})
        lines.append(
            f"- TTFT 分解（流式）：prefill_tps={ttft.get('prefill_tps')} "
            f"(R²={ttft.get('r2')}), decode_tps={decode.get('decode_tps')} "
            f"(R²={decode.get('r2')}), TTFT 截距={ttft.get('intercept_s')}s"
        )
    lines.append("")
    for agent_type, bucket in stats.items():
        think = bucket["think_time"]
        lines += [
            f"## {agent_type}",
            "",
            f"- 请求 {bucket['requests']} 个 / 会话 {bucket['sessions']} 个"
            f"（平均 {bucket['turns_per_session_mean']:.1f} 轮，最长 {bucket['max_turns']} 轮）",
            f"- think_time：中位数 {think['median']}s，均值 {think['mean']}s，"
            f"p95 {think['p95']}s；对数正态拟合 μ={think['lognormal_mu']}, "
            f"σ={think['lognormal_sigma']}（拟合中位数 {think['lognormal_median']}s）",
            f"- prompt 平均 {bucket['prompt_mean_tokens']} tokens，"
            f"前导（system+tools）占 {bucket['preamble_share']:.1%}，"
            f"四段占比 {bucket['segment_share']}",
            f"- 输出平均 {bucket['output_tokens']['mean']} tokens（p95 {bucket['output_tokens']['p95']}）",
            f"- 逐轮 prompt 增长：{bucket['turn_growth']}",
            "",
        ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = parse_probe_log(args.raw_log)
    print(f"parsed {len(report.requests)} requests, skipped {len(report.skipped)} lines")
    if not report.requests:
        print("no usable requests in raw log", file=sys.stderr)
        return 2

    stats = characterize(report, out_dir, Path(args.fig_dir))
    calibration = calibrate(report.timings)
    write_report(out_dir / "REPORT.md", stats, calibration, len(report.skipped))
    (out_dir / "calibration.json").write_text(
        json.dumps({"calibration": calibration, "stats": stats}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(json.dumps({"calibration": calibration}, indent=2))
    for agent_type, bucket in stats.items():
        print(
            f"{agent_type}: {bucket['requests']} req / {bucket['sessions']} sessions, "
            f"think median {bucket['think_time']['median']}s, "
            f"preamble share {bucket['preamble_share']}"
        )
    print(f"traces written to {out_dir}, report at {out_dir / 'REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
