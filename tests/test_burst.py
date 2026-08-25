"""M5-B 突发到达与外部 CSV 的单元测试。"""

import math

from ass.workload.synthetic import MMPPConfig, SyntheticConfig, generate_trace
from ass.workload.loaders import arrival_times_from_csv, arrivals_to_trace


def _cv(values: list[float]) -> float:
    mean = sum(values) / len(values)
    variance = sum((v - mean) ** 2 for v in values) / len(values)
    return math.sqrt(variance) / mean if mean > 0 else 0.0


def test_poisson_arrivals_cv_near_one() -> None:
    config = SyntheticConfig(num_sessions=400, turns_per_session=2)
    trace = generate_trace(config, seed=1)
    arrivals = [r.arrival_time for r in trace if r.turn_id == 1]
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    assert _cv(gaps) < 1.1  # 指数间隔 CV=1


def test_mmpp_arrivals_burstier_than_poisson() -> None:
    """MMPP 会话到达间隔 CV 显著大于泊松（ServeGen：生产 CV>1）。"""
    config = SyntheticConfig(
        num_sessions=400, turns_per_session=2,
        mmpp=MMPPConfig(background_rate=0.05, burst_rate=2.0,
                        mean_background_s=20.0, mean_burst_s=5.0),
    )
    trace = generate_trace(config, seed=1)
    arrivals = [r.arrival_time for r in trace if r.turn_id == 1]
    gaps = [b - a for a, b in zip(arrivals, arrivals[1:])]
    assert _cv(gaps) > 1.5


def test_mmpp_reproducible_and_valid() -> None:
    config = SyntheticConfig(num_sessions=50, turns_per_session=2,
                             mmpp=MMPPConfig())
    a = generate_trace(config, seed=7)
    b = generate_trace(config, seed=7)
    assert a == b
    first_turns = [r.arrival_time for r in a if r.turn_id == 1]
    assert first_turns == sorted(first_turns)  # 会话首轮到达单调不减
    try:
        MMPPConfig(background_rate=0)
        raised = False
    except ValueError:
        raised = True
    assert raised


def test_arrival_csv_roundtrip(tmp_path) -> None:
    """通用到达 CSV：读时间戳 → 映射单轮 trace（共享前缀流）。"""
    path = tmp_path / "arrivals.csv"
    path.write_text(
        "timestamp,prompt_tokens,completion_tokens\n"
        "0.0,120,30\n"
        "0.5,200,44\n"
        "3.25,180,12\n",
        encoding="utf-8",
    )
    times = arrival_times_from_csv(path)
    assert times == [0.0, 0.5, 3.25]

    trace = arrivals_to_trace(times, seed=3, shared_preamble_tokens=500,
                              new_tokens_mean=150, output_tokens_mean=40)
    assert len(trace) == 3
    assert all(r.turn_id == 1 and r.think_time == 0.0 for r in trace)
    assert all(r.prompt.system + r.prompt.tools == 500 for r in trace)
    assert all(r.agent_type == "chat" for r in trace)
    # 归一化到首个请求 0 时刻
    assert trace[0].arrival_time == 0.0 and trace[2].arrival_time == 3.25
    # 可复现
    again = arrivals_to_trace(times, seed=3, shared_preamble_tokens=500,
                              new_tokens_mean=150, output_tokens_mean=40)
    assert again == trace


def test_arrival_csv_skips_bad_rows(tmp_path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text(
        "timestamp\n0.0\nnot-a-number\n1.5\n\n2.0\n",
        encoding="utf-8",
    )
    times = arrival_times_from_csv(path)
    assert times == [0.0, 1.5, 2.0]
