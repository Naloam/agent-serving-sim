"""合成负载生成器的单元测试（FR-3）：复现性、前缀结构、统计量。"""

import math
from collections import defaultdict

import pytest

from ass.workload.synthetic import SyntheticConfig, generate_trace
from ass.workload.schema import write_trace, read_trace


CONFIG = SyntheticConfig(num_sessions=100, turns_per_session=6)


def test_same_seed_is_byte_identical(tmp_path) -> None:
    """固定 seed 两次生成结果逐字节一致（PRD 验收项）。"""
    path_a, path_b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_trace(generate_trace(CONFIG, seed=42), path_a)
    write_trace(generate_trace(CONFIG, seed=42), path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def test_different_seed_differs() -> None:
    trace_a = generate_trace(CONFIG, seed=42)
    trace_b = generate_trace(CONFIG, seed=43)
    assert trace_a != trace_b


def test_request_count_and_session_ids() -> None:
    trace = generate_trace(CONFIG, seed=42)
    assert len(trace) == CONFIG.num_sessions * CONFIG.turns_per_session
    sessions = {req.session_id for req in trace}
    assert len(sessions) == CONFIG.num_sessions


def test_history_accumulates_previous_turns() -> None:
    """history 逐轮累加前序 new + output（轮长增长与配置吻合）。"""
    trace = generate_trace(CONFIG, seed=42)
    by_session: dict[str, list] = defaultdict(list)
    for req in trace:
        by_session[req.session_id].append(req)
    for requests in by_session.values():
        requests.sort(key=lambda r: r.turn_id)
        running = 0
        for req in requests:
            assert req.prompt.history == running
            running += req.prompt.new + req.output_tokens


def test_preamble_shared_within_agent_type() -> None:
    """同 agent_type 的会话共享 system/tools 长度。"""
    trace = generate_trace(CONFIG, seed=42)
    seen: dict[str, tuple[int, int]] = {}
    for req in trace:
        key = (req.prompt.system, req.prompt.tools)
        if req.agent_type in seen:
            assert seen[req.agent_type] == key
        else:
            seen[req.agent_type] = key
    assert len(seen) >= 2  # 默认 mix 下两种类型都应出现


def test_agent_mix_matches_config() -> None:
    trace = generate_trace(CONFIG, seed=42)
    sessions = {req.session_id: req.agent_type for req in trace}
    coding = sum(1 for t in sessions.values() if t == "coding")
    expected = CONFIG.num_sessions * 0.7
    assert abs(coding - expected) < 0.2 * CONFIG.num_sessions


def test_session_arrivals_are_poisson() -> None:
    """会话首轮到达间隔应接近指数分布（均值 1/rate）。"""
    trace = generate_trace(CONFIG, seed=42)
    first_arrivals = sorted(
        req.arrival_time for req in trace if req.turn_id == 1
    )
    intervals = [b - a for a, b in zip(first_arrivals, first_arrivals[1:])]
    mean_interval = sum(intervals) / len(intervals)
    assert pytest_approx(mean_interval, 1.0 / CONFIG.session_arrival_rate, rel=0.25)


def test_turn_arrivals_increase_and_think_time_positive() -> None:
    trace = generate_trace(CONFIG, seed=42)
    by_session: dict[str, list] = defaultdict(list)
    for req in trace:
        by_session[req.session_id].append(req)
    for requests in by_session.values():
        requests.sort(key=lambda r: r.turn_id)
        for prev, curr in zip(requests, requests[1:]):
            assert curr.arrival_time > prev.arrival_time
            assert curr.think_time > 0.0
        assert requests[0].think_time == 0.0


def test_think_time_lognormal_mean() -> None:
    """think_time 样本均值接近对数正态理论均值 exp(mu + sigma^2/2)。"""
    wide = SyntheticConfig(num_sessions=200, turns_per_session=4)
    trace = generate_trace(wide, seed=7)
    thinks = [req.think_time for req in trace if req.turn_id > 1]
    expected = math.exp(wide.think_time_mu + wide.think_time_sigma**2 / 2)
    assert pytest_approx(sum(thinks) / len(thinks), expected, rel=0.2)


def test_agent_profiles_override_per_type() -> None:
    """AgentProfile：coding 快回转、search 长思考，各类取各自的生效值。"""
    from ass.workload.synthetic import AgentProfile

    config = SyntheticConfig(
        num_sessions=120,
        turns_per_session=4,
        agent_mix={"coding": 0.5, "search": 0.5},
        agent_profiles={
            "coding": AgentProfile(think_time_mu=1.61, think_time_sigma=0.5, new_tokens_mean=120.0),
            "search": AgentProfile(think_time_mu=3.0, think_time_sigma=0.7, new_tokens_mean=600.0),
        },
    )
    assert config.value("coding", "think_time_mu") == 1.61
    assert config.value("search", "think_time_mu") == 3.0
    assert config.value("coding", "new_tokens_std") == config.new_tokens_std  # 未覆盖沿用全局

    trace = generate_trace(config, seed=7)
    thinks: dict[str, list[float]] = {"coding": [], "search": []}
    news: dict[str, list[int]] = {"coding": [], "search": []}
    for req in trace:
        if req.turn_id > 1:
            thinks[req.agent_type].append(req.think_time)
        news[req.agent_type].append(req.prompt.new)
    assert _close(_mean(thinks["coding"]), math.exp(1.61 + 0.25 / 2), 0.35)
    assert _close(_mean(thinks["search"]), math.exp(3.0 + 0.49 / 2), 0.35)
    assert _close(_mean(news["coding"]), 120.0, 0.3)
    assert _close(_mean(news["search"]), 600.0, 0.3)


def test_agent_profiles_unknown_type_rejected() -> None:
    from ass.workload.synthetic import AgentProfile

    with pytest.raises(ValueError, match="unknown agent types"):
        SyntheticConfig(
            agent_mix={"coding": 1.0},
            agent_profiles={"ghost": AgentProfile(think_time_mu=1.0)},
        )


def test_agent_profiles_do_not_break_reproducibility(tmp_path) -> None:
    from ass.workload.synthetic import AgentProfile

    config = SyntheticConfig(
        num_sessions=40,
        turns_per_session=4,
        agent_mix={"coding": 0.5, "search": 0.5},
        agent_profiles={"coding": AgentProfile(think_time_mu=1.0)},
    )
    path_a, path_b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_trace(generate_trace(config, seed=42), path_a)
    write_trace(generate_trace(config, seed=42), path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def _close(value: float, expected: float, rel: float) -> bool:
    return abs(value - expected) <= rel * expected


def pytest_approx(value: float, expected: float, rel: float) -> bool:
    return abs(value - expected) <= rel * abs(expected)
