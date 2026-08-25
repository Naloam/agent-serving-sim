"""合成负载生成器。

建模约定（与模拟器的结构化前缀模型对齐）：

- **会话到达**：泊松过程（``session_arrival_rate``），会话内第 1 轮到达即
  会话到达时刻，``think_time`` 为 0；
- **前缀结构**：每个 ``agent_type`` 一次性抽取 system / tools 长度，
  同类型会话共享相同的前导（对应"同一应用共享 system prompt 与工具定义"）；
  ``history`` 逐轮累加前序轮次的 ``new + output``，与会话内 token 流
  按位置对齐的假设一致；
- **下轮到达时间**：trace 是离线数据，生成器用解析式服务时间估计
  （``est_prefill_tps`` / ``est_decode_tps``，按零命中保守估计）预排
  ``arrival_{t+1} = arrival_t + est_service_t + think_time_{t+1}``；
- **可复现**：全部随机性来自注入的 ``random.Random(seed)``。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Mapping

from ass.workload.schema import PromptBreakdown, TraceRequest


@dataclass(frozen=True)
class AgentProfile:
    """单个 agent_type 的参数覆盖：未设置的字段沿用全局配置。

    M3 异构负载实验用：例如 coding 快回转（短 think_time）+ search 长思考
    （长 think_time、大 new 段），用于考察 TTL/优先级/配额的差异化收益。
    """

    think_time_mu: float | None = None
    think_time_sigma: float | None = None
    system_tokens_mean: float | None = None
    system_tokens_std: float | None = None
    tools_tokens_mean: float | None = None
    tools_tokens_std: float | None = None
    new_tokens_mean: float | None = None
    new_tokens_std: float | None = None
    output_tokens_mean: float | None = None
    output_tokens_std: float | None = None


@dataclass(frozen=True)
class WorkflowConfig:
    """多 agent 工作流负载的参数（M5）。

    会话组织为流（flow）：根会话按泊松到达；每条流共享一次抽取的
    system+tools 前导（跨 agent 类型复用，前缀流按 ``flow:`` 标识）；
    根会话在随机轮次结束后按转移概率 ``transitions[根类型]`` 派生子
    会话（子会话可再派生一层），派生延迟与子会话参数独立配置。
    """

    transitions: Mapping[str, Mapping[str, float]]
    child_delay_mu: float = 1.0
    child_delay_sigma: float = 0.4
    children_per_flow: int = 3
    child_turns: int = 4
    grandchild_prob: float = 0.4


@dataclass(frozen=True)
class MMPPConfig:
    """两态 MMPP 会话到达（M5-B 突发到达建模）。

    背景/突发两态各自指数持留（``mean_background_s`` / ``mean_burst_s``），
    会话到达率随状态取 ``background_rate`` / ``burst_rate``。生产的 LLM
    到达呈 CV>1 的突发性（ServeGen, NSDI'26），泊松（CV≈1）无法刻画。
    """

    background_rate: float = 0.05
    burst_rate: float = 1.0
    mean_background_s: float = 20.0
    mean_burst_s: float = 5.0

    def __post_init__(self) -> None:
        for name in ("background_rate", "burst_rate", "mean_background_s", "mean_burst_s"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True)
class SyntheticConfig:
    """合成 trace 的全部可调参数。"""

    num_sessions: int = 100
    turns_per_session: int = 6
    session_arrival_rate: float = 0.5
    think_time_mu: float = 2.0
    think_time_sigma: float = 1.0
    system_tokens_mean: float = 800.0
    system_tokens_std: float = 0.0
    tools_tokens_mean: float = 1000.0
    tools_tokens_std: float = 0.0
    new_tokens_mean: float = 150.0
    new_tokens_std: float = 60.0
    output_tokens_mean: float = 350.0
    output_tokens_std: float = 150.0
    agent_mix: Mapping[str, float] = field(
        default_factory=lambda: {"coding": 0.7, "search": 0.3}
    )
    priority_mix: Mapping[int, float] = field(default_factory=lambda: {1: 1.0})
    # 按 agent_type 覆盖全局参数；出现在 agent_mix 但无 profile 的类型用全局值
    agent_profiles: Mapping[str, AgentProfile] = field(default_factory=dict)
    # 预排下轮到达所用的解析式服务时间估计（独立于 ServingConfig 的粗略假设）
    est_prefill_tps: float = 5000.0
    est_decode_tps: float = 200.0
    # 工作流负载模式（M5）：非 None 时切换到流式生成路径
    workflow: "WorkflowConfig | None" = None
    # 突发到达模式（M5-B）：非 None 时会话到达改为两态 MMPP
    mmpp: "MMPPConfig | None" = None

    def __post_init__(self) -> None:
        if self.num_sessions <= 0:
            raise ValueError(f"num_sessions must be positive, got {self.num_sessions}")
        if self.turns_per_session <= 0:
            raise ValueError(f"turns_per_session must be positive, got {self.turns_per_session}")
        if self.session_arrival_rate <= 0:
            raise ValueError(f"session_arrival_rate must be positive, got {self.session_arrival_rate}")
        for name in ("system_tokens", "tools_tokens", "new_tokens", "output_tokens"):
            mean = getattr(self, f"{name}_mean")
            std = getattr(self, f"{name}_std")
            if mean < 0 or std < 0:
                raise ValueError(f"{name}_mean/_std must be non-negative")
        if not self.agent_mix or sum(self.agent_mix.values()) <= 0:
            raise ValueError("agent_mix must be non-empty with positive weights")
        if not self.priority_mix or sum(self.priority_mix.values()) <= 0:
            raise ValueError("priority_mix must be non-empty with positive weights")
        known_types = set(self.agent_mix)
        if self.workflow is not None:
            for source, targets in self.workflow.transitions.items():
                if not targets or sum(targets.values()) <= 0:
                    raise ValueError(f"transitions[{source!r}] must have positive weights")
                known_types.add(source)
                known_types.update(targets)
        unknown = set(self.agent_profiles) - known_types
        if unknown:
            raise ValueError(f"agent_profiles for unknown agent types: {sorted(unknown)}")
        if self.est_prefill_tps <= 0 or self.est_decode_tps <= 0:
            raise ValueError("est_prefill_tps and est_decode_tps must be positive")

    def profile(self, agent_type: str) -> AgentProfile:
        """取该类型的覆盖（无则返回空 profile，即全部沿用全局）。"""
        return self.agent_profiles.get(agent_type, AgentProfile())

    def value(self, agent_type: str, name: str) -> float:
        """解析某参数在该类型下的生效值（profile 覆盖优先）。"""
        override = getattr(self.profile(agent_type), name)
        return override if override is not None else getattr(self, name)


def _session_arrival_draws(config: SyntheticConfig, rng: random.Random) -> list[float]:
    """生成 num_sessions 个会话到达时刻：泊松或两态 MMPP。"""
    times: list[float] = []
    clock = 0.0
    if config.mmpp is None:
        for _ in range(config.num_sessions):
            clock += rng.expovariate(config.session_arrival_rate)
            times.append(clock)
        return times
    mmpp = config.mmpp
    in_burst = False
    while len(times) < config.num_sessions:
        rate = mmpp.burst_rate if in_burst else mmpp.background_rate
        gap = rng.expovariate(rate)
        exit_rate = 1.0 / (mmpp.mean_burst_s if in_burst else mmpp.mean_background_s)
        sojourn = rng.expovariate(exit_rate)
        if sojourn < gap:
            # 持留先于下一次到达结束：先推进到状态切换点再继续抽取
            clock += sojourn
            in_burst = not in_burst
            continue
        clock += gap
        times.append(clock)
    return times


def generate_trace(config: SyntheticConfig, seed: int) -> list[TraceRequest]:
    """按配置与种子生成请求列表（同 seed 输出逐字节一致）。"""
    if config.workflow is not None:
        return _generate_workflow_trace(config, seed)
    rng = random.Random(seed)
    agent_types = list(config.agent_mix)
    agent_weights = [config.agent_mix[name] for name in agent_types]
    priorities = list(config.priority_mix)
    priority_weights = [config.priority_mix[p] for p in priorities]
    session_arrivals = _session_arrival_draws(config, rng)

    # 每个 agent_type 一次性抽取前导长度：同类型会话共享（模拟同一应用的固定 prompt）
    preamble_tokens: dict[str, tuple[int, int]] = {
        agent_type: (
            _draw_tokens(rng, config.value(agent_type, "system_tokens_mean"), config.value(agent_type, "system_tokens_std")),
            _draw_tokens(rng, config.value(agent_type, "tools_tokens_mean"), config.value(agent_type, "tools_tokens_std")),
        )
        for agent_type in agent_types
    }

    requests: list[TraceRequest] = []
    for index in range(config.num_sessions):
        session_clock = session_arrivals[index]
        agent_type = rng.choices(agent_types, weights=agent_weights, k=1)[0]
        priority = rng.choices(priorities, weights=priority_weights, k=1)[0]
        session_id = f"sess_{index:04d}"
        system_len, tools_len = preamble_tokens[agent_type]
        history = 0
        arrival = session_clock
        prev_est_end = session_clock
        for turn in range(1, config.turns_per_session + 1):
            if turn > 1:
                think_time = rng.lognormvariate(
                    config.value(agent_type, "think_time_mu"),
                    config.value(agent_type, "think_time_sigma"),
                )
                arrival = prev_est_end + think_time
            else:
                think_time = 0.0
            new_tokens = _draw_tokens(
                rng, config.value(agent_type, "new_tokens_mean"), config.value(agent_type, "new_tokens_std")
            )
            output_tokens = _draw_tokens(
                rng, config.value(agent_type, "output_tokens_mean"), config.value(agent_type, "output_tokens_std")
            )
            prompt = PromptBreakdown(
                system=system_len, tools=tools_len, history=history, new=new_tokens
            )
            requests.append(
                TraceRequest(
                    session_id=session_id,
                    turn_id=turn,
                    arrival_time=arrival,
                    prompt=prompt,
                    output_tokens=output_tokens,
                    think_time=think_time,
                    agent_type=agent_type,
                    priority=priority,
                )
            )
            history += new_tokens + output_tokens
            est_service = (
                prompt.total / config.est_prefill_tps
                + output_tokens / config.est_decode_tps
            )
            prev_est_end = arrival + est_service
    return requests


def _draw_tokens(rng: random.Random, mean: float, std: float) -> int:
    """正态抽取 token 数并截断到非负整数。"""
    return max(0, round(rng.gauss(mean, std)))


def _generate_workflow_trace(config: SyntheticConfig, seed: int) -> list[TraceRequest]:
    """工作流负载：根会话泊松到达，流内共享前导，按转移矩阵派生子会话。"""
    workflow = config.workflow
    rng = random.Random(seed)
    agent_types = list(config.agent_mix)
    agent_weights = [config.agent_mix[name] for name in agent_types]
    priorities = list(config.priority_mix)
    priority_weights = [config.priority_mix[p] for p in priorities]
    requests: list[TraceRequest] = []
    session_counter = 0

    def _spawn_session(
        session_id: str,
        agent_type: str,
        flow_id: str,
        parent_session: str | None,
        start_time: float,
        turns: int,
        first_think: float,
        system_len: int,
        tools_len: int,
    ) -> list[tuple[float, str]]:
        """生成一个会话的全部轮次；返回 (轮完成估计时刻, agent类型) 供派生。"""
        priority = rng.choices(priorities, weights=priority_weights, k=1)[0]
        history = 0
        arrival = start_time
        prev_est_end = start_time
        turn_ends: list[tuple[float, str]] = []
        for turn in range(1, turns + 1):
            if turn > 1:
                think_time = rng.lognormvariate(
                    config.value(agent_type, "think_time_mu"),
                    config.value(agent_type, "think_time_sigma"),
                )
                arrival = prev_est_end + think_time
            else:
                think_time = first_think
                if first_think > 0:
                    arrival = prev_est_end + first_think
            new_tokens = _draw_tokens(
                rng,
                config.value(agent_type, "new_tokens_mean"),
                config.value(agent_type, "new_tokens_std"),
            )
            output_tokens = _draw_tokens(
                rng,
                config.value(agent_type, "output_tokens_mean"),
                config.value(agent_type, "output_tokens_std"),
            )
            prompt = PromptBreakdown(
                system=system_len, tools=tools_len, history=history, new=new_tokens
            )
            requests.append(
                TraceRequest(
                    session_id=session_id,
                    turn_id=turn,
                    arrival_time=arrival,
                    prompt=prompt,
                    output_tokens=output_tokens,
                    think_time=think_time,
                    agent_type=agent_type,
                    priority=priority,
                    flow_id=flow_id,
                    parent_session=parent_session,
                )
            )
            history += new_tokens + output_tokens
            est_service = (
                prompt.total / config.est_prefill_tps
                + output_tokens / config.est_decode_tps
            )
            prev_est_end = arrival + est_service
            turn_ends.append((prev_est_end, agent_type))
        return turn_ends

    def _child_types(source: str, count: int) -> list[str]:
        targets = workflow.transitions.get(source)  # 缺省回落到 agent_mix
        if targets:
            names = list(targets)
            weights = [targets[name] for name in names]
        else:
            names, weights = agent_types, agent_weights
        return rng.choices(names, weights=weights, k=count)

    session_clock = 0.0
    for index in range(config.num_sessions):
        session_clock += rng.expovariate(config.session_arrival_rate)
        agent_type = rng.choices(agent_types, weights=agent_weights, k=1)[0]
        # 流级共享前导：一次抽取，流内全部成员相同（跨 agent 类型复用）
        system_len = _draw_tokens(rng, config.system_tokens_mean, config.system_tokens_std)
        tools_len = _draw_tokens(rng, config.tools_tokens_mean, config.tools_tokens_std)
        flow_id = f"flow_{index:04d}"
        root_id = f"sess_{session_counter:04d}"
        session_counter += 1
        turn_ends = _spawn_session(
            root_id, agent_type, flow_id, None, session_clock,
            config.turns_per_session, 0.0, system_len, tools_len,
        )
        child_count = rng.randint(1, max(1, workflow.children_per_flow))
        spawn_points = rng.sample(turn_ends, k=min(child_count, len(turn_ends)))
        for child_index, (end_time, source_type) in enumerate(spawn_points):
            child_type = _child_types(source_type, 1)[0]
            delay = rng.lognormvariate(workflow.child_delay_mu, workflow.child_delay_sigma)
            child_id = f"sess_{session_counter:04d}"
            session_counter += 1
            child_ends = _spawn_session(
                child_id, child_type, flow_id, root_id, end_time,
                workflow.child_turns, delay, system_len, tools_len,
            )
            # 二级派生（孙会话）：概率触发，仍计入流内
            if rng.random() < workflow.grandchild_prob and child_ends:
                grand_type = _child_types(child_type, 1)[0]
                grand_spawn = rng.choice(child_ends)
                grand_delay = rng.lognormvariate(
                    workflow.child_delay_mu, workflow.child_delay_sigma
                )
                grand_id = f"sess_{session_counter:04d}"
                session_counter += 1
                _spawn_session(
                    grand_id, grand_type, flow_id, child_id, grand_spawn[0],
                    max(2, workflow.child_turns - 1), grand_delay,
                    system_len, tools_len,
                )

    requests.sort(key=lambda r: r.arrival_time)
    return requests
