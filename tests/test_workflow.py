"""M5 工作流负载的单元测试：schema 兼容、生成器、flow 前缀共享、转移策略。"""

from collections import defaultdict

import pytest

from ass.cache.policies import TransitionPolicy
from ass.cache.radix import NodeMeta, RadixTree, Segment
from ass.scheduler.serving import ServingConfig, ServingSim
from ass.workload.schema import (
    PromptBreakdown,
    TraceRequest,
    TraceValidationError,
    request_from_dict,
    request_to_dict,
    write_trace,
    read_trace,
)
from ass.workload.synthetic import SyntheticConfig, WorkflowConfig, generate_trace


def make_request(session: str, turn: int, arrival: float, system: int,
                 new: int, output: int, agent: str = "coding",
                 flow: str | None = None, parent: str | None = None) -> TraceRequest:
    return TraceRequest(
        session_id=session, turn_id=turn, arrival_time=arrival,
        prompt=PromptBreakdown(system=system, tools=0, history=0, new=new),
        output_tokens=output, think_time=0.0, agent_type=agent, priority=1,
        flow_id=flow, parent_session=parent,
    )


# ---- schema：可选字段向后兼容 ----

def test_optional_fields_default_none_and_old_format_parses() -> None:
    legacy = {
        "session_id": "s1", "turn_id": 1, "arrival_time": 0.0,
        "prompt": {"system": 10, "tools": 0, "history": 0, "new": 5},
        "output_tokens": 3, "think_time": 0.0,
        "agent_type": "coding", "priority": 1,
    }
    request = request_from_dict(legacy)
    assert request.flow_id is None and request.parent_session is None
    # 往返后仍可读，可选字段显式为 null
    assert request_from_dict(request_to_dict(request)) == request


def test_optional_fields_roundtrip_with_values() -> None:
    request = make_request("s1", 1, 0.0, 10, 5, 3, flow="flow_0", parent="s0")
    restored = request_from_dict(request_to_dict(request))
    assert restored.flow_id == "flow_0" and restored.parent_session == "s0"


def test_optional_fields_type_validated() -> None:
    good = request_to_dict(make_request("s1", 1, 0.0, 10, 5, 3))
    with pytest.raises(TraceValidationError, match="flow_id"):
        request_from_dict({**good, "flow_id": 42})


# ---- 工作流生成器 ----

WORKFLOW = WorkflowConfig(
    transitions={"orchestrator": {"coder": 0.6, "searcher": 0.4},
                 "coder": {"critic": 1.0}},
    children_per_flow=2, child_turns=3, grandchild_prob=0.5,
)

WF_CONFIG = SyntheticConfig(
    num_sessions=12, turns_per_session=5, session_arrival_rate=0.5,
    agent_mix={"orchestrator": 1.0},
    workflow=WORKFLOW,
)


def test_workflow_trace_structure() -> None:
    trace = generate_trace(WF_CONFIG, seed=42)
    assert len(trace) > WF_CONFIG.num_sessions * WF_CONFIG.turns_per_session  # 含子会话
    flows: dict[str, set] = defaultdict(set)
    parents: dict[str, str] = {}
    for request in trace:
        assert request.flow_id is not None
        flows[request.flow_id].add(request.agent_type)
        if request.parent_session:
            parents[request.session_id] = request.parent_session
    # 流内有派生（子会话存在），且流内跨类型共享前导
    assert parents, "workflow 模式应产生派生会话"
    some_flow = next(iter(flows.values()))
    preamble = {}
    for request in trace:
        key = request.flow_id
        preamble.setdefault(key, request.prompt.system + request.prompt.tools)
        assert preamble[key] == request.prompt.system + request.prompt.tools
    assert len(flows) == WF_CONFIG.num_sessions


def test_workflow_children_types_follow_transitions() -> None:
    trace = generate_trace(WF_CONFIG, seed=42)
    by_session: dict[str, list] = defaultdict(list)
    for request in trace:
        by_session[request.session_id].append(request)
    # transitions 未覆盖的父类型按设计回落到 agent_mix（此处只有 orchestrator）
    allowed = {"orchestrator": {"coder", "searcher"},
               "coder": {"critic"},
               "searcher": {"orchestrator"},
               "critic": {"orchestrator"}}
    for session, requests in by_session.items():
        agent = requests[0].agent_type
        for request in requests:
            if request.parent_session:
                parent_agent = by_session[request.parent_session][0].agent_type
                assert agent in allowed[parent_agent], (parent_agent, agent)


def test_workflow_reproducible(tmp_path) -> None:
    path_a, path_b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write_trace(generate_trace(WF_CONFIG, seed=42), path_a)
    write_trace(generate_trace(WF_CONFIG, seed=42), path_b)
    assert path_a.read_bytes() == path_b.read_bytes()


# ---- flow 前缀共享（serving 集成） ----

def test_flow_members_share_preamble_across_agent_types() -> None:
    """同流不同 agent 类型的会话，前导互相命中（flow: 流）。"""
    first = make_request("s1", 1, 0.0, system=500, new=100, output=50,
                         agent="orchestrator", flow="flow_0")
    second = make_request("s2", 1, 10.0, system=500, new=80, output=40,
                          agent="coder", flow="flow_0", parent="s1")
    sim = ServingSim(ServingConfig(cache_capacity_tokens=100_000))
    sim.submit_all([first, second])
    sim.run()
    assert sim.collector.records[1].hit_tokens == 500


# ---- TransitionPolicy ----

def test_transition_policy_learns_and_ranks_by_hop() -> None:
    policy = TransitionPolicy(active_window_s=3.0)
    # 学习：orchestrator 派生 coder；coder 派生 critic
    policy.on_admit(make_request("root", 1, 0.0, 10, 5, 3, agent="orchestrator"), 0.0)
    policy.on_admit(make_request("c1", 1, 1.0, 10, 5, 3, agent="coder", parent="root"), 1.0)
    policy.on_admit(make_request("g1", 1, 2.0, 10, 5, 3, agent="critic", parent="c1"), 2.0)
    assert policy._transitions == {"orchestrator": {"coder": 1}, "coder": {"critic": 1}}
    # 刷新 coder 的活跃度：now=11 时前沿只有 coder（窗口 3s，root/g1 已出沿）
    policy.on_admit(make_request("c1", 2, 10.0, 10, 5, 3, agent="coder"), 10.0)

    tree = RadixTree(capacity_tokens=100000)
    tree.insert([Segment("sess:c1", 50)], now=10.5, meta=NodeMeta(agent_type="coder"))
    tree.insert([Segment("sess:g1", 50)], now=10.5, meta=NodeMeta(agent_type="critic"))
    tree.insert([Segment("sess:batch", 50)], now=10.5, meta=NodeMeta(agent_type="batch"))
    # 跳距 coder=0, critic=1（coder 边）；batch 未知 2.5 → 先驱逐
    victims = policy.select_victims(tree, 150, now=11.0)
    order = [node.segment.stream for node in victims]
    assert order == ["sess:batch", "sess:g1", "sess:c1"]


def test_transition_policy_preamble_flow_kept_last() -> None:
    policy = TransitionPolicy()
    policy.on_admit(make_request("s1", 1, 0.0, 10, 5, 3, agent="coder"), 0.0)
    tree = RadixTree(capacity_tokens=100000)
    tree.insert([Segment("flow:f0", 500)], now=1.0, meta=NodeMeta(agent_type="coder"))
    tree.insert([Segment("sess:s1", 50)], now=1.0, meta=NodeMeta(agent_type="coder"))
    victims = policy.select_victims(tree, 550, now=2.0)
    assert victims[0].segment.stream == "sess:s1"


def test_transition_policy_without_workflow_falls_back_to_lru() -> None:
    policy = TransitionPolicy()
    tree = RadixTree(capacity_tokens=10000)
    tree.insert([Segment("sess:old", 50)], now=1.0, meta=NodeMeta(agent_type="coder"))
    tree.insert([Segment("sess:new", 50)], now=9.0, meta=NodeMeta(agent_type="coder"))
    victims = policy.select_victims(tree, 100, now=10.0)
    assert victims[0].segment.stream == "sess:old"
