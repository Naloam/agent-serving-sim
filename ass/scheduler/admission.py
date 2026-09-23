"""可插拔准入策略（M6/FR-17 + M7/FR-18；动机见 exp010/exp012）。

极端突发负载下，并发准入（排队）取代缓存管理成为 JCT 主导项，而基线
队列语义是最朴素的 FIFO。本模块把"下一个准入谁"做成策略：

- :class:`FIFOAdmission`（默认）：只提供队头候选，**保持队头阻塞语义**
  （队头被缓存容量阻塞时后续候选不被尝试）——与旧实现逐位一致；
- :class:`PriorityAdmission`：按类权重排序（高权重先），同权重按到达序；
  低权重类在持续高压下可能饥饿，风险如实交给实验度量；
- :class:`ShortestJobAdmission`：按已知工作量（prompt+output token 数）
  从小到大——max_tokens 在真实系统中同样先验可得，排队论经典结论
  （SJF 最小化平均等待）直接适用；
- :class:`SessionChainAdmission`（FR-18）：**续链优先**——已准入过的会话
  的后续轮次排在新会话首轮之前。exp012 的负结果驱动：朴素 SJF 前缀盲，
  在会话结构负载上打乱轮次顺序会使前缀在等待中冷死（命中率 0.88→0.34）；
  续链优先以到达序保持轮次局部性，并把"完成在途会话"置于"开新会话"
  之前（会话级最短剩余工作）。无已见会话时退化为纯 FIFO。

非 FIFO 策略返回完整偏好序，被缓存容量阻塞的候选自然被跳过（后续
候选仍可准入），这是与基线的语义差异，由实验（exp011/exp013）度量。

观察钩子：内核在准入成功/完成时回调 :meth:`AdmissionPolicy.on_admit` /
:meth:`AdmissionPolicy.on_complete`（默认 no-op），供在线策略维护状态。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Mapping, Sequence

from ass.workload.schema import TraceRequest


class AdmissionPolicy(ABC):
    """准入策略基类：给出当前等待队列的准入偏好序。"""

    name: str = "abstract"

    @abstractmethod
    def order(self, queue: Sequence[TraceRequest], now: float) -> list[TraceRequest]:
        """返回按准入偏好排序的候选列表（首元素最优先）。"""

    def on_admit(self, request: TraceRequest, now: float) -> None:
        """观察钩子：候选被成功准入时由内核回调（默认 no-op）。"""

    def on_complete(self, request: TraceRequest, now: float) -> None:
        """观察钩子：请求完成时由内核回调（默认 no-op）。"""


class FIFOAdmission(AdmissionPolicy):
    """先进先出：仅提供队头候选，保持队头阻塞语义（基线，与旧行为一致）。"""

    name = "fifo"

    def order(self, queue: Sequence[TraceRequest], now: float) -> list[TraceRequest]:
        return [queue[0]] if queue else []


class PriorityAdmission(AdmissionPolicy):
    """类权重优先：高权重类的等待请求先准入，同权重按到达序。

    饥饿风险不做掩盖：持续高压下低权重类可能长期排队，exp011 以
    每类 JCT/p95 度量该代价。
    """

    name = "priority"

    def __init__(self, weights: Mapping[str, float]) -> None:
        if not weights:
            raise ValueError("weights must be non-empty")
        self.weights = dict(weights)

    def order(self, queue: Sequence[TraceRequest], now: float) -> list[TraceRequest]:
        return sorted(
            queue,
            key=lambda request: (
                -self.weights.get(request.agent_type, 1.0),
                request.arrival_time,
            ),
        )


class ShortestJobAdmission(AdmissionPolicy):
    """短作业优先：按工作量（prompt 总 token + 输出 token）升序准入。

    输出长度取自 trace 的 ``output_tokens``；真实系统虽不知实际生成量，
    但 max_tokens 先验可得，同构于带上限的 SJF。
    """

    name = "sjf"

    def order(self, queue: Sequence[TraceRequest], now: float) -> list[TraceRequest]:
        return sorted(
            queue,
            key=lambda request: (
                request.prompt.total + request.output_tokens,
                request.arrival_time,
            ),
        )


class SessionChainAdmission(AdmissionPolicy):
    """续链优先（FR-18）：已开头会话的后续轮次先于新会话首轮。

    排序键：``(0 if 会话已准入过 else 1, -最近完成时刻, arrival_time)``。

    - 第一键把"完成在途会话"（前缀已建）排在"开新会话"之前——会话级
      最短剩余工作 + 前缀驻留保持；
    - 第二键（前缀热度）续链内部按**前驱完成时间最近优先**：刚完成的
      会话前缀最热，优先准入把驻留换成命中（hit → 吞吐是重排影响
      均值 JCT 的唯一通道，排队守恒律之下其余重排只搬运等待）。
      **在途未完成**的会话该键取 -inf、排在所有已续链之后——其前缀
      不完整（本轮输出仍在生成），提前准入下一轮会 miss 未完成部分
      并触发增长封顶路径，刻意压后是语义而非疏漏；
    - 第三键类内保持 FIFO，轮次局部性由到达序自然维持。

    状态经 :meth:`on_admit` / :meth:`on_complete` 观察维护，不窥探缓存
    内部；前缀被逐出视为概率事件（已见仍优先），误差交给实验度量。

    无会话结构负载（每请求独立会话）上所有候选同为"新会话"，退化为
    纯 FIFO，不劣于基线。内存：观察表随会话数单调增长，实验规模
    （万级会话）下可忽略；超长运行可按时间裁剪。
    """

    name = "session-chain"

    def __init__(self) -> None:
        self._admit_time: dict[str, float] = {}
        self._last_complete: dict[str, float] = {}

    def on_admit(self, request: TraceRequest, now: float) -> None:
        self._admit_time[request.session_id] = now

    def on_complete(self, request: TraceRequest, now: float) -> None:
        self._last_complete[request.session_id] = now

    def order(self, queue: Sequence[TraceRequest], now: float) -> list[TraceRequest]:
        admit_time = self._admit_time
        last_complete = self._last_complete
        return sorted(
            queue,
            key=lambda request: (
                0 if request.session_id in admit_time else 1,
                -last_complete.get(request.session_id, float("-inf")),
                request.arrival_time,
            ),
        )
