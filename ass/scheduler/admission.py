"""可插拔准入策略（M6，对应 FR-17；动机见 exp010 的瓶颈迁移发现）。

极端突发负载下，并发准入（排队）取代缓存管理成为 JCT 主导项，而基线
队列语义是最朴素的 FIFO。本模块把"下一个准入谁"做成策略：

- :class:`FIFOAdmission`（默认）：只提供队头候选，**保持队头阻塞语义**
  （队头被缓存容量阻塞时后续候选不被尝试）——与旧实现逐位一致；
- :class:`PriorityAdmission`：按类权重排序（高权重先），同权重按到达序；
  低权重类在持续高压下可能饥饿，风险如实交给实验度量；
- :class:`ShortestJobAdmission`：按已知工作量（prompt+output token 数）
  从小到大——max_tokens 在真实系统中同样先验可得，排队论经典结论
  （SJF 最小化平均等待）直接适用。

非 FIFO 策略返回完整偏好序，被缓存容量阻塞的候选自然被跳过（后续
候选仍可准入），这是与基线的语义差异，由实验（exp011）度量其影响。
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
