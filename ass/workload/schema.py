"""Trace 格式定义（M1 实现，对应 PRD FR-2）。

规划内容：与 PLAN.md §4.3 一致的请求 dataclass（prompt 四段分解、think_time 等）
及 JSONL 读写与字段校验（prompt 四段非负、think_time >= 0 等）。
"""
