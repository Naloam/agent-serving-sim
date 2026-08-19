"""Radix tree KV cache 模型（M1 实现，对应 PRD FR-5）。

规划内容：节点为前缀段并记录 token 数；``match`` 前缀命中、``insert`` 插入、
按 token 数计费容量；被在途请求引用的节点不可淘汰（引用计数保护）。
"""
