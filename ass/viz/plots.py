"""可视化绘图。

统一使用 Agg 后端（无显示环境依赖），产出 PNG 存放 ``experiments/results/``。
本模块允许使用 numpy / matplotlib（PRD NFR-1 的例外范围）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")

from matplotlib import pyplot as plt  # noqa: E402 须在设置后端后导入


def plot_cdf(
    series: Mapping[str, Sequence[float]],
    path: str | Path,
    *,
    title: str = "",
    xlabel: str = "value",
    ylabel: str = "cumulative fraction",
) -> Path:
    """多组数值的经验 CDF 对比图。"""
    _require_series(series)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, values in series.items():
        if not values:
            raise ValueError(f"series {label!r} is empty")
        xs = sorted(values)
        ys = [(i + 1) / len(xs) for i in range(len(xs))]
        ax.plot(xs, ys, drawstyle="steps-post", label=label)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, path)


def plot_timeline(
    times: Sequence[float],
    values: Sequence[float],
    path: str | Path,
    *,
    title: str = "",
    xlabel: str = "time (s)",
    ylabel: str = "value",
) -> Path:
    """时间线图（如显存占用曲线）。"""
    if len(times) != len(values):
        raise ValueError("times and values must have equal length")
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(times, values, drawstyle="steps-post")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    return _save(fig, path)


def plot_sweep(
    param_name: str,
    param_values: Sequence[float],
    series: Mapping[str, Sequence[float]],
    path: str | Path,
    *,
    title: str = "",
    ylabel: str = "metric",
) -> Path:
    """参数扫描曲线：x 为扫描参数，每组 label 一条折线。"""
    _require_series(series)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for label, values in series.items():
        if len(values) != len(param_values):
            raise ValueError(f"series {label!r} length mismatch with param_values")
        ax.plot(param_values, values, marker="o", label=label)
    ax.set_xlabel(param_name)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend()
    return _save(fig, path)


def _require_series(series: Mapping[str, Sequence[float]]) -> None:
    if not series:
        raise ValueError("series must not be empty")


def _save(fig, path: str | Path) -> Path:
    file_path = Path(path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(file_path, dpi=150)
    plt.close(fig)
    return file_path
