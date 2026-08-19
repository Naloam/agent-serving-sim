"""可视化绘图的单元测试（FR-9）：产出非空 PNG、参数校验。"""

import pytest

from ass.viz.plots import plot_cdf, plot_sweep, plot_timeline


def test_plot_cdf_writes_png(tmp_path) -> None:
    path = tmp_path / "cdf.png"
    result = plot_cdf(
        {"lru": [1.0, 2.0, 3.0], "ttl": [0.5, 1.5, 2.5]},
        path,
        title="JCT CDF",
        xlabel="JCT (s)",
    )
    assert result == path
    assert path.stat().st_size > 0


def test_plot_timeline_writes_png(tmp_path) -> None:
    path = tmp_path / "timeline.png"
    plot_timeline([0.0, 1.0, 2.0], [100, 300, 200], path, title="cache usage", ylabel="tokens")
    assert path.stat().st_size > 0


def test_plot_sweep_writes_png(tmp_path) -> None:
    path = tmp_path / "sweep.png"
    plot_sweep(
        "ttl (s)",
        [5, 10, 20],
        {"hit_rate": [0.2, 0.4, 0.6], "jct_mean": [3.0, 2.5, 2.0]},
        path,
        ylabel="metric",
    )
    assert path.stat().st_size > 0


def test_empty_inputs_rejected(tmp_path) -> None:
    with pytest.raises(ValueError, match="empty"):
        plot_cdf({}, tmp_path / "a.png")
    with pytest.raises(ValueError, match="empty"):
        plot_cdf({"x": []}, tmp_path / "b.png")
    with pytest.raises(ValueError, match="equal length"):
        plot_timeline([1.0], [1, 2], tmp_path / "c.png")
    with pytest.raises(ValueError, match="length mismatch"):
        plot_sweep("t", [1, 2], {"a": [1.0]}, tmp_path / "d.png")


def test_creates_parent_directories(tmp_path) -> None:
    path = tmp_path / "nested" / "dir" / "out.png"
    plot_timeline([0.0], [1.0], path)
    assert path.exists()
