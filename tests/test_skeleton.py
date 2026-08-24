"""M0 骨架冒烟测试：主包与 规划的全部子模块必须可导入。"""

import importlib

import ass

PLANNED_SUBMODULES = [
    "ass.core.event",
    "ass.core.sim",
    "ass.workload.schema",
    "ass.workload.synthetic",
    "ass.workload.loaders",
    "ass.cache.radix",
    "ass.cache.policies",
    "ass.scheduler.serving",
    "ass.metrics.collector",
    "ass.viz.plots",
]


def test_package_exposes_version() -> None:
    """主包可导入且暴露非空版本号。"""
    assert isinstance(ass.__version__, str)
    assert ass.__version__


def test_all_planned_modules_importable() -> None:
    """§4.1 规划的全部模块均已存在且可导入。"""
    for module_name in PLANNED_SUBMODULES:
        importlib.import_module(module_name)
