"""
GeoPIM 模拟器模块

包含:
- HBMModel: HBM3 结构和访问模型
- PIMUnit: Per-Bank PIM 计算单元
- PIMController: 多 Bank 控制器
- GeoPIMSimulator: 完整模拟器
"""

from .hbm_model import HBMConfig, HBMModel, AccessStats
from .pim_unit import PIMUnitConfig, PIMUnit, GeometryAddressGenerator, ComputeUnit, ParamEntry
from .controller import PIMController
from .geopim_simulator import GeoPIMSimulator

__all__ = [
    "HBMConfig",
    "HBMModel",
    "AccessStats",
    "PIMUnitConfig",
    "PIMUnit",
    "GeometryAddressGenerator",
    "ComputeUnit",
    "ParamEntry",
    "PIMController",
    "GeoPIMSimulator",
]

