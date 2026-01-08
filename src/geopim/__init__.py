"""
GeoPIM v3.0: HBM-PIM 几何引导采样加速器

基于 HBM 存内计算的 3DGS 前馈生成 Encoder 加速模拟器。
"""

__version__ = "3.0.0"

# 延迟导入以避免循环依赖
def __getattr__(name):
    if name == "HBMConfig":
        from .simulator.hbm_model import HBMConfig
        return HBMConfig
    elif name == "HBMModel":
        from .simulator.hbm_model import HBMModel
        return HBMModel
    elif name == "AccessStats":
        from .simulator.hbm_model import AccessStats
        return AccessStats
    elif name == "PIMUnitConfig":
        from .simulator.pim_unit import PIMUnitConfig
        return PIMUnitConfig
    elif name == "PIMUnit":
        from .simulator.pim_unit import PIMUnit
        return PIMUnit
    elif name == "GeometryAddressGenerator":
        from .simulator.pim_unit import GeometryAddressGenerator
        return GeometryAddressGenerator
    elif name == "ComputeUnit":
        from .simulator.pim_unit import ComputeUnit
        return ComputeUnit
    elif name == "PIMController":
        from .simulator.controller import PIMController
        return PIMController
    elif name == "GeoPIMSimulator":
        from .simulator.geopim_simulator import GeoPIMSimulator
        return GeoPIMSimulator
    elif name == "TimingConfig":
        from .timing.cycle_model import TimingConfig
        return TimingConfig
    elif name == "CycleModel":
        from .timing.cycle_model import CycleModel
        return CycleModel
    elif name == "PowerModel":
        from .timing.power_model import PowerModel
        return PowerModel
    elif name == "BankScheduler":
        from .utils.bank_scheduler import BankScheduler
        return BankScheduler
    elif name == "GeoPIMSampler":
        from .interface.geopim_sampler import GeoPIMSampler
        return GeoPIMSampler
    elif name == "GeoPIMSamplerModule":
        from .interface.geopim_sampler import GeoPIMSamplerModule
        return GeoPIMSamplerModule
    raise AttributeError(f"module 'geopim' has no attribute '{name}'")

__all__ = [
    # Simulator
    "HBMConfig",
    "HBMModel", 
    "AccessStats",
    "PIMUnitConfig",
    "PIMUnit",
    "GeometryAddressGenerator",
    "ComputeUnit",
    "PIMController",
    "GeoPIMSimulator",
    # Timing
    "TimingConfig",
    "CycleModel",
    "PowerModel",
    # Utils
    "BankScheduler",
    # Interface
    "GeoPIMSampler",
    "GeoPIMSamplerModule",
]
