"""
功耗模型

设计约束:
- Per-bank: < 1mW
- 系统级 (512 banks): ~300mW
"""

from dataclasses import dataclass


@dataclass
class PowerConfig:
    """功耗配置"""
    
    # 28nm 工艺参数
    process_node_nm: int = 28
    supply_voltage: float = 0.9  # V
    
    # Per-bank 功耗估算 (mW)
    addr_gen_power: float = 0.1    # 地址生成器
    tile_buffer_power: float = 0.05  # Tile buffer
    mac_unit_power: float = 0.3    # 8-lane FP16 MAC
    control_power: float = 0.05    # 控制逻辑
    
    # 静态功耗
    leakage_per_gate: float = 1e-6  # mW/gate @ 28nm
    gates_per_bank: int = 9500
    
    @property
    def per_bank_dynamic(self) -> float:
        """Per-bank 动态功耗 (mW)"""
        return (self.addr_gen_power + self.tile_buffer_power + 
                self.mac_unit_power + self.control_power)
    
    @property
    def per_bank_static(self) -> float:
        """Per-bank 静态功耗 (mW)"""
        return self.leakage_per_gate * self.gates_per_bank


class PowerModel:
    """
    GeoPIM 功耗模型
    
    估算:
    - Per-bank: ~0.5mW 动态 + ~0.01mW 静态
    - 系统级 (512 banks): ~256mW 动态 + ~5mW 静态
    """
    
    def __init__(self, config: PowerConfig = None):
        self.config = config or PowerConfig()
        
    def get_per_bank_power(self, activity_factor: float = 1.0) -> float:
        """
        获取单 bank 功耗 (mW)
        
        Args:
            activity_factor: 活动因子 (0.0 ~ 1.0)
            
        Returns:
            功耗 (mW)
        """
        dynamic = self.config.per_bank_dynamic * activity_factor
        static = self.config.per_bank_static
        return dynamic + static
    
    def get_system_power(
        self, 
        num_banks: int = 512, 
        activity_factor: float = 1.0
    ) -> float:
        """
        获取系统级功耗 (mW)
        
        Args:
            num_banks: 活跃 bank 数
            activity_factor: 活动因子
            
        Returns:
            功耗 (mW)
        """
        per_bank = self.get_per_bank_power(activity_factor)
        return per_bank * num_banks
    
    def estimate_energy(
        self,
        num_banks: int,
        latency_ms: float,
        activity_factor: float = 0.8
    ) -> float:
        """
        估算能耗 (mJ)
        
        Args:
            num_banks: 活跃 bank 数
            latency_ms: 执行延迟 (ms)
            activity_factor: 活动因子
            
        Returns:
            能耗 (mJ)
        """
        power_mw = self.get_system_power(num_banks, activity_factor)
        return power_mw * latency_ms / 1000  # mW * ms / 1000 = mJ
    
    def get_power_breakdown(self, num_banks: int = 512) -> dict:
        """
        获取功耗分解
        
        Args:
            num_banks: bank 数
            
        Returns:
            功耗分解 dict
        """
        return {
            'per_bank': {
                'addr_gen': self.config.addr_gen_power,
                'tile_buffer': self.config.tile_buffer_power,
                'mac_unit': self.config.mac_unit_power,
                'control': self.config.control_power,
                'static': self.config.per_bank_static,
                'total': self.get_per_bank_power(),
            },
            'system': {
                'dynamic': self.config.per_bank_dynamic * num_banks,
                'static': self.config.per_bank_static * num_banks,
                'total': self.get_system_power(num_banks),
            },
            'constraints': {
                'per_bank_limit': 1.0,  # mW
                'per_bank_actual': self.get_per_bank_power(),
                'meets_constraint': self.get_per_bank_power() < 1.0,
            }
        }

