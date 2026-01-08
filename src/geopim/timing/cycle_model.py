"""
周期精确时序模型

每 tile 时序 (tile_C=8, C=128 → 16 tiles/sample):
- DRAM Fetch: 4-20 cycles (row hit: 4, miss: 20)
- Bilinear: 4 cycles (4 neighbors × 1 cycle)
- Accumulate: 1 cycle

Ping-Pong 隐藏延迟后:
- 理想 (100% row hit): ~5 cycles/tile
- 典型 (70% row hit): ~9 cycles/tile
"""

from dataclasses import dataclass
from typing import Optional

from ..simulator.hbm_model import HBMModel, HBMConfig


@dataclass
class TimingConfig:
    """时序配置"""
    
    pim_freq_mhz: int = 300
    tile_c: int = 8
    total_c: int = 128
    
    # 计算延迟
    bilinear_cycles: int = 4   # 4 neighbors sequential
    accum_cycles: int = 1
    
    @property
    def tiles_per_sample(self) -> int:
        """每采样点 tile 数"""
        return self.total_c // self.tile_c
    
    @property
    def compute_cycles_per_tile(self) -> int:
        """每 tile 计算周期"""
        return self.bilinear_cycles + self.accum_cycles


class CycleModel:
    """
    周期精确模型
    
    用于分析不同 row hit rate 下的性能。
    """
    
    def __init__(
        self, 
        hbm_model: Optional[HBMModel] = None,
        config: Optional[TimingConfig] = None
    ):
        self.hbm = hbm_model or HBMModel(HBMConfig())
        self.config = config or TimingConfig()
        
    def estimate_sample_cycles(self, row_hit_rate: float) -> int:
        """
        估算单个 sample 的周期数
        
        Args:
            row_hit_rate: Row buffer 命中率 (0.0 ~ 1.0)
            
        Returns:
            周期数
        """
        # 平均 fetch 延迟
        avg_fetch = (row_hit_rate * self.hbm.config.row_hit_latency + 
                     (1 - row_hit_rate) * self.hbm.config.row_miss_latency)
        
        # 计算延迟
        compute = self.config.compute_cycles_per_tile
        
        # Ping-pong 隐藏延迟: 取 max
        tile_cycles = max(avg_fetch, compute)
        
        # 每 sample 总周期
        return int(self.config.tiles_per_sample * tile_cycles)
    
    def estimate_throughput(self, num_banks: int, row_hit_rate: float) -> float:
        """
        估算系统吞吐量
        
        Args:
            num_banks: 活跃 bank 数
            row_hit_rate: Row buffer 命中率
            
        Returns:
            samples/sec
        """
        sample_cycles = self.estimate_sample_cycles(row_hit_rate)
        single_bank = self.config.pim_freq_mhz * 1e6 / sample_cycles
        return single_bank * num_banks
    
    def estimate_latency(
        self, 
        total_samples: int, 
        num_banks: int, 
        row_hit_rate: float
    ) -> float:
        """
        估算总延迟
        
        Args:
            total_samples: 总采样点数
            num_banks: 活跃 bank 数
            row_hit_rate: Row buffer 命中率
            
        Returns:
            延迟 (ms)
        """
        throughput = self.estimate_throughput(num_banks, row_hit_rate)
        return (total_samples / throughput) * 1000
    
    def estimate_speedup(
        self,
        total_samples: int,
        num_banks: int,
        row_hit_rate: float,
        gpu_baseline_ms: float = 19.65
    ) -> float:
        """
        估算相对于 GPU baseline 的加速比
        
        Args:
            total_samples: 总采样点数
            num_banks: 活跃 bank 数
            row_hit_rate: Row buffer 命中率
            gpu_baseline_ms: GPU baseline 延迟 (ms)
            
        Returns:
            加速比
        """
        geopim_ms = self.estimate_latency(total_samples, num_banks, row_hit_rate)
        return gpu_baseline_ms / geopim_ms
    
    def analyze_sensitivity(
        self,
        total_samples: int,
        num_banks: int,
        hit_rates: list = None
    ) -> dict:
        """
        Row hit rate 敏感性分析
        
        Args:
            total_samples: 总采样点数
            num_banks: 活跃 bank 数
            hit_rates: 要分析的 hit rate 列表
            
        Returns:
            {hit_rate: speedup} 映射
        """
        if hit_rates is None:
            hit_rates = [0.3, 0.5, 0.7, 0.9]
            
        results = {}
        for hr in hit_rates:
            results[hr] = {
                'latency_ms': self.estimate_latency(total_samples, num_banks, hr),
                'throughput': self.estimate_throughput(num_banks, hr),
                'speedup': self.estimate_speedup(total_samples, num_banks, hr),
            }
        return results

