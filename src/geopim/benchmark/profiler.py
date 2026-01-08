"""
GeoPIM 性能分析器

提供详细的性能分析和可视化。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import numpy as np

from ..timing.cycle_model import CycleModel, TimingConfig
from ..timing.power_model import PowerModel
from ..simulator.hbm_model import HBMModel, HBMConfig


@dataclass
class ProfileResult:
    """分析结果"""
    
    # 时序
    total_cycles: int
    latency_ms: float
    throughput_samples_per_sec: float
    
    # 内存
    row_hit_rate: float
    total_bytes_accessed: int
    bandwidth_utilization: float
    
    # 功耗
    dynamic_power_mw: float
    static_power_mw: float
    total_energy_mj: float
    
    # 加速比
    speedup_vs_gpu: float


class GeoPIMProfiler:
    """
    GeoPIM 性能分析器
    
    提供:
    - 时序分析
    - 内存带宽分析
    - 功耗分析
    - 敏感性分析
    """
    
    def __init__(
        self,
        hbm_config: Optional[HBMConfig] = None,
        timing_config: Optional[TimingConfig] = None
    ):
        self.hbm_model = HBMModel(hbm_config)
        self.timing_config = timing_config or TimingConfig()
        self.cycle_model = CycleModel(self.hbm_model, self.timing_config)
        self.power_model = PowerModel()
        
    def profile_workload(
        self,
        batch_size: int = 2,
        num_queries: int = 1024,
        num_samples: int = 512,
        num_views: int = 4,
        num_banks: int = 512,
        row_hit_rate: float = 0.7,
        gpu_baseline_ms: float = 19.65
    ) -> ProfileResult:
        """
        分析工作负载
        
        Args:
            batch_size: Batch 大小
            num_queries: Query 数量
            num_samples: 每 query 采样点数
            num_views: View 数量
            num_banks: 活跃 bank 数
            row_hit_rate: Row hit rate
            gpu_baseline_ms: GPU baseline 延迟
            
        Returns:
            ProfileResult
        """
        total_samples = batch_size * num_queries * num_samples * num_views
        
        # 时序分析
        sample_cycles = self.cycle_model.estimate_sample_cycles(row_hit_rate)
        throughput = self.cycle_model.estimate_throughput(num_banks, row_hit_rate)
        latency_ms = self.cycle_model.estimate_latency(total_samples, num_banks, row_hit_rate)
        total_cycles = int(sample_cycles * total_samples / num_banks)
        
        # 内存分析
        # 每 sample 访问: 4 neighbors × 16 tiles × 64B burst = 4KB
        bytes_per_sample = 4 * 16 * 64
        total_bytes = total_samples * bytes_per_sample
        
        # 带宽利用率
        # 理想带宽: HBM3 内部 ~8 TB/s
        ideal_bandwidth = self.hbm_model.config.internal_bandwidth_gbps * 1e9 / 8  # bytes/s
        actual_bandwidth = total_bytes / (latency_ms / 1000)
        bandwidth_utilization = actual_bandwidth / ideal_bandwidth
        
        # 功耗分析
        power_breakdown = self.power_model.get_power_breakdown(num_banks)
        dynamic_power = power_breakdown['system']['dynamic']
        static_power = power_breakdown['system']['static']
        energy = self.power_model.estimate_energy(num_banks, latency_ms)
        
        # 加速比
        speedup = gpu_baseline_ms / latency_ms
        
        return ProfileResult(
            total_cycles=total_cycles,
            latency_ms=latency_ms,
            throughput_samples_per_sec=throughput,
            row_hit_rate=row_hit_rate,
            total_bytes_accessed=total_bytes,
            bandwidth_utilization=bandwidth_utilization,
            dynamic_power_mw=dynamic_power,
            static_power_mw=static_power,
            total_energy_mj=energy,
            speedup_vs_gpu=speedup,
        )
    
    def sensitivity_analysis(
        self,
        param_name: str,
        param_values: List,
        **base_kwargs
    ) -> Dict:
        """
        参数敏感性分析
        
        Args:
            param_name: 参数名 ('row_hit_rate', 'num_banks', etc.)
            param_values: 参数值列表
            **base_kwargs: 基础配置
            
        Returns:
            {param_value: ProfileResult}
        """
        results = {}
        
        for value in param_values:
            kwargs = base_kwargs.copy()
            kwargs[param_name] = value
            results[value] = self.profile_workload(**kwargs)
        
        return results
    
    def row_hit_rate_sweep(
        self,
        hit_rates: List[float] = None,
        **kwargs
    ) -> Dict:
        """Row hit rate 敏感性分析"""
        if hit_rates is None:
            hit_rates = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        return self.sensitivity_analysis('row_hit_rate', hit_rates, **kwargs)
    
    def bank_count_sweep(
        self,
        bank_counts: List[int] = None,
        **kwargs
    ) -> Dict:
        """Bank 数量敏感性分析"""
        if bank_counts is None:
            bank_counts = [128, 256, 512, 768, 1024]
        return self.sensitivity_analysis('num_banks', bank_counts, **kwargs)
    
    def generate_summary(self, result: ProfileResult) -> str:
        """生成分析摘要"""
        return f"""
## GeoPIM 性能分析摘要

### 时序
- 总周期数: {result.total_cycles:,}
- 延迟: {result.latency_ms:.2f} ms
- 吞吐: {result.throughput_samples_per_sec/1e6:.2f} M samples/sec

### 内存
- Row Hit Rate: {result.row_hit_rate:.1%}
- 总访问字节: {result.total_bytes_accessed/1e6:.1f} MB
- 带宽利用率: {result.bandwidth_utilization:.1%}

### 功耗
- 动态功耗: {result.dynamic_power_mw:.1f} mW
- 静态功耗: {result.static_power_mw:.1f} mW
- 总能耗: {result.total_energy_mj:.3f} mJ

### 加速比
- vs GPU: **{result.speedup_vs_gpu:.2f}×**
"""

