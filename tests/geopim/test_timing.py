"""
时序模型单元测试

测试内容:
1. 周期计数准确性
2. Row hit rate 影响分析
3. 吞吐量估算
4. 加速比计算
"""

import pytest
from geopim.simulator.hbm_model import HBMConfig, HBMModel
from geopim.timing.cycle_model import TimingConfig, CycleModel
from geopim.timing.power_model import PowerModel


class TestTimingConfig:
    """时序配置测试"""
    
    def test_default_config(self):
        """测试默认时序配置"""
        config = TimingConfig()
        
        assert config.pim_freq_mhz == 300
        assert config.tile_c == 8
        assert config.total_c == 128
        assert config.bilinear_cycles == 4
        assert config.accum_cycles == 1
        
    def test_tiles_per_sample(self):
        """测试每采样点的 tile 数"""
        config = TimingConfig(tile_c=8, total_c=128)
        tiles = config.total_c // config.tile_c
        assert tiles == 16


class TestCycleModel:
    """周期模型测试"""
    
    def test_sample_cycles_100_hit_rate(self):
        """测试 100% row hit rate 的周期数"""
        hbm_config = HBMConfig()
        timing_config = TimingConfig()
        model = CycleModel(HBMModel(hbm_config), timing_config)
        
        # 100% hit rate: 每 tile 取 max(4, 5) = 5 cycles
        cycles = model.estimate_sample_cycles(row_hit_rate=1.0)
        
        # 16 tiles × 5 cycles = 80 cycles
        expected = 16 * 5
        assert cycles == expected
        
    def test_sample_cycles_0_hit_rate(self):
        """测试 0% row hit rate 的周期数"""
        hbm_config = HBMConfig()
        timing_config = TimingConfig()
        model = CycleModel(HBMModel(hbm_config), timing_config)
        
        # 0% hit rate: 每 tile 取 max(20, 5) = 20 cycles
        cycles = model.estimate_sample_cycles(row_hit_rate=0.0)
        
        # 16 tiles × 20 cycles = 320 cycles
        expected = 16 * 20
        assert cycles == expected
        
    def test_sample_cycles_typical_hit_rate(self):
        """测试典型 50% row hit rate 的周期数"""
        hbm_config = HBMConfig()
        timing_config = TimingConfig()
        model = CycleModel(HBMModel(hbm_config), timing_config)
        
        # 50% hit rate: avg_fetch = 0.5*4 + 0.5*20 = 12 cycles
        # 每 tile 取 max(12, 5) = 12 cycles
        cycles = model.estimate_sample_cycles(row_hit_rate=0.5)
        
        # 16 tiles × 12 cycles = 192 cycles
        expected = 16 * 12
        assert cycles == expected


class TestThroughput:
    """吞吐量测试"""
    
    def test_single_bank_throughput(self):
        """测试单 bank 吞吐量"""
        hbm_config = HBMConfig()
        timing_config = TimingConfig(pim_freq_mhz=300)
        model = CycleModel(HBMModel(hbm_config), timing_config)
        
        # 70% hit rate 时的单 bank 吞吐
        throughput = model.estimate_throughput(num_banks=1, row_hit_rate=0.7)
        
        # 预期: 300MHz / cycles_per_sample
        cycles = model.estimate_sample_cycles(0.7)
        expected = 300e6 / cycles
        
        assert abs(throughput - expected) < 1e3  # 允许小误差
        
    def test_multi_bank_throughput(self):
        """测试多 bank 并行吞吐量"""
        hbm_config = HBMConfig()
        timing_config = TimingConfig()
        model = CycleModel(HBMModel(hbm_config), timing_config)
        
        # 512 banks, 70% hit rate
        throughput = model.estimate_throughput(num_banks=512, row_hit_rate=0.7)
        
        single_throughput = model.estimate_throughput(num_banks=1, row_hit_rate=0.7)
        expected = single_throughput * 512
        
        assert abs(throughput - expected) < 1e6
        
    def test_transplat_latency(self):
        """测试 TransPlat 工作负载延迟"""
        hbm_config = HBMConfig()
        timing_config = TimingConfig()
        model = CycleModel(HBMModel(hbm_config), timing_config)
        
        # TransPlat: B=2, Q=1024, S=512 → 4M samples
        total_samples = 2 * 1024 * 512 * 4  # 考虑 4 个 view
        
        # 512 banks, 70% hit rate
        throughput = model.estimate_throughput(num_banks=512, row_hit_rate=0.7)
        
        latency_sec = total_samples / throughput
        latency_ms = latency_sec * 1000
        
        # 预期 3-5ms 范围内
        assert 2.0 < latency_ms < 10.0


class TestSpeedup:
    """加速比测试"""
    
    def test_speedup_vs_gpu_baseline(self):
        """测试相对于 GPU baseline 的加速比"""
        hbm_config = HBMConfig()
        timing_config = TimingConfig()
        model = CycleModel(HBMModel(hbm_config), timing_config)
        
        # GPU baseline: ~19.65ms (几何采样部分)
        gpu_baseline_ms = 19.65
        
        # GeoPIM 估算
        total_samples = 2 * 1024 * 512 * 4
        throughput = model.estimate_throughput(num_banks=512, row_hit_rate=0.7)
        geopim_ms = (total_samples / throughput) * 1000
        
        speedup = gpu_baseline_ms / geopim_ms
        
        # 预期加速比在 4-8x 范围内
        assert 3.0 < speedup < 10.0
        
    def test_speedup_sensitivity(self):
        """测试加速比对 row hit rate 的敏感性"""
        hbm_config = HBMConfig()
        timing_config = TimingConfig()
        model = CycleModel(HBMModel(hbm_config), timing_config)
        
        gpu_baseline_ms = 19.65
        total_samples = 2 * 1024 * 512 * 4
        
        speedups = []
        for hit_rate in [0.3, 0.5, 0.7, 0.9]:
            throughput = model.estimate_throughput(num_banks=512, row_hit_rate=hit_rate)
            geopim_ms = (total_samples / throughput) * 1000
            speedups.append(gpu_baseline_ms / geopim_ms)
        
        # 更高的 hit rate 应该带来更高的加速比
        assert speedups[0] < speedups[1] < speedups[2] < speedups[3]


class TestPowerModel:
    """功耗模型测试"""
    
    def test_per_bank_power(self):
        """测试单 bank 功耗"""
        power = PowerModel()
        
        # 设计约束: < 1mW/bank
        per_bank_power = power.get_per_bank_power()
        assert per_bank_power < 1.0  # mW
        
    def test_system_power(self):
        """测试系统级功耗"""
        power = PowerModel()
        
        # 512 banks 的总功耗
        system_power = power.get_system_power(num_banks=512)
        
        # 预期: ~256mW 动态 + ~51mW 静态
        assert 200 < system_power < 400  # mW


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
