"""
GeoPIM 集成测试

测试内容:
1. GeoPIM 输出与 PyTorch grid_sample 对比
2. 端到端采样流程
3. 数值精度验证
"""

import pytest
import numpy as np

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from geopim.simulator.pim_unit import GeometryAddressGenerator, ComputeUnit, PIMUnitConfig
from geopim.simulator.geopim_simulator import GeoPIMSimulator
from geopim.timing.cycle_model import TimingConfig, CycleModel
from geopim.simulator.hbm_model import HBMConfig, HBMModel


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestFunctionalCorrectness:
    """功能正确性测试"""
    
    def test_bilinear_vs_grid_sample(self):
        """测试双线性插值与 PyTorch grid_sample 对比"""
        # 创建测试特征图
        B, C, H, W = 1, 8, 32, 32
        feature_map = torch.randn(B, C, H, W, dtype=torch.float32)
        
        # 测试坐标
        test_coords = [
            (10.25, 15.75),
            (20.5, 25.5),
            (5.0, 5.0),  # 整数坐标
        ]
        
        addr_gen = GeometryAddressGenerator()
        config = PIMUnitConfig(tile_c=8)
        compute = ComputeUnit(config)
        
        for x, y in test_coords:
            # GeoPIM 计算
            _, bilinear_weights = addr_gen.generate_addrs((x, y), 0, W, C)
            
            # 获取 4 个邻域像素
            x0, y0 = int(x), int(y)
            x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
            
            neighbors = np.array([
                feature_map[0, :, y0, x0].numpy(),
                feature_map[0, :, y0, x1].numpy(),
                feature_map[0, :, y1, x0].numpy(),
                feature_map[0, :, y1, x1].numpy(),
            ], dtype=np.float32)
            
            geopim_result = compute.bilinear_interp(neighbors, bilinear_weights)
            
            # PyTorch grid_sample 计算
            # 归一化坐标 [-1, 1]
            norm_x = 2 * x / (W - 1) - 1
            norm_y = 2 * y / (H - 1) - 1
            grid = torch.tensor([[[[norm_x, norm_y]]]], dtype=torch.float32)
            
            torch_result = F.grid_sample(
                feature_map, grid, 
                mode='bilinear', 
                padding_mode='zeros',
                align_corners=True
            ).squeeze().numpy()
            
            # 验证结果接近
            np.testing.assert_allclose(geopim_result, torch_result, rtol=1e-3, atol=1e-4)


@pytest.mark.skipif(not HAS_TORCH, reason="PyTorch not available")
class TestNumericalPrecision:
    """数值精度测试"""
    
    def test_fp16_accumulation_error(self):
        """测试 FP16 累加误差"""
        config = PIMUnitConfig(tile_c=8, accum_precision="fp32")
        compute = ComputeUnit(config)
        compute.reset_accumulator()
        
        # 大量小值累加
        sample = np.array([1e-3] * 8, dtype=np.float16)
        
        # 累加 1000 次
        for _ in range(1000):
            compute.weighted_accum(sample, 1.0, 0, 8)
        
        result = compute.get_accumulator()[:8]
        expected = 1.0  # 1000 * 1e-3
        
        # FP32 累加应保持精度
        assert abs(result[0] - expected) < 0.1
        
    def test_large_weight_values(self):
        """测试大权重值的数值稳定性"""
        config = PIMUnitConfig(tile_c=8, accum_precision="fp32")
        compute = ComputeUnit(config)
        compute.reset_accumulator()
        
        # 大权重值
        sample = np.array([100.0] * 8, dtype=np.float16)
        
        for _ in range(100):
            compute.weighted_accum(sample, 10.0, 0, 8)
        
        result = compute.get_accumulator()[:8]
        expected = 100.0 * 10.0 * 100  # 100000
        
        # 验证无溢出
        assert result[0] > 50000  # 允许一些精度损失


class TestSimulatorEstimation:
    """模拟器性能估算测试"""
    
    def test_performance_estimation(self):
        """测试性能估算接口"""
        simulator = GeoPIMSimulator()
        
        result = simulator.estimate_performance(
            batch_size=2,
            num_queries=1024,
            num_samples=512,
            num_views=4,
            row_hit_rate=0.7
        )
        
        # 验证返回的字段
        assert 'total_samples' in result
        assert 'sample_cycles' in result
        assert 'estimated_ms' in result
        assert 'speedup' in result
        
        # 验证加速比在合理范围
        assert 3.0 < result['speedup'] < 10.0
        
    def test_speedup_with_different_hit_rates(self):
        """测试不同 hit rate 下的加速比"""
        simulator = GeoPIMSimulator()
        
        speedups = []
        for hit_rate in [0.3, 0.5, 0.7, 0.9]:
            result = simulator.estimate_performance(row_hit_rate=hit_rate)
            speedups.append(result['speedup'])
        
        # 更高的 hit rate 应该带来更高的加速比
        assert speedups[0] < speedups[1] < speedups[2] < speedups[3]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
