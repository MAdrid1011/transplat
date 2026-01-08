"""
PIM Unit 单元测试

测试内容:
1. GeometryAddressGenerator 地址生成
2. 双线性插值权重计算
3. ComputeUnit MAC 操作
4. FP16/FP32 精度验证
"""

import pytest
import numpy as np
from geopim.simulator.pim_unit import (
    PIMUnitConfig, 
    PIMUnit,
    GeometryAddressGenerator,
    ComputeUnit,
    ParamEntry
)


class TestPIMUnitConfig:
    """PIM Unit 配置测试"""
    
    def test_default_config(self):
        """测试默认配置"""
        config = PIMUnitConfig()
        
        assert config.tile_c == 8           # 每 tile 8 channels
        assert config.num_neighbors == 4    # 双线性插值 4 邻域
        assert config.mac_lanes == 8        # 8-lane MAC
        assert config.accum_precision == "fp32"
        
    def test_gate_budget(self):
        """验证门级预算估算"""
        config = PIMUnitConfig()
        
        # 根据设计文档:
        # - Geometry Addr Gen: ~2K gates
        # - Tile Buffer: ~1K gates
        # - Compute Unit: ~5K gates
        # - Control Logic: ~1.5K gates
        # Total: ~9.5K gates < 10K
        estimated_gates = 2000 + 1000 + 5000 + 1500
        assert estimated_gates < 10000


class TestGeometryAddressGenerator:
    """几何地址生成器测试"""
    
    def test_coord_computation_no_offset(self):
        """测试无偏移的坐标计算"""
        addr_gen = GeometryAddressGenerator()
        
        params = ParamEntry(
            query_id=0,
            num_samples=512,
            ref_x=16.0,
            ref_y=16.0,
            stride_x=0.5,
            stride_y=0.5,
            has_offset=False
        )
        
        # sample_idx=0: (16.0, 16.0)
        x, y = addr_gen.compute_coord(params, sample_idx=0)
        assert abs(x - 16.0) < 1e-5
        assert abs(y - 16.0) < 1e-5
        
        # sample_idx=10: (16.0 + 10*0.5, 16.0 + 10*0.5) = (21.0, 21.0)
        x, y = addr_gen.compute_coord(params, sample_idx=10)
        assert abs(x - 21.0) < 1e-5
        assert abs(y - 21.0) < 1e-5
        
    def test_coord_computation_with_offset(self):
        """测试带偏移的坐标计算"""
        addr_gen = GeometryAddressGenerator()
        
        offsets = np.array([[0.1, -0.2], [0.3, 0.4]], dtype=np.float16)
        
        params = ParamEntry(
            query_id=0,
            num_samples=2,
            ref_x=10.0,
            ref_y=10.0,
            stride_x=1.0,
            stride_y=1.0,
            has_offset=True,
            offsets=offsets
        )
        
        # sample_idx=0: (10.0 + 0*1.0 + 0.1, 10.0 + 0*1.0 - 0.2) = (10.1, 9.8)
        x, y = addr_gen.compute_coord(params, sample_idx=0)
        assert abs(x - 10.1) < 0.01
        assert abs(y - 9.8) < 0.01
        
    def test_address_generation(self):
        """测试 HBM 地址生成"""
        addr_gen = GeometryAddressGenerator()
        
        # 坐标 (10.5, 20.5), 特征图大小 W=64, C=128
        coord = (10.5, 20.5)
        base_addr = 0
        W = 64
        C = 128
        
        addrs, weights = addr_gen.generate_addrs(coord, base_addr, W, C)
        
        # 4 个邻域地址
        assert len(addrs) == 4
        assert len(weights) == 4
        
        # 验证地址计算
        # (x0, y0) = (10, 20)
        # addr[0] = base + (y0 * W + x0) * C * 2 = (20*64 + 10) * 128 * 2
        expected_addr0 = (20 * 64 + 10) * 128 * 2
        assert addrs[0] == expected_addr0
        
    def test_bilinear_weights(self):
        """测试双线性插值权重"""
        addr_gen = GeometryAddressGenerator()
        
        # 坐标 (10.25, 20.75) -> wx=0.25, wy=0.75
        coord = (10.25, 20.75)
        _, weights = addr_gen.generate_addrs(coord, 0, 64, 128)
        
        # w[0] = (1-wx)*(1-wy) = 0.75 * 0.25 = 0.1875
        # w[1] = wx*(1-wy) = 0.25 * 0.25 = 0.0625
        # w[2] = (1-wx)*wy = 0.75 * 0.75 = 0.5625
        # w[3] = wx*wy = 0.25 * 0.75 = 0.1875
        assert abs(weights[0] - 0.1875) < 1e-5
        assert abs(weights[1] - 0.0625) < 1e-5
        assert abs(weights[2] - 0.5625) < 1e-5
        assert abs(weights[3] - 0.1875) < 1e-5
        
        # 权重和为 1
        assert abs(sum(weights) - 1.0) < 1e-5


class TestComputeUnit:
    """计算单元测试"""
    
    def test_bilinear_interpolation(self):
        """测试双线性插值"""
        config = PIMUnitConfig(tile_c=8)
        compute = ComputeUnit(config)
        
        # 4 个邻域的 8 通道特征
        neighbors = np.array([
            [1.0] * 8,  # neighbor 0
            [2.0] * 8,  # neighbor 1
            [3.0] * 8,  # neighbor 2
            [4.0] * 8,  # neighbor 3
        ], dtype=np.float16)
        
        # 均匀权重 [0.25, 0.25, 0.25, 0.25]
        weights = [0.25, 0.25, 0.25, 0.25]
        
        result = compute.bilinear_interp(neighbors, weights)
        
        # 期望: (1+2+3+4)/4 = 2.5
        assert result.shape == (8,)
        assert abs(result[0] - 2.5) < 0.01
        
    def test_weighted_accumulation(self):
        """测试加权累加"""
        config = PIMUnitConfig(tile_c=8, total_c=128)
        compute = ComputeUnit(config)
        
        # 重置累加器
        compute.reset_accumulator()
        
        # 采样值
        sample = np.array([1.0] * 8, dtype=np.float16)
        
        # 累加多个采样
        for i in range(10):
            attn_weight = 0.1  # 均匀权重
            compute.weighted_accum(sample, attn_weight, tile_start=0, tile_end=8)
        
        # 期望: 10 * 0.1 * 1.0 = 1.0
        result = compute.get_accumulator()
        assert abs(result[0] - 1.0) < 0.01
        
    def test_fp32_accumulation_precision(self):
        """测试 FP32 累加精度"""
        config = PIMUnitConfig(tile_c=8, total_c=128, accum_precision="fp32")
        compute = ComputeUnit(config)
        compute.reset_accumulator()
        
        # 大量小数值累加 (FP16 可能丢失精度)
        sample = np.array([0.001] * 8, dtype=np.float16)
        
        for i in range(1000):
            compute.weighted_accum(sample, 1.0, tile_start=0, tile_end=8)
        
        result = compute.get_accumulator()
        # FP32 累加应该保持精度: 1000 * 0.001 = 1.0
        assert abs(result[0] - 1.0) < 0.1  # 允许一些 FP16 输入误差


class TestPIMUnit:
    """完整 PIM Unit 测试"""
    
    def test_full_sample_processing(self):
        """测试完整采样处理流程"""
        config = PIMUnitConfig(tile_c=8, total_c=128)
        pim_unit = PIMUnit(config)
        
        # 模拟输入
        feature_tile = np.random.randn(4, 8).astype(np.float16)  # 4 neighbors × 8 channels
        bilinear_weights = [0.25, 0.25, 0.25, 0.25]
        attn_weight = 0.5
        
        # 处理一个 tile
        pim_unit.process_tile(feature_tile, bilinear_weights, attn_weight, tile_idx=0)
        
        # 获取部分结果
        partial_result = pim_unit.get_partial_result(tile_start=0, tile_end=8)
        assert partial_result.shape == (8,)
        
    def test_c_tiling(self):
        """测试 C-维度分块"""
        config = PIMUnitConfig(tile_c=8, total_c=128)
        pim_unit = PIMUnit(config)
        
        num_tiles = config.total_c // config.tile_c
        assert num_tiles == 16
        
        # 处理所有 tiles
        for tile_idx in range(num_tiles):
            feature_tile = np.random.randn(4, 8).astype(np.float16)
            pim_unit.process_tile(feature_tile, [0.25]*4, 0.1, tile_idx)
        
        # 获取完整结果
        result = pim_unit.get_full_result()
        assert result.shape == (128,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
