"""
HBM 模型单元测试

测试内容:
1. HBMConfig 默认配置验证
2. Row buffer 命中/缺失延迟计算
3. AccessStats 统计准确性
4. 多 bank 并发访问模拟
"""

import pytest
from geopim.simulator.hbm_model import HBMConfig, HBMModel, AccessStats


class TestHBMConfig:
    """HBM 配置测试"""
    
    def test_default_config(self):
        """测试默认 HBM3 配置"""
        config = HBMConfig()
        
        # HBM3 6-stack 配置
        assert config.num_stacks == 6
        assert config.channels_per_stack == 8
        assert config.banks_per_channel == 32
        
        # 总 bank 数
        total_banks = config.num_stacks * config.channels_per_stack * config.banks_per_channel
        assert total_banks == 1536
        
    def test_row_buffer_size(self):
        """测试 row buffer 大小"""
        config = HBMConfig()
        assert config.row_buffer_size == 2048  # 2KB
        
    def test_latency_params(self):
        """测试延迟参数 @ 300MHz"""
        config = HBMConfig()
        assert config.row_hit_latency == 4   # cycles
        assert config.row_miss_latency == 20  # cycles
        assert config.burst_length == 64      # bytes


class TestHBMModel:
    """HBM 模型测试"""
    
    def test_row_buffer_hit(self):
        """测试 row buffer 命中"""
        model = HBMModel(HBMConfig())
        
        # 第一次访问 (miss)
        latency1 = model.access(bank_id=0, row=100, col=0)
        assert latency1 == model.config.row_miss_latency
        
        # 同一 row 再次访问 (hit)
        latency2 = model.access(bank_id=0, row=100, col=32)
        assert latency2 == model.config.row_hit_latency
        
    def test_row_buffer_miss(self):
        """测试 row buffer 缺失"""
        model = HBMModel(HBMConfig())
        
        # 第一次访问 row 100
        model.access(bank_id=0, row=100, col=0)
        
        # 访问不同 row (miss)
        latency = model.access(bank_id=0, row=200, col=0)
        assert latency == model.config.row_miss_latency
        
    def test_different_banks_independent(self):
        """测试不同 bank 的 row buffer 独立"""
        model = HBMModel(HBMConfig())
        
        # Bank 0 访问 row 100
        model.access(bank_id=0, row=100, col=0)
        
        # Bank 1 访问 row 100 (独立的 row buffer)
        latency = model.access(bank_id=1, row=100, col=0)
        assert latency == model.config.row_miss_latency  # 仍然是 miss
        
        # Bank 0 再次访问 row 100 (hit)
        latency = model.access(bank_id=0, row=100, col=0)
        assert latency == model.config.row_hit_latency


class TestAccessStats:
    """访问统计测试"""
    
    def test_stats_counting(self):
        """测试命中/缺失计数"""
        model = HBMModel(HBMConfig())
        
        # 10 次访问同一 row (1 miss + 9 hits)
        for i in range(10):
            model.access(bank_id=0, row=100, col=i * 64)
            
        assert model.stats.row_misses == 1
        assert model.stats.row_hits == 9
        
    def test_row_hit_rate(self):
        """测试 row hit rate 计算"""
        model = HBMModel(HBMConfig())
        
        # 模拟访问模式: 5 个不同 row，每个访问 4 次
        for row in range(5):
            for access in range(4):
                model.access(bank_id=0, row=row, col=access * 64)
        
        # 5 misses + 15 hits = 20 total
        assert model.stats.row_misses == 5
        assert model.stats.row_hits == 15
        
        hit_rate = model.stats.get_hit_rate()
        assert abs(hit_rate - 0.75) < 0.01  # 75% hit rate
        
    def test_stats_reset(self):
        """测试统计重置"""
        model = HBMModel(HBMConfig())
        
        # 一些访问
        for i in range(10):
            model.access(bank_id=0, row=i, col=0)
            
        # 重置统计
        model.stats.reset()
        
        assert model.stats.row_hits == 0
        assert model.stats.row_misses == 0


class TestMultiBankAccess:
    """多 bank 访问测试"""
    
    def test_parallel_bank_access(self):
        """测试多 bank 并行访问"""
        model = HBMModel(HBMConfig())
        
        # 512 个 bank 并行访问
        num_banks = 512
        
        for bank_id in range(num_banks):
            model.access(bank_id=bank_id, row=0, col=0)
            
        # 所有都是 miss (每个 bank 第一次访问)
        assert model.stats.row_misses == num_banks
        assert model.stats.row_hits == 0
        
    def test_locality_pattern(self):
        """测试具有局部性的访问模式"""
        model = HBMModel(HBMConfig())
        
        # 模拟几何采样的访问模式:
        # 每个 query 访问 4 个相邻像素 (通常在同一 row)
        num_queries = 100
        
        for q in range(num_queries):
            base_row = q * 10  # 每个 query 访问不同区域
            # 4 个邻域像素 (假设在同一 row)
            for neighbor in range(4):
                model.access(bank_id=q % 512, row=base_row, col=neighbor * 16)
        
        # 每个 query: 1 miss + 3 hits
        expected_misses = num_queries
        expected_hits = num_queries * 3
        
        assert model.stats.row_misses == expected_misses
        assert model.stats.row_hits == expected_hits


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
