"""
HBM3 结构模型

HBM3 6-stack 配置:
- 1536 banks total (6 stacks × 8 channels × 32 banks)
- Row buffer: 2KB per bank
- 内部带宽: ~8 TB/s
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class HBMConfig:
    """HBM3 配置参数"""
    
    # 结构参数
    num_stacks: int = 6
    channels_per_stack: int = 8
    banks_per_channel: int = 32
    row_buffer_size: int = 2048  # 2KB per bank
    
    # 延迟参数 (cycles @ 300MHz)
    row_hit_latency: int = 4
    row_miss_latency: int = 20
    burst_length: int = 64  # bytes
    
    # 带宽参数
    internal_bandwidth_gbps: float = 8000.0  # ~8 TB/s
    
    @property
    def total_banks(self) -> int:
        """总 bank 数"""
        return self.num_stacks * self.channels_per_stack * self.banks_per_channel


@dataclass
class AccessStats:
    """HBM 访问统计"""
    
    row_hits: int = 0
    row_misses: int = 0
    total_bytes: int = 0
    
    def reset(self):
        """重置统计"""
        self.row_hits = 0
        self.row_misses = 0
        self.total_bytes = 0
        
    def get_hit_rate(self) -> float:
        """计算 row hit rate"""
        total = self.row_hits + self.row_misses
        if total == 0:
            return 0.0
        return self.row_hits / total
    
    def __repr__(self) -> str:
        hit_rate = self.get_hit_rate()
        return (f"AccessStats(hits={self.row_hits}, misses={self.row_misses}, "
                f"hit_rate={hit_rate:.2%}, bytes={self.total_bytes})")


class HBMModel:
    """
    HBM3 访问模型
    
    模拟 HBM 的 row buffer 行为，追踪访问延迟和统计信息。
    """
    
    def __init__(self, config: Optional[HBMConfig] = None):
        self.config = config or HBMConfig()
        self.row_buffers: Dict[int, int] = {}  # bank_id -> current_row
        self.stats = AccessStats()
        
    def access(self, bank_id: int, row: int, col: int = 0) -> int:
        """
        模拟 HBM 访问
        
        Args:
            bank_id: Bank 编号
            row: Row 地址
            col: Column 地址 (用于统计)
            
        Returns:
            访问延迟 (cycles)
        """
        # 检查 row buffer 状态
        if self.row_buffers.get(bank_id) == row:
            # Row buffer hit
            self.stats.row_hits += 1
            latency = self.config.row_hit_latency
        else:
            # Row buffer miss - 需要激活新 row
            self.stats.row_misses += 1
            self.row_buffers[bank_id] = row
            latency = self.config.row_miss_latency
            
        # 更新字节统计
        self.stats.total_bytes += self.config.burst_length
        
        return latency
    
    def batch_access(self, accesses: list) -> int:
        """
        批量访问模拟
        
        Args:
            accesses: [(bank_id, row, col), ...] 访问序列
            
        Returns:
            总延迟 (cycles)
        """
        total_latency = 0
        for bank_id, row, col in accesses:
            total_latency += self.access(bank_id, row, col)
        return total_latency
    
    def get_bank_for_address(self, addr: int) -> int:
        """
        根据地址计算 bank ID
        
        使用简单的交织策略: addr 的低位决定 bank
        """
        # 假设 64B burst，使用 addr >> 6 的低 bits 选择 bank
        return (addr >> 6) % self.config.total_banks
    
    def get_row_for_address(self, addr: int, bank_id: int) -> int:
        """
        根据地址计算 row
        
        每个 bank 有 2KB row buffer
        """
        # 移除 bank 交织位后，计算 row
        bank_bits = (self.config.total_banks - 1).bit_length()
        row_size = self.config.row_buffer_size
        return (addr >> (6 + bank_bits)) // row_size
    
    def reset(self):
        """重置模型状态"""
        self.row_buffers.clear()
        self.stats.reset()
        
    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            'row_hits': self.stats.row_hits,
            'row_misses': self.stats.row_misses,
            'row_hit_rate': self.stats.get_hit_rate(),
            'total_bytes': self.stats.total_bytes,
        }

