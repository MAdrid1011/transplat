"""
Per-Bank PIM Unit 模拟

组件 (<10K gates total):
- Geometry Address Generator (~2K gates)
- Tile Buffer 256B, Ping-Pong (~1K gates)  
- 8-lane FP16 MAC + Accum (~5K gates)
- Control Logic (~1.5K gates)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional
import numpy as np


@dataclass
class ParamEntry:
    """几何参数寄存器条目 (32B)"""
    
    query_id: int = 0
    num_samples: int = 512
    ref_x: float = 0.0
    ref_y: float = 0.0
    stride_x: float = 0.0
    stride_y: float = 0.0
    weight_base: int = 0
    offset_base: int = 0
    has_offset: bool = False
    valid: bool = False
    offsets: Optional[np.ndarray] = None  # [num_samples, 2] if has_offset


@dataclass
class PIMUnitConfig:
    """PIM Unit 配置"""
    
    tile_c: int = 8           # channels per tile
    total_c: int = 128        # total channels
    num_neighbors: int = 4    # bilinear interpolation neighbors
    mac_lanes: int = 8        # parallel MAC units
    accum_precision: str = "fp32"
    
    # Buffer 大小
    tile_buffer_size: int = 256  # bytes, ping-pong
    param_regs: int = 4          # 参数寄存器条目数
    
    @property
    def tiles_per_sample(self) -> int:
        """每个采样点的 tile 数"""
        return self.total_c // self.tile_c


class GeometryAddressGenerator:
    """
    几何地址生成器
    
    根据几何参数计算采样坐标和 HBM 地址。
    门级估算: ~2K gates
    """
    
    def __init__(self, num_entries: int = 4):
        self.param_regs: List[Optional[ParamEntry]] = [None] * num_entries
        
    def load_params(self, entry_idx: int, params: ParamEntry):
        """加载参数到寄存器"""
        if 0 <= entry_idx < len(self.param_regs):
            self.param_regs[entry_idx] = params
            
    def compute_coord(self, params: ParamEntry, sample_idx: int) -> Tuple[float, float]:
        """
        计算采样坐标
        
        (x, y) = (ref_x, ref_y) + sample_idx × (stride_x, stride_y) + offset
        """
        x = params.ref_x + sample_idx * params.stride_x
        y = params.ref_y + sample_idx * params.stride_y
        
        if params.has_offset and params.offsets is not None:
            x += float(params.offsets[sample_idx, 0])
            y += float(params.offsets[sample_idx, 1])
            
        return x, y
    
    def generate_addrs(
        self, 
        coord: Tuple[float, float], 
        base_addr: int, 
        W: int, 
        C: int
    ) -> Tuple[List[int], List[float]]:
        """
        生成 4 个邻域的 HBM 地址和双线性权重
        
        Args:
            coord: (x, y) 浮点坐标
            base_addr: 特征图基地址
            W: 特征图宽度
            C: 通道数
            
        Returns:
            addrs: 4 个邻域地址
            weights: 4 个双线性插值权重
        """
        x, y = coord
        
        # 整数坐标
        x0, y0 = int(np.floor(x)), int(np.floor(y))
        x1, y1 = x0 + 1, y0 + 1
        
        # 4 个邻域地址 (假设 NCHW 布局, FP16 = 2 bytes)
        # addr = base + (y * W + x) * C * 2
        addrs = [
            base_addr + (y0 * W + x0) * C * 2,  # (x0, y0)
            base_addr + (y0 * W + x1) * C * 2,  # (x1, y0)
            base_addr + (y1 * W + x0) * C * 2,  # (x0, y1)
            base_addr + (y1 * W + x1) * C * 2,  # (x1, y1)
        ]
        
        # 双线性插值权重
        wx = x - x0
        wy = y - y0
        weights = [
            (1 - wx) * (1 - wy),  # w00
            wx * (1 - wy),        # w10
            (1 - wx) * wy,        # w01
            wx * wy,              # w11
        ]
        
        return addrs, weights


class ComputeUnit:
    """
    计算单元: 8-lane FP16 MAC + FP32 累加器
    
    门级估算: ~5K gates
    - 8× FP16 FMA: ~3K gates
    - 16× FP32 Acc: ~1K gates
    - 控制逻辑: ~1K gates
    """
    
    def __init__(self, config: PIMUnitConfig):
        self.config = config
        self.accumulator = np.zeros(config.total_c, dtype=np.float32)
        
    def reset_accumulator(self):
        """重置累加器"""
        self.accumulator.fill(0.0)
        
    def bilinear_interp(
        self, 
        neighbors: np.ndarray, 
        weights: List[float]
    ) -> np.ndarray:
        """
        双线性插值
        
        Args:
            neighbors: [4, tile_c] 4 个邻域的特征
            weights: [4] 双线性权重
            
        Returns:
            [tile_c] 插值结果
        """
        result = np.zeros(self.config.tile_c, dtype=np.float32)
        for i, w in enumerate(weights):
            result += w * neighbors[i].astype(np.float32)
        return result.astype(np.float16)
    
    def weighted_accum(
        self, 
        sample: np.ndarray, 
        attn_weight: float, 
        tile_start: int, 
        tile_end: int
    ):
        """
        加权累加
        
        acc[tile_start:tile_end] += attn_weight * sample
        """
        self.accumulator[tile_start:tile_end] += attn_weight * sample.astype(np.float32)
        
    def get_accumulator(self) -> np.ndarray:
        """获取累加器值 (FP32)"""
        return self.accumulator.copy()
    
    def get_result_fp16(self) -> np.ndarray:
        """获取结果 (转换为 FP16)"""
        return self.accumulator.astype(np.float16)


class TileBuffer:
    """
    Tile Buffer: 256B Ping-Pong 缓冲
    
    - Ping: 4 neighbors × 8 channels × 2B = 64B
    - Pong: 64B
    - 双缓冲隐藏 DRAM 延迟
    """
    
    def __init__(self, config: PIMUnitConfig):
        self.config = config
        buffer_size = config.num_neighbors * config.tile_c
        self.ping = np.zeros(buffer_size, dtype=np.float16)
        self.pong = np.zeros(buffer_size, dtype=np.float16)
        self.active_buffer = 0  # 0 = ping, 1 = pong
        
    def load(self, data: np.ndarray):
        """加载数据到非活跃缓冲区"""
        flat_data = data.flatten().astype(np.float16)
        if self.active_buffer == 0:
            self.pong[:len(flat_data)] = flat_data
        else:
            self.ping[:len(flat_data)] = flat_data
            
    def swap(self):
        """切换活跃缓冲区"""
        self.active_buffer = 1 - self.active_buffer
        
    def get_active(self) -> np.ndarray:
        """获取活跃缓冲区数据"""
        if self.active_buffer == 0:
            return self.ping.reshape(self.config.num_neighbors, self.config.tile_c)
        else:
            return self.pong.reshape(self.config.num_neighbors, self.config.tile_c)


class PIMUnit:
    """
    完整的 Per-Bank PIM Unit
    
    集成:
    - GeometryAddressGenerator
    - TileBuffer (Ping-Pong)
    - ComputeUnit (8-lane MAC + Accum)
    
    总门级: ~9.5K gates < 10K
    """
    
    def __init__(self, config: Optional[PIMUnitConfig] = None):
        self.config = config or PIMUnitConfig()
        self.addr_gen = GeometryAddressGenerator()
        self.tile_buffer = TileBuffer(self.config)
        self.compute = ComputeUnit(self.config)
        
        # 状态
        self.state = "IDLE"  # IDLE, LOAD, COMPUTE, OUTPUT
        self.current_query = None
        self.current_sample = 0
        self.current_tile = 0
        
    def reset(self):
        """重置单元状态"""
        self.compute.reset_accumulator()
        self.state = "IDLE"
        self.current_query = None
        self.current_sample = 0
        self.current_tile = 0
        
    def process_tile(
        self, 
        feature_tile: np.ndarray, 
        bilinear_weights: List[float],
        attn_weight: float,
        tile_idx: int
    ):
        """
        处理一个 tile
        
        Args:
            feature_tile: [4, tile_c] 4 个邻域的特征
            bilinear_weights: [4] 双线性权重
            attn_weight: 注意力权重
            tile_idx: Tile 索引
        """
        # 双线性插值
        sample = self.compute.bilinear_interp(feature_tile, bilinear_weights)
        
        # 加权累加
        tile_start = tile_idx * self.config.tile_c
        tile_end = tile_start + self.config.tile_c
        self.compute.weighted_accum(sample, attn_weight, tile_start, tile_end)
        
    def get_partial_result(self, tile_start: int, tile_end: int) -> np.ndarray:
        """获取部分结果"""
        return self.compute.get_accumulator()[tile_start:tile_end]
    
    def get_full_result(self) -> np.ndarray:
        """获取完整结果 (FP16)"""
        return self.compute.get_result_fp16()
    
    def estimate_cycles_per_tile(self, row_hit: bool) -> int:
        """估算每 tile 周期数"""
        fetch_cycles = 4 if row_hit else 20
        compute_cycles = 5  # bilinear (4) + accum (1)
        return max(fetch_cycles, compute_cycles)  # Ping-pong 隐藏延迟

