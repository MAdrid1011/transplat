"""
PIM 控制器

管理多个 Bank 的 PIM Unit，处理 Query 到 Bank 的分配和调度。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np

from .hbm_model import HBMConfig, HBMModel
from .pim_unit import PIMUnit, PIMUnitConfig, ParamEntry


@dataclass
class ControllerConfig:
    """控制器配置"""
    
    num_active_banks: int = 512      # 活跃 bank 数
    max_queries_per_bank: int = 64   # 每 bank 最大 query 数
    
    hbm_config: Optional[HBMConfig] = None
    pim_config: Optional[PIMUnitConfig] = None


class PIMController:
    """
    PIM 控制器
    
    功能:
    1. Query 到 Bank 分配
    2. 协调多 Bank 并行执行
    3. 收集结果和统计
    """
    
    def __init__(self, config: Optional[ControllerConfig] = None):
        self.config = config or ControllerConfig()
        
        # 初始化 HBM 模型
        self.hbm = HBMModel(self.config.hbm_config)
        
        # 初始化 PIM Units (按需创建)
        self.pim_config = self.config.pim_config or PIMUnitConfig()
        self.pim_units: Dict[int, PIMUnit] = {}
        
        # Bank 分配
        self.bank_assignments: Dict[int, List[int]] = {}  # bank_id -> [query_ids]
        
        # 统计
        self.total_cycles = 0
        self.queries_processed = 0
        
    def get_pim_unit(self, bank_id: int) -> PIMUnit:
        """获取或创建 PIM Unit"""
        if bank_id not in self.pim_units:
            self.pim_units[bank_id] = PIMUnit(self.pim_config)
        return self.pim_units[bank_id]
    
    def assign_queries_to_banks(
        self, 
        query_centers: np.ndarray,
        feature_shape: Tuple[int, int]
    ) -> Dict[int, List[int]]:
        """
        将 Query 分配到 Bank
        
        策略: 基于采样区域中心的空间局部性分配
        
        Args:
            query_centers: [Q, 2] 每个 query 的采样中心坐标
            feature_shape: (H, W) 特征图大小
            
        Returns:
            {bank_id: [query_indices]}
        """
        num_queries = len(query_centers)
        H, W = feature_shape
        
        self.bank_assignments = {i: [] for i in range(self.config.num_active_banks)}
        
        for q_idx in range(num_queries):
            cx, cy = query_centers[q_idx]
            
            # 基于空间位置计算 bank ID
            # 将特征图划分为 grid，相邻区域映射到相邻 bank
            grid_x = int(cx * 16 / W) % 16
            grid_y = int(cy * 16 / H) % 16
            tile_id = grid_y * 16 + grid_x
            bank_id = (tile_id * 2) % self.config.num_active_banks
            
            self.bank_assignments[bank_id].append(q_idx)
            
        return self.bank_assignments
    
    def execute_query(
        self,
        bank_id: int,
        query_id: int,
        feature_map: np.ndarray,
        params: ParamEntry,
        weights: np.ndarray,
        offsets: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, int]:
        """
        在指定 Bank 执行单个 Query
        
        Args:
            bank_id: Bank ID
            query_id: Query ID
            feature_map: [C, H, W] 特征图 (FP16)
            params: 几何参数
            weights: [S] 注意力权重
            offsets: [S, 2] 采样偏移 (可选)
            
        Returns:
            output: [C] 聚合结果
            cycles: 消耗周期数
        """
        pim_unit = self.get_pim_unit(bank_id)
        pim_unit.reset()
        
        C, H, W = feature_map.shape
        num_samples = len(weights)
        total_cycles = 0
        
        # 更新参数
        if offsets is not None:
            params.has_offset = True
            params.offsets = offsets
            
        pim_unit.addr_gen.load_params(0, params)
        
        # 处理每个采样点
        for s_idx in range(num_samples):
            # 计算坐标
            coord = pim_unit.addr_gen.compute_coord(params, s_idx)
            
            # 生成地址和双线性权重
            addrs, bilinear_weights = pim_unit.addr_gen.generate_addrs(
                coord, base_addr=0, W=W, C=C
            )
            
            attn_weight = float(weights[s_idx])
            
            # C-Tiling: 分块处理
            num_tiles = self.pim_config.tiles_per_sample
            tile_c = self.pim_config.tile_c
            
            for tile_idx in range(num_tiles):
                tile_start = tile_idx * tile_c
                tile_end = tile_start + tile_c
                
                # 模拟 HBM 访问
                x0, y0 = int(coord[0]), int(coord[1])
                x1, y1 = min(x0 + 1, W - 1), min(y0 + 1, H - 1)
                
                # 计算 row (简化模型)
                row = y0 * W // 64  # 假设每 row 64 pixels
                
                # 访问 4 个邻域
                for neighbor_idx in range(4):
                    latency = self.hbm.access(bank_id, row, tile_idx)
                    
                # 估算周期数 (使用 ping-pong 隐藏延迟)
                row_hit = (self.hbm.row_buffers.get(bank_id) == row)
                tile_cycles = pim_unit.estimate_cycles_per_tile(row_hit)
                total_cycles += tile_cycles
                
                # 获取特征 tile
                feature_tile = np.array([
                    feature_map[tile_start:tile_end, y0, x0],
                    feature_map[tile_start:tile_end, y0, x1],
                    feature_map[tile_start:tile_end, y1, x0],
                    feature_map[tile_start:tile_end, y1, x1],
                ], dtype=np.float16)
                
                # 处理 tile
                pim_unit.process_tile(feature_tile, bilinear_weights, attn_weight, tile_idx)
        
        output = pim_unit.get_full_result()
        return output, total_cycles
    
    def execute_batch(
        self,
        feature_map: np.ndarray,
        all_params: List[ParamEntry],
        all_weights: np.ndarray,
        all_offsets: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict]:
        """
        批量执行所有 Query
        
        Args:
            feature_map: [C, H, W] 特征图
            all_params: [Q] 所有 query 的参数
            all_weights: [Q, S] 所有权重
            all_offsets: [Q, S, 2] 所有偏移 (可选)
            
        Returns:
            outputs: [Q, C] 所有输出
            stats: 统计信息
        """
        num_queries = len(all_params)
        C = feature_map.shape[0]
        
        outputs = np.zeros((num_queries, C), dtype=np.float16)
        self.total_cycles = 0
        self.queries_processed = 0
        
        # 分配 queries 到 banks
        query_centers = np.array([
            [p.ref_x, p.ref_y] for p in all_params
        ])
        self.assign_queries_to_banks(query_centers, feature_map.shape[1:])
        
        # 并行执行每个 bank 的 queries
        # (在模拟中顺序执行，但统计按并行计算)
        max_bank_cycles = 0
        
        for bank_id, query_ids in self.bank_assignments.items():
            bank_cycles = 0
            
            for q_id in query_ids:
                offsets = all_offsets[q_id] if all_offsets is not None else None
                
                output, cycles = self.execute_query(
                    bank_id=bank_id,
                    query_id=q_id,
                    feature_map=feature_map,
                    params=all_params[q_id],
                    weights=all_weights[q_id],
                    offsets=offsets
                )
                
                outputs[q_id] = output
                bank_cycles += cycles
                self.queries_processed += 1
                
            max_bank_cycles = max(max_bank_cycles, bank_cycles)
        
        # 并行执行时，总时间取决于最慢的 bank
        self.total_cycles = max_bank_cycles
        
        stats = self.get_stats()
        return outputs, stats
    
    def get_stats(self) -> Dict:
        """获取统计信息"""
        hbm_stats = self.hbm.get_stats()
        
        return {
            'total_cycles': self.total_cycles,
            'queries_processed': self.queries_processed,
            'row_hit_rate': hbm_stats['row_hit_rate'],
            'total_bytes': hbm_stats['total_bytes'],
            'active_banks': len([b for b, q in self.bank_assignments.items() if q]),
        }
    
    def reset(self):
        """重置控制器"""
        self.hbm.reset()
        for pim_unit in self.pim_units.values():
            pim_unit.reset()
        self.bank_assignments.clear()
        self.total_cycles = 0
        self.queries_processed = 0

