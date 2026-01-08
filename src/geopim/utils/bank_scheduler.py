"""
几何感知 Bank 调度器

目标: 最大化 row buffer hit rate (目标 60-70%)

策略:
1. 将特征图按空间位置划分到不同 bank
2. 将采样区域相近的 queries 分配到同一 bank
3. 同一 bank 内的 queries 按采样区域排序
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np


@dataclass
class SchedulerConfig:
    """调度器配置"""
    
    num_banks: int = 512
    feature_height: int = 32
    feature_width: int = 32
    tile_grid_size: int = 16  # 特征图划分为 16x16 tile grid


class BankScheduler:
    """
    几何感知 Bank 调度器
    
    根据采样区域的空间局部性分配 queries 到 banks，
    以最大化 row buffer hit rate。
    """
    
    def __init__(self, config: Optional[SchedulerConfig] = None):
        self.config = config or SchedulerConfig()
        self.bank_assignments: Dict[int, List[int]] = {}
        self._query_centers: Optional[np.ndarray] = None
        
    def compute_bank_id(self, x: float, y: float) -> int:
        """
        计算坐标对应的 bank ID
        
        基于空间位置的 hash 映射，相邻区域映射到相邻 bank group。
        """
        H, W = self.config.feature_height, self.config.feature_width
        grid_size = self.config.tile_grid_size
        
        # 划分为 tile grid
        tile_x = int(x * grid_size / W) % grid_size
        tile_y = int(y * grid_size / H) % grid_size
        tile_id = tile_y * grid_size + tile_x
        
        # 每个 tile 映射到 2 个 banks (负载均衡)
        return (tile_id * 2) % self.config.num_banks
    
    def distribute_queries(
        self, 
        query_centers: np.ndarray
    ) -> Dict[int, List[int]]:
        """
        将 queries 分配到 banks
        
        Args:
            query_centers: [Q, 2] - 每个 query 的采样区域中心
            
        Returns:
            bank_assignments: {bank_id: [query_indices]}
        """
        self._query_centers = query_centers
        self.bank_assignments = {i: [] for i in range(self.config.num_banks)}
        
        for q_idx, (cx, cy) in enumerate(query_centers):
            bank_id = self.compute_bank_id(cx, cy)
            self.bank_assignments[bank_id].append(q_idx)
        
        # 同一 bank 内按 Morton code 排序以提高局部性
        for bank_id in self.bank_assignments:
            queries = self.bank_assignments[bank_id]
            if len(queries) > 1:
                queries.sort(key=lambda q: self._morton_code(query_centers[q]))
        
        return self.bank_assignments
    
    def _morton_code(self, coord: np.ndarray) -> int:
        """
        计算 Morton code (Z-order curve)
        
        将 2D 坐标映射到 1D，保持空间局部性。
        """
        x, y = int(coord[0] * 1000), int(coord[1] * 1000)
        code = 0
        for i in range(16):
            code |= ((x >> i) & 1) << (2*i)
            code |= ((y >> i) & 1) << (2*i + 1)
        return code
    
    def estimate_row_hit_rate(self) -> float:
        """
        估算预期 row hit rate
        
        基于 bank 内 query 数量和空间局部性估算。
        """
        if not self.bank_assignments or self._query_centers is None:
            return 0.5  # 默认值
        
        total_accesses = 0
        estimated_hits = 0
        
        for bank_id, queries in self.bank_assignments.items():
            if len(queries) <= 1:
                continue
            
            # 计算相邻 queries 之间的距离
            for i in range(1, len(queries)):
                q_prev = queries[i-1]
                q_curr = queries[i]
                
                # 距离越近，hit rate 越高
                dist = np.linalg.norm(
                    self._query_centers[q_curr] - self._query_centers[q_prev]
                )
                
                # 简化模型: 距离 < 2 pixels 时 80% hit rate
                if dist < 2.0:
                    hit_prob = 0.8
                elif dist < 5.0:
                    hit_prob = 0.5
                else:
                    hit_prob = 0.2
                
                total_accesses += 1
                estimated_hits += hit_prob
        
        if total_accesses == 0:
            return 0.5
        return estimated_hits / total_accesses
    
    def get_load_balance(self) -> Dict:
        """
        获取负载均衡统计
        
        Returns:
            负载分布统计
        """
        loads = [len(q) for q in self.bank_assignments.values()]
        
        return {
            'min_load': min(loads) if loads else 0,
            'max_load': max(loads) if loads else 0,
            'avg_load': np.mean(loads) if loads else 0,
            'std_load': np.std(loads) if loads else 0,
            'active_banks': sum(1 for l in loads if l > 0),
            'total_banks': self.config.num_banks,
        }
    
    def optimize_assignment(
        self,
        query_centers: np.ndarray,
        max_load_imbalance: float = 1.5
    ) -> Dict[int, List[int]]:
        """
        优化 bank 分配以平衡负载
        
        Args:
            query_centers: [Q, 2] query 中心坐标
            max_load_imbalance: 最大负载不均衡比 (max/avg)
            
        Returns:
            优化后的 bank 分配
        """
        # 首先使用基础分配
        self.distribute_queries(query_centers)
        
        # 检查负载均衡
        stats = self.get_load_balance()
        imbalance = stats['max_load'] / max(stats['avg_load'], 1)
        
        if imbalance > max_load_imbalance:
            # 重新分配过载 bank 的 queries
            avg_load = stats['avg_load']
            
            for bank_id, queries in list(self.bank_assignments.items()):
                if len(queries) > avg_load * max_load_imbalance:
                    # 移动多余的 queries 到负载较低的 banks
                    excess = queries[int(avg_load * max_load_imbalance):]
                    self.bank_assignments[bank_id] = queries[:int(avg_load * max_load_imbalance)]
                    
                    # 找到负载最低的 banks
                    for q in excess:
                        min_bank = min(
                            self.bank_assignments.keys(),
                            key=lambda b: len(self.bank_assignments[b])
                        )
                        self.bank_assignments[min_bank].append(q)
        
        return self.bank_assignments

