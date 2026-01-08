"""
GeoPIM 完整模拟器

整合所有组件，提供统一的模拟接口。
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union
import numpy as np

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from .hbm_model import HBMConfig, HBMModel
from .pim_unit import PIMUnitConfig, ParamEntry
from .controller import PIMController, ControllerConfig


@dataclass  
class SimulatorConfig:
    """模拟器配置"""
    
    # HBM 配置
    num_stacks: int = 6
    channels_per_stack: int = 8
    banks_per_channel: int = 32
    
    # PIM 配置
    tile_c: int = 8
    total_c: int = 128
    mac_lanes: int = 8
    
    # 控制器配置
    num_active_banks: int = 512
    
    # 时序参数
    pim_freq_mhz: int = 300
    row_hit_latency: int = 4
    row_miss_latency: int = 20


class GeoPIMSimulator:
    """
    GeoPIM v3.0 完整模拟器
    
    支持:
    - TransPlat 几何引导采样
    - PixelSplat 加权采样
    - 性能统计和分析
    """
    
    _default_instance = None
    
    def __init__(self, config: Optional[SimulatorConfig] = None):
        self.config = config or SimulatorConfig()
        
        # 创建配置对象
        hbm_config = HBMConfig(
            num_stacks=self.config.num_stacks,
            channels_per_stack=self.config.channels_per_stack,
            banks_per_channel=self.config.banks_per_channel,
            row_hit_latency=self.config.row_hit_latency,
            row_miss_latency=self.config.row_miss_latency,
        )
        
        pim_config = PIMUnitConfig(
            tile_c=self.config.tile_c,
            total_c=self.config.total_c,
            mac_lanes=self.config.mac_lanes,
        )
        
        controller_config = ControllerConfig(
            num_active_banks=self.config.num_active_banks,
            hbm_config=hbm_config,
            pim_config=pim_config,
        )
        
        # 创建控制器
        self.controller = PIMController(controller_config)
        
        # 最后一次执行的统计
        self._last_stats: Dict = {}
        
    @classmethod
    def get_default(cls) -> "GeoPIMSimulator":
        """获取默认模拟器实例"""
        if cls._default_instance is None:
            cls._default_instance = cls()
        return cls._default_instance
    
    def execute(
        self,
        feature_map: Union[np.ndarray, "torch.Tensor"],
        geo_params: Union[Dict, np.ndarray, "torch.Tensor"],
        weights: Union[np.ndarray, "torch.Tensor"],
        offsets: Optional[Union[np.ndarray, "torch.Tensor"]] = None
    ) -> Tuple[Union[np.ndarray, "torch.Tensor"], Dict]:
        """
        执行 GeoPIM 模拟
        
        Args:
            feature_map: [B, C, H, W] 或 [C, H, W] 特征图 (FP16)
            geo_params: 几何参数，可以是:
                - dict: {'ref_xy': [B, Q, 2], 'stride_xy': [B, Q, 2]}
                - tensor: [B, Q, param_size]
            weights: [B, Q, S] 或 [Q, S] 预计算权重
            offsets: [B, Q, S, 2] 或 [Q, S, 2] 采样偏移 (可选)
            
        Returns:
            output: [B, Q, C] 或 [Q, C] 聚合结果
            stats: 性能统计
        """
        # 转换输入
        feature_np, weights_np, offsets_np, geo_dict = self._convert_inputs(
            feature_map, geo_params, weights, offsets
        )
        
        # 确定批次大小
        if feature_np.ndim == 4:
            B, C, H, W = feature_np.shape
        else:
            B = 1
            C, H, W = feature_np.shape
            feature_np = feature_np[np.newaxis, ...]
            
        if weights_np.ndim == 2:
            weights_np = weights_np[np.newaxis, ...]
            
        Q, S = weights_np.shape[1], weights_np.shape[2]
        
        # 执行每个 batch
        all_outputs = []
        total_stats = {
            'total_cycles': 0,
            'row_hit_rate': 0,
            'total_bytes': 0,
            'queries_processed': 0,
        }
        
        for b in range(B):
            # 准备参数
            params_list = self._create_params_list(geo_dict, b, Q, S)
            
            # 执行
            self.controller.reset()
            outputs_b, stats_b = self.controller.execute_batch(
                feature_map=feature_np[b],  # [C, H, W]
                all_params=params_list,
                all_weights=weights_np[b],  # [Q, S]
                all_offsets=offsets_np[b] if offsets_np is not None else None,
            )
            
            all_outputs.append(outputs_b)
            
            # 累积统计
            total_stats['total_cycles'] = max(total_stats['total_cycles'], stats_b['total_cycles'])
            total_stats['row_hit_rate'] += stats_b['row_hit_rate']
            total_stats['total_bytes'] += stats_b['total_bytes']
            total_stats['queries_processed'] += stats_b['queries_processed']
        
        # 平均 hit rate
        total_stats['row_hit_rate'] /= B
        
        # 计算吞吐量
        total_stats['estimated_ms'] = total_stats['total_cycles'] / (self.config.pim_freq_mhz * 1e3)
        total_stats['throughput'] = total_stats['queries_processed'] * S / (total_stats['estimated_ms'] / 1000)
        
        # 注: GPU baseline 应由调用方提供，这里不计算 speedup
        # 避免硬编码 GPU 时间，让 benchmark 脚本根据实测值计算
        
        self._last_stats = total_stats
        
        # 合并输出
        output_np = np.stack(all_outputs, axis=0)  # [B, Q, C]
        
        # 转换回 torch (如果输入是 torch)
        if HAS_TORCH and isinstance(feature_map, torch.Tensor):
            output = torch.from_numpy(output_np).to(feature_map.device)
            return output, total_stats
        
        return output_np, total_stats
    
    def _convert_inputs(
        self,
        feature_map,
        geo_params,
        weights,
        offsets
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Dict]:
        """转换输入为 numpy"""
        
        def to_numpy(x):
            if x is None:
                return None
            if HAS_TORCH and isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
            return np.asarray(x)
        
        feature_np = to_numpy(feature_map).astype(np.float16)
        weights_np = to_numpy(weights).astype(np.float16)
        offsets_np = to_numpy(offsets)
        if offsets_np is not None:
            offsets_np = offsets_np.astype(np.float16)
        
        # 处理 geo_params
        if isinstance(geo_params, dict):
            geo_dict = {
                'ref_xy': to_numpy(geo_params.get('ref_xy')),
                'stride_xy': to_numpy(geo_params.get('stride_xy')),
            }
        else:
            # 假设是 tensor [B, Q, param_size]
            params_np = to_numpy(geo_params)
            geo_dict = {
                'ref_xy': params_np[..., :2],
                'stride_xy': params_np[..., 2:4] if params_np.shape[-1] > 2 else np.zeros_like(params_np[..., :2]),
            }
            
        return feature_np, weights_np, offsets_np, geo_dict
    
    def _create_params_list(
        self,
        geo_dict: Dict,
        batch_idx: int,
        num_queries: int,
        num_samples: int
    ) -> List[ParamEntry]:
        """创建参数列表"""
        params_list = []
        
        ref_xy = geo_dict['ref_xy']
        stride_xy = geo_dict['stride_xy']
        
        # 处理维度
        if ref_xy.ndim == 3:
            ref_xy_b = ref_xy[batch_idx]
            stride_xy_b = stride_xy[batch_idx]
        else:
            ref_xy_b = ref_xy
            stride_xy_b = stride_xy
        
        for q in range(num_queries):
            params = ParamEntry(
                query_id=q,
                num_samples=num_samples,
                ref_x=float(ref_xy_b[q, 0]),
                ref_y=float(ref_xy_b[q, 1]),
                stride_x=float(stride_xy_b[q, 0]) if stride_xy_b is not None else 0.0,
                stride_y=float(stride_xy_b[q, 1]) if stride_xy_b is not None else 0.0,
                valid=True,
            )
            params_list.append(params)
            
        return params_list
    
    def get_stats(self) -> Dict:
        """获取最后一次执行的统计"""
        return self._last_stats.copy()
    
    def estimate_performance(
        self,
        batch_size: int = 2,
        num_queries: int = 1024,
        num_samples: int = 512,
        num_views: int = 4,
        row_hit_rate: float = 0.7,
        gpu_baseline_ms: Optional[float] = None
    ) -> Dict:
        """
        估算性能 (不实际执行)
        
        Args:
            batch_size: Batch 大小
            num_queries: Query 数量
            num_samples: 每 query 采样点数
            num_views: View 数量
            row_hit_rate: 预估 row hit rate
            gpu_baseline_ms: GPU baseline 时间 (可选，用于计算加速比)
            
        Returns:
            性能估算
        """
        total_samples = batch_size * num_queries * num_samples * num_views
        
        # 每 tile 周期估算
        avg_fetch = (row_hit_rate * self.config.row_hit_latency + 
                     (1 - row_hit_rate) * self.config.row_miss_latency)
        compute_cycles = 5  # bilinear (4 cycles) + accum (1 cycle)
        tile_cycles = max(avg_fetch, compute_cycles)
        
        # 每 sample 周期
        tiles_per_sample = self.config.total_c // self.config.tile_c
        sample_cycles = tiles_per_sample * tile_cycles
        
        # 单 bank 吞吐
        single_bank_throughput = self.config.pim_freq_mhz * 1e6 / sample_cycles
        
        # 多 bank 并行
        total_throughput = single_bank_throughput * self.config.num_active_banks
        
        # 总延迟
        latency_sec = total_samples / total_throughput
        latency_ms = latency_sec * 1000
        
        result = {
            'total_samples': total_samples,
            'sample_cycles': int(sample_cycles),
            'single_bank_throughput': single_bank_throughput,
            'total_throughput': total_throughput,
            'estimated_ms': latency_ms,
            'row_hit_rate': row_hit_rate,
        }
        
        # 只有在提供了 GPU baseline 时才计算加速比
        if gpu_baseline_ms is not None:
            result['gpu_baseline_ms'] = gpu_baseline_ms
            result['speedup'] = gpu_baseline_ms / latency_ms
        
        return result

