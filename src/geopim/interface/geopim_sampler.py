"""
GeoPIM PyTorch 采样器

提供 PyTorch 接口，支持与 TransPlat/PixelSplat 集成。
"""

from typing import Optional, Dict, Tuple

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    
import numpy as np

from ..simulator.geopim_simulator import GeoPIMSimulator, SimulatorConfig


if HAS_TORCH:
    class GeoPIMSampler(torch.autograd.Function):
        """
        GeoPIM v3.0 采样器 (PyTorch autograd Function)
        
        统一预计算权重路径:
        - 对于所有方法（包括 PixelSplat），权重都由 GPU 预计算
        - PIM 只负责加权采样和聚合
        """
        
        @staticmethod
        def forward(
            ctx,
            feature_map: torch.Tensor,
            geo_params: torch.Tensor,
            weights: torch.Tensor,
            offsets: Optional[torch.Tensor] = None,
            simulator: Optional[GeoPIMSimulator] = None
        ) -> torch.Tensor:
            """
            前向传播
            
            Args:
                feature_map: [B, C, H, W] - FP16 特征图
                geo_params: [B, Q, param_size] - 几何参数
                weights: [B, Q, S] - 预计算权重 (GPU 已算好 Q·K→softmax)
                offsets: [B, Q, S, 2] - 采样偏移 (optional)
                simulator: GeoPIM 模拟器实例
                
            Returns:
                output: [B, Q, C] - FP16 聚合结果
            """
            if simulator is None:
                simulator = GeoPIMSimulator.get_default()
            
            # 执行模拟
            output, stats = simulator.execute(
                feature_map.contiguous(),
                geo_params.contiguous(),
                weights.contiguous(),
                offsets.contiguous() if offsets is not None else None
            )
            
            # 保存用于反向传播
            ctx.save_for_backward(feature_map, geo_params, weights, offsets)
            ctx.stats = stats
            ctx.simulator = simulator
            
            return output
        
        @staticmethod
        def backward(ctx, grad_output):
            """
            反向传播 (未实现 - 仅用于推理)
            """
            raise NotImplementedError(
                "GeoPIM backward not implemented. "
                "Use GeoPIM only for inference, not training."
            )


    class GeoPIMSamplerModule(nn.Module):
        """
        GeoPIM 采样器 Module
        
        提供更友好的接口和配置选项。
        """
        
        def __init__(
            self,
            simulator_config: Optional[SimulatorConfig] = None,
            enable_stats: bool = True
        ):
            super().__init__()
            self.simulator = GeoPIMSimulator(simulator_config)
            self.enable_stats = enable_stats
            self._last_stats: Dict = {}
        
        def forward(
            self,
            feature_map: torch.Tensor,
            geo_params: torch.Tensor,
            weights: torch.Tensor,
            offsets: Optional[torch.Tensor] = None
        ) -> torch.Tensor:
            """
            前向传播
            
            Args:
                feature_map: [B, C, H, W] - 特征图
                geo_params: [B, Q, param_size] 或 dict - 几何参数
                weights: [B, Q, S] - 预计算权重
                offsets: [B, Q, S, 2] - 采样偏移 (optional)
                
            Returns:
                output: [B, Q, C]
            """
            output, stats = self.simulator.execute(
                feature_map, geo_params, weights, offsets
            )
            
            if self.enable_stats:
                self._last_stats = stats
            
            return output
        
        def get_stats(self) -> Dict:
            """获取最后一次执行的统计"""
            return self._last_stats.copy()
        
        def estimate_performance(self, **kwargs) -> Dict:
            """估算性能"""
            return self.simulator.estimate_performance(**kwargs)
        
        def __repr__(self) -> str:
            return (f"GeoPIMSamplerModule("
                    f"num_banks={self.simulator.config.num_active_banks}, "
                    f"tile_c={self.simulator.config.tile_c})")

else:
    # 无 PyTorch 时的占位类
    class GeoPIMSampler:
        """GeoPIM Sampler (requires PyTorch)"""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for GeoPIMSampler")
    
    class GeoPIMSamplerModule:
        """GeoPIM Sampler Module (requires PyTorch)"""
        def __init__(self, *args, **kwargs):
            raise ImportError("PyTorch is required for GeoPIMSamplerModule")


def transplat_with_geopim(
    query: "torch.Tensor",
    geo_params: "torch.Tensor",
    feature_map: "torch.Tensor",
    offsets: "torch.Tensor",
    weight_predictor: nn.Module,
    simulator: Optional[GeoPIMSimulator] = None
) -> "torch.Tensor":
    """
    TransPlat 使用 GeoPIM 的完整流程
    
    权重直接由 query 通过 MLP 预测，不依赖采样值。
    
    Args:
        query: [B, Q, C] - Query 特征
        geo_params: [B, Q, param_size] - 几何参数
        feature_map: [B, C, H, W] - 特征图
        offsets: [B, Q, S, 2] - 采样偏移
        weight_predictor: 权重预测网络
        simulator: GeoPIM 模拟器
        
    Returns:
        output: [B, Q, C] - 聚合特征
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required")
    
    # Step 1: GPU 预测权重
    weights = weight_predictor(query)  # [B, Q, S]
    
    # Step 2: PIM 执行加权采样
    output = GeoPIMSampler.apply(
        feature_map, geo_params, weights, offsets, simulator
    )
    
    return output


def pixelsplat_with_geopim(
    query: "torch.Tensor",
    key_features: "torch.Tensor",
    geo_params: "torch.Tensor",
    feature_map: "torch.Tensor",
    simulator: Optional[GeoPIMSimulator] = None
) -> "torch.Tensor":
    """
    PixelSplat 使用 GeoPIM 的完整流程
    
    1. GPU: 采样 key features (用于计算 Q·K)
    2. GPU: 计算 Q·K → softmax → weights
    3. PIM: 使用预计算 weights 做加权采样
    
    Args:
        query: [B, Q, C] - Query 特征
        key_features: [B, C_key, H, W] - Key 特征图
        geo_params: [B, Q, param_size] - 几何参数
        feature_map: [B, C, H, W] - Value 特征图
        simulator: GeoPIM 模拟器
        
    Returns:
        output: [B, Q, C] - 聚合特征
    """
    if not HAS_TORCH:
        raise ImportError("PyTorch is required")
    
    import torch.nn.functional as F
    
    # Step 1: GPU 采样 key features (轻量级)
    # 简化实现: 使用 grid_sample
    B, Q, _ = query.shape
    
    # 从 geo_params 提取采样坐标
    ref_xy = geo_params[:, :, :2]  # [B, Q, 2]
    S = 32  # PixelSplat 默认采样点数
    
    # 生成采样网格
    grid = ref_xy.unsqueeze(2).expand(-1, -1, S, -1)  # [B, Q, S, 2]
    grid = grid.view(B, Q * S, 1, 2)
    
    # 归一化坐标
    H, W = key_features.shape[2:]
    grid[..., 0] = 2 * grid[..., 0] / (W - 1) - 1
    grid[..., 1] = 2 * grid[..., 1] / (H - 1) - 1
    
    key_samples = F.grid_sample(
        key_features, grid, mode='bilinear', 
        padding_mode='zeros', align_corners=True
    )
    key_samples = key_samples.view(B, -1, Q, S).permute(0, 2, 3, 1)  # [B, Q, S, C_key]
    
    # Step 2: GPU 计算 attention weights
    scores = torch.einsum('bqc,bqsc->bqs', query, key_samples)  # [B, Q, S]
    weights = F.softmax(scores, dim=-1)  # [B, Q, S]
    
    # Step 3: PIM 执行加权采样
    output = GeoPIMSampler.apply(
        feature_map, geo_params, weights, None, simulator
    )
    
    return output

