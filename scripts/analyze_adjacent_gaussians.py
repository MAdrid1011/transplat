#!/usr/bin/env python3
"""
相邻像素高斯基元相似性分析脚本

用于分析可泛化3DGS模型中相邻像素生成的高斯基元的相似程度，
评估高斯合并/压缩的潜力。

功能:
- 分析相邻像素高斯的位置、形状、颜色、不透明度相似性
- 估算可合并高斯的比例
- 计算潜在的 HBM 流量节省

使用方法:
    通过 run_all_timing_tests.sh 调用:
    bash scripts/run_all_timing_tests.sh transplat --analyze-adjacent
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional
import torch
import numpy as np


@dataclass
class AdjacentGaussianStats:
    """单个场景的相邻高斯统计"""
    scene_name: str
    total_gaussians: int
    n_views: int
    height: int
    width: int
    
    # 位置相似性
    pos_diff_mean: float
    pos_diff_median: float
    scene_scale: float
    
    # 协方差相似性
    cov_diff_mean: float
    cov_diff_median: float
    cov_scale: float
    
    # Opacity 相似性
    opacity_diff_mean: float
    opacity_diff_median: float
    
    # 球谐系数相似性
    sh_diff_mean: float
    sh_diff_median: float
    sh_scale: float
    
    # 综合可合并估算
    mergeable_loose: float  # 宽松标准
    mergeable_medium: float  # 中等标准
    mergeable_strict: float  # 严格标准
    
    # HBM 节省估算
    estimated_reduction_ratio: float
    estimated_hbm_saving_mb: float
    
    # 分布数据 (带默认值的放在最后)
    rel_pos_dist: Dict[float, float] = field(default_factory=dict)  # threshold -> percentage
    rel_cov_dist: Dict[float, float] = field(default_factory=dict)
    opacity_dist: Dict[float, float] = field(default_factory=dict)
    rel_sh_dist: Dict[float, float] = field(default_factory=dict)


class AdjacentGaussianAnalyzer:
    """相邻高斯相似性分析器"""
    
    def __init__(self):
        self.scene_stats: List[AdjacentGaussianStats] = []
    
    def analyze_scene(self, scene_name: str, gaussians) -> AdjacentGaussianStats:
        """分析单个场景的相邻高斯相似性"""
        means = gaussians.means[0]  # [N, 3]
        covariances = gaussians.covariances[0]  # [N, 3, 3]
        opacities = gaussians.opacities[0]  # [N]
        harmonics = gaussians.harmonics[0]  # [N, 3, K]
        
        N = means.shape[0]
        # 推断视图数和分辨率
        # 假设是 256x256 双视图 (可以从 N 推断)
        possible_resolutions = [(256, 256, 2), (128, 128, 2), (64, 64, 2)]
        for H, W, n_views in possible_resolutions:
            if N == H * W * n_views:
                break
        else:
            # 默认假设
            n_views = 2
            pixels_per_view = N // n_views
            H = W = int(np.sqrt(pixels_per_view))
        
        # 重塑为 [n_views, H, W, ...]
        means_2d = means.view(n_views, H, W, 3)
        cov_2d = covariances.view(n_views, H, W, 3, 3)
        op_2d = opacities.view(n_views, H, W)
        sh_2d = harmonics.view(n_views, H, W, harmonics.shape[1], harmonics.shape[2])
        
        # === 位置差异 ===
        h_pos_diff = (means_2d[:, :, 1:, :] - means_2d[:, :, :-1, :]).norm(dim=-1)
        v_pos_diff = (means_2d[:, 1:, :, :] - means_2d[:, :-1, :, :]).norm(dim=-1)
        all_pos_diff = torch.cat([h_pos_diff.flatten(), v_pos_diff.flatten()])
        
        scene_scale = means.std().item()
        rel_pos_diff = all_pos_diff / scene_scale
        
        rel_pos_dist = {}
        for t in [0.01, 0.02, 0.05, 0.1, 0.2, 0.5]:
            rel_pos_dist[t] = (rel_pos_diff < t).float().mean().item()
        
        # === 协方差差异 ===
        h_cov_diff = (cov_2d[:, :, 1:, :, :] - cov_2d[:, :, :-1, :, :]).norm(dim=(-2,-1))
        v_cov_diff = (cov_2d[:, 1:, :, :, :] - cov_2d[:, :-1, :, :, :]).norm(dim=(-2,-1))
        all_cov_diff = torch.cat([h_cov_diff.flatten(), v_cov_diff.flatten()])
        
        cov_scale = covariances.norm(dim=(-2,-1)).mean().item()
        rel_cov_diff = all_cov_diff / cov_scale
        
        rel_cov_dist = {}
        for t in [0.01, 0.05, 0.1, 0.2, 0.5]:
            rel_cov_dist[t] = (rel_cov_diff < t).float().mean().item()
        
        # === Opacity 差异 ===
        h_op_diff = (op_2d[:, :, 1:] - op_2d[:, :, :-1]).abs()
        v_op_diff = (op_2d[:, 1:, :] - op_2d[:, :-1, :]).abs()
        all_op_diff = torch.cat([h_op_diff.flatten(), v_op_diff.flatten()])
        
        opacity_dist = {}
        for t in [0.01, 0.05, 0.1, 0.2]:
            opacity_dist[t] = (all_op_diff < t).float().mean().item()
        
        # === 球谐系数差异 ===
        h_sh_diff = (sh_2d[:, :, 1:, :, :] - sh_2d[:, :, :-1, :, :]).norm(dim=(-2,-1))
        v_sh_diff = (sh_2d[:, 1:, :, :, :] - sh_2d[:, :-1, :, :, :]).norm(dim=(-2,-1))
        all_sh_diff = torch.cat([h_sh_diff.flatten(), v_sh_diff.flatten()])
        
        sh_scale = harmonics.norm(dim=(-2,-1)).mean().item()
        rel_sh_diff = all_sh_diff / sh_scale
        
        rel_sh_dist = {}
        for t in [0.01, 0.05, 0.1, 0.2, 0.5]:
            rel_sh_dist[t] = (rel_sh_diff < t).float().mean().item()
        
        # === 综合可合并估算 ===
        # 宽松: 位置<10%, 协方差<20%, opacity<0.1, SH<20%
        mergeable_loose = (
            (rel_pos_diff < 0.1) & 
            (rel_cov_diff < 0.2) & 
            (all_op_diff < 0.1) &
            (rel_sh_diff < 0.2)
        ).float().mean().item()
        
        # 中等: 位置<5%, 协方差<10%, opacity<0.05, SH<10%
        mergeable_medium = (
            (rel_pos_diff < 0.05) & 
            (rel_cov_diff < 0.1) & 
            (all_op_diff < 0.05) &
            (rel_sh_diff < 0.1)
        ).float().mean().item()
        
        # 严格: 位置<2%, 协方差<5%, opacity<0.02, SH<5%
        mergeable_strict = (
            (rel_pos_diff < 0.02) & 
            (rel_cov_diff < 0.05) & 
            (all_op_diff < 0.02) &
            (rel_sh_diff < 0.05)
        ).float().mean().item()
        
        # HBM 节省估算 (使用中等标准)
        estimated_reduction_ratio = mergeable_medium * 0.5  # 每对可合并贡献 50% 减少
        total_mb = N * 352 / 1024 / 1024
        estimated_hbm_saving_mb = total_mb * estimated_reduction_ratio
        
        stats = AdjacentGaussianStats(
            scene_name=scene_name,
            total_gaussians=N,
            n_views=n_views,
            height=H,
            width=W,
            # 位置
            pos_diff_mean=all_pos_diff.mean().item(),
            pos_diff_median=all_pos_diff.median().item(),
            scene_scale=scene_scale,
            # 协方差
            cov_diff_mean=all_cov_diff.mean().item(),
            cov_diff_median=all_cov_diff.median().item(),
            cov_scale=cov_scale,
            # Opacity
            opacity_diff_mean=all_op_diff.mean().item(),
            opacity_diff_median=all_op_diff.median().item(),
            # 球谐
            sh_diff_mean=all_sh_diff.mean().item(),
            sh_diff_median=all_sh_diff.median().item(),
            sh_scale=sh_scale,
            # 可合并估算
            mergeable_loose=mergeable_loose,
            mergeable_medium=mergeable_medium,
            mergeable_strict=mergeable_strict,
            # HBM 节省
            estimated_reduction_ratio=estimated_reduction_ratio,
            estimated_hbm_saving_mb=estimated_hbm_saving_mb,
            # 分布数据
            rel_pos_dist=rel_pos_dist,
            rel_cov_dist=rel_cov_dist,
            opacity_dist=opacity_dist,
            rel_sh_dist=rel_sh_dist,
        )
        
        self.scene_stats.append(stats)
        return stats
    
    def print_scene_report(self, stats: AdjacentGaussianStats):
        """打印单个场景的分析报告"""
        print(f"\n  ========== 相邻高斯相似性分析: {stats.scene_name} ==========")
        print(f"  高斯数: {stats.total_gaussians:,}, 视图: {stats.n_views}, 分辨率: {stats.height}x{stats.width}")
        
        print(f"\n  【位置差异】 场景尺度: {stats.scene_scale:.4f}")
        print(f"    mean={stats.pos_diff_mean:.4f}, median={stats.pos_diff_median:.4f}")
        for t, pct in sorted(stats.rel_pos_dist.items()):
            print(f"    < {t*100:.0f}% 场景尺度: {pct*100:.1f}%")
        
        print(f"\n  【协方差差异】 尺度: {stats.cov_scale:.6f}")
        for t, pct in sorted(stats.rel_cov_dist.items()):
            print(f"    < {t*100:.0f}%: {pct*100:.1f}%")
        
        print(f"\n  【Opacity 差异】")
        for t, pct in sorted(stats.opacity_dist.items()):
            print(f"    < {t}: {pct*100:.1f}%")
        
        print(f"\n  【球谐(颜色)差异】 尺度: {stats.sh_scale:.4f}")
        for t, pct in sorted(stats.rel_sh_dist.items()):
            print(f"    < {t*100:.0f}%: {pct*100:.1f}%")
        
        print(f"\n  【可合并估算】")
        print(f"    宽松 (位置<10%, 协方差<20%, opacity<0.1, SH<20%): {stats.mergeable_loose*100:.1f}%")
        print(f"    中等 (位置<5%, 协方差<10%, opacity<0.05, SH<10%): {stats.mergeable_medium*100:.1f}%")
        print(f"    严格 (位置<2%, 协方差<5%, opacity<0.02, SH<5%): {stats.mergeable_strict*100:.1f}%")
    
    def print_summary_report(self, model_name: str):
        """打印所有场景的汇总报告"""
        if not self.scene_stats:
            print("[Warning] No adjacent Gaussian stats collected")
            return
        
        n_scenes = len(self.scene_stats)
        
        # 汇总统计
        avg_gaussians = np.mean([s.total_gaussians for s in self.scene_stats])
        avg_scene_scale = np.mean([s.scene_scale for s in self.scene_stats])
        
        # 位置差异汇总
        avg_rel_pos_dist = {}
        for t in [0.01, 0.02, 0.05, 0.1, 0.2]:
            avg_rel_pos_dist[t] = np.mean([s.rel_pos_dist.get(t, 0) for s in self.scene_stats])
        
        # 可合并估算汇总
        avg_mergeable_loose = np.mean([s.mergeable_loose for s in self.scene_stats])
        avg_mergeable_medium = np.mean([s.mergeable_medium for s in self.scene_stats])
        avg_mergeable_strict = np.mean([s.mergeable_strict for s in self.scene_stats])
        
        # HBM 节省估算
        avg_reduction = np.mean([s.estimated_reduction_ratio for s in self.scene_stats])
        avg_saving_mb = np.mean([s.estimated_hbm_saving_mb for s in self.scene_stats])
        total_mb = avg_gaussians * 352 / 1024 / 1024
        
        print("\n" + "=" * 100)
        print(f"  ███ {model_name.upper()} 相邻高斯相似性分析总结 (Challenge 2) ███")
        print("=" * 100)
        
        print(f"\n  统计场景数: {n_scenes}")
        print(f"  平均高斯数: {avg_gaussians:,.0f}")
        print(f"  平均场景尺度: {avg_scene_scale:.4f}")
        
        print(f"\n  【相邻高斯位置相似性】")
        print(f"  ┌{'─'*20}┬{'─'*20}┐")
        print(f"  │{'相对距离阈值':^20}│{'相邻对比例':^20}│")
        print(f"  ├{'─'*20}┼{'─'*20}┤")
        for t in [0.01, 0.02, 0.05, 0.1, 0.2]:
            pct = avg_rel_pos_dist.get(t, 0) * 100
            print(f"  │  < {t*100:>4.0f}% 场景尺度  │{pct:>17.1f}%  │")
        print(f"  └{'─'*20}┴{'─'*20}┘")
        
        print(f"\n  【综合可合并估算】")
        print(f"  ┌{'─'*50}┬{'─'*20}┐")
        print(f"  │{'合并标准':^50}│{'可合并相邻对':^20}│")
        print(f"  ├{'─'*50}┼{'─'*20}┤")
        print(f"  │  宽松 (位置<10%, 协方差<20%, opacity<0.1, SH<20%)  │{avg_mergeable_loose*100:>17.1f}%  │")
        print(f"  │  中等 (位置<5%, 协方差<10%, opacity<0.05, SH<10%)  │{avg_mergeable_medium*100:>17.1f}%  │")
        print(f"  │  严格 (位置<2%, 协方差<5%, opacity<0.02, SH<5%)    │{avg_mergeable_strict*100:>17.1f}%  │")
        print(f"  └{'─'*50}┴{'─'*20}┘")
        
        print(f"\n  【HBM 节省估算 (中等标准)】")
        print(f"  原始数据量: {total_mb:.2f} MB")
        print(f"  可合并相邻对: {avg_mergeable_medium*100:.1f}%")
        print(f"  估算减少比例: {avg_reduction*100:.1f}%")
        print(f"  估算节省: {avg_saving_mb:.2f} MB")
        
        # 计算合并后高斯数
        new_gaussian_count = int(avg_gaussians * (1 - avg_reduction))
        print(f"\n  【合并前后对比】")
        print(f"  合并前高斯数: {avg_gaussians:,.0f}")
        print(f"  合并后高斯数: {new_gaussian_count:,} (减少 {avg_reduction*100:.1f}%)")
        
        print("=" * 100)
        print()


# 单例模式
_analyzer_instance: Optional[AdjacentGaussianAnalyzer] = None


def get_adjacent_analyzer() -> AdjacentGaussianAnalyzer:
    """获取全局分析器实例"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = AdjacentGaussianAnalyzer()
    return _analyzer_instance


def reset_adjacent_analyzer():
    """重置分析器"""
    global _analyzer_instance
    _analyzer_instance = None
