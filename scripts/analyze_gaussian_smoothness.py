#!/usr/bin/env python3
"""
高斯基元 K-NN 离散度分析脚本 (K-Nearest Neighbor Gaussian Dispersion)

分析 Encoder 生成的高斯基元在 3D 空间中的局部离散程度，生成热力图：
- 红色区域：离散度高，高斯差异大，需要保留细节（小范围合并）
- 蓝色区域：离散度低，高斯相似，可以大规模合并

K-NN Gaussian Dispersion 定义：
  对每个高斯，计算其与 K 个最近邻高斯在位置、形状、颜色、不透明度上的综合差异

使用方法:
    通过 run_all_timing_tests.sh 调用:
    bash scripts/run_all_timing_tests.sh transplat --analyze-smoothness
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from pathlib import Path
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.colors as mcolors
from scipy.ndimage import zoom as scipy_zoom


@dataclass
class SmoothnessStats:
    """单个场景的平滑度统计"""
    scene_name: str
    total_gaussians: int
    n_views: int
    height: int
    width: int
    
    # 变化度统计
    variability_mean: float
    variability_std: float
    variability_min: float
    variability_max: float
    
    # 分层统计
    smooth_ratio: float      # 平滑区域比例 (变化度 < 0.2)
    medium_ratio: float      # 中等区域比例 (0.2 <= 变化度 < 0.5)
    complex_ratio: float     # 复杂区域比例 (变化度 >= 0.5)
    
    # 合并潜力估算
    smooth_merge_potential: float   # 平滑区域合并潜力 (可合并 ~80%)
    medium_merge_potential: float   # 中等区域合并潜力 (可合并 ~50%)
    complex_merge_potential: float  # 复杂区域合并潜力 (可合并 ~20%)
    total_merge_potential: float    # 总体合并潜力


class GaussianSmoothnessAnalyzer:
    """高斯基元空间变化度分析器"""
    
    def __init__(self, output_dir: str = "outputs/smoothness_analysis"):
        self.scene_stats: List[SmoothnessStats] = []
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 保存用于可视化的数据
        self.last_positions = None
        self.last_variability = None
        self.last_scene_name = None
        
        # 统一 3D 高斯场的数据
        self.last_all_positions = None  # [N, 3] 所有高斯的 3D 位置
        self.last_3d_variability = None  # [N] 每个高斯在 3D 空间中的变化度
        
        # 图像和相机参数（用于生成完整的可视化）
        self.last_context_images = None  # [B, V, C, H, W]
        self.last_context_extrinsics = None  # [B, V, 4, 4]
        self.last_context_intrinsics = None  # [B, V, 3, 3]
        self.last_target_images = None  # [B, V, C, H, W]
        self.last_target_extrinsics = None  # [B, V, 4, 4]
        self.last_target_intrinsics = None  # [B, V, 3, 3]
    
    def compute_local_variability(self, gaussians, kernel_size: int = 3) -> torch.Tensor:
        """
        计算每个高斯的局部变化度
        
        变化度 = 该高斯与其邻居在所有维度上的综合差异
        """
        means = gaussians.means[0]  # [N, 3]
        covariances = gaussians.covariances[0]  # [N, 3, 3]
        opacities = gaussians.opacities[0]  # [N]
        harmonics = gaussians.harmonics[0]  # [N, 3, K]
        
        N = means.shape[0]
        # 推断布局
        n_views = 2
        pixels_per_view = N // n_views
        H = W = int(np.sqrt(pixels_per_view))
        
        # 重塑为 2D 网格
        means_2d = means.view(n_views, H, W, 3)
        cov_2d = covariances.view(n_views, H, W, 3, 3)
        op_2d = opacities.view(n_views, H, W)
        sh_2d = harmonics.view(n_views, H, W, harmonics.shape[1], harmonics.shape[2])
        
        # 计算尺度用于归一化
        scene_scale = means.std()
        cov_scale = covariances.norm(dim=(-2,-1)).mean()
        sh_scale = harmonics.norm(dim=(-2,-1)).mean()
        
        # 初始化变化度张量
        variability = torch.zeros(n_views, H, W, device=means.device)
        
        # 使用滑动窗口计算局部变化度
        pad = kernel_size // 2
        
        for v in range(n_views):
            for i in range(H):
                for j in range(W):
                    # 获取邻域范围
                    i_min, i_max = max(0, i - pad), min(H, i + pad + 1)
                    j_min, j_max = max(0, j - pad), min(W, j + pad + 1)
                    
                    # 当前高斯
                    curr_pos = means_2d[v, i, j]
                    curr_cov = cov_2d[v, i, j]
                    curr_op = op_2d[v, i, j]
                    curr_sh = sh_2d[v, i, j]
                    
                    # 邻域高斯
                    neighbor_pos = means_2d[v, i_min:i_max, j_min:j_max]
                    neighbor_cov = cov_2d[v, i_min:i_max, j_min:j_max]
                    neighbor_op = op_2d[v, i_min:i_max, j_min:j_max]
                    neighbor_sh = sh_2d[v, i_min:i_max, j_min:j_max]
                    
                    # 计算与邻居的差异
                    pos_diff = (neighbor_pos - curr_pos).norm(dim=-1) / scene_scale
                    cov_diff = (neighbor_cov - curr_cov).norm(dim=(-2,-1)) / cov_scale
                    op_diff = (neighbor_op - curr_op).abs()
                    sh_diff = (neighbor_sh - curr_sh).norm(dim=(-2,-1)) / sh_scale
                    
                    # 综合变化度 (加权平均)
                    local_var = (
                        0.4 * pos_diff.mean() +  # 位置权重最高
                        0.2 * cov_diff.mean() +
                        0.2 * op_diff.mean() +
                        0.2 * sh_diff.mean()
                    )
                    
                    variability[v, i, j] = local_var
        
        return variability, (n_views, H, W)
    
    def compute_local_variability_fast(self, gaussians) -> Tuple[torch.Tensor, Tuple[int, int, int]]:
        """
        快速计算局部变化度 (使用卷积)
        """
        means = gaussians.means[0]
        covariances = gaussians.covariances[0]
        opacities = gaussians.opacities[0]
        harmonics = gaussians.harmonics[0]
        
        N = means.shape[0]
        n_views = 2
        pixels_per_view = N // n_views
        H = W = int(np.sqrt(pixels_per_view))
        
        # 重塑
        means_2d = means.view(n_views, H, W, 3)
        cov_2d = covariances.view(n_views, H, W, 3, 3)
        op_2d = opacities.view(n_views, H, W)
        sh_2d = harmonics.view(n_views, H, W, harmonics.shape[1], harmonics.shape[2])
        
        # 尺度
        scene_scale = means.std() + 1e-6
        cov_scale = covariances.norm(dim=(-2,-1)).mean() + 1e-6
        sh_scale = harmonics.norm(dim=(-2,-1)).mean() + 1e-6
        
        variability = torch.zeros(n_views, H, W, device=means.device)
        
        for v in range(n_views):
            # 位置变化度：与4邻域的差异
            pos = means_2d[v]  # [H, W, 3]
            
            # 水平差异
            h_diff = torch.zeros_like(pos[:, :, 0])
            h_diff[:, :-1] += (pos[:, 1:] - pos[:, :-1]).norm(dim=-1)
            h_diff[:, 1:] += (pos[:, :-1] - pos[:, 1:]).norm(dim=-1)
            
            # 垂直差异
            v_diff = torch.zeros_like(pos[:, :, 0])
            v_diff[:-1, :] += (pos[1:, :] - pos[:-1, :]).norm(dim=-1)
            v_diff[1:, :] += (pos[:-1, :] - pos[1:, :]).norm(dim=-1)
            
            pos_var = (h_diff + v_diff) / 4 / scene_scale
            
            # 协方差变化度
            cov = cov_2d[v]  # [H, W, 3, 3]
            h_cov_diff = torch.zeros(H, W, device=means.device)
            h_cov_diff[:, :-1] += (cov[:, 1:] - cov[:, :-1]).norm(dim=(-2,-1))
            h_cov_diff[:, 1:] += (cov[:, :-1] - cov[:, 1:]).norm(dim=(-2,-1))
            
            v_cov_diff = torch.zeros(H, W, device=means.device)
            v_cov_diff[:-1, :] += (cov[1:, :] - cov[:-1, :]).norm(dim=(-2,-1))
            v_cov_diff[1:, :] += (cov[:-1, :] - cov[1:, :]).norm(dim=(-2,-1))
            
            cov_var = (h_cov_diff + v_cov_diff) / 4 / cov_scale
            
            # Opacity 变化度
            op = op_2d[v]
            h_op_diff = torch.zeros(H, W, device=means.device)
            h_op_diff[:, :-1] += (op[:, 1:] - op[:, :-1]).abs()
            h_op_diff[:, 1:] += (op[:, :-1] - op[:, 1:]).abs()
            
            v_op_diff = torch.zeros(H, W, device=means.device)
            v_op_diff[:-1, :] += (op[1:, :] - op[:-1, :]).abs()
            v_op_diff[1:, :] += (op[:-1, :] - op[1:, :]).abs()
            
            op_var = (h_op_diff + v_op_diff) / 4
            
            # SH 变化度
            sh = sh_2d[v]
            h_sh_diff = torch.zeros(H, W, device=means.device)
            h_sh_diff[:, :-1] += (sh[:, 1:] - sh[:, :-1]).norm(dim=(-2,-1))
            h_sh_diff[:, 1:] += (sh[:, :-1] - sh[:, 1:]).norm(dim=(-2,-1))
            
            v_sh_diff = torch.zeros(H, W, device=means.device)
            v_sh_diff[:-1, :] += (sh[1:, :] - sh[:-1, :]).norm(dim=(-2,-1))
            v_sh_diff[1:, :] += (sh[:-1, :] - sh[1:, :]).norm(dim=(-2,-1))
            
            sh_var = (h_sh_diff + v_sh_diff) / 4 / sh_scale
            
            # 综合变化度
            variability[v] = 0.4 * pos_var + 0.2 * cov_var + 0.2 * op_var + 0.2 * sh_var
        
        return variability, (n_views, H, W), means_2d
    
    def compute_3d_variability(self, gaussians, k_neighbors: int = 8) -> torch.Tensor:
        """
        计算每个高斯与其 K 近邻的余弦相似度（分别统计）
        
        返回协方差余弦相似度（用于热力图），同时打印三个指标的分布
        """
        means = gaussians.means[0]  # [N, 3]
        covariances = gaussians.covariances[0]  # [N, 3, 3]
        opacities = gaussians.opacities[0]  # [N]
        harmonics = gaussians.harmonics[0]  # [N, 3, K]
        
        N = means.shape[0]
        device = means.device
        
        # 展平
        cov_flat = covariances.view(N, -1)  # [N, 9]
        sh_flat = harmonics.view(N, -1)  # [N, C]
        
        # 归一化（用于余弦相似度）
        cov_norm = cov_flat / (cov_flat.norm(dim=-1, keepdim=True) + 1e-8)
        sh_norm = sh_flat / (sh_flat.norm(dim=-1, keepdim=True) + 1e-8)
        pos_norm = means / (means.norm(dim=-1, keepdim=True) + 1e-8)
        
        # 存储四个指标
        cov_sim_all = torch.zeros(N, device=device)
        sh_sim_all = torch.zeros(N, device=device)
        pos_sim_all = torch.zeros(N, device=device)
        op_diff_all = torch.zeros(N, device=device)  # 不透明度差异
        
        batch_size = 8192
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_pos = means[start:end]
            batch_pos_norm = pos_norm[start:end]
            batch_cov = cov_norm[start:end]
            batch_sh = sh_norm[start:end]
            batch_op = opacities[start:end]
            
            # 采样
            sample_size = min(2000, N)
            sample_idx = torch.randperm(N, device=device)[:sample_size]
            sample_pos = means[sample_idx]
            sample_pos_norm = pos_norm[sample_idx]
            sample_cov = cov_norm[sample_idx]
            sample_sh = sh_norm[sample_idx]
            sample_op = opacities[sample_idx]
            
            # 找 K 近邻（基于位置距离）
            pos_dist = torch.cdist(batch_pos, sample_pos)
            _, nearest_idx = pos_dist.topk(k_neighbors + 1, dim=1, largest=False)
            nearest_idx = nearest_idx[:, 1:]  # 排除自己
            
            B = end - start
            
            # 近邻属性
            nn_cov = sample_cov[nearest_idx.flatten()].view(B, k_neighbors, -1)
            nn_sh = sample_sh[nearest_idx.flatten()].view(B, k_neighbors, -1)
            nn_pos = sample_pos_norm[nearest_idx.flatten()].view(B, k_neighbors, -1)
            nn_op = sample_op[nearest_idx.flatten()].view(B, k_neighbors)
            
            # 1. 协方差余弦相似度
            cov_cos = torch.bmm(batch_cov.unsqueeze(1), nn_cov.transpose(1, 2)).squeeze(1)  # [B, k]
            cov_sim_all[start:end] = cov_cos.mean(dim=1)
            
            # 2. SH 余弦相似度
            sh_cos = torch.bmm(batch_sh.unsqueeze(1), nn_sh.transpose(1, 2)).squeeze(1)  # [B, k]
            sh_sim_all[start:end] = sh_cos.mean(dim=1)
            
            # 3. 位置余弦相似度
            pos_cos = torch.bmm(batch_pos_norm.unsqueeze(1), nn_pos.transpose(1, 2)).squeeze(1)  # [B, k]
            pos_sim_all[start:end] = pos_cos.mean(dim=1)
            
            # 4. 不透明度 L2 距离（绝对差）
            op_l2 = (nn_op - batch_op.unsqueeze(1)).abs()  # [B, k]
            op_diff_all[start:end] = op_l2.mean(dim=1)
        
        # 统计分布
        def print_stats(name, data):
            q = torch.tensor([0, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0], device=device)
            percentiles = torch.quantile(data, q)
            print(f"  [{name}]")
            print(f"    min={percentiles[0]:.4f}, 10%={percentiles[1]:.4f}, 25%={percentiles[2]:.4f}")
            print(f"    50%={percentiles[3]:.4f}, 75%={percentiles[4]:.4f}, 90%={percentiles[5]:.4f}")
            print(f"    95%={percentiles[6]:.4f}, max={percentiles[7]:.4f}")
        
        print("\n===== 四个相似度/差异分布 =====")
        print_stats("协方差 cos_sim", cov_sim_all)
        print_stats("球谐函数 cos_sim", sh_sim_all)
        print_stats("位置 cos_sim", pos_sim_all)
        print_stats("不透明度 L2_diff", op_diff_all)
        
        # 统计满足条件的交集（多组阈值）
        thresholds = [
            (0.97, 0.99, 0.1, "严格"),
            (0.7, 0.7, 0.2, "宽松"),
        ]
        
        print(f"\n===== 高相似度交集统计 =====")
        print(f"  总高斯数: {N}")
        
        for cov_thresh, sh_thresh, op_thresh, label in thresholds:
            cov_mask = cov_sim_all > cov_thresh
            sh_mask = sh_sim_all > sh_thresh
            op_mask = op_diff_all < op_thresh
            
            cov_count = cov_mask.sum().item()
            sh_count = sh_mask.sum().item()
            op_count = op_mask.sum().item()
            
            joint_mask = cov_mask & sh_mask & op_mask
            joint_count = joint_mask.sum().item()
            
            print(f"\n  [{label}阈值] cov>{cov_thresh}, sh>{sh_thresh}, op<{op_thresh}")
            print(f"    协方差 cos > {cov_thresh}: {cov_count} ({cov_count/N*100:.1f}%)")
            print(f"    球谐函数 cos > {sh_thresh}: {sh_count} ({sh_count/N*100:.1f}%)")
            print(f"    不透明度 L2 < {op_thresh}: {op_count} ({op_count/N*100:.1f}%)")
            print(f"    --> 三者交集: {joint_count} ({joint_count/N*100:.1f}%)")
        
        print("================================\n")
        
        # 保存到实例变量供后续使用
        self.cov_sim = cov_sim_all
        self.sh_sim = sh_sim_all
        self.pos_sim = pos_sim_all
        self.op_diff = op_diff_all
        
        # 返回协方差相似度用于默认热力图
        return cov_sim_all
    
    def analyze_scene(
        self, 
        scene_name: str, 
        gaussians,
        context_images: Optional[torch.Tensor] = None,
        context_extrinsics: Optional[torch.Tensor] = None,
        context_intrinsics: Optional[torch.Tensor] = None,
        target_images: Optional[torch.Tensor] = None,
        target_extrinsics: Optional[torch.Tensor] = None,
        target_intrinsics: Optional[torch.Tensor] = None,
    ) -> SmoothnessStats:
        """分析单个场景的高斯平滑度"""
        
        variability, (n_views, H, W), means_2d = self.compute_local_variability_fast(gaussians)
        
        # 计算统一 3D 空间的变化度
        variability_3d = self.compute_3d_variability(gaussians, k_neighbors=8)
        
        # 保存用于 2D 可视化
        self.last_positions = means_2d.cpu()
        self.last_variability = variability.cpu()
        self.last_scene_name = scene_name
        
        # 保存用于 3D 可视化（统一高斯场）
        self.last_all_positions = gaussians.means[0]  # [N, 3]
        self.last_3d_variability = variability_3d  # [N]
        
        # 保存图像和相机参数
        self.last_context_images = context_images.cpu() if context_images is not None else None
        self.last_context_extrinsics = context_extrinsics.cpu() if context_extrinsics is not None else None
        self.last_context_intrinsics = context_intrinsics.cpu() if context_intrinsics is not None else None
        self.last_target_images = target_images.cpu() if target_images is not None else None
        self.last_target_extrinsics = target_extrinsics.cpu() if target_extrinsics is not None else None
        self.last_target_intrinsics = target_intrinsics.cpu() if target_intrinsics is not None else None
        
        # 统计
        var_flat = variability.flatten()
        
        # 分层阈值
        smooth_thresh = 0.05
        medium_thresh = 0.15
        
        smooth_mask = var_flat < smooth_thresh
        medium_mask = (var_flat >= smooth_thresh) & (var_flat < medium_thresh)
        complex_mask = var_flat >= medium_thresh
        
        smooth_ratio = smooth_mask.float().mean().item()
        medium_ratio = medium_mask.float().mean().item()
        complex_ratio = complex_mask.float().mean().item()
        
        # 合并潜力估算
        # 平滑区域：80% 可合并（大范围合并）
        # 中等区域：50% 可合并（中等范围合并）
        # 复杂区域：20% 可合并（小范围合并）
        smooth_merge = smooth_ratio * 0.8
        medium_merge = medium_ratio * 0.5
        complex_merge = complex_ratio * 0.2
        total_merge = smooth_merge + medium_merge + complex_merge
        
        stats = SmoothnessStats(
            scene_name=scene_name,
            total_gaussians=gaussians.means.shape[1],
            n_views=n_views,
            height=H,
            width=W,
            variability_mean=var_flat.mean().item(),
            variability_std=var_flat.std().item(),
            variability_min=var_flat.min().item(),
            variability_max=var_flat.max().item(),
            smooth_ratio=smooth_ratio,
            medium_ratio=medium_ratio,
            complex_ratio=complex_ratio,
            smooth_merge_potential=smooth_merge,
            medium_merge_potential=medium_merge,
            complex_merge_potential=complex_merge,
            total_merge_potential=total_merge,
        )
        
        self.scene_stats.append(stats)
        return stats
    
    def generate_heatmap_2d(self, stats: SmoothnessStats, save_path: Optional[Path] = None):
        """生成 2D 变化度热力图"""
        if self.last_variability is None:
            return
        
        variability = self.last_variability.numpy()
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        for v in range(variability.shape[0]):
            ax = axes[v]
            im = ax.imshow(variability[v], cmap='RdYlBu', vmin=0, vmax=0.3)
            ax.set_title(f'View {v+1} - Gaussian Variability\n(Red=Smooth, Blue=Complex)')
            ax.set_xlabel('Width')
            ax.set_ylabel('Height')
            plt.colorbar(im, ax=ax, label='Variability')
        
        plt.suptitle(f'Scene: {stats.scene_name}\n'
                     f'Smooth: {stats.smooth_ratio*100:.1f}% | '
                     f'Medium: {stats.medium_ratio*100:.1f}% | '
                     f'Complex: {stats.complex_ratio*100:.1f}%')
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"{stats.scene_name}_heatmap_2d.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"  2D 热力图已保存: {save_path}")
    
    def generate_heatmap_3d(self, stats: SmoothnessStats, save_path: Optional[Path] = None,
                           downsample: int = 4, percentile_clip: float = 2.0):
        """生成 3D 变化度热力图 (合并所有视图)
        
        Args:
            percentile_clip: 裁剪掉位置在 percentile_clip% 和 (100-percentile_clip)% 之外的点
        """
        if self.last_all_positions is None or self.last_3d_variability is None:
            return
        
        positions = self.last_all_positions.cpu().numpy()  # [N, 3]
        variability = self.last_3d_variability.cpu().numpy()  # [N]
        
        # 过滤离群点：使用百分位数裁剪
        x_low, x_high = np.percentile(positions[:, 0], [percentile_clip, 100 - percentile_clip])
        y_low, y_high = np.percentile(positions[:, 1], [percentile_clip, 100 - percentile_clip])
        z_low, z_high = np.percentile(positions[:, 2], [percentile_clip, 100 - percentile_clip])
        
        # 创建掩码
        mask = (
            (positions[:, 0] >= x_low) & (positions[:, 0] <= x_high) &
            (positions[:, 1] >= y_low) & (positions[:, 1] <= y_high) &
            (positions[:, 2] >= z_low) & (positions[:, 2] <= z_high)
        )
        
        positions_filtered = positions[mask]
        variability_filtered = variability[mask]
        
        # 下采样
        step = downsample
        pos = positions_filtered[::step]
        var = variability_filtered[::step]
        
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        
        # 归一化颜色
        c_norm = np.clip(var / 0.3, 0, 1)
        
        filtered_ratio = (1 - mask.sum() / len(mask)) * 100
        print(f"  过滤离群点: {filtered_ratio:.1f}% (保留 {percentile_clip:.0f}-{100-percentile_clip:.0f}% 范围)")
        
        # 创建图形
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        # 使用 RdYlBu: 红色=低变化度(平滑)，蓝色=高变化度(复杂)
        scatter = ax.scatter(x, y, z, c=c_norm, cmap='RdYlBu', 
                            s=3, alpha=0.6, vmin=0, vmax=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Scene: {stats.scene_name}\n'
                     f'K-NN Gaussian Dispersion ({len(pos):,} / {stats.total_gaussians:,} Gaussians)\n'
                     f'Red=Low Dispersion (Mergeable), Blue=High Dispersion (Keep)')
        
        # 添加颜色条（显示 K-NN Gaussian Dispersion 数值）
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.6, pad=0.1, label='K-NN Gaussian Dispersion')
        cbar.set_ticks([0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0])
        cbar.set_ticklabels(['0', '0.05', '0.10', '0.15', '0.20', '0.25', '0.30+'])
        
        # 添加统计信息
        textstr = (f'Smooth: {stats.smooth_ratio*100:.1f}%\n'
                   f'Medium: {stats.medium_ratio*100:.1f}%\n'
                   f'Complex: {stats.complex_ratio*100:.1f}%\n'
                   f'Merge Potential: {stats.total_merge_potential*100:.1f}%')
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text2D(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                  verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = self.output_dir / f"{stats.scene_name}_heatmap_3d.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"  3D 统一高斯场热力图已保存: {save_path}")
    
    def generate_heatmap_3d_multiview(self, stats: SmoothnessStats, save_path: Optional[Path] = None,
                                       downsample: int = 4, percentile_clip: float = 2.0):
        """生成多角度 3D 热力图"""
        if self.last_all_positions is None or self.last_3d_variability is None:
            return
        
        positions = self.last_all_positions.cpu().numpy()
        variability = self.last_3d_variability.cpu().numpy()
        
        # 过滤离群点
        x_low, x_high = np.percentile(positions[:, 0], [percentile_clip, 100 - percentile_clip])
        y_low, y_high = np.percentile(positions[:, 1], [percentile_clip, 100 - percentile_clip])
        z_low, z_high = np.percentile(positions[:, 2], [percentile_clip, 100 - percentile_clip])
        
        mask = (
            (positions[:, 0] >= x_low) & (positions[:, 0] <= x_high) &
            (positions[:, 1] >= y_low) & (positions[:, 1] <= y_high) &
            (positions[:, 2] >= z_low) & (positions[:, 2] <= z_high)
        )
        
        positions_filtered = positions[mask]
        variability_filtered = variability[mask]
        
        step = downsample
        pos = positions_filtered[::step]
        var = variability_filtered[::step]
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        c_norm = np.clip(var / 0.3, 0, 1)
        
        # 创建 2x2 多角度视图
        fig = plt.figure(figsize=(16, 14))
        
        views = [
            (30, 45, 'View 1 (Default)'),
            (30, 135, 'View 2 (Rotated 90°)'),
            (60, 45, 'View 3 (Top-down)'),
            (0, 0, 'View 4 (Front)')
        ]
        
        for i, (elev, azim, title) in enumerate(views):
            ax = fig.add_subplot(2, 2, i + 1, projection='3d')
            scatter = ax.scatter(x, y, z, c=c_norm, cmap='RdYlBu', 
                                s=2, alpha=0.5, vmin=0, vmax=1)
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(title)
        
        plt.suptitle(f'Scene: {stats.scene_name} - K-NN Gaussian Dispersion (Multi-view)\n'
                     f'Total: {stats.total_gaussians:,} | Merge Potential: {stats.total_merge_potential*100:.1f}%',
                     fontsize=14)
        
        # 添加颜色条（显示 K-NN Gaussian Dispersion 数值）
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        sm = plt.cm.ScalarMappable(cmap='RdYlBu', norm=plt.Normalize(0, 1))
        cbar = fig.colorbar(sm, cax=cbar_ax, label='K-NN Dispersion')
        cbar.set_ticks([0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0])
        cbar.set_ticklabels(['0', '0.05', '0.10', '0.15', '0.20', '0.25', '0.30+'])
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        if save_path is None:
            save_path = self.output_dir / f"{stats.scene_name}_heatmap_3d_multiview.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"  多角度 3D 热力图已保存: {save_path}")
    
    def generate_comprehensive_visualization(self, stats: SmoothnessStats, 
                                             downsample: int = 4, percentile_clip: float = 2.0):
        """
        生成完整的 4 张可视化图：
        1. Context 图像（用于生成高斯的输入图之一）
        2. Target 图像（推理视角）
        3. 从推理视角看的 3D 热力图
        4. 将热力图映射到推理视角图像上的 overlay
        """
        if (self.last_all_positions is None or self.last_3d_variability is None or
            self.last_context_images is None or self.last_target_images is None or
            self.last_target_extrinsics is None or self.last_target_intrinsics is None):
            print("  [Warning] 缺少必要数据，无法生成完整可视化")
            return
        
        scene_name = stats.scene_name
        
        # 准备数据
        positions = self.last_all_positions.cpu().numpy()  # [N, 3]
        variability = self.last_3d_variability.cpu().numpy()  # [N]
        
        # 过滤离群点
        x_low, x_high = np.percentile(positions[:, 0], [percentile_clip, 100 - percentile_clip])
        y_low, y_high = np.percentile(positions[:, 1], [percentile_clip, 100 - percentile_clip])
        z_low, z_high = np.percentile(positions[:, 2], [percentile_clip, 100 - percentile_clip])
        
        mask = (
            (positions[:, 0] >= x_low) & (positions[:, 0] <= x_high) &
            (positions[:, 1] >= y_low) & (positions[:, 1] <= y_high) &
            (positions[:, 2] >= z_low) & (positions[:, 2] <= z_high)
        )
        
        positions_filtered = positions[mask]
        variability_filtered = variability[mask]
        
        # 下采样
        step = downsample
        pos = positions_filtered[::step]
        var = variability_filtered[::step]
        
        # 归一化到 [0, 1] 用于可视化
        # L2=0 → 0 (红色，相似), L2=0.5 → 1 (蓝色，差异大)
        c_norm = np.clip(var / 0.5, 0, 1)
        
        # 获取图像 (取第一个 batch, 第一个 view)
        context_img = self.last_context_images[0, 0].permute(1, 2, 0).numpy()  # [H, W, 3]
        context_img = np.clip(context_img, 0, 1)
        
        target_img = self.last_target_images[0, 0].permute(1, 2, 0).numpy()  # [H, W, 3]
        target_img = np.clip(target_img, 0, 1)
        
        # 获取 target 相机参数
        target_ext = self.last_target_extrinsics[0, 0].numpy()  # [4, 4]
        target_int = self.last_target_intrinsics[0, 0].numpy()  # [3, 3]
        
        # 计算从 target 相机视角的视角参数
        # 相机位置
        R = target_ext[:3, :3]
        t = target_ext[:3, 3]
        cam_pos = -R.T @ t  # 相机在世界坐标系中的位置
        
        # 相机朝向（光轴方向）
        cam_forward = R[2, :]  # 第三行是 z 轴方向
        
        # 计算 matplotlib 3D 视角参数
        # azimuth: 绕 z 轴旋转角度
        # elevation: 仰角
        azim = np.degrees(np.arctan2(cam_forward[1], cam_forward[0]))
        elev = np.degrees(np.arcsin(-cam_forward[2]))
        
        # ========== 图 1: Context 图像（无标题、无白色边框）==========
        fig1, ax1 = plt.subplots(figsize=(10, 8))
        ax1.imshow(context_img)
        ax1.axis('off')
        save_path1 = self.output_dir / f"{scene_name}_1_context_image.png"
        plt.savefig(save_path1, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"  图1 Context图像已保存: {save_path1}")
        
        # ========== 图 2: Target 图像（无标题、无白色边框）==========
        fig2, ax2 = plt.subplots(figsize=(10, 8))
        ax2.imshow(target_img)
        ax2.axis('off')
        save_path2 = self.output_dir / f"{scene_name}_2_target_image.png"
        plt.savefig(save_path2, dpi=300, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"  图2 Target图像已保存: {save_path2}")
        
        # ========== 图 3a: 协方差余弦相似度 3D 热力图 ==========
        if hasattr(self, 'cov_sim') and self.cov_sim is not None:
            cov_sim_np = self.cov_sim.cpu().numpy()
            cov_sim_filtered = cov_sim_np[mask][::step]
            
            fig3a = plt.figure(figsize=(10, 8))
            ax3a = fig3a.add_subplot(111, projection='3d')
            
            x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
            # 相似度 0→1 映射到颜色：红色=低相似度, 蓝色=高相似度
            scatter = ax3a.scatter(x, y, z, c=cov_sim_filtered, cmap='RdYlBu', 
                                  s=3, alpha=0.6, vmin=0, vmax=1)
            
            view_azim = azim + 15  # 相对于 target 视角旋转 15 度
            ax3a.view_init(elev=25, azim=view_azim)
            ax3a.set_xlabel('X')
            ax3a.set_ylabel('Y')
            ax3a.set_zlabel('Z')
            # 不要标题
            
            cbar = plt.colorbar(scatter, ax=ax3a, shrink=0.6, pad=0.02, label='Cosine Similarity')
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            
            fig3a.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            save_path3a = self.output_dir / f"{scene_name}_3a_heatmap_cov_sim.png"
            plt.savefig(save_path3a, dpi=300, bbox_inches='tight', pad_inches=0.02)
            plt.close()
            print(f"  图3a 协方差余弦相似度热力图已保存: {save_path3a}")
        
        # ========== 图 3b: 球谐函数余弦相似度 3D 热力图 ==========
        if hasattr(self, 'sh_sim') and self.sh_sim is not None:
            sh_sim_np = self.sh_sim.cpu().numpy()
            sh_sim_filtered = sh_sim_np[mask][::step]
            
            fig3b = plt.figure(figsize=(10, 8))
            ax3b = fig3b.add_subplot(111, projection='3d')
            
            # SH 相似度可能是负数，归一化到 [0, 1]：-1→0, 1→1
            sh_sim_norm = (sh_sim_filtered + 1) / 2
            scatter = ax3b.scatter(x, y, z, c=sh_sim_norm, cmap='RdYlBu', 
                                  s=3, alpha=0.6, vmin=0, vmax=1)
            
            ax3b.view_init(elev=25, azim=view_azim)
            ax3b.set_xlabel('X')
            ax3b.set_ylabel('Y')
            ax3b.set_zlabel('Z')
            # 不要标题
            
            cbar = plt.colorbar(scatter, ax=ax3b, shrink=0.6, pad=0.02, label='Cosine Similarity')
            cbar.set_ticks([0, 0.25, 0.5, 0.75, 1.0])
            cbar.set_ticklabels(['-1', '-0.5', '0', '0.5', '1'])
            
            fig3b.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            save_path3b = self.output_dir / f"{scene_name}_3b_heatmap_sh_sim.png"
            plt.savefig(save_path3b, dpi=300, bbox_inches='tight', pad_inches=0.02)
            plt.close()
            print(f"  图3b 球谐函数余弦相似度热力图已保存: {save_path3b}")
        
        # ========== 图 3-RGB: RGB 通道编码综合热力图 ==========
        if (hasattr(self, 'cov_sim') and self.cov_sim is not None and
            hasattr(self, 'sh_sim') and self.sh_sim is not None and
            hasattr(self, 'op_diff') and self.op_diff is not None):
            
            cov_sim_np = self.cov_sim.cpu().numpy()
            sh_sim_np = self.sh_sim.cpu().numpy()
            op_diff_np = self.op_diff.cpu().numpy()
            
            # 过滤和下采样
            cov_filtered = cov_sim_np[mask][::step]
            sh_filtered = sh_sim_np[mask][::step]
            op_filtered = op_diff_np[mask][::step]
            
            # RGB 编码：
            # R = 1 - cov_sim（协方差不相似 → 红）
            # G = 1 - sh_sim（球谐不相似 → 绿），sh_sim 范围是 [-1, 1]，先归一化到 [0, 1]
            # B = op_diff（不透明度差异大 → 蓝）
            
            r_channel = np.clip(1 - cov_filtered, 0, 1)
            g_channel = np.clip(1 - (sh_filtered + 1) / 2, 0, 1)  # [-1,1] -> [0,1] -> 反转
            b_channel = np.clip(op_filtered / 0.5, 0, 1)  # 归一化到 [0, 1]
            
            # 组合成 RGB 颜色
            colors = np.stack([r_channel, g_channel, b_channel], axis=1)
            
            # 设置字体为 Nimbus Roman (Times New Roman 替代)
            plt.rcParams['font.family'] = 'serif'
            plt.rcParams['font.serif'] = ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif']
            
            fig_rgb = plt.figure(figsize=(10, 8))
            ax_rgb = fig_rgb.add_subplot(111, projection='3d')
            ax_rgb.scatter(x, y, z, c=colors, s=3, alpha=0.6)
            # 旋转 15 度
            ax_rgb.view_init(elev=25, azim=azim + 15)
            ax_rgb.set_xlabel('X', fontsize=12, fontfamily='serif', labelpad=8)
            ax_rgb.set_ylabel('Y', fontsize=12, fontfamily='serif', labelpad=8)
            ax_rgb.set_zlabel('Z', fontsize=12, fontfamily='serif', labelpad=8)
            # 不要标题
            
            # 设置刻度标签字体
            ax_rgb.tick_params(axis='x', labelsize=10, pad=2)
            ax_rgb.tick_params(axis='y', labelsize=10, pad=2)
            ax_rgb.tick_params(axis='z', labelsize=10, pad=2)
            for label in ax_rgb.get_xticklabels() + ax_rgb.get_yticklabels() + ax_rgb.get_zticklabels():
                label.set_fontfamily('serif')
            
            # 调整布局确保 Z 轴标签可见
            fig_rgb.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            save_path_rgb = self.output_dir / f"{scene_name}_3_rgb_combined.png"
            plt.savefig(save_path_rgb, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.02)
            plt.close()
            print(f"  图3-RGB 综合热力图已保存: {save_path_rgb}")
            
            # ===== Target 视角的 RGB 热力图（在 target 基础上仰角+15度，纸面内旋转15度让Z轴更左倾）=====
            fig_rgb_target = plt.figure(figsize=(10, 8))
            ax_rgb_target = fig_rgb_target.add_subplot(111, projection='3d')
            ax_rgb_target.scatter(x, y, z, c=colors, s=3, alpha=0.6)
            # 仰角 +15 度，保持原方位角，roll=15 让 Z 轴向左倾斜
            ax_rgb_target.view_init(elev=elev + 15, azim=azim, roll=15)
            ax_rgb_target.set_xlabel('X', fontsize=12, fontfamily='serif', labelpad=8)
            ax_rgb_target.set_ylabel('Y', fontsize=12, fontfamily='serif', labelpad=8)
            ax_rgb_target.set_zlabel('Z', fontsize=12, fontfamily='serif', labelpad=8)
            # 不要标题
            
            # 设置刻度标签字体
            ax_rgb_target.tick_params(axis='x', labelsize=10, pad=2)
            ax_rgb_target.tick_params(axis='y', labelsize=10, pad=2)
            ax_rgb_target.tick_params(axis='z', labelsize=10, pad=2)
            for label in ax_rgb_target.get_xticklabels() + ax_rgb_target.get_yticklabels() + ax_rgb_target.get_zticklabels():
                label.set_fontfamily('serif')
            
            fig_rgb_target.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            save_path_rgb_target = self.output_dir / f"{scene_name}_3_rgb_target_view.png"
            plt.savefig(save_path_rgb_target, dpi=300, bbox_inches='tight', facecolor='white', pad_inches=0.02)
            plt.close()
            print(f"  图3-RGB Target视角热力图已保存: {save_path_rgb_target}")
            
            # ===== 单独的 RGB 色盘图例 =====
            fig_legend, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # 左图：8个角点色块
            ax_cube = axes[0]
            ax_cube.set_xlim(-0.5, 3.5)
            ax_cube.set_ylim(-0.5, 2.5)
            ax_cube.set_aspect('equal')
            ax_cube.axis('off')
            ax_cube.set_title('RGB Color Cube Corners', fontsize=14, fontweight='bold')
            
            corners = [
                (0, 2, (0, 0, 0), 'Black\n(All Similar)'),
                (1, 2, (1, 0, 0), 'Red\n(Cov diff)'),
                (2, 2, (0, 1, 0), 'Green\n(SH diff)'),
                (3, 2, (0, 0, 1), 'Blue\n(Op diff)'),
                (0, 0, (1, 1, 0), 'Yellow\n(Cov+SH)'),
                (1, 0, (1, 0, 1), 'Magenta\n(Cov+Op)'),
                (2, 0, (0, 1, 1), 'Cyan\n(SH+Op)'),
                (3, 0, (1, 1, 1), 'White\n(All diff)'),
            ]
            
            for cx, cy, color, label in corners:
                circle = plt.Circle((cx, cy), 0.35, color=color, ec='black', linewidth=2)
                ax_cube.add_patch(circle)
                ax_cube.text(cx, cy - 0.6, label, ha='center', va='top', fontsize=9)
            
            # 右图：三个渐变条
            ax_grad = axes[1]
            ax_grad.axis('off')
            ax_grad.set_title('RGB Gradient Slices', fontsize=14, fontweight='bold')
            
            n = 100
            gradient = np.linspace(0, 1, n).reshape(1, -1)
            
            ax_r = fig_legend.add_axes([0.55, 0.68, 0.35, 0.1])
            r_img = np.zeros((20, n, 3))
            r_img[:, :, 0] = gradient
            ax_r.imshow(r_img, aspect='auto')
            ax_r.set_xticks([0, n-1])
            ax_r.set_xticklabels(['Similar', 'Different'])
            ax_r.set_yticks([])
            ax_r.set_title('R: Covariance', fontsize=11, loc='left')
            
            ax_g = fig_legend.add_axes([0.55, 0.42, 0.35, 0.1])
            g_img = np.zeros((20, n, 3))
            g_img[:, :, 1] = gradient
            ax_g.imshow(g_img, aspect='auto')
            ax_g.set_xticks([0, n-1])
            ax_g.set_xticklabels(['Similar', 'Different'])
            ax_g.set_yticks([])
            ax_g.set_title('G: Spherical Harmonics', fontsize=11, loc='left')
            
            ax_b = fig_legend.add_axes([0.55, 0.16, 0.35, 0.1])
            b_img = np.zeros((20, n, 3))
            b_img[:, :, 2] = gradient
            ax_b.imshow(b_img, aspect='auto')
            ax_b.set_xticks([0, n-1])
            ax_b.set_xticklabels(['Similar', 'Different'])
            ax_b.set_yticks([])
            ax_b.set_title('B: Opacity', fontsize=11, loc='left')
            
            save_path_legend = self.output_dir / f"{scene_name}_3_rgb_legend.png"
            plt.savefig(save_path_legend, dpi=150, bbox_inches='tight', facecolor='white')
            plt.close()
            print(f"  图3-RGB 色盘图例已保存: {save_path_legend}")
        
        # ========== 图 3c: 不透明度 L2 距离 3D 热力图 ==========
        if hasattr(self, 'op_diff') and self.op_diff is not None:
            op_diff_np = self.op_diff.cpu().numpy()
            op_diff_filtered = op_diff_np[mask][::step]
            
            fig3c = plt.figure(figsize=(10, 8))
            ax3c = fig3c.add_subplot(111, projection='3d')
            
            # 不透明度差异 0→1：红色=低差异（相似）, 蓝色=高差异
            scatter = ax3c.scatter(x, y, z, c=op_diff_filtered, cmap='RdYlBu', 
                                  s=3, alpha=0.6, vmin=0, vmax=0.5)
            
            ax3c.view_init(elev=25, azim=view_azim)
            ax3c.set_xlabel('X')
            ax3c.set_ylabel('Y')
            ax3c.set_zlabel('Z')
            # 不要标题
            
            cbar = plt.colorbar(scatter, ax=ax3c, shrink=0.6, pad=0.02, label='L2 Distance')
            cbar.set_ticks([0, 0.125, 0.25, 0.375, 0.5])
            
            fig3c.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)
            save_path3c = self.output_dir / f"{scene_name}_3c_heatmap_op_diff.png"
            plt.savefig(save_path3c, dpi=300, bbox_inches='tight', pad_inches=0.02)
            plt.close()
            print(f"  图3c 不透明度L2距离热力图已保存: {save_path3c}")
        
        # ========== 图 3d: 分布直方图（三个指标，每个单独一张图）==========
        # 设置字体为 Nimbus Roman（Times New Roman 的开源替代）
        plt.rcParams['font.family'] = 'serif'
        plt.rcParams['font.serif'] = ['Nimbus Roman', 'Times New Roman', 'DejaVu Serif']
        plt.rcParams['mathtext.fontset'] = 'stix'
        
        # 用于保存 CSV 的数据
        csv_rows = []
        
        def setup_histogram_style(ax, data, bins, xlabel, color='steelblue'):
            """统一设置直方图样式"""
            counts, bin_edges, patches = ax.hist(
                data, bins=bins, 
                color=color, alpha=1.0,  # 不透明
                edgecolor='black', linewidth=0.3,
                rwidth=0.8  # 柱子宽度比例
            )
            
            # 设置四面边框粗细为 2pt
            for spine in ax.spines.values():
                spine.set_linewidth(2)
            
            # 添加水平网格线（灰色实线，在最底层）
            ax.yaxis.grid(True, linestyle='-', linewidth=0.5, color='gray', alpha=0.7, zorder=0)
            ax.set_axisbelow(True)  # 确保网格线在柱子下面
            
            # 设置刻度线：左侧和下侧朝外，长度 10pt，粗细 2pt
            ax.tick_params(axis='y', which='major', length=10, width=2, 
                          labelsize=12, direction='out')  # 左侧朝外
            ax.tick_params(axis='x', which='major', length=10, width=2, 
                          labelsize=12, direction='out')  # 下侧朝外
            ax.tick_params(axis='both', which='minor', length=5, width=1.5, 
                          direction='out')
            # 右侧和上侧不显示刻度线
            ax.tick_params(axis='both', which='both', right=False, top=False)
            
            # 设置字体（使用 fontdict 确保字体生效）
            ax.set_xlabel(xlabel, fontsize=14, fontfamily='serif')
            ax.set_ylabel('Number of Gaussians', fontsize=14, fontfamily='serif')
            
            # 设置刻度标签字体
            for label in ax.get_xticklabels() + ax.get_yticklabels():
                label.set_fontfamily('serif')
                label.set_fontsize(12)
            
            # 自动范围：从有数据的区域开始
            nonzero_idx = np.where(counts > 0)[0]
            if len(nonzero_idx) > 0:
                margin = (bins[1] - bins[0]) * 2
                ax.set_xlim(bin_edges[nonzero_idx[0]] - margin, 
                           bin_edges[nonzero_idx[-1] + 1] + margin)
            
            return counts, bin_edges
        
        # 协方差余弦相似度分布（蓝色）
        if hasattr(self, 'cov_sim') and self.cov_sim is not None:
            fig_cov, ax_cov = plt.subplots(figsize=(6, 4.5))
            cov_data = self.cov_sim.cpu().numpy()
            cov_bins = np.arange(0, 1.01, 0.01)
            counts, bin_edges = setup_histogram_style(
                ax_cov, cov_data, cov_bins, 
                'Covariance Cosine Similarity',
                color='#4472C4'  # 蓝色
            )
            for i in range(len(counts)):
                csv_rows.append(['covariance', round(bin_edges[i], 2), round(bin_edges[i+1], 2), int(counts[i])])
            
            # 在 0.7 处画红色虚线，右侧标注
            threshold = 0.7
            high_sim_ratio = (cov_data > threshold).sum() / len(cov_data) * 100
            ax_cov.axvline(x=threshold, color='red', linestyle='--', linewidth=1, zorder=5)
            y_max = ax_cov.get_ylim()[1]
            ax_cov.text(threshold + 0.02, y_max * 0.85, f'High Similarity\n({high_sim_ratio:.1f}%)', 
                       fontsize=10, color='red', fontfamily='serif', ha='left', va='top')
            
            plt.tight_layout()
            save_cov = self.output_dir / f"{scene_name}_hist_cov.png"
            plt.savefig(save_cov, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  协方差直方图已保存: {save_cov}")
        
        # 球谐函数余弦相似度分布（橙色）
        if hasattr(self, 'sh_sim') and self.sh_sim is not None:
            fig_sh, ax_sh = plt.subplots(figsize=(6, 4.5))
            sh_data = self.sh_sim.cpu().numpy()
            sh_bins = np.arange(-1, 1.01, 0.02)
            counts, bin_edges = setup_histogram_style(
                ax_sh, sh_data, sh_bins, 
                'SH Cosine Similarity',
                color='#ED7D31'  # 橙色
            )
            for i in range(len(counts)):
                csv_rows.append(['spherical_harmonics', round(bin_edges[i], 2), round(bin_edges[i+1], 2), int(counts[i])])
            
            # 在 0.7 处画红色虚线，右侧标注（多行避免超出）
            threshold = 0.7
            high_sim_ratio = (sh_data > threshold).sum() / len(sh_data) * 100
            ax_sh.axvline(x=threshold, color='red', linestyle='--', linewidth=1, zorder=5)
            y_max = ax_sh.get_ylim()[1]
            ax_sh.text(threshold + 0.03, y_max * 0.55, f'High\nSimilarity\n({high_sim_ratio:.1f}%)', 
                       fontsize=10, color='red', fontfamily='serif', ha='left', va='top')
            
            plt.tight_layout()
            save_sh = self.output_dir / f"{scene_name}_hist_sh.png"
            plt.savefig(save_sh, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  球谐函数直方图已保存: {save_sh}")
        
        # 不透明度 L2 距离分布（绿色）
        if hasattr(self, 'op_diff') and self.op_diff is not None:
            fig_op, ax_op = plt.subplots(figsize=(6, 4.5))
            op_data = self.op_diff.cpu().numpy()
            op_bins = np.arange(0, 1.01, 0.01)
            counts, bin_edges = setup_histogram_style(
                ax_op, op_data, op_bins, 
                'Opacity L2 Distance',
                color='#70AD47'  # 绿色
            )
            for i in range(len(counts)):
                csv_rows.append(['opacity', round(bin_edges[i], 2), round(bin_edges[i+1], 2), int(counts[i])])
            
            # 在 0.2 处画红色虚线，左侧标注（因为 L2 距离越小越相似，换行避免重叠）
            threshold = 0.2
            high_sim_ratio = (op_data < threshold).sum() / len(op_data) * 100
            ax_op.axvline(x=threshold, color='red', linestyle='--', linewidth=1, zorder=5)
            y_max = ax_op.get_ylim()[1]
            ax_op.text(threshold - 0.01, y_max * 0.95, f'High\nSimilarity\n({high_sim_ratio:.1f}%)', 
                       fontsize=10, color='red', fontfamily='serif', ha='right', va='top')
            
            plt.tight_layout()
            save_op = self.output_dir / f"{scene_name}_hist_opacity.png"
            plt.savefig(save_op, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"  不透明度直方图已保存: {save_op}")
        
        # 恢复默认字体设置
        plt.rcParams['font.family'] = 'sans-serif'
        
        # 保存 CSV（分三个文件）
        import csv
        
        # 按 metric 分组
        cov_rows = [r for r in csv_rows if r[0] == 'covariance']
        sh_rows = [r for r in csv_rows if r[0] == 'spherical_harmonics']
        op_rows = [r for r in csv_rows if r[0] == 'opacity']
        
        # 协方差
        csv_cov = self.output_dir / f"{scene_name}_hist_covariance.csv"
        with open(csv_cov, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['bin_left', 'bin_right', 'count'])
            for r in cov_rows:
                writer.writerow([r[1], r[2], r[3]])
        print(f"  协方差直方图数据: {csv_cov}")
        
        # 球谐函数
        csv_sh = self.output_dir / f"{scene_name}_hist_spherical_harmonics.csv"
        with open(csv_sh, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['bin_left', 'bin_right', 'count'])
            for r in sh_rows:
                writer.writerow([r[1], r[2], r[3]])
        print(f"  球谐函数直方图数据: {csv_sh}")
        
        # 不透明度
        csv_op = self.output_dir / f"{scene_name}_hist_opacity.csv"
        with open(csv_op, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['bin_left', 'bin_right', 'count'])
            for r in op_rows:
                writer.writerow([r[1], r[2], r[3]])
        print(f"  不透明度直方图数据: {csv_op}")
        
        # ========== 图 4: K-NN 散度投影到 Target 图像上 ==========
        # 使用 3D K-NN 散度（而非 2D variability）投影到 target 视角
        h, w = target_img.shape[:2]
        
        # 使用过滤后的高斯数据（不下采样）
        pos_full = positions_filtered  # 已过滤离群点
        knn_dispersion = variability_filtered  # 3D K-NN 散度
        
        # 世界坐标 -> 相机坐标 -> 图像坐标
        pos_homo = np.hstack([pos_full, np.ones((len(pos_full), 1))])
        pos_cam = (target_ext @ pos_homo.T).T[:, :3]
        
        # 只保留在相机前方的点 (z > 0.1)
        front_mask = pos_cam[:, 2] > 0.1
        pos_cam = pos_cam[front_mask]
        knn_proj = knn_dispersion[front_mask]
        
        # 投影到图像平面
        pos_proj = (target_int @ pos_cam.T).T
        pos_2d = pos_proj[:, :2] / pos_proj[:, 2:3]
        
        # 检查是否是归一化坐标
        if pos_2d[:, 0].max() < 2.0 and pos_2d[:, 1].max() < 2.0:
            pos_2d[:, 0] = pos_2d[:, 0] * w
            pos_2d[:, 1] = pos_2d[:, 1] * h
        
        # 过滤在图像范围内的点
        in_image = (
            (pos_2d[:, 0] >= 0) & (pos_2d[:, 0] < w) &
            (pos_2d[:, 1] >= 0) & (pos_2d[:, 1] < h)
        )
        pos_2d = pos_2d[in_image]
        knn_proj = knn_proj[in_image]
        
        # 创建像素级热力图：每个像素累积投影到该像素的所有高斯的 K-NN 散度
        heatmap = np.zeros((h, w))
        count_map = np.zeros((h, w))
        
        px = pos_2d[:, 0].astype(int)
        py = pos_2d[:, 1].astype(int)
        
        for x, y, v in zip(px, py, knn_proj):
            heatmap[y, x] += v
            count_map[y, x] += 1
        
        # 计算每个像素的平均 K-NN 散度
        mask_valid = count_map >= 1  # 只要有高斯投影就显示
        heatmap[mask_valid] = heatmap[mask_valid] / count_map[mask_valid]
        heatmap[~mask_valid] = np.nan  # 无投影的像素不显示
        
        # 裁剪到有效区域
        rows_with_data = np.any(mask_valid, axis=1)
        cols_with_data = np.any(mask_valid, axis=0)
        
        if rows_with_data.any() and cols_with_data.any():
            row_min, row_max = np.where(rows_with_data)[0][[0, -1]]
            col_min, col_max = np.where(cols_with_data)[0][[0, -1]]
            
            margin = 5
            row_min = max(0, row_min - margin)
            row_max = min(h - 1, row_max + margin)
            col_min = max(0, col_min - margin)
            col_max = min(w - 1, col_max + margin)
            
            heatmap_cropped = heatmap[row_min:row_max+1, col_min:col_max+1]
        else:
            heatmap_cropped = heatmap
            row_min, col_min = 0, 0
        
        # 绘制热力图
        fig4, ax4 = plt.subplots(figsize=(10, 8))
        im = ax4.imshow(heatmap_cropped, cmap='RdYlBu', vmin=0, vmax=0.3,
                       extent=[col_min, col_min + heatmap_cropped.shape[1],
                               row_min + heatmap_cropped.shape[0], row_min])
        ax4.set_title(f'Target View - Average K-NN Dispersion (Projected)\n'
                      f'(Red=Low Dispersion/Smooth, Blue=High Dispersion/Complex)\n'
                      f'{len(pos_2d):,} projected Gaussians', fontsize=12)
        ax4.set_xlabel('Width')
        ax4.set_ylabel('Height')
        plt.colorbar(im, ax=ax4, label='K-NN Dispersion')
        
        plt.suptitle(f'Scene: {scene_name}', fontsize=14)
        plt.tight_layout()
        save_path4 = self.output_dir / f"{scene_name}_4_heatmap_target_view.png"
        plt.savefig(save_path4, dpi=150)
        plt.close()
        print(f"  图4 Target视角热力图已保存: {save_path4}")
    
    def print_scene_report(self, stats: SmoothnessStats):
        """打印单个场景的分析报告"""
        print(f"\n  ========== 高斯平滑度分析: {stats.scene_name} ==========")
        print(f"  高斯数: {stats.total_gaussians:,}, 分辨率: {stats.height}x{stats.width}x{stats.n_views}")
        
        print(f"\n  【变化度统计】")
        print(f"    Mean: {stats.variability_mean:.4f}")
        print(f"    Std:  {stats.variability_std:.4f}")
        print(f"    Range: [{stats.variability_min:.4f}, {stats.variability_max:.4f}]")
        
        print(f"\n  【区域分布】")
        print(f"    ┌{'─'*20}┬{'─'*15}┬{'─'*20}┐")
        print(f"    │{'区域类型':^20}│{'占比':^15}│{'合并潜力':^20}│")
        print(f"    ├{'─'*20}┼{'─'*15}┼{'─'*20}┤")
        print(f"    │  平滑 (var<0.05)   │{stats.smooth_ratio*100:>12.1f}%  │  ~80% 可大范围合并  │")
        print(f"    │  中等 (0.05-0.15)  │{stats.medium_ratio*100:>12.1f}%  │  ~50% 可中范围合并  │")
        print(f"    │  复杂 (var>=0.15)  │{stats.complex_ratio*100:>12.1f}%  │  ~20% 可小范围合并  │")
        print(f"    └{'─'*20}┴{'─'*15}┴{'─'*20}┘")
        
        print(f"\n  【合并潜力估算】")
        print(f"    平滑区域贡献: {stats.smooth_merge_potential*100:.1f}%")
        print(f"    中等区域贡献: {stats.medium_merge_potential*100:.1f}%")
        print(f"    复杂区域贡献: {stats.complex_merge_potential*100:.1f}%")
        print(f"    ─────────────────────────")
        print(f"    总体合并潜力: {stats.total_merge_potential*100:.1f}%")
        
        # 生成热力图
        self.generate_heatmap_2d(stats)
        self.generate_heatmap_3d(stats)
        self.generate_heatmap_3d_multiview(stats)
        
        # 生成完整的 4 张可视化图
        self.generate_comprehensive_visualization(stats)
    
    def print_summary_report(self, model_name: str):
        """打印所有场景的汇总报告"""
        if not self.scene_stats:
            print("[Warning] No smoothness stats collected")
            return
        
        n_scenes = len(self.scene_stats)
        
        # 汇总
        avg_smooth = np.mean([s.smooth_ratio for s in self.scene_stats])
        avg_medium = np.mean([s.medium_ratio for s in self.scene_stats])
        avg_complex = np.mean([s.complex_ratio for s in self.scene_stats])
        avg_merge_potential = np.mean([s.total_merge_potential for s in self.scene_stats])
        
        avg_gaussians = np.mean([s.total_gaussians for s in self.scene_stats])
        total_mb = avg_gaussians * 352 / 1024 / 1024
        
        print("\n" + "=" * 100)
        print(f"  ███ {model_name.upper()} 高斯平滑度分析总结 ███")
        print("=" * 100)
        
        print(f"\n  统计场景数: {n_scenes}")
        print(f"  平均高斯数: {avg_gaussians:,.0f}")
        
        print(f"\n  【区域分布汇总】")
        print(f"  ┌{'─'*25}┬{'─'*15}┬{'─'*25}┐")
        print(f"  │{'区域类型':^25}│{'平均占比':^15}│{'合并策略':^25}│")
        print(f"  ├{'─'*25}┼{'─'*15}┼{'─'*25}┤")
        print(f"  │  🔵 平滑区域 (var<0.05)  │{avg_smooth*100:>12.1f}%  │  大范围合并 (4x4+)       │")
        print(f"  │  🟡 中等区域 (0.05-0.15) │{avg_medium*100:>12.1f}%  │  中范围合并 (2x2)        │")
        print(f"  │  🔴 复杂区域 (var>=0.15) │{avg_complex*100:>12.1f}%  │  小范围合并 (相邻对)     │")
        print(f"  └{'─'*25}┴{'─'*15}┴{'─'*25}┘")
        
        print(f"\n  【HBM 节省估算】")
        print(f"  原始数据量: {total_mb:.2f} MB")
        print(f"  总体合并潜力: {avg_merge_potential*100:.1f}%")
        print(f"  估算可节省: {total_mb * avg_merge_potential:.2f} MB")
        
        new_count = int(avg_gaussians * (1 - avg_merge_potential))
        print(f"\n  【合并前后对比】")
        print(f"  合并前高斯数: {avg_gaussians:,.0f}")
        print(f"  合并后高斯数: {new_count:,} (减少 {avg_merge_potential*100:.1f}%)")
        
        print(f"\n  热力图已保存至: {self.output_dir}")
        print("=" * 100)
        print()


# 单例模式
_analyzer_instance: Optional[GaussianSmoothnessAnalyzer] = None


def get_smoothness_analyzer() -> GaussianSmoothnessAnalyzer:
    """获取全局分析器实例"""
    global _analyzer_instance
    if _analyzer_instance is None:
        _analyzer_instance = GaussianSmoothnessAnalyzer()
    return _analyzer_instance


def reset_smoothness_analyzer():
    """重置分析器"""
    global _analyzer_instance
    _analyzer_instance = None
