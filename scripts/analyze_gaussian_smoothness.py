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
        计算统一 3D 高斯场中每个高斯的变化度
        使用 K 近邻来计算每个高斯与其 3D 空间邻居的差异
        """
        means = gaussians.means[0]  # [N, 3]
        covariances = gaussians.covariances[0]  # [N, 3, 3]
        opacities = gaussians.opacities[0]  # [N]
        harmonics = gaussians.harmonics[0]  # [N, 3, K]
        
        N = means.shape[0]
        device = means.device
        
        # 尺度
        scene_scale = means.std() + 1e-6
        cov_scale = covariances.norm(dim=(-2,-1)).mean() + 1e-6
        sh_scale = harmonics.norm(dim=(-2,-1)).mean() + 1e-6
        
        # 使用简单的网格化方法找邻居（避免 KNN 的计算开销）
        # 将 3D 空间划分为网格，每个高斯与同网格/相邻网格的高斯比较
        
        # 简化版本：随机采样邻居进行比较
        variability = torch.zeros(N, device=device)
        
        # 为了效率，使用批量距离计算
        # 对每个高斯，随机采样 k_neighbors 个其他高斯作为"伪邻居"
        # 然后计算与最近的几个的差异
        
        batch_size = 4096
        for start in range(0, N, batch_size):
            end = min(start + batch_size, N)
            batch_means = means[start:end]  # [B, 3]
            batch_cov = covariances[start:end]  # [B, 3, 3]
            batch_op = opacities[start:end]  # [B]
            batch_sh = harmonics[start:end]  # [B, 3, K]
            
            # 计算与所有其他高斯的距离（采样）
            sample_size = min(1000, N)
            sample_idx = torch.randperm(N, device=device)[:sample_size]
            sample_means = means[sample_idx]  # [S, 3]
            sample_cov = covariances[sample_idx]
            sample_op = opacities[sample_idx]
            sample_sh = harmonics[sample_idx]
            
            # 计算距离 [B, S]
            dist = torch.cdist(batch_means, sample_means)
            
            # 找到 k 个最近邻（排除自己）
            _, nearest_idx = dist.topk(k_neighbors + 1, dim=1, largest=False)
            nearest_idx = nearest_idx[:, 1:]  # 排除自己，[B, k]
            
            # 计算与最近邻的差异
            B = end - start
            for i in range(B):
                nn_idx = nearest_idx[i]  # [k]
                
                # 位置差异
                pos_diff = (sample_means[nn_idx] - batch_means[i]).norm(dim=-1).mean() / scene_scale
                
                # 协方差差异
                cov_diff = (sample_cov[nn_idx] - batch_cov[i]).norm(dim=(-2,-1)).mean() / cov_scale
                
                # Opacity 差异
                op_diff = (sample_op[nn_idx] - batch_op[i]).abs().mean()
                
                # SH 差异
                sh_diff = (sample_sh[nn_idx] - batch_sh[i]).norm(dim=(-2,-1)).mean() / sh_scale
                
                # 综合变化度
                variability[start + i] = 0.4 * pos_diff + 0.2 * cov_diff + 0.2 * op_diff + 0.2 * sh_diff
        
        return variability
    
    def analyze_scene(self, scene_name: str, gaussians) -> SmoothnessStats:
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
            im = ax.imshow(variability[v], cmap='RdYlBu_r', vmin=0, vmax=0.3)
            ax.set_title(f'View {v+1} - Gaussian Variability\n(Red=Complex, Blue=Smooth)')
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
        
        # 使用 RdYlBu_r: 红色=高变化度，蓝色=低变化度
        scatter = ax.scatter(x, y, z, c=c_norm, cmap='RdYlBu_r', 
                            s=3, alpha=0.6, vmin=0, vmax=1)
        
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.set_title(f'Scene: {stats.scene_name}\n'
                     f'K-NN Gaussian Dispersion ({len(pos):,} / {stats.total_gaussians:,} Gaussians)\n'
                     f'Blue=Low Dispersion (Mergeable), Red=High Dispersion (Keep)')
        
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
            scatter = ax.scatter(x, y, z, c=c_norm, cmap='RdYlBu_r', 
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
        sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=plt.Normalize(0, 1))
        cbar = fig.colorbar(sm, cax=cbar_ax, label='K-NN Dispersion')
        cbar.set_ticks([0, 0.167, 0.333, 0.5, 0.667, 0.833, 1.0])
        cbar.set_ticklabels(['0', '0.05', '0.10', '0.15', '0.20', '0.25', '0.30+'])
        
        plt.tight_layout(rect=[0, 0, 0.9, 0.95])
        
        if save_path is None:
            save_path = self.output_dir / f"{stats.scene_name}_heatmap_3d_multiview.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"  多角度 3D 热力图已保存: {save_path}")
    
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
