#!/usr/bin/env python3
"""
高斯基元冗余分析脚本 (Challenge 2 Profiling)

分析可泛化3DGS中高斯基元的结构性冗余:
1. 视角相关的有效性差异：不同视角下可见/高贡献高斯的比例和重叠度
2. 像素级生成的空间冗余：特征相似像素产生位置相近的高斯
3. 深度概率分布特性：单峰vs多峰，置信度分布

这些数据用于支撑 FSGG (Feature-Similarity Guided Gaussian Grouping) 
和 DPGH (Depth-Probability Guided Hierarchical Gaussians) 的设计。

使用方法:
    # 嵌入到模型推理中使用
    from scripts.analyze_gaussian_redundancy import GaussianRedundancyAnalyzer
    analyzer = GaussianRedundancyAnalyzer()
    analyzer.analyze_scene(gaussians, features, depth_pdf, ...)
    analyzer.print_report()
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class ViewPairStats:
    """视角对之间的高斯共享统计"""
    view_i: int
    view_j: int
    visible_i: int          # 视角i可见的高斯数
    visible_j: int          # 视角j可见的高斯数
    shared_visible: int     # 两个视角都可见的高斯数
    jaccard_similarity: float  # Jaccard相似度
    

@dataclass  
class DepthPDFStats:
    """深度PDF特性统计"""
    num_pixels: int
    single_peak_ratio: float      # 单峰分布比例 (pdf_max > 0.5)
    multi_peak_ratio: float       # 多峰分布比例 (second_peak > 0.2)
    uncertain_ratio: float        # 不确定分布比例 (spread > threshold)
    avg_peak_confidence: float    # 平均峰值置信度
    avg_spread: float             # 平均分布宽度


@dataclass
class SpatialClusterStats:
    """空间聚类统计"""
    num_clusters: int
    avg_cluster_size: int
    max_cluster_size: int
    cluster_size_distribution: List[int]
    avg_intra_cluster_distance: float  # 簇内平均距离
    avg_inter_cluster_distance: float  # 簇间平均距离


@dataclass
class GaussianMemoryStats:
    """高斯内存占用统计"""
    total_gaussians: int
    sh_degree: int
    d_sh: int                      # (sh_degree + 1)^2
    bytes_per_gaussian: int        # 每个高斯的字节数
    total_bytes: int               # 总字节数
    total_mb: float                # 总 MB
    # 分项统计
    means_bytes: int               # 3 * 4 = 12
    covariances_bytes: int         # 3 * 3 * 4 = 36
    harmonics_bytes: int           # 3 * d_sh * 4
    opacities_bytes: int           # 1 * 4 = 4


@dataclass
class SceneRedundancyStats:
    """单个场景的冗余统计"""
    scene_name: str
    total_gaussians: int
    num_views: int
    
    # 高斯内存统计
    memory_stats: Optional[GaussianMemoryStats] = None
    
    # 视角相关统计
    avg_visible_ratio: float = 0.0         # 平均可见比例
    avg_high_contrib_ratio: float = 0.0    # 平均高贡献比例
    view_pair_stats: List[ViewPairStats] = field(default_factory=list)
    avg_jaccard_similarity: float = 0.0    # 平均Jaccard相似度
    
    # 深度PDF统计
    depth_pdf_stats: Optional[DepthPDFStats] = None
    
    # 空间聚类统计
    spatial_cluster_stats: Optional[SpatialClusterStats] = None
    
    # 特征-位置相关性
    feature_position_correlation: float = 0.0  # 特征相似性与位置接近度的相关系数


class GaussianRedundancyAnalyzer:
    """高斯基元冗余分析器"""
    
    def __init__(
        self, 
        device: str = 'cuda',
        num_sample_pairs: int = 10000,
        num_clusters: int = 256,
    ):
        self.device = device
        self.num_sample_pairs = num_sample_pairs
        self.num_clusters = num_clusters
        
        # 累积统计
        self.all_scene_stats: List[SceneRedundancyStats] = []
    
    def compute_gaussian_memory(self, gaussians) -> GaussianMemoryStats:
        """
        计算高斯基元的精确内存占用
        
        Gaussians 数据结构:
        - means: [batch, gaussian, 3] -> 3 * 4 = 12 bytes
        - covariances: [batch, gaussian, 3, 3] -> 9 * 4 = 36 bytes
        - harmonics: [batch, gaussian, 3, d_sh] -> 3 * d_sh * 4 bytes
        - opacities: [batch, gaussian] -> 1 * 4 = 4 bytes
        """
        # 获取张量形状
        total_gaussians = gaussians.means.shape[1]
        
        # 推断 sh_degree: harmonics shape is [batch, gaussian, 3, d_sh]
        d_sh = gaussians.harmonics.shape[-1]
        sh_degree = int(d_sh ** 0.5) - 1  # d_sh = (sh_degree + 1)^2
        
        # 计算各部分字节数 (假设 float32)
        means_bytes = 3 * 4  # 12
        covariances_bytes = 3 * 3 * 4  # 36
        harmonics_bytes = 3 * d_sh * 4
        opacities_bytes = 1 * 4  # 4
        
        bytes_per_gaussian = means_bytes + covariances_bytes + harmonics_bytes + opacities_bytes
        total_bytes = total_gaussians * bytes_per_gaussian
        total_mb = total_bytes / (1024 * 1024)
        
        return GaussianMemoryStats(
            total_gaussians=total_gaussians,
            sh_degree=sh_degree,
            d_sh=d_sh,
            bytes_per_gaussian=bytes_per_gaussian,
            total_bytes=total_bytes,
            total_mb=total_mb,
            means_bytes=means_bytes,
            covariances_bytes=covariances_bytes,
            harmonics_bytes=harmonics_bytes,
            opacities_bytes=opacities_bytes,
        )
        
    def analyze_view_redundancy(
        self,
        gaussians_means: torch.Tensor,      # [B, G, 3]
        gaussians_opacities: torch.Tensor,  # [B, G]
        target_extrinsics: torch.Tensor,    # [B, V, 4, 4]
        target_intrinsics: torch.Tensor,    # [B, V, 3, 3]
        near: torch.Tensor,                 # [B, V]
        far: torch.Tensor,                  # [B, V]
        image_shape: Tuple[int, int],
        opacity_threshold: float = 0.01,
    ) -> Tuple[List[float], List[float], List[ViewPairStats]]:
        """
        分析不同视角下高斯的可见性和重叠度
        
        Returns:
            visible_ratios: 每个视角的可见高斯比例
            high_contrib_ratios: 每个视角的高贡献高斯比例
            view_pair_stats: 视角对之间的共享统计
        """
        b, g, _ = gaussians_means.shape
        _, v, _, _ = target_extrinsics.shape
        h, w = image_shape
        
        visible_masks = []  # 每个视角的可见掩码
        high_contrib_masks = []  # 每个视角的高贡献掩码
        
        for vi in range(v):
            # 获取当前视角的相机参数
            extr_c2w = target_extrinsics[0, vi]  # [4, 4] camera-to-world
            extr_w2c = torch.inverse(extr_c2w)   # world-to-camera
            intr = target_intrinsics[0, vi]  # [3, 3]
            
            # 变换到相机坐标系 (world -> camera)
            means_homo = F.pad(gaussians_means[0], (0, 1), value=1.0)  # [G, 4]
            means_cam = (extr_w2c @ means_homo.T).T[:, :3]  # [G, 3]
            
            # 深度筛选 (相机坐标系 z > 0 才在相机前方)
            depths = means_cam[:, 2]
            depth_mask = (depths > near[0, vi]) & (depths < far[0, vi]) & (depths > 0)
            
            # 投影到图像平面
            means_proj = (intr @ means_cam.T).T  # [G, 3]
            means_2d = means_proj[:, :2] / (means_proj[:, 2:3] + 1e-6)  # [G, 2]
            
            # 视锥裁剪
            frustum_mask = (
                (means_2d[:, 0] >= 0) & (means_2d[:, 0] < w) &
                (means_2d[:, 1] >= 0) & (means_2d[:, 1] < h) &
                depth_mask
            )
            
            visible_masks.append(frustum_mask)
            
            # 高贡献掩码 (可见 + opacity > threshold)
            high_contrib_mask = frustum_mask & (gaussians_opacities[0] > opacity_threshold)
            high_contrib_masks.append(high_contrib_mask)
        
        # 计算比例
        visible_ratios = [m.float().mean().item() for m in visible_masks]
        high_contrib_ratios = [m.float().mean().item() for m in high_contrib_masks]
        
        # 计算视角对之间的共享度
        view_pair_stats = []
        for i in range(min(v, 10)):  # 限制分析的视角数
            for j in range(i + 1, min(v, 10)):
                shared = (visible_masks[i] & visible_masks[j]).sum().item()
                union = (visible_masks[i] | visible_masks[j]).sum().item()
                jaccard = shared / union if union > 0 else 0
                
                view_pair_stats.append(ViewPairStats(
                    view_i=i,
                    view_j=j,
                    visible_i=visible_masks[i].sum().item(),
                    visible_j=visible_masks[j].sum().item(),
                    shared_visible=shared,
                    jaccard_similarity=jaccard,
                ))
        
        return visible_ratios, high_contrib_ratios, view_pair_stats
    
    def analyze_depth_pdf(
        self,
        depth_pdf: torch.Tensor,  # [B, V, D, H, W] or [VB, D, H, W]
        depth_candidates: torch.Tensor,  # [D] or other shapes
    ) -> DepthPDFStats:
        """
        分析深度概率分布的特性
        
        用于DPGH设计：根据PDF特性决定高斯层级
        """
        # 处理不同的输入格式
        if depth_pdf.dim() == 5:
            b, v, d, h, w = depth_pdf.shape
            pdf = depth_pdf.view(-1, d, h * w).permute(0, 2, 1)  # [BV, HW, D]
        else:
            vb, d, h, w = depth_pdf.shape
            pdf = depth_pdf.view(vb, d, h * w).permute(0, 2, 1)  # [VB, HW, D]
        
        pdf = pdf.reshape(-1, pdf.shape[-1])  # [N, D] where N = BV*HW
        num_pixels = pdf.shape[0]
        d_dim = pdf.shape[1]
        
        # 1. 峰值分析
        pdf_max, pdf_argmax = pdf.max(dim=1)
        
        # 单峰: pdf_max > 0.5
        single_peak_mask = pdf_max > 0.5
        single_peak_ratio = single_peak_mask.float().mean().item()
        
        # 2. 次峰分析 (用于识别多模态深度)
        # 将主峰位置的值置零，找次峰
        pdf_masked = pdf.clone()
        pdf_masked.scatter_(1, pdf_argmax.unsqueeze(1), 0)
        # 同时将主峰相邻位置也置零（避免同一峰被算两次）
        for offset in [-1, 1]:
            neighbor_idx = (pdf_argmax + offset).clamp(0, d_dim - 1)
            pdf_masked.scatter_(1, neighbor_idx.unsqueeze(1), 0)
        
        second_peak, _ = pdf_masked.max(dim=1)
        multi_peak_mask = second_peak > 0.2
        multi_peak_ratio = multi_peak_mask.float().mean().item()
        
        # 3. 分布宽度 (标准差)
        # Handle depth_candidates with various shapes
        if depth_candidates.dim() > 1:
            # Flatten and take unique values
            depth_candidates = depth_candidates.flatten()
            if len(depth_candidates) > d_dim:
                # Take first d_dim values
                depth_candidates = depth_candidates[:d_dim]
        
        if len(depth_candidates) != d_dim:
            # Create synthetic depth candidates if dimensions don't match
            depth_candidates = torch.linspace(0.5, 10.0, d_dim, device=pdf.device)
        
        depth_candidates_expanded = depth_candidates.unsqueeze(0).expand(num_pixels, -1)
        mean_depth = (pdf * depth_candidates_expanded).sum(dim=1)
        variance = (pdf * (depth_candidates_expanded - mean_depth.unsqueeze(1))**2).sum(dim=1)
        spread = variance.sqrt()
        
        # 不确定: spread > threshold (相对于深度范围)
        depth_range = depth_candidates[-1] - depth_candidates[0]
        if depth_range.abs() < 1e-6:
            depth_range = torch.tensor(1.0, device=pdf.device)
        uncertain_mask = spread > 0.1 * depth_range
        uncertain_ratio = uncertain_mask.float().mean().item()
        
        return DepthPDFStats(
            num_pixels=num_pixels,
            single_peak_ratio=single_peak_ratio,
            multi_peak_ratio=multi_peak_ratio,
            uncertain_ratio=uncertain_ratio,
            avg_peak_confidence=pdf_max.mean().item(),
            avg_spread=(spread / depth_range).mean().item(),
        )
    
    def analyze_spatial_clustering(
        self,
        gaussians_means: torch.Tensor,  # [B, G, 3]
        features: torch.Tensor,         # [B, V, C, H, W]
        feature_to_gaussian_map: Optional[torch.Tensor] = None,  # 特征像素到高斯的映射
    ) -> Tuple[SpatialClusterStats, float]:
        """
        分析高斯的空间聚类特性
        
        用于FSGG设计：基于特征相似性的高斯分组
        
        Returns:
            spatial_stats: 空间聚类统计
            feature_position_correlation: 特征相似性与位置接近度的相关系数
        """
        means = gaussians_means[0]  # [G, 3]
        g = means.shape[0]
        
        # 1. 简单空间网格聚类
        # 归一化到 [0, num_clusters)
        means_min = means.min(dim=0).values
        means_max = means.max(dim=0).values
        means_normalized = (means - means_min) / (means_max - means_min + 1e-6)
        
        # 量化到网格
        grid_size = int(self.num_clusters ** (1/3))  # 立方根
        cluster_ids = (means_normalized * grid_size).long().clamp(0, grid_size - 1)
        cluster_ids = (
            cluster_ids[:, 0] * grid_size * grid_size + 
            cluster_ids[:, 1] * grid_size + 
            cluster_ids[:, 2]
        )
        
        # 统计每个簇的大小
        unique_clusters, cluster_counts = cluster_ids.unique(return_counts=True)
        num_clusters = len(unique_clusters)
        cluster_sizes = cluster_counts.cpu().numpy().tolist()
        
        # 计算簇内和簇间距离（采样）
        sample_size = min(1000, g)
        sample_idx = torch.randperm(g)[:sample_size]
        sample_means = means[sample_idx]
        sample_clusters = cluster_ids[sample_idx]
        
        # 簇内距离
        intra_distances = []
        for cid in unique_clusters[:50]:  # 只采样部分簇
            mask = sample_clusters == cid
            if mask.sum() > 1:
                cluster_means = sample_means[mask]
                dists = torch.cdist(cluster_means, cluster_means)
                # 取上三角（排除对角线）
                triu_mask = torch.triu(torch.ones_like(dists), diagonal=1).bool()
                if triu_mask.any():
                    intra_distances.extend(dists[triu_mask].cpu().numpy().tolist())
        
        avg_intra = np.mean(intra_distances) if intra_distances else 0
        
        # 簇间距离（簇中心之间）
        cluster_centers = []
        for cid in unique_clusters[:50]:
            mask = cluster_ids == cid
            cluster_centers.append(means[mask].mean(dim=0))
        
        if len(cluster_centers) > 1:
            centers = torch.stack(cluster_centers)
            inter_dists = torch.cdist(centers, centers)
            triu_mask = torch.triu(torch.ones_like(inter_dists), diagonal=1).bool()
            avg_inter = inter_dists[triu_mask].mean().item()
        else:
            avg_inter = 0
        
        spatial_stats = SpatialClusterStats(
            num_clusters=num_clusters,
            avg_cluster_size=int(np.mean(cluster_sizes)),
            max_cluster_size=int(np.max(cluster_sizes)),
            cluster_size_distribution=cluster_sizes,
            avg_intra_cluster_distance=avg_intra,
            avg_inter_cluster_distance=avg_inter,
        )
        
        # 2. 特征-位置相关性分析
        feature_position_corr = self._compute_feature_position_correlation(
            means, features
        )
        
        return spatial_stats, feature_position_corr
    
    def _compute_feature_position_correlation(
        self,
        means: torch.Tensor,    # [G, 3]
        features: torch.Tensor, # [B, V, C, H, W]
    ) -> float:
        """
        计算特征相似性与高斯位置接近度的相关系数
        
        核心假设: 特征相似的像素 → 高斯位置接近
        """
        # 取第一个视角的特征
        feat = features[0, 0]  # [C, H, W]
        c, h, w = feat.shape
        
        # 假设高斯与像素一一对应
        g = means.shape[0]
        if g != h * w:
            # 如果不是一一对应，采样分析
            sample_size = min(g, h * w, 5000)
            sample_idx = torch.randperm(min(g, h * w))[:sample_size]
        else:
            sample_size = min(g, 5000)
            sample_idx = torch.randperm(g)[:sample_size]
        
        # 采样像素对
        num_pairs = min(self.num_sample_pairs, sample_size * (sample_size - 1) // 2)
        pair_idx_a = torch.randint(0, sample_size, (num_pairs,), device=self.device)
        pair_idx_b = torch.randint(0, sample_size, (num_pairs,), device=self.device)
        
        # 特征相似性
        feat_flat = feat.view(c, -1).T  # [HW, C]
        if feat_flat.shape[0] > sample_size:
            feat_flat = feat_flat[:sample_size]
        feat_norm = F.normalize(feat_flat, p=2, dim=1)
        
        feat_a = feat_norm[pair_idx_a]
        feat_b = feat_norm[pair_idx_b]
        similarities = (feat_a * feat_b).sum(dim=1)  # [num_pairs]
        
        # 位置接近度（距离的负数）
        means_sample = means[:sample_size] if means.shape[0] >= sample_size else means
        if means_sample.shape[0] < sample_size:
            # 不够样本，返回0
            return 0.0
        
        pos_a = means_sample[pair_idx_a]
        pos_b = means_sample[pair_idx_b]
        distances = (pos_a - pos_b).norm(dim=1)  # [num_pairs]
        
        # 计算皮尔逊相关系数
        # 特征相似度高 → 距离小 (负相关)
        sim_np = similarities.cpu().numpy()
        dist_np = distances.cpu().numpy()
        
        # 相关系数（取负，因为我们期望负相关）
        if np.std(sim_np) > 0 and np.std(dist_np) > 0:
            corr = -np.corrcoef(sim_np, dist_np)[0, 1]
        else:
            corr = 0.0
        
        return corr
    
    def analyze_scene(
        self,
        scene_name: str,
        gaussians,  # Gaussians dataclass
        target_extrinsics: torch.Tensor,
        target_intrinsics: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        image_shape: Tuple[int, int],
        features: Optional[torch.Tensor] = None,
        depth_pdf: Optional[torch.Tensor] = None,
        depth_candidates: Optional[torch.Tensor] = None,
        opacity_threshold: float = 0.01,
    ) -> SceneRedundancyStats:
        """
        分析单个场景的高斯冗余
        """
        gaussians_means = gaussians.means  # [B, G, 3]
        gaussians_opacities = gaussians.opacities  # [B, G]
        
        total_gaussians = gaussians_means.shape[1]
        num_views = target_extrinsics.shape[1]
        
        # 0. 高斯内存统计
        memory_stats = self.compute_gaussian_memory(gaussians)
        
        # 1. 视角相关冗余分析
        visible_ratios, high_contrib_ratios, view_pair_stats = self.analyze_view_redundancy(
            gaussians_means, gaussians_opacities,
            target_extrinsics, target_intrinsics,
            near, far, image_shape, opacity_threshold
        )
        
        avg_visible = np.mean(visible_ratios)
        avg_high_contrib = np.mean(high_contrib_ratios)
        avg_jaccard = np.mean([vps.jaccard_similarity for vps in view_pair_stats]) if view_pair_stats else 0
        
        # 2. 深度PDF分析
        depth_pdf_stats = None
        if depth_pdf is not None:
            try:
                depth_pdf_stats = self.analyze_depth_pdf(
                    depth_pdf, 
                    depth_candidates if depth_candidates is not None else torch.linspace(0.5, 10, 32, device=depth_pdf.device)
                )
            except Exception as e:
                print(f"[Warning] Depth PDF analysis failed: {e}")
        
        # 3. 空间聚类分析
        spatial_stats = None
        feature_position_corr = 0.0
        if features is not None:
            try:
                spatial_stats, feature_position_corr = self.analyze_spatial_clustering(
                    gaussians_means, features
                )
            except Exception as e:
                print(f"[Warning] Spatial clustering analysis failed: {e}")
        
        stats = SceneRedundancyStats(
            scene_name=scene_name,
            total_gaussians=total_gaussians,
            num_views=num_views,
            memory_stats=memory_stats,
            avg_visible_ratio=avg_visible,
            avg_high_contrib_ratio=avg_high_contrib,
            view_pair_stats=view_pair_stats,
            avg_jaccard_similarity=avg_jaccard,
            depth_pdf_stats=depth_pdf_stats,
            spatial_cluster_stats=spatial_stats,
            feature_position_correlation=feature_position_corr,
        )
        
        self.all_scene_stats.append(stats)
        return stats
    
    def print_scene_report(self, stats: SceneRedundancyStats):
        """打印单个场景的分析报告"""
        print(f"\n{'='*80}")
        print(f"  [Challenge 2 Profiling] 场景: {stats.scene_name}")
        print(f"{'='*80}")
        
        print(f"\n  【基础统计】")
        print(f"  总高斯数: {stats.total_gaussians:,}")
        print(f"  目标视角数: {stats.num_views}")
        
        # 高斯内存统计
        if stats.memory_stats:
            mem = stats.memory_stats
            print(f"\n  【高斯内存占用】(精确测量)")
            print(f"  SH阶数: {mem.sh_degree} (d_sh = {mem.d_sh})")
            print(f"  每个高斯: {mem.bytes_per_gaussian} bytes")
            print(f"    - means:       {mem.means_bytes:>4} B (3 x float32)")
            print(f"    - covariances: {mem.covariances_bytes:>4} B (3x3 x float32)")
            print(f"    - harmonics:   {mem.harmonics_bytes:>4} B (3 x {mem.d_sh} x float32)")
            print(f"    - opacities:   {mem.opacities_bytes:>4} B (1 x float32)")
            print(f"  总数据量: {mem.total_bytes:,} bytes = {mem.total_mb:.2f} MB")
        
        print(f"\n  【视角相关冗余】")
        print(f"  平均可见比例: {stats.avg_visible_ratio*100:.2f}%")
        print(f"  平均高贡献比例: {stats.avg_high_contrib_ratio*100:.2f}%")
        print(f"  视角间Jaccard相似度: {stats.avg_jaccard_similarity:.3f}")
        redundancy = 1 - stats.avg_visible_ratio
        print(f"  → 视角无关冗余: {redundancy*100:.1f}% 的高斯在视锥外")
        if stats.memory_stats:
            wasted_mb = stats.memory_stats.total_mb * redundancy
            print(f"  → 无效HBM流量: {wasted_mb:.2f} MB ({redundancy*100:.1f}%)")
        
        if stats.depth_pdf_stats:
            pdf = stats.depth_pdf_stats
            print(f"\n  【深度PDF特性】(用于DPGH分层)")
            print(f"  单峰分布 (高置信度, >0.5): {pdf.single_peak_ratio*100:.1f}%")
            print(f"  多峰分布 (次峰>0.2): {pdf.multi_peak_ratio*100:.1f}%")
            print(f"  不确定分布 (spread>10%): {pdf.uncertain_ratio*100:.1f}%")
            print(f"  平均峰值置信度: {pdf.avg_peak_confidence:.3f}")
            print(f"  平均分布宽度 (归一化): {pdf.avg_spread:.3f}")
        
        if stats.spatial_cluster_stats:
            sc = stats.spatial_cluster_stats
            print(f"\n  【空间聚类特性】(用于FSGG分组)")
            print(f"  有效簇数: {sc.num_clusters}")
            print(f"  平均簇大小: {sc.avg_cluster_size}")
            print(f"  最大簇大小: {sc.max_cluster_size}")
            print(f"  簇内平均距离: {sc.avg_intra_cluster_distance:.4f}")
            print(f"  簇间平均距离: {sc.avg_inter_cluster_distance:.4f}")
            if sc.avg_intra_cluster_distance > 0:
                ratio = sc.avg_inter_cluster_distance / sc.avg_intra_cluster_distance
                print(f"  簇分离度 (inter/intra): {ratio:.1f}x")
        
        if stats.feature_position_correlation > 0:
            print(f"\n  【特征-位置相关性】")
            print(f"  相关系数: {stats.feature_position_correlation:.3f}")
            print(f"  → {'强' if stats.feature_position_correlation > 0.5 else '中等' if stats.feature_position_correlation > 0.3 else '弱'}相关")
    
    def print_summary_report(self, model_name: str = ""):
        """打印所有场景的汇总报告"""
        if not self.all_scene_stats:
            print("没有分析数据")
            return
        
        print("\n" + "=" * 90)
        print(f"  ███ Challenge 2 高斯冗余分析汇总报告 - {model_name.upper() if model_name else 'ALL'} ███")
        print(f"  (共 {len(self.all_scene_stats)} 个场景)")
        print("=" * 90)
        
        # 内存统计汇总
        mem_stats = [s.memory_stats for s in self.all_scene_stats if s.memory_stats]
        if mem_stats:
            print(f"\n  【高斯内存占用统计】")
            avg_gaussians = np.mean([m.total_gaussians for m in mem_stats])
            avg_bytes_per = np.mean([m.bytes_per_gaussian for m in mem_stats])
            avg_mb = np.mean([m.total_mb for m in mem_stats])
            sh_degree = mem_stats[0].sh_degree
            d_sh = mem_stats[0].d_sh
            
            print(f"  SH阶数: {sh_degree} (d_sh = {d_sh})")
            print(f"  平均高斯数: {avg_gaussians:,.0f}")
            print(f"  每高斯字节: {avg_bytes_per:.0f} B")
            print(f"    - means:       {mem_stats[0].means_bytes:>4} B")
            print(f"    - covariances: {mem_stats[0].covariances_bytes:>4} B")
            print(f"    - harmonics:   {mem_stats[0].harmonics_bytes:>4} B")
            print(f"    - opacities:   {mem_stats[0].opacities_bytes:>4} B")
            print(f"  平均总数据量: {avg_mb:.2f} MB")
        
        # 视角冗余统计
        avg_gaussians = np.mean([s.total_gaussians for s in self.all_scene_stats])
        avg_visible = np.mean([s.avg_visible_ratio for s in self.all_scene_stats])
        avg_high_contrib = np.mean([s.avg_high_contrib_ratio for s in self.all_scene_stats])
        avg_jaccard = np.mean([s.avg_jaccard_similarity for s in self.all_scene_stats])
        
        print(f"\n  【视角相关冗余统计】")
        print(f"  {'指标':<25} {'平均值':<15} {'最小值':<15} {'最大值':<15}")
        print(f"  {'-'*70}")
        print(f"  {'总高斯数':<25} {avg_gaussians:>12,.0f}")
        print(f"  {'可见比例':<25} {avg_visible*100:>12.2f}% "
              f"{min(s.avg_visible_ratio for s in self.all_scene_stats)*100:>12.2f}% "
              f"{max(s.avg_visible_ratio for s in self.all_scene_stats)*100:>12.2f}%")
        print(f"  {'高贡献比例':<25} {avg_high_contrib*100:>12.2f}% "
              f"{min(s.avg_high_contrib_ratio for s in self.all_scene_stats)*100:>12.2f}% "
              f"{max(s.avg_high_contrib_ratio for s in self.all_scene_stats)*100:>12.2f}%")
        print(f"  {'视角间Jaccard相似度':<25} {avg_jaccard:>12.3f}")
        
        # 计算冗余流量
        redundancy_ratio = 1 - avg_visible
        if mem_stats:
            wasted_mb = avg_mb * redundancy_ratio
            print(f"\n  【HBM流量冗余】")
            print(f"  视角无关冗余: {redundancy_ratio*100:.1f}%")
            print(f"  每帧无效流量: {wasted_mb:.2f} MB")
            print(f"  每帧有效流量: {avg_mb - wasted_mb:.2f} MB")
        
        # 深度PDF汇总
        pdf_stats = [s.depth_pdf_stats for s in self.all_scene_stats if s.depth_pdf_stats]
        if pdf_stats:
            print(f"\n  【深度PDF特性汇总】(DPGH分层依据)")
            avg_single = np.mean([p.single_peak_ratio for p in pdf_stats])
            avg_multi = np.mean([p.multi_peak_ratio for p in pdf_stats])
            avg_uncertain = np.mean([p.uncertain_ratio for p in pdf_stats])
            avg_conf = np.mean([p.avg_peak_confidence for p in pdf_stats])
            avg_spread = np.mean([p.avg_spread for p in pdf_stats])
            
            print(f"  单峰分布 (Level 0, 高置信度): {avg_single*100:.1f}%")
            print(f"  多峰分布 (Level 1, 深度模糊): {avg_multi*100:.1f}%")
            print(f"  不确定分布 (Level 2): {avg_uncertain*100:.1f}%")
            print(f"  平均峰值置信度: {avg_conf:.3f}")
            print(f"  平均分布宽度: {avg_spread:.3f}")
        
        # 空间聚类汇总
        cluster_stats = [s.spatial_cluster_stats for s in self.all_scene_stats if s.spatial_cluster_stats]
        if cluster_stats:
            print(f"\n  【空间聚类特性汇总】(FSGG分组依据)")
            avg_clusters = np.mean([c.num_clusters for c in cluster_stats])
            avg_cluster_size = np.mean([c.avg_cluster_size for c in cluster_stats])
            avg_intra = np.mean([c.avg_intra_cluster_distance for c in cluster_stats])
            avg_inter = np.mean([c.avg_inter_cluster_distance for c in cluster_stats])
            separation = avg_inter / (avg_intra + 1e-6)
            
            print(f"  平均有效簇数: {avg_clusters:.0f}")
            print(f"  平均簇大小: {avg_cluster_size:.0f}")
            print(f"  簇内平均距离: {avg_intra:.4f}")
            print(f"  簇间平均距离: {avg_inter:.4f}")
            print(f"  簇分离度: {separation:.1f}x")
        
        # 特征-位置相关性汇总
        corrs = [s.feature_position_correlation for s in self.all_scene_stats 
                 if s.feature_position_correlation > 0]
        if corrs:
            print(f"\n  【特征-位置相关性】")
            print(f"  平均相关系数: {np.mean(corrs):.3f}")
        
        # Challenge 2 关键发现
        print(f"\n  {'='*70}")
        print(f"  ███ Challenge 2 关键发现 (用于论文) ███")
        print(f"  {'='*70}")
        
        print(f"\n  1. 高斯数据量")
        if mem_stats:
            print(f"     - 总高斯数: {avg_gaussians:,.0f}")
            print(f"     - 每高斯: {avg_bytes_per:.0f} bytes")
            print(f"     - 总数据: {avg_mb:.2f} MB")
        
        print(f"\n  2. 视角相关冗余")
        print(f"     - 平均可见比例: {avg_visible*100:.1f}%")
        print(f"     - 视角无关冗余: {redundancy_ratio*100:.1f}%")
        print(f"     - Jaccard相似度: {avg_jaccard:.3f}")
        if mem_stats:
            print(f"     - 无效HBM流量: {wasted_mb:.2f} MB ({redundancy_ratio*100:.1f}%)")
        
        if pdf_stats:
            print(f"\n  3. 深度PDF特性 (DPGH依据)")
            print(f"     - 高置信单峰: {avg_single*100:.1f}%")
            print(f"     - 多峰分布: {avg_multi*100:.1f}%")
            print(f"     - 不确定分布: {avg_uncertain*100:.1f}%")
        
        if cluster_stats:
            print(f"\n  4. 空间聚类 (FSGG依据)")
            print(f"     - 语义簇数: {avg_clusters:.0f}")
            print(f"     - 每簇高斯: {avg_cluster_size:.0f}")
            print(f"     - 簇分离度: {separation:.1f}x")
        
        print()
    
    def reset(self):
        """重置累积统计"""
        self.all_scene_stats = []


# 全局实例
_global_analyzer: Optional[GaussianRedundancyAnalyzer] = None


def get_analyzer() -> GaussianRedundancyAnalyzer:
    """获取全局分析器实例"""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = GaussianRedundancyAnalyzer()
    return _global_analyzer


def reset_analyzer():
    """重置全局分析器"""
    global _global_analyzer
    if _global_analyzer is not None:
        _global_analyzer.reset()


if __name__ == "__main__":
    print("="*80)
    print("Challenge 2 高斯冗余分析脚本")
    print("="*80)
    print()
    print("此脚本需要嵌入到真实模型推理中使用，不能独立运行。")
    print()
    print("使用方法:")
    print("  bash scripts/run_all_timing_tests.sh transplat --analyze-redundancy")
    print("  bash scripts/run_all_timing_tests.sh mvsplat --analyze-redundancy")
    print("  bash scripts/run_all_timing_tests.sh depthsplat --analyze-redundancy")
    print("  bash scripts/run_all_timing_tests.sh all --analyze-redundancy")
    print()
    print("分析内容:")
    print("  1. 高斯内存占用 (精确测量每高斯字节数)")
    print("  2. 视角相关冗余 (可见比例、Jaccard相似度)")
    print("  3. 深度PDF特性 (单峰/多峰/不确定分布比例)")
    print("  4. 空间聚类特性 (簇数、簇大小、分离度)")
    print()
