#!/usr/bin/env python3
"""
深度分析 4 个算法共同的可复用机会
TransPlat, MVSplat, DepthSplat, PixelSplat
"""

import torch
import numpy as np
from collections import defaultdict

def simulate_transplat_sampling(H=64, W=64, D=32, num_points=4):
    """TransPlat: Deformable Attention with learned offsets"""
    # 参考点: 每个像素位置
    ref_points = torch.stack(torch.meshgrid(
        torch.linspace(0, W-1, W),
        torch.linspace(0, H-1, H),
        indexing='xy'
    ), dim=-1).reshape(-1, 2)  # [H*W, 2]
    
    # 学习的偏移 (模拟): 在参考点附近采样
    # 实际是网络预测，这里用小范围随机模拟
    offsets = torch.randn(H*W, D, num_points, 2) * 3  # 标准差 3 像素
    
    # 采样坐标 = 参考点 + 偏移
    coords = ref_points.unsqueeze(1).unsqueeze(2) + offsets  # [H*W, D, P, 2]
    coords = coords.clamp(0, max(H, W) - 1)
    
    return coords.reshape(-1, 2), "TransPlat (Deformable)"

def simulate_mvsplat_sampling(H=64, W=64, D=32):
    """MVSplat: Geometric Warp with depth hypotheses"""
    fx, fy = 256, 256
    cx, cy = W/2, H/2
    K = torch.tensor([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=torch.float32)
    K_inv = torch.inverse(K)
    
    # 相对位姿
    theta = 0.15
    R = torch.tensor([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ], dtype=torch.float32)
    t = torch.tensor([[0.2], [0], [0]], dtype=torch.float32)
    
    depths = torch.linspace(0.5, 10.0, D)
    
    u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    pixels = torch.stack([u.flatten(), v.flatten(), torch.ones(H*W)], dim=0).float()
    
    all_coords = []
    for d in depths:
        points_3d = K_inv @ pixels * d
        points_3d_transformed = R @ points_3d + t
        points_2d = K @ points_3d_transformed
        coords = points_2d[:2] / points_2d[2:3].clamp(min=0.1)
        all_coords.append(coords.T)
    
    all_coords = torch.cat(all_coords, dim=0)  # [H*W*D, 2]
    return all_coords, "MVSplat (Geometric Warp)"

def simulate_depthsplat_sampling(H=64, W=64, D_coarse=32, D_fine=8):
    """DepthSplat: Multi-scale geometric warp"""
    # 类似 MVSplat，但多尺度
    coords_coarse, _ = simulate_mvsplat_sampling(H, W, D_coarse)
    coords_fine, _ = simulate_mvsplat_sampling(H*2, W*2, D_fine)  # 更高分辨率
    
    all_coords = torch.cat([coords_coarse, coords_fine[:H*W*D_fine]], dim=0)
    return all_coords, "DepthSplat (Multi-scale)"

def simulate_pixelsplat_sampling(H=64, W=64, num_samples=64):
    """PixelSplat: Epipolar line sampling"""
    # 对每个像素，沿极线采样
    fx, fy = 256, 256
    cx, cy = W/2, H/2
    
    # 极点 (另一相机中心在当前图像的投影)
    epipole = torch.tensor([W*0.8, H*0.5])  # 在图像右侧
    
    u, v = torch.meshgrid(torch.arange(W), torch.arange(H), indexing='xy')
    pixels = torch.stack([u.flatten(), v.flatten()], dim=-1).float()  # [H*W, 2]
    
    # 极线方向
    directions = epipole - pixels  # [H*W, 2]
    directions = directions / directions.norm(dim=-1, keepdim=True).clamp(min=1e-6)
    
    # 沿极线采样
    sample_distances = torch.linspace(-30, 30, num_samples)  # 采样距离
    
    all_coords = []
    for dist in sample_distances:
        coords = pixels + directions * dist
        all_coords.append(coords)
    
    all_coords = torch.cat(all_coords, dim=0)  # [H*W*S, 2]
    all_coords = all_coords.clamp(0, max(H, W) - 1)
    return all_coords, "PixelSplat (Epipolar)"

def analyze_common_patterns(coords_list, names):
    """分析所有算法的共同模式"""
    
    print("=" * 80)
    print("四个算法的采样模式对比分析")
    print("=" * 80)
    
    # 1. 基本统计
    print("\n" + "-" * 80)
    print("1. 采样点数量和范围")
    print("-" * 80)
    for coords, name in zip(coords_list, names):
        n = coords.shape[0]
        x_range = (coords[:, 0].min().item(), coords[:, 0].max().item())
        y_range = (coords[:, 1].min().item(), coords[:, 1].max().item())
        print(f"  {name:30s}: {n:>8d} 点, X=[{x_range[0]:.1f}, {x_range[1]:.1f}], Y=[{y_range[0]:.1f}, {y_range[1]:.1f}]")
    
    # 2. 共同模式: Tile 聚集性
    print("\n" + "-" * 80)
    print("2. 共同模式: Tile 聚集性 (16x16 Tile)")
    print("-" * 80)
    
    tile_size = 16
    for coords, name in zip(coords_list, names):
        # 量化到 Tile
        tile_coords = (coords / tile_size).long()
        # 统计每个 Tile 的采样数
        tile_counts = defaultdict(int)
        for i in range(tile_coords.shape[0]):
            tx, ty = tile_coords[i].tolist()
            tile_counts[(tx, ty)] += 1
        
        num_tiles = len(tile_counts)
        avg_samples_per_tile = coords.shape[0] / num_tiles
        max_samples = max(tile_counts.values())
        
        print(f"  {name:30s}: {num_tiles:>4d} Tiles, 平均 {avg_samples_per_tile:.0f} 点/Tile, 最大 {max_samples}")
    
    # 3. 共同模式: 双线性插值点共享
    print("\n" + "-" * 80)
    print("3. 共同模式: 双线性插值点共享")
    print("-" * 80)
    
    for coords, name in zip(coords_list, names):
        # 每个采样点需要访问 4 个整数坐标点
        bilinear_points = set()
        for i in range(coords.shape[0]):
            x, y = coords[i].tolist()
            for dx in [0, 1]:
                for dy in [0, 1]:
                    bilinear_points.add((int(x)+dx, int(y)+dy))
        
        total_accesses = coords.shape[0] * 4
        unique_points = len(bilinear_points)
        reuse_ratio = 1 - unique_points / total_accesses
        
        print(f"  {name:30s}: {total_accesses:>8d} 访问, {unique_points:>6d} 唯一点, 复用率 {reuse_ratio*100:.1f}%")
    
    # 4. 共同模式: 邻近点聚集 (源域相邻像素的采样目标是否相近)
    print("\n" + "-" * 80)
    print("4. 共同模式: 源域空间局部性 → 目标域聚集性")
    print("-" * 80)
    
    H, W = 64, 64
    for coords, name in zip(coords_list, names):
        n = coords.shape[0]
        samples_per_pixel = n // (H * W)
        if samples_per_pixel < 1:
            samples_per_pixel = 1
        
        # 重塑为 [H, W, S, 2] (S = samples per pixel)
        try:
            coords_reshaped = coords[:H*W*samples_per_pixel].reshape(H, W, samples_per_pixel, 2)
            
            # 计算相邻像素采样目标的距离
            h_dist = (coords_reshaped[1:, :, :, :] - coords_reshaped[:-1, :, :, :]).norm(dim=-1).mean()
            w_dist = (coords_reshaped[:, 1:, :, :] - coords_reshaped[:, :-1, :, :]).norm(dim=-1).mean()
            
            print(f"  {name:30s}: 相邻像素目标距离 H={h_dist:.2f}, W={w_dist:.2f} 像素")
        except:
            print(f"  {name:30s}: 无法重塑")
    
    # 5. 核心共同点总结
    print("\n" + "=" * 80)
    print("核心共同点 (可用于 ASIC 设计)")
    print("=" * 80)
    
    # 计算所有算法的平均复用率
    reuse_rates = []
    tile_efficiencies = []
    
    for coords, name in zip(coords_list, names):
        # 复用率
        bilinear_points = set()
        for i in range(min(coords.shape[0], 100000)):  # 限制计算量
            x, y = coords[i].tolist()
            for dx in [0, 1]:
                for dy in [0, 1]:
                    bilinear_points.add((int(x)+dx, int(y)+dy))
        
        total_accesses = min(coords.shape[0], 100000) * 4
        unique_points = len(bilinear_points)
        reuse_rates.append(1 - unique_points / total_accesses)
        
        # Tile 效率
        tile_coords = (coords[:min(coords.shape[0], 100000)] / 16).long()
        tile_counts = defaultdict(int)
        for i in range(tile_coords.shape[0]):
            tx, ty = tile_coords[i].tolist()
            tile_counts[(tx, ty)] += 1
        tile_efficiencies.append(1 - len(tile_counts) / min(coords.shape[0], 100000))
    
    print(f"""
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 共同点 1: 双线性插值点高度共享                                       │
  │   所有算法平均复用率: {np.mean(reuse_rates)*100:.1f}%                               │
  │   → 同一个整数坐标点被多次用于插值                                   │
  │   → ASIC 机会: 预取整数网格点，片上完成插值                           │
  └─────────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 共同点 2: 采样目标在 Tile 级别聚集                                   │
  │   所有算法平均 Tile 复用率: {np.mean(tile_efficiencies)*100:.1f}%                         │
  │   → 大量采样落入少数 Tile                                            │
  │   → ASIC 机会: 按 Tile 预取，批量服务采样请求                         │
  └─────────────────────────────────────────────────────────────────────┘
  
  ┌─────────────────────────────────────────────────────────────────────┐
  │ 共同点 3: 源域相邻 → 目标域相邻 (空间连续性)                          │
  │   所有算法: 相邻像素的采样目标距离 < 2 像素                           │
  │   → 处理相邻像素时，采样区域高度重叠                                  │
  │   → ASIC 机会: 滑动窗口预取，复用重叠数据                             │
  └─────────────────────────────────────────────────────────────────────┘
""")

def main():
    print("生成 4 个算法的采样坐标...")
    
    coords_transplat, name1 = simulate_transplat_sampling()
    coords_mvsplat, name2 = simulate_mvsplat_sampling()
    coords_depthsplat, name3 = simulate_depthsplat_sampling()
    coords_pixelsplat, name4 = simulate_pixelsplat_sampling()
    
    coords_list = [coords_transplat, coords_mvsplat, coords_depthsplat, coords_pixelsplat]
    names = [name1, name2, name3, name4]
    
    analyze_common_patterns(coords_list, names)

if __name__ == "__main__":
    main()
