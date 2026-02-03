#!/usr/bin/env python3
"""
Analyze correlation between feature similarity and depth difference.

This script validates the hypothesis for Challenge 1:
- Within local tiles of the feature map, how many pixel pairs have similar features?
- Do similar features correlate with similar depths?

Analysis approach:
1. Divide 64×64 feature map into 4×4 tiles (16×16 = 256 tiles)
2. Within each tile, compute all pairwise feature similarities
3. Bin pairs by similarity threshold and compute depth difference stats
4. Average across all tiles
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional, List
import numpy as np
from collections import defaultdict


class FeatureDepthCorrelationAnalyzer:
    """Analyzes correlation between feature similarity and depth differences.
    
    Divides feature map into tiles and analyzes similarity distribution within each tile.
    """
    
    def __init__(self, tile_size: int = 4, device: str = 'cuda'):
        """
        Args:
            tile_size: Size of each tile (e.g., 4 means 4×4 tiles)
            device: Device to run computations on
        """
        self.tile_size = tile_size
        self.device = device
        
        # Similarity bins for analysis (threshold 0.7 for "high similarity")
        self.sim_bins = [
            (0.7, 1.01, ">0.7 (high)"),
            (0.5, 0.7, "0.5-0.7 (medium)"),
            (0.0, 0.5, "<0.5 (low)"),
        ]
        
        # Accumulated statistics across all tiles and scenes
        self.accumulated_stats = {
            'tile_stats': [],  # List of per-tile statistics
            'global_by_sim': defaultdict(lambda: {
                'count': 0,
                'depth_diffs': [],
                'rel_depth_diffs': [],
            }),
        }
        self.total_scenes = 0
        self.total_tiles = 0
        
    def analyze_single_scene(
        self,
        features: torch.Tensor,
        depths: torch.Tensor,
        view_idx: int = 0,
    ) -> Dict:
        """
        Analyze feature-depth correlation for a single scene.
        
        Args:
            features: Feature tensor [B, V, C, H_feat, W_feat] 
            depths: Depth tensor [B, V, H, W] or [B, V, H*W, 1, 1]
            view_idx: Which view to analyze
            
        Returns:
            Dictionary with correlation statistics
        """
        # Handle different depth tensor formats
        if depths.dim() == 5:  # [B, V, H*W, 1, 1]
            b, v, hw, _, _ = depths.shape
            h = w = int(np.sqrt(hw))
            depths = depths.view(b, v, h, w)
        
        b, v, h_depth, w_depth = depths.shape
        _, _, c, h_feat, w_feat = features.shape
        
        # Use first batch
        feat = features[0, view_idx]  # [C, H_feat, W_feat]
        depth = depths[0, view_idx]   # [H_depth, W_depth]
        
        # Work at feature map resolution (typically 64×64)
        # Downsample depth to match feature map if needed
        if h_feat != h_depth or w_feat != w_depth:
            depth = F.interpolate(
                depth.unsqueeze(0).unsqueeze(0), 
                size=(h_feat, w_feat), 
                mode='bilinear', 
                align_corners=False
            ).squeeze()  # [H_feat, W_feat]
        
        h, w = h_feat, w_feat
        
        # Normalize features for cosine similarity
        feat_norm = F.normalize(feat, p=2, dim=0)  # [C, H, W]
        
        mean_depth = depth.mean()
        
        # Analyze by tiles
        num_tiles_h = h // self.tile_size
        num_tiles_w = w // self.tile_size
        
        scene_tile_stats = []
        
        for ti in range(num_tiles_h):
            for tj in range(num_tiles_w):
                # Extract tile
                y_start = ti * self.tile_size
                y_end = y_start + self.tile_size
                x_start = tj * self.tile_size
                x_end = x_start + self.tile_size
                
                tile_feat = feat_norm[:, y_start:y_end, x_start:x_end]  # [C, T, T]
                tile_depth = depth[y_start:y_end, x_start:x_end]  # [T, T]
                
                tile_stats = self._analyze_tile(tile_feat, tile_depth, mean_depth)
                scene_tile_stats.append(tile_stats)
                self.total_tiles += 1
        
        self.total_scenes += 1
        
        return {
            'num_tiles': len(scene_tile_stats),
            'feature_shape': (h, w),
            'tile_size': self.tile_size,
        }
    
    def _analyze_tile(
        self, 
        tile_feat: torch.Tensor, 
        tile_depth: torch.Tensor,
        mean_depth: torch.Tensor
    ) -> Dict:
        """Analyze all pairwise similarities within a single tile."""
        c, th, tw = tile_feat.shape
        num_points = th * tw  # e.g., 16 for 4×4 tile
        
        # Flatten tile (use contiguous() because tile is a slice of larger tensor)
        feat_flat = tile_feat.contiguous().view(c, -1).t()  # [num_points, C]
        depth_flat = tile_depth.contiguous().view(-1)  # [num_points]
        
        # Compute all pairwise similarities
        # sim_matrix[i, j] = cosine_similarity(feat[i], feat[j])
        sim_matrix = feat_flat @ feat_flat.t()  # [num_points, num_points]
        
        # Get upper triangle indices (excluding diagonal)
        triu_indices = torch.triu_indices(num_points, num_points, offset=1, device=self.device)
        similarities = sim_matrix[triu_indices[0], triu_indices[1]]
        
        # Compute pairwise depth differences
        depth_diff_matrix = torch.abs(depth_flat.unsqueeze(1) - depth_flat.unsqueeze(0))
        depth_diffs = depth_diff_matrix[triu_indices[0], triu_indices[1]]
        rel_depth_diffs = depth_diffs / mean_depth * 100  # percentage
        
        # Count pairs in each similarity bin
        tile_stats = {
            'total_pairs': similarities.numel(),
            'bins': {}
        }
        
        for low, high, name in self.sim_bins:
            mask = (similarities >= low) & (similarities < high)
            count = mask.sum().item()
            
            tile_stats['bins'][name] = {
                'count': count,
                'percentage': count / similarities.numel() * 100 if similarities.numel() > 0 else 0,
            }
            
            if count > 0:
                bin_depth_diffs = depth_diffs[mask]
                bin_rel_diffs = rel_depth_diffs[mask]
                
                tile_stats['bins'][name]['depth_diff_median'] = bin_rel_diffs.median().item()
                tile_stats['bins'][name]['depth_diff_mean'] = bin_rel_diffs.mean().item()
                
                # Accumulate globally
                self.accumulated_stats['global_by_sim'][name]['count'] += count
                self.accumulated_stats['global_by_sim'][name]['depth_diffs'].extend(
                    bin_depth_diffs.cpu().numpy().tolist()
                )
                self.accumulated_stats['global_by_sim'][name]['rel_depth_diffs'].extend(
                    bin_rel_diffs.cpu().numpy().tolist()
                )
        
        self.accumulated_stats['tile_stats'].append(tile_stats)
        
        return tile_stats
    
    def get_accumulated_stats(self) -> Dict:
        """Get accumulated statistics across all analyzed scenes."""
        results = {
            'total_scenes': self.total_scenes,
            'total_tiles': self.total_tiles,
            'tile_size': self.tile_size,
            'points_per_tile': self.tile_size * self.tile_size,
            'pairs_per_tile': (self.tile_size * self.tile_size) * (self.tile_size * self.tile_size - 1) // 2,
            'by_similarity': {},
            'tile_averages': {},
        }
        
        # Compute global statistics by similarity bin
        for name, data in self.accumulated_stats['global_by_sim'].items():
            if data['count'] > 0:
                rel_diffs = np.array(data['rel_depth_diffs'])
                results['by_similarity'][name] = {
                    'total_pairs': data['count'],
                    'median_depth_diff_pct': float(np.median(rel_diffs)),
                    'mean_depth_diff_pct': float(np.mean(rel_diffs)),
                    'std_depth_diff_pct': float(np.std(rel_diffs)),
                    'p25': float(np.percentile(rel_diffs, 25)),
                    'p75': float(np.percentile(rel_diffs, 75)),
                }
        
        # Compute average percentages across all tiles
        if self.accumulated_stats['tile_stats']:
            for name in [b[2] for b in self.sim_bins]:
                percentages = []
                for ts in self.accumulated_stats['tile_stats']:
                    if name in ts['bins']:
                        percentages.append(ts['bins'][name]['percentage'])
                
                if percentages:
                    results['tile_averages'][name] = {
                        'mean_percentage': float(np.mean(percentages)),
                        'std_percentage': float(np.std(percentages)),
                        'min_percentage': float(np.min(percentages)),
                        'max_percentage': float(np.max(percentages)),
                    }
        
        return results
    
    def print_report(self, model_name: str = ""):
        """Print formatted analysis report."""
        stats = self.get_accumulated_stats()
        
        print("\n" + "=" * 100)
        print(f"  FEATURE-DEPTH CORRELATION ANALYSIS (TILE-BASED)")
        print(f"  {model_name.upper() if model_name else 'MODEL'}")
        print("=" * 100)
        print(f"  Scenes analyzed: {stats['total_scenes']}")
        print(f"  Total tiles analyzed: {stats['total_tiles']}")
        print(f"  Tile size: {stats['tile_size']}×{stats['tile_size']} = {stats['points_per_tile']} points")
        print(f"  Pairs per tile: {stats['pairs_per_tile']}")
        print("=" * 100)
        
        # Part 1: Similarity distribution within tiles
        print("\n" + "-" * 100)
        print("  PART 1: FEATURE SIMILARITY DISTRIBUTION WITHIN TILES")
        print("  (What percentage of point pairs in each tile fall into each similarity bin?)")
        print("-" * 100)
        print(f"{'Similarity Range':<20} {'Mean %':>12} {'Std %':>12} {'Min %':>12} {'Max %':>12}")
        print("-" * 100)
        
        ordered_bins = [">0.7 (high)", "0.5-0.7 (medium)", "<0.5 (low)"]
        
        for name in ordered_bins:
            if name in stats['tile_averages']:
                data = stats['tile_averages'][name]
                print(f"{name:<20} {data['mean_percentage']:>11.1f}% {data['std_percentage']:>11.1f}% "
                      f"{data['min_percentage']:>11.1f}% {data['max_percentage']:>11.1f}%")
        
        print("-" * 100)
        
        # Part 2: Depth difference by similarity
        print("\n" + "-" * 100)
        print("  PART 2: DEPTH DIFFERENCE BY FEATURE SIMILARITY")
        print("  (For pairs in each similarity bin, what is the depth difference?)")
        print("-" * 100)
        print(f"{'Similarity':<20} {'Total Pairs':>15} {'Median Δd%':>12} {'Mean Δd%':>12} "
              f"{'Std':>10} {'P25':>8} {'P75':>8}")
        print("-" * 100)
        
        for name in ordered_bins:
            if name in stats['by_similarity']:
                data = stats['by_similarity'][name]
                print(f"{name:<20} {data['total_pairs']:>15,} {data['median_depth_diff_pct']:>11.2f}% "
                      f"{data['mean_depth_diff_pct']:>11.2f}% {data['std_depth_diff_pct']:>9.2f} "
                      f"{data['p25']:>7.2f} {data['p75']:>7.2f}")
        
        print("-" * 100)
        
        # Summary
        print("\n" + "=" * 100)
        print("  KEY FINDINGS FOR CHALLENGE 1 (threshold=0.7):")
        print("=" * 100)
        
        high_key = ">0.7 (high)"
        med_key = "0.5-0.7 (medium)"
        low_key = "<0.5 (low)"
        
        if high_key in stats['tile_averages'] and low_key in stats['tile_averages']:
            high_pct = stats['tile_averages'][high_key]['mean_percentage']
            low_pct = stats['tile_averages'][low_key]['mean_percentage']
            med_pct = stats['tile_averages'].get(med_key, {}).get('mean_percentage', 0)
            print(f"  1. Within each {stats['tile_size']}×{stats['tile_size']} tile:")
            print(f"     - {high_pct:.1f}% of point pairs have HIGH similarity (>0.7)")
            print(f"     - {med_pct:.1f}% of point pairs have MEDIUM similarity (0.5-0.7)")
            print(f"     - {low_pct:.1f}% of point pairs have LOW similarity (<0.5)")
        
        if high_key in stats['by_similarity'] and low_key in stats['by_similarity']:
            high_diff = stats['by_similarity'][high_key]['median_depth_diff_pct']
            low_diff = stats['by_similarity'][low_key]['median_depth_diff_pct']
            print(f"\n  2. Feature similarity predicts depth similarity:")
            print(f"     - HIGH sim (>0.7):  depth diff median = {high_diff:.2f}%")
            print(f"     - LOW sim (<0.5):   depth diff median = {low_diff:.2f}%")
            if high_diff > 0:
                print(f"     - Ratio: {low_diff/high_diff:.1f}x larger depth difference for low-similarity pairs")
        
        if high_key in stats['tile_averages']:
            reusable = stats['tile_averages'][high_key]['mean_percentage']
            print(f"\n  3. Depth reuse potential:")
            print(f"     - {reusable:.1f}% of point pairs within tiles have similarity > 0.7")
            print(f"     - These pairs are candidates for depth reuse in FSDR")
        
        print("=" * 100)
        print()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.accumulated_stats = {
            'tile_stats': [],
            'global_by_sim': defaultdict(lambda: {
                'count': 0,
                'depth_diffs': [],
                'rel_depth_diffs': [],
            }),
        }
        self.total_scenes = 0
        self.total_tiles = 0


# Global analyzer instance for integration with models
_global_analyzer: Optional[FeatureDepthCorrelationAnalyzer] = None


def get_analyzer(tile_size: int = 4) -> FeatureDepthCorrelationAnalyzer:
    """Get or create global analyzer instance."""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = FeatureDepthCorrelationAnalyzer(tile_size=tile_size)
    return _global_analyzer


def reset_analyzer():
    """Reset global analyzer."""
    global _global_analyzer
    if _global_analyzer is not None:
        _global_analyzer.reset()


def analyze_and_report(features: torch.Tensor, depths: torch.Tensor, 
                       model_name: str = "", view_idx: int = 0):
    """
    Convenience function to analyze a single scene and accumulate stats.
    
    Args:
        features: Feature tensor [B, V, C, H_feat, W_feat]
        depths: Depth tensor [B, V, H, W] or [B, V, H*W, 1, 1]
        model_name: Model name for reporting
        view_idx: Which view to analyze
    """
    analyzer = get_analyzer()
    analyzer.analyze_single_scene(features, depths, view_idx)


def print_final_report(model_name: str = ""):
    """Print final accumulated report."""
    analyzer = get_analyzer()
    analyzer.print_report(model_name)


if __name__ == "__main__":
    # Test with synthetic data
    print("Testing FeatureDepthCorrelationAnalyzer with synthetic data...")
    
    analyzer = FeatureDepthCorrelationAnalyzer(tile_size=4)
    
    # Create synthetic data with spatial correlation
    B, V, C, H, W = 1, 2, 128, 64, 64
    
    # Create smooth feature map (neighboring pixels are similar)
    base_features = torch.randn(B, V, C, H//4, W//4).cuda()
    features = F.interpolate(base_features.view(B*V, C, H//4, W//4), 
                            size=(H, W), mode='bilinear', align_corners=False)
    features = features.view(B, V, C, H, W)
    features = features + 0.1 * torch.randn_like(features)  # Add small noise
    
    # Create smooth depth map
    base_depths = torch.randn(B, V, H//4, W//4).cuda().abs() * 5 + 2
    depths = F.interpolate(base_depths, size=(H, W), mode='bilinear', align_corners=False)
    depths = depths + 0.05 * torch.randn_like(depths)  # Add small noise
    
    # Analyze
    for _ in range(2):  # Simulate 2 scenes
        stats = analyzer.analyze_single_scene(features, depths)
    
    analyzer.print_report("synthetic_test")
