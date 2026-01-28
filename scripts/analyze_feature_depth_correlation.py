#!/usr/bin/env python3
"""
Analyze correlation between feature similarity and depth difference.

This script validates the hypothesis:
- Pixels with feature cosine similarity > 0.92 have depth difference median ~2.3% of mean depth
- Pixels with feature cosine similarity < 0.7 have depth difference median ~12.7% of mean depth
"""

import torch
import torch.nn.functional as F
from typing import Dict, Tuple, Optional
import numpy as np
from collections import defaultdict


class FeatureDepthCorrelationAnalyzer:
    """Analyzes correlation between feature similarity and depth differences."""
    
    def __init__(self, num_samples: int = 4096, device: str = 'cuda'):
        """
        Args:
            num_samples: Number of pixel pairs to sample for analysis
            device: Device to run computations on
        """
        self.num_samples = num_samples
        self.device = device
        
        # Similarity bins for analysis
        self.sim_bins = [
            (0.92, 1.0, "high (>0.92)"),
            (0.8, 0.92, "medium-high (0.8-0.92)"),
            (0.7, 0.8, "medium (0.7-0.8)"),
            (0.5, 0.7, "low (0.5-0.7)"),
            (0.0, 0.5, "very low (<0.5)"),
        ]
        
        # Accumulate statistics across multiple scenes
        self.accumulated_stats = defaultdict(lambda: {
            'depth_diffs': [],
            'rel_depth_diffs': [],
            'count': 0
        })
        self.total_scenes = 0
        
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
        
        # Upsample or downsample to match sizes
        if h_feat != h_depth or w_feat != w_depth:
            # Interpolate features to depth resolution
            feat = F.interpolate(
                feat.unsqueeze(0), 
                size=(h_depth, w_depth), 
                mode='bilinear', 
                align_corners=False
            ).squeeze(0)  # [C, H, W]
        
        # Flatten to [C, H*W] and [H*W]
        feat_flat = feat.view(c, -1).t()  # [H*W, C]
        depth_flat = depth.view(-1)       # [H*W]
        
        num_pixels = feat_flat.shape[0]
        
        # Normalize features for cosine similarity
        feat_norm = F.normalize(feat_flat, p=2, dim=1)  # [H*W, C]
        
        # Sample random pixel pairs
        indices_a = torch.randint(0, num_pixels, (self.num_samples,), device=self.device)
        indices_b = torch.randint(0, num_pixels, (self.num_samples,), device=self.device)
        
        # Compute cosine similarities
        feat_a = feat_norm[indices_a]  # [num_samples, C]
        feat_b = feat_norm[indices_b]  # [num_samples, C]
        similarities = (feat_a * feat_b).sum(dim=1)  # [num_samples]
        
        # Compute depth differences
        depth_a = depth_flat[indices_a]  # [num_samples]
        depth_b = depth_flat[indices_b]  # [num_samples]
        depth_diffs = torch.abs(depth_a - depth_b)  # [num_samples]
        
        # Compute relative depth differences (% of mean depth)
        mean_depth = depth_flat.mean()
        rel_depth_diffs = depth_diffs / mean_depth * 100  # percentage
        
        # Accumulate statistics by similarity bin
        for low, high, name in self.sim_bins:
            mask = (similarities >= low) & (similarities < high)
            if mask.any():
                bin_depth_diffs = depth_diffs[mask].cpu().numpy()
                bin_rel_diffs = rel_depth_diffs[mask].cpu().numpy()
                
                self.accumulated_stats[name]['depth_diffs'].extend(bin_depth_diffs.tolist())
                self.accumulated_stats[name]['rel_depth_diffs'].extend(bin_rel_diffs.tolist())
                self.accumulated_stats[name]['count'] += mask.sum().item()
        
        self.total_scenes += 1
        
        return self._compute_scene_stats(similarities, depth_diffs, rel_depth_diffs, mean_depth)
    
    def _compute_scene_stats(
        self, 
        similarities: torch.Tensor, 
        depth_diffs: torch.Tensor,
        rel_depth_diffs: torch.Tensor,
        mean_depth: torch.Tensor
    ) -> Dict:
        """Compute statistics for a single scene."""
        stats = {
            'mean_depth': mean_depth.item(),
            'bins': {}
        }
        
        for low, high, name in self.sim_bins:
            mask = (similarities >= low) & (similarities < high)
            if mask.any():
                bin_rel_diffs = rel_depth_diffs[mask]
                stats['bins'][name] = {
                    'count': mask.sum().item(),
                    'median_rel_diff': bin_rel_diffs.median().item(),
                    'mean_rel_diff': bin_rel_diffs.mean().item(),
                    'std_rel_diff': bin_rel_diffs.std().item(),
                }
        
        return stats
    
    def get_accumulated_stats(self) -> Dict:
        """Get accumulated statistics across all analyzed scenes."""
        results = {
            'total_scenes': self.total_scenes,
            'bins': {}
        }
        
        for name, data in self.accumulated_stats.items():
            if data['count'] > 0:
                rel_diffs = np.array(data['rel_depth_diffs'])
                results['bins'][name] = {
                    'count': data['count'],
                    'median_rel_diff': float(np.median(rel_diffs)),
                    'mean_rel_diff': float(np.mean(rel_diffs)),
                    'std_rel_diff': float(np.std(rel_diffs)),
                    'p25': float(np.percentile(rel_diffs, 25)),
                    'p75': float(np.percentile(rel_diffs, 75)),
                }
        
        return results
    
    def print_report(self, model_name: str = ""):
        """Print formatted analysis report."""
        stats = self.get_accumulated_stats()
        
        print("\n" + "=" * 90)
        print(f"  FEATURE-DEPTH CORRELATION ANALYSIS{' - ' + model_name.upper() if model_name else ''}")
        print(f"  (Analyzed {stats['total_scenes']} scene(s), {self.num_samples} pixel pairs per scene)")
        print("=" * 90)
        print()
        print(f"{'Similarity Range':<25} {'Count':>10} {'Median Δd%':>12} {'Mean Δd%':>12} {'Std':>10} {'P25':>8} {'P75':>8}")
        print("-" * 90)
        
        # Sort bins by similarity (high to low)
        ordered_bins = ["high (>0.92)", "medium-high (0.8-0.92)", "medium (0.7-0.8)", 
                       "low (0.5-0.7)", "very low (<0.5)"]
        
        for name in ordered_bins:
            if name in stats['bins']:
                data = stats['bins'][name]
                print(f"{name:<25} {data['count']:>10} {data['median_rel_diff']:>11.2f}% "
                      f"{data['mean_rel_diff']:>11.2f}% {data['std_rel_diff']:>9.2f} "
                      f"{data['p25']:>7.2f} {data['p75']:>7.2f}")
        
        print("-" * 90)
        
        # Summary comparison
        if "high (>0.92)" in stats['bins'] and "low (0.5-0.7)" in stats['bins']:
            high_sim = stats['bins']["high (>0.92)"]['median_rel_diff']
            low_sim = stats['bins']["low (0.5-0.7)"]['median_rel_diff']
            print()
            print("KEY FINDINGS:")
            print(f"  • High similarity (>0.92): median depth diff = {high_sim:.2f}% of mean depth")
            print(f"  • Low similarity (<0.7):   median depth diff = {low_sim:.2f}% of mean depth")
            print(f"  • Ratio: {low_sim/high_sim:.1f}x larger depth variance for low-similarity pairs")
        
        print("=" * 90)
        print()
    
    def reset(self):
        """Reset accumulated statistics."""
        self.accumulated_stats = defaultdict(lambda: {
            'depth_diffs': [],
            'rel_depth_diffs': [],
            'count': 0
        })
        self.total_scenes = 0


# Global analyzer instance for integration with models
_global_analyzer: Optional[FeatureDepthCorrelationAnalyzer] = None


def get_analyzer(num_samples: int = 4096) -> FeatureDepthCorrelationAnalyzer:
    """Get or create global analyzer instance."""
    global _global_analyzer
    if _global_analyzer is None:
        _global_analyzer = FeatureDepthCorrelationAnalyzer(num_samples=num_samples)
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
    # Test with random data
    print("Testing FeatureDepthCorrelationAnalyzer with synthetic data...")
    
    analyzer = FeatureDepthCorrelationAnalyzer(num_samples=10000)
    
    # Create synthetic data where similar features have similar depths
    B, V, C, H, W = 1, 2, 128, 64, 64
    
    # Create clustered features (pixels close together have similar features)
    features = torch.randn(B, V, C, H, W).cuda()
    
    # Create depth map correlated with feature patterns
    depths = torch.randn(B, V, H, W).cuda().abs() * 10 + 1
    
    # Analyze
    for _ in range(3):  # Simulate 3 scenes
        stats = analyzer.analyze_single_scene(features, depths)
    
    analyzer.print_report("synthetic_test")
