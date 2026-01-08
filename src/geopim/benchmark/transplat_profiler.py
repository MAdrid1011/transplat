"""
TransPlat 真实推理性能分析

对真实 TransPlat 模型进行 profiling，测量几何采样部分的性能。

用法:
    conda activate transplat
    python -m src.geopim.benchmark.transplat_profiler \
        --checkpoint checkpoints/re10k.ckpt \
        --config experiment=re10k
"""

import argparse
import time
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple
from contextlib import contextmanager
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@dataclass
class ProfilingResult:
    """Profiling 结果"""
    
    name: str
    total_time_ms: float
    call_count: int = 1
    sub_results: Dict[str, 'ProfilingResult'] = field(default_factory=dict)
    
    @property
    def avg_time_ms(self) -> float:
        return self.total_time_ms / self.call_count if self.call_count > 0 else 0
    
    def add_sub(self, name: str, time_ms: float):
        if name not in self.sub_results:
            self.sub_results[name] = ProfilingResult(name, 0, 0)
        self.sub_results[name].total_time_ms += time_ms
        self.sub_results[name].call_count += 1


class CUDATimer:
    """CUDA 计时器"""
    
    def __init__(self):
        self.start_event = torch.cuda.Event(enable_timing=True)
        self.end_event = torch.cuda.Event(enable_timing=True)
        
    @contextmanager
    def time(self):
        self.start_event.record()
        yield
        self.end_event.record()
        torch.cuda.synchronize()
        
    def elapsed_ms(self) -> float:
        return self.start_event.elapsed_time(self.end_event)


class TransPlatProfiler:
    """
    TransPlat 推理 Profiler
    
    Hook 到模型各个部分测量时间。
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.timers: Dict[str, CUDATimer] = {}
        self.results: Dict[str, ProfilingResult] = {}
        
    def get_timer(self, name: str) -> CUDATimer:
        if name not in self.timers:
            self.timers[name] = CUDATimer()
        return self.timers[name]
    
    def record(self, name: str, time_ms: float):
        if name not in self.results:
            self.results[name] = ProfilingResult(name, 0, 0)
        self.results[name].total_time_ms += time_ms
        self.results[name].call_count += 1
        
    def profile_depth_predictor_forward(
        self,
        depth_predictor,
        features: torch.Tensor,
        intrinsics: torch.Tensor,
        extrinsics: torch.Tensor,
        near: torch.Tensor,
        far: torch.Tensor,
        **kwargs
    ) -> Tuple[torch.Tensor, ...]:
        """
        Profile DepthPredictorTrans forward
        
        测量几何采样的核心计算。
        """
        # 准备阶段
        timer = self.get_timer('prepare')
        with timer.time():
            from src.model.encoder.matching.depth_predictor_trans import prepare_feat_proj_data_lists
            feat_comb_lists, intr_curr, pose_curr_lists, disp_candi_curr = (
                prepare_feat_proj_data_lists(
                    features,
                    intrinsics,
                    extrinsics,
                    near,
                    far,
                    num_samples=depth_predictor.num_depth_candidates,
                )
            )
        self.record('prepare_data', timer.elapsed_ms())
        
        # Grid 计算 (几何采样核心)
        timer = self.get_timer('grid_compute')
        with timer.time():
            from src.model.encoder.matching.depth_predictor_trans import calculate_grid
            b, v, c, h, w = features.shape
            
            grid, depth = calculate_grid(
                intr_curr,
                pose_curr_lists[0],
                1.0 / disp_candi_curr.repeat([1, 1, *features.shape[-2:]]),
            )
        self.record('grid_compute', timer.elapsed_ms())
        
        # Grid sample (关键的采样操作)
        timer = self.get_timer('grid_sample')
        with timer.time():
            # 模拟 match_two 中的 grid_sample
            features_rearranged = rearrange(features, "b v c h w -> (v b) c h w")
            sampled = F.grid_sample(
                features_rearranged,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
        self.record('grid_sample', timer.elapsed_ms())
        
        # 完整 forward
        timer = self.get_timer('full_forward')
        with timer.time():
            result = depth_predictor(
                features, intrinsics, extrinsics, near, far, **kwargs
            )
        self.record('full_forward', timer.elapsed_ms())
        
        return result
    
    def print_summary(self):
        """打印 profiling 总结"""
        print("\n" + "=" * 60)
        print("TransPlat Profiling Summary")
        print("=" * 60)
        
        for name, result in sorted(self.results.items()):
            print(f"\n{name}:")
            print(f"  Total: {result.total_time_ms:.2f} ms ({result.call_count} calls)")
            print(f"  Avg:   {result.avg_time_ms:.2f} ms")


def profile_geometry_sampling_operations(device='cuda', num_iterations=50):
    """
    Profile 几何采样相关操作
    
    不需要完整模型，直接测试核心操作。
    """
    print("\n" + "=" * 60)
    print("Profiling Geometry Sampling Operations")
    print("=" * 60)
    
    dtype = torch.float16
    results = {}
    
    # TransPlat 实际配置
    # 根据 depth_predictor_trans.py:
    # - feature size: [B, V, C, H, W] where H=W=64, C=128
    # - depth candidates: 32
    # - transformer queries: 64x64 = 4096
    
    configs = [
        {
            'name': 'TransPlat-V2-H64',
            'B': 1, 'V': 2, 'C': 128, 'H': 64, 'W': 64,
            'D': 32,  # depth candidates
        },
        {
            'name': 'TransPlat-V2-H128',
            'B': 1, 'V': 2, 'C': 128, 'H': 128, 'W': 128,
            'D': 32,
        },
        {
            'name': 'TransPlat-V4-H64',
            'B': 1, 'V': 4, 'C': 128, 'H': 64, 'W': 64,
            'D': 32,
        },
    ]
    
    for cfg in configs:
        B, V, C, H, W, D = cfg['B'], cfg['V'], cfg['C'], cfg['H'], cfg['W'], cfg['D']
        name = cfg['name']
        
        print(f"\n{'='*40}")
        print(f"Config: {name}")
        print(f"  Features: [{B}, {V}, {C}, {H}, {W}]")
        print(f"  Depth candidates: {D}")
        print(f"  Grid size: [{V*B}, {D}, {H*W}, 2]")
        
        # 创建测试数据
        features = torch.randn(V * B, C, H, W, device=device, dtype=dtype)
        grid = torch.rand(V * B, D, H * W, 2, device=device, dtype=dtype) * 2 - 1
        
        # 计算总采样点数
        total_samples = V * B * D * H * W
        print(f"  Total samples: {total_samples:,}")
        
        # Warmup
        for _ in range(10):
            _ = F.grid_sample(features, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
        torch.cuda.synchronize()
        
        # Benchmark grid_sample
        times_grid_sample = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            _ = F.grid_sample(features, grid, mode='bilinear', padding_mode='zeros', align_corners=True)
            torch.cuda.synchronize()
            times_grid_sample.append((time.perf_counter() - start) * 1000)
        
        # Benchmark attention-like aggregation
        # 模拟 cost volume 聚合
        cost_volume = torch.randn(V * B, D, H, W, device=device, dtype=dtype)
        softmax_weights = torch.softmax(cost_volume, dim=1)
        
        times_aggregation = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            # 加权聚合 (depth 维度)
            _ = (cost_volume * softmax_weights).sum(dim=1)
            torch.cuda.synchronize()
            times_aggregation.append((time.perf_counter() - start) * 1000)
        
        # 计算统计
        gs_mean = np.mean(times_grid_sample)
        gs_std = np.std(times_grid_sample)
        agg_mean = np.mean(times_aggregation)
        agg_std = np.std(times_aggregation)
        total_mean = gs_mean + agg_mean
        
        print(f"\nResults:")
        print(f"  Grid Sample:   {gs_mean:.2f} ± {gs_std:.2f} ms")
        print(f"  Aggregation:   {agg_mean:.2f} ± {agg_std:.2f} ms")
        print(f"  Total:         {total_mean:.2f} ms")
        print(f"  Throughput:    {total_samples / (gs_mean / 1000) / 1e6:.2f} M samples/sec")
        
        results[name] = {
            'grid_sample_ms': gs_mean,
            'aggregation_ms': agg_mean,
            'total_ms': total_mean,
            'total_samples': total_samples,
            'throughput': total_samples / (gs_mean / 1000),
        }
    
    return results


def compare_with_geopim(gpu_results: Dict, row_hit_rates=[0.5, 0.7, 0.9]):
    """与 GeoPIM 模拟对比"""
    from geopim.simulator.geopim_simulator import GeoPIMSimulator
    
    print("\n" + "=" * 60)
    print("Comparison with GeoPIM")
    print("=" * 60)
    
    simulator = GeoPIMSimulator()
    
    for name, gpu_result in gpu_results.items():
        print(f"\n{name}:")
        print(f"  GPU baseline: {gpu_result['total_ms']:.2f} ms")
        print(f"  GPU throughput: {gpu_result['throughput']/1e6:.2f} M samples/sec")
        
        total_samples = gpu_result['total_samples']
        
        print(f"\n  GeoPIM estimates (total samples: {total_samples:,}):")
        for hr in row_hit_rates:
            geopim_result = simulator.estimate_performance(
                batch_size=1,
                num_queries=total_samples // 128,
                num_samples=128,
                num_views=1,
                row_hit_rate=hr
            )
            
            geopim_ms = geopim_result['estimated_ms']
            speedup = gpu_result['total_ms'] / geopim_ms
            
            print(f"    {hr:.0%} hit rate: {geopim_ms:.2f} ms ({speedup:.2f}× speedup)")


def main():
    parser = argparse.ArgumentParser(description='TransPlat Profiler')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of iterations')
    parser.add_argument('--device', type=str, default='cuda')
    
    args = parser.parse_args()
    
    # Profile 几何采样操作
    gpu_results = profile_geometry_sampling_operations(
        device=args.device,
        num_iterations=args.iterations
    )
    
    # 与 GeoPIM 对比
    compare_with_geopim(gpu_results)
    
    # 总结
    print("\n" + "=" * 60)
    print("Summary Table")
    print("=" * 60)
    print("\n| Config | GPU (ms) | Samples | GPU Throughput | GeoPIM@70% | Speedup |")
    print("|--------|----------|---------|----------------|------------|---------|")
    
    from geopim.simulator.geopim_simulator import GeoPIMSimulator
    simulator = GeoPIMSimulator()
    
    for name, result in gpu_results.items():
        geopim_result = simulator.estimate_performance(
            batch_size=1,
            num_queries=result['total_samples'] // 128,
            num_samples=128,
            num_views=1,
            row_hit_rate=0.7
        )
        speedup = result['total_ms'] / geopim_result['estimated_ms']
        
        print(f"| {name} | {result['total_ms']:.2f} | {result['total_samples']:,} | "
              f"{result['throughput']/1e6:.1f} M/s | {geopim_result['estimated_ms']:.2f} ms | {speedup:.2f}× |")


if __name__ == "__main__":
    main()

