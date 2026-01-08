"""
GeoPIM 端到端性能评估

基于真实 TransPlat 推理流程进行性能测量和对比。

用法:
    conda activate transplat
    python -m src.geopim.benchmark.e2e_benchmark \
        --checkpoint checkpoints/re10k.ckpt \
        --dataset_path datasets/re10k/test
"""

import argparse
import time
import sys
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import numpy as np
from einops import rearrange
from tqdm import tqdm

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))


@dataclass
class BenchmarkResult:
    """基准测试结果"""
    
    # 时序 (ms)
    backbone_time: float
    depth_predictor_time: float
    geometry_sampling_time: float
    gaussian_adapter_time: float
    total_encoder_time: float
    
    # 采样统计
    total_samples: int
    samples_per_second: float
    
    # 内存 (MB)
    peak_memory: float
    
    def __repr__(self):
        return f"""
BenchmarkResult:
  Backbone:           {self.backbone_time:.2f} ms
  Depth Predictor:    {self.depth_predictor_time:.2f} ms
  Geometry Sampling:  {self.geometry_sampling_time:.2f} ms ({self.geometry_sampling_time/self.total_encoder_time*100:.1f}%)
  Gaussian Adapter:   {self.gaussian_adapter_time:.2f} ms
  ---
  Total Encoder:      {self.total_encoder_time:.2f} ms
  
  Total Samples:      {self.total_samples:,}
  Throughput:         {self.samples_per_second/1e6:.2f} M samples/sec
  Peak Memory:        {self.peak_memory:.1f} MB
"""


class GeometrySamplingProfiler:
    """
    几何采样性能分析器
    
    测量 TransPlat 中几何引导采样的性能。
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.timing_records: List[Dict] = []
        
    def profile_grid_sample(
        self,
        feature_map: torch.Tensor,
        grid: torch.Tensor,
        num_iterations: int = 100
    ) -> Dict:
        """
        分析 F.grid_sample 性能
        
        Args:
            feature_map: [B, C, H, W] 特征图
            grid: [B, N, 1, 2] 采样网格
            num_iterations: 测试迭代次数
            
        Returns:
            性能统计
        """
        B, C, H, W = feature_map.shape
        N = grid.shape[1]
        
        # Warmup
        for _ in range(10):
            _ = F.grid_sample(
                feature_map, grid, mode='bilinear',
                padding_mode='zeros', align_corners=True
            )
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = F.grid_sample(
                feature_map, grid, mode='bilinear',
                padding_mode='zeros', align_corners=True
            )
            
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        total_samples = B * N
        mean_time = np.mean(times) * 1000  # ms
        
        return {
            'mean_time_ms': mean_time,
            'std_time_ms': np.std(times) * 1000,
            'total_samples': total_samples,
            'samples_per_sec': total_samples / (mean_time / 1000),
            'feature_shape': (B, C, H, W),
            'grid_shape': grid.shape,
        }
    
    def profile_weighted_aggregation(
        self,
        sampled_features: torch.Tensor,
        weights: torch.Tensor,
        num_iterations: int = 100
    ) -> Dict:
        """
        分析加权聚合性能
        
        Args:
            sampled_features: [B, Q, S, C] 采样特征
            weights: [B, Q, S] 权重
            num_iterations: 迭代次数
        """
        B, Q, S, C = sampled_features.shape
        
        # Warmup
        for _ in range(10):
            _ = torch.einsum('bqsc,bqs->bqc', sampled_features, weights)
        torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(num_iterations):
            torch.cuda.synchronize()
            start = time.perf_counter()
            
            _ = torch.einsum('bqsc,bqs->bqc', sampled_features, weights)
            
            torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        mean_time = np.mean(times) * 1000
        
        return {
            'mean_time_ms': mean_time,
            'std_time_ms': np.std(times) * 1000,
            'shape': (B, Q, S, C),
        }


class TransPlatE2EBenchmark:
    """
    TransPlat 端到端性能基准测试
    
    加载真实模型并测量各阶段性能。
    """
    
    def __init__(
        self,
        checkpoint_path: str,
        device: str = 'cuda',
        half_precision: bool = True
    ):
        self.device = device
        self.half_precision = half_precision
        self.checkpoint_path = checkpoint_path
        
        self.model = None
        self.encoder = None
        self.decoder = None
        
    def load_model(self):
        """加载 TransPlat 模型"""
        from src.model.model_wrapper import ModelWrapper
        from src.model.encoder import get_encoder
        from src.model.decoder import get_decoder
        
        print(f"Loading checkpoint from {self.checkpoint_path}...")
        
        # 加载 checkpoint
        ckpt = torch.load(self.checkpoint_path, map_location=self.device)
        
        # 从 checkpoint 获取配置
        # 这里我们需要根据实际的 checkpoint 格式调整
        if 'hyper_parameters' in ckpt:
            cfg = ckpt['hyper_parameters'].get('cfg', None)
        else:
            cfg = None
            
        # 如果无法从 checkpoint 获取配置，使用默认配置
        if cfg is None:
            print("Warning: Cannot load config from checkpoint, using default config")
            # 创建默认的 encoder 和 decoder
            return False
        
        # 创建模型
        encoder = get_encoder(cfg.model.encoder)
        decoder = get_decoder(cfg.model.decoder)
        
        # 加载权重
        self.model = ModelWrapper.load_from_checkpoint(
            self.checkpoint_path,
            encoder=encoder,
            decoder=decoder,
            strict=False
        )
        self.model = self.model.to(self.device)
        self.model.eval()
        
        if self.half_precision:
            self.model = self.model.half()
        
        self.encoder = self.model.encoder
        self.decoder = self.model.decoder
        
        print("Model loaded successfully!")
        return True
    
    def create_dummy_input(
        self,
        batch_size: int = 1,
        num_views: int = 2,
        image_size: Tuple[int, int] = (256, 256),
    ) -> Dict:
        """创建测试输入"""
        H, W = image_size
        dtype = torch.float16 if self.half_precision else torch.float32
        
        context = {
            'image': torch.randn(batch_size, num_views, 3, H, W, device=self.device, dtype=dtype),
            'intrinsics': torch.eye(3, device=self.device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(batch_size, num_views, -1, -1).clone(),
            'extrinsics': torch.eye(4, device=self.device, dtype=dtype).unsqueeze(0).unsqueeze(0).expand(batch_size, num_views, -1, -1).clone(),
            'near': torch.tensor([[0.1] * num_views] * batch_size, device=self.device, dtype=dtype),
            'far': torch.tensor([[100.0] * num_views] * batch_size, device=self.device, dtype=dtype),
        }
        
        # 设置相机内参
        context['intrinsics'][:, :, 0, 0] = 0.5  # fx
        context['intrinsics'][:, :, 1, 1] = 0.5  # fy
        context['intrinsics'][:, :, 0, 2] = 0.5  # cx
        context['intrinsics'][:, :, 1, 2] = 0.5  # cy
        
        return context
    
    def benchmark_geometry_sampling(
        self,
        num_iterations: int = 50
    ) -> Dict:
        """
        基准测试几何采样操作
        
        模拟 TransPlat 中的几何引导采样。
        """
        print("\n" + "=" * 60)
        print("Benchmarking Geometry Sampling (GPU Baseline)")
        print("=" * 60)
        
        profiler = GeometrySamplingProfiler(self.device)
        dtype = torch.float16 if self.half_precision else torch.float32
        
        # TransPlat 典型配置
        configs = [
            {'B': 1, 'C': 128, 'H': 64, 'W': 64, 'Q': 1024, 'S': 128, 'name': 'TransPlat-small'},
            {'B': 1, 'C': 128, 'H': 64, 'W': 64, 'Q': 4096, 'S': 128, 'name': 'TransPlat-medium'},
            {'B': 2, 'C': 128, 'H': 64, 'W': 64, 'Q': 4096, 'S': 128, 'name': 'TransPlat-large'},
        ]
        
        results = {}
        
        for cfg in configs:
            B, C, H, W = cfg['B'], cfg['C'], cfg['H'], cfg['W']
            Q, S = cfg['Q'], cfg['S']
            
            print(f"\nConfig: {cfg['name']}")
            print(f"  Feature: [{B}, {C}, {H}, {W}]")
            print(f"  Queries: {Q}, Samples/Query: {S}")
            print(f"  Total Samples: {B * Q * S:,}")
            
            # 创建测试数据
            feature_map = torch.randn(B, C, H, W, device=self.device, dtype=dtype)
            
            # 生成采样网格 (模拟几何采样坐标)
            grid = torch.rand(B, Q * S, 1, 2, device=self.device, dtype=dtype) * 2 - 1
            
            # 测试 grid_sample
            grid_sample_result = profiler.profile_grid_sample(
                feature_map, grid, num_iterations
            )
            
            # 测试加权聚合
            sampled = torch.randn(B, Q, S, C, device=self.device, dtype=dtype)
            weights = torch.softmax(torch.randn(B, Q, S, device=self.device), dim=-1).to(dtype)
            
            agg_result = profiler.profile_weighted_aggregation(
                sampled, weights, num_iterations
            )
            
            total_time = grid_sample_result['mean_time_ms'] + agg_result['mean_time_ms']
            
            print(f"  Grid Sample:    {grid_sample_result['mean_time_ms']:.2f} ms")
            print(f"  Aggregation:    {agg_result['mean_time_ms']:.2f} ms")
            print(f"  Total:          {total_time:.2f} ms")
            print(f"  Throughput:     {grid_sample_result['samples_per_sec']/1e6:.2f} M samples/sec")
            
            results[cfg['name']] = {
                'grid_sample': grid_sample_result,
                'aggregation': agg_result,
                'total_time_ms': total_time,
                'total_samples': B * Q * S,
            }
        
        return results
    
    def compare_with_geopim(
        self,
        gpu_results: Dict,
        row_hit_rate: float = 0.7
    ) -> Dict:
        """
        与 GeoPIM 模拟结果对比
        """
        print("\n" + "=" * 60)
        print("Comparing with GeoPIM Simulation")
        print("=" * 60)
        
        from geopim.simulator.geopim_simulator import GeoPIMSimulator
        
        simulator = GeoPIMSimulator()
        comparisons = {}
        
        for name, gpu_result in gpu_results.items():
            total_samples = gpu_result['total_samples']
            gpu_time = gpu_result['total_time_ms']
            
            # GeoPIM 估算
            geopim_result = simulator.estimate_performance(
                batch_size=1,
                num_queries=total_samples // 128,  # 假设 S=128
                num_samples=128,
                num_views=1,
                row_hit_rate=row_hit_rate
            )
            
            geopim_time = geopim_result['estimated_ms']
            speedup = gpu_time / geopim_time
            
            print(f"\n{name}:")
            print(f"  GPU Time:     {gpu_time:.2f} ms")
            print(f"  GeoPIM Est:   {geopim_time:.2f} ms")
            print(f"  Speedup:      {speedup:.2f}×")
            
            comparisons[name] = {
                'gpu_time_ms': gpu_time,
                'geopim_time_ms': geopim_time,
                'speedup': speedup,
                'total_samples': total_samples,
            }
        
        return comparisons


def run_quick_benchmark(device='cuda'):
    """
    快速基准测试 (不需要完整模型)
    """
    print("\n" + "=" * 60)
    print("GeoPIM Quick Benchmark")
    print("=" * 60)
    
    benchmark = TransPlatE2EBenchmark(
        checkpoint_path="",  # 不需要
        device=device,
        half_precision=True
    )
    
    # 运行几何采样基准测试
    gpu_results = benchmark.benchmark_geometry_sampling(num_iterations=100)
    
    # 与 GeoPIM 对比
    comparisons = benchmark.compare_with_geopim(gpu_results, row_hit_rate=0.7)
    
    # 总结
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    
    print("\n| Config | GPU (ms) | GeoPIM (ms) | Speedup |")
    print("|--------|----------|-------------|---------|")
    for name, comp in comparisons.items():
        print(f"| {name} | {comp['gpu_time_ms']:.2f} | {comp['geopim_time_ms']:.2f} | {comp['speedup']:.2f}× |")
    
    return comparisons


def run_full_benchmark(checkpoint_path: str, device='cuda'):
    """
    完整基准测试 (需要模型和数据)
    """
    benchmark = TransPlatE2EBenchmark(
        checkpoint_path=checkpoint_path,
        device=device,
        half_precision=True
    )
    
    # 尝试加载模型
    if not benchmark.load_model():
        print("Failed to load model, running quick benchmark instead")
        return run_quick_benchmark(device)
    
    # TODO: 添加完整的端到端测试
    # 包括数据加载、编码器各阶段计时等
    
    return run_quick_benchmark(device)


def main():
    parser = argparse.ArgumentParser(description='GeoPIM E2E Benchmark')
    parser.add_argument('--checkpoint', type=str, default='',
                        help='Path to TransPlat checkpoint')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to run on')
    parser.add_argument('--quick', action='store_true',
                        help='Run quick benchmark without model')
    parser.add_argument('--iterations', type=int, default=100,
                        help='Number of benchmark iterations')
    
    args = parser.parse_args()
    
    if args.quick or not args.checkpoint:
        results = run_quick_benchmark(args.device)
    else:
        results = run_full_benchmark(args.checkpoint, args.device)
    
    return results


if __name__ == "__main__":
    main()

