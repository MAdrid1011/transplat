"""
TransPlat GeoPIM Benchmark

测试配置:
- TransPlat: B=2, Q=1024 (32×32), C=128, P=4, S=512 → 4M samples
- PixelSplat: B=2, Q=4096 (64×64), S=32 → 0.5M samples
"""

from dataclasses import dataclass
from typing import Dict, Optional
import numpy as np
import time

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from ..simulator.geopim_simulator import GeoPIMSimulator


@dataclass
class BenchmarkConfig:
    """Benchmark 配置"""
    
    # 工作负载参数
    batch_size: int = 2
    num_queries: int = 1024  # 32x32
    channels: int = 128
    samples_per_query: int = 512
    feature_height: int = 32
    feature_width: int = 32
    
    # 运行参数
    num_runs: int = 10
    warmup_runs: int = 3
    
    # GPU baseline 参考值
    gpu_baseline_ms: float = 19.65
    
    @property
    def total_samples(self) -> int:
        """总采样点数"""
        return self.batch_size * self.num_queries * self.samples_per_query


class TransPlatBenchmark:
    """
    TransPlat GeoPIM Benchmark
    
    对比 GPU baseline 和 GeoPIM 模拟性能。
    """
    
    def __init__(self, config: Optional[BenchmarkConfig] = None):
        self.config = config or BenchmarkConfig()
        self.simulator = GeoPIMSimulator()
        self._results: Dict = {}
    
    def generate_inputs(self, device: str = 'cpu'):
        """生成测试输入"""
        cfg = self.config
        
        if HAS_TORCH:
            feature_map = torch.randn(
                cfg.batch_size, cfg.channels, 
                cfg.feature_height, cfg.feature_width,
                dtype=torch.float16, device=device
            )
            geo_params = {
                'ref_xy': torch.rand(cfg.batch_size, cfg.num_queries, 2, 
                                     dtype=torch.float16, device=device) * 30 + 1,
                'stride_xy': torch.rand(cfg.batch_size, cfg.num_queries, 2,
                                        dtype=torch.float16, device=device) * 0.5,
            }
            weights = torch.softmax(
                torch.randn(cfg.batch_size, cfg.num_queries, cfg.samples_per_query, device=device),
                dim=-1
            ).half()
            offsets = torch.randn(
                cfg.batch_size, cfg.num_queries, cfg.samples_per_query, 2,
                dtype=torch.float16, device=device
            ) * 0.5
        else:
            feature_map = np.random.randn(
                cfg.batch_size, cfg.channels,
                cfg.feature_height, cfg.feature_width
            ).astype(np.float16)
            geo_params = {
                'ref_xy': np.random.rand(cfg.batch_size, cfg.num_queries, 2).astype(np.float16) * 30 + 1,
                'stride_xy': np.random.rand(cfg.batch_size, cfg.num_queries, 2).astype(np.float16) * 0.5,
            }
            weights_raw = np.random.randn(cfg.batch_size, cfg.num_queries, cfg.samples_per_query)
            weights = (np.exp(weights_raw) / np.exp(weights_raw).sum(axis=-1, keepdims=True)).astype(np.float16)
            offsets = (np.random.randn(cfg.batch_size, cfg.num_queries, cfg.samples_per_query, 2) * 0.5).astype(np.float16)
        
        return feature_map, geo_params, weights, offsets
    
    def run_geopim_benchmark(self) -> Dict:
        """运行 GeoPIM benchmark"""
        feature_map, geo_params, weights, offsets = self.generate_inputs()
        
        times = []
        stats_list = []
        
        # Warmup
        for _ in range(self.config.warmup_runs):
            _, _ = self.simulator.execute(feature_map, geo_params, weights, offsets)
        
        # Benchmark
        for _ in range(self.config.num_runs):
            start = time.perf_counter()
            output, stats = self.simulator.execute(feature_map, geo_params, weights, offsets)
            elapsed = time.perf_counter() - start
            times.append(elapsed)
            stats_list.append(stats)
        
        # 聚合结果
        avg_stats = {
            'total_cycles': np.mean([s['total_cycles'] for s in stats_list]),
            'row_hit_rate': np.mean([s['row_hit_rate'] for s in stats_list]),
            'estimated_ms': np.mean([s['estimated_ms'] for s in stats_list]),
            'speedup': np.mean([s['speedup'] for s in stats_list]),
        }
        
        return {
            'simulation_time_ms': np.mean(times) * 1000,
            'simulation_time_std': np.std(times) * 1000,
            **avg_stats,
        }
    
    def run_gpu_baseline(self) -> Dict:
        """运行 GPU baseline (如果有 PyTorch)"""
        if not HAS_TORCH:
            return {
                'mean_ms': self.config.gpu_baseline_ms,
                'std_ms': 0.0,
                'note': 'Using reference value (PyTorch not available)'
            }
        
        import torch.nn.functional as F
        
        feature_map, geo_params, weights, offsets = self.generate_inputs('cuda' if torch.cuda.is_available() else 'cpu')
        
        times = []
        
        # 实现简化的 GPU 采样
        def gpu_sampling():
            B, Q, S = weights.shape
            C = feature_map.shape[1]
            H, W = feature_map.shape[2:]
            
            # 生成采样坐标
            ref_xy = geo_params['ref_xy']  # [B, Q, 2]
            stride_xy = geo_params['stride_xy']  # [B, Q, 2]
            
            # 扩展为所有采样点
            sample_idx = torch.arange(S, device=weights.device).float()  # [S]
            coords = ref_xy.unsqueeze(2) + sample_idx.view(1, 1, S, 1) * stride_xy.unsqueeze(2)
            coords = coords + offsets  # [B, Q, S, 2]
            
            # 归一化
            coords_norm = coords.clone()
            coords_norm[..., 0] = 2 * coords[..., 0] / (W - 1) - 1
            coords_norm[..., 1] = 2 * coords[..., 1] / (H - 1) - 1
            
            # Grid sample
            coords_flat = coords_norm.view(B, Q * S, 1, 2)
            sampled = F.grid_sample(feature_map.float(), coords_flat.float(), 
                                   mode='bilinear', padding_mode='zeros', align_corners=True)
            sampled = sampled.view(B, C, Q, S).permute(0, 2, 3, 1)  # [B, Q, S, C]
            
            # 加权聚合
            output = torch.einsum('bqs,bqsc->bqc', weights.float(), sampled)
            return output
        
        # Warmup
        for _ in range(self.config.warmup_runs):
            _ = gpu_sampling()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        
        # Benchmark
        for _ in range(self.config.num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start = time.perf_counter()
            _ = gpu_sampling()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            times.append(time.perf_counter() - start)
        
        return {
            'mean_ms': np.mean(times) * 1000,
            'std_ms': np.std(times) * 1000,
        }
    
    def compare(self) -> Dict:
        """对比 GPU 和 GeoPIM"""
        gpu_results = self.run_gpu_baseline()
        geopim_results = self.run_geopim_benchmark()
        
        actual_speedup = gpu_results['mean_ms'] / geopim_results['estimated_ms']
        
        self._results = {
            'gpu_baseline': gpu_results,
            'geopim': geopim_results,
            'actual_speedup': actual_speedup,
            'config': {
                'batch_size': self.config.batch_size,
                'num_queries': self.config.num_queries,
                'samples_per_query': self.config.samples_per_query,
                'total_samples': self.config.total_samples,
            }
        }
        
        return self._results
    
    def generate_report(self) -> str:
        """生成性能报告"""
        if not self._results:
            self.compare()
        
        r = self._results
        
        return f"""
# GeoPIM v3.0 性能评估报告

## 配置
- Batch 大小: {r['config']['batch_size']}
- Query 数量: {r['config']['num_queries']} (32×32)
- 采样点/Query: {r['config']['samples_per_query']}
- 总采样点数: {r['config']['total_samples']:,}

## GPU Baseline
- 平均时间: {r['gpu_baseline']['mean_ms']:.2f} ms
- 标准差: {r['gpu_baseline']['std_ms']:.2f} ms

## GeoPIM 模拟
- 预估时间: {r['geopim']['estimated_ms']:.2f} ms
- Row Hit Rate: {r['geopim']['row_hit_rate']:.1%}
- 总周期数: {r['geopim']['total_cycles']:,.0f}

## 加速比
- **{r['actual_speedup']:.2f}×** 加速

## 结论
{'✓ 达到目标加速比 (4-8×)' if 4 <= r['actual_speedup'] <= 8 else '⚠ 未达到目标加速比'}
"""


class PixelSplatBenchmark(TransPlatBenchmark):
    """PixelSplat 配置的 Benchmark"""
    
    def __init__(self):
        config = BenchmarkConfig(
            batch_size=2,
            num_queries=4096,  # 64x64
            channels=128,
            samples_per_query=32,
            feature_height=64,
            feature_width=64,
            gpu_baseline_ms=5.0,  # PixelSplat 估计值
        )
        super().__init__(config)

