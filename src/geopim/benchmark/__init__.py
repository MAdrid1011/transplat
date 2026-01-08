"""
GeoPIM Benchmark 模块

提供端到端性能测试和分析工具。

用法:
    # 快速基准测试 (不需要模型)
    python -m src.geopim.benchmark.e2e_benchmark --quick
    
    # 真实配置测试
    python -m src.geopim.benchmark.transplat_profiler
    
    # 详细 profiling (需要 checkpoint)
    python -m src.geopim.benchmark.detailed_profiler --checkpoint checkpoints/re10k.ckpt
    
    # 完整端到端测试
    python -m src.geopim.benchmark.full_e2e_test --checkpoint checkpoints/re10k.ckpt
"""

from .e2e_benchmark import (
    TransPlatE2EBenchmark,
    GeometrySamplingProfiler,
    run_quick_benchmark,
)

from .transplat_profiler import (
    profile_geometry_sampling_operations,
    compare_with_geopim,
)

from .realistic_benchmark import (
    measure_memory_overhead,
    benchmark_batch_inference,
    analyze_geopim_advantages,
)

__all__ = [
    'TransPlatE2EBenchmark',
    'GeometrySamplingProfiler',
    'run_quick_benchmark',
    'profile_geometry_sampling_operations',
    'compare_with_geopim',
    'measure_memory_overhead',
    'benchmark_batch_inference',
    'analyze_geopim_advantages',
]
