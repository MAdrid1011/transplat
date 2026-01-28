import json
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from time import time

import numpy as np
import torch
from torch.profiler import profile, ProfilerActivity

# Try to import NVTX for marking regions in nsys profiler
try:
    import torch.cuda.nvtx as nvtx
    HAS_NVTX = True
except ImportError:
    HAS_NVTX = False


class Benchmarker:
    def __init__(self):
        self.execution_times = defaultdict(list)
        self.memory_stats = defaultdict(list)  # Memory allocation statistics
        self.hbm_traffic_stats = defaultdict(list)  # Real HBM traffic statistics
        self.profile_memory = False  # Flag to enable memory allocation profiling
        self.profile_hbm_traffic = False  # Flag to enable real HBM traffic profiling
        self._profiler_depth = 0  # Track nesting depth to avoid nested profilers

    def enable_memory_profiling(self, enable: bool = True):
        """Enable or disable memory allocation profiling."""
        self.profile_memory = enable
        if enable:
            # Reset CUDA memory stats
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()

    def enable_hbm_traffic_profiling(self, enable: bool = True):
        """Enable or disable real HBM traffic profiling using PyTorch profiler."""
        self.profile_hbm_traffic = enable
        if enable:
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            self._profiler_depth = 0

    @contextmanager
    def time(self, tag: str, num_calls: int = 1):
        profiler_ctx = None
        mem_start_allocated = 0
        mem_start_reserved = 0
        should_profile_hbm = False
        nvtx_range = None
        
        try:
            start_time = time()
            
            # Add NVTX marker for nsys profiling
            if HAS_NVTX:
                nvtx_range = nvtx.range_push(tag)
            
            # Memory allocation profiling: record start state
            if self.profile_memory:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                mem_start_allocated = torch.cuda.memory_allocated()
                mem_start_reserved = torch.cuda.memory_reserved()
            
            # HBM Traffic profiling using PyTorch profiler
            # Only profile if no other profiler is active (avoid nesting)
            if self.profile_hbm_traffic and self._profiler_depth == 0:
                should_profile_hbm = True
                self._profiler_depth += 1
                torch.cuda.synchronize()
                profiler_ctx = profile(
                    activities=[ProfilerActivity.CUDA],
                    profile_memory=True,
                    record_shapes=False,
                    with_stack=False,
                )
                profiler_ctx.__enter__()
            elif self.profile_hbm_traffic:
                # Track depth for nested calls
                self._profiler_depth += 1
            
            yield
        finally:
            end_time = time()
            
            # HBM Traffic profiling: collect results
            if should_profile_hbm and profiler_ctx is not None:
                torch.cuda.synchronize()
                profiler_ctx.__exit__(None, None, None)
                self._profiler_depth -= 1
                
                # Extract memory traffic from profiler events
                cuda_mem_self = 0
                total_cuda_time = 0
                kernel_count = 0
                
                try:
                    for event in profiler_ctx.key_averages():
                        if event.device_type == torch.autograd.DeviceType.CUDA:
                            # self_cuda_memory_usage gives memory allocated by this operation
                            if hasattr(event, 'cuda_memory_usage') and event.cuda_memory_usage:
                                cuda_mem_self += abs(event.cuda_memory_usage)
                            if hasattr(event, 'self_cuda_memory_usage') and event.self_cuda_memory_usage:
                                cuda_mem_self += abs(event.self_cuda_memory_usage)
                            if event.cuda_time_total > 0:
                                total_cuda_time += event.cuda_time_total
                                kernel_count += 1
                except Exception as e:
                    print(f"Warning: Could not extract profiler stats for {tag}: {e}")
                
                self.hbm_traffic_stats[tag].append({
                    'cuda_memory_usage_bytes': cuda_mem_self,
                    'cuda_time_us': total_cuda_time,
                    'kernel_count': kernel_count,
                })
            elif self.profile_hbm_traffic and not should_profile_hbm:
                # Decrement depth for nested calls
                self._profiler_depth -= 1
            
            # Memory allocation profiling: record end state
            if self.profile_memory:
                torch.cuda.synchronize()
                mem_end_allocated = torch.cuda.memory_allocated()
                mem_end_reserved = torch.cuda.memory_reserved()
                mem_peak = torch.cuda.max_memory_allocated()
                
                # Calculate memory metrics
                mem_delta_allocated = mem_end_allocated - mem_start_allocated
                mem_peak_during = mem_peak - mem_start_allocated
                
                # Store memory stats
                self.memory_stats[tag].append({
                    'allocated_delta_bytes': mem_delta_allocated,
                    'peak_bytes': mem_peak_during,
                    'reserved_delta_bytes': mem_end_reserved - mem_start_reserved,
                    'total_allocated_end': mem_end_allocated,
                })
            
            for _ in range(num_calls):
                self.execution_times[tag].append((end_time - start_time) / num_calls)
            
            # Pop NVTX marker
            if HAS_NVTX:
                nvtx.range_pop()

    def get_memory_summary(self):
        """Get summary of memory allocation statistics for all tags."""
        summary = {}
        for tag, stats_list in self.memory_stats.items():
            if stats_list:
                avg_peak = np.mean([s['peak_bytes'] for s in stats_list])
                avg_delta = np.mean([s['allocated_delta_bytes'] for s in stats_list])
                summary[tag] = {
                    'avg_peak_mb': avg_peak / 1e6,
                    'avg_delta_mb': avg_delta / 1e6,
                    'calls': len(stats_list),
                }
        return summary

    def get_hbm_traffic_summary(self):
        """Get summary of HBM traffic statistics for all tags."""
        summary = {}
        for tag, stats_list in self.hbm_traffic_stats.items():
            if stats_list:
                avg_mem = np.mean([s['cuda_memory_usage_bytes'] for s in stats_list])
                avg_time = np.mean([s['cuda_time_us'] for s in stats_list])
                avg_kernels = np.mean([s['kernel_count'] for s in stats_list])
                summary[tag] = {
                    'avg_cuda_mem_mb': avg_mem / 1e6,
                    'avg_cuda_time_ms': avg_time / 1000,
                    'avg_kernel_count': avg_kernels,
                    'calls': len(stats_list),
                }
        return summary

    def dump(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(dict(self.execution_times), f)

    def dump_memory(self, path: Path) -> None:
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump(torch.cuda.memory_stats()["allocated_bytes.all.peak"], f)

    def dump_memory_stats(self, path: Path) -> None:
        """Dump detailed memory allocation statistics to file."""
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump({
                'per_stage': dict(self.memory_stats),
                'summary': self.get_memory_summary()
            }, f, indent=2)

    def dump_hbm_traffic_stats(self, path: Path) -> None:
        """Dump detailed HBM traffic statistics to file."""
        path.parent.mkdir(exist_ok=True, parents=True)
        with path.open("w") as f:
            json.dump({
                'per_stage': dict(self.hbm_traffic_stats),
                'summary': self.get_hbm_traffic_summary()
            }, f, indent=2)

    def summarize(self) -> None:
        for tag, times in self.execution_times.items():
            print(f"{tag}: {len(times)} calls, avg. {np.mean(times)} seconds per call")

    def clear_history(self) -> None:
        self.execution_times = defaultdict(list)
        self.memory_stats = defaultdict(list)
        self.hbm_traffic_stats = defaultdict(list)
