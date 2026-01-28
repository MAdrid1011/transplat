#!/usr/bin/env python3
"""
Analyze HBM traffic from nsys/ncu profile reports.
Uses NVTX markers and CUPTI events for memcpy, and ncu for kernel DRAM traffic.
"""

import sqlite3
import argparse
import subprocess
import sys
import re
import csv
import json
from pathlib import Path
from collections import defaultdict

# Path to NVIDIA ncu
NCU_PATH = "/usr/local/cuda/bin/ncu"


def run_ncu_profile(python_cmd: str, output_csv: str = "ncu_metrics.csv"):
    """
    Run ncu to profile DRAM traffic for all CUDA kernels.
    Returns path to CSV output file.
    """
    metrics = [
        "dram__bytes_read.sum",
        "dram__bytes_write.sum",
        "lts__t_bytes.sum",  # L2 cache bytes
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    ]
    
    ncu_cmd = [
        NCU_PATH,
        "--metrics", ",".join(metrics),
        "--csv",
        "--log-file", output_csv,
        "--nvtx",
        "--nvtx-include", "encoder*,decoder*",
        "python"
    ] + python_cmd.split()[2:]  # Skip "python -m"
    
    print(f"Running ncu profiler...")
    print(f"Command: {' '.join(ncu_cmd[:8])}...")
    
    result = subprocess.run(ncu_cmd, capture_output=True, text=True)
    
    if result.returncode != 0:
        print(f"ncu error: {result.stderr[:500]}")
        return None
    
    return output_csv


def parse_ncu_csv(csv_path: str):
    """Parse ncu CSV output to extract per-kernel DRAM traffic."""
    if not Path(csv_path).exists():
        return {}
    
    kernel_traffic = defaultdict(lambda: {
        'dram_read_bytes': 0,
        'dram_write_bytes': 0,
        'l2_bytes': 0,
        'kernel_count': 0,
        'nvtx_range': ''
    })
    
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    # Find the header row (starts with "ID")
    header_idx = -1
    for i, line in enumerate(lines):
        if line.startswith('"ID"') or line.startswith('ID,'):
            header_idx = i
            break
    
    if header_idx < 0:
        print(f"Could not find header in {csv_path}")
        return kernel_traffic
    
    # Parse CSV from header
    reader = csv.DictReader(lines[header_idx:])
    
    # Group by kernel ID to aggregate metrics
    kernel_data = defaultdict(lambda: {'name': '', 'read': 0, 'write': 0})
    
    for row in reader:
        kernel_id = row.get('ID', '')
        kernel_name = row.get('Kernel Name', '')
        metric_name = row.get('Metric Name', '')
        metric_value = row.get('Metric Value', '0')
        
        # Parse metric value (may have commas)
        try:
            value = float(metric_value.replace(',', '')) if metric_value else 0
        except:
            value = 0
        
        kernel_data[kernel_id]['name'] = kernel_name
        
        if 'dram__bytes_read' in metric_name:
            kernel_data[kernel_id]['read'] = value
        elif 'dram__bytes_write' in metric_name:
            kernel_data[kernel_id]['write'] = value
    
    # Aggregate by simplified kernel name
    for kid, data in kernel_data.items():
        kernel_name = data['name']
        
        # Simplify kernel name for grouping
        if 'gemm' in kernel_name.lower() or 'cutlass' in kernel_name.lower():
            key = 'GEMM/MatMul'
        elif 'conv' in kernel_name.lower():
            key = 'Convolution'
        elif 'elementwise' in kernel_name.lower():
            key = 'Elementwise'
        elif 'reduce' in kernel_name.lower():
            key = 'Reduction'
        elif 'softmax' in kernel_name.lower():
            key = 'Softmax'
        elif 'attention' in kernel_name.lower() or 'flash' in kernel_name.lower():
            key = 'Attention'
        elif 'batch_norm' in kernel_name.lower() or 'layer_norm' in kernel_name.lower():
            key = 'Normalization'
        elif 'grid_sampler' in kernel_name.lower() or 'warp' in kernel_name.lower():
            key = 'GridSample/Warp'
        elif 'distribution' in kernel_name.lower() or 'random' in kernel_name.lower():
            key = 'Random/Init'
        elif 'index' in kernel_name.lower() or 'gather' in kernel_name.lower() or 'scatter' in kernel_name.lower():
            key = 'Index/Gather/Scatter'
        else:
            # Use first 40 chars of kernel name
            key = kernel_name[:40] + '...' if len(kernel_name) > 40 else kernel_name
        
        kernel_traffic[key]['dram_read_bytes'] += data['read']
        kernel_traffic[key]['dram_write_bytes'] += data['write']
        kernel_traffic[key]['kernel_count'] += 1
    
    return dict(kernel_traffic)


def run_nsys_profile(cmd: str, output_name: str = "hbm_profile"):
    """Run nsys profile on a command."""
    nsys_cmd = [
        "nsys", "profile",
        "--stats=true",
        "--force-overwrite=true",
        "-o", output_name,
    ] + cmd.split()
    
    print(f"Running: {' '.join(nsys_cmd)}")
    result = subprocess.run(nsys_cmd, capture_output=False)
    return result.returncode == 0


def export_sqlite(nsys_rep: str):
    """Export nsys report to SQLite."""
    sqlite_path = nsys_rep.replace('.nsys-rep', '.sqlite')
    cmd = ["nsys", "stats", "--force-export=true", "--report", "nvtx_sum", nsys_rep]
    subprocess.run(cmd, capture_output=True)
    return sqlite_path


def analyze_memcpy_traffic(sqlite_path: str):
    """Analyze memcpy HBM traffic from SQLite database."""
    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    
    # Query per-stage memcpy traffic
    query = """
    WITH nvtx_ranges AS (
        SELECT 
            text as stage,
            start as nvtx_start,
            end as nvtx_end,
            (end - start) / 1e9 as duration_s
        FROM NVTX_EVENTS 
        WHERE eventType = 59
    )
    SELECT 
        n.stage,
        n.duration_s,
        e.label as copy_type,
        COALESCE(SUM(m.bytes), 0) as total_bytes,
        COUNT(m.bytes) as num_ops
    FROM nvtx_ranges n
    LEFT JOIN CUPTI_ACTIVITY_KIND_MEMCPY m 
        ON m.start >= n.nvtx_start AND m.end <= n.nvtx_end
    LEFT JOIN ENUM_CUDA_MEMCPY_OPER e ON m.copyKind = e.id
    GROUP BY n.stage, e.label
    ORDER BY n.nvtx_start, total_bytes DESC
    """
    
    cursor.execute(query)
    rows = cursor.fetchall()
    
    # Aggregate by stage
    stage_traffic = {}
    for stage, duration, copy_type, total_bytes, num_ops in rows:
        if stage not in stage_traffic:
            stage_traffic[stage] = {
                'duration_s': duration,
                'h2d_bytes': 0,
                'd2h_bytes': 0,
                'd2d_bytes': 0,
                'total_ops': 0
            }
        
        if copy_type == 'Host-to-Device':
            stage_traffic[stage]['h2d_bytes'] = total_bytes
        elif copy_type == 'Device-to-Host':
            stage_traffic[stage]['d2h_bytes'] = total_bytes
        elif copy_type == 'Device-to-Device':
            stage_traffic[stage]['d2d_bytes'] = total_bytes
        
        stage_traffic[stage]['total_ops'] += num_ops
    
    conn.close()
    return stage_traffic


def print_memcpy_report(traffic_data: dict):
    """Print formatted memcpy traffic report."""
    print("\n" + "=" * 100)
    print("HBM TRAFFIC ANALYSIS - Explicit Memcpy Operations (nsys)")
    print("=" * 100)
    print(f"\n{'Stage':<45} {'Time(s)':>8} {'H2D(MB)':>10} {'D2H(MB)':>10} {'D2D(MB)':>10} {'Total(MB)':>12}")
    print("-" * 100)
    
    total_h2d = 0
    total_d2h = 0
    total_d2d = 0
    
    for stage, data in traffic_data.items():
        h2d_mb = data['h2d_bytes'] / 1e6
        d2h_mb = data['d2h_bytes'] / 1e6
        d2d_mb = data['d2d_bytes'] / 1e6
        total_mb = h2d_mb + d2h_mb + d2d_mb
        
        total_h2d += h2d_mb
        total_d2h += d2h_mb
        total_d2d += d2d_mb
        
        print(f"{stage:<45} {data['duration_s']:>8.3f} {h2d_mb:>10.2f} {d2h_mb:>10.2f} {d2d_mb:>10.2f} {total_mb:>12.2f}")
    
    print("-" * 100)
    total_all = total_h2d + total_d2h + total_d2d
    print(f"{'TOTAL':<45} {'':<8} {total_h2d:>10.2f} {total_d2h:>10.2f} {total_d2d:>10.2f} {total_all:>12.2f}")
    print("=" * 100 + "\n")


def print_ncu_report(kernel_traffic: dict):
    """Print formatted ncu DRAM traffic report."""
    print("\n" + "=" * 110)
    print("HBM TRAFFIC ANALYSIS - Kernel DRAM Access (ncu)")
    print("=" * 110)
    print(f"\n{'Stage/Kernel':<50} {'Read(MB)':>12} {'Write(MB)':>12} {'Total(MB)':>12} {'Kernels':>10} {'% of Total':>12}")
    print("-" * 110)
    
    total_read = 0
    total_write = 0
    
    # Sort by total traffic
    sorted_traffic = sorted(
        kernel_traffic.items(),
        key=lambda x: x[1]['dram_read_bytes'] + x[1]['dram_write_bytes'],
        reverse=True
    )
    
    for stage, data in sorted_traffic:
        read_mb = data['dram_read_bytes'] / 1e6
        write_mb = data['dram_write_bytes'] / 1e6
        total_mb = read_mb + write_mb
        
        total_read += read_mb
        total_write += write_mb
    
    grand_total = total_read + total_write
    
    for stage, data in sorted_traffic:
        read_mb = data['dram_read_bytes'] / 1e6
        write_mb = data['dram_write_bytes'] / 1e6
        total_mb = read_mb + write_mb
        pct = (total_mb / grand_total * 100) if grand_total > 0 else 0
        
        # Truncate long stage names
        stage_display = stage[:48] + '..' if len(stage) > 50 else stage
        
        print(f"{stage_display:<50} {read_mb:>12.2f} {write_mb:>12.2f} {total_mb:>12.2f} {data['kernel_count']:>10} {pct:>11.1f}%")
    
    print("-" * 110)
    print(f"{'TOTAL':<50} {total_read:>12.2f} {total_write:>12.2f} {grand_total:>12.2f}")
    print("=" * 110 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Analyze HBM traffic from nsys/ncu profile")
    parser.add_argument("input_file", nargs='?', help="Path to nsys SQLite/.nsys-rep or ncu CSV file")
    parser.add_argument("--ncu-csv", help="Path to ncu CSV output file")
    parser.add_argument("--run-ncu", metavar="CMD", help="Run ncu profiler on command")
    args = parser.parse_args()
    
    # If running ncu profiler
    if args.run_ncu:
        csv_path = run_ncu_profile(args.run_ncu)
        if csv_path:
            kernel_traffic = parse_ncu_csv(csv_path)
            print_ncu_report(kernel_traffic)
        return
    
    # If analyzing ncu CSV
    if args.ncu_csv:
        kernel_traffic = parse_ncu_csv(args.ncu_csv)
        print_ncu_report(kernel_traffic)
        return
    
    # Analyze nsys SQLite
    if args.input_file:
        sqlite_path = args.input_file
        if sqlite_path.endswith('.nsys-rep'):
            print("Exporting to SQLite...")
            sqlite_path = export_sqlite(args.input_file)
        
        if not Path(sqlite_path).exists():
            print(f"Error: {sqlite_path} not found")
            sys.exit(1)
        
        print(f"Analyzing: {sqlite_path}")
        traffic_data = analyze_memcpy_traffic(sqlite_path)
        print_memcpy_report(traffic_data)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
