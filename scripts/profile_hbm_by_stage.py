#!/usr/bin/env python3
"""
Profile HBM DRAM traffic by inference stage using ncu.
Supports multiple models: transplat, mvsplat, depthsplat, pixelsplat.
"""

import subprocess
import sys
import csv
import json
import re
from pathlib import Path
from collections import defaultdict

NCU_PATH = "/usr/local/cuda/bin/ncu"

# Define inference stages for each model
MODEL_STAGES = {
    'transplat': {
        'order': [
            'encoder_1_prep_intrinsics',
            'encoder_2_backbone', 
            'encoder_3_depth_anything',
            'encoder_4_depth_predictor',
            'encoder_5_gaussian_adapter',
            'decoder',
        ],
        'parent_map': {
            # Map sub-stages to parent
            'encoder_4a_prep_features': 'encoder_4_depth_predictor',
            'encoder_4b_cost_volume_matching': 'encoder_4_depth_predictor',
            'encoder_4c_cost_volume_unet': 'encoder_4_depth_predictor',
            'encoder_4d_coarse_depth': 'encoder_4_depth_predictor',
            'encoder_4e_depth_refine_unet': 'encoder_4_depth_predictor',
            'encoder_4f_gaussian_head': 'encoder_4_depth_predictor',
        },
        'skip': ['encoder'],  # Skip parent container
    },
    'mvsplat': {
        'order': [
            'encoder_1_backbone',
            'encoder_2_depth_predictor',
            'encoder_3_gaussian_adapter',
            'decoder',
        ],
        'parent_map': {
            # Sub-stages of depth_predictor
            'encoder_2a_prep_features': 'encoder_2_depth_predictor',
            'encoder_2b_cost_volume': 'encoder_2_depth_predictor',
            'encoder_2c_cost_volume_unet': 'encoder_2_depth_predictor',
            'encoder_2d_coarse_depth': 'encoder_2_depth_predictor',
            'encoder_2e_depth_refine_unet': 'encoder_2_depth_predictor',
            'encoder_2f_gaussian_head': 'encoder_2_depth_predictor',
        },
        'skip': ['encoder'],
    },
    'depthsplat': {
        'order': [
            'encoder_1_depth_predictor',
            'encoder_2_feature_upsampler',
            'encoder_3_gaussian_regressor',
            'encoder_4_gaussian_head',
            'encoder_5_gaussian_adapter',
            'decoder',
        ],
        'parent_map': {
            # Sub-stages of depth_predictor
            'encoder_1a_cnn_backbone': 'encoder_1_depth_predictor',
            'encoder_1b_mv_transformer': 'encoder_1_depth_predictor',
            'encoder_1c_vit_backbone': 'encoder_1_depth_predictor',
            'encoder_1d_cost_volume': 'encoder_1_depth_predictor',
            'encoder_1e_unet_regressor': 'encoder_1_depth_predictor',
            'encoder_1f_depth_head': 'encoder_1_depth_predictor',
            'encoder_1g_depth_upsampler': 'encoder_1_depth_predictor',
        },
        'skip': ['encoder'],
    },
    'pixelsplat': {
        'order': [
            'encoder_1_backbone',
            'encoder_2_backbone_projection',
            'encoder_3_epipolar_transformer',
            'encoder_4_high_res_skip',
            'encoder_5_depth_predictor',
            'encoder_6_gaussian_adapter',
            'decoder',
        ],
        'parent_map': {},
        'skip': ['encoder'],
    },
}


def extract_nvtx_stage(nvtx_string: str) -> str:
    """Extract the innermost NVTX stage from the ncu NVTX column."""
    if not nvtx_string or nvtx_string == '""':
        return 'unknown'
    
    # Pattern: "<default domain>:stage_name:none:none:..."
    # Find all stage names in the string
    pattern = r'<default domain>:([^:]+):'
    matches = re.findall(pattern, nvtx_string)
    
    if not matches:
        return 'unknown'
    
    # Return the innermost (last) stage
    return matches[-1]


def parse_ncu_csv_with_nvtx(csv_path: str, model: str = 'transplat'):
    """Parse ncu CSV to get DRAM traffic and cache metrics grouped by NVTX stage."""
    if not Path(csv_path).exists():
        print(f"Error: CSV file not found: {csv_path}")
        return {}, 0, 0, False
    
    model_config = MODEL_STAGES.get(model, MODEL_STAGES['transplat'])
    parent_map = model_config['parent_map']
    skip_stages = model_config['skip']
    
    # Extended stats including cache metrics
    stage_traffic = defaultdict(lambda: {
        'read': 0, 'write': 0, 'kernels': 0,
        'l1_sectors': 0, 'l1_hit_rate_sum': 0, 'l1_hit_rate_count': 0,
        'l2_sectors': 0, 'l2_hit_rate_sum': 0, 'l2_hit_rate_count': 0,
    })
    total_read = 0
    total_write = 0
    has_cache_metrics = False
    
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    # Find header
    header_idx = -1
    for i, line in enumerate(lines):
        if line.startswith('"ID"'):
            header_idx = i
            break
    
    if header_idx < 0:
        print("Error: Could not find CSV header")
        return {}, 0, 0, False
    
    reader = csv.DictReader(lines[header_idx:])
    
    # Find the NVTX column - it might have different names
    nvtx_col = None
    for col in ['thread Domain:Push/Pop_Range:PL_Type:PL_Value:CLR_Type:Color:Msg_Type:Msg',
                'NVTX Range', 'nvtx_range']:
        if col in reader.fieldnames:
            nvtx_col = col
            break
    
    seen_ids = set()
    
    for row in reader:
        kernel_id = row.get('ID', '')
        metric_name = row.get('Metric Name', '')
        metric_value = row.get('Metric Value', '0')
        
        # Get NVTX stage
        nvtx_string = row.get(nvtx_col, '') if nvtx_col else ''
        stage = extract_nvtx_stage(nvtx_string)
        
        # Map sub-stages to parent
        if stage in parent_map:
            stage = parent_map[stage]
        elif stage in skip_stages:
            stage = 'unknown'
        
        try:
            value = float(metric_value.replace(',', '')) if metric_value else 0
        except:
            value = 0
        
        if 'dram__bytes_read' in metric_name:
            total_read += value
            stage_traffic[stage]['read'] += value
            if kernel_id not in seen_ids:
                stage_traffic[stage]['kernels'] += 1
                seen_ids.add(kernel_id)
        elif 'dram__bytes_write' in metric_name:
            total_write += value
            stage_traffic[stage]['write'] += value
        # Cache metrics - ncu outputs with suffixes like .sum, .pct, .ratio
        elif metric_name == 'l1tex__t_sectors.sum':
            stage_traffic[stage]['l1_sectors'] += value
            has_cache_metrics = True
        elif metric_name == 'l1tex__t_sector_hit_rate.pct':
            # Hit rate in percentage (0-100)
            stage_traffic[stage]['l1_hit_rate_sum'] += value
            stage_traffic[stage]['l1_hit_rate_count'] += 1
            has_cache_metrics = True
        elif metric_name == 'lts__t_sectors.sum':
            stage_traffic[stage]['l2_sectors'] += value
            has_cache_metrics = True
        elif metric_name == 'lts__t_sector_hit_rate.pct':
            # Hit rate in percentage (0-100)
            stage_traffic[stage]['l2_hit_rate_sum'] += value
            stage_traffic[stage]['l2_hit_rate_count'] += 1
            has_cache_metrics = True
    
    return dict(stage_traffic), total_read, total_write, has_cache_metrics


def parse_ncu_csv(csv_path: str):
    """Parse ncu CSV to get total DRAM traffic (legacy function)."""
    if not Path(csv_path).exists():
        return 0, 0, 0
    
    total_read = 0
    total_write = 0
    kernel_count = 0
    
    with open(csv_path, 'r') as f:
        lines = f.readlines()
    
    # Find header
    header_idx = -1
    for i, line in enumerate(lines):
        if line.startswith('"ID"'):
            header_idx = i
            break
    
    if header_idx < 0:
        return 0, 0, 0
    
    reader = csv.DictReader(lines[header_idx:])
    seen_ids = set()
    
    for row in reader:
        kernel_id = row.get('ID', '')
        metric_name = row.get('Metric Name', '')
        metric_value = row.get('Metric Value', '0')
        
        try:
            value = float(metric_value.replace(',', '')) if metric_value else 0
        except:
            value = 0
        
        if 'dram__bytes_read' in metric_name:
            total_read += value
            if kernel_id not in seen_ids:
                kernel_count += 1
                seen_ids.add(kernel_id)
        elif 'dram__bytes_write' in metric_name:
            total_write += value
    
    return total_read, total_write, kernel_count


def correlate_with_nsys(nsys_sqlite: str, ncu_csv: str, model: str = 'transplat'):
    """
    Correlate ncu kernel data with nsys NVTX ranges.
    Uses kernel timestamps to accurately map traffic to stages.
    """
    import sqlite3
    
    if not Path(nsys_sqlite).exists() or not Path(ncu_csv).exists():
        return {}
    
    model_config = MODEL_STAGES.get(model, MODEL_STAGES['transplat'])
    parent_map = model_config['parent_map']
    skip_stages = model_config['skip']
    
    # Get NVTX ranges from nsys
    conn = sqlite3.connect(nsys_sqlite)
    cursor = conn.cursor()
    
    cursor.execute("""
        SELECT text, start, end, (end - start) / 1e9 as duration_s
        FROM NVTX_EVENTS 
        WHERE eventType = 59
        ORDER BY start
    """)
    
    nvtx_ranges = []
    for row in cursor.fetchall():
        nvtx_ranges.append({
            'name': row[0],
            'start': row[1],
            'end': row[2],
            'duration': row[3]
        })
    
    # Get kernel timestamps from nsys - map each kernel to its NVTX range
    cursor.execute("""
        SELECT k.start, k.end, k.shortName,
               (SELECT n.text FROM NVTX_EVENTS n 
                WHERE n.eventType = 59 
                AND k.start >= n.start AND k.end <= n.end
                ORDER BY n.start DESC LIMIT 1) as nvtx_range
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        ORDER BY k.start
    """)
    
    # Build mapping: kernel order -> NVTX stage
    kernel_to_stage = []
    for row in cursor.fetchall():
        nvtx = row[3] if row[3] else 'unknown'
        kernel_to_stage.append({
            'name': row[2],
            'stage': nvtx
        })
    
    conn.close()
    
    # Parse ncu data
    with open(ncu_csv, 'r') as f:
        lines = f.readlines()
    
    header_idx = -1
    for i, line in enumerate(lines):
        if line.startswith('"ID"'):
            header_idx = i
            break
    
    if header_idx < 0:
        return {}
    
    reader = csv.DictReader(lines[header_idx:])
    
    # Build ncu kernel data by ID (in order)
    ncu_kernels = []
    ncu_data = {}
    
    for row in reader:
        kid = row.get('ID', '')
        metric_name = row.get('Metric Name', '')
        metric_value = row.get('Metric Value', '0')
        kernel_name = row.get('Kernel Name', '')
        
        try:
            value = float(metric_value.replace(',', '')) if metric_value else 0
        except:
            value = 0
        
        if kid not in ncu_data:
            ncu_data[kid] = {'read': 0, 'write': 0, 'name': kernel_name}
            ncu_kernels.append(kid)
        
        if 'dram__bytes_read' in metric_name:
            ncu_data[kid]['read'] = value
        elif 'dram__bytes_write' in metric_name:
            ncu_data[kid]['write'] = value
    
    # Map ncu kernels to stages using kernel order correlation
    stage_traffic = defaultdict(lambda: {'read': 0, 'write': 0, 'kernels': 0, 'duration': 0})
    
    # Match by kernel order (ncu and nsys should have same kernel order)
    for i, kid in enumerate(ncu_kernels):
        if i < len(kernel_to_stage):
            stage = kernel_to_stage[i]['stage']
            
            # Map sub-stages to parent using model-specific config
            if stage in parent_map:
                stage = parent_map[stage]
            elif stage in skip_stages:
                continue
            
            stage_traffic[stage]['read'] += ncu_data[kid]['read']
            stage_traffic[stage]['write'] += ncu_data[kid]['write']
            stage_traffic[stage]['kernels'] += 1
    
    # Add duration from nvtx_ranges
    for nvtx in nvtx_ranges:
        if nvtx['name'] in stage_traffic:
            stage_traffic[nvtx['name']]['duration'] = nvtx['duration']
    
    return dict(stage_traffic)


def print_cache_report(stage_traffic: dict, stage_order: list, model: str):
    """Print cache hit rate statistics per stage."""
    print(f"\n{'=' * 100}")
    print(f"CACHE HIT RATE STATISTICS - {model.upper()} (per stage via ncu)")
    print("=" * 100)
    
    print(f"\n{'Stage':<40} {'L1 Sectors':>12} {'L1 Hit%':>10} {'L2 Sectors':>12} {'L2 Hit%':>10}")
    print("-" * 100)
    
    total_l1_sectors = 0
    total_l1_hit_sum = 0
    total_l1_hit_count = 0
    total_l2_sectors = 0
    total_l2_hit_sum = 0
    total_l2_hit_count = 0
    
    for stage in stage_order:
        if stage in stage_traffic:
            data = stage_traffic[stage]
            l1_sectors = data.get('l1_sectors', 0)
            l1_hit_count = data.get('l1_hit_rate_count', 0)
            # Hit rate sum is already in percentage (0-100), just average it
            l1_hit_rate = data.get('l1_hit_rate_sum', 0) / l1_hit_count if l1_hit_count > 0 else 0
            
            l2_sectors = data.get('l2_sectors', 0)
            l2_hit_count = data.get('l2_hit_rate_count', 0)
            l2_hit_rate = data.get('l2_hit_rate_sum', 0) / l2_hit_count if l2_hit_count > 0 else 0
            
            total_l1_sectors += l1_sectors
            total_l1_hit_sum += data.get('l1_hit_rate_sum', 0)
            total_l1_hit_count += l1_hit_count
            total_l2_sectors += l2_sectors
            total_l2_hit_sum += data.get('l2_hit_rate_sum', 0)
            total_l2_hit_count += l2_hit_count
            
            # Format sectors (in millions)
            l1_str = f"{l1_sectors/1e6:.2f}M" if l1_sectors > 0 else "N/A"
            l1_hit_str = f"{l1_hit_rate:.1f}%" if l1_hit_count > 0 else "N/A"
            l2_str = f"{l2_sectors/1e6:.2f}M" if l2_sectors > 0 else "N/A"
            l2_hit_str = f"{l2_hit_rate:.1f}%" if l2_hit_count > 0 else "N/A"
            
            print(f"{stage:<40} {l1_str:>12} {l1_hit_str:>10} {l2_str:>12} {l2_hit_str:>10}")
    
    # Print other stages not in order
    for stage, data in stage_traffic.items():
        if stage not in stage_order and stage != 'unknown':
            l1_sectors = data.get('l1_sectors', 0)
            l1_hit_count = data.get('l1_hit_rate_count', 0)
            l1_hit_rate = data.get('l1_hit_rate_sum', 0) / l1_hit_count if l1_hit_count > 0 else 0
            
            l2_sectors = data.get('l2_sectors', 0)
            l2_hit_count = data.get('l2_hit_rate_count', 0)
            l2_hit_rate = data.get('l2_hit_rate_sum', 0) / l2_hit_count if l2_hit_count > 0 else 0
            
            if l1_sectors > 0 or l2_sectors > 0:
                l1_str = f"{l1_sectors/1e6:.2f}M" if l1_sectors > 0 else "N/A"
                l1_hit_str = f"{l1_hit_rate:.1f}%" if l1_hit_count > 0 else "N/A"
                l2_str = f"{l2_sectors/1e6:.2f}M" if l2_sectors > 0 else "N/A"
                l2_hit_str = f"{l2_hit_rate:.1f}%" if l2_hit_count > 0 else "N/A"
                
                print(f"{stage:<40} {l1_str:>12} {l1_hit_str:>10} {l2_str:>12} {l2_hit_str:>10}")
    
    print("-" * 100)
    
    # Calculate weighted averages
    avg_l1_hit = total_l1_hit_sum / total_l1_hit_count if total_l1_hit_count > 0 else 0
    avg_l2_hit = total_l2_hit_sum / total_l2_hit_count if total_l2_hit_count > 0 else 0
    
    l1_total_str = f"{total_l1_sectors/1e6:.2f}M" if total_l1_sectors > 0 else "N/A"
    l2_total_str = f"{total_l2_sectors/1e6:.2f}M" if total_l2_sectors > 0 else "N/A"
    
    print(f"{'TOTAL / AVERAGE':<40} {l1_total_str:>12} {avg_l1_hit:>9.1f}% {l2_total_str:>12} {avg_l2_hit:>9.1f}%")
    print("=" * 100)
    print("")


def print_stage_report(stage_traffic: dict, total_read: float, total_write: float, model: str = 'transplat', has_cache_metrics: bool = False):
    """Print formatted per-stage HBM traffic report."""
    model_config = MODEL_STAGES.get(model, MODEL_STAGES['transplat'])
    stage_order = model_config['order']
    
    print(f"\n{'=' * 95}")
    print(f"HBM DRAM TRAFFIC STATISTICS - {model.upper()} (per stage via ncu)")
    print("=" * 95)
    
    # Calculate inference-only traffic (excluding init/unknown)
    inference_read = 0
    inference_write = 0
    
    for stage in stage_order:
        if stage in stage_traffic:
            inference_read += stage_traffic[stage]['read']
            inference_write += stage_traffic[stage]['write']
    
    # Also add any stages found but not in order
    for stage, data in stage_traffic.items():
        if stage not in stage_order and stage != 'unknown':
            inference_read += data['read']
            inference_write += data['write']
    
    inference_total = inference_read + inference_write
    
    print(f"\n{'Stage':<40} {'Read(GB)':>10} {'Write(GB)':>10} {'Total(GB)':>10} {'% of Inf':>12} {'% of Encoder':>14}")
    print("-" * 95)
    
    encoder_total_read = 0
    encoder_total_write = 0
    decoder_read = 0
    decoder_write = 0
    
    # Print stages in order
    for stage in stage_order:
        if stage in stage_traffic:
            data = stage_traffic[stage]
            read_gb = data['read'] / 1e9
            write_gb = data['write'] / 1e9
            total_gb = read_gb + write_gb
            pct_inf = (data['read'] + data['write']) / inference_total * 100 if inference_total > 0 else 0
            
            if stage == 'decoder':
                decoder_read = data['read']
                decoder_write = data['write']
                pct_enc = '--'
            else:
                encoder_total_read += data['read']
                encoder_total_write += data['write']
                enc_total = sum(stage_traffic.get(s, {}).get('read', 0) + stage_traffic.get(s, {}).get('write', 0) 
                               for s in stage_order if s != 'decoder')
                pct_enc = f"{((data['read'] + data['write']) / enc_total * 100):.1f}%" if enc_total > 0 else '--'
            
            print(f"{stage:<40} {read_gb:>10.2f} {write_gb:>10.2f} {total_gb:>10.2f} {pct_inf:>11.1f}% {pct_enc:>14}")
    
    # Print stages not in order (other detected stages)
    for stage, data in stage_traffic.items():
        if stage not in stage_order and stage != 'unknown' and (data['read'] + data['write']) > 1e6:
            read_gb = data['read'] / 1e9
            write_gb = data['write'] / 1e9
            total_gb = read_gb + write_gb
            pct_inf = (data['read'] + data['write']) / inference_total * 100 if inference_total > 0 else 0
            print(f"{stage:<40} {read_gb:>10.2f} {write_gb:>10.2f} {total_gb:>10.2f} {pct_inf:>11.1f}%")
    
    encoder_total = encoder_total_read + encoder_total_write
    decoder_total = decoder_read + decoder_write
    
    print("-" * 95)
    print(f"{'ENCODER TOTAL':<40} {encoder_total_read/1e9:>10.2f} {encoder_total_write/1e9:>10.2f} {encoder_total/1e9:>10.2f} {encoder_total/inference_total*100 if inference_total > 0 else 0:>11.1f}%")
    print(f"{'DECODER TOTAL':<40} {decoder_read/1e9:>10.2f} {decoder_write/1e9:>10.2f} {decoder_total/1e9:>10.2f} {decoder_total/inference_total*100 if inference_total > 0 else 0:>11.1f}%")
    print("=" * 95)
    print(f"{'INFERENCE HBM TRAFFIC':<40} {inference_read/1e9:>10.2f} {inference_write/1e9:>10.2f} {inference_total/1e9:>10.2f} GB")
    print("=" * 95)
    
    # Show total including init
    total = total_read + total_write
    if total > inference_total * 1.01:  # More than 1% extra
        init_traffic = total - inference_total
        print(f"\n(Init/loading overhead: {init_traffic/1e9:.2f} GB, {init_traffic/total*100:.1f}% of total {total/1e9:.2f} GB)")
    
    print("")
    
    # Print cache hit rate table if metrics available
    if has_cache_metrics:
        print_cache_report(stage_traffic, stage_order, model)
    
    # Save to JSON
    output_data = {
        'model': model,
        'stages': {},
        'summary': {
            'encoder_read_gb': encoder_total_read / 1e9,
            'encoder_write_gb': encoder_total_write / 1e9,
            'encoder_total_gb': encoder_total / 1e9,
            'decoder_read_gb': decoder_read / 1e9,
            'decoder_write_gb': decoder_write / 1e9,
            'decoder_total_gb': decoder_total / 1e9,
            'inference_total_gb': inference_total / 1e9,
            'total_with_init_gb': total / 1e9,
        }
    }
    
    for stage, data in stage_traffic.items():
        output_data['stages'][stage] = {
            'read_gb': data['read'] / 1e9,
            'write_gb': data['write'] / 1e9,
            'total_gb': (data['read'] + data['write']) / 1e9,
            'kernels': data.get('kernels', 0),
        }
    
    return output_data


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Profile HBM traffic by stage")
    parser.add_argument("--ncu-csv", help="Existing ncu CSV file")
    parser.add_argument("--nsys-sqlite", help="nsys SQLite file for correlation (optional if ncu has NVTX)")
    parser.add_argument("--model", default="transplat", 
                       choices=['transplat', 'mvsplat', 'depthsplat', 'pixelsplat'],
                       help="Model type for stage configuration")
    parser.add_argument("--output-json", help="Output JSON file path")
    args = parser.parse_args()
    
    if args.ncu_csv:
        # Try to parse NVTX directly from ncu CSV first
        stage_traffic, total_read, total_write, has_cache_metrics = parse_ncu_csv_with_nvtx(args.ncu_csv, args.model)
        
        # If no stages found and nsys file provided, try correlation
        if not any(s for s in stage_traffic if s != 'unknown') and args.nsys_sqlite:
            print("No NVTX data in ncu CSV, using nsys correlation...")
            total_read, total_write, _ = parse_ncu_csv(args.ncu_csv)
            stage_traffic = correlate_with_nsys(args.nsys_sqlite, args.ncu_csv, args.model)
            has_cache_metrics = False
        
        output_data = print_stage_report(stage_traffic, total_read, total_write, args.model, has_cache_metrics)
        
        # Save to JSON if requested
        if args.output_json:
            with open(args.output_json, 'w') as f:
                json.dump(output_data, f, indent=2)
            print(f"Results saved to {args.output_json}")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
