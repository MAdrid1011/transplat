#!/usr/bin/env python3
"""
Generate a 4-phase summary table for model timing analysis.

Phases:
1. Feature Extraction - CNN/ViT backbone, feature projection
2. Depth Prediction - Cross-view matching, cost volume, depth estimation
3. Gaussian Generation - Gaussian regressor, head, adapter
4. Decoder - Splatting/rasterization
"""

import sys
import json
import re
from pathlib import Path
from collections import defaultdict


# Stage-to-phase mapping for each model
# Note: For stages with sub-stages (e.g., encoder_1_depth_predictor with encoder_1a_*, encoder_1b_*, etc.),
# the script will use detailed sub-stages if available to avoid double-counting
PHASE_MAPPING = {
    'transplat': {
        'Feature Extraction': [
            'encoder_1_prep_intrinsics',
            'encoder_2_backbone',
        ],
        'Depth Prediction': [
            'encoder_3_depth_anything',  # 单目深度先验
            'encoder_4_depth_predictor',
            'encoder_4a_prep_features',
            'encoder_4b_cost_volume_matching', 
            'encoder_4c_cost_volume_unet',
            'encoder_4d_coarse_depth',
            'encoder_4e_depth_refine_unet',
        ],
        'Gaussian Generation': [
            'encoder_4f_gaussian_head',
            'encoder_5_gaussian_adapter',
        ],
        'Decoder': [
            'decoder',
        ],
    },
    'mvsplat': {
        'Feature Extraction': [
            'encoder_1_backbone',
        ],
        'Depth Prediction': [
            'encoder_2_depth_predictor',
            'encoder_2a_prep_features',
            'encoder_2b_cost_volume',
            'encoder_2c_cost_volume_unet',
            'encoder_2d_coarse_depth',
            'encoder_2e_depth_refine_unet',
        ],
        'Gaussian Generation': [
            'encoder_2f_gaussian_head',
            'encoder_3_gaussian_adapter',
        ],
        'Decoder': [
            'decoder',
        ],
    },
    'depthsplat': {
        'Feature Extraction': [
            'encoder_1a_cnn_backbone',
            'encoder_1b_mv_transformer',
            'encoder_1c_vit_backbone',
        ],
        'Depth Prediction': [
            # Timing data uses detailed sub-stages
            'encoder_1d_cost_volume',
            'encoder_1e_unet_regressor',
            'encoder_1f_depth_head',
            'encoder_1g_depth_upsampler',
            'encoder_2_feature_upsampler',
        ],
        'Gaussian Generation': [
            'encoder_3_gaussian_regressor',
            'encoder_4_gaussian_head',
            'encoder_5_gaussian_adapter',
        ],
        'Decoder': [
            'decoder',
        ],
        # Special: HBM data uses parent tag, need to split by timing ratio
        '_hbm_split': {
            'encoder_1_depth_predictor': {
                'Feature Extraction': 0.50,  # 1a+1b+1c timing ratio
                'Depth Prediction': 0.50,    # 1d+1e+1f+1g timing ratio
            }
        },
    },
    'pixelsplat': {
        'Feature Extraction': [
            'encoder_1_backbone',
            'encoder_2_backbone_projection',
        ],
        'Depth Prediction': [
            'encoder_3_epipolar_transformer',
            'encoder_4_high_res_skip',
            'encoder_5_depth_predictor',
        ],
        'Gaussian Generation': [
            'encoder_6_gaussian_adapter',
        ],
        'Decoder': [
            'decoder',
        ],
    },
}


def parse_timing_output(output_text):
    """Parse timing statistics from model output text."""
    timing_data = {}
    
    # Look for the DETAILED TIMING STATISTICS section
    in_timing_section = False
    for line in output_text.split('\n'):
        if 'DETAILED TIMING STATISTICS' in line:
            in_timing_section = True
            continue
        # End of timing section when we hit TOTAL INFERENCE or HBM DRAM TRAFFIC
        if in_timing_section and ('TOTAL INFERENCE TIME' in line or 'HBM DRAM TRAFFIC' in line):
            in_timing_section = False
            continue
        if in_timing_section and '====' in line:
            continue
        if in_timing_section and line.strip().startswith('Stage'):
            continue
        if in_timing_section and line.startswith('---'):
            continue
        if in_timing_section and ('ENCODER TOTAL' in line or 'DECODER TOTAL' in line):
            continue
            
        if in_timing_section and line.strip():
            # Parse lines like: "encoder_1_backbone                       1   77088.10     77.088        21.0%         22.5%"
            parts = line.split()
            if len(parts) >= 4 and (parts[0].startswith('encoder') or parts[0].startswith('decoder')):
                stage_name = parts[0]
                try:
                    # Calls is parts[1], Avg(ms) is parts[2], Total(s) is parts[3]
                    avg_ms = float(parts[2])
                    total_s = float(parts[3])
                    timing_data[stage_name] = {
                        'avg_ms': avg_ms,
                        'total_s': total_s,
                    }
                except (ValueError, IndexError):
                    pass
    
    return timing_data


def parse_hbm_traffic_output(output_text):
    """Parse HBM traffic statistics from model output text."""
    hbm_data = {}
    
    # Look for HBM traffic section (from ncu analysis)
    # Format: "Stage   Read(GB)  Write(GB)  Total(GB)   % of Inf   % of Encoder"
    in_hbm_section = False
    for line in output_text.split('\n'):
        if 'HBM DRAM TRAFFIC STATISTICS' in line:
            in_hbm_section = True
            continue
        if in_hbm_section and '====' in line:
            if 'INFERENCE HBM TRAFFIC' in line:
                in_hbm_section = False
            continue
        if in_hbm_section and line.strip().startswith('Stage'):
            continue
        if in_hbm_section and line.startswith('---'):
            continue
        if in_hbm_section and ('ENCODER TOTAL' in line or 'DECODER TOTAL' in line):
            continue
            
        if in_hbm_section and line.strip():
            parts = line.split()
            # Format: encoder_1_backbone    3.94    3.12    7.05    40.5%    41.8%
            if len(parts) >= 4:
                stage_name = parts[0]
                if stage_name.startswith('encoder') or stage_name.startswith('decoder'):
                    try:
                        # Read(GB) is parts[1], Write(GB) is parts[2], Total(GB) is parts[3]
                        total_gb = float(parts[3])
                        hbm_data[stage_name] = total_gb * 1e9  # Convert to bytes
                    except (ValueError, IndexError):
                        pass
    
    return hbm_data


def parse_cache_output(output_text):
    """Parse cache hit rate statistics from model output text."""
    cache_data = {}
    
    # Look for CACHE HIT RATE STATISTICS section
    # Format: "Stage   L1 Sectors   L1 Hit%   L2 Sectors   L2 Hit%"
    in_cache_section = False
    header_passed = False
    for line in output_text.split('\n'):
        if 'CACHE HIT RATE STATISTICS' in line:
            in_cache_section = True
            header_passed = False
            continue
        if in_cache_section and '====' in line:
            if header_passed:
                # Second ==== marks end of section
                in_cache_section = False
            else:
                # First ==== is just after header
                header_passed = True
            continue
        if in_cache_section and line.strip().startswith('Stage'):
            continue
        if in_cache_section and line.startswith('---'):
            continue
        if in_cache_section and 'TOTAL' in line:
            continue
            
        if in_cache_section and line.strip():
            parts = line.split()
            # Format: encoder_1_backbone    376.97M    14.9%    478.88M    64.4%
            if len(parts) >= 5:
                stage_name = parts[0]
                if stage_name.startswith('encoder') or stage_name.startswith('decoder'):
                    try:
                        # L1 Sectors is parts[1] (e.g., "376.97M"), L1 Hit% is parts[2]
                        # L2 Sectors is parts[3], L2 Hit% is parts[4]
                        l1_sectors_str = parts[1]
                        l1_hit_str = parts[2].rstrip('%')
                        l2_sectors_str = parts[3]
                        l2_hit_str = parts[4].rstrip('%')
                        
                        # Parse sectors (can be M for millions)
                        l1_sectors = 0
                        if l1_sectors_str != 'N/A':
                            if l1_sectors_str.endswith('M'):
                                l1_sectors = float(l1_sectors_str[:-1]) * 1e6
                            else:
                                l1_sectors = float(l1_sectors_str)
                        
                        l2_sectors = 0
                        if l2_sectors_str != 'N/A':
                            if l2_sectors_str.endswith('M'):
                                l2_sectors = float(l2_sectors_str[:-1]) * 1e6
                            else:
                                l2_sectors = float(l2_sectors_str)
                        
                        l1_hit = float(l1_hit_str) if l1_hit_str != 'N/A' else 0
                        l2_hit = float(l2_hit_str) if l2_hit_str != 'N/A' else 0
                        
                        cache_data[stage_name] = {
                            'l1_sectors': l1_sectors,
                            'l1_hit': l1_hit,
                            'l2_sectors': l2_sectors,
                            'l2_hit': l2_hit,
                        }
                    except (ValueError, IndexError):
                        pass
    
    return cache_data


def aggregate_to_phases(timing_data, hbm_data, model_name, cache_data=None):
    """Aggregate detailed stages into 4 high-level phases."""
    mapping = PHASE_MAPPING.get(model_name, {})
    
    phase_timing = defaultdict(float)
    phase_hbm = defaultdict(float)
    phase_cache = defaultdict(lambda: {'l1_sectors': 0, 'l1_hit_weighted': 0, 'l2_sectors': 0, 'l2_hit_weighted': 0})
    
    if cache_data is None:
        cache_data = {}
    
    # Track which stages have been assigned
    assigned_timing_stages = set()
    assigned_hbm_stages = set()
    assigned_cache_stages = set()
    
    # First, identify parent-child relationships
    # Pattern: encoder_2_depth_predictor is parent of encoder_2a_*, encoder_2b_*, etc.
    # Or: encoder_1_depth_predictor is parent of encoder_1a_*, encoder_1b_*, etc.
    def get_parent_prefix(stage_name):
        """Extract parent prefix like 'encoder_2' from 'encoder_2a_prep_features'."""
        import re
        match = re.match(r'(encoder_\d+)[a-z]_', stage_name)
        if match:
            return match.group(1)
        return None
    
    def has_detailed_children(parent_pattern, all_stages):
        """Check if a parent stage has detailed sub-stages (e.g., encoder_2_* has encoder_2a_*, encoder_2b_*)."""
        # Extract the prefix number like 'encoder_2' from 'encoder_2_depth_predictor'
        import re
        match = re.match(r'(encoder_\d+)_', parent_pattern)
        if match:
            prefix = match.group(1)
            # Check if there are stages like encoder_2a_*, encoder_2b_*, etc.
            for s in all_stages:
                if re.match(rf'{prefix}[a-z]_', s):
                    return True
        return False
    
    # Process explicit mappings (skip special keys starting with _)
    for phase_name, stage_patterns in mapping.items():
        if phase_name.startswith('_'):
            continue
        for pattern in stage_patterns:
            for stage_name, stage_data in timing_data.items():
                if stage_name == pattern:
                    # Check if this parent stage has detailed children
                    if has_detailed_children(stage_name, timing_data.keys()):
                        # Skip parent if detailed children exist (they'll be added separately)
                        assigned_timing_stages.add(stage_name)  # Mark as processed but don't add time
                    else:
                        phase_timing[phase_name] += stage_data.get('total_s', 0) * 1000
                        assigned_timing_stages.add(stage_name)
            
            # For HBM, use whatever is available (usually parent-level)
            for stage_name, traffic in hbm_data.items():
                if stage_name == pattern and stage_name not in assigned_hbm_stages:
                    phase_hbm[phase_name] += traffic
                    assigned_hbm_stages.add(stage_name)
    
    # Handle unassigned stages - try to categorize them
    for stage_name, stage_data in timing_data.items():
        if stage_name not in assigned_timing_stages:
            # Skip top-level encoder/decoder aggregates
            if stage_name in ('encoder', 'decoder'):
                continue
            
            # Skip parent stages that have detailed children (to avoid double counting)
            if has_detailed_children(stage_name, timing_data.keys()):
                assigned_timing_stages.add(stage_name)  # Mark as processed
                continue
            
            # Heuristic categorization based on stage name
            stage_lower = stage_name.lower()
            if 'backbone' in stage_lower or '_da_' in stage_lower or 'vit' in stage_lower or 'projection' in stage_lower:
                if 'cnn' in stage_lower or 'mv_transformer' in stage_lower or '1a_' in stage_lower or '1b_' in stage_lower or '1c_' in stage_lower:
                    phase_timing['Feature Extraction'] += stage_data.get('total_s', 0) * 1000
                    assigned_timing_stages.add(stage_name)
                elif 'backbone' in stage_lower and 'encoder_1_' in stage_lower:
                    phase_timing['Feature Extraction'] += stage_data.get('total_s', 0) * 1000
                    assigned_timing_stages.add(stage_name)
            elif 'depth' in stage_lower or 'cost' in stage_lower or 'epipolar' in stage_lower or 'unet' in stage_lower or 'prep_features' in stage_lower or 'coarse' in stage_lower or 'refine' in stage_lower or 'upsampler' in stage_lower:
                phase_timing['Depth Prediction'] += stage_data.get('total_s', 0) * 1000
                assigned_timing_stages.add(stage_name)
            elif 'gaussian' in stage_lower:
                phase_timing['Gaussian Generation'] += stage_data.get('total_s', 0) * 1000
                assigned_timing_stages.add(stage_name)
    
    # Add decoder separately
    if 'decoder' in timing_data and 'decoder' not in assigned_timing_stages:
        phase_timing['Decoder'] += timing_data['decoder'].get('total_s', 0) * 1000
        assigned_timing_stages.add('decoder')
    
    # Handle special HBM split configuration (e.g., DepthSplat's encoder_1_depth_predictor)
    hbm_split_config = mapping.get('_hbm_split', {})
    for stage_name, traffic in hbm_data.items():
        if stage_name in hbm_split_config and stage_name not in assigned_hbm_stages:
            # Split this HBM traffic across phases according to configured ratios
            split_ratios = hbm_split_config[stage_name]
            for phase_name, ratio in split_ratios.items():
                phase_hbm[phase_name] += traffic * ratio
            assigned_hbm_stages.add(stage_name)
    
    # Handle unassigned HBM stages
    for stage_name, traffic in hbm_data.items():
        if stage_name not in assigned_hbm_stages:
            stage_lower = stage_name.lower()
            if 'backbone' in stage_lower or 'feature' in stage_lower:
                phase_hbm['Feature Extraction'] += traffic
            elif 'depth' in stage_lower or 'cost' in stage_lower or 'epipolar' in stage_lower:
                phase_hbm['Depth Prediction'] += traffic
            elif 'gaussian' in stage_lower:
                phase_hbm['Gaussian Generation'] += traffic
            elif 'decoder' in stage_lower:
                phase_hbm['Decoder'] += traffic
    
    # Aggregate cache data by phase (handle split stages like encoder_1_depth_predictor for DepthSplat)
    cache_split_config = mapping.get('_hbm_split', {})  # Reuse HBM split ratios for cache
    
    for stage_name, cache_info in cache_data.items():
        stage_lower = stage_name.lower()
        
        l1_sectors = cache_info.get('l1_sectors', 0)
        l2_sectors = cache_info.get('l2_sectors', 0)
        l1_hit = cache_info.get('l1_hit', 0)
        l2_hit = cache_info.get('l2_hit', 0)
        
        # Check if this stage needs to be split across phases
        if stage_name in cache_split_config:
            split_ratios = cache_split_config[stage_name]
            for phase_name, ratio in split_ratios.items():
                phase_cache[phase_name]['l1_sectors'] += l1_sectors * ratio
                phase_cache[phase_name]['l1_hit_weighted'] += l1_hit * l1_sectors * ratio
                phase_cache[phase_name]['l2_sectors'] += l2_sectors * ratio
                phase_cache[phase_name]['l2_hit_weighted'] += l2_hit * l2_sectors * ratio
        else:
            # Determine phase based on stage name
            phase_name = None
            if 'backbone' in stage_lower or 'prep_intrinsics' in stage_lower or 'projection' in stage_lower:
                if 'depth' not in stage_lower:
                    phase_name = 'Feature Extraction'
            if phase_name is None and ('depth' in stage_lower or 'cost' in stage_lower or 'epipolar' in stage_lower or 'unet' in stage_lower or 'prep_features' in stage_lower or 'coarse' in stage_lower or 'refine' in stage_lower or 'upsampler' in stage_lower or 'skip' in stage_lower):
                phase_name = 'Depth Prediction'
            if phase_name is None and 'gaussian' in stage_lower:
                phase_name = 'Gaussian Generation'
            if phase_name is None and 'decoder' in stage_lower:
                phase_name = 'Decoder'
            
            if phase_name:
                phase_cache[phase_name]['l1_sectors'] += l1_sectors
                phase_cache[phase_name]['l1_hit_weighted'] += l1_hit * l1_sectors
                phase_cache[phase_name]['l2_sectors'] += l2_sectors
                phase_cache[phase_name]['l2_hit_weighted'] += l2_hit * l2_sectors
    
    return phase_timing, phase_hbm, dict(phase_cache)


def print_phase_summary(model_name, phase_timing, phase_hbm, phase_cache=None):
    """Print the 4-phase summary table."""
    
    phases = ['Feature Extraction', 'Depth Prediction', 'Gaussian Generation', 'Decoder']
    
    total_time = sum(phase_timing.values())
    total_hbm = sum(phase_hbm.values())
    
    if phase_cache is None:
        phase_cache = {}
    
    has_cache = any(phase_cache.get(p, {}).get('l1_sectors', 0) > 0 or phase_cache.get(p, {}).get('l2_sectors', 0) > 0 for p in phases)
    
    print("")
    if has_cache:
        print("=" * 120)
    else:
        print("=" * 90)
    print(f"  {model_name.upper()} - UNIFIED 4-PHASE SUMMARY")
    print("  (Encoder = Feature Extraction + Depth Prediction + Gaussian Generation)")
    if has_cache:
        print("=" * 120)
        print("")
        print(f"{'Phase':<25} {'Time(ms)':>12} {'Time%':>8} {'HBM Traffic':>12} {'HBM%':>8} {'L1 Hit%':>10} {'L2 Hit%':>10}")
        print("-" * 120)
    else:
        print("=" * 90)
        print("")
        print(f"{'Phase':<25} {'Time(ms)':>12} {'Time%':>10} {'HBM Traffic':>15} {'HBM%':>10}")
        print("-" * 90)
    
    total_l1_sectors = 0
    total_l1_hit_weighted = 0
    total_l2_sectors = 0
    total_l2_hit_weighted = 0
    
    for phase in phases:
        time_ms = phase_timing.get(phase, 0)
        time_pct = (time_ms / total_time * 100) if total_time > 0 else 0
        
        hbm_bytes = phase_hbm.get(phase, 0)
        hbm_pct = (hbm_bytes / total_hbm * 100) if total_hbm > 0 else 0
        
        # Format HBM traffic
        if hbm_bytes >= 1e9:
            hbm_str = f"{hbm_bytes/1e9:.2f} GB"
        elif hbm_bytes >= 1e6:
            hbm_str = f"{hbm_bytes/1e6:.2f} MB"
        elif hbm_bytes > 0:
            hbm_str = f"{hbm_bytes/1e3:.2f} KB"
        else:
            hbm_str = "N/A"
        
        hbm_pct_str = f"{hbm_pct:.1f}%" if hbm_bytes > 0 else "N/A"
        
        if has_cache:
            cache_info = phase_cache.get(phase, {})
            l1_sectors = cache_info.get('l1_sectors', 0)
            l1_hit_weighted = cache_info.get('l1_hit_weighted', 0)
            l2_sectors = cache_info.get('l2_sectors', 0)
            l2_hit_weighted = cache_info.get('l2_hit_weighted', 0)
            
            l1_hit = l1_hit_weighted / l1_sectors if l1_sectors > 0 else 0
            l2_hit = l2_hit_weighted / l2_sectors if l2_sectors > 0 else 0
            
            total_l1_sectors += l1_sectors
            total_l1_hit_weighted += l1_hit_weighted
            total_l2_sectors += l2_sectors
            total_l2_hit_weighted += l2_hit_weighted
            
            l1_str = f"{l1_hit:.1f}%" if l1_sectors > 0 else "N/A"
            l2_str = f"{l2_hit:.1f}%" if l2_sectors > 0 else "N/A"
            
            print(f"{phase:<25} {time_ms:>12.2f} {time_pct:>7.1f}% {hbm_str:>12} {hbm_pct_str:>8} {l1_str:>10} {l2_str:>10}")
        else:
            print(f"{phase:<25} {time_ms:>12.2f} {time_pct:>9.1f}% {hbm_str:>15} {hbm_pct_str:>10}")
    
    if has_cache:
        print("-" * 120)
    else:
        print("-" * 90)
    
    # Totals
    if total_hbm >= 1e9:
        total_hbm_str = f"{total_hbm/1e9:.2f} GB"
    elif total_hbm >= 1e6:
        total_hbm_str = f"{total_hbm/1e6:.2f} MB"
    elif total_hbm > 0:
        total_hbm_str = f"{total_hbm/1e3:.2f} KB"
    else:
        total_hbm_str = "N/A"
    
    if has_cache:
        avg_l1 = total_l1_hit_weighted / total_l1_sectors if total_l1_sectors > 0 else 0
        avg_l2 = total_l2_hit_weighted / total_l2_sectors if total_l2_sectors > 0 else 0
        l1_avg_str = f"{avg_l1:.1f}%" if total_l1_sectors > 0 else "N/A"
        l2_avg_str = f"{avg_l2:.1f}%" if total_l2_sectors > 0 else "N/A"
        print(f"{'TOTAL':<25} {total_time:>12.2f} {'100.0%':>8} {total_hbm_str:>12} {'100.0%' if total_hbm > 0 else 'N/A':>8} {l1_avg_str:>10} {l2_avg_str:>10}")
        print("=" * 120)
    else:
        print(f"{'TOTAL':<25} {total_time:>12.2f} {'100.0%':>10} {total_hbm_str:>15} {'100.0%' if total_hbm > 0 else 'N/A':>10}")
        print("=" * 90)
    print("")


def main():
    if len(sys.argv) < 2:
        print("Usage: python generate_phase_summary.py <model_name> [timing_json] [hbm_json]")
        print("  model_name: transplat, mvsplat, depthsplat, pixelsplat")
        print("")
        print("If no JSON files provided, reads from stdin for timing output parsing.")
        sys.exit(1)
    
    model_name = sys.argv[1].lower()
    
    timing_data = {}
    hbm_data = {}
    cache_data = {}
    
    # Try to read from JSON files if provided
    if len(sys.argv) >= 3:
        timing_json_path = Path(sys.argv[2])
        if timing_json_path.exists():
            with open(timing_json_path) as f:
                raw_timing = json.load(f)
                # Convert from {tag: [times]} to {tag: {total_s: sum}}
                for tag, times in raw_timing.items():
                    timing_data[tag] = {
                        'avg_ms': sum(times) / len(times) * 1000 if times else 0,
                        'total_s': sum(times),
                    }
    
    if len(sys.argv) >= 4:
        hbm_json_path = Path(sys.argv[3])
        if hbm_json_path.exists():
            with open(hbm_json_path) as f:
                hbm_data = json.load(f)
    
    # If no timing data from JSON, try to parse from stdin
    if not timing_data:
        try:
            # Read from stdin (piped output)
            if not sys.stdin.isatty():
                output_text = sys.stdin.read()
                timing_data = parse_timing_output(output_text)
                hbm_data = parse_hbm_traffic_output(output_text)
                cache_data = parse_cache_output(output_text)
        except Exception as e:
            print(f"Warning: Could not parse stdin: {e}")
    
    if not timing_data:
        print(f"Warning: No timing data available for {model_name}")
        # Still print empty table
        timing_data = {}
    
    # Aggregate to phases
    phase_timing, phase_hbm, phase_cache = aggregate_to_phases(timing_data, hbm_data, model_name, cache_data)
    
    # Print summary
    print_phase_summary(model_name, phase_timing, phase_hbm, phase_cache)


if __name__ == "__main__":
    main()
