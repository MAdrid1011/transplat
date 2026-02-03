#!/bin/bash
# Script to run timing analysis for all four 3D Gaussian Splatting models
# Usage: bash scripts/run_all_timing_tests.sh [model] [--memory] [--hbm-traffic] [--ncu] [--ncu-cache] [--analyze-fd] [--analyze-redundancy] [--parse-only]
# model: all, transplat, mvsplat, depthsplat, pixelsplat
# --memory: Enable memory allocation profiling (all models)
# --hbm-traffic: Enable real HBM traffic measurement via nsys (with NVTX markers)
# --ncu: Enable kernel-level DRAM traffic via Nsight Compute (slow but accurate)
# --ncu-cache: Enable ncu with additional cache hit rate metrics (even slower)
# --analyze-fd: Analyze feature-depth correlation (validates similar features = similar depths)
# --analyze-redundancy: Analyze Gaussian redundancy (Challenge 2: view-related + spatial redundancy)
# --parse-only: Only parse existing output files without running models

set -e

# Base directory
BASE_DIR="/home/mazirui/transplat"
CONDA_ENV="transplat"
PROFILE_DIR="$BASE_DIR/profile_outputs"

# Create profile output directory if not exists
mkdir -p "$PROFILE_DIR"

# Parse arguments
PROFILE_MEMORY="false"
PROFILE_HBM_TRAFFIC="false"
PROFILE_NCU="false"
PROFILE_NCU_CACHE="false"
ANALYZE_FD="false"
ANALYZE_REDUNDANCY="false"
ANALYZE_CONTRIBUTION="false"
ANALYZE_ADJACENT="false"
ANALYZE_SMOOTHNESS="false"
PARSE_ONLY="false"
MODEL="all"

for arg in "$@"; do
    case $arg in
        --memory)
            PROFILE_MEMORY="true"
            ;;
        --hbm-traffic)
            PROFILE_HBM_TRAFFIC="true"
            ;;
        --ncu)
            PROFILE_NCU="true"
            ;;
        --ncu-cache)
            PROFILE_NCU="true"
            PROFILE_NCU_CACHE="true"
            ;;
        --analyze-fd)
            ANALYZE_FD="true"
            ;;
        --analyze-redundancy)
            ANALYZE_REDUNDANCY="true"
            ;;
        --analyze-contribution)
            ANALYZE_CONTRIBUTION="true"
            ;;
        --analyze-adjacent)
            ANALYZE_ADJACENT="true"
            ;;
        --analyze-smoothness)
            ANALYZE_SMOOTHNESS="true"
            ;;
        --parse-only)
            PARSE_ONLY="true"
            ;;
        *)
            MODEL="$arg"
            ;;
    esac
done

# Define ncu metrics based on options
NCU_METRICS="dram__bytes_read.sum,dram__bytes_write.sum"
if [ "$PROFILE_NCU_CACHE" = "true" ]; then
    # Add L1 and L2 cache metrics
    NCU_METRICS="$NCU_METRICS,l1tex__t_sector_hit_rate,l1tex__t_sectors,lts__t_sector_hit_rate,lts__t_sectors"
fi

# Activate conda environment

# Function to print section header
print_header() {
    echo ""
    echo "================================================================"
    echo "  $1"
    echo "================================================================"
    echo ""
}

# Function to generate 4-phase summary
generate_phase_summary() {
    local model_name=$1
    local output_file=$2
    
    if [ -f "$output_file" ]; then
        echo ""
        cat "$output_file" | python "$BASE_DIR/scripts/generate_phase_summary.py" "$model_name"
    fi
}

# Function to run TransPlat
run_transplat() {
    print_header "Running TransPlat Timing Analysis"
    cd "$BASE_DIR"
    
    # Output file for capturing results
    local OUTPUT_FILE="$PROFILE_DIR/transplat_output.txt"
    local NCU_CSV="$PROFILE_DIR/transplat_ncu_metrics.csv"
    
    # Parse-only mode: parse existing CSV files
    if [ "$PARSE_ONLY" = "true" ]; then
        echo "[Parse-only mode: parsing existing data files]"
        
        # Parse ncu CSV if exists
        if [ -f "$NCU_CSV" ]; then
            echo "Parsing ncu CSV: $NCU_CSV"
            python "$BASE_DIR/scripts/profile_hbm_by_stage.py" --ncu-csv "$NCU_CSV" --model transplat | tee "$PROFILE_DIR/transplat_hbm_parsed.txt"
            
            # If we also have a previous output file with timing, combine them
            if [ -f "$OUTPUT_FILE" ]; then
                echo ""
                echo "Combining with timing data from: $OUTPUT_FILE"
                # Create combined output for phase summary
                cat "$OUTPUT_FILE" "$PROFILE_DIR/transplat_hbm_parsed.txt" > "$PROFILE_DIR/transplat_combined.txt"
                generate_phase_summary "transplat" "$PROFILE_DIR/transplat_combined.txt"
            else
                generate_phase_summary "transplat" "$PROFILE_DIR/transplat_hbm_parsed.txt"
            fi
        elif [ -f "$OUTPUT_FILE" ]; then
            echo "No ncu CSV found, parsing output file: $OUTPUT_FILE"
            generate_phase_summary "transplat" "$OUTPUT_FILE"
        else
            echo "Error: No data files found."
            echo "  Expected: $NCU_CSV or $OUTPUT_FILE"
            echo "Run without --parse-only first to generate data."
        fi
        return
    fi
    
    # Check if memory profiling is enabled
    MEMORY_FLAG=""
    if [ "$PROFILE_MEMORY" = "true" ]; then
        MEMORY_FLAG="test.profile_memory=true"
        echo "[Memory Allocation Profiling Enabled]"
    fi
    
    # Check if feature-depth analysis is enabled
    ANALYZE_FD_FLAG=""
    if [ "$ANALYZE_FD" = "true" ]; then
        ANALYZE_FD_FLAG="test.analyze_feature_depth=true"
        echo "[Feature-Depth Correlation Analysis Enabled]"
    fi
    
    # Check if Gaussian redundancy analysis is enabled
    ANALYZE_REDUNDANCY_FLAG=""
    if [ "$ANALYZE_REDUNDANCY" = "true" ]; then
        ANALYZE_REDUNDANCY_FLAG="+test.analyze_gaussian_redundancy=true"
        echo "[Gaussian Redundancy Analysis Enabled (Challenge 2)]"
    fi
    
    # Check if adjacent Gaussian analysis is enabled
    ANALYZE_ADJACENT_FLAG=""
    if [ "$ANALYZE_ADJACENT" = "true" ]; then
        ANALYZE_ADJACENT_FLAG="+test.analyze_adjacent_gaussians=true"
        echo "[Adjacent Gaussian Similarity Analysis Enabled]"
    fi
    
    # Check if Gaussian smoothness analysis is enabled
    ANALYZE_SMOOTHNESS_FLAG=""
    if [ "$ANALYZE_SMOOTHNESS" = "true" ]; then
        ANALYZE_SMOOTHNESS_FLAG="+test.analyze_gaussian_smoothness=true"
        echo "[Gaussian Smoothness/Variability Analysis Enabled]"
    fi
    
    # Check if Gaussian contribution analysis is enabled
    if [ "$ANALYZE_CONTRIBUTION" = "true" ]; then
        echo "[Gaussian Contribution Analysis Enabled]"
        echo ""
        python "$BASE_DIR/scripts/analyze_gaussian_contribution.py" \
            --model transplat \
            --checkpoint "$BASE_DIR/checkpoints/re10k.ckpt" \
            --num-samples 5 \
            --thresholds 0.01 0.1 0.2 0.3 0.5 0.7 0.9
        return
    fi
    
    # Build the Python command
    PYTHON_CMD="python -m src.main +experiment=re10k mode=test dataset/view_sampler=evaluation test.compute_scores=true dataset.test_len=1 test.eval_time_skip_steps=0 checkpointing.load=checkpoints/re10k.ckpt $MEMORY_FLAG $ANALYZE_FD_FLAG $ANALYZE_REDUNDANCY_FLAG $ANALYZE_ADJACENT_FLAG $ANALYZE_SMOOTHNESS_FLAG"
    
    # Check if HBM traffic profiling is enabled
    if [ "$PROFILE_HBM_TRAFFIC" = "true" ]; then
        echo "[HBM Traffic Profiling Enabled via nsys]"
        nsys profile --stats=true --force-overwrite=true -o "$PROFILE_DIR/transplat_hbm_profile" $PYTHON_CMD 2>&1 | tee "$OUTPUT_FILE"
        echo ""
        echo "Analyzing HBM traffic (memcpy)..."
        python scripts/analyze_hbm_traffic.py "$PROFILE_DIR/transplat_hbm_profile.sqlite" | tee -a "$OUTPUT_FILE"
    elif [ "$PROFILE_NCU" = "true" ]; then
        echo "[HBM DRAM Traffic Profiling Enabled via ncu]"
        if [ "$PROFILE_NCU_CACHE" = "true" ]; then
            echo "Running ncu with cache metrics (this takes ~20+ minutes)..."
        else
            echo "Running ncu for kernel DRAM traffic (this takes ~10 minutes)..."
        fi
        /usr/local/cuda/bin/ncu --metrics $NCU_METRICS \
            --csv --log-file "$PROFILE_DIR/transplat_ncu_metrics.csv" \
            --target-processes all \
            --nvtx \
            --replay-mode kernel \
            $PYTHON_CMD 2>&1 | tee "$OUTPUT_FILE"
        
        echo ""
        echo "Analyzing HBM DRAM traffic by stage..."
        python scripts/profile_hbm_by_stage.py --ncu-csv "$PROFILE_DIR/transplat_ncu_metrics.csv" --model transplat | tee -a "$OUTPUT_FILE"
    else
        $PYTHON_CMD 2>&1 | tee "$OUTPUT_FILE"
    fi
    
    # Generate 4-phase summary
    generate_phase_summary "transplat" "$OUTPUT_FILE"
}

# Function to run MVSplat
run_mvsplat() {
    print_header "Running MVSplat Timing Analysis"
    cd "$BASE_DIR/mvsplat-main"
    
    # Output file for capturing results
    local OUTPUT_FILE="$PROFILE_DIR/mvsplat_output.txt"
    local NCU_CSV="$PROFILE_DIR/mvsplat_ncu_metrics.csv"
    
    # Parse-only mode: parse existing CSV files
    if [ "$PARSE_ONLY" = "true" ]; then
        echo "[Parse-only mode: parsing existing data files]"
        
        # Parse ncu CSV if exists
        if [ -f "$NCU_CSV" ]; then
            echo "Parsing ncu CSV: $NCU_CSV"
            python "$BASE_DIR/scripts/profile_hbm_by_stage.py" --ncu-csv "$NCU_CSV" --model mvsplat | tee "$PROFILE_DIR/mvsplat_hbm_parsed.txt"
            
            # If we also have a previous output file with timing, combine them
            if [ -f "$OUTPUT_FILE" ]; then
                echo ""
                echo "Combining with timing data from: $OUTPUT_FILE"
                cat "$OUTPUT_FILE" "$PROFILE_DIR/mvsplat_hbm_parsed.txt" > "$PROFILE_DIR/mvsplat_combined.txt"
                generate_phase_summary "mvsplat" "$PROFILE_DIR/mvsplat_combined.txt"
            else
                generate_phase_summary "mvsplat" "$PROFILE_DIR/mvsplat_hbm_parsed.txt"
            fi
        elif [ -f "$OUTPUT_FILE" ]; then
            echo "No ncu CSV found, parsing output file: $OUTPUT_FILE"
            generate_phase_summary "mvsplat" "$OUTPUT_FILE"
        else
            echo "Error: No data files found."
            echo "  Expected: $NCU_CSV or $OUTPUT_FILE"
            echo "Run without --parse-only first to generate data."
        fi
        return
    fi
    
    # Check if memory profiling is enabled
    MEMORY_FLAG=""
    if [ "$PROFILE_MEMORY" = "true" ]; then
        MEMORY_FLAG="test.profile_memory=true"
        echo "[Memory Profiling Enabled]"
    fi
    
    # Check if feature-depth analysis is enabled
    ANALYZE_FD_FLAG=""
    if [ "$ANALYZE_FD" = "true" ]; then
        ANALYZE_FD_FLAG="test.analyze_feature_depth=true"
        echo "[Feature-Depth Correlation Analysis Enabled]"
    fi
    
    # Check if Gaussian redundancy analysis is enabled
    ANALYZE_REDUNDANCY_FLAG=""
    if [ "$ANALYZE_REDUNDANCY" = "true" ]; then
        ANALYZE_REDUNDANCY_FLAG="+test.analyze_gaussian_redundancy=true"
        echo "[Gaussian Redundancy Analysis Enabled (Challenge 2)]"
    fi
    
    # Check if Gaussian contribution analysis is enabled
    if [ "$ANALYZE_CONTRIBUTION" = "true" ]; then
        echo "[Gaussian Contribution Analysis Enabled]"
        echo ""
        python "$BASE_DIR/scripts/analyze_gaussian_contribution.py" \
            --model mvsplat \
            --checkpoint "$BASE_DIR/mvsplat-main/checkpoints/re10k.ckpt" \
            --num-samples 5 \
            --thresholds 0.01 0.1 0.2 0.3 0.5 0.7 0.9
        return
    fi
    
    # Check if Gaussian smoothness analysis is enabled
    ANALYZE_SMOOTHNESS_FLAG=""
    if [ "$ANALYZE_SMOOTHNESS" = "true" ]; then
        ANALYZE_SMOOTHNESS_FLAG="+test.analyze_gaussian_smoothness=true"
        echo "[Gaussian Smoothness/Variability Analysis Enabled]"
    fi
    
    PYTHON_CMD="python -m src.main +experiment=re10k mode=test dataset/view_sampler=evaluation test.compute_scores=true dataset.test_len=1 test.eval_time_skip_steps=0 checkpointing.load=checkpoints/re10k.ckpt $MEMORY_FLAG $ANALYZE_FD_FLAG $ANALYZE_REDUNDANCY_FLAG $ANALYZE_SMOOTHNESS_FLAG"
    
    if [ "$PROFILE_HBM_TRAFFIC" = "true" ]; then
        echo "[HBM Traffic Profiling Enabled via nsys]"
        nsys profile --stats=true --force-overwrite=true -o "$PROFILE_DIR/mvsplat_hbm_profile" $PYTHON_CMD 2>&1 | tee "$OUTPUT_FILE"
        echo ""
        echo "Analyzing HBM traffic (memcpy)..."
        python "$BASE_DIR/scripts/analyze_hbm_traffic.py" "$PROFILE_DIR/mvsplat_hbm_profile.sqlite" | tee -a "$OUTPUT_FILE"
    elif [ "$PROFILE_NCU" = "true" ]; then
        echo "[HBM DRAM Traffic Profiling Enabled via ncu]"
        if [ "$PROFILE_NCU_CACHE" = "true" ]; then
            echo "Running ncu with cache metrics (this takes ~20+ minutes)..."
        else
            echo "Running ncu for kernel DRAM traffic (this takes ~10 minutes)..."
        fi
        /usr/local/cuda/bin/ncu --metrics $NCU_METRICS \
            --csv --log-file "$PROFILE_DIR/mvsplat_ncu_metrics.csv" \
            --target-processes all \
            --nvtx \
            --replay-mode kernel \
            $PYTHON_CMD 2>&1 | tee "$OUTPUT_FILE"
        
        echo ""
        echo "Analyzing HBM DRAM traffic by stage..."
        python "$BASE_DIR/scripts/profile_hbm_by_stage.py" --ncu-csv "$PROFILE_DIR/mvsplat_ncu_metrics.csv" --model mvsplat | tee -a "$OUTPUT_FILE"
    else
        $PYTHON_CMD 2>&1 | tee "$OUTPUT_FILE"
    fi
    
    # Generate 4-phase summary
    generate_phase_summary "mvsplat" "$OUTPUT_FILE"
}

# Function to run DepthSplat
run_depthsplat() {
    print_header "Running DepthSplat Timing Analysis"
    cd "$BASE_DIR/depthsplat-main"
    
    # Output file for capturing results
    local OUTPUT_FILE="$PROFILE_DIR/depthsplat_output.txt"
    local NCU_CSV="$PROFILE_DIR/depthsplat_ncu_metrics.csv"
    
    # Parse-only mode: parse existing CSV files
    if [ "$PARSE_ONLY" = "true" ]; then
        echo "[Parse-only mode: parsing existing data files]"
        
        # Parse ncu CSV if exists
        if [ -f "$NCU_CSV" ]; then
            echo "Parsing ncu CSV: $NCU_CSV"
            python "$BASE_DIR/scripts/profile_hbm_by_stage.py" --ncu-csv "$NCU_CSV" --model depthsplat | tee "$PROFILE_DIR/depthsplat_hbm_parsed.txt"
            
            # If we also have a previous output file with timing, combine them
            if [ -f "$OUTPUT_FILE" ]; then
                echo ""
                echo "Combining with timing data from: $OUTPUT_FILE"
                cat "$OUTPUT_FILE" "$PROFILE_DIR/depthsplat_hbm_parsed.txt" > "$PROFILE_DIR/depthsplat_combined.txt"
                generate_phase_summary "depthsplat" "$PROFILE_DIR/depthsplat_combined.txt"
            else
                generate_phase_summary "depthsplat" "$PROFILE_DIR/depthsplat_hbm_parsed.txt"
            fi
        elif [ -f "$OUTPUT_FILE" ]; then
            echo "No ncu CSV found, parsing output file: $OUTPUT_FILE"
            generate_phase_summary "depthsplat" "$OUTPUT_FILE"
        else
            echo "Error: No data files found."
            echo "  Expected: $NCU_CSV or $OUTPUT_FILE"
            echo "Run without --parse-only first to generate data."
        fi
        return
    fi
    
    # Check if memory profiling is enabled
    MEMORY_FLAG=""
    if [ "$PROFILE_MEMORY" = "true" ]; then
        MEMORY_FLAG="test.profile_memory=true"
        echo "[Memory Profiling Enabled]"
    fi
    
    # Check if feature-depth analysis is enabled
    ANALYZE_FD_FLAG=""
    if [ "$ANALYZE_FD" = "true" ]; then
        ANALYZE_FD_FLAG="test.analyze_feature_depth=true"
        echo "[Feature-Depth Correlation Analysis Enabled]"
    fi
    
    # Check if Gaussian redundancy analysis is enabled
    ANALYZE_REDUNDANCY_FLAG=""
    if [ "$ANALYZE_REDUNDANCY" = "true" ]; then
        ANALYZE_REDUNDANCY_FLAG="+test.analyze_gaussian_redundancy=true"
        echo "[Gaussian Redundancy Analysis Enabled (Challenge 2)]"
    fi
    
    # Check if Gaussian contribution analysis is enabled
    if [ "$ANALYZE_CONTRIBUTION" = "true" ]; then
        echo "[Gaussian Contribution Analysis Enabled]"
        echo ""
        python "$BASE_DIR/scripts/analyze_gaussian_contribution.py" \
            --model depthsplat \
            --checkpoint "$BASE_DIR/depthsplat-main/checkpoints/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth" \
            --num-samples 5 \
            --thresholds 0.01 0.1 0.2 0.3 0.5 0.7 0.9
        return
    fi
    
    # Check if Gaussian smoothness analysis is enabled
    ANALYZE_SMOOTHNESS_FLAG=""
    if [ "$ANALYZE_SMOOTHNESS" = "true" ]; then
        ANALYZE_SMOOTHNESS_FLAG="+test.analyze_gaussian_smoothness=true"
        echo "[Gaussian Smoothness/Variability Analysis Enabled]"
    fi
    
    PYTHON_CMD="python -m src.main +experiment=re10k dataset.test_chunk_interval=1 model.encoder.upsample_factor=4 model.encoder.lowest_feature_resolution=4 checkpointing.pretrained_model=checkpoints/depthsplat-gs-small-re10k-256x256-view2-cfeab6b1.pth mode=test dataset/view_sampler=evaluation dataset.test_len=1 test.eval_time_skip_steps=0 $MEMORY_FLAG $ANALYZE_FD_FLAG $ANALYZE_REDUNDANCY_FLAG $ANALYZE_SMOOTHNESS_FLAG"
    
    if [ "$PROFILE_HBM_TRAFFIC" = "true" ]; then
        echo "[HBM Traffic Profiling Enabled via nsys]"
        nsys profile --stats=true --force-overwrite=true -o "$PROFILE_DIR/depthsplat_hbm_profile" $PYTHON_CMD 2>&1 | tee "$OUTPUT_FILE"
        echo ""
        echo "Analyzing HBM traffic (memcpy)..."
        python "$BASE_DIR/scripts/analyze_hbm_traffic.py" "$PROFILE_DIR/depthsplat_hbm_profile.sqlite" | tee -a "$OUTPUT_FILE"
    elif [ "$PROFILE_NCU" = "true" ]; then
        echo "[HBM DRAM Traffic Profiling Enabled via ncu]"
        if [ "$PROFILE_NCU_CACHE" = "true" ]; then
            echo "Running ncu with cache metrics (this takes ~20+ minutes)..."
        else
            echo "Running ncu for kernel DRAM traffic (this takes ~10 minutes)..."
        fi
        /usr/local/cuda/bin/ncu --metrics $NCU_METRICS \
            --csv --log-file "$PROFILE_DIR/depthsplat_ncu_metrics.csv" \
            --target-processes all \
            --nvtx \
            --replay-mode kernel \
            $PYTHON_CMD 2>&1 | tee "$OUTPUT_FILE"
        
        echo ""
        echo "Analyzing HBM DRAM traffic by stage..."
        python "$BASE_DIR/scripts/profile_hbm_by_stage.py" --ncu-csv "$PROFILE_DIR/depthsplat_ncu_metrics.csv" --model depthsplat | tee -a "$OUTPUT_FILE"
    else
        $PYTHON_CMD 2>&1 | tee "$OUTPUT_FILE"
    fi
    
    # Generate 4-phase summary
    generate_phase_summary "depthsplat" "$OUTPUT_FILE"
}

# Function to run PixelSplat
run_pixelsplat() {
    print_header "Running PixelSplat Timing Analysis"
    cd "$BASE_DIR/pixelsplat"
    
    # Output file for capturing results
    local OUTPUT_FILE="$PROFILE_DIR/pixelsplat_output.txt"
    local NCU_CSV="$PROFILE_DIR/pixelsplat_ncu_metrics.csv"
    
    # Parse-only mode: parse existing CSV files
    if [ "$PARSE_ONLY" = "true" ]; then
        echo "[Parse-only mode: parsing existing data files]"
        
        # Parse ncu CSV if exists
        if [ -f "$NCU_CSV" ]; then
            echo "Parsing ncu CSV: $NCU_CSV"
            python "$BASE_DIR/scripts/profile_hbm_by_stage.py" --ncu-csv "$NCU_CSV" --model pixelsplat | tee "$PROFILE_DIR/pixelsplat_hbm_parsed.txt"
            
            # If we also have a previous output file with timing, combine them
            if [ -f "$OUTPUT_FILE" ]; then
                echo ""
                echo "Combining with timing data from: $OUTPUT_FILE"
                cat "$OUTPUT_FILE" "$PROFILE_DIR/pixelsplat_hbm_parsed.txt" > "$PROFILE_DIR/pixelsplat_combined.txt"
                generate_phase_summary "pixelsplat" "$PROFILE_DIR/pixelsplat_combined.txt"
            else
                generate_phase_summary "pixelsplat" "$PROFILE_DIR/pixelsplat_hbm_parsed.txt"
            fi
        elif [ -f "$OUTPUT_FILE" ]; then
            echo "No ncu CSV found, parsing output file: $OUTPUT_FILE"
            generate_phase_summary "pixelsplat" "$OUTPUT_FILE"
        else
            echo "Error: No data files found."
            echo "  Expected: $NCU_CSV or $OUTPUT_FILE"
            echo "Run without --parse-only first to generate data."
        fi
        return
    fi
    
    # Check if memory profiling is enabled
    MEMORY_FLAG=""
    if [ "$PROFILE_MEMORY" = "true" ]; then
        MEMORY_FLAG="test.profile_memory=true"
        echo "[Memory Profiling Enabled]"
    fi
    
    PYTHON_CMD="python -m src.main +experiment=re10k mode=test dataset/view_sampler=evaluation dataset.overfit_to_scene=5aca87f95a9412c6 dataset.test_len=1 $MEMORY_FLAG"
    
    if [ "$PROFILE_HBM_TRAFFIC" = "true" ]; then
        echo "[HBM Traffic Profiling Enabled via nsys]"
        nsys profile --stats=true --force-overwrite=true -o "$PROFILE_DIR/pixelsplat_hbm_profile" $PYTHON_CMD 2>&1 | tee "$OUTPUT_FILE"
        echo ""
        echo "Analyzing HBM traffic (memcpy)..."
        python "$BASE_DIR/scripts/analyze_hbm_traffic.py" "$PROFILE_DIR/pixelsplat_hbm_profile.sqlite" | tee -a "$OUTPUT_FILE"
    elif [ "$PROFILE_NCU" = "true" ]; then
        echo "[HBM DRAM Traffic Profiling Enabled via ncu]"
        if [ "$PROFILE_NCU_CACHE" = "true" ]; then
            echo "Running ncu with cache metrics (this takes ~20+ minutes)..."
        else
            echo "Running ncu for kernel DRAM traffic (this takes ~10 minutes)..."
        fi
        /usr/local/cuda/bin/ncu --metrics $NCU_METRICS \
            --csv --log-file "$PROFILE_DIR/pixelsplat_ncu_metrics.csv" \
            --target-processes all \
            --nvtx \
            --replay-mode kernel \
            $PYTHON_CMD 2>&1 | tee "$OUTPUT_FILE"
        
        echo ""
        echo "Analyzing HBM DRAM traffic by stage..."
        python "$BASE_DIR/scripts/profile_hbm_by_stage.py" --ncu-csv "$PROFILE_DIR/pixelsplat_ncu_metrics.csv" --model pixelsplat | tee -a "$OUTPUT_FILE"
    else
        $PYTHON_CMD 2>&1 | tee "$OUTPUT_FILE"
    fi
    
    # Generate 4-phase summary
    generate_phase_summary "pixelsplat" "$OUTPUT_FILE"
}

# Function to run all models
run_all() {
    run_transplat
    run_mvsplat
    run_depthsplat
    run_pixelsplat
}

# Main script
case $MODEL in
    transplat)
        run_transplat
        ;;
    mvsplat)
        run_mvsplat
        ;;
    depthsplat)
        run_depthsplat
        ;;
    pixelsplat)
        run_pixelsplat
        ;;
    all)
        run_all
        ;;
    *)
        echo "Usage: $0 [model] [--memory] [--hbm-traffic] [--ncu] [--ncu-cache] [--analyze-fd] [--analyze-redundancy] [--analyze-contribution] [--parse-only]"
        echo ""
        echo "Models:"
        echo "  all         - Run all four models (default)"
        echo "  transplat   - Run TransPlat only"
        echo "  mvsplat     - Run MVSplat only"
        echo "  depthsplat  - Run DepthSplat only"
        echo "  pixelsplat  - Run PixelSplat only (note: pixelsplat does not support --analyze-fd/--analyze-redundancy)"
        echo ""
        echo "Options:"
        echo "  --memory            - Enable memory allocation profiling (all models)"
        echo "  --hbm-traffic       - Enable HBM traffic via nsys memcpy analysis (all models)"
        echo "  --ncu               - Enable kernel-level DRAM traffic via Nsight Compute (all models, slow)"
        echo "  --ncu-cache         - Enable ncu with L1/L2 cache hit rate metrics (even slower)"
        echo "  --analyze-fd        - Analyze feature-depth correlation (transplat, mvsplat, depthsplat)"
        echo "  --analyze-redundancy - Analyze Gaussian redundancy for Challenge 2 (transplat, mvsplat, depthsplat)"
        echo "  --analyze-contribution - Analyze Gaussian contribution by opacity threshold (shows HBM saving potential)"
        echo "  --analyze-adjacent  - Analyze adjacent pixel Gaussian similarity (mergeable estimation)"
        echo "  --analyze-smoothness - Analyze Gaussian variability and generate 2D/3D heatmaps"
        echo "  --parse-only        - Only parse existing output files (no model execution)"
        echo ""
        echo "Profile outputs are saved to: $PROFILE_DIR"
        echo ""
        echo "Examples:"
        echo "  $0 mvsplat --memory         # Run MVSplat with memory profiling"
        echo "  $0 transplat --hbm-traffic  # Run TransPlat with nsys memcpy analysis"
        echo "  $0 transplat --ncu          # Run TransPlat with ncu DRAM profiling (accurate)"
        echo "  $0 all --ncu                # Run all models with ncu profiling (takes hours)"
        echo "  $0 all                      # Run all models (timing only)"
        echo "  $0 all --parse-only         # Parse existing outputs and show 4-phase summaries"
        echo "  $0 mvsplat --parse-only     # Parse existing MVSplat output only"
        exit 1
        ;;
esac

print_header "All Tests Completed"
