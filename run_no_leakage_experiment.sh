#!/bin/bash
# No Data Leakage Experiment (Quick Version)
# Only tests GROUPED split - true generalization ability
# Tests: 2 centering options × 3 models = 6 training runs

set -e  # Exit on error

echo "================================================================"
echo "NO DATA LEAKAGE EXPERIMENT (真實泛化能力測試)"
echo "================================================================"
echo ""
echo "This experiment will test TRUE generalization ability using:"
echo ""
echo "Split Strategy:"
echo "  ✓ Grouped (NO data leakage) - samples from same mesh stay together"
echo "  ✗ Stratified (NOT included) - use run_full_comparison_experiment.sh for that"
echo ""
echo "Centering Options:"
echo "  1. With Centering (normalize_center=true)"
echo "  2. No Centering (normalize_center=false)"
echo ""
echo "Models:"
echo "  1. MLP"
echo "  2. CNN1D (kernel_size=3)"
echo "  3. PointNet"
echo ""
echo "Total: 2 × 3 = 6 training runs"
echo "Estimated time: ~2-3 hours"
echo ""
read -p "Press Enter to start..."

# Create results directory
RESULTS_DIR="results/no_leakage_experiment"
mkdir -p $RESULTS_DIR

# Function to backup config
backup_config() {
    cp config.yaml config.yaml.backup
}

# Function to restore config
restore_config() {
    if [ -f config.yaml.backup ]; then
        mv config.yaml.backup config.yaml
    fi
}

# Function to set split strategy
set_split_strategy() {
    local strategy=$1
    sed -i '' "s/strategy: .*/strategy: \"$strategy\"  # Experiment setting/" config.yaml
}

# Function to set normalize_center
set_normalize_center() {
    local value=$1
    sed -i '' "s/normalize_center: .*/normalize_center: $value  # Experiment setting/" config.yaml
}

# Function to clean data for reprocessing
clean_data() {
    echo "Cleaning processed data..."
    rm -f data/processed/faust_pc_grouped.npz
}

# Function to train a model
train_model() {
    local model=$1
    local center_setting=$2
    
    echo ""
    echo "================================================================"
    echo "Training: $model"
    echo "  Split: GROUPED (no leakage)"
    echo "  Centering: $center_setting"
    echo "================================================================"
    
    # Clean checkpoints and logs for this model
    rm -rf results/checkpoints/${model}/*
    rm -rf results/tensorboard/${model}/*
    
    # Train
    python src/train.py --config config.yaml --model $model
    
    # Copy results
    local result_dir="$RESULTS_DIR/grouped_${center_setting}_${model}"
    mkdir -p $result_dir
    cp -r results/checkpoints/${model} $result_dir/checkpoints
    cp -r results/tensorboard/${model} $result_dir/tensorboard
    
    echo "Results saved to: $result_dir"
}

# Backup original config
backup_config

# Set to grouped split and keep normalize_scale false
sed -i '' "s/normalize_scale: .*/normalize_scale: false  # Keep absolute body size/" config.yaml
set_split_strategy "grouped"

# ========================================
# Experiment: GROUPED SPLIT ONLY
# ========================================
echo ""
echo "================================================================"
echo "GROUPED SPLIT (NO DATA LEAKAGE)"
echo "Testing TRUE generalization ability - unseen poses"
echo "================================================================"

# 1: Grouped + With Centering
echo ""
echo "--- Phase 1: Grouped + With Centering ---"
set_normalize_center "true"
clean_data

train_model "mlp" "with_center"
train_model "cnn1d" "with_center"
train_model "pointnet" "with_center"

# 2: Grouped + No Centering
echo ""
echo "--- Phase 2: Grouped + No Centering ---"
set_normalize_center "false"
clean_data

train_model "mlp" "no_center"
train_model "cnn1d" "no_center"
train_model "pointnet" "no_center"

# Restore original config
restore_config

# ========================================
# Generate summary
# ========================================
echo ""
echo "================================================================"
echo "EXPERIMENT COMPLETED!"
echo "================================================================"
echo ""
echo "Results are saved in: $RESULTS_DIR"
echo ""
echo "Directory structure:"
echo "  grouped_with_center_mlp/         - With Centering + MLP"
echo "  grouped_with_center_cnn1d/       - With Centering + CNN1D"
echo "  grouped_with_center_pointnet/    - With Centering + PointNet"
echo "  grouped_no_center_mlp/           - No Centering + MLP"
echo "  grouped_no_center_cnn1d/         - No Centering + CNN1D"
echo "  grouped_no_center_pointnet/      - No Centering + PointNet"
echo ""
echo "Processed dataset saved:"
echo "  data/processed/faust_pc_grouped.npz - No data leakage (真實泛化能力)"
echo ""
echo "Expected results:"
echo "  - Validation accuracy: 50-75% (真實的泛化能力)"
echo "  - Much lower than stratified split (which would be 99%+)"
echo "  - PointNet likely performs best due to T-Net"
echo ""
echo "To view TensorBoard logs:"
echo "  tensorboard --logdir=$RESULTS_DIR"
echo ""
echo "Next step: Run the analysis script to compare results"
echo "  python analyze_no_leakage_results.py"
echo ""
echo "Want to compare with data leakage? Run:"
echo "  bash run_full_comparison_experiment.sh"

