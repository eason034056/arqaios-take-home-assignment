#!/bin/bash
# Full Comparison Experiment
# Tests: (2 split strategies) × (2 centering options) × (3 models) = 12 training runs

set -e  # Exit on error

echo "================================================================"
echo "FULL COMPARISON EXPERIMENT"
echo "================================================================"
echo ""
echo "This experiment will test:"
echo ""
echo "Split Strategies (執行順序):"
echo "  1. Grouped (NO data leakage) - 先測真實泛化能力"
echo "  2. Stratified (WITH data leakage) - 再測有洩漏情況"
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
echo "Total: 2 × 2 × 3 = 12 training runs"
echo "Estimated time: ~4-6 hours"
echo ""
read -p "Press Enter to start..."

# Create results directory
RESULTS_DIR="results/full_comparison_experiment"
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
    rm -f data/processed/faust_pc_*.npz
}

# Function to train a model
train_model() {
    local model=$1
    local split_strategy=$2
    local center_setting=$3
    
    echo ""
    echo "================================================================"
    echo "Training: $model"
    echo "  Split: $split_strategy"
    echo "  Centering: $center_setting"
    echo "================================================================"
    
    # Clean checkpoints and logs for this model
    rm -rf results/checkpoints/${model}/*
    rm -rf results/tensorboard/${model}/*
    
    # Train
    python src/train.py --config config.yaml --model $model
    
    # Copy results
    local result_dir="$RESULTS_DIR/${split_strategy}_${center_setting}_${model}"
    mkdir -p $result_dir
    cp -r results/checkpoints/${model} $result_dir/checkpoints
    cp -r results/tensorboard/${model} $result_dir/tensorboard
    
    echo "Results saved to: $result_dir"
}

# Backup original config
backup_config

# Ensure normalize_scale is false for all experiments
sed -i '' "s/normalize_scale: .*/normalize_scale: false  # Keep absolute body size/" config.yaml

# ========================================
# Experiment Group 1: GROUPED (NO data leakage) - 先測真實泛化能力
# ========================================
echo ""
echo "================================================================"
echo "EXPERIMENT GROUP 1: GROUPED SPLIT (NO DATA LEAKAGE)"
echo "Testing TRUE generalization ability - unseen poses"
echo "================================================================"

set_split_strategy "grouped"

# 1.1: Grouped + With Centering
echo ""
echo "--- 1.1: Grouped + With Centering ---"
set_normalize_center "true"
clean_data

train_model "mlp" "grouped" "with_center"
train_model "cnn1d" "grouped" "with_center"
train_model "pointnet" "grouped" "with_center"

# 1.2: Grouped + No Centering
echo ""
echo "--- 1.2: Grouped + No Centering ---"
set_normalize_center "false"
clean_data

train_model "mlp" "grouped" "no_center"
train_model "cnn1d" "grouped" "no_center"
train_model "pointnet" "grouped" "no_center"

# ========================================
# Experiment Group 2: STRATIFIED (WITH data leakage) - 再測有洩漏情況
# ========================================
echo ""
echo "================================================================"
echo "EXPERIMENT GROUP 2: STRATIFIED SPLIT (WITH DATA LEAKAGE)"
echo "Testing with data leakage - seen pose variants"
echo "================================================================"

set_split_strategy "stratified"

# 2.1: Stratified + With Centering
echo ""
echo "--- 2.1: Stratified + With Centering ---"
set_normalize_center "true"
clean_data

train_model "mlp" "stratified" "with_center"
train_model "cnn1d" "stratified" "with_center"
train_model "pointnet" "stratified" "with_center"

# 2.2: Stratified + No Centering
echo ""
echo "--- 2.2: Stratified + No Centering ---"
set_normalize_center "false"
clean_data

train_model "mlp" "stratified" "no_center"
train_model "cnn1d" "stratified" "no_center"
train_model "pointnet" "stratified" "no_center"

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
echo "Directory structure (執行順序):"
echo ""
echo "Group 1 - NO LEAKAGE (先執行):"
echo "  grouped_with_center_mlp/         - Grouped + Centering + MLP"
echo "  grouped_with_center_cnn1d/       - Grouped + Centering + CNN1D"
echo "  grouped_with_center_pointnet/    - Grouped + Centering + PointNet"
echo "  grouped_no_center_mlp/           - Grouped + No Centering + MLP"
echo "  grouped_no_center_cnn1d/         - Grouped + No Centering + CNN1D"
echo "  grouped_no_center_pointnet/      - Grouped + No Centering + PointNet"
echo ""
echo "Group 2 - WITH LEAKAGE (後執行):"
echo "  stratified_with_center_mlp/      - Stratified + Centering + MLP"
echo "  stratified_with_center_cnn1d/    - Stratified + Centering + CNN1D"
echo "  stratified_with_center_pointnet/ - Stratified + Centering + PointNet"
echo "  stratified_no_center_mlp/        - Stratified + No Centering + MLP"
echo "  stratified_no_center_cnn1d/      - Stratified + No Centering + CNN1D"
echo "  stratified_no_center_pointnet/   - Stratified + No Centering + PointNet"
echo ""
echo "Processed datasets saved:"
echo "  data/processed/faust_pc_grouped.npz     - No data leakage (真實泛化能力)"
echo "  data/processed/faust_pc_stratified.npz  - With data leakage (虛高準確率)"
echo ""
echo "To view TensorBoard logs:"
echo "  tensorboard --logdir=$RESULTS_DIR"
echo ""
echo "Next step: Run the analysis script to compare results"
echo "  python analyze_full_comparison_results.py"

