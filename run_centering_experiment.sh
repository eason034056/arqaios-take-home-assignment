#!/bin/bash
# Centering Comparison Experiment
# Tests impact of centering on MLP, CNN1D (kernel=3), and PointNet

set -e  # Exit on error

echo "=================================="
echo "Centering Comparison Experiment"
echo "=================================="
echo ""
echo "This experiment will:"
echo "1. Train 3 models WITH centering (normalize_center=true, normalize_scale=false)"
echo "2. Train 3 models WITHOUT centering (normalize_center=false, normalize_scale=false)"
echo "3. CNN1D uses kernel_size=3 for local pattern recognition"
echo "4. Compare results"
echo ""
echo "Total: 6 training runs"
echo "Estimated time: ~2-3 hours"
echo ""
read -p "Press Enter to start..."

# Create results directory
RESULTS_DIR="results/centering_experiment"
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

# Function to update normalize_center in config
set_normalize_center() {
    local value=$1
    # Use sed to replace the line
    sed -i '' "s/normalize_center: .*/normalize_center: $value  # Experiment setting/" config.yaml
}

# Function to clean data for reprocessing
clean_data() {
    echo "Cleaning processed data..."
    rm -f data/processed/faust_pc.npz
}

# Function to train a model
train_model() {
    local model=$1
    local center_setting=$2
    
    echo ""
    echo "=================================="
    echo "Training: $model (normalize_center=$center_setting)"
    echo "=================================="
    
    # Clean checkpoints and logs for this model
    rm -rf results/checkpoints/${model}/*
    rm -rf results/tensorboard/${model}/*
    
    # Train
    python src/train.py --config config.yaml --model $model
    
    # Copy results
    local result_dir="$RESULTS_DIR/${center_setting}_${model}"
    mkdir -p $result_dir
    cp -r results/checkpoints/${model} $result_dir/checkpoints
    cp -r results/tensorboard/${model} $result_dir/tensorboard
    
    echo "Results saved to: $result_dir"
}

# Backup original config
backup_config

# Ensure normalize_scale is false for both experiments
sed -i '' "s/normalize_scale: .*/normalize_scale: false  # Keep absolute body size/" config.yaml

# ========================================
# Experiment 1: WITH centering
# ========================================
echo ""
echo "========================================"
echo "EXPERIMENT 1: normalize_center=true"
echo "Center at origin, keep absolute body size"
echo "========================================"

set_normalize_center "true"
clean_data

# Train all three models
train_model "mlp" "with_center"
clean_data  # Clean between models to ensure same preprocessing

train_model "cnn1d" "with_center"
clean_data

train_model "pointnet" "with_center"

# ========================================
# Experiment 2: WITHOUT centering
# ========================================
echo ""
echo "========================================"
echo "EXPERIMENT 2: normalize_center=false"
echo "Keep original coordinates and body size"
echo "========================================"

set_normalize_center "false"
clean_data

# Train all three models
train_model "mlp" "no_center"
clean_data

train_model "cnn1d" "no_center"
clean_data

train_model "pointnet" "no_center"

# Restore original config
restore_config

# ========================================
# Generate summary
# ========================================
echo ""
echo "========================================"
echo "EXPERIMENT COMPLETED!"
echo "========================================"
echo ""
echo "Results are saved in: $RESULTS_DIR"
echo ""
echo "Directory structure:"
echo "  with_center_mlp/      - MLP with centering"
echo "  with_center_cnn1d/    - CNN1D (kernel=3) with centering"
echo "  with_center_pointnet/ - PointNet with centering"
echo "  no_center_mlp/        - MLP without centering"
echo "  no_center_cnn1d/      - CNN1D (kernel=3) without centering"
echo "  no_center_pointnet/   - PointNet without centering"
echo ""
echo "Note: CNN1D uses kernel_size=3 for local pattern recognition"
echo ""
echo "To view TensorBoard logs:"
echo "  tensorboard --logdir=$RESULTS_DIR"
echo ""
echo "Next step: Run the analysis script to compare results"
echo "  python analyze_centering_results.py"

