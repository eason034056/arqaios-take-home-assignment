#!/bin/bash
# Run the three models with grouped split and with_center settings
# Run: split_type=grouped, center_type=with_center, models=mlp+cnn1d+pointnet

set -e  # Exit immediately if any command fails

echo "================================================================"
echo "Running ML Pipeline: Grouped Split + With Centering"
echo "================================================================"
echo ""
echo "Configuration:"
echo "  split_type: grouped (no data leakage, tests true generalization)"
echo "  center_type: with_center (perform centering preprocessing)"
echo "  models: MLP, CNN1D, PointNet"
echo ""
echo "Expected time: ~1-2 hours"
echo ""

# ========================================
# Step 1: Backup original config file
# ========================================
echo "Step 1: Backup config file..."
# This command: copy config.yaml to config.yaml.backup
# cp = copy
# config.yaml = source file
# config.yaml.backup = target file
cp config.yaml config.yaml.backup
echo "✓ Config file backed up to config.yaml.backup"
echo ""

# ========================================
# Step 2: Set experiment parameters
# ========================================
echo "Step 2: Set experiment parameters..."

# 2.1: Set split strategy to grouped
# sed = stream editor, used to edit file content
# -i '' = in-place editing (for Mac, '' is required)
# "s/old/new/" = substitute command
# strategy: .* = match "strategy: " followed by any content
# Replace with "strategy: \"grouped\""
echo "  Setting split_strategy = grouped..."
sed -i '' "s/strategy: .*/strategy: \"grouped\"  # Experiment setting/" config.yaml

# 2.2: Set normalize_center to true
# Enables centering for point cloud
echo "  Setting normalize_center = true..."
sed -i '' "s/normalize_center: .*/normalize_center: true  # Experiment setting/" config.yaml

# 2.3: Ensure normalize_scale is false
# Keep absolute body size, do not scale
echo "  Setting normalize_scale = false..."
sed -i '' "s/normalize_scale: .*/normalize_scale: false  # Keep absolute body size/" config.yaml

echo "✓ Configuration set:"
echo "    split_strategy = grouped"
echo "    normalize_center = true"
echo "    normalize_scale = false"
echo ""

# ========================================
# Step 3: Clean up old processed data
# ========================================
echo "Step 3: Clean up old processed data..."
# rm = remove
# -f = force (do not prompt)
# data/processed/faust_pc_*.npz = match all .npz files
rm -f data/processed/faust_pc_*.npz
echo "✓ Old data cleaned. Will reprocess."
echo ""

# ========================================
# Step 4: Create results directory
# ========================================
echo "Step 4: Create results directory..."
RESULTS_DIR="results/grouped_with_center_experiment"
# mkdir = make directory
# -p = create parent directories if not exist
mkdir -p $RESULTS_DIR
echo "✓ Results will be saved to: $RESULTS_DIR"
echo ""

# ========================================
# Step 5: Train the three models
# ========================================
echo "Step 5: Start model training..."
echo ""

# Define a function to train a model
# This avoids repeating code for each model
train_model() {
    # $1 = first argument to function (model name)
    local model=$1
    
    echo "================================================================"
    echo "Training: $model"
    echo "================================================================"
    
    # Clean previous checkpoints and logs for this model
    # Make sure each run is fresh
    echo "  Cleaning old checkpoints and logs..."
    rm -rf results/checkpoints/${model}/*
    rm -rf results/tensorboard/${model}/*
    
    # Run the training script
    # python = Python interpreter
    # src/train.py = path to training script
    # --config = specify config file
    # --model = specify model type to train
    echo "  Starting training..."
    python src/train.py --config config.yaml --model $model
    
    # After training, copy results to experiment directory
    local result_dir="$RESULTS_DIR/${model}"
    mkdir -p $result_dir
    
    # cp -r = copy recursively (include subdirectories)
    echo "  Saving results..."
    cp -r results/checkpoints/${model} $result_dir/checkpoints
    cp -r results/tensorboard/${model} $result_dir/tensorboard
    
    echo "✓ $model training complete! Results saved to: $result_dir"
    echo ""
}

# 5.1: Train MLP model
echo "--- 5.1: Train MLP Baseline ---"
train_model "mlp"

# 5.2: Train CNN1D model
echo "--- 5.2: Train CNN1D model ---"
train_model "cnn1d"

# 5.3: Train PointNet model
echo "--- 5.3: Train Tiny PointNet model ---"
train_model "pointnet"

# ========================================
# Step 6: Restore original config
# ========================================
echo "Step 6: Restore original config..."
# mv = move/rename
# Rename backup file back to original filename
mv config.yaml.backup config.yaml
echo "✓ Config file restored"
echo ""

# ========================================
# Done
# ========================================
echo "================================================================"
echo "✓ All training complete!"
echo "================================================================"
echo ""
echo "Results summary:"
echo "  Results directory: $RESULTS_DIR"
echo ""
echo "  Trained models:"
echo "    1. MLP Baseline      → $RESULTS_DIR/mlp/"
echo "    2. CNN1D Model       → $RESULTS_DIR/cnn1d/"
echo "    3. Tiny PointNet     → $RESULTS_DIR/pointnet/"
echo ""
echo "  Each model directory contains:"
echo "    - checkpoints/    → model weights (.pth)"
echo "    - tensorboard/    → training logs (for visualization)"
echo ""
echo "Processed data:"
echo "  data/processed/faust_pc_grouped.npz"
echo ""
echo "Next steps:"
echo ""
echo "  1. View TensorBoard training curves:"
echo "     tensorboard --logdir=$RESULTS_DIR"
echo ""
echo "  2. Evaluate model performance:"
echo "     python src/evaluate.py --model mlp --checkpoint $RESULTS_DIR/mlp/checkpoints/model_best.pth"
echo "     python src/evaluate.py --model cnn1d --checkpoint $RESULTS_DIR/cnn1d/checkpoints/model_best.pth"
echo "     python src/evaluate.py --model pointnet --checkpoint $RESULTS_DIR/pointnet/checkpoints/model_best.pth"
echo ""
echo "  3. Compare all three models:"
echo "     python src/evaluate.py --compare \\"
echo "       --models mlp cnn1d pointnet \\"
echo "       --checkpoints \\"
echo "         $RESULTS_DIR/mlp/checkpoints/model_best.pth \\"
echo "         $RESULTS_DIR/cnn1d/checkpoints/model_best.pth \\"
echo "         $RESULTS_DIR/pointnet/checkpoints/model_best.pth"
echo ""

