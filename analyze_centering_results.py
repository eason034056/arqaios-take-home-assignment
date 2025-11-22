"""
Analyze Centering Experiment Results
Compares performance of MLP, CNN1D (kernel=3), and PointNet with/without centering.
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

def extract_best_results(model_dir):
    """Extract best validation and test accuracy from a model directory."""
    checkpoint_dir = Path(model_dir) / 'checkpoints'
    
    # Try to find model_best.pth and extract metadata
    best_model = checkpoint_dir / 'model_best.pth'
    
    if not best_model.exists():
        print(f"Warning: {best_model} not found")
        return None
    
    # Read checkpoint
    import torch
    try:
        checkpoint = torch.load(best_model, map_location='cpu')
        
        # Extract metrics
        val_acc = checkpoint.get('val_acc', 0.0)
        val_loss = checkpoint.get('val_loss', 0.0)
        epoch = checkpoint.get('epoch', 0)
        
        return {
            'val_acc': val_acc,
            'val_loss': val_loss,
            'epoch': epoch
        }
    except Exception as e:
        print(f"Error loading {best_model}: {e}")
        return None

def parse_results_directory(results_dir):
    """Parse all experiment results."""
    results = []
    
    models = ['mlp', 'cnn1d', 'pointnet']
    center_settings = ['no_center', 'with_center']
    
    for center in center_settings:
        for model in models:
            dir_name = f"{center}_{model}"
            model_dir = Path(results_dir) / dir_name
            
            if model_dir.exists():
                metrics = extract_best_results(model_dir)
                if metrics:
                    results.append({
                        'model': model.upper() + (' (k=3)' if model == 'cnn1d' else ''),
                        'centering': 'With Centering' if center == 'with_center' else 'No Centering',
                        'center_type': center,
                        'val_acc': metrics['val_acc'],
                        'val_loss': metrics['val_loss'],
                        'best_epoch': metrics['epoch']
                    })
    
    return pd.DataFrame(results)

def create_comparison_plots(df, output_dir):
    """Create visualization plots comparing results."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Accuracy comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Pivot data for grouped bar chart
    pivot_acc = df.pivot(index='model', columns='centering', values='val_acc')
    pivot_acc.plot(kind='bar', ax=ax, width=0.7, color=['#FF6B6B', '#4ECDC4'])
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Impact of Centering on Model Performance\n(CNN1D kernel_size=3)', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Data Processing', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', padding=3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'accuracy_comparison.png'}")
    
    # Plot 2: Loss comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot_loss = df.pivot(index='model', columns='centering', values='val_loss')
    pivot_loss.plot(kind='bar', ax=ax, width=0.7, color=['#FF6B6B', '#4ECDC4'])
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
    ax.set_title('Validation Loss with Different Centering Strategies\n(CNN1D kernel_size=3)', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Data Processing', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f', padding=3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'loss_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'loss_comparison.png'}")
    
    # Plot 3: Performance improvement/degradation
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Calculate difference (with_center - no_center)
    improvements = []
    for model in df['model'].unique():
        no_center_acc = df[(df['model'] == model) & (df['center_type'] == 'no_center')]['val_acc'].values
        with_center_acc = df[(df['model'] == model) & (df['center_type'] == 'with_center')]['val_acc'].values
        
        if len(no_center_acc) > 0 and len(with_center_acc) > 0:
            diff = with_center_acc[0] - no_center_acc[0]
            improvements.append({'model': model, 'improvement': diff})
    
    if improvements:
        imp_df = pd.DataFrame(improvements)
        colors = ['green' if x > 0 else 'red' for x in imp_df['improvement']]
        
        ax.bar(range(len(imp_df)), imp_df['improvement'], color=colors, alpha=0.7)
        ax.set_xticks(range(len(imp_df)))
        ax.set_xticklabels(imp_df['model'], rotation=45, ha='right')
        ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Accuracy Change (%)', fontsize=12, fontweight='bold')
        ax.set_title('Performance Change: With Centering vs No Centering\n(Positive = Centering Helps)', 
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (model, imp) in enumerate(zip(imp_df['model'], imp_df['improvement'])):
            ax.text(i, imp + (0.5 if imp > 0 else -0.5), f'{imp:.2f}%', 
                    ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'improvement_comparison.png'}")

def generate_summary_report(df, output_dir):
    """Generate a text summary report."""
    output_dir = Path(output_dir)
    report_path = output_dir / 'summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("CENTERING EXPERIMENT SUMMARY REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Experiment Overview:\n")
        f.write("-" * 80 + "\n")
        f.write("This experiment compares two data processing strategies:\n")
        f.write("1. No Centering: Keep original coordinates and body size\n")
        f.write("2. With Centering: Center at origin, keep body size (no scaling)\n\n")
        
        f.write("Models Tested:\n")
        f.write("- MLP: Multi-Layer Perceptron\n")
        f.write("- CNN1D (k=3): 1D CNN with kernel_size=3 (local patterns)\n")
        f.write("- PointNet: PointNet with T-Net spatial alignment\n")
        f.write("=" * 80 + "\n\n")
        
        # Results table
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        # Calculate improvements
        f.write("PERFORMANCE CHANGE (With Centering - No Centering):\n")
        f.write("-" * 80 + "\n")
        for model in df['model'].unique():
            no_center = df[(df['model'] == model) & (df['center_type'] == 'no_center')]
            with_center = df[(df['model'] == model) & (df['center_type'] == 'with_center')]
            
            if not no_center.empty and not with_center.empty:
                acc_diff = with_center['val_acc'].values[0] - no_center['val_acc'].values[0]
                loss_diff = with_center['val_loss'].values[0] - no_center['val_loss'].values[0]
                
                f.write(f"\n{model}:\n")
                f.write(f"  Accuracy change: {acc_diff:+.2f}%\n")
                f.write(f"  Loss change: {loss_diff:+.4f}\n")
                
                if acc_diff > 0:
                    f.write(f"  ✓ Centering HELPS (accuracy improved)\n")
                else:
                    f.write(f"  ✗ Centering HURTS (accuracy decreased)\n")
        
        f.write("\n" + "=" * 80 + "\n")
        f.write("CONCLUSIONS:\n")
        f.write("-" * 80 + "\n")
        
        # Find best overall
        best_row = df.loc[df['val_acc'].idxmax()]
        f.write(f"\nBest overall performance:\n")
        f.write(f"  Model: {best_row['model']}\n")
        f.write(f"  Centering: {best_row['centering']}\n")
        f.write(f"  Validation Accuracy: {best_row['val_acc']:.2f}%\n")
        f.write(f"  Validation Loss: {best_row['val_loss']:.4f}\n")
        
        # Average improvement
        avg_improvement = 0
        count = 0
        for model in df['model'].unique():
            no_center = df[(df['model'] == model) & (df['center_type'] == 'no_center')]
            with_center = df[(df['model'] == model) & (df['center_type'] == 'with_center')]
            if not no_center.empty and not with_center.empty:
                acc_diff = with_center['val_acc'].values[0] - no_center['val_acc'].values[0]
                avg_improvement += acc_diff
                count += 1
        
        if count > 0:
            avg_improvement /= count
            f.write(f"\nAverage accuracy change across all models: {avg_improvement:+.2f}%\n")
            if avg_improvement > 0:
                f.write("→ Overall, centering IMPROVES performance\n")
            else:
                f.write("→ Overall, centering DEGRADES performance\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Saved: {report_path}")

def main():
    results_dir = Path('results/centering_experiment')
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Please run the experiment first: bash run_centering_experiment.sh")
        return
    
    print("Analyzing centering experiment results...")
    print("-" * 80)
    
    # Parse results
    df = parse_results_directory(results_dir)
    
    if df.empty:
        print("Error: No results found in the experiment directory")
        return
    
    print(f"\nFound {len(df)} experiment results")
    print("\nResults DataFrame:")
    print(df.to_string(index=False))
    print()
    
    # Create visualizations
    print("\nGenerating comparison plots...")
    create_comparison_plots(df, results_dir)
    
    # Generate summary report
    print("\nGenerating summary report...")
    generate_summary_report(df, results_dir)
    
    print("\n" + "=" * 80)
    print("Analysis complete!")
    print(f"All outputs saved to: {results_dir}")
    print("=" * 80)

if __name__ == '__main__':
    main()

