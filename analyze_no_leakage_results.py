"""
Analyze No Data Leakage Experiment Results
Focuses on TRUE generalization ability with grouped split
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
    best_model = checkpoint_dir / 'model_best.pth'
    
    if not best_model.exists():
        print(f"Warning: {best_model} not found")
        return None
    
    import torch
    try:
        checkpoint = torch.load(best_model, map_location='cpu')
        return {
            'val_acc': checkpoint.get('val_acc', 0.0),
            'val_loss': checkpoint.get('val_loss', 0.0),
            'epoch': checkpoint.get('epoch', 0)
        }
    except Exception as e:
        print(f"Error loading {best_model}: {e}")
        return None

def parse_results_directory(results_dir):
    """Parse experiment results (grouped split only)."""
    results = []
    
    models = ['mlp', 'cnn1d', 'pointnet']
    center_settings = ['no_center', 'with_center']
    
    for center in center_settings:
        for model in models:
            dir_name = f"grouped_{center}_{model}"
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
    """Create visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Bar chart comparison
    fig, ax = plt.subplots(figsize=(10, 6))
    
    pivot = df.pivot(index='model', columns='centering', values='val_acc')
    pivot.plot(kind='bar', ax=ax, width=0.7, color=['#FF6B6B', '#4ECDC4'])
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('TRUE Generalization Ability (No Data Leakage)\nGrouped Split - Unseen Poses', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Data Processing', fontsize=10)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2f%%', padding=3)
    
    # Add reference line for random guess
    ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random Guess (10%)')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'accuracy_comparison.png'}")
    
    # Plot 2: Model ranking
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Average across centering options
    avg_acc = df.groupby('model')['val_acc'].mean().sort_values(ascending=False)
    colors = ['#4ECDC4' if val >= 70 else '#FFB86C' if val >= 50 else '#FF6B6B' for val in avg_acc.values]
    
    bars = ax.bar(range(len(avg_acc)), avg_acc.values, color=colors, alpha=0.7)
    ax.set_xticks(range(len(avg_acc)))
    ax.set_xticklabels(avg_acc.index, fontsize=11)
    ax.set_ylabel('Average Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Ranking (Average Performance)\nNo Data Leakage', 
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels and rank
    for i, (bar, val) in enumerate(zip(bars, avg_acc.values)):
        ax.text(i, val + 1, f'#{i+1}\n{val:.2f}%', ha='center', fontweight='bold')
    
    # Add reference line
    ax.axhline(y=10, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random (10%)')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_ranking.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'model_ranking.png'}")
    
    # Plot 3: Impact of centering
    fig, ax = plt.subplots(figsize=(10, 6))
    
    improvements = []
    for model in df['model'].unique():
        no_center = df[(df['model'] == model) & (df['center_type'] == 'no_center')]['val_acc']
        with_center = df[(df['model'] == model) & (df['center_type'] == 'with_center')]['val_acc']
        
        if len(no_center) > 0 and len(with_center) > 0:
            diff = with_center.values[0] - no_center.values[0]
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
        ax.set_title('Impact of Centering on TRUE Generalization\n(Positive = Centering Helps)', 
                     fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (model, imp) in enumerate(zip(imp_df['model'], imp_df['improvement'])):
            ax.text(i, imp + (0.5 if imp > 0 else -0.5), f'{imp:+.2f}%', 
                    ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'centering_impact.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'centering_impact.png'}")

def generate_summary_report(df, output_dir):
    """Generate text summary report."""
    output_dir = Path(output_dir)
    report_path = output_dir / 'summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("NO DATA LEAKAGE EXPERIMENT SUMMARY\n")
        f.write("TRUE GENERALIZATION ABILITY TEST\n")
        f.write("=" * 80 + "\n\n")
        
        f.write("Experiment Overview:\n")
        f.write("-" * 80 + "\n")
        f.write("Split Strategy: GROUPED (No Data Leakage)\n")
        f.write("  - Same mesh samples stay together in one split\n")
        f.write("  - Val/Test see completely unseen poses\n")
        f.write("  - Tests TRUE pose generalization ability\n\n")
        
        f.write("Centering Options:\n")
        f.write("  1. With Centering: Center at origin (normalize_center=true)\n")
        f.write("  2. No Centering: Keep original coordinates (normalize_center=false)\n\n")
        
        f.write("Models: MLP, CNN1D (k=3), PointNet\n")
        f.write("=" * 80 + "\n\n")
        
        # Results table
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 80 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        # Model ranking
        f.write("MODEL RANKING (Average Accuracy):\n")
        f.write("-" * 80 + "\n")
        avg_acc = df.groupby('model')['val_acc'].mean().sort_values(ascending=False)
        for rank, (model, acc) in enumerate(avg_acc.items(), 1):
            f.write(f"  {rank}. {model}: {acc:.2f}%\n")
        
        # Best configuration
        f.write("\nBEST CONFIGURATION:\n")
        f.write("-" * 80 + "\n")
        best_row = df.loc[df['val_acc'].idxmax()]
        f.write(f"  Model: {best_row['model']}\n")
        f.write(f"  Centering: {best_row['centering']}\n")
        f.write(f"  Accuracy: {best_row['val_acc']:.2f}%\n")
        f.write(f"  Loss: {best_row['val_loss']:.4f}\n")
        f.write(f"  Epoch: {best_row['best_epoch']}\n")
        
        # Centering impact
        f.write("\nIMPACT OF CENTERING:\n")
        f.write("-" * 80 + "\n")
        for model in df['model'].unique():
            no_center = df[(df['model'] == model) & (df['center_type'] == 'no_center')]['val_acc']
            with_center = df[(df['model'] == model) & (df['center_type'] == 'with_center')]['val_acc']
            
            if len(no_center) > 0 and len(with_center) > 0:
                diff = with_center.values[0] - no_center.values[0]
                f.write(f"\n{model}:\n")
                f.write(f"  No Centering: {no_center.values[0]:.2f}%\n")
                f.write(f"  With Centering: {with_center.values[0]:.2f}%\n")
                f.write(f"  Difference: {diff:+.2f}%\n")
                if abs(diff) > 5:
                    f.write(f"  → Centering has SIGNIFICANT impact\n")
                elif abs(diff) > 2:
                    f.write(f"  → Centering has MODERATE impact\n")
                else:
                    f.write(f"  → Centering has MINIMAL impact\n")
        
        # Key findings
        f.write("\n" + "=" * 80 + "\n")
        f.write("KEY FINDINGS:\n")
        f.write("-" * 80 + "\n\n")
        
        best_model = avg_acc.index[0]
        best_acc = avg_acc.values[0]
        worst_model = avg_acc.index[-1]
        worst_acc = avg_acc.values[-1]
        
        f.write(f"1. Best Model: {best_model} ({best_acc:.2f}%)\n")
        f.write(f"   - Shows best generalization to unseen poses\n\n")
        
        f.write(f"2. Worst Model: {worst_model} ({worst_acc:.2f}%)\n")
        f.write(f"   - Struggles with pose variations\n\n")
        
        f.write(f"3. Performance Gap: {best_acc - worst_acc:.2f}%\n")
        if best_acc - worst_acc > 20:
            f.write(f"   - Large gap indicates model architecture matters significantly\n")
        else:
            f.write(f"   - Small gap indicates similar generalization ability\n")
        
        f.write("\n4. These are TRUE generalization scores (no data leakage)\n")
        f.write("   - Much lower than stratified split would show (~99%)\n")
        f.write("   - But represents REAL-WORLD performance on unseen poses\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Saved: {report_path}")

def main():
    results_dir = Path('results/no_leakage_experiment')
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Please run the experiment first: bash run_no_leakage_experiment.sh")
        return
    
    print("Analyzing no data leakage experiment results...")
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
    print("\nThese results show TRUE generalization ability (no data leakage)")
    print("Want to compare with data leakage? Run: bash run_full_comparison_experiment.sh")

if __name__ == '__main__':
    main()

