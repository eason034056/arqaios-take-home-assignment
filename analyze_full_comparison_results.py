"""
Analyze Full Comparison Experiment Results
Compares: (2 split strategies) × (2 centering options) × (3 models) = 12 experiments
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
plt.rcParams['figure.figsize'] = (14, 10)

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
    """Parse all 12 experiment results."""
    results = []
    
    models = ['mlp', 'cnn1d', 'pointnet']
    split_strategies = ['grouped', 'stratified']  # 先 grouped（無洩漏），再 stratified（有洩漏）
    center_settings = ['with_center', 'no_center']  # 先 with_center，再 no_center
    
    for split in split_strategies:
        for center in center_settings:
            for model in models:
                dir_name = f"{split}_{center}_{model}"
                model_dir = Path(results_dir) / dir_name
                
                if model_dir.exists():
                    metrics = extract_best_results(model_dir)
                    if metrics:
                        results.append({
                            'model': model.upper() + (' (k=3)' if model == 'cnn1d' else ''),
                            'split_strategy': 'With Leakage' if split == 'stratified' else 'No Leakage',
                            'split_type': split,
                            'centering': 'With Centering' if center == 'with_center' else 'No Centering',
                            'center_type': center,
                            'val_acc': metrics['val_acc'],
                            'val_loss': metrics['val_loss'],
                            'best_epoch': metrics['epoch']
                        })
    
    return pd.DataFrame(results)

def create_comparison_plots(df, output_dir):
    """Create comprehensive visualization plots."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Plot 1: Grouped bar chart - All conditions
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    for idx, split in enumerate(['No Leakage', 'With Leakage']):
        ax = axes[idx]
        df_subset = df[df['split_strategy'] == split]
        
        # Pivot for grouped bar chart
        pivot = df_subset.pivot_table(
            index='model', 
            columns='centering', 
            values='val_acc'
        )
        
        pivot.plot(kind='bar', ax=ax, width=0.7, color=['#FF6B6B', '#4ECDC4'])
        
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('Validation Accuracy (%)', fontsize=12, fontweight='bold')
        ax.set_title(f'{split}\nImpact of Centering on Model Performance', 
                     fontsize=13, fontweight='bold')
        ax.legend(title='Data Processing', fontsize=10)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for container in ax.containers:
            ax.bar_label(container, fmt='%.2f%%', padding=3, fontsize=9)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'split_strategy_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'split_strategy_comparison.png'}")
    
    # Plot 2: Heatmap showing all combinations
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Create pivot table for heatmap
    heatmap_data = df.pivot_table(
        index=['split_strategy', 'centering'],
        columns='model',
        values='val_acc'
    )
    
    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=70, vmin=40, vmax=100, ax=ax, cbar_kws={'label': 'Accuracy (%)'})
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Configuration', fontsize=12, fontweight='bold')
    ax.set_title('Performance Heatmap: All Configurations', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'performance_heatmap.png'}")
    
    # Plot 3: Data Leakage Impact
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Calculate average accuracy for each combination
    summary = df.groupby(['split_strategy', 'centering'])['val_acc'].mean().reset_index()
    
    # Reorder: No Leakage first, then With Leakage
    order = [
        ('No Leakage', 'With Centering'),
        ('No Leakage', 'No Centering'),
        ('With Leakage', 'With Centering'),
        ('With Leakage', 'No Centering')
    ]
    summary['sort_key'] = summary.apply(lambda x: order.index((x['split_strategy'], x['centering'])), axis=1)
    summary = summary.sort_values('sort_key').reset_index(drop=True)
    summary['config'] = summary['split_strategy'] + '\n' + summary['centering']
    
    colors = ['#4ECDC4', '#9FE6E6', '#FF6B6B', '#FFB6B6']  # Grouped 用綠色系，Stratified 用紅色系
    bars = ax.bar(range(len(summary)), summary['val_acc'], color=colors, alpha=0.8)
    ax.set_xticks(range(len(summary)))
    ax.set_xticklabels(summary['config'], fontsize=10)
    ax.set_ylabel('Average Validation Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Average Performance Across All Models', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, summary['val_acc'])):
        ax.text(i, val + 1, f'{val:.2f}%', ha='center', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'average_performance.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'average_performance.png'}")
    
    # Plot 4: Model-wise comparison across conditions
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, model in enumerate(df['model'].unique()):
        ax = axes[idx]
        df_model = df[df['model'] == model]
        
        # Create grouped bar chart
        df_model['config'] = df_model['split_type'] + '_' + df_model['center_type']
        
        x_pos = np.arange(4)
        configs = ['grouped_with_center', 'grouped_no_center',
                   'stratified_with_center', 'stratified_no_center']
        config_labels = ['Group+Center', 'Group+NoCenter',
                        'Strat+Center', 'Strat+NoCenter']
        
        accs = [df_model[df_model['config'] == cfg]['val_acc'].values[0] 
                if len(df_model[df_model['config'] == cfg]) > 0 else 0
                for cfg in configs]
        
        colors_map = ['#4ECDC4', '#FF6B6B', '#FFB6B6', '#FFA6A6']  # Grouped 用綠色系，Stratified 用紅色系
        bars = ax.bar(x_pos, accs, color=colors_map, alpha=0.7)
        
        ax.set_xticks(x_pos)
        ax.set_xticklabels(config_labels, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Validation Accuracy (%)', fontsize=11, fontweight='bold')
        ax.set_title(f'{model}\nPerformance Across Conditions', 
                     fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, accs)):
            ax.text(i, val + 1, f'{val:.1f}%', ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_wise_comparison.png', dpi=300, bbox_inches='tight')
    print(f"Saved: {output_dir / 'model_wise_comparison.png'}")

def generate_summary_report(df, output_dir):
    """Generate comprehensive text summary report."""
    output_dir = Path(output_dir)
    report_path = output_dir / 'summary_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 100 + "\n")
        f.write("FULL COMPARISON EXPERIMENT SUMMARY REPORT\n")
        f.write("=" * 100 + "\n\n")
        
        f.write("Experiment Design:\n")
        f.write("-" * 100 + "\n")
        f.write("This comprehensive experiment tests:\n\n")
        f.write("Split Strategies (執行順序):\n")
        f.write("  1. Grouped (NO data leakage): Same mesh samples stay together [先測]\n")
        f.write("  2. Stratified (WITH data leakage): Same mesh samples in train/val/test [後測]\n\n")
        f.write("Centering Options:\n")
        f.write("  1. With Centering: Center at origin (normalize_center=true)\n")
        f.write("  2. No Centering: Keep original coordinates (normalize_center=false)\n\n")
        f.write("Models:\n")
        f.write("  1. MLP: Multi-Layer Perceptron\n")
        f.write("  2. CNN1D (k=3): 1D CNN with kernel_size=3\n")
        f.write("  3. PointNet: PointNet with T-Net\n\n")
        f.write("Total: 2 × 2 × 3 = 12 experiments\n")
        f.write("=" * 100 + "\n\n")
        
        # Detailed results table
        f.write("DETAILED RESULTS:\n")
        f.write("-" * 100 + "\n")
        f.write(df.to_string(index=False))
        f.write("\n\n")
        
        # Key findings
        f.write("KEY FINDINGS:\n")
        f.write("=" * 100 + "\n\n")
        
        # 1. Data leakage impact
        f.write("1. IMPACT OF DATA LEAKAGE:\n")
        f.write("-" * 100 + "\n")
        for model in df['model'].unique():
            strat_avg = df[(df['model'] == model) & (df['split_type'] == 'stratified')]['val_acc'].mean()
            group_avg = df[(df['model'] == model) & (df['split_type'] == 'grouped')]['val_acc'].mean()
            diff = strat_avg - group_avg
            
            f.write(f"\n{model}:\n")
            f.write(f"  With Leakage (Stratified): {strat_avg:.2f}%\n")
            f.write(f"  No Leakage (Grouped): {group_avg:.2f}%\n")
            f.write(f"  Difference: {diff:+.2f}%\n")
            if diff > 20:
                f.write(f"  → Severe data leakage effect! Performance drops {diff:.1f}% without leakage\n")
            elif diff > 5:
                f.write(f"  → Moderate data leakage effect\n")
            else:
                f.write(f"  → Minimal data leakage effect\n")
        
        # 2. Centering impact
        f.write("\n2. IMPACT OF CENTERING:\n")
        f.write("-" * 100 + "\n")
        for split in ['grouped', 'stratified']:
            f.write(f"\n{split.upper()}:\n")
            for model in df['model'].unique():
                with_center = df[(df['model'] == model) & (df['split_type'] == split) & 
                                (df['center_type'] == 'with_center')]['val_acc']
                no_center = df[(df['model'] == model) & (df['split_type'] == split) & 
                              (df['center_type'] == 'no_center')]['val_acc']
                
                if len(with_center) > 0 and len(no_center) > 0:
                    diff = with_center.values[0] - no_center.values[0]
                    f.write(f"  {model}: {diff:+.2f}% ")
                    if diff > 0:
                        f.write("(Centering helps)\n")
                    else:
                        f.write("(Centering hurts)\n")
        
        # 3. Best configurations
        f.write("\n3. BEST CONFIGURATIONS:\n")
        f.write("-" * 100 + "\n")
        
        # Overall best
        best_row = df.loc[df['val_acc'].idxmax()]
        f.write(f"\nOverall Best:\n")
        f.write(f"  Model: {best_row['model']}\n")
        f.write(f"  Split: {best_row['split_strategy']}\n")
        f.write(f"  Centering: {best_row['centering']}\n")
        f.write(f"  Accuracy: {best_row['val_acc']:.2f}%\n")
        
        # Best without leakage
        df_no_leak = df[df['split_type'] == 'grouped']
        best_no_leak = df_no_leak.loc[df_no_leak['val_acc'].idxmax()]
        f.write(f"\nBest Without Data Leakage:\n")
        f.write(f"  Model: {best_no_leak['model']}\n")
        f.write(f"  Centering: {best_no_leak['centering']}\n")
        f.write(f"  Accuracy: {best_no_leak['val_acc']:.2f}%\n")
        
        # 4. Model rankings
        f.write("\n4. MODEL RANKINGS (No Leakage):\n")
        f.write("-" * 100 + "\n")
        model_avg = df[df['split_type'] == 'grouped'].groupby('model')['val_acc'].mean().sort_values(ascending=False)
        for rank, (model, acc) in enumerate(model_avg.items(), 1):
            f.write(f"  {rank}. {model}: {acc:.2f}%\n")
        
        f.write("\n" + "=" * 100 + "\n")
    
    print(f"Saved: {report_path}")

def main():
    results_dir = Path('results/full_comparison_experiment')
    
    if not results_dir.exists():
        print(f"Error: Results directory not found: {results_dir}")
        print("Please run the experiment first: bash run_full_comparison_experiment.sh")
        return
    
    print("Analyzing full comparison experiment results...")
    print("-" * 100)
    
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
    
    print("\n" + "=" * 100)
    print("Analysis complete!")
    print(f"All outputs saved to: {results_dir}")
    print("=" * 100)

if __name__ == '__main__':
    main()

