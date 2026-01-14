#!/usr/bin/env python3
"""
Visualization comparison program: Compare prediction results with reference data
Compare grid_pred_case0.csv with merged_fibers_output_z_m0p005.txt
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.interpolate import griddata
import matplotlib.colors as mcolors

# Use default font settings
plt.rcParams['axes.unicode_minus'] = False

def load_reference_data(filepath):
    """Load reference data (merged_fibers_output_z_m0p005.txt)"""
    df = pd.read_csv(filepath, sep='\t')
    return df

def load_prediction_data(filepath):
    """Load prediction data (grid_pred_case0.csv)"""
    df = pd.read_csv(filepath)
    return df

def interpolate_to_common_grid(ref_df, pred_df, grid_resolution=50):
    """Interpolate both datasets to a common grid for comparison"""
    # Determine common x, y range
    x_min = max(ref_df['x'].min(), pred_df['x'].min())
    x_max = min(ref_df['x'].max(), pred_df['x'].max())
    y_min = max(ref_df['y'].min(), pred_df['y'].min())
    y_max = min(ref_df['y'].max(), pred_df['y'].max())
    
    # Create common grid
    xi = np.linspace(x_min, x_max, grid_resolution)
    yi = np.linspace(y_min, y_max, grid_resolution)
    Xi, Yi = np.meshgrid(xi, yi)
    
    return Xi, Yi, x_min, x_max, y_min, y_max

def create_comparison_plot(ref_df, pred_df, output_path='comparison_visualization.png'):
    """Create comparison visualization"""
    
    # Set dark theme style
    plt.style.use('dark_background')
    
    fig = plt.figure(figsize=(20, 16))
    fig.patch.set_facecolor('#0a0a0f')
    
    # Create grid layout
    gs = GridSpec(3, 4, figure=fig, hspace=0.3, wspace=0.3)
    
    # Define colormaps
    cmap_ref = plt.cm.viridis
    cmap_pred = plt.cm.plasma
    cmap_diff = plt.cm.RdBu_r
    
    # Get common grid
    Xi, Yi, x_min, x_max, y_min, y_max = interpolate_to_common_grid(ref_df, pred_df)
    
    # Variables to compare
    variables = ['u', 'v', 'w']
    var_labels = ['U (X-displacement)', 'V (Y-displacement)', 'W (Z-displacement)']
    
    for idx, (var, label) in enumerate(zip(variables, var_labels)):
        # Reference data scatter plot
        ax1 = fig.add_subplot(gs[idx, 0])
        ref_points = ref_df[['x', 'y']].values
        ref_values = ref_df[var].values
        
        scatter1 = ax1.scatter(ref_df['x'], ref_df['y'], c=ref_values, 
                               cmap=cmap_ref, s=15, alpha=0.8, edgecolors='none')
        ax1.set_xlim(x_min - 0.01, x_max + 0.01)
        ax1.set_ylim(y_min - 0.01, y_max + 0.01)
        ax1.set_xlabel('X', fontsize=10, color='#aaaaaa')
        ax1.set_ylabel('Y', fontsize=10, color='#aaaaaa')
        ax1.set_title(f'Reference - {label}', fontsize=11, color='#00ff88', fontweight='bold')
        ax1.set_facecolor('#0f0f18')
        ax1.tick_params(colors='#888888')
        for spine in ax1.spines.values():
            spine.set_color('#333344')
        cbar1 = plt.colorbar(scatter1, ax=ax1, shrink=0.8)
        cbar1.ax.tick_params(colors='#888888')
        
        # Prediction data scatter plot
        ax2 = fig.add_subplot(gs[idx, 1])
        pred_points = pred_df[['x', 'y']].values
        pred_values = pred_df[var].values
        
        scatter2 = ax2.scatter(pred_df['x'], pred_df['y'], c=pred_values, 
                               cmap=cmap_pred, s=3, alpha=0.6, edgecolors='none')
        ax2.set_xlim(x_min - 0.01, x_max + 0.01)
        ax2.set_ylim(y_min - 0.01, y_max + 0.01)
        ax2.set_xlabel('X', fontsize=10, color='#aaaaaa')
        ax2.set_ylabel('Y', fontsize=10, color='#aaaaaa')
        ax2.set_title(f'Prediction - {label}', fontsize=11, color='#ff6688', fontweight='bold')
        ax2.set_facecolor('#0f0f18')
        ax2.tick_params(colors='#888888')
        for spine in ax2.spines.values():
            spine.set_color('#333344')
        cbar2 = plt.colorbar(scatter2, ax=ax2, shrink=0.8)
        cbar2.ax.tick_params(colors='#888888')
        
        # Interpolate to common grid for difference calculation
        ref_interp = griddata(ref_points, ref_values, (Xi, Yi), method='linear')
        pred_interp = griddata(pred_points, pred_values, (Xi, Yi), method='linear')
        
        # Difference plot
        ax3 = fig.add_subplot(gs[idx, 2])
        diff = pred_interp - ref_interp
        
        # Symmetric color range
        vmax = np.nanmax(np.abs(diff))
        if vmax == 0:
            vmax = 1e-10
        
        im3 = ax3.pcolormesh(Xi, Yi, diff, cmap=cmap_diff, vmin=-vmax, vmax=vmax, shading='auto')
        ax3.set_xlim(x_min, x_max)
        ax3.set_ylim(y_min, y_max)
        ax3.set_xlabel('X', fontsize=10, color='#aaaaaa')
        ax3.set_ylabel('Y', fontsize=10, color='#aaaaaa')
        ax3.set_title(f'Difference (Pred-Ref) - {label}', fontsize=11, color='#ffaa00', fontweight='bold')
        ax3.set_facecolor('#0f0f18')
        ax3.tick_params(colors='#888888')
        for spine in ax3.spines.values():
            spine.set_color('#333344')
        cbar3 = plt.colorbar(im3, ax=ax3, shrink=0.8)
        cbar3.ax.tick_params(colors='#888888')
        
        # Statistics panel
        ax4 = fig.add_subplot(gs[idx, 3])
        ax4.set_facecolor('#0f0f18')
        
        # Calculate statistics
        valid_mask = ~np.isnan(diff)
        if np.any(valid_mask):
            diff_valid = diff[valid_mask]
            ref_valid = ref_interp[valid_mask]
            pred_valid = pred_interp[valid_mask]
            
            mae = np.mean(np.abs(diff_valid))
            rmse = np.sqrt(np.mean(diff_valid**2))
            max_diff = np.max(np.abs(diff_valid))
            
            # Relative error (avoid division by zero)
            ref_range = np.max(ref_valid) - np.min(ref_valid)
            if ref_range > 0:
                rel_error = rmse / ref_range * 100
            else:
                rel_error = 0
            
            stats_text = f"""Statistics - {var.upper()}
            
Reference Data Range:
  Min: {np.min(ref_valid):.6e}
  Max: {np.max(ref_valid):.6e}
  
Prediction Data Range:
  Min: {np.min(pred_valid):.6e}
  Max: {np.max(pred_valid):.6e}

Error Analysis:
  MAE: {mae:.6e}
  RMSE: {rmse:.6e}
  Max Abs Error: {max_diff:.6e}
  Relative Error: {rel_error:.2f}%
"""
        else:
            stats_text = f"Statistics - {var.upper()}\n\nNo valid data for comparison"
        
        ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, 
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                color='#cccccc', bbox=dict(boxstyle='round', facecolor='#1a1a2e', 
                                           edgecolor='#333366', alpha=0.9))
        ax4.axis('off')
    
    # Add main title
    fig.suptitle('Prediction vs Reference Data Comparison\n(grid_pred_case0.csv vs merged_fibers_output_z_m0p005.txt)', 
                 fontsize=16, color='#ffffff', fontweight='bold', y=0.98)
    
    plt.savefig(output_path, dpi=150, facecolor='#0a0a0f', edgecolor='none', 
                bbox_inches='tight', pad_inches=0.3)
    plt.close()
    print(f"Visualization saved to: {output_path}")

def create_line_comparison(ref_df, pred_df, output_path='line_comparison.png'):
    """Create line comparison plots along specific directions"""
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(3, 2, figsize=(16, 12))
    fig.patch.set_facecolor('#0a0a0f')
    
    variables = ['u', 'v', 'w']
    var_labels = ['U (X-displacement)', 'V (Y-displacement)', 'W (Z-displacement)']
    
    # Select specific x values for line comparison
    x_values_to_compare = [0.0, 0.15]
    
    for row, (var, label) in enumerate(zip(variables, var_labels)):
        for col, x_val in enumerate(x_values_to_compare):
            ax = axes[row, col]
            ax.set_facecolor('#0f0f18')
            
            # Filter reference data near this x value
            tol = 0.02
            ref_mask = np.abs(ref_df['x'] - x_val) < tol
            ref_subset = ref_df[ref_mask].sort_values('y')
            
            # Filter prediction data near this x value
            pred_mask = np.abs(pred_df['x'] - x_val) < tol
            pred_subset = pred_df[pred_mask].sort_values('y')
            
            if len(ref_subset) > 0:
                ax.plot(ref_subset['y'], ref_subset[var], 'o-', 
                       color='#00ff88', label='Reference', markersize=6, linewidth=2, alpha=0.8)
            
            if len(pred_subset) > 0:
                ax.plot(pred_subset['y'], pred_subset[var], 's--', 
                       color='#ff6688', label='Prediction', markersize=3, linewidth=1.5, alpha=0.7)
            
            ax.set_xlabel('Y', fontsize=10, color='#aaaaaa')
            ax.set_ylabel(var.upper(), fontsize=10, color='#aaaaaa')
            ax.set_title(f'{label} @ X={x_val:.2f}', fontsize=11, color='#ffaa00', fontweight='bold')
            ax.legend(loc='best', fontsize=9, facecolor='#1a1a2e', edgecolor='#333366')
            ax.grid(True, alpha=0.2, color='#444466')
            ax.tick_params(colors='#888888')
            for spine in ax.spines.values():
                spine.set_color('#333344')
    
    fig.suptitle('Displacement Distribution Along Y-direction', fontsize=14, color='#ffffff', fontweight='bold', y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=150, facecolor='#0a0a0f', edgecolor='none', 
                bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Line comparison saved to: {output_path}")

def create_scatter_correlation(ref_df, pred_df, output_path='scatter_correlation.png'):
    """Create scatter correlation plots"""
    
    plt.style.use('dark_background')
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.patch.set_facecolor('#0a0a0f')
    
    variables = ['u', 'v', 'w']
    var_labels = ['U', 'V', 'W']
    colors = ['#00ff88', '#ff6688', '#ffaa00']
    
    # Get common grid for interpolation
    Xi, Yi, _, _, _, _ = interpolate_to_common_grid(ref_df, pred_df, grid_resolution=30)
    
    for idx, (var, label, color) in enumerate(zip(variables, var_labels, colors)):
        ax = axes[idx]
        ax.set_facecolor('#0f0f18')
        
        # Interpolation
        ref_points = ref_df[['x', 'y']].values
        pred_points = pred_df[['x', 'y']].values
        
        ref_interp = griddata(ref_points, ref_df[var].values, (Xi, Yi), method='linear')
        pred_interp = griddata(pred_points, pred_df[var].values, (Xi, Yi), method='linear')
        
        # Flatten and remove NaN
        ref_flat = ref_interp.flatten()
        pred_flat = pred_interp.flatten()
        valid = ~(np.isnan(ref_flat) | np.isnan(pred_flat))
        
        if np.sum(valid) > 0:
            ref_valid = ref_flat[valid]
            pred_valid = pred_flat[valid]
            
            ax.scatter(ref_valid, pred_valid, c=color, alpha=0.5, s=20, edgecolors='none')
            
            # Add ideal line y=x
            all_vals = np.concatenate([ref_valid, pred_valid])
            min_val, max_val = np.min(all_vals), np.max(all_vals)
            ax.plot([min_val, max_val], [min_val, max_val], '--', 
                   color='#ffffff', linewidth=2, alpha=0.7, label='Ideal (y=x)')
            
            # Calculate correlation coefficient
            corr = np.corrcoef(ref_valid, pred_valid)[0, 1]
            ax.text(0.05, 0.95, f'Correlation: {corr:.4f}', transform=ax.transAxes,
                   fontsize=11, color='#ffffff', verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='#1a1a2e', edgecolor=color, alpha=0.9))
        
        ax.set_xlabel(f'Reference {label}', fontsize=11, color='#aaaaaa')
        ax.set_ylabel(f'Prediction {label}', fontsize=11, color='#aaaaaa')
        ax.set_title(f'{label} Correlation Analysis', fontsize=12, color=color, fontweight='bold')
        ax.legend(loc='lower right', fontsize=9, facecolor='#1a1a2e', edgecolor='#333366')
        ax.grid(True, alpha=0.2, color='#444466')
        ax.tick_params(colors='#888888')
        for spine in ax.spines.values():
            spine.set_color('#333344')
        ax.set_aspect('equal', adjustable='box')
    
    fig.suptitle('Prediction vs Reference Correlation Analysis', fontsize=14, color='#ffffff', fontweight='bold', y=1.02)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, facecolor='#0a0a0f', edgecolor='none', 
                bbox_inches='tight', pad_inches=0.2)
    plt.close()
    print(f"Correlation plot saved to: {output_path}")

def main():
    # File paths
    ref_path = '/home/misaka/fiberglass/data/merged_fibers_output_z_m0p005.txt'
    pred_path = '/home/misaka/fiberglass/runs/plate_basis/grid_pred_case0.csv'
    
    print("=" * 60)
    print("Prediction vs Reference Data Visualization")
    print("=" * 60)
    
    # Load data
    print("\n[1/4] Loading data...")
    ref_df = load_reference_data(ref_path)
    pred_df = load_prediction_data(pred_path)
    
    print(f"  Reference data: {len(ref_df)} points")
    print(f"  Prediction data: {len(pred_df)} points")
    print(f"  Reference columns: {list(ref_df.columns)}")
    print(f"  Prediction columns: {list(pred_df.columns)}")
    
    # Create comparison visualization
    print("\n[2/4] Generating main comparison plot...")
    create_comparison_plot(ref_df, pred_df, 
                          '/home/misaka/fiberglass/comparison_visualization.png')
    
    # Create line comparison
    print("\n[3/4] Generating line comparison plot...")
    create_line_comparison(ref_df, pred_df,
                          '/home/misaka/fiberglass/line_comparison.png')
    
    # Create correlation plot
    print("\n[4/4] Generating correlation analysis plot...")
    create_scatter_correlation(ref_df, pred_df,
                              '/home/misaka/fiberglass/scatter_correlation.png')
    
    print("\n" + "=" * 60)
    print("Visualization complete! Generated files:")
    print("  1. comparison_visualization.png - Main comparison")
    print("  2. line_comparison.png - Line comparison")
    print("  3. scatter_correlation.png - Correlation analysis")
    print("=" * 60)

if __name__ == '__main__':
    main()
