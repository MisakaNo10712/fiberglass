#!/usr/bin/env python
"""E: w_compare.csv collapse analysis"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import json

csv_path = "/home/misaka/fiberglass/project/runs/20260114-201551/w_compare.csv"
output_dir = "/home/misaka/fiberglass/diagnostic_figs"
Path(output_dir).mkdir(parents=True, exist_ok=True)

df = pd.read_csv(csv_path)
print(f"Loaded {len(df)} rows")

# 1. Amplitude
w_true_max = float(df['w_true'].abs().max())
w_pred_max = float(df['w_pred'].abs().max())
ratio = w_pred_max / w_true_max if w_true_max > 0 else 0

print(f"\n=== Amplitude ===")
print(f"max|w_true|: {w_true_max:.6f}")
print(f"max|w_pred|: {w_pred_max:.6f}")
print(f"Ratio: {ratio:.4f}")

# 2. Collapse check
collapse_by_point = df.groupby('point_id')['w_pred'].std()
collapse_by_sample = df.groupby('sample_name')['w_pred'].std()

point_var_mean = float(collapse_by_point.mean())
sample_var_mean = float(collapse_by_sample.mean())

print(f"\n=== Collapse Check ===")
print(f"Cross-sample variance (same point): {point_var_mean:.8f}")
print(f"Spatial variance (same sample): {sample_var_mean:.8f}")

is_collapsed = point_var_mean < 1e-5
if is_collapsed:
    print("*** SEVERE COLLAPSE DETECTED ***")

# 3. Error
mae = float(df['w_abs_error'].mean())
rmse = float(np.sqrt(df['w_sq_error'].mean()))

print(f"\n=== Error ===")
print(f"MAE: {mae:.8f}")
print(f"RMSE: {rmse:.8f}")

# 4. Plots
fig, axes = plt.subplots(2, 3, figsize=(18, 12))

ax = axes[0, 0]
ax.scatter(df['w_true'], df['w_pred'], alpha=0.3, s=1)
lim = max(abs(df['w_true']).max(), abs(df['w_pred']).max())
ax.plot([-lim, lim], [-lim, lim], 'r--', lw=2)
ax.set_xlabel('w_true')
ax.set_ylabel('w_pred')
ax.set_title('w_true vs w_pred')
ax.grid(True, alpha=0.3)

ax = axes[0, 1]
ax.hist(df['w_abs_error'], bins=50, alpha=0.7)
ax.set_xlabel('Absolute Error')
ax.set_title('Error Distribution')
ax.axvline(mae, color='r', linestyle='--', lw=2, label=f'MAE={mae:.6f}')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[0, 2]
ax.hist(collapse_by_point, bins=50, alpha=0.7)
ax.set_xlabel('Std across samples')
ax.set_title('Cross-sample variance')
ax.axvline(point_var_mean, color='r', linestyle='--', lw=2, label=f'Mean={point_var_mean:.8f}')
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
for sample in df['sample_name'].unique()[:5]:
    subset = df[df['sample_name'] == sample].sort_values('point_id')
    ax.plot(subset['point_id'], subset['w_pred'], alpha=0.7, label=sample[:15])
ax.set_xlabel('point_id')
ax.set_ylabel('w_pred')
ax.set_title('w_pred curves (5 samples)')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
point_ids = sorted(df['point_id'].unique())[::max(1, len(df['point_id'].unique())//20)]
for pid in point_ids[:10]:
    subset = df[df['point_id'] == pid].sort_values('sample_id')
    ax.plot(subset['sample_id'], subset['w_pred'], marker='o', alpha=0.6, label=f'pt{pid}')
ax.set_xlabel('sample_id')
ax.set_ylabel('w_pred')
ax.set_title('w_pred across samples')
ax.legend(fontsize=7, ncol=2)
ax.grid(True, alpha=0.3)

ax = axes[1, 2]
ax.hist(collapse_by_sample, bins=30, alpha=0.7)
ax.set_xlabel('Std within sample')
ax.set_title('Spatial variance')
ax.axvline(sample_var_mean, color='r', linestyle='--', lw=2, label=f'Mean={sample_var_mean:.8f}')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f"{output_dir}/w_collapse_analysis.png", dpi=150)
print(f"\nFigure saved: {output_dir}/w_collapse_analysis.png")

results = {
    'amplitude': {'w_true_max': w_true_max, 'w_pred_max': w_pred_max, 'ratio': ratio},
    'collapse': {'point_var_mean': point_var_mean, 'sample_var_mean': sample_var_mean, 'is_collapsed': is_collapsed},
    'error': {'mae': mae, 'rmse': rmse}
}

with open(f"{output_dir}/w_collapse_results.json", 'w') as f:
    json.dump(results, f, indent=2)

print("\n" + "="*60)
print("CONCLUSION:")
if is_collapsed:
    print("X SEVERE COLLAPSE: Different samples -> identical w_pred")
if ratio < 0.1:
    print(f"X AMPLITUDE SUPPRESSED: w_pred only {ratio*100:.1f}% of w_true")
