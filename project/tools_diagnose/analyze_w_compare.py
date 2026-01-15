#!/usr/bin/env python
"""
诊断脚本：分析 w_compare.csv 检测条件塌缩和误差模式

输出：
- 全局幅值对比统计
- 条件塌缩检测（同一 point_id 跨样本的 w_pred 方差）
- 误差随空间位置变化分析
- 可视化图表
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # 非交互式后端

# 设置中文字体（如果可用）
plt.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 150

def load_w_compare(csv_path: str | Path) -> pd.DataFrame:
    """加载 w_compare.csv"""
    df = pd.read_csv(csv_path)
    print(f"[INFO] 加载 {csv_path}")
    print(f"[INFO] 数据形状: {df.shape}")
    print(f"[INFO] 列名: {list(df.columns)}")
    return df


def analyze_global_amplitude(df: pd.DataFrame) -> dict:
    """全局幅值对比分析"""
    w_true = df['w_true'].values
    w_pred = df['w_pred'].values
    
    stats = {
        'w_true_min': float(np.min(w_true)),
        'w_true_max': float(np.max(w_true)),
        'w_true_range': float(np.max(w_true) - np.min(w_true)),
        'w_true_mean': float(np.mean(w_true)),
        'w_true_std': float(np.std(w_true)),
        'w_pred_min': float(np.min(w_pred)),
        'w_pred_max': float(np.max(w_pred)),
        'w_pred_range': float(np.max(w_pred) - np.min(w_pred)),
        'w_pred_mean': float(np.mean(w_pred)),
        'w_pred_std': float(np.std(w_pred)),
        'amplitude_ratio': float(np.max(np.abs(w_pred)) / max(np.max(np.abs(w_true)), 1e-12)),
        'mae': float(np.mean(np.abs(w_pred - w_true))),
        'rmse': float(np.sqrt(np.mean((w_pred - w_true)**2))),
        'correlation': float(np.corrcoef(w_true, w_pred)[0, 1]) if len(w_true) > 1 else 0.0,
    }
    
    print("\n" + "="*60)
    print("全局幅值对比分析")
    print("="*60)
    print(f"w_true: range=[{stats['w_true_min']:.6e}, {stats['w_true_max']:.6e}], "
          f"span={stats['w_true_range']:.6e}")
    print(f"w_pred: range=[{stats['w_pred_min']:.6e}, {stats['w_pred_max']:.6e}], "
          f"span={stats['w_pred_range']:.6e}")
    print(f"幅值比 (max|w_pred|/max|w_true|): {stats['amplitude_ratio']:.4f}")
    print(f"MAE: {stats['mae']:.6e}")
    print(f"RMSE: {stats['rmse']:.6e}")
    print(f"Pearson相关系数: {stats['correlation']:.4f}")
    
    return stats


def analyze_collapse_by_point(df: pd.DataFrame) -> pd.DataFrame:
    """
    条件塌缩检测：对每个 point_id，计算不同 sample_name 的 w_pred 方差
    如果方差接近 0，说明模型对不同样本输出相同的 w_pred（塌缩）
    """
    print("\n" + "="*60)
    print("条件塌缩检测（按 point_id 分组）")
    print("="*60)
    
    # 按 point_id 分组，计算 w_pred 的方差
    collapse_stats = df.groupby('point_id').agg({
        'w_pred': ['mean', 'std', 'var', 'min', 'max', 'count'],
        'w_true': ['mean', 'std', 'var', 'min', 'max'],
    }).reset_index()
    
    collapse_stats.columns = ['point_id', 
                               'w_pred_mean', 'w_pred_std', 'w_pred_var', 'w_pred_min', 'w_pred_max', 'sample_count',
                               'w_true_mean', 'w_true_std', 'w_true_var', 'w_true_min', 'w_true_max']
    
    # 计算变异系数 (CV) = std / |mean|
    collapse_stats['w_pred_cv'] = collapse_stats['w_pred_std'] / (np.abs(collapse_stats['w_pred_mean']) + 1e-12)
    collapse_stats['w_true_cv'] = collapse_stats['w_true_std'] / (np.abs(collapse_stats['w_true_mean']) + 1e-12)
    
    # 塌缩指标：w_pred 的方差相对于 w_true 方差的比值
    collapse_stats['variance_ratio'] = collapse_stats['w_pred_var'] / (collapse_stats['w_true_var'] + 1e-12)
    
    # 统计
    n_points = len(collapse_stats)
    n_collapsed = (collapse_stats['w_pred_std'] < 1e-6).sum()
    n_low_variance = (collapse_stats['variance_ratio'] < 0.1).sum()
    
    print(f"总 point_id 数量: {n_points}")
    print(f"每个 point_id 的样本数: {collapse_stats['sample_count'].iloc[0]} (假设均匀)")
    print(f"w_pred 标准差 < 1e-6 的点数（严重塌缩）: {n_collapsed} ({100*n_collapsed/n_points:.1f}%)")
    print(f"方差比 < 0.1 的点数（显著塌缩）: {n_low_variance} ({100*n_low_variance/n_points:.1f}%)")
    
    print(f"\nw_pred 跨样本标准差统计:")
    print(f"  mean(std): {collapse_stats['w_pred_std'].mean():.6e}")
    print(f"  median(std): {collapse_stats['w_pred_std'].median():.6e}")
    print(f"  max(std): {collapse_stats['w_pred_std'].max():.6e}")
    
    print(f"\nw_true 跨样本标准差统计:")
    print(f"  mean(std): {collapse_stats['w_true_std'].mean():.6e}")
    print(f"  median(std): {collapse_stats['w_true_std'].median():.6e}")
    print(f"  max(std): {collapse_stats['w_true_std'].max():.6e}")
    
    print(f"\n方差比 (w_pred_var / w_true_var) 统计:")
    print(f"  mean: {collapse_stats['variance_ratio'].mean():.4f}")
    print(f"  median: {collapse_stats['variance_ratio'].median():.4f}")
    print(f"  min: {collapse_stats['variance_ratio'].min():.4f}")
    print(f"  max: {collapse_stats['variance_ratio'].max():.4f}")
    
    return collapse_stats


def analyze_collapse_by_sample(df: pd.DataFrame) -> pd.DataFrame:
    """
    按样本分析：每个 sample_name 的 w_pred 空间方差
    如果空间方差很小，说明模型输出接近常数平面
    """
    print("\n" + "="*60)
    print("空间塌缩检测（按 sample_name 分组）")
    print("="*60)
    
    sample_stats = df.groupby('sample_name').agg({
        'w_pred': ['mean', 'std', 'var', 'min', 'max', 'count'],
        'w_true': ['mean', 'std', 'var', 'min', 'max'],
    }).reset_index()
    
    sample_stats.columns = ['sample_name',
                            'w_pred_mean', 'w_pred_std', 'w_pred_var', 'w_pred_min', 'w_pred_max', 'point_count',
                            'w_true_mean', 'w_true_std', 'w_true_var', 'w_true_min', 'w_true_max']
    
    # 空间方差比
    sample_stats['spatial_variance_ratio'] = sample_stats['w_pred_var'] / (sample_stats['w_true_var'] + 1e-12)
    
    print(f"样本数量: {len(sample_stats)}")
    print(f"每个样本的点数: {sample_stats['point_count'].iloc[0]}")
    
    print(f"\n各样本 w_pred 空间标准差:")
    for _, row in sample_stats.iterrows():
        print(f"  {row['sample_name']}: std={row['w_pred_std']:.6e}, "
              f"range=[{row['w_pred_min']:.6e}, {row['w_pred_max']:.6e}]")
    
    print(f"\n各样本 w_true 空间标准差:")
    for _, row in sample_stats.iterrows():
        print(f"  {row['sample_name']}: std={row['w_true_std']:.6e}, "
              f"range=[{row['w_true_min']:.6e}, {row['w_true_max']:.6e}]")
    
    print(f"\n空间方差比 (w_pred_var / w_true_var):")
    print(f"  mean: {sample_stats['spatial_variance_ratio'].mean():.4f}")
    print(f"  min: {sample_stats['spatial_variance_ratio'].min():.4f}")
    print(f"  max: {sample_stats['spatial_variance_ratio'].max():.4f}")
    
    return sample_stats


def plot_scatter_comparison(df: pd.DataFrame, output_dir: Path):
    """绘制 w_true vs w_pred 散点图"""
    fig, ax = plt.subplots(figsize=(8, 8))
    
    w_true = df['w_true'].values
    w_pred = df['w_pred'].values
    
    ax.scatter(w_true, w_pred, alpha=0.3, s=5, c='blue')
    
    # 理想线
    lims = [min(w_true.min(), w_pred.min()), max(w_true.max(), w_pred.max())]
    ax.plot(lims, lims, 'r--', linewidth=2, label='Ideal (y=x)')
    
    # 统计信息
    corr = np.corrcoef(w_true, w_pred)[0, 1]
    mae = np.mean(np.abs(w_pred - w_true))
    
    ax.set_xlabel('w_true', fontsize=12)
    ax.set_ylabel('w_pred', fontsize=12)
    ax.set_title(f'w_true vs w_pred Scatter Plot\nCorr={corr:.4f}, MAE={mae:.2e}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal', adjustable='box')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'scatter_w_true_vs_w_pred.png', dpi=150)
    plt.close()
    print(f"[PLOT] 保存: {output_dir / 'scatter_w_true_vs_w_pred.png'}")


def plot_collapse_evidence(df: pd.DataFrame, collapse_stats: pd.DataFrame, output_dir: Path):
    """绘制塌缩证据图：同一 point_id 跨样本的 w_pred 对比"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 1. 方差比分布直方图
    ax = axes[0, 0]
    variance_ratios = collapse_stats['variance_ratio'].values
    variance_ratios_clipped = np.clip(variance_ratios, 0, 2)  # 裁剪以便可视化
    ax.hist(variance_ratios_clipped, bins=50, edgecolor='black', alpha=0.7)
    ax.axvline(x=1.0, color='r', linestyle='--', label='Ideal ratio=1')
    ax.axvline(x=0.1, color='orange', linestyle='--', label='Collapse threshold=0.1')
    ax.set_xlabel('Variance Ratio (w_pred_var / w_true_var)', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title('Distribution of Variance Ratio by Point ID', fontsize=12)
    ax.legend()
    
    # 2. w_pred std vs w_true std
    ax = axes[0, 1]
    ax.scatter(collapse_stats['w_true_std'], collapse_stats['w_pred_std'], alpha=0.5, s=10)
    max_std = max(collapse_stats['w_true_std'].max(), collapse_stats['w_pred_std'].max())
    ax.plot([0, max_std], [0, max_std], 'r--', label='y=x')
    ax.set_xlabel('w_true std (across samples)', fontsize=10)
    ax.set_ylabel('w_pred std (across samples)', fontsize=10)
    ax.set_title('Cross-sample Std: w_pred vs w_true', fontsize=12)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 3. 选取几个 point_id，展示不同样本的 w_pred 值
    ax = axes[1, 0]
    sample_names = df['sample_name'].unique()
    n_samples = len(sample_names)
    
    # 选取方差比最低的几个点（最塌缩的）
    worst_points = collapse_stats.nsmallest(5, 'variance_ratio')['point_id'].values
    
    for i, pid in enumerate(worst_points):
        point_data = df[df['point_id'] == pid]
        w_preds = point_data['w_pred'].values
        w_trues = point_data['w_true'].values
        
        x_pos = np.arange(len(w_preds)) + i * 0.15
        ax.scatter(x_pos, w_preds, label=f'point {pid} pred', marker='o', s=30)
        ax.scatter(x_pos, w_trues, label=f'point {pid} true', marker='x', s=30, alpha=0.5)
    
    ax.set_xlabel('Sample Index', fontsize=10)
    ax.set_ylabel('w value', fontsize=10)
    ax.set_title('Collapse Evidence: w_pred (o) vs w_true (x) for worst points', fontsize=12)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # 4. 每个样本的 w_pred 分布箱线图
    ax = axes[1, 1]
    sample_data = [df[df['sample_name'] == s]['w_pred'].values for s in sample_names]
    bp = ax.boxplot(sample_data, labels=[s.replace('merged_fibers_output_', '').replace('.parquet', '') 
                                          for s in sample_names], vert=True)
    ax.set_xlabel('Sample', fontsize=10)
    ax.set_ylabel('w_pred', fontsize=10)
    ax.set_title('w_pred Distribution by Sample', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'collapse_evidence.png', dpi=150)
    plt.close()
    print(f"[PLOT] 保存: {output_dir / 'collapse_evidence.png'}")


def plot_sample_curves(df: pd.DataFrame, output_dir: Path):
    """绘制每个样本的 w_pred 和 w_true 曲线叠图"""
    sample_names = df['sample_name'].unique()
    
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # 按 point_id 排序
    for sample in sample_names:
        sample_df = df[df['sample_name'] == sample].sort_values('point_id')
        label = sample.replace('merged_fibers_output_', '').replace('.parquet', '')
        axes[0].plot(sample_df['point_id'], sample_df['w_true'], label=label, alpha=0.7)
        axes[1].plot(sample_df['point_id'], sample_df['w_pred'], label=label, alpha=0.7)
    
    axes[0].set_ylabel('w_true', fontsize=12)
    axes[0].set_title('w_true curves by sample', fontsize=14)
    axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('point_id', fontsize=12)
    axes[1].set_ylabel('w_pred', fontsize=12)
    axes[1].set_title('w_pred curves by sample (check for collapse: curves should differ)', fontsize=14)
    axes[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'sample_curves_comparison.png', dpi=150)
    plt.close()
    print(f"[PLOT] 保存: {output_dir / 'sample_curves_comparison.png'}")


def plot_error_analysis(df: pd.DataFrame, output_dir: Path):
    """误差分析图"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    df['abs_error'] = np.abs(df['w_pred'] - df['w_true'])
    df['rel_error'] = df['abs_error'] / (np.abs(df['w_true']) + 1e-12)
    
    # 1. 误差分布直方图
    ax = axes[0, 0]
    ax.hist(df['abs_error'], bins=50, edgecolor='black', alpha=0.7)
    ax.set_xlabel('Absolute Error |w_pred - w_true|', fontsize=10)
    ax.set_ylabel('Count', fontsize=10)
    ax.set_title(f'Error Distribution (MAE={df["abs_error"].mean():.2e})', fontsize=12)
    ax.axvline(df['abs_error'].mean(), color='r', linestyle='--', label=f'Mean={df["abs_error"].mean():.2e}')
    ax.legend()
    
    # 2. 误差 vs w_true 幅值
    ax = axes[0, 1]
    ax.scatter(np.abs(df['w_true']), df['abs_error'], alpha=0.3, s=5)
    ax.set_xlabel('|w_true|', fontsize=10)
    ax.set_ylabel('Absolute Error', fontsize=10)
    ax.set_title('Error vs True Amplitude', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 3. 误差按 point_id 分布
    ax = axes[1, 0]
    error_by_point = df.groupby('point_id')['abs_error'].mean()
    ax.plot(error_by_point.index, error_by_point.values, 'b-', alpha=0.7)
    ax.set_xlabel('point_id', fontsize=10)
    ax.set_ylabel('Mean Absolute Error', fontsize=10)
    ax.set_title('MAE by Point ID (spatial pattern)', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    # 4. 残差图
    ax = axes[1, 1]
    residuals = df['w_pred'] - df['w_true']
    ax.scatter(df['w_true'], residuals, alpha=0.3, s=5)
    ax.axhline(y=0, color='r', linestyle='--')
    ax.set_xlabel('w_true', fontsize=10)
    ax.set_ylabel('Residual (w_pred - w_true)', fontsize=10)
    ax.set_title('Residual Plot', fontsize=12)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_analysis.png', dpi=150)
    plt.close()
    print(f"[PLOT] 保存: {output_dir / 'error_analysis.png'}")


def main():
    # 路径配置
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    csv_path = project_dir / 'runs' / '20260114-201551' / 'w_compare.csv'
    output_dir = project_dir / 'diagnostic_figs'
    log_dir = project_dir / 'diagnostic_logs'
    
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    
    # 重定向输出到日志文件
    log_file = log_dir / 'w_compare_analysis.log'
    
    # 同时输出到控制台和文件
    import io
    from contextlib import redirect_stdout
    
    output_buffer = io.StringIO()
    
    class Tee:
        def __init__(self, *files):
            self.files = files
        def write(self, obj):
            for f in self.files:
                f.write(obj)
        def flush(self):
            for f in self.files:
                f.flush()
    
    with open(log_file, 'w') as f:
        tee = Tee(sys.stdout, f)
        old_stdout = sys.stdout
        sys.stdout = tee
        
        try:
            print("="*70)
            print("W_COMPARE.CSV 诊断分析报告")
            print("="*70)
            
            # 加载数据
            df = load_w_compare(csv_path)
            
            # 全局幅值分析
            global_stats = analyze_global_amplitude(df)
            
            # 条件塌缩检测（按 point_id）
            collapse_stats = analyze_collapse_by_point(df)
            
            # 空间塌缩检测（按 sample_name）
            sample_stats = analyze_collapse_by_sample(df)
            
            # 生成图表
            print("\n" + "="*60)
            print("生成可视化图表")
            print("="*60)
            
            plot_scatter_comparison(df, output_dir)
            plot_collapse_evidence(df, collapse_stats, output_dir)
            plot_sample_curves(df, output_dir)
            plot_error_analysis(df, output_dir)
            
            # 总结
            print("\n" + "="*70)
            print("诊断结论")
            print("="*70)
            
            # 判断塌缩程度
            mean_variance_ratio = collapse_stats['variance_ratio'].mean()
            median_variance_ratio = collapse_stats['variance_ratio'].median()
            
            if median_variance_ratio < 0.1:
                collapse_severity = "严重塌缩"
                collapse_desc = "模型对不同样本输出几乎相同的 w_pred，完全丧失了条件区分能力"
            elif median_variance_ratio < 0.5:
                collapse_severity = "中度塌缩"
                collapse_desc = "模型对不同样本的区分能力显著不足"
            elif median_variance_ratio < 0.9:
                collapse_severity = "轻度塌缩"
                collapse_desc = "模型有一定区分能力，但方差被压缩"
            else:
                collapse_severity = "无明显塌缩"
                collapse_desc = "模型能够区分不同样本"
            
            print(f"\n1. 塌缩诊断: {collapse_severity}")
            print(f"   - 方差比中位数: {median_variance_ratio:.4f}")
            print(f"   - 解释: {collapse_desc}")
            
            print(f"\n2. 幅值诊断:")
            print(f"   - w_true 范围: [{global_stats['w_true_min']:.2e}, {global_stats['w_true_max']:.2e}]")
            print(f"   - w_pred 范围: [{global_stats['w_pred_min']:.2e}, {global_stats['w_pred_max']:.2e}]")
            print(f"   - 幅值比: {global_stats['amplitude_ratio']:.4f}")
            
            if global_stats['amplitude_ratio'] < 0.5:
                print(f"   - 警告: w_pred 幅值明显偏小，可能是系数被正则化压缩或梯度消失")
            elif global_stats['amplitude_ratio'] > 2.0:
                print(f"   - 警告: w_pred 幅值偏大，可能存在数值不稳定")
            
            print(f"\n3. 相关性诊断:")
            print(f"   - Pearson 相关系数: {global_stats['correlation']:.4f}")
            if global_stats['correlation'] < 0.5:
                print(f"   - 警告: 相关性很低，模型预测与真值几乎无关")
            
            print(f"\n4. 误差诊断:")
            print(f"   - MAE: {global_stats['mae']:.2e}")
            print(f"   - RMSE: {global_stats['rmse']:.2e}")
            
            # 保存统计结果
            collapse_stats.to_csv(log_dir / 'collapse_stats_by_point.csv', index=False)
            sample_stats.to_csv(log_dir / 'collapse_stats_by_sample.csv', index=False)
            
            print(f"\n[INFO] 统计结果已保存到 {log_dir}")
            print(f"[INFO] 图表已保存到 {output_dir}")
            
        finally:
            sys.stdout = old_stdout
    
    print(f"\n[INFO] 日志已保存到 {log_file}")


if __name__ == '__main__':
    main()
