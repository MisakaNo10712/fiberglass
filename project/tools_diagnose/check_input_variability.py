#!/usr/bin/env python
"""
诊断脚本：检查数据集输入变异性和 mask 健全性

检查项目：
1. 输入通道的统计信息（mean/std/min/max）
2. 不同样本之间的输入差异
3. mask 的有效性
4. kappa_t 通道是否有足够变化
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def load_parquet_samples(samples_dir: Path, manifest_path: Path = None):
    """加载所有 parquet 样本"""
    if manifest_path and manifest_path.exists():
        with open(manifest_path) as f:
            manifest = json.load(f)
        files = [samples_dir / entry.get('output_file', entry.get('path', '')) 
                 for entry in manifest.get('files', [])]
    else:
        files = sorted(samples_dir.glob('*.parquet'))
    
    samples = []
    for f in files:
        if f.exists():
            df = pd.read_parquet(f)
            samples.append({'name': f.stem, 'df': df})
    return samples


def analyze_input_channels(samples: list) -> dict:
    """分析输入通道的统计信息"""
    print("\n" + "="*70)
    print("输入通道统计分析")
    print("="*70)
    
    channels = ['x', 'y', 'tx', 'ty', 'kappa_t', 'mask']
    if 'w' in samples[0]['df'].columns:
        channels.append('w')
    
    all_stats = {}
    
    for ch in channels:
        if ch not in samples[0]['df'].columns:
            print(f"[WARN] 通道 '{ch}' 不存在")
            continue
            
        all_values = np.concatenate([s['df'][ch].values for s in samples])
        
        stats = {
            'mean': float(np.mean(all_values)),
            'std': float(np.std(all_values)),
            'min': float(np.min(all_values)),
            'max': float(np.max(all_values)),
            'range': float(np.max(all_values) - np.min(all_values)),
            'n_zeros': int(np.sum(all_values == 0)),
            'n_total': len(all_values),
            'zero_ratio': float(np.sum(all_values == 0) / len(all_values)),
        }
        all_stats[ch] = stats
        
        print(f"\n通道 '{ch}':")
        print(f"  mean={stats['mean']:.6e}, std={stats['std']:.6e}")
        print(f"  range=[{stats['min']:.6e}, {stats['max']:.6e}], span={stats['range']:.6e}")
        print(f"  零值比例: {stats['zero_ratio']*100:.2f}% ({stats['n_zeros']}/{stats['n_total']})")
    
    return all_stats


def analyze_cross_sample_variability(samples: list) -> dict:
    """分析不同样本之间的输入差异"""
    print("\n" + "="*70)
    print("跨样本输入变异性分析")
    print("="*70)
    
    # 检查 kappa_t 通道在不同样本间的差异
    kappa_arrays = []
    for s in samples:
        kappa_arrays.append(s['df']['kappa_t'].values)
    
    # 计算任意两个样本之间的最大差异
    n_samples = len(samples)
    max_diffs = []
    mean_diffs = []
    
    for i in range(n_samples):
        for j in range(i+1, n_samples):
            # 取相同长度
            min_len = min(len(kappa_arrays[i]), len(kappa_arrays[j]))
            diff = np.abs(kappa_arrays[i][:min_len] - kappa_arrays[j][:min_len])
            max_diffs.append(np.max(diff))
            mean_diffs.append(np.mean(diff))
    
    print(f"\nkappa_t 跨样本差异:")
    print(f"  样本对数: {len(max_diffs)}")
    print(f"  max|kappa_i - kappa_j|: mean={np.mean(max_diffs):.6e}, max={np.max(max_diffs):.6e}")
    print(f"  mean|kappa_i - kappa_j|: mean={np.mean(mean_diffs):.6e}")
    
    # 检查每个样本的 kappa_t 统计
    print(f"\n各样本 kappa_t 统计:")
    for s in samples:
        kappa = s['df']['kappa_t'].values
        print(f"  {s['name']}: mean={np.mean(kappa):.6e}, std={np.std(kappa):.6e}, "
              f"range=[{np.min(kappa):.6e}, {np.max(kappa):.6e}]")
    
    # 检查 w 通道（如果存在）
    if 'w' in samples[0]['df'].columns:
        print(f"\n各样本 w (真值) 统计:")
        for s in samples:
            w = s['df']['w'].values
            print(f"  {s['name']}: mean={np.mean(w):.6e}, std={np.std(w):.6e}, "
                  f"range=[{np.min(w):.6e}, {np.max(w):.6e}]")
    
    return {
        'max_kappa_diff_mean': float(np.mean(max_diffs)),
        'max_kappa_diff_max': float(np.max(max_diffs)),
        'mean_kappa_diff_mean': float(np.mean(mean_diffs)),
    }


def analyze_mask_sanity(samples: list) -> dict:
    """分析 mask 的健全性"""
    print("\n" + "="*70)
    print("Mask 健全性检查")
    print("="*70)
    
    mask_stats = []
    for s in samples:
        mask = s['df']['mask'].values
        stats = {
            'name': s['name'],
            'total': len(mask),
            'valid': int(np.sum(mask > 0)),
            'invalid': int(np.sum(mask == 0)),
            'valid_ratio': float(np.sum(mask > 0) / len(mask)),
            'mask_sum': float(np.sum(mask)),
            'mask_mean': float(np.mean(mask)),
        }
        mask_stats.append(stats)
        
        print(f"\n{s['name']}:")
        print(f"  总点数: {stats['total']}")
        print(f"  有效点 (mask>0): {stats['valid']} ({stats['valid_ratio']*100:.1f}%)")
        print(f"  无效点 (mask=0): {stats['invalid']}")
        print(f"  mask.sum(): {stats['mask_sum']:.2f}")
    
    # 检查是否有样本 mask 全为 0
    all_zero_samples = [s for s in mask_stats if s['valid'] == 0]
    if all_zero_samples:
        print(f"\n[严重警告] 以下样本 mask 全为 0:")
        for s in all_zero_samples:
            print(f"  - {s['name']}")
        print("这会导致输入被全部置零，模型无法学习！")
    
    # 检查 mask 是否过于稀疏
    sparse_samples = [s for s in mask_stats if 0 < s['valid_ratio'] < 0.1]
    if sparse_samples:
        print(f"\n[警告] 以下样本 mask 有效率 < 10%:")
        for s in sparse_samples:
            print(f"  - {s['name']}: {s['valid_ratio']*100:.1f}%")
    
    return mask_stats


def analyze_input_after_masking(samples: list) -> dict:
    """分析 mask 后的有效输入"""
    print("\n" + "="*70)
    print("Mask 后有效输入分析")
    print("="*70)
    
    for s in samples:
        df = s['df']
        mask = df['mask'].values > 0
        
        if mask.sum() == 0:
            print(f"\n{s['name']}: [跳过] mask 全为 0")
            continue
        
        print(f"\n{s['name']} (有效点数: {mask.sum()}):")
        
        for ch in ['x', 'y', 'kappa_t']:
            if ch in df.columns:
                valid_values = df[ch].values[mask]
                print(f"  {ch}: mean={np.mean(valid_values):.6e}, "
                      f"std={np.std(valid_values):.6e}, "
                      f"range=[{np.min(valid_values):.6e}, {np.max(valid_values):.6e}]")


def main():
    script_dir = Path(__file__).parent
    project_dir = script_dir.parent
    
    samples_dir = project_dir / 'data' / 'processed2'
    manifest_path = samples_dir / 'manifest.json'
    log_dir = project_dir / 'diagnostic_logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = log_dir / 'input_variability_analysis.log'
    
    # 同时输出到控制台和文件
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
            print("数据集输入变异性与 Mask 健全性诊断")
            print("="*70)
            print(f"数据目录: {samples_dir}")
            
            # 加载样本
            samples = load_parquet_samples(samples_dir, manifest_path)
            print(f"加载样本数: {len(samples)}")
            
            if not samples:
                print("[错误] 未找到任何样本文件")
                return
            
            # 分析
            channel_stats = analyze_input_channels(samples)
            variability_stats = analyze_cross_sample_variability(samples)
            mask_stats = analyze_mask_sanity(samples)
            analyze_input_after_masking(samples)
            
            # 诊断结论
            print("\n" + "="*70)
            print("诊断结论")
            print("="*70)
            
            # 检查 kappa_t 是否有足够变化
            kappa_stats = channel_stats.get('kappa_t', {})
            if kappa_stats:
                if kappa_stats['std'] < 1e-10:
                    print("\n[严重问题] kappa_t 标准差接近 0，输入几乎无变化")
                    print("  -> 模型无法从输入中学习到有意义的信息")
                elif kappa_stats['std'] < 1e-6:
                    print("\n[警告] kappa_t 标准差很小，输入变化可能被归一化压扁")
            
            # 检查跨样本差异
            if variability_stats['max_kappa_diff_max'] < 1e-10:
                print("\n[严重问题] 不同样本的 kappa_t 几乎相同")
                print("  -> 模型无法区分不同样本，必然导致条件塌缩")
            
            # 检查 mask
            all_valid = all(s['valid'] > 0 for s in mask_stats)
            if not all_valid:
                print("\n[严重问题] 存在 mask 全为 0 的样本")
                print("  -> 这些样本的输入会被完全置零")
            
            print(f"\n[INFO] 日志已保存到 {log_file}")
            
        finally:
            sys.stdout = old_stdout


if __name__ == '__main__':
    main()
