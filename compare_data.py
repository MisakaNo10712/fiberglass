#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
数据比较分析工具（轻量级版本）
"""

import csv
import math

def load_csv(filename):
    """加载CSV文件"""
    data = []
    with open(filename, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            data.append({k: float(v) for k, v in row.items()})
    return data

def load_txt(filename):
    """加载TXT文件（制表符分隔）"""
    data = []
    with open(filename, 'r') as f:
        lines = f.readlines()
        headers = lines[0].strip().split('\t')
        for line in lines[1:]:
            values = line.strip().split('\t')
            if len(values) == len(headers):
                row = {headers[i]: float(values[i]) for i in range(len(headers))}
                data.append(row)
    return data

def find_nearest(fiber_data, x, y):
    """在纤维数据中找到最近的点"""
    min_dist = float('inf')
    nearest = None
    for point in fiber_data:
        dist = math.sqrt((point['x'] - x)**2 + (point['y'] - y)**2)
        if dist < min_dist:
            min_dist = dist
            nearest = point
    return nearest, min_dist

def calculate_statistics(values):
    """计算统计量"""
    n = len(values)
    if n == 0:
        return {}
    
    mean = sum(values) / n
    variance = sum((x - mean)**2 for x in values) / n
    std = math.sqrt(variance)
    min_val = min(values)
    max_val = max(values)
    
    return {
        'mean': mean,
        'std': std,
        'min': min_val,
        'max': max_val,
        'count': n
    }

def main():
    print("="*80)
    print("数据比较分析工具")
    print("="*80)
    
    # 加载数据
    print("\n正在加载数据...")
    pred_data = load_csv('runs/plate_basis/grid_pred_case0.csv')
    fiber_data = load_txt('data/merged_fibers_output_z_m0p005.txt')
    
    print(f"预测数据: {len(pred_data)} 个点")
    print(f"纤维数据: {len(fiber_data)} 个点")
    
    # 分析每个变量
    variables = ['u', 'v', 'w']
    
    for var in variables:
        print(f"\n{'='*80}")
        print(f"变量: {var}")
        print(f"{'='*80}")
        
        # 预测数据统计
        pred_values = [p[var] for p in pred_data]
        pred_stats = calculate_statistics(pred_values)
        
        print(f"\n预测数据统计:")
        print(f"  数量: {pred_stats['count']}")
        print(f"  均值: {pred_stats['mean']:.6e}")
        print(f"  标准差: {pred_stats['std']:.6e}")
        print(f"  最小值: {pred_stats['min']:.6e}")
        print(f"  最大值: {pred_stats['max']:.6e}")
        
        # 纤维数据统计
        fiber_values = [f[var] for f in fiber_data]
        fiber_stats = calculate_statistics(fiber_values)
        
        print(f"\n纤维数据统计:")
        print(f"  数量: {fiber_stats['count']}")
        print(f"  均值: {fiber_stats['mean']:.6e}")
        print(f"  标准差: {fiber_stats['std']:.6e}")
        print(f"  最小值: {fiber_stats['min']:.6e}")
        print(f"  最大值: {fiber_stats['max']:.6e}")
        
        # 找到匹配点并计算差异
        print(f"\n正在计算差异（使用最近邻匹配）...")
        differences = []
        matched_pred = []
        matched_fiber = []
        
        for i, pred_point in enumerate(pred_data):
            if i % 1000 == 0:
                print(f"  处理进度: {i}/{len(pred_data)}", end='\r')
            
            nearest, dist = find_nearest(fiber_data, pred_point['x'], pred_point['y'])
            
            # 只考虑距离小于0.01的匹配点
            if nearest and dist < 0.01:
                diff = pred_point[var] - nearest[var]
                differences.append(diff)
                matched_pred.append(pred_point[var])
                matched_fiber.append(nearest[var])
        
        print(f"  处理进度: {len(pred_data)}/{len(pred_data)}")
        
        if len(differences) > 0:
            diff_stats = calculate_statistics(differences)
            
            # 计算RMSE和MAE
            rmse = math.sqrt(sum(d**2 for d in differences) / len(differences))
            mae = sum(abs(d) for d in differences) / len(differences)
            
            # 计算相关系数
            n = len(matched_pred)
            mean_pred = sum(matched_pred) / n
            mean_fiber = sum(matched_fiber) / n
            
            cov = sum((matched_pred[i] - mean_pred) * (matched_fiber[i] - mean_fiber) 
                     for i in range(n)) / n
            std_pred = math.sqrt(sum((x - mean_pred)**2 for x in matched_pred) / n)
            std_fiber = math.sqrt(sum((x - mean_fiber)**2 for x in matched_fiber) / n)
            
            if std_pred > 0 and std_fiber > 0:
                correlation = cov / (std_pred * std_fiber)
                r_squared = correlation ** 2
            else:
                correlation = 0
                r_squared = 0
            
            print(f"\n差异分析 (预测 - 纤维):")
            print(f"  匹配点数: {len(differences)}")
            print(f"  差异均值: {diff_stats['mean']:.6e}")
            print(f"  差异标准差: {diff_stats['std']:.6e}")
            print(f"  最小差异: {diff_stats['min']:.6e}")
            print(f"  最大差异: {diff_stats['max']:.6e}")
            print(f"  RMSE: {rmse:.6e}")
            print(f"  MAE: {mae:.6e}")
            print(f"  相关系数 (R): {correlation:.6f}")
            print(f"  决定系数 (R²): {r_squared:.6f}")
            
            if abs(mean_fiber) > 1e-10:
                relative_rmse = rmse / abs(mean_fiber) * 100
                print(f"  相对RMSE: {relative_rmse:.2f}%")
        else:
            print("\n警告: 没有找到匹配的点！")
    
    print(f"\n{'='*80}")
    print("分析完成！")
    print(f"{'='*80}")

if __name__ == '__main__':
    main()
