
# ML 诊断报告摘要

## 关键发现

1. **严重条件塌缩**（置信度 100%）
   - 所有21个样本在每个位置的 w_pred 完全相同（方差 < 1e-10）
   - 模型输出了一个与输入无关的常数场

2. **幅值严重抑制**（置信度 100%）
   - w_pred 最大幅值仅为 w_true 的 2.78%
   - 系数被压缩到接近 0

3. **Loss 权重严重失衡**（置信度 95%）
   - kappa_loss 占总 loss 的 90%+
   - w_loss 虽然 lambda=1.0，但量级小100倍，实际贡献 < 10%

## 根本原因

模型学习到了输出接近0的常数场作为最优解：
- kappa_loss 主导优化方向
- HF 正则（lambda_hf=0.01）持续压制系数
- w_loss 权重不足以纠正这个偏好

## 最小修复方案

**修改 config.yaml**:


**验证步骤**:
1. 单样本 overfit 测试（1个样本，100 epochs）
2. 检查 w_pred 幅值是否恢复到 > 80% w_true
3. 两样本测试，检查系数差异 > 0.1

## 数据证据

从 w_compare.csv (5418行，21样本):
- max|w_true|: 0.046164
- max|w_pred|: 0.001283
- 跨样本方差: 0.00000000 (完全塌缩)
- MAE: 0.00108844

从 metrics.csv:
- Epoch 1: loss_kappa=18.587, loss_w=0.00347 (kappa占99.98%)
- Epoch 100: loss_kappa=0.0732, loss_w=0.00716 (kappa占90%+)

## 生成的文件

- diagnostic_figs/w_collapse_analysis.png
- diagnostic_figs/w_collapse_results.json
- diagnostic_logs/w_collapse_analysis.log
- REPORT_diagnosis.md (完整报告)
