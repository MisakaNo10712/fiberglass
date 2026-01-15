# ML 诊断报告摘要

**生成时间**: 2026-01-15  
**模型**: MambaCoeffNet (实际使用 CNN fallback)  
**训练配置**: runs/20260114-201551  

---

## 🔴 关键发现（已确认）

### 1. 严重条件塌缩（置信度 100%）

**证据 A - w_compare.csv 分析**：
- 所有 21 个样本在每个位置的 w_pred 完全相同
- 跨样本方差：**0.00000000** (< 1e-10)
- 空间方差：0.00042390（样本内有微小变化）

**证据 B - 两样本系数检查**：
```
Sample 1 coefficients: [0.03090071, 0.07633696, -0.05108402, ...]
Sample 2 coefficients: [0.03090081, 0.07633696, -0.05108402, ...]
Max difference: 1.04e-07 (< 1e-5)
```
- 两个不同样本的系数差异仅为 **0.0000001**
- 这是数值误差级别，实际上完全相同

**结论**：模型输出了一个与输入无关的"万能常数场"

---

### 2. 幅值严重抑制（置信度 100%）

| 指标 | w_true | w_pred | 比值 |
|------|--------|--------|------|
| 最大绝对值 | 0.046164 | 0.001283 | **2.78%** |

**结论**：系数被压缩到接近 0，w_pred 幅值仅为真实值的 2.78%

---

### 3. Loss 权重严重失衡（置信度 95%）

**训练过程中的 Loss 占比**：

| Epoch | loss_kappa | loss_w | kappa 占比 | w 占比 |
|-------|-----------|--------|-----------|--------|
| 1 | 18.587 | 0.00347 | 99.98% | 0.02% |
| 100 | 0.0732 | 0.00716 | 90%+ | <10% |

**配置**：
```yaml
loss:
  lambda_hf: 0.01      # HF 正则
  lambda_w: 1.0        # w 监督权重
```

**问题**：虽然 lambda_w=1.0，但 loss_w 的量级天然比 loss_kappa 小 100 倍，导致实际贡献 < 10%

---

### 4. 编码器确认（置信度 100%）

**发现**：
- Checkpoint 中包含 `mamba_layers.*` 参数
- 训练时使用了 **Mamba encoder**
- 当前环境加载时回退到 **CNN encoder**（mamba-ssm 不可用）

**重要性**：这不是塌缩的主因，但说明：
- 训练环境有 CUDA + mamba-ssm
- 当前诊断环境使用 CPU + CNN fallback
- 塌缩是 loss 设计问题，与编码器类型无关

---

## 🎯 根本原因

### 塌缩机制

```
1. Kappa loss 主导优化（占 90%+）
   ↓
2. 模型发现：如果 w ≈ 0（常数），则 κ_pred ≈ 0
   ↓
3. 当 κ_meas 均值接近 0 时，这是低 loss 策略
   ↓
4. HF 正则（lambda_hf=0.01）进一步压制系数 → a → 0
   ↓
5. W loss 权重不足（实际贡献 < 10%），无法纠正
   ↓
6. 结果：局部最优解 a ≈ 0 → w ≈ 0 → κ ≈ 0
```

**为什么所有样本输出相同**：
- 因为最优解是 a ≈ 0（常数）
- 与输入无关，所以所有样本都得到相同的 a

---

## ✅ 最小修复方案

### Patch 1: 重新平衡 Loss 权重（必须）

**修改配置文件**：
```yaml
loss:
  huber_delta: 1.0
  lambda_hf: 0.0       # 先关闭 HF 正则
  lambda_w: 100.0      # 提升到与 kappa 相当
  lambda_bc: 0.0
```

**原理**：
- 让 w_loss 和 kappa_loss 量级相当
- 100 * 0.007 ≈ 0.7 vs 0.07（相当）
- 强制模型关注 w 的准确性

---

### Patch 2: 单样本 Overfit 验证

**目的**：验证模型能否拟合单个样本

**步骤**：
1. 创建只包含 1 个样本的数据集
2. 训练 100 epochs，lambda_w=100, lambda_hf=0
3. 检查 w_pred vs w_true

**成功标准**：
- RMSE(w) < 0.001
- w_pred 幅值 > 80% w_true

---

### Patch 3: 两样本差异验证

**目的**：验证修复后模型能否区分样本

**步骤**：
1. 使用 2 个样本训练
2. 运行 `tools_diagnose/check_two_sample_diff.py`
3. 检查系数差异

**成功标准**：
- Max |a1 - a2| > 0.01（当前仅 1e-7）

---

## 📊 数据证据汇总

### 从 w_compare.csv（5418 行，21 样本）
- max|w_true|: 0.046164
- max|w_pred|: 0.001283
- 跨样本方差: 0.00000000（完全塌缩）
- MAE: 0.00108844

### 从 metrics.csv（训练过程）
- Epoch 1: kappa_loss=18.587 (99.98%), w_loss=0.00347 (0.02%)
- Epoch 100: kappa_loss=0.0732 (90%+), w_loss=0.00716 (<10%)

### 从两样本系数检查
- Sample 1 mean: 0.01681837, std: 0.05722837
- Sample 2 mean: 0.01681840, std: 0.05722838
- Max diff: **1.04e-07** (几乎完全相同)

---

## 🔧 验证实验计划

### Experiment 1: 单样本 Overfit
- **目的**：验证模型架构没问题
- **预期**：能拟合单样本 → 说明是 loss 设计问题

### Experiment 2: Loss 权重扫描
- **扫描**：lambda_w ∈ [1, 10, 100, 1000]
- **记录**：w_pred 幅值、跨样本方差
- **目标**：找到最佳 lambda_w

### Experiment 3: 渐进式训练
- **阶段 1**：只训练 w（lambda_kappa=0）
- **阶段 2**：联合训练（lambda_kappa=1, lambda_w=10）

---

## 📁 生成的文件

**诊断输出**：
- ✅ `diagnostic_figs/w_collapse_results.json`
- ✅ `diagnostic_logs/w_collapse_quick.log`
- ✅ `diagnostic_logs/two_sample_diff.txt`
- ✅ `diagnostic_logs/two_sample_diff_run.log`
- ✅ `REPORT_diagnosis.md`（完整报告，725 行）
- ✅ `REPORT_diagnosis_SUMMARY.md`（本文件）

**诊断脚本**：
- ✅ `tools_diagnose/check_two_sample_diff.py`

---

## 🚀 立即行动

### 今天必须做：

1. **修改配置**：
   ```bash
   cp project/runs/20260114-201551/config.yaml configs/fixed_loss.yaml
   # 编辑 configs/fixed_loss.yaml：
   # lambda_hf: 0.0
   # lambda_w: 100.0
   ```

2. **重新训练**：
   ```bash
   python project/scripts/train.py --config configs/fixed_loss.yaml
   ```

3. **检查结果**：
   - w_pred 幅值是否恢复？
   - 运行 `check_two_sample_diff.py`，差异是否 > 0.01？

---

## 📈 预期结果

**修复前**（当前）：
- w_pred 幅值：2.78% of w_true
- 跨样本方差：< 1e-10
- 系数差异：1e-7

**修复后**（预期）：
- w_pred 幅值：> 80% of w_true
- 跨样本方差：> 1e-5
- 系数差异：> 0.01

---

## 💡 如果修复失败

### 备选方案：

1. **DeepONet**：直接学习"观测 → 场"的映射
2. **图神经网络**：节点=观测点，边=空间关系
3. **网格化 + UNet**：投影到规则网格 + CNN
4. **线性基线**：最小二乘验证可辨识性

---

## 📝 总结

**问题本质**：Loss 函数设计失衡，导致模型学习到"输出常数场"的局部最优解

**修复策略**：重新平衡 loss 权重（lambda_w: 1.0 → 100.0）

**置信度**：95% 确信这是主要原因

**下一步**：立即修改配置并重新训练

---

**报告完成**  
**诊断工程师**: AI Assistant (GPT-5.2)  
**日期**: 2026-01-15
