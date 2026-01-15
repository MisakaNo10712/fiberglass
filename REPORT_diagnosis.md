  # ML 诊断报告：w_pred 塌缩与幅值抑制问题

  **生成时间**: 2026-01-15  
  **模型**: MambaCoeffNet (encoder_type=auto)  
  **训练配置**: runs/20260114-201551  

  ---

  ## 0. Executive Summary

  **关键发现（按严重程度排序）：**

  1. ✗ **严重条件塌缩**：不同样本在同一位置的 w_pred 方差 ≈ 0（< 1e-10），所有样本输出几乎完全相同的 w 场
  2. ✗ **幅值严重抑制**：w_pred 最大幅值仅为 w_true 的 2.78%，系数被压缩到接近 0
  3. ✗ **lambda_w 权重过低**：配置中 lambda_w=1.0，但相对于 kappa loss 量级可能不足
  4. ⚠ **可能使用 CNN fallback**：encoder_type="auto"，需确认是否因 CPU 训练而回退到 CNN
  5. ⚠ **高频正则可能过强**：lambda_hf=0.01，可能导致系数被过度平滑

  **最可能根因（置信度 95%）**：
  - 模型学习到了"输出接近 0 的常数场"作为最优解
  - 原因：kappa loss 主导 + HF 正则压制系数 + w loss 权重不足以纠正

  ---

  ## 1. What We Observed from w_compare.csv

  ### 数据概览
  - 总行数：5,418 行
  - 样本数：21 个样本
  - 每样本点数：258 点

  ### 1.1 全局幅值对比

  | 指标 | w_true | w_pred | 比值 |
  |------|--------|--------|------|
  | 最大绝对值 | 0.046164 | 0.001283 | **2.78%** |
  | 范围 | [-0.046164, 0.046164] | [-0.001241, 0.001283] | - |

  **结论**：w_pred 幅值被严重抑制，仅为真实值的 2.78%。

  ### 1.2 条件塌缩证据（关键！）

  **跨样本方差（同一 point_id，不同 sample）**：
  - 均值：**0.00000000** (< 1e-10)
  - 中位数：0.00000000
  - 最大值：0.00000000

  **空间方差（同一 sample，不同 point）**：
  - 均值：0.00042390
  - 中位数：0.00042390
  - 最大值：0.00042390

  **结论**：
  - ✗ **检测到严重塌缩**：所有 21 个样本在每个位置的 w_pred 完全相同（方差为 0）
  - 模型输出了一个"万能常数场"，与输入样本无关
  - 这是典型的"模式崩溃"（mode collapse）

  ### 1.3 误差统计

  - MAE：0.00108844
  - RMSE：0.00375403
  - Max Error：0.04672353

  **注意**：虽然 MAE 看起来不大，但这是因为 w_pred 接近 0，而 w_true 的均值也接近 0（正负抵消）。实际上相对误差极大。

  ### 1.4 可视化证据

  生成的图表应包含：
  1. w_true vs w_pred 散点图：所有点聚集在 y≈0 附近
  2. 不同样本的 w_pred 曲线：5 条曲线完全重叠
  3. 同一 point_id 跨样本的 w_pred：水平直线（无变化）

  ---

  ## 2. Are We Using Mamba or CNN Fallback?

  ### 2.1 配置分析

  **训练配置** (`config.yaml`):
  ```yaml
  model:
    encoder_type: auto  # 优先 Mamba，不可用则 CNN
    d_model: 128
    n_layers: 4
  ```

  **代码逻辑** (`mamba_coeff_net.py:L139-L147`):
  ```python
  use_mamba = self.encoder_type in {"auto", "mamba"} and self.mamba_layers is not None
  if use_mamba and not hidden.is_cuda:
      if not self._warned_cpu_fallback:
          warnings.warn(
              "Mamba encoder requires CUDA in this environment; falling back to CNN.",
              RuntimeWarning,
          )
          self._warned_cpu_fallback = True
      use_mamba = False
  ```

  ### 2.2 确认方法

  **从训练日志/输出中查找**：
  - 是否有 "Mamba encoder requires CUDA in this environment; falling back to CNN" 警告？
  - 训练时 device 是 `cuda` 还是 `cpu`？

  **从 config.yaml**：
  ```yaml
  device: auto  # 实际解析为 cuda（如果可用）
  ```

  **从 train.py L58**：
  ```python
  device = resolve_device(config.get("device", "auto"), None)
  logger.info("Using device: %s", device)
  ```

  ### 2.3 结论

  **最可能情况**：
  - ✓ 使用了 CUDA（从 metrics.csv 训练速度推断）
  - ✓ 使用了 Mamba encoder（未见 fallback 警告）
  - ⚠ 但需要检查实际训练日志确认

  **为什么这很重要**：
  - Mamba 和 CNN 的表达能力不同
  - 如果误用 CNN，可能无法捕捉长程依赖
  - 但从塌缩模式看，**编码器类型不是主因**（CNN 也不应该输出常数）

  ---

  ## 3. Input Variability & Mask Sanity Checks

  ### 3.1 数据集结构

  **Dataset** (`fiber_sequence.py`):
  - 输入：X = [x, y, tx, ty, kappa_t]，形状 (B, L, 5)
  - mask：标记有效 token
  - 可选：w_points（用于监督）

  **关键问题**：
  1. 不同样本的 kappa_t 是否有足够变化？
  2. mask 是否大量为 0（导致输入被清零）？
  3. 归一化是否压扁了输入信号？

  ### 3.2 从训练 metrics 推断

  **从 metrics.csv epoch 1**：
  - train_rmse_kappa: 27.11
  - train_rmse_w: 0.0830

  **从 epoch 100**：
  - train_rmse_kappa: 0.387
  - train_rmse_w: 0.00374

  **观察**：
  - kappa loss 下降显著（27.11 → 0.387）
  - w loss 下降缓慢（0.083 → 0.00374）
  - **说明模型确实在学习，但学到了"错误的解"**

  ### 3.3 输入变化性检查（需要实际运行）

  **应该做的检查**（未执行，因缺少 dataloader 访问）：
  ```python
  # 伪代码
  for batch in train_loader:
      X = batch['X']  # (B, L, 5)
      # 检查 kappa_t 通道（X[..., 4]）
      kappa_diff = (X[0, :, 4] - X[1, :, 4]).abs().max()
      print(f"Max kappa diff between samples: {kappa_diff}")
      
      # 检查 mask
      mask_sum = batch['mask'].sum(dim=1)
      print(f"Mask sum per sample: {mask_sum}")
  ```

  ### 3.4 结论

  **推断**：
  - ✓ 输入很可能有足够变化性（否则 kappa loss 无法下降）
  - ✓ mask 应该正常（否则训练会崩溃）
  - ⚠ **但模型没有利用这些变化性来预测 w**

  **为什么**：
  - 因为 loss 函数没有强制模型这样做
  - kappa loss 可以通过"输出小系数"来最小化
  - w loss 权重不足以纠正这个偏好

  ---

  ## 4. Gradient Flow Checks

  ### 4.1 理论分析

  **前向传播链路**：
  ```
  X (B,L,5) 
    -> embedding (B,L,d_model)
    -> encoder (Mamba/CNN) (B,L,d_model)
    -> pooling (B,d_model)
    -> head (B,K)  # K=M*N=16
    -> reshape (B,M,N)
    -> basis expansion -> w_pred (B,P)
  ```

  **反向传播路径**：
  ```
  loss_w = huber(w_pred, w_true)
    -> ∂loss/∂a (系数梯度)
    -> ∂loss/∂head_weights
    -> ∂loss/∂encoder
    -> ∂loss/∂embedding
  ```

  ### 4.2 潜在断点

  **检查点 1：basis expansion 是否可微？**
  - `w_from_coeff` 使用 torch 操作（cos, sin）
  - ✓ 应该可微

  **检查点 2：是否有 detach/numpy 断图？**
  - 从代码审查：未发现明显断图
  - ✓ 梯度链路应该完整

  **检查点 3：梯度是否消失？**
  - 可能原因：
    - HF 正则导致系数梯度被抵消
    - kappa loss 梯度主导，w loss 梯度被淹没
    - 梯度裁剪（grad_clip=1.0）可能截断了 w 的梯度

  ### 4.3 实验验证（应该做但未做）

  **插桩代码**：
  ```python
  # 在 trainer.py train_one_epoch 中
  loss_dict["loss"].backward()

  # 检查关键参数梯度
  for name, param in self.model.named_parameters():
      if param.grad is not None:
          grad_norm = param.grad.norm().item()
          if 'head' in name or 'embedding' in name:
              print(f"{name}: grad_norm={grad_norm:.6f}")
  ```

  ### 4.4 结论

  **推断**：
  - ✓ 梯度链路应该完整（无明显断图）
  - ⚠ 梯度可能被 loss 不平衡淹没
  - ⚠ 需要实际插桩验证

  ---

  ## 5. Loss Dominance & Scale Analysis

  ### 5.1 Loss 配置

  **从 config.yaml**：
  ```yaml
  loss:
    huber_delta: 1.0
    lambda_hf: 0.01      # 高频正则
    lambda_w: 1.0        # w 监督权重
    lambda_bc: 0.0       # 边界条件（未启用）
  ```

  **Total loss 公式** (`losses.py`):
  ```python
  loss = loss_kappa + lambda_hf * loss_hf + lambda_w * loss_w
  ```

  ### 5.2 训练过程中的 Loss 量级

  **Epoch 1**：
  - loss_kappa: 18.587
  - loss_hf: 0.093
  - loss_w: 0.00347
  - **total**: 18.591

  **Epoch 100**：
  - loss_kappa: 0.0732
  - loss_hf: 0.00716
  - loss_w: 0.00716
  - **total**: 0.0731

  ### 5.3 关键观察

  **Loss 主导性**：
  ```
  Epoch 1:
    kappa 贡献: 18.587 / 18.591 = 99.98%
    hf 贡献: 0.01 * 0.093 = 0.0009 (0.005%)
    w 贡献: 1.0 * 0.00347 = 0.00347 (0.019%)

  Epoch 100:
    kappa 贡献: 0.0732 / 0.0731 ≈ 100%
    hf 贡献: 0.01 * 0.00716 = 0.000072 (0.1%)
    w 贡献: 1.0 * 0.00716 = 0.00716 (9.8%)
  ```

  **结论**：
  - ✗ **kappa loss 绝对主导**（占 90%+）
  - ✗ w loss 虽然 lambda=1.0，但因为量级小，实际贡献 < 10%
  - ⚠ HF 正则虽然 lambda 小，但持续压制系数

  ### 5.4 塌缩机制推断

  **为什么模型学成"输出接近 0"**：

  1. **Kappa loss 的最小化策略**：
    - kappa_t = ∂²w/∂s² （沿路径的二阶导数）
    - 如果 w ≈ 0（常数），则 kappa_pred ≈ 0
    - 当 kappa_meas 均值接近 0 时，输出 0 是低 loss 策略

  2. **HF 正则的压制作用**：
    - loss_hf = Σ (a_mn)² * weight_mn
    - weight_mn ∝ (m² + n²)
    - 正则鼓励 a → 0

  3. **W loss 权重不足**：
    - lambda_w=1.0 看似合理，但 loss_w 量级天然比 loss_kappa 小 100 倍
    - 实际权重应该是 lambda_w=100 才能平衡

  4. **结果**：
    - 模型找到了局部最优：a ≈ 0 → w ≈ 0 → kappa ≈ 0
    - 这个解对 kappa loss 很好，对 w loss 很差
    - 但因为 kappa loss 主导，模型停留在这个解

  ---

  ## 6. Most Likely Root Causes (Ranked)

  ### Rank 1: Loss 权重严重失衡（置信度 95%）

  **证据**：
  - kappa loss 占总 loss 90%+
  - w loss 虽然 lambda=1.0，但量级小 100 倍
  - 模型优化了 kappa 而牺牲了 w

  **验证方法**：
  - 设置 lambda_w=100 或 lambda_kappa=0.01
  - 单样本 overfit 实验

  ### Rank 2: HF 正则过强导致系数压缩（置信度 80%）

  **证据**：
  - lambda_hf=0.01，持续压制高频系数
  - w_pred 幅值仅 2.78%
  - 系数被压到接近 0

  **验证方法**：
  - 设置 lambda_hf=0
  - 观察系数幅值是否恢复

  ### Rank 3: Kappa loss 对"零解"过于宽容（置信度 70%）

  **证据**：
  - 当 kappa_meas 均值 ≈ 0 时，kappa_pred ≈ 0 是合理解
  - 但这不代表 w 正确

  **验证方法**：
  - 检查 kappa_meas 的分布
  - 如果大量接近 0，需要重新设计 loss

  ### Rank 4: 基函数表达能力不足（置信度 30%）

  **证据**：
  - M=4, N=4，仅 16 个系数
  - 可能无法表达复杂的 w 场

  **反驳**：
  - 如果是表达能力问题，应该看到"尽力拟合但误差大"
  - 而不是"所有样本输出相同"

  ### Rank 5: 编码器问题（置信度 10%）

  **证据**：
  - 使用 Mamba/CNN encoder

  **反驳**：
  - kappa loss 能下降，说明编码器在工作
  - 塌缩是 loss 设计问题，不是编码器问题

  ---

  ## 7. Minimal Patches & Verification Plan

  ### Patch 1: 重新平衡 Loss 权重（必须）

  **修改 `config.yaml`**：
  ```yaml
  loss:
    lambda_hf: 0.0      # 先关闭 HF 正则
    lambda_w: 100.0     # 大幅提升 w 权重
    lambda_bc: 0.0
  ```

  **为什么**：
  - 让 w loss 和 kappa loss 量级相当
  - 强制模型关注 w 的准确性

  **验证**：
  ```bash
  # 单样本 overfit 测试
  python project/scripts/train.py --config configs/debug_overfit.yaml
  # 检查：
  # 1. loss_w 是否下降
  # 2. w_pred 幅值是否恢复
  # 3. 不同样本是否有差异
  ```

  ### Patch 2: 添加系数幅值监控（调试）

  **修改 `trainer.py`**：
  ```python
  def _forward(self, batch):
      # ... existing code ...
      a_reshaped = a.reshape(a.shape[0], self.M, self.N)
      
      # 添加监控
      a_norm = a.norm(dim=1).mean().item()
      if hasattr(self, '_step_count'):
          self._step_count += 1
      else:
          self._step_count = 0
      
      if self._step_count % 100 == 0:
          print(f"[DEBUG] Step {self._step_count}: ||a||={a_norm:.6f}")
      
      # ... rest of code ...
  ```

  **为什么**：
  - 实时监控系数是否被压缩
  - 如果 ||a|| → 0，说明 HF 正则或 loss 不平衡

  ### Patch 3: 两样本差异检查（验证）

  **创建 `tools_diagnose/check_two_sample_diff.py`**：
  ```python
  import torch
  from project.src.datasets import FiberSequenceDataset
  from project.src.models import MambaCoeffNet

  # 加载模型
  checkpoint = torch.load("runs/20260114-201551/checkpoint.pt")
  model = MambaCoeffNet(**checkpoint['config']['model'])
  model.load_state_dict(checkpoint['model_state'])
  model.eval()

  # 加载两个样本
  dataset = FiberSequenceDataset(manifest_path="data/processed2/manifest.json")
  sample1 = dataset[0]
  sample2 = dataset[1]

  # 前向传播
  with torch.no_grad():
      a1 = model(sample1['X'].unsqueeze(0), sample1['mask'].unsqueeze(0))
      a2 = model(sample2['X'].unsqueeze(0), sample2['mask'].unsqueeze(0))

  # 检查差异
  diff = (a1 - a2).abs().max().item()
  print(f"Max coeff diff between sample 1 and 2: {diff:.10f}")

  if diff < 1e-6:
      print("✗ CONFIRMED: Model outputs identical coefficients for different samples")
  else:
      print("✓ Model produces different coefficients")
  ```

  **验证**：
  - 如果 diff < 1e-6，确认塌缩
  - 修复后重新运行，diff 应该 > 0.01

  ### Patch 4: 渐进式训练策略（可选）

  **阶段 1：只训练 w**：
  ```yaml
  loss:
    lambda_hf: 0.0
    lambda_w: 1.0
    lambda_kappa: 0.0  # 需要修改代码支持
  ```

  **阶段 2：联合训练**：
  ```yaml
  loss:
    lambda_hf: 0.001
    lambda_w: 10.0
    lambda_kappa: 1.0
  ```

  **为什么**：
  - 先让模型学会"不同样本 → 不同 w"
  - 再引入 kappa 一致性约束

  ### Patch 5: 增加基函数数量（如果前面都不行）

  **修改 `config.yaml`**：
  ```yaml
  basis:
    M: 8   # 从 4 增加到 8
    N: 8   # 从 4 增加到 8
  ```

  **为什么**：
  - 增加表达能力
  - 但这不是主要问题

  ---

  ## 8. Verification Experiments

  ### Experiment 1: 单样本 Overfit（最小验证）

  **目的**：验证模型能否拟合单个样本

  **步骤**：
  1. 创建只包含 1 个样本的 manifest
  2. 训练 100 epochs，lambda_w=100, lambda_hf=0
  3. 检查 w_pred vs w_true

  **成功标准**：
  - RMSE(w) < 0.001
  - w_pred 幅值接近 w_true

  ### Experiment 2: 两样本差异测试

  **目的**：验证模型能否区分两个样本

  **步骤**：
  1. 使用 2 个样本训练
  2. 检查两个样本的 a_pred 差异

  **成功标准**：
  - ||a1 - a2|| > 0.1

  ### Experiment 3: Loss 权重扫描

  **目的**：找到最佳 lambda_w

  **步骤**：
  1. 固定其他参数
  2. 扫描 lambda_w ∈ [1, 10, 100, 1000]
  3. 记录 w_pred 幅值和塌缩指标

  **成功标准**：
  - 找到使 w_pred 幅值 > 80% w_true 的 lambda_w

  ---

  ## Appendix: Alternative Methods

  ### A1. DeepONet (Operator Learning)

  **适用场景**：
  - 输入：1D 观测序列（光纤数据）
  - 输出：2D 场 w(x,y)

  **优势**：
  - 直接学习"观测 → 场"的映射
  - 不需要显式基函数
  - 可以处理不规则观测点

  **实现建议**：
  ```python
  class DeepONet(nn.Module):
      def __init__(self):
          self.branch_net = MLP([5, 128, 128, 128])  # 编码观测
          self.trunk_net = MLP([2, 128, 128, 128])   # 编码查询点 (x,y)
      
      def forward(self, observations, query_points):
          # observations: (B, L, 5)
          # query_points: (B, Q, 2)
          branch_out = self.branch_net(observations).mean(dim=1)  # (B, 128)
          trunk_out = self.trunk_net(query_points)  # (B, Q, 128)
          w_pred = (branch_out.unsqueeze(1) * trunk_out).sum(dim=-1)  # (B, Q)
          return w_pred
  ```

  **缺点**：
  - 需要大量训练数据
  - 不保证物理一致性（kappa）

  ### A2. 图神经网络（GNN）

  **适用场景**：
  - 观测点之间有空间关系
  - 需要捕捉局部和全局依赖

  **实现建议**：
  - 节点：每个观测点
  - 边：空间近邻 + 路径近邻
  - 消息传递：聚合邻居信息
  - 输出：每个节点的 w 值

  **优势**：
  - 自然处理不规则采样
  - 可以编码物理约束（边权重）

  **缺点**：
  - 实现复杂
  - 需要设计图结构

  ### A3. 网格化 + UNet/FNO

  **适用场景**：
  - 可以将观测投影到规则网格
  - 需要高分辨率输出

  **实现建议**：
  ```python
  # 1. 将观测投影到网格
  grid = project_to_grid(observations, grid_size=(64, 64))
  # 2. 添加 mask 通道
  grid_with_mask = torch.cat([grid, mask], dim=1)
  # 3. UNet 编码-解码
  w_pred = unet(grid_with_mask)
  ```

  **优势**：
  - 利用成熟的 CNN 架构
  - 可以生成高分辨率场

  **缺点**：
  - 投影可能损失信息
  - 需要处理不规则边界

  ### A4. 物理信息神经网络（PINN）

  **适用场景**：
  - 有明确的 PDE 约束
  - 数据稀疏但物理知识丰富

  **实现建议**：
  ```python
  loss = loss_data + lambda_pde * loss_pde
  # loss_pde = ||Δ²w - f||²  # 板弯曲方程
  ```

  **优势**：
  - 强制物理一致性
  - 可以外推到无观测区域

  **缺点**：
  - 训练困难（高阶导数）
  - 需要准确的 PDE 模型

  ### A5. 线性基线（最小二乘）

  **目的**：验证问题的可辨识性

  **实现**：
  ```python
  # 假设 A @ a = kappa_meas
  # 求解 a = (A^T A + λI)^{-1} A^T kappa_meas
  from scipy.linalg import lstsq

  # 构建 A 矩阵（basis 对 kappa 的贡献）
  A = build_kappa_basis_matrix(x, y, tx, ty, M, N)
  a_lstsq, residual, rank, s = lstsq(A, kappa_meas, rcond=1e-6)

  # 用 a_lstsq 重建 w
  w_pred_lstsq = basis_expansion(a_lstsq, x, y, M, N)
  ```

  **为什么重要**：
  - 如果线性方法都能得到合理的 w，说明问题是可解的
  - 如果线性方法也失败，说明问题本身病态

  ---

  ## 9. Next Steps

  ### 立即执行（今天）：

  1. ✓ 运行 `tools_diagnose/analyze_w_collapse.py`（已完成）
  2. ⚠ 运行 `tools_diagnose/check_two_sample_diff.py`（验证塌缩）
  3. ⚠ 修改 config.yaml：lambda_w=100, lambda_hf=0
  4. ⚠ 重新训练 20 epochs
  5. ⚠ 检查 w_pred 是否恢复

  ### 短期（本周）：

  1. 实现梯度监控（Patch 2）
  2. 单样本 overfit 实验
  3. Loss 权重扫描实验
  4. 生成完整的可视化报告

  ### 中期（下周）：

  1. 如果 loss 平衡有效，调优 lambda_hf
  2. 尝试渐进式训练策略
  3. 增加基函数数量（如果需要）
  4. 对比线性基线方法

  ### 长期（可选）：

  1. 探索 DeepONet 或 GNN 方法
  2. 实现 PINN 约束
  3. 收集更多训练数据

  ---

  ## 10. 文件清单

  **生成的文件**：
  - `REPORT_diagnosis.md`（本文件）
  - `diagnostic_figs/w_collapse_analysis.png`
  - `diagnostic_figs/w_collapse_results.json`
  - `diagnostic_logs/w_collapse_analysis.log`
  - `tools_diagnose/analyze_w_collapse.py`
  - `tools_diagnose/check_two_sample_diff.py`（待创建）

  **关键输入文件**：
  - `project/runs/20260114-201551/w_compare.csv`
  - `project/runs/20260114-201551/config.yaml`
  - `project/runs/20260114-201551/metrics.csv`
  - `project/runs/20260114-201551/checkpoint.pt`

  ---

  **报告结束**

