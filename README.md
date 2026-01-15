# Fiberglass 数据管线脚手架

<!--
工程背景:
- 输入光纤路径序列 token X=[x,y,tx,ty,kappa_t]，回归二维基函数系数 a (长度 K=M*N)。
- 通过可微算子计算 κ_pred = t^T H(w) t，与测得 κ_meas 做一致性损失。
- 若有 COMSOL 挠度 w_points 可做监督损失；欠定问题用高频正则 hf_weights(a)。
- 可选边界项约束 y=Ly 处夹持（挠度/斜率接近 0）。
-->

COMSOL 导出的光纤应变数据（列：`x y z strain u v w`）最小可运行仓库，用于完成 txt -> parquet 的处理与 manifest 生成，并提供完整的 PyTorch 训练闭环（Dataset/Loss/Trainer/脚本）。

## 背景与假设
- 数据源：COMSOL 导出为纯文本 txt，空格或 Tab 分隔，每行一个点；列为 `x, y, z, strain, u, v, w`。
- 阶段：包含完整训练闭环，模型为序列编码器（Mamba/CNN）回归系数。
- 运行环境：Python >= 3.10，Linux/macOS/Windows 均可（无平台相关代码）。
- 预处理：支持路径排序与上下表面配对（stage2），更复杂的几何校正仍未覆盖。

## 数据格式
- `x, y, z`：坐标（单位 m，默认为全局坐标系）
- `strain`：沿光纤方向的应变（或等价投影）
- `u, v, w`：位移分量（`w` 常作挠度真值）
- 分隔符：空格或 Tab；有表头需包含全部 7 列名；无表头时按列序自动命名。

## 目录结构
```
configs/           # 基础配置（包含背景与假设元数据）
  plate.yaml       # 板/数据相关占位配置（厚度、网格、固定区域等）
  train.yaml       # 训练配置占位（batch_size、lr、epochs 等）
data/
  raw/             # 放置原始 COMSOL 导出的 txt
  processed/       # prepare_dataset 输出的 parquet 与 manifest
  processed_top/   # 上表面 parquet
  processed_bottom/ # 下表面 parquet
  processed2/      # stage2 输出（单面占位）
  processed2_pair/ # stage2 输出（上下表面配对）
scripts/
  prepare_dataset.py  # txt -> parquet + manifest CLI
  prepare_stage2.py   # path 特征 + kappa_t/mask
  compute_dataset_stats.py  # 统计量计算
  train.py            # 训练入口 CLI
  eval.py             # 评估入口 CLI
src/
  file_io/comsol_txt.py # 读取、基础校验、列统计
  datasets/        # FiberSequenceDataset + collate
  losses/          # 训练损失
  train/           # Trainer + metrics
  utils/logging.py # 统一日志
tests/
  test_smoke.py    # 基础导入与读取测试
  test_train_smoke.py  # 端到端训练 smoke test
```

## 安装
```bash
python -m venv .venv
source .venv/bin/activate  # Windows 使用 .venv\\Scripts\\activate
pip install -e .
```

## 使用
- 预处理（txt -> parquet）
  ```bash
  # 单面输入
  python scripts/prepare_dataset.py --input data/raw --output data/processed

  # 上下表面输入（推荐）
  python scripts/prepare_dataset.py --input data/raw --pattern "*_top_dedup_filtered.txt" --output data/processed_top
  python scripts/prepare_dataset.py --input data/raw --pattern "*_bottom_dedup_filtered.txt" --output data/processed_bottom
  ```
  - 每个输入文件生成同名 `.parquet` 至输出目录。
  - 生成 `manifest.json`，包含处理时间、文件列表、行数、列统计、是否有表头、输出路径。

- Stage‑2（路径特征 + kappa_t + mask）
  ```bash
  # 上下表面配对：kappa_t = (strain_bot - strain_top) / h
  python scripts/prepare_stage2.py --top data/processed_top --bottom data/processed_bottom --output data/processed2_pair --h 0.005

  # 单面占位：kappa_t = strain / h
  python scripts/prepare_stage2.py --input data/processed --output data/processed2 --h 0.005
  ```
  - 配对规则：文件名需包含 `_top_`/`_bottom_`，例如 `merged_z_m0p002_top_dedup_filtered.txt` 与 `merged_z_m0p002_bottom_dedup_filtered.txt`。
  - 可选参数：`--pair-method s|xy`、`--s-tol`、`--xy-tol`、`--fill-strategy zeros|nearest`、`--order-mode as_is|nn_graph`、`--jump-threshold`、`--k`、`--dedup-eps`。

- 统计量（训练前必须）
  ```bash
  python scripts/compute_dataset_stats.py --data_dir data/processed2_pair --output data/processed2_pair/stats.json
  ```

- 训练入口
  ```bash
  python scripts/train.py --config configs/train.yaml
  ```
  - 确保 `configs/train.yaml` 指向新的数据与 stats：
    ```yaml
    data:
      manifest: "data/processed2_pair/manifest.json"
      samples_dir: "data/processed2_pair"
    stats_path: "data/processed2_pair/stats.json"
    ```
  - 训练输出写入 `runs/<timestamp>/`，包含 `config.yaml`、`metrics.csv`、`checkpoint.pt`。

- 评估入口
  ```bash
  python scripts/eval.py --checkpoint runs/<timestamp>/checkpoint.pt
  python scripts/predict.py --checkpoint runs/<timestamp>/checkpoint.pt --data_dir data/processed2_pair
  ```

- 诊断包导出（定位 kappa_pred 尺度问题）
  ```bash
  python tools_diagnose/collect_kappa_debug_pack.py --checkpoint runs/<timestamp>/checkpoint.pt --sample_id 5 --grid_nx 80 --grid_ny 80
  ```
  - 默认输出目录：`diagnostic_pack/<run_name>_sid<id>/`，包含网格 CSV、point_pack、pack_stats.json、PNG。
  - 解读要点：`scale_ratio_std/max` 判断 kappa_meas 比例问题；`hessian_fd_error`/`kappa_fd_error` 判断解析 Hessian 尺度；`t_norm_mean` 判断 tx/ty 归一化。

## 配置
- `configs/plate.yaml`：几何/网格/光纤布置等占位参数，以及项目背景元数据。
- `configs/train.yaml`：数据、训练、设备、模型与损失权重配置。

## 常见问题
- txt 无表头怎么办？使用 `--no-header`，脚本会按列序自动赋名 `x y z strain u v w`。
- 分隔符是空格还是 Tab？两者均自动支持（使用正则分隔）。
- 为什么行数或列校验失败？脚本要求至少 10 行，且 7 列必须齐全，数据中不得包含 NaN/Inf。
- 上下表面无法配对？请检查文件名是否含 `_top_`/`_bottom_`，并确保成对出现。

## 测试
```bash
pytest -q
```

## 可选：预提交检查
启用 pre-commit（已提供 `.pre-commit-config.yaml`）:
```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
```
