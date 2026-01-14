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
- 约束：当前不做路径排序、上下表面配对等复杂几何预处理。

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
scripts/
  prepare_dataset.py  # txt -> parquet + manifest CLI
  train.py            # 训练入口 CLI
  eval.py             # 评估入口 CLI
src/
  io/comsol_txt.py # 读取、基础校验、列统计
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
- 预处理（生成 parquet 与 manifest）
  ```bash
  python scripts/prepare_dataset.py --input data/raw --output data/processed
  # 指定无表头:
  python scripts/prepare_dataset.py --input data/raw --no-header
  # 自定义模式:
  python scripts/prepare_dataset.py --input data/raw --pattern "*.dat"

  PYTHONPATH=.:src python scripts/prepare_stage2.py --input data/processed --output data/processed2

  ```
  - 每个输入文件生成同名 `.parquet` 至 `data/processed/`。
  - 生成 `manifest.json`，包含处理时间、文件列表、行数、列统计（min/max/mean）、是否有表头、输出路径。
  ```bash
  python operators/curvature_projection.py
  ```
- 训练入口
  ```bash
  PYTHONPATH=.:src python scripts/train.py --config configs/train.yaml
  ```
  - 自动读取 manifest 或单个 parquet 文件并训练。
  - 训练输出写入 `runs/<timestamp>/`，包含 `config.yaml`、`metrics.csv`、`checkpoint.pt`。

- 评估入口
  ```bash
  python scripts/eval.py --checkpoint runs/<timestamp>/checkpoint.pt
  ```

## 配置
- `configs/plate.yaml`：几何/网格/光纤布置等占位参数，以及项目背景元数据。
- `configs/train.yaml`：数据、训练、设备、模型与损失权重配置。

## 常见问题
- txt 无表头怎么办？使用 `--no-header`，脚本会按列序自动赋名 `x y z strain u v w`。
- 分隔符是空格还是 Tab？两者均自动支持（使用正则分隔）。
- 为什么行数或列校验失败？脚本要求至少 10 行，且 7 列必须齐全，数据中不得包含 NaN/Inf。

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
