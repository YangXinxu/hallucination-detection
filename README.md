# Hallucination Detection Framework

一个可扩展的 LLM 幻觉检测框架，支持多种检测方法和数据集。

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/YangXinxu/hallucination-detection.git
cd hallucination-detection
pip install -e .
pip install -r requirements.txt
```

### 一键运行 (DVC)

```bash
# 1. 配置参数 (编辑 params.yaml)
# 默认配置: RAGTruth 全部类型 + Mistral-7B + 所有方法

# 2. 运行完整流水线
dvc repro

# 3. 查看结果
dvc metrics show
cat outputs/results/summary.json
```

**注意**: 如果遇到 `dvc.lock is git-ignored` 错误，请确保 `.gitignore` 中没有 `/dvc.lock`。DVC 需要跟踪 `dvc.lock` 文件来记录管道状态。

## 📁 项目结构

```
hallucination-detection/
├── config/
│   ├── config.yaml          # 主配置文件
│   ├── dataset/             # 数据集配置
│   ├── model/               # 模型配置
│   ├── method/              # 方法配置
│   └── features/            # 特征提取配置
├── scripts/
│   ├── generate_activations.py  # 特征提取
│   ├── train_probe.py           # 训练
│   ├── evaluate.py              # 评估
│   └── aggregate_results.py     # 结果汇总
├── src/                     # 源代码
├── params.yaml              # DVC 参数
├── dvc.yaml                 # DVC 流水线
└── outputs/                 # 输出目录
    ├── features/            # 提取的特征
    ├── models/              # 训练的模型
    └── results/             # 评估结果
```

## ⚙️ 配置说明

### params.yaml (DVC 参数)

```yaml
# 数据集配置
datasets:
  - name: ragtruth
    task_types: null  # null = 所有类型 (QA, Summary, Data2txt)
    splits: [train, test]

# 模型配置
models:
  - name: mistral_7b
    path: /path/to/model

# 方法配置 (可选择运行的方法)
methods:
  - lapeigvals
  - entropy
  - lookback_lens
  - hypergraph
  - ensemble

# 特征提取模式
features:
  mode: teacher_forcing  # teacher_forcing or generation

# 其他
seed: 42
cv_folds: 5
```

### 运行 RAGTruth 所有 task-type + teacher_forcing + 所有方法

1. **确保 `params.yaml` 配置正确**:

```yaml
datasets:
  - name: ragtruth
    task_types: null  # null = 运行所有 task_type (QA, Summary, Data2txt)
    splits: [train, test]

models:
  - name: mistral_7b
    path: /path/to/your/model

methods:
  - lapeigvals
  - entropy
  - lookback_lens
  - hypergraph
  - ensemble

features:
  mode: teacher_forcing  # 使用 teacher_forcing 模式
```

2. **运行 DVC 流水线**:

```bash
# 运行整个流水线
dvc repro

# 或强制重新运行
dvc repro -f
```

### 运行特定 task_type

```yaml
# params.yaml
datasets:
  - name: ragtruth
    task_types: [QA]  # 只运行 QA
```

### 运行多个方法

```yaml
# params.yaml
methods:
  - lapeigvals
  - entropy
  - lookback_lens
  - hypergraph
  - ensemble
```

## 🔧 命令行使用

### 单独运行各阶段

```bash
# 特征提取
python scripts/generate_activations.py dataset=ragtruth model=mistral_7b

# 训练
python scripts/train_probe.py dataset=ragtruth method=lapeigvals

# 评估
python scripts/evaluate.py dataset=ragtruth method=lapeigvals
```

### Hydra 多重运行

```bash
# 多个方法
python scripts/train_probe.py --multirun method=lapeigvals,entropy

# 多个数据集
python scripts/generate_activations.py --multirun dataset=ragtruth,truthfulqa
```

## 🔍 常见问题排查

### DVC 报错 "dvc.lock is git-ignored"

**原因**: `.gitignore` 中包含了 `/dvc.lock`，但 DVC 需要 Git 跟踪此文件。

**解决方案**: 从 `.gitignore` 中移除 `/dvc.lock` 行。

### 特征提取模式 (teacher_forcing vs generation)

- **teacher_forcing**: 使用真实的响应文本进行特征提取，适合训练和评估
- **generation**: 让模型自动生成响应，适合实际应用场景

## 📊 输出格式

### eval_results.json

```json
{
  "metrics": {
    "auroc": 0.8234,
    "auprc": 0.7891,
    "f1": 0.7456
  },
  "by_task_type": {
    "QA": {"auroc": 0.82, "f1": 0.75, "n_samples": 150},
    "Summary": {"auroc": 0.85, "f1": 0.78, "n_samples": 150},
    "Data2txt": {"auroc": 0.80, "f1": 0.72, "n_samples": 150}
  },
  "n_samples": 450
}
```

## 📝 License

MIT License