# Hallucination Detection Framework

一个可扩展的 LLM 幻觉检测框架，支持多种检测方法和数据集。

## 🚀 快速开始

### 安装

```bash
git clone https://github.com/DwightEd/hallucination-detection. git
cd hallucination-detection
pip install -e . 
pip install -r requirements.txt
```

### 一键运行 (DVC)

```bash
# 1. 配置参数 (编辑 params.yaml)
# 默认配置:  RAGTruth 全部类型 + Mistral-7B + LapEigvals

# 2. 运行完整流水线
dvc repro

# 3. 查看结果
dvc metrics show
cat outputs/results/summary.json
```

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

# 方法配置
methods: 
  - lapeigvals

# 其他
seed: 42
```

### 运行特定 task_type

```yaml
# params.yaml
datasets:
  - name: ragtruth
    task_types:  [QA]  # 只运行 QA
```

### 运行多个方法

```yaml
# params.yaml
methods:
  - lapeigvals
  - entropy
  - lookback_lens
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
    "Summary": {"auroc":  0.85, "f1":  0.78, "n_samples": 150},
    "Data2txt": {"auroc":  0.80, "f1":  0.72, "n_samples":  150}
  },
  "n_samples": 450
}
```

## 📝 License

MIT License