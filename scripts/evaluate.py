#!/usr/bin/env python3
"""Evaluate a trained method on test data."""
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    MethodConfig, EvalMetrics, Prediction,
    set_seed, setup_logging,
)
from src.methods import create_method, BaseMethod
from src.evaluation import compute_metrics, find_optimal_threshold

logger = logging.getLogger(__name__)


def get_model_short_name(cfg: DictConfig) -> str:
    """Get short name for model."""
    if hasattr(cfg.model, 'short_name') and cfg.model.short_name:
        return cfg.model.short_name
    return cfg.model.name.split("/")[-1]


def get_task_suffix(cfg: DictConfig) -> str:
    """Get task type suffix for paths."""
    task_types = cfg.dataset.get('task_types', None)
    if task_types is None or task_types == 'null':
        return "all"
    elif isinstance(task_types, list):
        return "_".join(task_types)
    return str(task_types)


def find_model_path(cfg: DictConfig) -> Path:
    """Find the trained model path based on config."""
    base_dir = Path(cfg.models_dir)
    dataset_name = cfg.dataset.name
    task_suffix = get_task_suffix(cfg)
    model_name = get_model_short_name(cfg)
    method_name = cfg.method.name
    
    model_dir = base_dir / f"{dataset_name}_{task_suffix}" / model_name / method_name / f"seed_{cfg.seed}"
    model_path = model_dir / "model.pkl"
    
    return model_path


def find_features_path(cfg: DictConfig) -> Path:
    """Find the features path based on config."""
    base_dir = Path(cfg.features_dir)
    dataset_name = cfg.dataset.name
    task_suffix = get_task_suffix(cfg)
    model_name = get_model_short_name(cfg)
    
    features_dir = base_dir / f"{dataset_name}_{task_suffix}" / model_name / f"seed_{cfg.seed}"
    features_path = features_dir / "features.pkl"
    
    return features_path


def load_model(model_path: Path, method_config: MethodConfig) -> BaseMethod:
    """Load trained model."""
    method = create_method(method_config.name, config=method_config)
    method.load(model_path)
    return method


def load_features(features_path: Path):
    """Load features from pickle file."""
    with open(features_path, "rb") as f:
        data = pickle.load(f)
    
    features_list = data.get("features", [])
    samples = data.get("samples", [])
    
    return features_list, samples


def get_test_data(features_list, samples):
    """Get test split data."""
    test_features = []
    test_labels = []
    
    for feat, sample in zip(features_list, samples):
        # 使用 test split
        if sample.split and sample.split.value == "test":
            test_features.append(feat)
            test_labels.append(feat.label)
    
    # 如果没有 test split，使用所有数据
    if len(test_features) == 0:
        logger.warning("No test split found, using all data for evaluation")
        test_features = features_list
        test_labels = [f.label for f in features_list]
    
    return test_features, test_labels


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation."""
    
    setup_logging(level=logging.INFO)
    set_seed(cfg.seed)
    
    logger.info("=" * 60)
    logger.info("Evaluate")
    logger.info("=" * 60)
    
    # Find paths
    model_path = find_model_path(cfg)
    features_path = find_features_path(cfg)
    
    if not model_path.exists():
        logger.error(f"Model not found: {model_path}")
        logger.error("Please run train_probe.py first")
        return
    
    if not features_path.exists():
        logger.error(f"Features not found: {features_path}")
        logger.error("Please run generate_activations.py first")
        return
    
    logger.info(f"Model: {model_path}")
    logger.info(f"Features: {features_path}")
    
    # Load model
    method_config = MethodConfig(**OmegaConf.to_container(cfg.method, resolve=True))
    method = load_model(model_path, method_config)
    logger.info(f"Loaded method: {method_config.name}")
    
    # Load features
    features_list, samples = load_features(features_path)
    
    # Get test data
    test_features, test_labels = get_test_data(features_list, samples)
    logger.info(f"Test samples: {len(test_features)}")
    
    n_pos = sum(1 for l in test_labels if l == 1)
    n_neg = sum(1 for l in test_labels if l == 0)
    logger.info(f"Test labels: {n_pos} positive, {n_neg} negative")
    
    # Predict
    logger.info("Predicting...")
    predictions = method.predict_batch(test_features)
    logger.info(f"Generated {len(predictions)} predictions")
    
    # Get threshold
    threshold = cfg.get("evaluation", {}).get("threshold", None)
    scores = [p.score for p in predictions]
    
    if threshold is None:
        threshold, _ = find_optimal_threshold(scores, test_labels)
        logger.info(f"Optimal threshold: {threshold:.4f}")
    else:
        logger.info(f"Using specified threshold: {threshold:.4f}")
    
    # Compute metrics
    metrics = compute_metrics(scores, test_labels, threshold=threshold)
    
    logger.info("=" * 40)
    logger.info("Evaluation Results:")
    logger.info("=" * 40)
    logger.info(f"  AUROC:     {metrics.auroc:.4f}")
    logger.info(f"  AUPRC:     {metrics.auprc:.4f}")
    logger.info(f"  F1:        {metrics.f1:.4f}")
    logger.info(f"  Precision: {metrics.precision:.4f}")
    logger.info(f"  Recall:    {metrics.recall:.4f}")
    logger.info(f"  Accuracy:  {metrics.accuracy:.4f}")
    logger.info(f"  Threshold: {threshold:.4f}")
    logger.info("=" * 40)
    
    # Save results
    output_dir = model_path.parent
    results_path = output_dir / "eval_results.json"
    
    # 按 task_type 分组的结果
    by_task = {}
    for i, (feat, pred) in enumerate(zip(test_features, predictions)):
        task = feat.metadata.get("task_type_str", "unknown")
        if task not in by_task:
            by_task[task] = {"scores": [], "labels": []}
        by_task[task]["scores"].append(pred.score)
        by_task[task]["labels"].append(test_labels[i])
    
    task_metrics = {}
    for task, data in by_task.items():
        if len(set(data["labels"])) > 1:  # 至少有两类
            task_m = compute_metrics(data["scores"], data["labels"], threshold=threshold)
            task_metrics[task] = {
                "auroc": task_m.auroc,
                "auprc": task_m.auprc,
                "f1": task_m.f1,
                "n_samples": len(data["labels"]),
                "n_positive": sum(1 for l in data["labels"] if l == 1),
            }
    
    results = {
        "metrics": metrics.to_dict(),
        "by_task_type": task_metrics,
        "threshold": threshold,
        "n_samples": len(predictions),
        "config": {
            "dataset": cfg.dataset.name,
            "model": get_model_short_name(cfg),
            "method": cfg.method.name,
            "seed": cfg.seed,
        }
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    # Print by task type
    if task_metrics:
        logger.info("\nResults by Task Type:")
        logger.info("-" * 60)
        for task, m in task_metrics.items():
            logger.info(f"  {task}: AUROC={m['auroc']:.4f}, F1={m['f1']:.4f}, N={m['n_samples']}")
    
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()