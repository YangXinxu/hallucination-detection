#!/usr/bin/env python3
"""Train probe/classifier for hallucination detection."""
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional, List

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    MethodConfig, ExtractedFeatures, SplitType,
    set_seed, setup_logging,
)
from src.methods import create_method

logger = logging.getLogger(__name__)


def get_model_short_name(cfg: DictConfig) -> str:
    """Get short name for model."""
    if hasattr(cfg. model, 'short_name') and cfg.model.short_name:
        return cfg.model.short_name
    return cfg.model.name. split("/")[-1]


def find_features_dir(cfg: DictConfig) -> Path:
    """Find the features directory based on config."""
    base_dir = Path(cfg.features_dir)
    dataset_name = cfg.dataset. name

    # 处理 task_types
    task_types = cfg.dataset.get('task_types', None)
    if task_types is None or task_types == 'null':
        task_suffix = "all"
    elif isinstance(task_types, list):
        task_suffix = "_".join(task_types)
    else: 
        task_suffix = str(task_types)

    model_name = get_model_short_name(cfg)
    seed_str = f"seed_{cfg.seed}"

    features_dir = base_dir / f"{dataset_name}_{task_suffix}" / model_name / seed_str

    if not features_dir. exists():
        # 尝试查找
        pattern_dir = base_dir / f"{dataset_name}_{task_suffix}" / model_name
        if pattern_dir.exists():
            subdirs = list(pattern_dir.iterdir())
            if subdirs:
                features_dir = subdirs[0]
                logger.warning(f"Using found features directory: {features_dir}")

    return features_dir


def load_features(features_path: Path) -> tuple: 
    """Load features from pickle file."""
    with open(features_path, "rb") as f:
        data = pickle. load(f)

    features_list = data. get("features", [])
    samples = data.get("samples", [])

    return features_list, samples


def split_by_dataset_split(
    features_list: List[ExtractedFeatures],
    samples: list
) -> tuple:
    """Split features by dataset split (train/test).

    Samples should already have split field set.
    """
    train_features = []
    train_labels = []
    test_features = []
    test_labels = []

    for feat, sample in zip(features_list, samples):
        label = feat.label if feat.label is not None else (sample.label if sample.label is not None else 0)

        if sample.split and sample.split == SplitType. TRAIN:
            train_features.append(feat)
            train_labels.append(label)
        else:  # test or validation or None
            test_features.append(feat)
            test_labels.append(label)

    return train_features, train_labels, test_features, test_labels


def build_output_dir(cfg: DictConfig) -> Path:
    """Build output directory for trained model."""
    base_dir = Path(cfg.models_dir)
    dataset_name = cfg.dataset.name

    task_types = cfg. dataset.get('task_types', None)
    if task_types is None or task_types == 'null':
        task_suffix = "all"
    elif isinstance(task_types, list):
        task_suffix = "_". join(task_types)
    else:
        task_suffix = str(task_types)

    model_name = get_model_short_name(cfg)
    method_name = cfg. method.name

    # 添加 probe 子目录，避免与 evaluate 输出路径重叠
    output_dir = base_dir / f"{dataset_name}_{task_suffix}" / model_name / method_name / f"seed_{cfg.seed}" / "probe"
    return output_dir


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training."""

    setup_logging(level=logging. INFO)
    set_seed(cfg. seed)

    logger.info("=" * 60)
    logger.info("Train Probe")
    logger.info("=" * 60)
    logger.info(f"Method: {cfg.method.name}")
    logger.info(f"Classifier: {cfg.method.classifier}")

    # Find features
    features_dir = find_features_dir(cfg)
    features_path = features_dir / "features.pkl"

    if not features_path. exists():
        logger.error(f"Features not found:  {features_path}")
        logger.error("Please run generate_activations.py first")
        return

    logger.info(f"Loading features from {features_path}")

    # Load features
    features_list, samples = load_features(features_path)
    logger.info(f"Loaded {len(features_list)} feature sets")

    # Split by dataset split
    train_features, train_labels, test_features, test_labels = split_by_dataset_split(
        features_list, samples
    )
    logger.info(f"Train:  {len(train_features)}, Test: {len(test_features)}")

    # 如果没有 split 信息，使用所有数据训练
    if len(train_features) == 0:
        logger.warning("No train split found, using all data for training")
        train_features = features_list
        train_labels = [f.label if f.label is not None else 0 for f in features_list]

    n_pos = sum(1 for l in train_labels if l == 1)
    n_neg = sum(1 for l in train_labels if l == 0)
    logger.info(f"Train labels: {n_pos} positive, {n_neg} negative")

    # Build method config
    method_config = MethodConfig(**OmegaConf.to_container(cfg.method, resolve=True))

    # Create method
    logger.info(f"Creating method:  {method_config. name}")
    method = create_method(method_config. name, config=method_config)

    # Train
    logger.info("Training...")
    try:
        metrics = method.fit(train_features, train_labels, cv=True)
    except Exception as e: 
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return

    logger.info("Training complete!")
    logger.info(f"Metrics:  {metrics}")

    # Build output directory
    output_dir = build_output_dir(cfg)
    output_dir. mkdir(parents=True, exist_ok=True)

    # Save model
    model_path = output_dir / "model. pkl"
    method.save(model_path)
    logger.info(f"Model saved to {model_path}")

    # Save metrics
    metrics_path = output_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Save config
    OmegaConf.save(cfg, output_dir / "config.yaml")

    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()