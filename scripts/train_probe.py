#!/usr/bin/env python3
"""Train hallucination detection methods. 

This is the unified training script for all detection methods.
All parameters are managed via Hydra configuration. 

Usage:
    # Train with default config
    python scripts/train_probe. py

    # Train specific method
    python scripts/train_probe.py method=lapeigvals
    python scripts/train_probe.py method=hypergraph
    python scripts/train_probe.py method=entropy

    # Override parameters
    python scripts/train_probe.py method=hypergraph method. params.hidden_dim=256

    # Multi-run
    python scripts/train_probe.py --multirun method=lapeigvals,entropy,hypergraph
"""
import sys
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import pickle

import hydra
from omegaconf import DictConfig, OmegaConf
import torch
import numpy as np

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    ExtractedFeatures, MethodConfig, EvalMetrics,
    set_seed, setup_logging, ensure_dir,
)
from src.methods import create_method, BaseMethod
from src. data. splitter import split_features
from src.evaluation import compute_metrics

logger = logging.getLogger(__name__)


def load_features(features_dir: Path, split: Optional[str] = None) -> List[ExtractedFeatures]:
    """Load extracted features from directory. 

    Args:
        features_dir: Directory containing features
        split: Optional split to load (train, val, test)

    Returns:
        List of ExtractedFeatures
    """
    features_list = []

    # Try loading from pickle first
    if split: 
        pkl_path = features_dir / f"{split}_features.pkl"
    else:
        pkl_path = features_dir / "features.pkl"

    if pkl_path. exists():
        logger.info(f"Loading features from {pkl_path}")
        with open(pkl_path, "rb") as f:
            data = pickle.load(f)

        if isinstance(data, dict):
            features_list = data.get("features", [])
        elif isinstance(data, list):
            features_list = data
        else:
            features_list = [data]

        return features_list

    # Try loading individual . pt files
    pt_files = sorted(features_dir.glob("*.pt"))
    if pt_files:
        logger.info(f"Loading {len(pt_files)} feature files from {features_dir}")
        for pt_file in pt_files:
            try:
                feat = ExtractedFeatures.load(pt_file)
                features_list.append(feat)
            except Exception as e:
                logger.warning(f"Failed to load {pt_file}: {e}")

    return features_list


def load_labels(labels_path: Path) -> Optional[torch.Tensor]: 
    """Load labels from file."""
    if labels_path.exists():
        labels = torch.load(labels_path)
        return labels
    return None


def check_and_split_features(
    features_list: List[ExtractedFeatures],
    cfg: DictConfig,
) -> Tuple[List[ExtractedFeatures], List[ExtractedFeatures], List[ExtractedFeatures]]: 
    """Check if features have split info, split if necessary. 

    Args: 
        features_list:  All features
        cfg:  Configuration

    Returns: 
        (train_features, val_features, test_features)
    """
    # Check if already split by checking metadata or separate files
    train_features = []
    val_features = []
    test_features = []

    for feat in features_list:
        split_info = feat.metadata.get("split", None)
        if split_info == "train":
            train_features.append(feat)
        elif split_info in ["val", "validation"]:
            val_features.append(feat)
        elif split_info == "test":
            test_features.append(feat)

    # If we found split info, use it
    if train_features or val_features or test_features:
        logger.info(f"Using existing splits: train={len(train_features)}, "
                   f"val={len(val_features)}, test={len(test_features)}")

        # If no val, split from train
        if not val_features and train_features: 
            train_features, val_features, _ = split_features(
                train_features,
                train_ratio=0.9,
                val_ratio=0.1,
                random_seed=cfg.seed,
            )

        return train_features, val_features, test_features

    # No split info, perform splitting
    logger.info("No split info found, performing auto-split")

    train_ratio = cfg.get("train_ratio", 0.8)
    val_ratio = cfg.get("val_ratio", 0.1)

    train_features, val_features, test_features = split_features(
        features_list,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        random_seed=cfg.seed,
        stratify=True,
    )

    return train_features, val_features, test_features


def save_split_features(
    train_features: List[ExtractedFeatures],
    val_features: List[ExtractedFeatures],
    test_features:  List[ExtractedFeatures],
    output_dir: Path,
):
    """Save split features to separate files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for split_name, features in [
        ("train", train_features),
        ("val", val_features),
        ("test", test_features),
    ]:
        if features:
            path = output_dir / f"{split_name}_features.pkl"
            with open(path, "wb") as f:
                pickle.dump({"features": features}, f)
            logger.info(f"Saved {len(features)} {split_name} features to {path}")


def build_method_config(cfg: DictConfig) -> MethodConfig: 
    """Build MethodConfig from Hydra config."""
    method_cfg = cfg.method

    params = OmegaConf.to_container(method_cfg.get("params", {}), resolve=True)

    return MethodConfig(
        name=method_cfg.name,
        classifier=method_cfg. get("classifier", "logistic"),
        cv_folds=method_cfg.get("cv_folds", 5),
        random_seed=cfg.seed,
        params=params,
    )


def train_method(
    method:  BaseMethod,
    train_features: List[ExtractedFeatures],
    val_features: List[ExtractedFeatures],
) -> Dict[str, float]:
    """Train method on features.

    Args:
        method: Method instance
        train_features: Training features
        val_features:  Validation features (for monitoring)

    Returns:
        Training metrics
    """
    # Get labels
    train_labels = [f.label for f in train_features]

    # Train
    logger.info(f"Training on {len(train_features)} samples...")
    metrics = method.fit(train_features, labels=train_labels, cv=True)

    return metrics


def evaluate_method(
    method: BaseMethod,
    test_features:  List[ExtractedFeatures],
) -> EvalMetrics: 
    """Evaluate method on test features. 

    Args:
        method:  Trained method
        test_features:  Test features

    Returns:
        Evaluation metrics
    """
    predictions = method.predict_batch(test_features)

    y_true = [f.label for f in test_features if f.label is not None]
    y_scores = [p.score for p in predictions[: len(y_true)]]

    return compute_metrics(y_true, y_scores)


@hydra.main(version_base=None, config_path="../config", config_name="train")
def main(cfg: DictConfig) -> float:
    """Main training entry point."""
    # Setup
    setup_logging(level=logging.INFO)
    set_seed(cfg. seed)

    logger.info("=" * 60)
    logger.info("Train Hallucination Detection Method")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")

    # Build paths
    features_dir = Path(cfg.features_dir)
    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save config
    OmegaConf.save(cfg, output_dir / "config.yaml")

    # Load features
    logger.info(f"Loading features from {features_dir}")

    # Try loading pre-split features first
    train_features = load_features(features_dir, "train")
    val_features = load_features(features_dir, "val")
    test_features = load_features(features_dir, "test")

    if not train_features: 
        # Load all and split
        all_features = load_features(features_dir)
        if not all_features:
            raise ValueError(f"No features found in {features_dir}")

        train_features, val_features, test_features = check_and_split_features(all_features, cfg)

        # Save split features for future use
        if cfg.get("save_splits", True):
            save_split_features(train_features, val_features, test_features, features_dir)

    logger.info(f"Features:  train={len(train_features)}, val={len(val_features)}, test={len(test_features)}")

    # Check labels
    n_pos = sum(1 for f in train_features if f.label == 1)
    n_neg = sum(1 for f in train_features if f.label == 0)
    logger.info(f"Training labels: {n_pos} positive, {n_neg} negative")