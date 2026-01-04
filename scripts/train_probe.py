#!/usr/bin/env python3
"""Train probe/classifier for hallucination detection. 

Trains a detection method on extracted features. 
All parameters are configured via Hydra configuration files. 

Usage:
    # Default config
    python scripts/train_probe. py
    
    # Override method
    python scripts/train_probe.py method=entropy
    
    # Override classifier
    python scripts/train_probe.py method. classifier=random_forest
    
    # Multi-run
    python scripts/train_probe.py --multirun method=lapeigvals,entropy,lookback_lens
"""
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, Optional

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    MethodConfig, FeaturesConfig,
    set_seed, setup_logging,
)
from src.methods import create_method

logger = logging.getLogger(__name__)


def find_features_dir(cfg: DictConfig) -> Path:
    """Find the features directory based on config."""
    base_dir = Path(cfg.features_dir)
    dataset_name = cfg.dataset. name
    model_name = cfg.model.short_name or cfg.model.name.split("/")[-1]
    
    # Match the pattern from generate_activations
    temp_str = f"temp_{cfg.generation_config.temperature}"
    prompt_str = cfg.prompt. name
    seed_str = f"seed_{cfg.seed}"
    
    features_dir = base_dir / dataset_name / model_name / f"{temp_str}__{prompt_str}__{seed_str}"
    
    if not features_dir. exists():
        # Try to find any matching directory
        pattern_dir = base_dir / dataset_name / model_name
        if pattern_dir.exists():
            subdirs = list(pattern_dir.iterdir())
            if subdirs:
                features_dir = subdirs[0]
                logger.warning(f"Using found features directory: {features_dir}")
    
    return features_dir


def load_features(features_path: Path):
    """Load features from pickle file."""
    with open(features_path, "rb") as f:
        data = pickle.load(f)
    
    features_list = data. get("features", [])
    samples = data.get("samples", [])
    
    return features_list, samples


def build_output_dir(cfg: DictConfig, features_dir: Path) -> Path:
    """Build output directory for trained model."""
    base_dir = Path(cfg.models_dir)
    dataset_name = cfg.dataset. name
    model_name = cfg.model.short_name or cfg.model.name.split("/")[-1]
    method_name = cfg. method.name
    
    output_dir = base_dir / dataset_name / model_name / method_name / f"seed_{cfg.seed}"
    return output_dir


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for training."""
    
    # Setup
    setup_logging(level=logging.INFO)
    set_seed(cfg.seed)
    
    logger.info("=" * 60)
    logger.info("Train Probe")
    logger.info("=" * 60)
    
    # Print config
    logger. info(f"Method: {cfg.method.name}")
    logger.info(f"Classifier: {cfg. method.classifier}")
    
    # Find features
    features_dir = find_features_dir(cfg)
    features_path = features_dir / "features. pkl"
    
    if not features_path. exists():
        logger.error(f"Features not found:  {features_path}")
        logger.error("Please run generate_activations. py first")
        return
    
    logger.info(f"Loading features from {features_path}")
    
    # Load features
    features_list, samples = load_features(features_path)
    logger.info(f"Loaded {len(features_list)} feature sets")
    
    # Get labels
    labels = [f.label for f in features_list]
    n_pos = sum(1 for l in labels if l == 1)
    n_neg = sum(1 for l in labels if l == 0)
    logger.info(f"Labels: {n_pos} positive, {n_neg} negative")
    
    # Build method config
    method_config = MethodConfig(**OmegaConf.to_container(cfg.method, resolve=True))
    
    # Create method
    logger.info(f"Creating method: {method_config.name}")
    method = create_method(method_config. name, config=method_config)
    
    # Train
    logger.info("Training...")
    try:
        metrics = method.fit(features_list, labels, cv=True)
    except Exception as e: 
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return
    
    logger.info("Training complete!")
    logger.info(f"Metrics: {metrics}")
    
    # Build output directory
    output_dir = build_output_dir(cfg, features_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save model
    model_path = output_dir / "model. pkl"
    method.save(model_path)
    logger.info(f"Model saved to {model_path}")
    
    # Save metrics
    metrics_path = output_dir / "train_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Metrics saved to {metrics_path}")
    
    # Save config
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()