#!/usr/bin/env python3
"""Evaluate a trained method on test data. 

All parameters are configured via Hydra configuration files. 

Usage:
    # Default config
    python scripts/evaluate. py
    
    # Override method
    python scripts/evaluate.py method=entropy
    
    # Specify threshold
    python scripts/evaluate.py evaluation. threshold=0.5
"""
import sys
import json
import pickle
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    MethodConfig, EvalMetrics, Prediction,
    set_seed, setup_logging,
)
from src.methods import create_method, BaseMethod
from src.evaluation import compute_metrics, find_optimal_threshold

logger = logging.getLogger(__name__)


def find_model_path(cfg: DictConfig) -> Path:
    """Find the trained model path based on config."""
    base_dir = Path(cfg.models_dir)
    dataset_name = cfg.dataset. name
    model_name = cfg.model.short_name or cfg.model.name.split("/")[-1]
    method_name = cfg. method.name
    
    model_dir = base_dir / dataset_name / model_name / method_name / f"seed_{cfg. seed}"
    model_path = model_dir / "model. pkl"
    
    return model_path


def find_features_path(cfg: DictConfig) -> Path:
    """Find the features path based on config."""
    base_dir = Path(cfg.features_dir)
    dataset_name = cfg.dataset.name
    model_name = cfg.model.short_name or cfg. model.name.split("/")[-1]
    
    temp_str = f"temp_{cfg.generation_config.temperature}"
    prompt_str = cfg.prompt. name
    seed_str = f"seed_{cfg.seed}"
    
    features_dir = base_dir / dataset_name / model_name / f"{temp_str}__{prompt_str}__{seed_str}"
    features_path = features_dir / "features. pkl"
    
    return features_path


def load_model(model_path:  Path, method_config: MethodConfig) -> BaseMethod:
    """Load trained model."""
    method = create_method(method_config.name, config=method_config)
    method.load(model_path)
    return method


def load_features(features_path: Path):
    """Load features from pickle file."""
    with open(features_path, "rb") as f:
        data = pickle. load(f)
    
    features_list = data.get("features", [])
    samples = data.get("samples", [])
    
    return features_list, samples


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for evaluation."""
    
    # Setup
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
    
    logger.info(f"Model:  {model_path}")
    logger.info(f"Features:  {features_path}")
    
    # Load model
    method_config = MethodConfig(**OmegaConf.to_container(cfg.method, resolve=True))
    method = load_model(model_path, method_config)
    logger.info(f"Loaded method: {method_config.name}")
    
    # Load features
    features_list, samples = load_features(features_path)
    labels = [f.label for f in features_list]
    logger.info(f"Loaded {len(features_list)} samples for evaluation")
    
    # Predict
    logger.info("Predicting...")
    predictions = method.predict_batch(features_list)
    logger.info(f"Generated {len(predictions)} predictions")
    
    # Get threshold
    threshold = cfg.get("evaluation", {}).get("threshold", None)
    if threshold is None:
        scores = [p.score for p in predictions]
        threshold, _ = find_optimal_threshold(scores, labels)
        logger.info(f"Optimal threshold: {threshold:. 4f}")
    else:
        logger. info(f"Using specified threshold: {threshold:. 4f}")
    
    # Compute metrics
    scores = [p.score for p in predictions]
    metrics = compute_metrics(scores, labels, threshold=threshold)
    
    logger.info("=" * 40)
    logger.info("Evaluation Results:")
    logger.info("=" * 40)
    logger.info(f"  AUROC:      {metrics. auroc:.4f}")
    logger.info(f"  AUPRC:     {metrics.auprc:.4f}")
    logger.info(f"  F1:        {metrics.f1:.4f}")
    logger.info(f"  Precision: {metrics.precision:.4f}")
    logger.info(f"  Recall:    {metrics. recall:.4f}")
    logger.info(f"  Accuracy:  {metrics. accuracy:.4f}")
    logger.info(f"  Threshold: {threshold:.4f}")
    logger.info("=" * 40)
    
    # Save results
    output_dir = model_path.parent
    results_path = output_dir / "eval_results.json"
    
    results = {
        "metrics": metrics. to_dict(),
        "threshold": threshold,
        "n_samples": len(predictions),
        "predictions": [
            {
                "sample_id": p.sample_id,
                "score": p.score,
                "predicted_label": 1 if p. score >= threshold else 0,
                "true_label": labels[i] if i < len(labels) else None,
            }
            for i, p in enumerate(predictions)
        ],
    }
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to {results_path}")
    
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()