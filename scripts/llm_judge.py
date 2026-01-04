#!/usr/bin/env python3
"""Run LLM-as-Judge evaluation on samples.

All parameters are configured via Hydra configuration files. 

Usage:
    # Default config
    python scripts/llm_judge.py
    
    # Override LLM API
    python scripts/llm_judge. py llm_api=openai
    
    # Override model
    python scripts/llm_judge. py llm_api.model=gpt-4o
"""
import sys
import json
import os
import logging
from pathlib import Path
from typing import Dict, Any, List

import hydra
from omegaconf import DictConfig, OmegaConf

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path. insert(0, str(PROJECT_ROOT))

from src.core import (
    DatasetConfig, LLMAPIConfig, Sample, JudgeResult,
    set_seed, setup_logging,
)
from src.data import get_dataset
from src.evaluation import create_judge, compute_metrics

logger = logging.getLogger(__name__)


def find_output_dir(cfg:  DictConfig) -> Path:
    """Find/create output directory for judge results."""
    base_dir = Path(cfg.results_dir) / "llm_judge"
    dataset_name = cfg. dataset.name
    llm_name = cfg.llm_api.model. replace("/", "_")
    
    output_dir = base_dir / dataset_name / llm_name / f"seed_{cfg. seed}"
    return output_dir


def get_api_key(cfg: DictConfig) -> str:
    """Get API key from environment."""
    env_var = cfg.llm_api.api_key_env
    api_key = os. environ.get(env_var, "")
    
    if not api_key:
        raise ValueError(
            f"API key not found.  Please set {env_var} environment variable."
        )
    
    return api_key


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for LLM judge evaluation."""
    
    # Setup
    setup_logging(level=logging.INFO)
    set_seed(cfg.seed)
    
    logger.info("=" * 60)
    logger.info("LLM-as-Judge Evaluation")
    logger.info("=" * 60)
    
    # Get API key
    try:
        api_key = get_api_key(cfg)
    except ValueError as e:
        logger.error(str(e))
        return
    
    # Build configs
    dataset_config = DatasetConfig(**OmegaConf.to_container(cfg.dataset, resolve=True))
    llm_config = LLMAPIConfig(**OmegaConf.to_container(cfg.llm_api, resolve=True))
    
    logger.info(f"Dataset: {dataset_config.name}")
    logger.info(f"LLM:  {llm_config.provider}/{llm_config. model}")
    
    # Load dataset
    logger. info(f"Loading dataset: {dataset_config.name}")
    dataset = get_dataset(config=dataset_config, split=dataset_config.test_split_name)
    samples = dataset.load(max_samples=dataset_config.max_samples)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Create judge
    logger. info("Creating LLM judge...")
    judge = create_judge(llm_config, api_key=api_key)
    
    # Run evaluation
    logger. info("Running LLM judge evaluation...")
    results:  List[JudgeResult] = []
    
    for i, sample in enumerate(samples):
        try:
            result = judge.judge(sample)
            results.append(result)
            
            if (i + 1) % 10 == 0:
                logger.info(f"Processed {i + 1}/{len(samples)} samples")
                
        except Exception as e:
            logger.warning(f"Failed to judge sample {sample.id}: {e}")
            # Create a failed result
            results.append(JudgeResult(
                sample_id=sample.id,
                label=-1,  # Unknown
                confidence=0.0,
                explanation=f"Error: {str(e)}",
                raw_response="",
                model=llm_config. model,
            ))
    
    logger.info(f"Judged {len(results)} samples")
    
    # Statistics
    n_hallucinated = sum(1 for r in results if r.label == 1)
    n_clean = sum(1 for r in results if r.label == 0)
    n_failed = sum(1 for r in results if r. label == -1)
    
    logger.info(f"Results: {n_hallucinated} hallucinated, {n_clean} clean, {n_failed} failed")
    
    # Compare with ground truth if available
    has_labels = any(s.label is not None for s in samples)
    metrics = None
    
    if has_labels: 
        true_labels = [s.label for s in samples]
        pred_labels = [r.label for r in results]
        
        # Filter out failed predictions
        valid_idx = [i for i, r in enumerate(results) if r.label != -1]
        valid_true = [true_labels[i] for i in valid_idx]
        valid_pred = [pred_labels[i] for i in valid_idx]
        
        if valid_true and valid_pred:
            # Use confidence as score
            valid_scores = [results[i].confidence for i in valid_idx]
            metrics = compute_metrics(valid_scores, valid_true)
            
            logger.info("=" * 40)
            logger.info("Comparison with Ground Truth:")
            logger.info("=" * 40)
            logger.info(f"  AUROC:     {metrics.auroc:.4f}")
            logger.info(f"  AUPRC:     {metrics.auprc:.4f}")
            logger.info(f"  F1:        {metrics. f1:.4f}")
            logger. info(f"  Accuracy:  {metrics. accuracy:.4f}")
            logger.info("=" * 40)
    
    # Save results
    output_dir = find_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save judge results
    results_data = [r.to_dict() for r in results]
    results_path = output_dir / "judge_results.json"
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(results_data, f, ensure_ascii=False, indent=2)
    logger.info(f"Results saved to {results_path}")
    
    # Save metrics if available
    if metrics: 
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics. to_dict(), f, indent=2)
        logger.info(f"Metrics saved to {metrics_path}")
    
    # Save config
    config_path = output_dir / "config. yaml"
    OmegaConf.save(cfg, config_path)
    
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__": 
    main()