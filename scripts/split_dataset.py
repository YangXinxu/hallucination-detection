#!/usr/bin/env python3
"""Split dataset into train/test sets. 

This script is a preprocessing step for datasets that don't have split information.
It reads the dataset, applies train/test splitting, and saves the result. 

Usage:
    python scripts/split_dataset. py dataset.name=ragtruth dataset. train_ratio=0.9

After running this script, the dataset will have split information that can be
used by generate_activations.py, train_probe.py, and evaluate.py. 
"""
import sys
import json
import logging
import pickle
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    DatasetConfig, Sample, SplitType,
    set_seed, setup_logging,
)
from src.data import get_dataset, auto_split_dataset, DatasetSplitter

logger = logging.getLogger(__name__)


def save_samples(samples: List[Sample], output_path: Path) -> None:
    """Save samples to JSON file."""
    data = [s.to_dict() for s in samples]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved {len(samples)} samples to {output_path}")


def save_samples_pickle(samples: List[Sample], output_path: Path) -> None:
    """Save samples to pickle file."""
    with open(output_path, "wb") as f:
        pickle.dump(samples, f)
    logger.info(f"Saved {len(samples)} samples to {output_path}")


def print_split_statistics(samples: List[Sample]) -> None:
    """Print statistics about the split."""
    by_split = {}
    by_split_label = {}
    
    for s in samples:
        split_name = s.split. value if s.split else "none"
        by_split[split_name] = by_split.get(split_name, 0) + 1
        
        key = (split_name, s. label)
        by_split_label[key] = by_split_label.get(key, 0) + 1
    
    logger.info("=" * 50)
    logger.info("Split Statistics:")
    logger.info("=" * 50)
    
    for split_name, count in sorted(by_split. items()):
        n_pos = by_split_label. get((split_name, 1), 0)
        n_neg = by_split_label.get((split_name, 0), 0)
        logger.info(f"  {split_name}: {count} samples (pos={n_pos}, neg={n_neg})")
    
    logger.info("=" * 50)


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for dataset splitting."""
    
    setup_logging(level=logging.INFO)
    set_seed(cfg. seed)
    
    logger.info("=" * 60)
    logger.info("Split Dataset")
    logger.info("=" * 60)
    
    # Build dataset config
    dataset_dict = OmegaConf.to_container(cfg.dataset, resolve=True)
    if dataset_dict.get('task_types') == 'null' or dataset_dict.get('task_types') is None:
        dataset_dict['task_types'] = None
    
    dataset_config = DatasetConfig(**dataset_dict)
    
    logger.info(f"Dataset:  {dataset_config. name}")
    logger.info(f"Path: {dataset_config.path}")
    
    # Get train_ratio from config
    train_ratio = dataset_dict.get('train_ratio', 0.9)
    split_seed = dataset_dict.get('split_seed', None) or cfg.seed
    force_split = dataset_dict.get('force_split', False)
    
    logger.info(f"Train ratio: {train_ratio}")
    logger.info(f"Split seed: {split_seed}")
    logger.info(f"Force split:  {force_split}")
    
    # Load dataset
    logger.info("Loading dataset...")
    dataset = get_dataset(config=dataset_config)
    samples = dataset.load(max_samples=dataset_config.max_samples)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Check existing splits
    n_with_split = sum(1 for s in samples if s.split is not None)
    logger.info(f"Samples with existing split info: {n_with_split}/{len(samples)}")
    
    if n_with_split == len(samples) and not force_split:
        logger. info("All samples already have split information.  Use force_split=True to re-split.")
        print_split_statistics(samples)
        return
    
    # Apply splitting
    logger.info("Applying train/test split...")
    samples = auto_split_dataset(
        samples=samples,
        train_ratio=train_ratio,
        seed=split_seed,
        config=dataset_config,
        force_split=force_split,
    )
    
    # Print statistics
    print_split_statistics(samples)
    
    # Build output directory
    output_dir = Path(cfg.get('output_dir', 'outputs')) / "split_data" / dataset_config. name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save results
    save_samples(samples, output_dir / "samples. json")
    save_samples_pickle(samples, output_dir / "samples.pkl")
    
    # Save config
    OmegaConf.save(cfg, output_dir / "config.yaml")
    
    # Save split info
    split_info = {
        "dataset":  dataset_config.name,
        "train_ratio": train_ratio,
        "split_seed": split_seed,
        "total_samples": len(samples),
        "train_samples": sum(1 for s in samples if s. split == SplitType. TRAIN),
        "test_samples":  sum(1 for s in samples if s.split == SplitType.TEST),
    }
    with open(output_dir / "split_info.json", "w") as f:
        json.dump(split_info, f, indent=2)
    
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()