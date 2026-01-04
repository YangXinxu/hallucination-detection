#!/usr/bin/env python3
"""Generate activations (features) for hallucination detection. 

This is the main entry point for feature extraction, following lapeigvals design.
All parameters are configured via Hydra configuration files. 

Usage:
    # Default config
    python scripts/generate_activations.py
    
    # Override dataset
    python scripts/generate_activations. py dataset=truthfulqa
    
    # Override model
    python scripts/generate_activations.py model=llama3. 1_8b
    
    # Multi-run
    python scripts/generate_activations.py --multirun dataset=ragtruth,truthfulqa
"""
import sys
import json
import logging
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    DatasetConfig, ModelConfig, FeaturesConfig, Sample, ExtractedFeatures,
    set_seed, setup_logging,
)
from src.data import get_dataset
from src.models import get_model, unload_all_models
from src. features import create_extractor

logger = logging.getLogger(__name__)


def build_output_dir(cfg: DictConfig) -> Path:
    """Build output directory path from config."""
    base_dir = Path(cfg.features_dir)
    dataset_name = cfg.dataset. name
    model_name = cfg.model.short_name or cfg.model.name. split("/")[-1]
    
    # Include temperature and prompt info if available
    temp_str = f"temp_{cfg.generation_config.temperature}"
    prompt_str = cfg.prompt. name
    seed_str = f"seed_{cfg. seed}"
    
    output_dir = base_dir / dataset_name / model_name / f"{temp_str}__{prompt_str}__{seed_str}"
    return output_dir


def save_features_lapeigvals_style(features_list: List[ExtractedFeatures], output_dir: Path) -> None:
    """Save features in lapeigvals format. 
    
    Saves:
    - attn_diags. pt: Attention diagonals [n_samples, n_layers, n_heads, seq_len]
    - laplacian_diags.pt: Laplacian diagonals
    - hidden_states. pt: Hidden states (if available)
    """
    # Collect attention diagonals
    attn_diags = []
    laplacian_diags = []
    hidden_states = []
    attn_entropy = []
    token_probs = []
    
    for feat in features_list: 
        if feat.attn_diags is not None:
            attn_diags.append(feat. attn_diags.cpu())
        if feat.laplacian_diags is not None:
            laplacian_diags.append(feat.laplacian_diags.cpu())
        if feat. hidden_states is not None:
            hidden_states.append(feat.hidden_states.cpu())
        if feat.attn_entropy is not None:
            attn_entropy.append(feat.attn_entropy.cpu())
        if feat.token_probs is not None:
            token_probs.append(feat.token_probs.cpu())
    
    if attn_diags:
        torch. save(attn_diags, output_dir / "attn_diags.pt")
        logger.info(f"Saved attention diagonals:  {len(attn_diags)} samples")
    
    if laplacian_diags:
        torch.save(laplacian_diags, output_dir / "laplacian_diags.pt")
        logger.info(f"Saved Laplacian diagonals: {len(laplacian_diags)} samples")
    
    if hidden_states: 
        torch.save(hidden_states, output_dir / "hidden_states.pt")
        logger.info(f"Saved hidden states: {len(hidden_states)} samples")
    
    if attn_entropy: 
        torch.save(attn_entropy, output_dir / "attn_entropy.pt")
        logger.info(f"Saved attention entropy: {len(attn_entropy)} samples")
    
    if token_probs:
        torch. save(token_probs, output_dir / "token_probs.pt")
        logger.info(f"Saved token probabilities: {len(token_probs)} samples")


def save_answers(samples:  List[Sample], output_path: Path) -> None:
    """Save sample answers/responses to JSON."""
    answers = []
    for sample in samples: 
        answers.append({
            "id": sample.id,
            "prompt":  sample.prompt,
            "response": sample.response,
            "reference": sample.reference,
            "label":  sample.label,
            "task_type": sample.task_type. value if sample.task_type else None,
        })
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    
    logger. info(f"Answers saved to {output_path}")


def save_metadata(samples: List[Sample], features_list: List[ExtractedFeatures], output_path: Path) -> None:
    """Save extraction metadata."""
    metadata = {
        "n_samples": len(samples),
        "n_features": len(features_list),
        "n_positive":  sum(1 for s in samples if s. label == 1),
        "n_negative": sum(1 for s in samples if s.label == 0),
        "sample_ids": [s.id for s in samples],
        "prompt_lengths": [f. prompt_len for f in features_list],
        "response_lengths": [f.response_len for f in features_list],
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {output_path}")


@hydra.main(version_base=None, config_path="../config/generation", config_name="default")
def main(cfg: DictConfig) -> None:
    """Main entry point for feature extraction."""
    
    # Setup
    setup_logging(level=logging.INFO)
    set_seed(cfg. seed)
    
    logger.info("=" * 60)
    logger.info("Generate Activations")
    logger.info("=" * 60)
    
    # Print config
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Build output directory
    output_dir = build_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Save config
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    logger.info(f"Config saved to {config_path}")
    
    # Build typed configs
    dataset_config = DatasetConfig(**OmegaConf.to_container(cfg.dataset, resolve=True))
    model_config = ModelConfig(**OmegaConf.to_container(cfg. model, resolve=True))
    features_config = FeaturesConfig(**OmegaConf.to_container(cfg. features, resolve=True))
    
    # Load dataset
    logger.info(f"Loading dataset:  {dataset_config. name}")
    dataset = get_dataset(config=dataset_config, split=dataset_config.test_split_name)
    samples = dataset.load(max_samples=dataset_config.max_samples)
    logger.info(f"Loaded {len(samples)} samples")
    
    # Count labels
    n_pos = sum(1 for s in samples if s.label == 1)
    n_neg = sum(1 for s in samples if s.label == 0)
    n_unknown = sum(1 for s in samples if s. label is None)
    logger.info(f"Labels: {n_pos} positive, {n_neg} negative, {n_unknown} unknown")
    
    # Load model
    logger.info(f"Loading model:  {model_config. name}")
    model = get_model(model_config)
    logger.info(f"Model loaded: {model.num_layers} layers, {model. num_heads} heads")
    
    # Update config with actual model architecture
    model_config.n_layers = model.num_layers
    model_config.n_heads = model.num_heads
    model_config.hidden_size = model.hidden_size
    
    # Create extractor
    extractor = create_extractor(model, features_config)
    
    # Extract features
    logger. info("Extracting features...")
    features_list = extractor.extract_batch(samples, show_progress=True)
    logger.info(f"Extracted features for {len(features_list)} samples")
    
    # Save features
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    
    # Save each feature type separately (lapeigvals style)
    save_features_lapeigvals_style(features_list, features_dir)
    
    # Also save as single pickle for convenience
    import pickle
    pickle_path = output_dir / "features.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump({
            "features":  features_list,
            "samples": samples,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, f)
    logger.info(f"Features saved to {pickle_path}")
    
    # Save answers/responses
    save_answers(samples, output_dir / "answers.json")
    
    # Save labels
    labels = torch. tensor([s.label if s.label is not None else -1 for s in samples])
    torch.save(labels, output_dir / "labels.pt")
    logger.info(f"Labels saved to {output_dir / 'labels.pt'}")
    
    # Save metadata
    save_metadata(samples, features_list, output_dir / "metadata.json")
    
    # Cleanup
    unload_all_models()
    
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__": 
    main()