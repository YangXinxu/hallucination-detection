#!/usr/bin/env python3
"""Generate activations (features) for hallucination detection."""
import sys
import json
import logging
import pickle
from pathlib import Path
from typing import List

import hydra
from omegaconf import DictConfig, OmegaConf
import torch

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.core import (
    DatasetConfig, ModelConfig, FeaturesConfig, Sample, ExtractedFeatures,
    set_seed, setup_logging,
)
from src.data import get_dataset
from src.models import get_model, unload_all_models
from src.features import create_extractor

logger = logging.getLogger(__name__)


def get_model_short_name(cfg: DictConfig) -> str:
    """Get short name for model."""
    if hasattr(cfg.model, 'short_name') and cfg.model.short_name:
        return cfg.model.short_name
    # Extract from path
    model_name = cfg.model.name
    return model_name.split("/")[-1]


def build_output_dir(cfg: DictConfig) -> Path:
    """Build output directory path from config."""
    base_dir = Path(cfg.features_dir)
    dataset_name = cfg.dataset.name
    
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
    
    output_dir = base_dir / f"{dataset_name}_{task_suffix}" / model_name / seed_str
    return output_dir


def save_features_lapeigvals_style(features_list: List[ExtractedFeatures], output_dir: Path) -> None:
    """Save features in lapeigvals format."""
    attn_diags = []
    laplacian_diags = []
    hidden_states = []
    attn_entropy = []
    token_probs = []
    
    for feat in features_list:
        if feat.attn_diags is not None:
            attn_diags.append(feat.attn_diags.cpu())
        if feat.laplacian_diags is not None:
            laplacian_diags.append(feat.laplacian_diags.cpu())
        if feat.hidden_states is not None:
            hidden_states.append(feat.hidden_states.cpu())
        if feat.attn_entropy is not None:
            attn_entropy.append(feat.attn_entropy.cpu())
        if feat.token_probs is not None:
            token_probs.append(feat.token_probs.cpu())
    
    if attn_diags:
        torch.save(attn_diags, output_dir / "attn_diags.pt")
        logger.info(f"Saved attention diagonals: {len(attn_diags)} samples")
    
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
        torch.save(token_probs, output_dir / "token_probs.pt")
        logger.info(f"Saved token probabilities: {len(token_probs)} samples")


def save_answers(samples: List[Sample], output_path: Path) -> None:
    """Save sample answers/responses to JSON with hallucination span annotations.
    
    Format aligned with RAGTruth for consistency - includes hallucination spans
    with character positions when available.
    """
    answers = []
    for sample in samples:
        sample_data = {
            "id": sample.id,
            "prompt": sample.prompt,
            "response": sample.response,
            "reference": sample.reference,
            "label": sample.label,
            "task_type": sample.task_type.value if sample.task_type else None,
            "split": sample.split.value if sample.split else None,
        }
        
        # Include hallucination spans if available (RAGTruth format)
        if sample.metadata.get("hallucination_spans"):
            sample_data["labels"] = [
                {
                    "start": span.get("start", 0),
                    "end": span.get("end", 0),
                    "text": span.get("text", ""),
                    "label_type": span.get("type", span.get("label_type", "")),
                }
                for span in sample.metadata["hallucination_spans"]
            ]
        else:
            sample_data["labels"] = []
        
        answers.append(sample_data)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(answers, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Answers saved to {output_path}")


def save_metadata(samples: List[Sample], features_list: List[ExtractedFeatures], output_path: Path) -> None:
    """Save extraction metadata including hallucination token label statistics."""
    # 按 task_type 统计
    by_task = {}
    for s in samples:
        task = s.task_type.value if s.task_type else "unknown"
        if task not in by_task:
            by_task[task] = {"total": 0, "positive": 0, "negative": 0}
        by_task[task]["total"] += 1
        if s.label == 1:
            by_task[task]["positive"] += 1
        elif s.label == 0:
            by_task[task]["negative"] += 1
    
    # Count samples with token-level hallucination labels
    n_with_token_labels = sum(1 for f in features_list if f.hallucination_labels is not None)
    n_with_token_spans = sum(1 for f in features_list if f.hallucination_token_spans)
    
    metadata = {
        "n_samples": len(samples),
        "n_features": len(features_list),
        "n_positive": sum(1 for s in samples if s.label == 1),
        "n_negative": sum(1 for s in samples if s.label == 0),
        "n_with_token_labels": n_with_token_labels,
        "n_with_token_spans": n_with_token_spans,
        "by_task_type": by_task,
        "sample_ids": [s.id for s in samples],
        "prompt_lengths": [f.prompt_len for f in features_list],
        "response_lengths": [f.response_len for f in features_list],
    }
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Metadata saved to {output_path}")


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point for feature extraction."""
    
    setup_logging(level=logging.INFO)
    set_seed(cfg.seed)
    
    logger.info("=" * 60)
    logger.info("Generate Activations")
    logger.info("=" * 60)
    logger.info(f"Config:\n{OmegaConf.to_yaml(cfg)}")
    
    # Build output directory
    output_dir = build_output_dir(cfg)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Save config
    config_path = output_dir / "config.yaml"
    OmegaConf.save(cfg, config_path)
    
    # Build dataset config
    dataset_dict = OmegaConf.to_container(cfg.dataset, resolve=True)
    # 处理 task_types
    if dataset_dict.get('task_types') == 'null' or dataset_dict.get('task_types') is None:
        dataset_dict['task_types'] = None  # None 表示所有类型
    
    dataset_config = DatasetConfig(**dataset_dict)
    model_config = ModelConfig(**OmegaConf.to_container(cfg.model, resolve=True))
    features_config = FeaturesConfig(**OmegaConf.to_container(cfg.features, resolve=True))
    
    # Load dataset
    logger.info(f"Loading dataset: {dataset_config.name}")
    logger.info(f"Task types filter: {dataset_config.task_types}")
    
    dataset = get_dataset(config=dataset_config)
    samples = dataset.load(max_samples=dataset_config.max_samples)
    logger.info(f"Loaded {len(samples)} samples")
    
    # 按 task_type 统计
    task_counts = {}
    for s in samples:
        task = s.task_type.value if s.task_type else "unknown"
        task_counts[task] = task_counts.get(task, 0) + 1
    logger.info(f"Samples by task type: {task_counts}")
    
    # Count labels
    n_pos = sum(1 for s in samples if s.label == 1)
    n_neg = sum(1 for s in samples if s.label == 0)
    n_unknown = sum(1 for s in samples if s.label is None)
    logger.info(f"Labels: {n_pos} positive, {n_neg} negative, {n_unknown} unknown")
    
    # Load model
    logger.info(f"Loading model: {model_config.name}")
    model = get_model(model_config)
    logger.info(f"Model loaded: {model.num_layers} layers, {model.num_heads} heads")
    
    # Create extractor
    extractor = create_extractor(model, features_config)
    
    # Extract features
    logger.info("Extracting features...")
    features_list = extractor.extract_batch(samples, show_progress=True)
    logger.info(f"Extracted features for {len(features_list)} samples")
    
    # Save features
    features_dir = output_dir / "features"
    features_dir.mkdir(parents=True, exist_ok=True)
    save_features_lapeigvals_style(features_list, features_dir)
    
    # Save as pickle
    pickle_path = output_dir / "features.pkl"
    with open(pickle_path, "wb") as f:
        pickle.dump({
            "features": features_list,
            "samples": samples,
            "config": OmegaConf.to_container(cfg, resolve=True),
        }, f)
    logger.info(f"Features saved to {pickle_path}")
    
    # Save answers
    save_answers(samples, output_dir / "answers.json")
    
    # Save labels
    labels = torch.tensor([s.label if s.label is not None else -1 for s in samples])
    torch.save(labels, output_dir / "labels.pt")
    
    # Save metadata
    save_metadata(samples, features_list, output_dir / "metadata.json")
    
    # Cleanup
    unload_all_models()
    
    logger.info("=" * 60)
    logger.info("Done!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()