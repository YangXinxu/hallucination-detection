"""Data loading module for hallucination detection. 

Provides: 
- Dataset classes for various benchmarks
- Factory functions for dataset creation
- Formatters for prompt preparation
"""

from .base import BaseDataset, JsonDataset, JsonlDataset
from . factory import get_dataset, prepare_dataset, load_dataset_with_labels
from . formatter import (
    DatasetFormatter,
    QaFormatter,
    RAGTruthFormatter,
    HaluEvalFormatter,
    TruthfulQAFormatter,
    get_formatter,
)
from .ragtruth import RAGTruthDataset
from .truthfulqa import TruthfulQADataset
from .halueval import HaluEvalDataset, HaluEvalQADataset

__all__ = [
    # Base
    "BaseDataset",
    "JsonDataset",
    "JsonlDataset",
    # Factory
    "get_dataset",
    "prepare_dataset",
    "load_dataset_with_labels",
    # Formatters
    "DatasetFormatter",
    "QaFormatter",
    "RAGTruthFormatter",
    "HaluEvalFormatter",
    "TruthfulQAFormatter",
    "get_formatter",
    # Datasets
    "RAGTruthDataset",
    "TruthfulQADataset",
    "HaluEvalDataset",
    "HaluEvalQADataset",
]