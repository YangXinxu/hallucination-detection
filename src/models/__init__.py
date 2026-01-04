"""Model loading utilities for hallucination detection."""

from .loader import (
    LoadedModel,
    load_model,
    ModelManager,
    get_model_manager,
    unload_model,
    unload_all_models,
)
from .factory import (
    get_model,
    create_model,
    get_model_from_hydra,
    get_model_info,
)

__all__ = [
    # Loader
    "LoadedModel",
    "load_model",
    "ModelManager",
    "get_model_manager",
    "unload_model",
    "unload_all_models",
    # Factory
    "get_model",
    "create_model",
    "get_model_from_hydra",
    "get_model_info",
]