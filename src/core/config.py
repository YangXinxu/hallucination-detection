"""Configuration management using Pydantic models. 

Design aligned with lapeigvals for compatibility. 
All configs use Pydantic BaseModel for validation and serialization. 
"""
from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Dict, Any, Union, Literal
import re

from pydantic import BaseModel, Field, model_validator

from .types import ExtractionMode, StorageMode, TaskType


# ==============================================================================
# Layer Selection - Unified parsing
# ==============================================================================

def parse_layers(spec: Union[str, List[int], None], n_layers: int) -> List[int]: 
    """Parse layer specification to concrete indices. 
    
    Args:
        spec:  Layer specification, can be:
            - "all": All layers [0, 1, .. ., n_layers-1]
            - "first": First layer only [0]
            - "last": Last layer only [n_layers-1]
            - "first_n: k": First k layers [0, 1, ..., k-1]
            - "last_n: k": Last k layers [n_layers-k, .. ., n_layers-1]
            - [0, 4, 8, ...]: Explicit list of indices
            - "[0, 4, 8]": String representation of list
            - None:  Defaults to "all"
        n_layers: Total number of layers in model
        
    Returns:
        List of layer indices (0-indexed, validated)
    """
    if spec is None:
        spec = "all"
    
    if isinstance(spec, (list, tuple)):
        return [i for i in spec if 0 <= i < n_layers]
    
    spec = str(spec).strip().lower()
    
    if spec == "all":
        return list(range(n_layers))
    if spec == "first":
        return [0]
    if spec == "last": 
        return [n_layers - 1]
    
    match = re.match(r"(first|last)_n[:\s]*(\d+)", spec)
    if match:
        mode, n = match. groups()
        n = int(n)
        if mode == "first":
            return list(range(min(n, n_layers)))
        else: 
            return list(range(max(0, n_layers - n), n_layers))
    
    try:
        cleaned = spec.strip("[]() ")
        if cleaned: 
            indices = [int(x. strip()) for x in cleaned.split(",")]
            return [i for i in indices if 0 <= i < n_layers]
    except ValueError:
        pass
    
    return list(range(n_layers))


# ==============================================================================
# Pydantic Configuration Models
# ==============================================================================

class DatasetConfig(BaseModel, extra="forbid"):
    """Dataset configuration."""
    name: str
    path: Optional[Path] = None
    cls_path: Optional[str] = None
    subset: Optional[str] = None
    
    splits: Optional[List[str]] = None
    task_types: Optional[List[str]] = None
    max_samples: Optional[int] = None
    exclude_quality: Optional[List[str]] = None
    
    max_answer_tokens:  int = 256
    target_column_name: str = "answer"
    test_split_name: Optional[str] = None


class ModelConfig(BaseModel, extra="forbid"):
    """Model configuration."""
    name: str
    short_name: Optional[str] = None
    
    n_layers: int = 32
    n_heads: int = 32
    hidden_size: int = 4096
    context_size: int = 8192
    
    dtype: str = "bfloat16"
    device_map: str = "auto"
    trust_remote_code: bool = True
    attn_implementation: str = "eager"
    load_in_4bit: bool = False
    load_in_8bit: bool = False
    
    tokenizer_name:  Optional[str] = None
    tokenizer_padding_side:  Literal["left", "right"] = "left"
    
    quantization: Optional[Dict[str, Any]] = None
    
    @model_validator(mode="after")
    def set_defaults(self) -> "ModelConfig":
        if self.tokenizer_name is None:
            self. tokenizer_name = self.name
        if self.short_name is None: 
            self.short_name = self. name.split("/")[-1].replace("-", "_").lower()
        return self


class PromptConfig(BaseModel, extra="forbid"):
    """Base prompt configuration."""
    name: str = "default"
    cls_path: Optional[str] = None
    content: str = "{question}"


class QaPromptConfig(PromptConfig, extra="forbid"):
    """QA prompt configuration."""
    question_key: str = "question"
    context_key: Optional[str] = None
    num_few_shot_examples:  Optional[int] = None


class RAGTruthPromptConfig(PromptConfig, extra="forbid"):
    """RAGTruth specific prompt configuration."""
    use_original_prompt: bool = True
    question_key: str = "prompt"
    context_key: Optional[str] = None


class FeaturesConfig(BaseModel, extra="forbid"):
    """Feature extraction configuration."""
    mode: str = "teacher_forcing"
    stored_features: str = "attention_diags"
    
    attention_enabled: bool = True
    attention_layers: Union[str, List[int]] = "all"
    attention_storage:  str = "diagonal"
    
    hidden_states_enabled: bool = True
    hidden_states_layers:  Optional[Union[str, List[int]]] = "last_n:4"
    hidden_states_pooling: Optional[str] = "last_token"
    
    token_probs_enabled:  bool = True
    token_probs_top_k: int = 10
    
    max_length: int = 4096
    batch_size: int = 1
    
    def get_attention_layers(self, n_layers: int) -> List[int]:
        if not self.attention_enabled:
            return []
        return parse_layers(self.attention_layers, n_layers)
    
    def get_hidden_layers(self, n_layers: int) -> List[int]: 
        if not self. hidden_states_enabled:
            return []
        return parse_layers(self.hidden_states_layers, n_layers)


class GenerationConfig(BaseModel, extra="forbid"):
    """Generation configuration."""
    max_new_tokens:  int = 256
    temperature: float = 0.7
    top_p:  float = 0.9
    top_k: int = 50
    do_sample: bool = True
    repetition_penalty: float = 1.0


class MethodConfig(BaseModel, extra="forbid"):
    """Detection method configuration."""
    name: str
    cls_path:  Optional[str] = None
    
    classifier: str = "logistic"
    cv_folds: int = 5
    random_seed: int = 42
    
    params: Dict[str, Any] = Field(default_factory=dict)


class LLMAPIConfig(BaseModel, extra="forbid"):
    """LLM API configuration for judge."""
    provider: str = "qwen"
    model: str = "qwen-plus"
    api_key_env: str = "DASHSCOPE_API_KEY"
    api_key: Optional[str] = None  # Can be set directly or loaded from env
    base_url: Optional[str] = None
    system_prompt: Optional[str] = None
    
    temperature: float = 0.0
    max_tokens: int = 1024
    timeout: int = 60
    max_retries: int = 3
    rate_limit: int = 60


class Config(BaseModel):
    """Main configuration container."""
    dataset: DatasetConfig
    model: ModelConfig
    prompt: Union[PromptConfig, QaPromptConfig, RAGTruthPromptConfig] = Field(
        default_factory=PromptConfig
    )
    features: FeaturesConfig = Field(default_factory=FeaturesConfig)
    generation_config: GenerationConfig = Field(default_factory=GenerationConfig)
    method: MethodConfig = Field(default_factory=lambda: MethodConfig(name="lapeigvals"))
    llm_api: LLMAPIConfig = Field(default_factory=LLMAPIConfig)
    
    seed: int = 42
    device: str = "cuda"
    output_dir: str = "outputs"
    results_dir:  str = "outputs/results"
    features_dir: str = "outputs/features"
    models_dir: str = "outputs/models"
    
    def get_output_path(self) -> Path:
        """Get output path based on config."""
        return Path(self.results_dir) / self.dataset.name / self. model.short_name


# ==============================================================================
# Config Loading Utilities
# ==============================================================================

def load_config_from_hydra(cfg) -> Config:
    """Convert Hydra DictConfig to typed Config."""
    from omegaconf import OmegaConf
    cfg_dict = OmegaConf. to_container(cfg, resolve=True)
    return Config(**cfg_dict)


def save_config(cfg:  Config, path: Path) -> None:
    """Save config to YAML file."""
    import yaml
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        yaml. dump(cfg.model_dump(), f, default_flow_style=False, sort_keys=False)


def print_config(cfg:  Config) -> None:
    """Print configuration."""
    import json
    print(json.dumps(cfg.model_dump(), indent=2, default=str))