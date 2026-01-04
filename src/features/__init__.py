"""Feature extraction module for hallucination detection.

Provides:
- FeatureExtractor: Main class for extracting features
- Utility functions for attention, hidden states, and token probabilities
- Hallucination span to token-level label conversion

Example:
    from src.features import FeatureExtractor, create_extractor
    from src.models import load_model
    from src.core import FeaturesConfig
    
    extractor = FeatureExtractor(model, FeaturesConfig(
        mode="teacher_forcing",
        attention_layers="last_n:4",
        hidden_states_layers="last_n:4",
    ))
    
    features = extractor.extract(sample)
    # features.attn_diags: [n_layers, n_heads, seq_len]
    # features.laplacian_diags: [n_layers, n_heads, seq_len]
    # features.hidden_states: [n_layers, hidden_size]
    # features.token_probs: [response_len]
    # features.hallucination_labels: [seq_len] - 0/1 for each token
"""

from .extractor import (
    # Main extractor
    FeatureExtractor,
    create_extractor,
    
    # Attention utilities
    extract_attention_diagonal,
    compute_laplacian_diagonal,
    compute_attention_entropy,
    stack_layer_attentions,
    
    # Hidden state utilities
    pool_hidden_states,
    stack_layer_hidden_states,
    
    # Token probability utilities
    compute_token_probs,
    compute_token_entropy,
    compute_top_k_probs,
)

from .hallucination_spans import (
    calculate_hallucination_token_spans,
    get_token_hallucination_labels,
    extract_hallucination_info_from_sample,
    calculate_hallucination_labels_for_input,
    needs_llm_judge_for_spans,
    HallucinationSpanInfo,
)

__all__ = [
    "FeatureExtractor",
    "create_extractor",
    "extract_attention_diagonal",
    "compute_laplacian_diagonal",
    "compute_attention_entropy",
    "stack_layer_attentions",
    "pool_hidden_states",
    "stack_layer_hidden_states",
    "compute_token_probs",
    "compute_token_entropy",
    "compute_top_k_probs",
    # Hallucination span utilities
    "calculate_hallucination_token_spans",
    "get_token_hallucination_labels",
    "extract_hallucination_info_from_sample",
    "calculate_hallucination_labels_for_input",
    "needs_llm_judge_for_spans",
    "HallucinationSpanInfo",
]
