"""Prompt templates and utilities for hallucination detection."""

from .base import (
    PromptConfig,
    QaPromptConfig,
    RAGTruthPromptConfig,
    PromptTemplate,
    ChatMessage,
)
from .templates import (
    QA_PROMPT,
    QA_WITH_CONTEXT_PROMPT,
    RAGTRUTH_PROMPT,
    get_prompt_template,
)

__all__ = [
    "PromptConfig",
    "QaPromptConfig",
    "RAGTruthPromptConfig",
    "PromptTemplate",
    "ChatMessage",
    "QA_PROMPT",
    "QA_WITH_CONTEXT_PROMPT",
    "RAGTRUTH_PROMPT",
    "get_prompt_template",
]