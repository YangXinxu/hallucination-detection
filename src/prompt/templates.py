"""Predefined prompt templates."""
from __future__ import annotations
from typing import Optional

from .base import PromptTemplate


# Basic QA prompt
QA_PROMPT = PromptTemplate(
    name="qa",
    template="Question: {question}\nAnswer:",
)

# QA with context
QA_WITH_CONTEXT_PROMPT = PromptTemplate(
    name="qa_with_context",
    template="Context:  {context}\n\nQuestion: {question}\nAnswer:",
)

# RAGTruth (passthrough)
RAGTRUTH_PROMPT = PromptTemplate(
    name="ragtruth",
    template="{prompt}",
)

# TruthfulQA
TRUTHFULQA_PROMPT = PromptTemplate(
    name="truthfulqa",
    template="Q: {question}\nA:",
)

# HaluEval QA
HALUEVAL_QA_PROMPT = PromptTemplate(
    name="halueval_qa",
    template="Question: {question}\nAnswer:",
)

# Template registry
_TEMPLATES = {
    "qa": QA_PROMPT,
    "qa_with_context": QA_WITH_CONTEXT_PROMPT,
    "ragtruth":  RAGTRUTH_PROMPT,
    "truthfulqa":  TRUTHFULQA_PROMPT,
    "halueval_qa": HALUEVAL_QA_PROMPT,
}


def get_prompt_template(name: str) -> Optional[PromptTemplate]:
    """Get prompt template by name. 
    
    Args:
        name: Template name
        
    Returns:
        PromptTemplate or None if not found
    """
    return _TEMPLATES.get(name.lower())


def register_template(name: str, template: PromptTemplate) -> None:
    """Register a new prompt template.
    
    Args:
        name: Template name
        template: PromptTemplate instance
    """
    _TEMPLATES[name.lower()] = template