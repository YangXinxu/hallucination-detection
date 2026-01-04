"""Base classes for prompt handling."""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class PromptConfig(BaseModel, extra="forbid"):
    """Base prompt configuration."""
    name: str = "default"
    cls_path: Optional[str] = None
    content: str = "{question}"


class QaPromptConfig(PromptConfig, extra="forbid"):
    """QA prompt configuration."""
    question_key: str = "question"
    context_key: Optional[str] = None
    num_few_shot_examples: Optional[int] = None


class RAGTruthPromptConfig(PromptConfig, extra="forbid"):
    """RAGTruth specific prompt configuration."""
    use_original_prompt: bool = True
    question_key: str = "prompt"
    context_key: Optional[str] = None


@dataclass
class ChatMessage:
    """Single chat message."""
    role: str  # "system", "user", "assistant"
    content: str
    
    def to_dict(self) -> Dict[str, str]:
        return {"role": self.role, "content": self.content}


@dataclass
class PromptTemplate: 
    """Template for generating prompts."""
    name: str
    template: str
    system_message: Optional[str] = None
    few_shot_examples:  List[Dict[str, str]] = field(default_factory=list)
    
    def format(self, **kwargs) -> str:
        """Format template with given arguments."""
        return self.template.format(**kwargs)
    
    def to_messages(self, **kwargs) -> List[ChatMessage]:
        """Convert to chat messages format."""
        messages = []
        
        if self.system_message:
            messages.append(ChatMessage(role="system", content=self.system_message))
        
        for example in self.few_shot_examples:
            messages.append(ChatMessage(role="user", content=example["question"]))
            messages.append(ChatMessage(role="assistant", content=example["answer"]))
        
        user_content = self.format(**kwargs)
        messages.append(ChatMessage(role="user", content=user_content))
        
        return messages