"""
LLM Provider Abstraction Layer

Provides a unified interface for interacting with different LLM providers
(Claude, OpenAI) using their native tool-calling APIs.
"""

from src.nl2api.llm.factory import create_llm_provider
from src.nl2api.llm.protocols import (
    LLMMessage,
    LLMProvider,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
    MessageRole,
)

__all__ = [
    "LLMMessage",
    "LLMProvider",
    "LLMResponse",
    "LLMToolCall",
    "LLMToolDefinition",
    "MessageRole",
    "create_llm_provider",
]
