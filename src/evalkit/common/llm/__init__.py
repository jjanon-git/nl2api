"""
Shared LLM Client Infrastructure

Provides factory functions and utilities for creating LLM clients.
All OpenAI and Anthropic client usage should go through this module
to ensure consistent configuration (e.g., max_completion_tokens for OpenAI).
"""

from src.evalkit.common.llm.clients import (
    anthropic_message_create,
    create_anthropic_client,
    create_openai_client,
    openai_chat_completion,
)

__all__ = [
    "create_openai_client",
    "create_anthropic_client",
    "openai_chat_completion",
    "anthropic_message_create",
]
