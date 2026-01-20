"""
LLM Provider Factory

Factory function to create LLM providers based on configuration.
"""

from __future__ import annotations

from typing import Literal

from src.nl2api.llm.protocols import LLMProvider


def create_llm_provider(
    provider: Literal["claude", "openai", "azure_openai"],
    api_key: str,
    model: str | None = None,
    **kwargs,
) -> LLMProvider:
    """
    Create an LLM provider instance.

    Args:
        provider: Provider type ("claude", "openai", or "azure_openai")
        api_key: API key for the provider
        model: Model name (uses provider default if not specified)
        **kwargs: Additional provider-specific arguments

    Returns:
        LLMProvider instance

    Raises:
        ValueError: If unknown provider is specified
    """
    if provider == "claude":
        from src.nl2api.llm.claude import ClaudeProvider

        return ClaudeProvider(
            api_key=api_key,
            model=model or "claude-sonnet-4-20250514",
            base_url=kwargs.get("base_url"),
        )

    elif provider == "openai":
        from src.nl2api.llm.openai import OpenAIProvider

        return OpenAIProvider(
            api_key=api_key,
            model=model or "gpt-4o",
            base_url=kwargs.get("base_url"),
        )

    elif provider == "azure_openai":
        from src.nl2api.llm.openai import OpenAIProvider

        if "azure_endpoint" not in kwargs:
            raise ValueError("azure_endpoint is required for Azure OpenAI")

        return OpenAIProvider(
            api_key=api_key,
            model=model or "gpt-4o",
            azure_endpoint=kwargs["azure_endpoint"],
            api_version=kwargs.get("api_version"),
        )

    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
