"""
Shared LLM Client Factories

Provides factory functions for creating OpenAI and Anthropic clients
with consistent configuration. This is the SINGLE source of truth for
LLM client creation across the codebase.

Key features:
- OpenAI uses `max_completion_tokens` (required for gpt-5-nano and newer)
- Consistent API key resolution from environment variables
- Support for both sync and async clients
- Azure OpenAI support
"""

from __future__ import annotations

import logging
import os
from typing import TYPE_CHECKING, Any, Literal, overload

if TYPE_CHECKING:
    from anthropic import Anthropic, AsyncAnthropic
    from openai import AsyncAzureOpenAI, AsyncOpenAI, OpenAI

logger = logging.getLogger(__name__)


# =============================================================================
# OpenAI Client Factory
# =============================================================================


def get_openai_api_key() -> str:
    """
    Get OpenAI API key from environment.

    Checks in order:
    1. OPENAI_API_KEY
    2. NL2API_OPENAI_API_KEY

    Raises:
        RuntimeError: If no API key is found
    """
    api_key = os.getenv("OPENAI_API_KEY") or os.getenv("NL2API_OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OpenAI API key required. Set OPENAI_API_KEY or NL2API_OPENAI_API_KEY")
    return api_key


@overload
def create_openai_client(
    *,
    async_client: Literal[True] = True,
    api_key: str | None = None,
    base_url: str | None = None,
    azure_endpoint: str | None = None,
    api_version: str | None = None,
) -> AsyncOpenAI | AsyncAzureOpenAI: ...


@overload
def create_openai_client(
    *,
    async_client: Literal[False],
    api_key: str | None = None,
    base_url: str | None = None,
    azure_endpoint: str | None = None,
    api_version: str | None = None,
) -> OpenAI: ...


def create_openai_client(
    *,
    async_client: bool = True,
    api_key: str | None = None,
    base_url: str | None = None,
    azure_endpoint: str | None = None,
    api_version: str | None = None,
) -> AsyncOpenAI | AsyncAzureOpenAI | OpenAI:
    """
    Create an OpenAI client with consistent configuration.

    Args:
        async_client: If True (default), returns AsyncOpenAI. If False, returns OpenAI.
        api_key: API key. If None, reads from environment.
        base_url: Optional custom base URL for OpenAI.
        azure_endpoint: If provided, creates Azure OpenAI client instead.
        api_version: Azure API version (default: 2024-02-15-preview).

    Returns:
        OpenAI client (async or sync based on async_client parameter)

    Raises:
        ImportError: If openai package is not installed
        RuntimeError: If no API key is found
    """
    try:
        import openai
    except ImportError:
        raise ImportError("openai package required. Install with: pip install openai")

    resolved_key = api_key or get_openai_api_key()

    if azure_endpoint:
        if async_client:
            return openai.AsyncAzureOpenAI(
                api_key=resolved_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version or "2024-02-15-preview",
            )
        else:
            return openai.AzureOpenAI(
                api_key=resolved_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version or "2024-02-15-preview",
            )
    else:
        if async_client:
            return openai.AsyncOpenAI(api_key=resolved_key, base_url=base_url)
        else:
            return openai.OpenAI(api_key=resolved_key, base_url=base_url)


def _is_gpt5_model(model: str) -> bool:
    """Check if model is a GPT-5 reasoning model (doesn't support temperature)."""
    return model.startswith("gpt-5") or model.startswith("o1") or model.startswith("o3")


async def openai_chat_completion(
    client: AsyncOpenAI | AsyncAzureOpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Make an OpenAI chat completion request with correct parameters.

    IMPORTANT: This function uses `max_completion_tokens` instead of `max_tokens`
    because gpt-5-nano and newer models require it.

    For GPT-5 reasoning models:
    - temperature is NOT supported (removed from request)
    - reasoning_effort controls determinism: "minimal", "low", "medium", "high"
    - Use "minimal" for deterministic judge/evaluation tasks

    Args:
        client: OpenAI async client
        model: Model name
        messages: Chat messages
        max_tokens: Maximum tokens for completion (passed as max_completion_tokens)
        temperature: Sampling temperature (ignored for GPT-5 models)
        reasoning_effort: Reasoning effort for GPT-5 models ("minimal", "low", "medium", "high")
        tools: Optional tools for function calling
        tool_choice: Optional tool choice specification
        **kwargs: Additional parameters passed to the API

    Returns:
        OpenAI ChatCompletion response
    """
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        **kwargs,
    }

    # GPT-5 reasoning models don't support temperature - use reasoning_effort instead
    if _is_gpt5_model(model):
        # Default to "minimal" for deterministic outputs (judge/eval use case)
        effort = reasoning_effort or "minimal"
        request_kwargs["reasoning"] = {"effort": effort}
        # Note: temperature parameter is intentionally omitted for GPT-5 models
    else:
        request_kwargs["temperature"] = temperature

    if tools:
        request_kwargs["tools"] = tools
    if tool_choice:
        request_kwargs["tool_choice"] = tool_choice

    return await client.chat.completions.create(**request_kwargs)


def openai_chat_completion_sync(
    client: OpenAI,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    reasoning_effort: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | str | None = None,
    **kwargs: Any,
) -> Any:
    """
    Make a synchronous OpenAI chat completion request with correct parameters.

    IMPORTANT: This function uses `max_completion_tokens` instead of `max_tokens`
    because gpt-5-nano and newer models require it.

    For GPT-5 reasoning models:
    - temperature is NOT supported (removed from request)
    - reasoning_effort controls determinism: "minimal", "low", "medium", "high"
    - Use "minimal" for deterministic judge/evaluation tasks

    Args:
        client: OpenAI sync client
        model: Model name
        messages: Chat messages
        max_tokens: Maximum tokens for completion (passed as max_completion_tokens)
        temperature: Sampling temperature (ignored for GPT-5 models)
        reasoning_effort: Reasoning effort for GPT-5 models ("minimal", "low", "medium", "high")
        tools: Optional tools for function calling
        tool_choice: Optional tool choice specification
        **kwargs: Additional parameters passed to the API

    Returns:
        OpenAI ChatCompletion response
    """
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_completion_tokens": max_tokens,
        **kwargs,
    }

    # GPT-5 reasoning models don't support temperature - use reasoning_effort instead
    if _is_gpt5_model(model):
        # Default to "minimal" for deterministic outputs (judge/eval use case)
        effort = reasoning_effort or "minimal"
        request_kwargs["reasoning"] = {"effort": effort}
        # Note: temperature parameter is intentionally omitted for GPT-5 models
    else:
        request_kwargs["temperature"] = temperature

    if tools:
        request_kwargs["tools"] = tools
    if tool_choice:
        request_kwargs["tool_choice"] = tool_choice

    return client.chat.completions.create(**request_kwargs)


# =============================================================================
# Anthropic Client Factory
# =============================================================================


def get_anthropic_api_key() -> str:
    """
    Get Anthropic API key from environment.

    Checks in order:
    1. ANTHROPIC_API_KEY
    2. NL2API_ANTHROPIC_API_KEY
    3. RAG_UI_ANTHROPIC_API_KEY

    Raises:
        RuntimeError: If no API key is found
    """
    api_key = (
        os.getenv("ANTHROPIC_API_KEY")
        or os.getenv("NL2API_ANTHROPIC_API_KEY")
        or os.getenv("RAG_UI_ANTHROPIC_API_KEY")
    )
    if not api_key:
        raise RuntimeError(
            "Anthropic API key required. Set ANTHROPIC_API_KEY, "
            "NL2API_ANTHROPIC_API_KEY, or RAG_UI_ANTHROPIC_API_KEY"
        )
    return api_key


@overload
def create_anthropic_client(
    *,
    async_client: Literal[True] = True,
    api_key: str | None = None,
    base_url: str | None = None,
) -> AsyncAnthropic: ...


@overload
def create_anthropic_client(
    *,
    async_client: Literal[False],
    api_key: str | None = None,
    base_url: str | None = None,
) -> Anthropic: ...


def create_anthropic_client(
    *,
    async_client: bool = True,
    api_key: str | None = None,
    base_url: str | None = None,
) -> AsyncAnthropic | Anthropic:
    """
    Create an Anthropic client with consistent configuration.

    Args:
        async_client: If True (default), returns AsyncAnthropic. If False, returns Anthropic.
        api_key: API key. If None, reads from environment.
        base_url: Optional custom base URL.

    Returns:
        Anthropic client (async or sync based on async_client parameter)

    Raises:
        ImportError: If anthropic package is not installed
        RuntimeError: If no API key is found
    """
    try:
        import anthropic
    except ImportError:
        raise ImportError("anthropic package required. Install with: pip install anthropic")

    resolved_key = api_key or get_anthropic_api_key()

    if async_client:
        return anthropic.AsyncAnthropic(api_key=resolved_key, base_url=base_url)
    else:
        return anthropic.Anthropic(api_key=resolved_key, base_url=base_url)


async def anthropic_message_create(
    client: AsyncAnthropic,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """
    Make an Anthropic message creation request with consistent parameters.

    Args:
        client: Anthropic async client
        model: Model name
        messages: Chat messages
        max_tokens: Maximum tokens for response
        temperature: Sampling temperature
        system: Optional system prompt
        tools: Optional tools for function calling
        tool_choice: Optional tool choice specification
        **kwargs: Additional parameters passed to the API

    Returns:
        Anthropic Message response
    """
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs,
    }

    if system:
        request_kwargs["system"] = system
    if tools:
        request_kwargs["tools"] = tools
    if tool_choice:
        request_kwargs["tool_choice"] = tool_choice

    return await client.messages.create(**request_kwargs)


def anthropic_message_create_sync(
    client: Anthropic,
    *,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int = 4096,
    temperature: float = 0.0,
    system: str | None = None,
    tools: list[dict[str, Any]] | None = None,
    tool_choice: dict[str, Any] | None = None,
    **kwargs: Any,
) -> Any:
    """
    Make a synchronous Anthropic message creation request.

    Args:
        client: Anthropic sync client
        model: Model name
        messages: Chat messages
        max_tokens: Maximum tokens for response
        temperature: Sampling temperature
        system: Optional system prompt
        tools: Optional tools for function calling
        tool_choice: Optional tool choice specification
        **kwargs: Additional parameters passed to the API

    Returns:
        Anthropic Message response
    """
    request_kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        **kwargs,
    }

    if system:
        request_kwargs["system"] = system
    if tools:
        request_kwargs["tools"] = tools
    if tool_choice:
        request_kwargs["tool_choice"] = tool_choice

    return client.messages.create(**request_kwargs)
