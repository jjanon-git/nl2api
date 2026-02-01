"""
OpenAI LLM Provider

Implementation of LLMProvider using the OpenAI API.
Supports both OpenAI and Azure OpenAI endpoints.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
from typing import Any

from src.evalkit.common.telemetry import get_tracer
from src.nl2api.llm.protocols import (
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
    MessageRole,
)

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


def _extract_retry_after(error: Exception) -> float | None:
    """
    Extract retry-after value from rate limit error headers.

    OpenAI's RateLimitError inherits from APIStatusError which has
    a response object with headers. The retry-after header indicates
    how long to wait before retrying.

    Args:
        error: The exception (expected to be RateLimitError)

    Returns:
        Seconds to wait, or None if not available
    """
    try:
        # OpenAI SDK exposes headers via response.headers
        if hasattr(error, "response") and error.response is not None:
            headers = getattr(error.response, "headers", {})
            retry_after = headers.get("retry-after")
            if retry_after:
                return float(retry_after)
    except (ValueError, TypeError, AttributeError):
        pass
    return None


def _calculate_wait_time(
    attempt: int,
    retry_after: float | None = None,
    base_delay: float = 1.0,
    max_delay: float = 60.0,
    jitter_factor: float = 0.25,
) -> float:
    """
    Calculate wait time with retry-after header support and jitter.

    If retry_after is provided (from API headers), uses that as the base.
    Otherwise falls back to exponential backoff.
    Adds jitter to prevent thundering herd when multiple workers hit rate limits.

    Args:
        attempt: Current retry attempt (0-indexed)
        retry_after: Seconds from retry-after header (if available)
        base_delay: Base delay for exponential backoff
        max_delay: Maximum delay cap
        jitter_factor: Random jitter as fraction of wait time (0.25 = ±25%)

    Returns:
        Seconds to wait before next retry
    """
    if retry_after is not None:
        # Use API-provided retry-after, but cap it
        wait = min(retry_after, max_delay)
    else:
        # Exponential backoff: 1s, 2s, 4s, 8s, ...
        wait = min(base_delay * (2**attempt), max_delay)

    # Add jitter to prevent thundering herd
    # Random value in range [wait * (1 - jitter), wait * (1 + jitter)]
    jitter = wait * jitter_factor * (2 * random.random() - 1)
    return max(0.1, wait + jitter)  # Never wait less than 100ms


class OpenAIProvider:
    """
    LLM provider using OpenAI's API.

    Supports tool-calling via OpenAI's function calling API.
    Compatible with both OpenAI and Azure OpenAI endpoints.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o",
        base_url: str | None = None,
        azure_endpoint: str | None = None,
        api_version: str | None = None,
    ):
        """
        Initialize OpenAI provider.

        For standard OpenAI:
            OpenAIProvider(api_key="sk-...", model="gpt-4o")

        For Azure OpenAI:
            OpenAIProvider(
                api_key="...",
                model="deployment-name",
                azure_endpoint="https://xxx.openai.azure.com",
                api_version="2024-02-15-preview"
            )

        Args:
            api_key: OpenAI or Azure API key
            model: Model name or Azure deployment name
            base_url: Optional custom base URL for OpenAI
            azure_endpoint: Azure OpenAI endpoint URL
            api_version: Azure OpenAI API version
        """
        from src.evalkit.common.llm import create_openai_client

        self._model = model
        self._client = create_openai_client(
            async_client=True,
            api_key=api_key,
            base_url=base_url,
            azure_endpoint=azure_endpoint,
            api_version=api_version,
        )

    @property
    def model_name(self) -> str:
        """Return the model name."""
        return self._model

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Generate a completion using OpenAI.

        Args:
            messages: Conversation history
            tools: Optional tools for the model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with content and/or tool calls
        """
        with tracer.start_as_current_span("llm.complete") as span:
            span.set_attribute("llm.provider", "openai")
            span.set_attribute("llm.model", self._model)
            span.set_attribute("llm.temperature", temperature)
            span.set_attribute("llm.max_tokens", max_tokens)
            span.set_attribute("llm.message_count", len(messages))
            span.set_attribute("llm.tools_count", len(tools) if tools else 0)

            # Convert messages to OpenAI format
            openai_messages = []

            for msg in messages:
                if msg.role == MessageRole.SYSTEM:
                    openai_messages.append(
                        {
                            "role": "system",
                            "content": msg.content,
                        }
                    )
                elif msg.role == MessageRole.USER:
                    openai_messages.append(
                        {
                            "role": "user",
                            "content": msg.content,
                        }
                    )
                elif msg.role == MessageRole.ASSISTANT:
                    assistant_msg: dict[str, Any] = {
                        "role": "assistant",
                        "content": msg.content or None,
                    }
                    if msg.tool_calls:
                        assistant_msg["tool_calls"] = [
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.name,
                                    "arguments": json.dumps(tc.arguments),
                                },
                            }
                            for tc in msg.tool_calls
                        ]
                    openai_messages.append(assistant_msg)
                elif msg.role == MessageRole.TOOL:
                    openai_messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": msg.tool_call_id,
                            "content": msg.content,
                        }
                    )

            # Build request kwargs
            # gpt-5-nano and newer models require max_completion_tokens instead of max_tokens
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": openai_messages,
                "max_completion_tokens": max_tokens,
                "temperature": temperature,
            }

            if tools:
                kwargs["tools"] = [t.to_openai_format() for t in tools]

            # Make API call
            response = await self._client.chat.completions.create(**kwargs)

            # Parse response
            choice = response.choices[0]
            content = choice.message.content or ""
            tool_calls: list[LLMToolCall] = []

            if choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    try:
                        arguments = json.loads(tc.function.arguments)
                    except json.JSONDecodeError:
                        arguments = {}
                    tool_calls.append(
                        LLMToolCall(
                            id=tc.id,
                            name=tc.function.name,
                            arguments=arguments,
                        )
                    )

            # Record usage metrics
            prompt_tokens = response.usage.prompt_tokens if response.usage else 0
            completion_tokens = response.usage.completion_tokens if response.usage else 0
            total_tokens = response.usage.total_tokens if response.usage else 0
            span.set_attribute("llm.prompt_tokens", prompt_tokens)
            span.set_attribute("llm.completion_tokens", completion_tokens)
            span.set_attribute("llm.total_tokens", total_tokens)
            span.set_attribute("llm.tool_calls_count", len(tool_calls))
            span.set_attribute("llm.finish_reason", choice.finish_reason or "stop")

            return LLMResponse(
                content=content,
                tool_calls=tuple(tool_calls),
                finish_reason=choice.finish_reason or "stop",
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": total_tokens,
                },
            )

    async def complete_with_retry(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> LLMResponse:
        """
        Generate a completion with automatic retry on transient errors.

        Args:
            messages: Conversation history
            tools: Optional tools for the model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            max_retries: Maximum retry attempts

        Returns:
            LLMResponse with content and/or tool calls
        """
        import openai

        with tracer.start_as_current_span("llm.complete_with_retry") as span:
            span.set_attribute("llm.provider", "openai")
            span.set_attribute("llm.model", self._model)
            span.set_attribute("llm.max_retries", max_retries)

            last_error: Exception | None = None
            for attempt in range(max_retries):
                try:
                    result = await self.complete(messages, tools, temperature, max_tokens)
                    span.set_attribute("llm.retry_attempts", attempt)
                    return result
                except openai.RateLimitError as e:
                    last_error = e
                    # Honor retry-after header from API if available
                    retry_after = _extract_retry_after(e)
                    wait_time = _calculate_wait_time(attempt, retry_after)
                    span.add_event(
                        "retry",
                        {
                            "attempt": attempt + 1,
                            "error_type": "rate_limit",
                            "wait_time": wait_time,
                            "retry_after_header": retry_after,
                        },
                    )
                    logger.warning(
                        f"Rate limited, retrying in {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries}, "
                        f"retry-after={retry_after})"
                    )
                    await asyncio.sleep(wait_time)
                except openai.APIConnectionError as e:
                    last_error = e
                    wait_time = _calculate_wait_time(attempt)
                    span.add_event(
                        "retry",
                        {
                            "attempt": attempt + 1,
                            "error_type": "connection_error",
                            "wait_time": wait_time,
                        },
                    )
                    logger.warning(
                        f"Connection error, retrying in {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                except openai.InternalServerError as e:
                    last_error = e
                    wait_time = _calculate_wait_time(attempt)
                    span.add_event(
                        "retry",
                        {
                            "attempt": attempt + 1,
                            "error_type": "server_error",
                            "wait_time": wait_time,
                        },
                    )
                    logger.warning(
                        f"Server error, retrying in {wait_time:.1f}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)

            # All retries exhausted
            span.set_attribute("llm.retry_exhausted", True)
            span.set_attribute("llm.retry_attempts", max_retries)
            if last_error:
                raise last_error
            raise RuntimeError("Unexpected error during retry")
