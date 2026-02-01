"""
Claude (Anthropic) LLM Provider

Implementation of LLMProvider using the Anthropic API.
"""

from __future__ import annotations

import asyncio
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

    Anthropic's RateLimitError inherits from APIStatusError which has
    a response object with headers. The retry-after header indicates
    how long to wait before retrying.

    Args:
        error: The exception (expected to be RateLimitError)

    Returns:
        Seconds to wait, or None if not available
    """
    try:
        # Anthropic SDK exposes headers via response.headers
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


class ClaudeProvider:
    """
    LLM provider using Anthropic's Claude API.

    Supports tool-calling via Anthropic's native tool_use API.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "claude-sonnet-4-20250514",
        base_url: str | None = None,
    ):
        """
        Initialize Claude provider.

        Args:
            api_key: Anthropic API key
            model: Model name (e.g., claude-sonnet-4-20250514)
            base_url: Optional custom base URL
        """
        from src.evalkit.common.llm import create_anthropic_client

        self._model = model
        self._client = create_anthropic_client(
            async_client=True,
            api_key=api_key,
            base_url=base_url,
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
        Generate a completion using Claude.

        Args:
            messages: Conversation history
            tools: Optional tools for the model to use
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with content and/or tool calls
        """
        with tracer.start_as_current_span("llm.complete") as span:
            span.set_attribute("llm.provider", "anthropic")
            span.set_attribute("llm.model", self._model)
            span.set_attribute("llm.temperature", temperature)
            span.set_attribute("llm.max_tokens", max_tokens)
            span.set_attribute("llm.message_count", len(messages))
            span.set_attribute("llm.tools_count", len(tools) if tools else 0)

            # Convert messages to Anthropic format
            system_message = ""
            anthropic_messages = []

            for msg in messages:
                if msg.role == MessageRole.SYSTEM:
                    system_message = msg.content
                elif msg.role == MessageRole.USER:
                    anthropic_messages.append(
                        {
                            "role": "user",
                            "content": msg.content,
                        }
                    )
                elif msg.role == MessageRole.ASSISTANT:
                    if msg.tool_calls:
                        # Assistant message with tool use
                        content = []
                        if msg.content:
                            content.append({"type": "text", "text": msg.content})
                        for tc in msg.tool_calls:
                            content.append(
                                {
                                    "type": "tool_use",
                                    "id": tc.id,
                                    "name": tc.name,
                                    "input": tc.arguments,
                                }
                            )
                        anthropic_messages.append(
                            {
                                "role": "assistant",
                                "content": content,
                            }
                        )
                    else:
                        anthropic_messages.append(
                            {
                                "role": "assistant",
                                "content": msg.content,
                            }
                        )
                elif msg.role == MessageRole.TOOL:
                    # Tool result message
                    anthropic_messages.append(
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "tool_result",
                                    "tool_use_id": msg.tool_call_id,
                                    "content": msg.content,
                                }
                            ],
                        }
                    )

            # Build request kwargs
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": anthropic_messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }

            if system_message:
                kwargs["system"] = system_message

            if tools:
                kwargs["tools"] = [t.to_anthropic_format() for t in tools]

            # Make API call
            response = await self._client.messages.create(**kwargs)

            # Parse response
            content = ""
            tool_calls: list[LLMToolCall] = []

            for block in response.content:
                if block.type == "text":
                    content = block.text
                elif block.type == "tool_use":
                    tool_calls.append(
                        LLMToolCall(
                            id=block.id,
                            name=block.name,
                            arguments=block.input if isinstance(block.input, dict) else {},
                        )
                    )

            # Record usage metrics
            span.set_attribute("llm.prompt_tokens", response.usage.input_tokens)
            span.set_attribute("llm.completion_tokens", response.usage.output_tokens)
            span.set_attribute(
                "llm.total_tokens", response.usage.input_tokens + response.usage.output_tokens
            )
            span.set_attribute("llm.tool_calls_count", len(tool_calls))
            span.set_attribute("llm.finish_reason", response.stop_reason or "stop")

            return LLMResponse(
                content=content,
                tool_calls=tuple(tool_calls),
                finish_reason=response.stop_reason or "stop",
                usage={
                    "prompt_tokens": response.usage.input_tokens,
                    "completion_tokens": response.usage.output_tokens,
                    "total_tokens": response.usage.input_tokens + response.usage.output_tokens,
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
        import anthropic

        with tracer.start_as_current_span("llm.complete_with_retry") as span:
            span.set_attribute("llm.provider", "anthropic")
            span.set_attribute("llm.model", self._model)
            span.set_attribute("llm.max_retries", max_retries)

            last_error: Exception | None = None
            for attempt in range(max_retries):
                try:
                    result = await self.complete(messages, tools, temperature, max_tokens)
                    span.set_attribute("llm.retry_attempts", attempt)
                    return result
                except anthropic.RateLimitError as e:
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
                except anthropic.APIConnectionError as e:
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
                except anthropic.InternalServerError as e:
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
