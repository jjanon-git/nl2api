"""
Claude (Anthropic) LLM Provider

Implementation of LLMProvider using the Anthropic API.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from src.common.telemetry import get_tracer
from src.nl2api.llm.protocols import (
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    LLMToolDefinition,
    MessageRole,
)

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


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
        try:
            import anthropic
        except ImportError:
            raise ImportError("anthropic package required. Install with: pip install anthropic")

        self._model = model
        self._client = anthropic.AsyncAnthropic(
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
                    wait_time = 2**attempt  # Exponential backoff
                    span.add_event(
                        "retry",
                        {
                            "attempt": attempt + 1,
                            "error_type": "rate_limit",
                            "wait_time": wait_time,
                        },
                    )
                    logger.warning(
                        f"Rate limited, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                except anthropic.APIConnectionError as e:
                    last_error = e
                    wait_time = 2**attempt
                    span.add_event(
                        "retry",
                        {
                            "attempt": attempt + 1,
                            "error_type": "connection_error",
                            "wait_time": wait_time,
                        },
                    )
                    logger.warning(
                        f"Connection error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)
                except anthropic.InternalServerError as e:
                    last_error = e
                    wait_time = 2**attempt
                    span.add_event(
                        "retry",
                        {
                            "attempt": attempt + 1,
                            "error_type": "server_error",
                            "wait_time": wait_time,
                        },
                    )
                    logger.warning(
                        f"Server error, retrying in {wait_time}s (attempt {attempt + 1}/{max_retries})"
                    )
                    await asyncio.sleep(wait_time)

            # All retries exhausted
            span.set_attribute("llm.retry_exhausted", True)
            span.set_attribute("llm.retry_attempts", max_retries)
            if last_error:
                raise last_error
            raise RuntimeError("Unexpected error during retry")
