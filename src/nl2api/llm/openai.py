"""
OpenAI LLM Provider

Implementation of LLMProvider using the OpenAI API.
Supports both OpenAI and Azure OpenAI endpoints.
"""

from __future__ import annotations

import asyncio
import json
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
        try:
            import openai
        except ImportError:
            raise ImportError("openai package required. Install with: pip install openai")

        self._model = model

        if azure_endpoint:
            # Azure OpenAI
            self._client = openai.AsyncAzureOpenAI(
                api_key=api_key,
                azure_endpoint=azure_endpoint,
                api_version=api_version or "2024-02-15-preview",
            )
        else:
            # Standard OpenAI
            self._client = openai.AsyncOpenAI(
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
            kwargs: dict[str, Any] = {
                "model": self._model,
                "messages": openai_messages,
                "max_tokens": max_tokens,
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
                    wait_time = 2**attempt
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
                except openai.APIConnectionError as e:
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
                except openai.InternalServerError as e:
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
