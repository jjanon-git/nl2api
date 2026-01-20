"""
LLM Provider Protocols

Defines the interface for LLM providers supporting tool-calling.
Uses typing.Protocol for duck-typed interface definitions.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class MessageRole(str, Enum):
    """Role of a message in the conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


@dataclass(frozen=True)
class LLMToolDefinition:
    """
    Definition of a tool that the LLM can call.

    Uses JSON Schema format for parameters, compatible with both
    OpenAI and Anthropic tool-calling APIs.
    """

    name: str
    description: str
    parameters: dict[str, Any] = field(default_factory=dict)

    def to_openai_format(self) -> dict[str, Any]:
        """Convert to OpenAI function format."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }

    def to_anthropic_format(self) -> dict[str, Any]:
        """Convert to Anthropic tool format."""
        return {
            "name": self.name,
            "description": self.description,
            "input_schema": self.parameters,
        }


@dataclass(frozen=True)
class LLMToolCall:
    """
    A tool call made by the LLM.

    Represents a single invocation of a tool with arguments.
    """

    id: str
    name: str
    arguments: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LLMMessage:
    """
    A message in the conversation.

    Can represent user input, assistant responses, or tool results.
    """

    role: MessageRole
    content: str = ""
    tool_calls: tuple[LLMToolCall, ...] = ()
    tool_call_id: str | None = None  # For tool result messages
    name: str | None = None  # Tool name for tool result messages


@dataclass(frozen=True)
class LLMResponse:
    """
    Response from an LLM completion request.

    Contains the generated content and any tool calls.
    """

    content: str = ""
    tool_calls: tuple[LLMToolCall, ...] = ()
    finish_reason: str = "stop"
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def has_tool_calls(self) -> bool:
        """Check if the response contains tool calls."""
        return len(self.tool_calls) > 0


@runtime_checkable
class LLMProvider(Protocol):
    """
    Protocol for LLM providers supporting tool-calling.

    Implementations must provide async completion with optional tools.
    Both Anthropic Claude and OpenAI APIs are supported.
    """

    @property
    def model_name(self) -> str:
        """Return the model name/identifier."""
        ...

    async def complete(
        self,
        messages: list[LLMMessage],
        tools: list[LLMToolDefinition] | None = None,
        temperature: float = 0.0,
        max_tokens: int = 4096,
    ) -> LLMResponse:
        """
        Generate a completion for the given messages.

        Args:
            messages: Conversation history
            tools: Optional list of tools the model can call
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens in response

        Returns:
            LLMResponse with content and/or tool calls
        """
        ...

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
            tools: Optional list of tools the model can call
            temperature: Sampling temperature (0.0 for deterministic)
            max_tokens: Maximum tokens in response
            max_retries: Maximum number of retry attempts

        Returns:
            LLMResponse with content and/or tool calls
        """
        ...
