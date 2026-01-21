"""
Query Routing Protocols

Defines the interfaces for query routing strategies and tool providers.
Designed to support both agent-based and MCP-based routing.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from src.nl2api.llm.protocols import LLMToolDefinition


@dataclass(frozen=True)
class RouterResult:
    """
    Result of query routing.

    Contains the selected domain, confidence score, and metadata.
    """

    domain: str
    confidence: float  # 0.0 - 1.0
    reasoning: str | None = None
    alternative_domains: tuple[str, ...] = ()
    cached: bool = False
    latency_ms: int = 0
    model_used: str | None = None  # For escalation tracking


@runtime_checkable
class QueryRouter(Protocol):
    """
    Protocol for query routing strategies.

    Implementations can use keyword matching, LLM tool selection,
    embeddings, or other strategies to route queries to domains.
    """

    async def route(
        self,
        query: str,
        context: dict[str, Any] | None = None,
    ) -> RouterResult:
        """
        Route a query to the appropriate domain agent.

        Args:
            query: Natural language query to route
            context: Optional context (resolved entities, conversation history, etc.)

        Returns:
            RouterResult with domain, confidence, and metadata
        """
        ...


@runtime_checkable
class ToolProvider(Protocol):
    """
    Protocol for tool discovery.

    Abstracts tool discovery to support both domain agents and MCP servers.
    This enables gradual migration to MCP without rewriting the router.
    """

    @property
    def provider_name(self) -> str:
        """Return the name/identifier of this provider."""
        ...

    @property
    def provider_description(self) -> str:
        """Return a description of what this provider handles."""
        ...

    async def list_tools(self) -> list[LLMToolDefinition]:
        """
        List available tools from this provider.

        Returns:
            List of tool definitions
        """
        ...

    async def get_tool_description(self, tool_name: str) -> str | None:
        """
        Get detailed description for a specific tool.

        Args:
            tool_name: Name of the tool

        Returns:
            Detailed description or None if tool not found
        """
        ...


@runtime_checkable
class ToolExecutor(Protocol):
    """
    Protocol for tool execution.

    Abstracts tool execution to support both domain agents and MCP servers.
    """

    async def execute(
        self,
        tool_name: str,
        arguments: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Execute a tool and return results.

        Args:
            tool_name: Name of the tool to execute
            arguments: Arguments for the tool

        Returns:
            Execution result as dictionary
        """
        ...


@dataclass(frozen=True)
class RoutingToolDefinition:
    """
    Tool definition used for routing decisions.

    A simplified version of LLMToolDefinition specifically for routing.
    The router exposes each domain/provider as a "routing tool" to let
    the LLM select the appropriate one.
    """

    name: str  # e.g., "route_to_datastream"
    description: str  # Domain description for the LLM
    capabilities: tuple[str, ...] = ()  # Data types this domain handles
    example_queries: tuple[str, ...] = ()  # Example queries for context

    def to_llm_tool(self) -> LLMToolDefinition:
        """Convert to LLMToolDefinition for use with LLM providers."""
        # Build a rich description including capabilities and examples
        description_parts = [self.description]

        if self.capabilities:
            caps = ", ".join(self.capabilities)
            description_parts.append(f"Handles: {caps}")

        if self.example_queries:
            examples = "; ".join(f'"{q}"' for q in self.example_queries[:3])
            description_parts.append(f"Examples: {examples}")

        full_description = ". ".join(description_parts)

        return LLMToolDefinition(
            name=self.name,
            description=full_description,
            parameters={
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Brief explanation for why this domain is appropriate",
                    },
                    "confidence": {
                        "type": "number",
                        "description": "Confidence score 0.0-1.0 that this is the correct domain",
                        "minimum": 0.0,
                        "maximum": 1.0,
                    },
                },
                "required": ["reasoning"],
            },
        )


def deprecated_can_handle() -> None:
    """
    Emit deprecation warning for can_handle() method.

    Called by agents that still use the legacy can_handle() interface.
    """
    warnings.warn(
        "can_handle() is deprecated and will be removed in v2.0. "
        "Use LLMToolRouter for query routing instead.",
        DeprecationWarning,
        stacklevel=3,
    )
