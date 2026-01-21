"""
Domain Agent Protocols

Defines the interface for domain-specific agents that translate
natural language queries into API calls.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

from CONTRACTS import ToolCall


@dataclass(frozen=True)
class AgentContext:
    """
    Context passed to a domain agent for processing.

    Contains the user query, resolved entities, and retrieved context.
    """

    # User input
    query: str
    expanded_query: str | None = None  # After context expansion

    # Resolved entities
    resolved_entities: dict[str, str] = field(default_factory=dict)
    # e.g., {"Apple Inc.": "AAPL.O", "Microsoft": "MSFT.O"}

    # Retrieved context from RAG
    field_codes: list[dict[str, Any]] = field(default_factory=list)
    # e.g., [{"code": "TR.EPSMean", "description": "Mean EPS estimate"}]

    query_examples: list[dict[str, Any]] = field(default_factory=list)
    # e.g., [{"query": "What is Apple's EPS?", "api_call": {...}}]

    # Conversation history (for multi-turn)
    # Can be a string (formatted history text) or list of turn dicts
    conversation_history: str | list[dict[str, Any]] = ""

    # Session metadata
    session_id: str | None = None
    turn_number: int = 1


@dataclass(frozen=True)
class AgentResult:
    """
    Result from a domain agent's processing.

    Contains the generated API calls and metadata.
    """

    # Success case
    tool_calls: tuple[ToolCall, ...] = ()
    confidence: float = 0.0
    reasoning: str = ""

    # Clarification case
    needs_clarification: bool = False
    clarification_questions: tuple[str, ...] = ()

    # Metadata
    domain: str = ""
    raw_llm_response: str = ""
    tokens_used: int = 0

    # Metrics fields (for observability)
    used_llm: bool = True  # Whether LLM was used (vs rule-based)
    rule_matched: str | None = None  # Which rule pattern matched (if any)
    tokens_prompt: int = 0  # Prompt tokens used
    tokens_completion: int = 0  # Completion tokens used
    llm_model: str | None = None  # Model used for processing


@runtime_checkable
class DomainAgent(Protocol):
    """
    Protocol for domain-specific agents.

    Each agent handles a specific API domain (e.g., Estimates, Fundamentals).
    """

    @property
    def domain_name(self) -> str:
        """Return the domain name this agent handles."""
        ...

    @property
    def domain_description(self) -> str:
        """Return a description of what this domain handles."""
        ...

    @property
    def capabilities(self) -> tuple[str, ...]:
        """
        Return the data types this agent handles.

        Used by LLMToolRouter for better routing context.

        Returns:
            Tuple of capability strings (e.g., ("stock prices", "market cap"))
        """
        ...

    @property
    def example_queries(self) -> tuple[str, ...]:
        """
        Return example queries this agent handles well.

        Used by LLMToolRouter for routing context and training.

        Returns:
            Tuple of example query strings
        """
        ...

    async def process(
        self,
        context: AgentContext,
    ) -> AgentResult:
        """
        Process a query and generate API calls.

        Args:
            context: AgentContext with query, entities, and retrieved context

        Returns:
            AgentResult with tool calls or clarification request
        """
        ...

    async def can_handle(self, query: str) -> float:
        """
        Check if this agent can handle the given query.

        .. deprecated::
            Will be removed in v2.0. Use LLMToolRouter for query routing instead.

        Returns a confidence score (0.0 to 1.0) indicating how likely
        this agent is the right one for the query.

        Args:
            query: Natural language query

        Returns:
            Confidence score (0.0 to 1.0)
        """
        ...
