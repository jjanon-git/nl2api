"""
Protocol Definitions for Evalkit.

Provides protocols and types for LLM and entity resolution integration
without depending on specific implementations (nl2api, langchain, etc.).

This module defines:
- LLM protocols: MessageRole, LLMMessage, LLMResponse, LLMProviderProtocol
- Entity resolution protocols: ResolvedEntity, EntityResolver
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable


class MessageRole(str, Enum):
    """Role of a message in an LLM conversation."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass
class LLMMessage:
    """A message in an LLM conversation."""

    role: MessageRole
    content: str


@dataclass
class LLMResponse:
    """Response from an LLM provider."""

    content: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""


@runtime_checkable
class LLMProviderProtocol(Protocol):
    """
    Protocol for LLM providers.

    Any LLM provider that implements this protocol can be used with
    evalkit's semantics evaluator and other LLM-dependent components.
    """

    async def complete_with_retry(
        self,
        messages: list[LLMMessage],
        temperature: float = 0.0,
        max_tokens: int = 1024,
        max_retries: int = 3,
    ) -> LLMResponse:
        """
        Complete a conversation with retry logic.

        Args:
            messages: List of messages in the conversation
            temperature: Sampling temperature (0.0 = deterministic)
            max_tokens: Maximum tokens in response
            max_retries: Number of retries on failure

        Returns:
            LLMResponse with the completion
        """
        ...


def create_default_llm_provider() -> LLMProviderProtocol | None:
    """
    Create a default LLM provider if nl2api is available.

    Returns None if nl2api is not installed or configured.
    This allows evalkit to work standalone without nl2api.
    """
    try:
        from src.nl2api.config import NL2APIConfig
        from src.nl2api.llm.claude import ClaudeProvider

        cfg = NL2APIConfig()
        api_key = cfg.get_llm_api_key()
        if not api_key:
            return None

        return ClaudeProvider(api_key=api_key)
    except ImportError:
        return None
    except Exception:
        return None


# =============================================================================
# Entity Resolution Protocols
# =============================================================================


@dataclass(frozen=True)
class ResolvedEntity:
    """
    A resolved entity with its identifier and metadata.

    Represents the result of resolving a natural language entity reference
    (e.g., "Apple") to a standardized identifier (e.g., "AAPL.O").
    """

    original: str  # Original text (e.g., "Apple")
    identifier: str  # Resolved identifier (e.g., "AAPL.O")
    entity_type: str  # Type of entity (e.g., "company", "index")
    confidence: float = 1.0
    alternatives: tuple[str, ...] = ()  # Alternative identifiers if ambiguous
    metadata: dict[str, str] = field(default_factory=dict)


@runtime_checkable
class EntityResolver(Protocol):
    """
    Protocol for entity resolution.

    Resolves natural language entity references to standardized identifiers.
    Any entity resolver that implements this protocol can be used with
    evalkit's response generators and evaluators.
    """

    async def resolve(
        self,
        query: str,
    ) -> dict[str, str]:
        """
        Extract and resolve entities from a query.

        Args:
            query: Natural language query containing entity references

        Returns:
            Dictionary mapping entity names to resolved identifiers
            e.g., {"Apple": "AAPL.O", "Microsoft": "MSFT.O"}
        """
        ...

    async def resolve_single(
        self,
        entity: str,
        entity_type: str | None = None,
    ) -> ResolvedEntity | None:
        """
        Resolve a single entity.

        Args:
            entity: Entity name to resolve (e.g., "Apple Inc.")
            entity_type: Optional entity type hint (e.g., "company")

        Returns:
            ResolvedEntity if found, None otherwise
        """
        ...

    async def resolve_batch(
        self,
        entities: list[str],
    ) -> list[ResolvedEntity]:
        """
        Resolve multiple entities in batch.

        Args:
            entities: List of entity names to resolve

        Returns:
            List of ResolvedEntity (may be shorter than input if some not found)
        """
        ...
