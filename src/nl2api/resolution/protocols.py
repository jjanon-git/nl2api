"""
Entity Resolution Protocols

Defines the interface for entity resolution (company names to RICs, etc.).
ResolvedEntity is imported from the canonical shared location.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from src.evalkit.common.entity_resolution import ResolvedEntity

# Re-export ResolvedEntity for backwards compatibility
__all__ = ["ResolvedEntity", "EntityResolver"]


@runtime_checkable
class EntityResolver(Protocol):
    """
    Protocol for entity resolution.

    Resolves natural language entity references to standardized identifiers.
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
