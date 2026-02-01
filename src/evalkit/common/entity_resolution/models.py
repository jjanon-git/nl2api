"""
Entity Resolution Models

Canonical data models for entity resolution results.
Used by both standalone service and embedded resolver.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EntityType(str, Enum):
    """Types of entities that can be resolved."""

    COMPANY = "company"
    TICKER = "ticker"
    INDEX = "index"
    CURRENCY = "currency"
    COMMODITY = "commodity"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class ResolvedEntity:
    """
    A resolved entity with its identifier and metadata.

    Attributes:
        original: Original text (e.g., "Apple")
        identifier: Resolved identifier (e.g., "AAPL.O")
        entity_type: Type of entity (e.g., "company", "ticker")
        confidence: Resolution confidence (0.0-1.0)
        alternatives: Alternative identifiers if ambiguous
        metadata: Additional metadata (ticker, company_name, etc.)
    """

    original: str
    identifier: str
    entity_type: str = "company"
    confidence: float = 1.0
    alternatives: tuple[str, ...] = ()
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "original": self.original,
            "identifier": self.identifier,
            "entity_type": self.entity_type,
            "confidence": self.confidence,
            "alternatives": list(self.alternatives),
            "metadata": dict(self.metadata),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ResolvedEntity:
        """Create from dictionary."""
        return cls(
            original=data["original"],
            identifier=data["identifier"],
            entity_type=data.get("entity_type", "company"),
            confidence=data.get("confidence", 1.0),
            alternatives=tuple(data.get("alternatives", [])),
            metadata=data.get("metadata", {}),
        )
