"""
Entity Resolution Models

Data classes for entity resolution results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


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
    """

    original: str  # Original text (e.g., "Apple")
    identifier: str  # Resolved identifier (e.g., "AAPL.O")
    entity_type: str  # Type of entity (e.g., "company", "index")
    confidence: float = 1.0
    alternatives: tuple[str, ...] = ()  # Alternative identifiers if ambiguous
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict:
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
    def from_dict(cls, data: dict) -> ResolvedEntity:
        """Create from dictionary."""
        return cls(
            original=data["original"],
            identifier=data["identifier"],
            entity_type=data.get("entity_type", "company"),
            confidence=data.get("confidence", 1.0),
            alternatives=tuple(data.get("alternatives", [])),
            metadata=data.get("metadata", {}),
        )
