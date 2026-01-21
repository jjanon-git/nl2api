"""
Accuracy test configuration.

Defines tiers, thresholds, and settings for accuracy testing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto


class Tier(Enum):
    """Test execution tiers with different sample sizes and thresholds."""

    TIER1 = auto()  # Quick check: 50 samples, ~5 min
    TIER2 = auto()  # Standard: 200 samples, ~15 min
    TIER3 = auto()  # Comprehensive: All samples, ~1 hr+


@dataclass(frozen=True)
class CategoryThreshold:
    """Accuracy threshold for a specific category."""

    category: str
    threshold: float
    rationale: str = ""


@dataclass
class AccuracyConfig:
    """Configuration for accuracy tests."""

    # Global minimum threshold
    global_threshold: float = 0.80

    # Per-category thresholds
    category_thresholds: dict[str, float] = field(default_factory=lambda: {
        "lookups": 0.85,
        "temporal": 0.80,
        "comparisons": 0.75,
        "screening": 0.75,
        "complex": 0.70,
        "errors": 0.90,
    })

    # Tier sample sizes
    tier_samples: dict[Tier, int] = field(default_factory=lambda: {
        Tier.TIER1: 50,
        Tier.TIER2: 200,
        Tier.TIER3: -1,  # -1 means all
    })

    # LLM settings
    model: str = "claude-3-haiku-20240307"
    max_retries: int = 2
    parallel_requests: int = 5

    # Clarification threshold
    confidence_threshold: float = 0.5

    def get_threshold(self, category: str | None = None) -> float:
        """Get threshold for a category (or global if not specified)."""
        if category and category in self.category_thresholds:
            return self.category_thresholds[category]
        return self.global_threshold

    def get_samples(self, tier: Tier) -> int | None:
        """Get sample count for a tier (-1 or None means all)."""
        count = self.tier_samples.get(tier, -1)
        return None if count == -1 else count


# Default configuration instance
DEFAULT_CONFIG = AccuracyConfig()
