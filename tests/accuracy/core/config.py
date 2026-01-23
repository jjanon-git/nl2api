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
    category_thresholds: dict[str, float] = field(
        default_factory=lambda: {
            "lookups": 0.85,
            "temporal": 0.80,
            "comparisons": 0.75,
            "screening": 0.75,
            "complex": 0.70,
            "errors": 0.90,
        }
    )

    # Tier sample sizes
    tier_samples: dict[Tier, int] = field(
        default_factory=lambda: {
            Tier.TIER1: 50,
            Tier.TIER2: 200,
            Tier.TIER3: -1,  # -1 means all
        }
    )

    # LLM settings
    model: str = "claude-3-haiku-20240307"

    # Batch API settings (default - 50% cheaper, higher rate limits)
    use_batch_api: bool = True
    batch_poll_interval: float = 5.0  # seconds between status checks
    batch_timeout: float = 3600.0  # max wait time for batch completion (1 hour)

    # Real-time API settings (fallback when use_batch_api=False)
    parallel_requests: int = 3  # concurrent requests (reduced from 5)
    request_delay: float = 0.5  # delay between request batches (seconds)

    # Retry settings (applies to real-time API)
    max_retries: int = 3
    retry_base_delay: float = 1.0  # base delay for exponential backoff
    retry_max_delay: float = 30.0  # max delay cap
    retry_jitter: float = 0.5  # jitter factor (0-1)

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

# Module-level exports for backwards compatibility
DEFAULT_MIN_ACCURACY = DEFAULT_CONFIG.global_threshold
CATEGORY_THRESHOLDS = DEFAULT_CONFIG.category_thresholds
