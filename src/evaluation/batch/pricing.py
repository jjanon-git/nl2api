"""
LLM Pricing Table

Provides model-aware cost calculation for evaluation tracking.
Prices are per 1 million tokens as of January 2026.
"""

from __future__ import annotations

# Pricing table: model_id -> (input_cost_per_million, output_cost_per_million)
# Source: Official provider pricing pages
MODEL_PRICING: dict[str, tuple[float, float]] = {
    # Anthropic Claude models
    "claude-3-5-sonnet-20241022": (3.0, 15.0),
    "claude-3-5-sonnet-20240620": (3.0, 15.0),
    "claude-3-5-haiku-20241022": (0.80, 4.0),
    "claude-3-haiku-20240307": (0.25, 1.25),
    "claude-3-sonnet-20240229": (3.0, 15.0),
    "claude-3-opus-20240229": (15.0, 75.0),
    "claude-opus-4-5-20251101": (15.0, 75.0),
    "claude-sonnet-4-20250514": (3.0, 15.0),
    # OpenAI models
    "gpt-4o": (2.50, 10.0),
    "gpt-4o-2024-11-20": (2.50, 10.0),
    "gpt-4o-mini": (0.15, 0.60),
    "gpt-4-turbo": (10.0, 30.0),
    "gpt-4": (30.0, 60.0),
    "gpt-3.5-turbo": (0.50, 1.50),
    # o1 models
    "o1": (15.0, 60.0),
    "o1-mini": (3.0, 12.0),
    "o1-preview": (15.0, 60.0),
}

# Default pricing (Claude Sonnet 3.5) for unknown models
DEFAULT_PRICING = (3.0, 15.0)


def get_model_pricing(model: str | None) -> tuple[float, float]:
    """
    Get pricing for a model.

    Args:
        model: Model identifier (e.g., "claude-3-5-sonnet-20241022")

    Returns:
        Tuple of (input_cost_per_million, output_cost_per_million)
    """
    if model is None:
        return DEFAULT_PRICING

    # Try exact match first
    if model in MODEL_PRICING:
        return MODEL_PRICING[model]

    # Try partial match for model families
    model_lower = model.lower()
    for known_model, pricing in MODEL_PRICING.items():
        if known_model in model_lower or model_lower in known_model:
            return pricing

    return DEFAULT_PRICING


def calculate_cost(
    input_tokens: int | None,
    output_tokens: int | None,
    model: str | None = None,
) -> float | None:
    """
    Calculate estimated cost for a request.

    Args:
        input_tokens: Number of input/prompt tokens
        output_tokens: Number of output/completion tokens
        model: Model identifier for pricing lookup

    Returns:
        Estimated cost in USD, or None if tokens are not available
    """
    if input_tokens is None or output_tokens is None:
        return None

    input_rate, output_rate = get_model_pricing(model)

    return (
        (input_tokens / 1_000_000) * input_rate +
        (output_tokens / 1_000_000) * output_rate
    )


def format_cost(cost: float | None) -> str:
    """Format cost for display."""
    if cost is None:
        return "-"
    if cost < 0.01:
        return f"${cost:.6f}"
    return f"${cost:.4f}"
