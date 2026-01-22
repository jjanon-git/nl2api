"""
Unit tests for the LLM pricing module.
"""

import pytest

from src.evaluation.batch.pricing import (
    MODEL_PRICING,
    calculate_cost,
    format_cost,
    get_model_pricing,
)


class TestGetModelPricing:
    """Tests for get_model_pricing function."""

    def test_exact_match_claude_sonnet(self):
        """Test exact match for Claude 3.5 Sonnet."""
        pricing = get_model_pricing("claude-3-5-sonnet-20241022")
        assert pricing == (3.0, 15.0)

    def test_exact_match_claude_haiku(self):
        """Test exact match for Claude 3.5 Haiku."""
        pricing = get_model_pricing("claude-3-5-haiku-20241022")
        assert pricing == (0.80, 4.0)

    def test_exact_match_gpt4o(self):
        """Test exact match for GPT-4o."""
        pricing = get_model_pricing("gpt-4o")
        assert pricing == (2.50, 10.0)

    def test_exact_match_opus(self):
        """Test exact match for Claude Opus 4.5."""
        pricing = get_model_pricing("claude-opus-4-5-20251101")
        assert pricing == (15.0, 75.0)

    def test_none_returns_default(self):
        """Test that None returns default pricing."""
        pricing = get_model_pricing(None)
        assert pricing == (3.0, 15.0)  # Default is Sonnet 3.5

    def test_unknown_model_returns_default(self):
        """Test that unknown model returns default pricing."""
        pricing = get_model_pricing("unknown-model-xyz")
        assert pricing == (3.0, 15.0)

    def test_partial_match(self):
        """Test that partial match works for model families."""
        # gpt-4o should match gpt-4o-2024-11-20
        pricing = get_model_pricing("gpt-4o-2024-11-20")
        assert pricing == (2.50, 10.0)


class TestCalculateCost:
    """Tests for calculate_cost function."""

    def test_basic_cost_calculation(self):
        """Test basic cost calculation with known values."""
        # 1M input + 1M output with Sonnet pricing: $3 + $15 = $18
        cost = calculate_cost(1_000_000, 1_000_000, "claude-3-5-sonnet-20241022")
        assert cost == 18.0

    def test_haiku_is_cheaper(self):
        """Test that Haiku is significantly cheaper than Sonnet."""
        haiku_cost = calculate_cost(1000, 1000, "claude-3-5-haiku-20241022")
        sonnet_cost = calculate_cost(1000, 1000, "claude-3-5-sonnet-20241022")
        assert haiku_cost < sonnet_cost

    def test_opus_is_expensive(self):
        """Test that Opus is more expensive than Sonnet."""
        opus_cost = calculate_cost(1000, 1000, "claude-opus-4-5-20251101")
        sonnet_cost = calculate_cost(1000, 1000, "claude-3-5-sonnet-20241022")
        assert opus_cost > sonnet_cost

    def test_none_input_tokens_returns_none(self):
        """Test that None input tokens returns None."""
        cost = calculate_cost(None, 1000, "claude-3-5-sonnet-20241022")
        assert cost is None

    def test_none_output_tokens_returns_none(self):
        """Test that None output tokens returns None."""
        cost = calculate_cost(1000, None, "claude-3-5-sonnet-20241022")
        assert cost is None

    def test_zero_tokens_returns_zero_cost(self):
        """Test that zero tokens returns zero cost."""
        cost = calculate_cost(0, 0, "claude-3-5-sonnet-20241022")
        assert cost == 0.0

    def test_realistic_token_counts(self):
        """Test with realistic token counts for a typical query."""
        # Typical query: ~500 input, ~200 output
        cost = calculate_cost(500, 200, "claude-3-5-sonnet-20241022")
        # $3 * 500/1M + $15 * 200/1M = $0.0015 + $0.003 = $0.0045
        assert cost == pytest.approx(0.0045, abs=0.0001)

    def test_default_model_used_when_none(self):
        """Test that default model pricing is used when model is None."""
        cost_none = calculate_cost(1000, 1000, None)
        cost_sonnet = calculate_cost(1000, 1000, "claude-3-5-sonnet-20241022")
        assert cost_none == cost_sonnet


class TestFormatCost:
    """Tests for format_cost function."""

    def test_format_none(self):
        """Test formatting None cost."""
        assert format_cost(None) == "-"

    def test_format_small_cost(self):
        """Test formatting very small cost."""
        result = format_cost(0.000045)
        assert result.startswith("$")
        assert "0.000045" in result

    def test_format_normal_cost(self):
        """Test formatting normal cost."""
        result = format_cost(0.05)
        assert result == "$0.0500"

    def test_format_larger_cost(self):
        """Test formatting larger cost."""
        result = format_cost(1.23)
        assert result == "$1.2300"


class TestModelPricingTable:
    """Tests for MODEL_PRICING table completeness."""

    def test_has_claude_models(self):
        """Test that Claude models are in pricing table."""
        claude_models = [k for k in MODEL_PRICING.keys() if "claude" in k]
        assert len(claude_models) >= 5  # At least Sonnet, Haiku, Opus variants

    def test_has_openai_models(self):
        """Test that OpenAI models are in pricing table."""
        openai_models = [k for k in MODEL_PRICING.keys() if "gpt" in k or "o1" in k]
        assert len(openai_models) >= 5  # GPT-4o, GPT-4, GPT-3.5, o1 variants

    def test_all_prices_are_positive(self):
        """Test that all prices are positive numbers."""
        for model, (input_price, output_price) in MODEL_PRICING.items():
            assert input_price > 0, f"Input price for {model} should be positive"
            assert output_price > 0, f"Output price for {model} should be positive"

    def test_output_price_greater_than_input(self):
        """Test that output price is >= input price for all models."""
        for model, (input_price, output_price) in MODEL_PRICING.items():
            assert output_price >= input_price, \
                f"Output price for {model} should be >= input price"
