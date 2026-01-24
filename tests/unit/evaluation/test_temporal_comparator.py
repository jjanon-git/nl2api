"""
Unit tests for TemporalComparator.

Tests temporal-aware tool call comparison including:
- STRUCTURAL mode (normalize dates before comparison)
- BEHAVIORAL mode (validate expressions only)
- DATA mode (exact match required)
- Custom date fields
- Error handling
"""

from datetime import date

import pytest

from CONTRACTS import TemporalValidationMode, ToolCall
from src.evalkit.core.temporal import (
    DateResolver,
    TemporalComparator,
    compare_tool_calls_temporal,
)


@pytest.fixture
def resolver():
    """DateResolver with fixed reference date for reproducible tests."""
    return DateResolver(reference_date=date(2026, 1, 21))


@pytest.fixture
def structural_comparator(resolver):
    """TemporalComparator in STRUCTURAL mode."""
    return TemporalComparator(
        date_resolver=resolver,
        validation_mode=TemporalValidationMode.STRUCTURAL,
    )


@pytest.fixture
def behavioral_comparator(resolver):
    """TemporalComparator in BEHAVIORAL mode."""
    return TemporalComparator(
        date_resolver=resolver,
        validation_mode=TemporalValidationMode.BEHAVIORAL,
    )


@pytest.fixture
def data_comparator(resolver):
    """TemporalComparator in DATA mode."""
    return TemporalComparator(
        date_resolver=resolver,
        validation_mode=TemporalValidationMode.DATA,
    )


class TestStructuralMode:
    """Tests for STRUCTURAL validation mode (normalize dates before comparison)."""

    def test_structural_normalizes_relative_to_absolute(self, structural_comparator):
        """Relative and absolute dates should match after normalization."""
        expected = (
            ToolCall(
                tool_name="get_data",
                arguments={"tickers": ["AAPL.O"], "start": "-1D", "end": "-1D"},
            ),
        )
        actual = (
            ToolCall(
                tool_name="get_data",
                arguments={"tickers": ["AAPL.O"], "start": "2026-01-20", "end": "2026-01-20"},
            ),
        )

        result = structural_comparator.compare(expected, actual)

        assert result.matches is True
        assert result.score == 1.0
        assert result.date_normalization_applied is True

    def test_structural_fails_different_dates(self, structural_comparator):
        """Different dates should not match even after normalization."""
        expected = (
            ToolCall(tool_name="get_data", arguments={"tickers": ["AAPL.O"], "start": "-1D"}),
        )
        actual = (
            ToolCall(
                tool_name="get_data",
                arguments={"tickers": ["AAPL.O"], "start": "2026-01-19"},  # Different day
            ),
        )

        result = structural_comparator.compare(expected, actual)

        assert result.matches is False

    def test_structural_ignores_non_date_fields(self, structural_comparator):
        """Non-date fields should be compared exactly."""
        expected = (
            ToolCall(
                tool_name="get_data",
                arguments={"tickers": ["AAPL.O"], "start": "-1D", "fields": ["P", "MV"]},
            ),
        )
        actual = (
            ToolCall(
                tool_name="get_data",
                arguments={"tickers": ["AAPL.O"], "start": "2026-01-20", "fields": ["P", "MV"]},
            ),
        )

        result = structural_comparator.compare(expected, actual)

        assert result.matches is True

    def test_structural_fails_different_non_date_fields(self, structural_comparator):
        """Different non-date fields should cause mismatch."""
        expected = (
            ToolCall(
                tool_name="get_data",
                arguments={"tickers": ["AAPL.O"], "start": "-1D", "fields": ["P"]},
            ),
        )
        actual = (
            ToolCall(
                tool_name="get_data",
                arguments={"tickers": ["AAPL.O"], "start": "2026-01-20", "fields": ["MV"]},
            ),
        )

        result = structural_comparator.compare(expected, actual)

        assert result.matches is False

    def test_structural_records_normalizations(self, structural_comparator):
        """Normalizations should be recorded in result."""
        expected = (ToolCall(tool_name="get_data", arguments={"start": "-1D", "end": "-7D"}),)
        actual = (
            ToolCall(tool_name="get_data", arguments={"start": "2026-01-20", "end": "2026-01-14"}),
        )

        result = structural_comparator.compare(expected, actual)

        assert result.date_normalization_applied is True
        assert "expected[0]" in result.normalized_dates
        assert "start" in result.normalized_dates["expected[0]"]
        assert "end" in result.normalized_dates["expected[0]"]


class TestBehavioralMode:
    """Tests for BEHAVIORAL validation mode (validate expressions only)."""

    def test_behavioral_accepts_both_relative(self, behavioral_comparator):
        """Both relative expressions should be accepted."""
        expected = (
            ToolCall(tool_name="get_data", arguments={"tickers": ["AAPL.O"], "start": "-1D"}),
        )
        actual = (
            ToolCall(
                tool_name="get_data",
                arguments={"tickers": ["AAPL.O"], "start": "-7D"},  # Different but valid
            ),
        )

        result = behavioral_comparator.compare(expected, actual)

        # In behavioral mode, date values are stripped, so only tool name matters
        assert result.matches is True
        assert result.date_normalization_applied is False

    def test_behavioral_accepts_mixed_formats(self, behavioral_comparator):
        """Mixed relative and absolute should be accepted if both valid."""
        expected = (
            ToolCall(tool_name="get_data", arguments={"tickers": ["AAPL.O"], "start": "-1D"}),
        )
        actual = (
            ToolCall(
                tool_name="get_data",
                arguments={"tickers": ["AAPL.O"], "start": "2026-01-15"},  # Different but valid
            ),
        )

        result = behavioral_comparator.compare(expected, actual)

        assert result.matches is True

    def test_behavioral_rejects_invalid_expression(self, behavioral_comparator):
        """Invalid temporal expressions should cause failure."""
        expected = (ToolCall(tool_name="get_data", arguments={"start": "-1D"}),)
        actual = (
            ToolCall(
                tool_name="get_data",
                arguments={"start": "invalid-date"},  # Invalid
            ),
        )

        result = behavioral_comparator.compare(expected, actual)

        assert result.matches is False
        assert len(result.temporal_validation_errors) > 0

    def test_behavioral_compares_non_date_fields(self, behavioral_comparator):
        """Non-date fields should still be compared exactly."""
        expected = (
            ToolCall(tool_name="get_data", arguments={"tickers": ["AAPL.O"], "start": "-1D"}),
        )
        actual = (
            ToolCall(
                tool_name="get_data",
                arguments={"tickers": ["MSFT.O"], "start": "-1D"},  # Different ticker
            ),
        )

        result = behavioral_comparator.compare(expected, actual)

        assert result.matches is False


class TestDataMode:
    """Tests for DATA validation mode (exact match required)."""

    def test_data_mode_exact_match_required(self, data_comparator):
        """DATA mode requires exact match, no normalization."""
        expected = (ToolCall(tool_name="get_data", arguments={"start": "-1D"}),)
        actual = (
            ToolCall(
                tool_name="get_data",
                arguments={"start": "2026-01-20"},  # Same date but different format
            ),
        )

        result = data_comparator.compare(expected, actual)

        assert result.matches is False
        assert result.date_normalization_applied is False

    def test_data_mode_exact_match_passes(self, data_comparator):
        """DATA mode passes with exact string match."""
        expected = (ToolCall(tool_name="get_data", arguments={"start": "2026-01-20"}),)
        actual = (ToolCall(tool_name="get_data", arguments={"start": "2026-01-20"}),)

        result = data_comparator.compare(expected, actual)

        assert result.matches is True


class TestCustomDateFields:
    """Tests for custom date field configuration."""

    def test_custom_date_fields_recognized(self, resolver):
        """Custom date fields should be normalized."""
        comparator = TemporalComparator(
            date_resolver=resolver,
            validation_mode=TemporalValidationMode.STRUCTURAL,
            relative_date_fields=("custom_start", "custom_end"),
        )

        expected = (ToolCall(tool_name="get_data", arguments={"custom_start": "-1D"}),)
        actual = (ToolCall(tool_name="get_data", arguments={"custom_start": "2026-01-20"}),)

        result = comparator.compare(expected, actual)

        assert result.matches is True
        assert result.date_normalization_applied is True

    def test_default_date_fields_not_normalized_when_custom(self, resolver):
        """Default date fields should not be normalized when custom is set."""
        comparator = TemporalComparator(
            date_resolver=resolver,
            validation_mode=TemporalValidationMode.STRUCTURAL,
            relative_date_fields=("custom_start",),  # Only custom_start
        )

        expected = (
            ToolCall(
                tool_name="get_data",
                arguments={"start": "-1D"},  # Default field, not in custom list
            ),
        )
        actual = (ToolCall(tool_name="get_data", arguments={"start": "2026-01-20"}),)

        result = comparator.compare(expected, actual)

        # 'start' is not in custom fields, so no normalization
        # Strings "-1D" != "2026-01-20"
        assert result.matches is False


class TestMultipleToolCalls:
    """Tests for multiple tool call comparison."""

    def test_multiple_calls_all_match(self, structural_comparator):
        """All calls should match for overall match."""
        expected = (
            ToolCall(tool_name="get_data", arguments={"start": "-1D"}),
            ToolCall(tool_name="get_price", arguments={"start": "-7D"}),
        )
        actual = (
            ToolCall(tool_name="get_data", arguments={"start": "2026-01-20"}),
            ToolCall(tool_name="get_price", arguments={"start": "2026-01-14"}),
        )

        result = structural_comparator.compare(expected, actual)

        assert result.matches is True
        assert len(result.matched_calls) == 2

    def test_multiple_calls_partial_match(self, structural_comparator):
        """Partial match should fail overall."""
        expected = (
            ToolCall(tool_name="get_data", arguments={"start": "-1D"}),
            ToolCall(tool_name="get_price", arguments={"start": "-7D"}),
        )
        actual = (
            ToolCall(tool_name="get_data", arguments={"start": "2026-01-20"}),
            ToolCall(tool_name="get_price", arguments={"start": "2026-01-01"}),  # Wrong date
        )

        result = structural_comparator.compare(expected, actual)

        assert result.matches is False


class TestConvenienceFunction:
    """Tests for the compare_tool_calls_temporal convenience function."""

    def test_convenience_function_structural(self):
        """Convenience function should use STRUCTURAL mode by default."""
        expected = (ToolCall(tool_name="get_data", arguments={"start": "-1D"}),)
        actual = (ToolCall(tool_name="get_data", arguments={"start": "2026-01-20"}),)

        result = compare_tool_calls_temporal(
            expected,
            actual,
            reference_date=date(2026, 1, 21),
        )

        assert result.matches is True

    def test_convenience_function_with_mode(self):
        """Convenience function should accept mode parameter."""
        expected = (ToolCall(tool_name="get_data", arguments={"start": "-1D"}),)
        actual = (ToolCall(tool_name="get_data", arguments={"start": "2026-01-20"}),)

        result = compare_tool_calls_temporal(
            expected,
            actual,
            reference_date=date(2026, 1, 21),
            validation_mode=TemporalValidationMode.DATA,
        )

        assert result.matches is False  # Exact match required


class TestEdgeCases:
    """Tests for edge cases."""

    def test_empty_tool_calls(self, structural_comparator):
        """Empty tool call lists should match."""
        result = structural_comparator.compare((), ())
        assert result.matches is True
        assert result.score == 1.0

    def test_no_date_fields(self, structural_comparator):
        """Tool calls without date fields should work normally."""
        expected = (ToolCall(tool_name="get_data", arguments={"tickers": ["AAPL.O"]}),)
        actual = (ToolCall(tool_name="get_data", arguments={"tickers": ["AAPL.O"]}),)

        result = structural_comparator.compare(expected, actual)

        assert result.matches is True
        assert result.date_normalization_applied is False

    def test_missing_call(self, structural_comparator):
        """Missing calls should be detected."""
        expected = (
            ToolCall(tool_name="get_data", arguments={"start": "-1D"}),
            ToolCall(tool_name="get_price", arguments={"start": "-1D"}),
        )
        actual = (ToolCall(tool_name="get_data", arguments={"start": "2026-01-20"}),)

        result = structural_comparator.compare(expected, actual)

        assert result.matches is False
        assert len(result.missing_calls) == 1

    def test_extra_call(self, structural_comparator):
        """Extra calls should be detected."""
        expected = (ToolCall(tool_name="get_data", arguments={"start": "-1D"}),)
        actual = (
            ToolCall(tool_name="get_data", arguments={"start": "2026-01-20"}),
            ToolCall(tool_name="extra_call", arguments={}),
        )

        result = structural_comparator.compare(expected, actual)

        assert result.matches is False
        assert len(result.extra_calls) == 1
