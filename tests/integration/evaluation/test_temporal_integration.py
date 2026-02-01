"""
Integration tests for temporal-aware evaluation.

Verifies that NL2APIPack correctly uses TemporalComparator
when temporal evaluation is enabled.
"""

from datetime import date

import pytest

from CONTRACTS import (
    EvalContext,
    EvaluationConfig,
    SystemResponse,
    TemporalValidationMode,
    TestCase,
    TestCaseMetadata,
    ToolCall,
)
from src.nl2api.evaluation import NL2APIPack


def _make_metadata(category: str = "temporal", subcategory: str = "test") -> TestCaseMetadata:
    """Create test case metadata with required fields."""
    return TestCaseMetadata(
        api_version="1.0.0",
        complexity_level=1,
        category=category,
        subcategory=subcategory,
        source="test",
    )


@pytest.fixture
def test_case_with_relative_dates() -> TestCase:
    """Test case using relative date expressions."""
    return TestCase(
        id="temporal-test-001",
        nl_query="Get Apple data for yesterday",
        expected_tool_calls=(
            ToolCall(
                tool_name="get_data",
                arguments={"tickers": ["AAPL.O"], "start": "-1D", "end": "-1D"},
            ),
        ),
        metadata=_make_metadata(category="temporal", subcategory="relative_dates"),
    )


@pytest.fixture
def response_with_absolute_dates() -> SystemResponse:
    """Response using absolute dates (equivalent to -1D when ref date is 2026-01-21)."""
    return SystemResponse(
        raw_output='[{"tool_name": "get_data", "arguments": {"tickers": ["AAPL.O"], "start": "2026-01-20", "end": "2026-01-20"}}]',
        latency_ms=100,
    )


@pytest.fixture
def response_with_wrong_dates() -> SystemResponse:
    """Response with different dates that shouldn't match."""
    return SystemResponse(
        raw_output='[{"tool_name": "get_data", "arguments": {"tickers": ["AAPL.O"], "start": "2026-01-15", "end": "2026-01-15"}}]',
        latency_ms=100,
    )


def _make_pack_from_config(config: EvaluationConfig) -> NL2APIPack:
    """Create NL2APIPack from EvaluationConfig."""
    return NL2APIPack(
        execution_enabled=config.execution_stage_enabled,
        semantics_enabled=config.semantics_stage_enabled,
        numeric_tolerance=config.numeric_tolerance,
        temporal_mode=config.temporal_mode,
        evaluation_date=config.evaluation_date,
        relative_date_fields=config.relative_date_fields,
        fiscal_year_end_month=config.fiscal_year_end_month,
    )


async def _evaluate_with_pack(
    pack: NL2APIPack,
    test_case: TestCase,
    response: SystemResponse,
):
    """Evaluate using pack with converted response."""
    system_output = {
        "raw_output": response.raw_output,
        "nl_response": response.nl_response,
    }
    return await pack.evaluate(
        test_case=test_case,
        system_output=system_output,
        context=EvalContext(worker_id="test-worker"),
    )


class TestNL2APIPackTemporalIntegration:
    """Tests for NL2APIPack with temporal comparison."""

    @pytest.mark.asyncio
    async def test_structural_mode_matches_equivalent_dates(
        self,
        test_case_with_relative_dates: TestCase,
        response_with_absolute_dates: SystemResponse,
    ):
        """STRUCTURAL mode should match -1D with 2026-01-20 when ref date is 2026-01-21."""
        config = EvaluationConfig(
            temporal_mode=TemporalValidationMode.STRUCTURAL,
            evaluation_date=date(2026, 1, 21),
        )
        pack = _make_pack_from_config(config)

        scorecard = await _evaluate_with_pack(
            pack, test_case_with_relative_dates, response_with_absolute_dates
        )

        assert scorecard.syntax_result.passed is True
        assert scorecard.logic_result is not None
        assert scorecard.logic_result.passed is True
        assert scorecard.logic_result.score == 1.0

    @pytest.mark.asyncio
    async def test_structural_mode_fails_different_dates(
        self,
        test_case_with_relative_dates: TestCase,
        response_with_wrong_dates: SystemResponse,
    ):
        """STRUCTURAL mode should fail when dates don't match after normalization."""
        config = EvaluationConfig(
            temporal_mode=TemporalValidationMode.STRUCTURAL,
            evaluation_date=date(2026, 1, 21),
        )
        pack = _make_pack_from_config(config)

        scorecard = await _evaluate_with_pack(
            pack, test_case_with_relative_dates, response_with_wrong_dates
        )

        assert scorecard.syntax_result.passed is True
        assert scorecard.logic_result is not None
        assert scorecard.logic_result.passed is False

    @pytest.mark.asyncio
    async def test_data_mode_requires_exact_match(
        self,
        test_case_with_relative_dates: TestCase,
        response_with_absolute_dates: SystemResponse,
    ):
        """DATA mode should require exact string match (no normalization)."""
        config = EvaluationConfig(
            temporal_mode=TemporalValidationMode.DATA,
            evaluation_date=date(2026, 1, 21),
        )
        pack = _make_pack_from_config(config)

        scorecard = await _evaluate_with_pack(
            pack, test_case_with_relative_dates, response_with_absolute_dates
        )

        assert scorecard.syntax_result.passed is True
        assert scorecard.logic_result is not None
        # Should fail because "-1D" != "2026-01-20" in DATA mode
        assert scorecard.logic_result.passed is False

    @pytest.mark.asyncio
    async def test_behavioral_mode_validates_expressions(
        self,
        test_case_with_relative_dates: TestCase,
        response_with_absolute_dates: SystemResponse,
    ):
        """BEHAVIORAL mode should accept any valid date expressions."""
        config = EvaluationConfig(
            temporal_mode=TemporalValidationMode.BEHAVIORAL,
            evaluation_date=date(2026, 1, 21),
        )
        pack = _make_pack_from_config(config)

        scorecard = await _evaluate_with_pack(
            pack, test_case_with_relative_dates, response_with_absolute_dates
        )

        assert scorecard.syntax_result.passed is True
        assert scorecard.logic_result is not None
        # Should pass because both expressions are valid
        assert scorecard.logic_result.passed is True

    @pytest.mark.asyncio
    async def test_default_mode_is_structural(
        self,
        test_case_with_relative_dates: TestCase,
        response_with_absolute_dates: SystemResponse,
    ):
        """Default config should use STRUCTURAL mode."""
        # No explicit temporal_mode - should default to STRUCTURAL
        config = EvaluationConfig(
            evaluation_date=date(2026, 1, 21),
        )
        pack = _make_pack_from_config(config)

        scorecard = await _evaluate_with_pack(
            pack, test_case_with_relative_dates, response_with_absolute_dates
        )

        assert scorecard.logic_result is not None
        assert scorecard.logic_result.passed is True

    @pytest.mark.asyncio
    async def test_fiscal_year_comparison(self):
        """Fiscal year expressions should be normalized correctly."""
        test_case = TestCase(
            id="temporal-test-fy",
            nl_query="Get Apple FY2025 data",
            expected_tool_calls=(
                ToolCall(
                    tool_name="get_financials",
                    # Note: "Period" (capital P) matches the default date field names
                    arguments={"ticker": "AAPL.O", "Period": "FY-1"},
                ),
            ),
            metadata=_make_metadata(category="temporal", subcategory="fiscal_year"),
        )
        # FY-1 from 2026 with Dec year-end = 2025-12-31
        response = SystemResponse(
            raw_output='[{"tool_name": "get_financials", "arguments": {"ticker": "AAPL.O", "Period": "2025-12-31"}}]',
            latency_ms=100,
        )

        config = EvaluationConfig(
            temporal_mode=TemporalValidationMode.STRUCTURAL,
            evaluation_date=date(2026, 1, 21),
            fiscal_year_end_month=12,
        )
        pack = _make_pack_from_config(config)

        scorecard = await _evaluate_with_pack(pack, test_case, response)

        assert scorecard.logic_result is not None
        assert scorecard.logic_result.passed is True


class TestTemporalWithNonDateFields:
    """Tests that non-date fields are still compared correctly."""

    @pytest.mark.asyncio
    async def test_non_date_fields_still_compared(self):
        """Non-date fields should be compared exactly even in temporal mode."""
        test_case = TestCase(
            id="temporal-test-non-date",
            nl_query="Get Apple price yesterday",
            expected_tool_calls=(
                ToolCall(
                    tool_name="get_data",
                    arguments={"tickers": ["AAPL.O"], "start": "-1D", "fields": ["P"]},
                ),
            ),
            metadata=_make_metadata(category="temporal", subcategory="mixed"),
        )
        # Different field
        response = SystemResponse(
            raw_output='[{"tool_name": "get_data", "arguments": {"tickers": ["AAPL.O"], "start": "2026-01-20", "fields": ["MV"]}}]',
            latency_ms=100,
        )

        config = EvaluationConfig(
            temporal_mode=TemporalValidationMode.STRUCTURAL,
            evaluation_date=date(2026, 1, 21),
        )
        pack = _make_pack_from_config(config)

        scorecard = await _evaluate_with_pack(pack, test_case, response)

        assert scorecard.logic_result is not None
        # Should fail because fields don't match
        assert scorecard.logic_result.passed is False

    @pytest.mark.asyncio
    async def test_tool_name_mismatch_fails(self):
        """Tool name mismatch should fail regardless of temporal mode."""
        test_case = TestCase(
            id="temporal-test-tool",
            nl_query="Get Apple data",
            expected_tool_calls=(
                ToolCall(
                    tool_name="get_data",
                    arguments={"tickers": ["AAPL.O"], "start": "-1D"},
                ),
            ),
            metadata=_make_metadata(category="temporal", subcategory="tool_mismatch"),
        )
        response = SystemResponse(
            raw_output='[{"tool_name": "get_price", "arguments": {"tickers": ["AAPL.O"], "start": "2026-01-20"}}]',
            latency_ms=100,
        )

        config = EvaluationConfig(
            temporal_mode=TemporalValidationMode.STRUCTURAL,
            evaluation_date=date(2026, 1, 21),
        )
        pack = _make_pack_from_config(config)

        scorecard = await _evaluate_with_pack(pack, test_case, response)

        assert scorecard.logic_result is not None
        assert scorecard.logic_result.passed is False
