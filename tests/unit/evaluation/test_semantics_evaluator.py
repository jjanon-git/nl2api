"""
Unit tests for SemanticsEvaluator (LLM-as-Judge).

Tests cover:
- NL generation from expected_response
- Semantic comparison (similar, different, partial match)
- Skip behavior when expected fields are NULL
- Error handling (LLM errors, timeouts, parse errors)
- Score calculation with weights
- OTEL telemetry (spans and metrics)
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from CONTRACTS import (
    ErrorCode,
    EvaluationStage,
    LLMJudgeConfig,
    TestCase,
    TestCaseMetadata,
    ToolCall,
)
from src.evaluation.core.semantics.evaluator import (
    ComparisonScores,
    SemanticsEvaluator,
)


@pytest.fixture
def mock_llm():
    """Create a mock LLM provider."""
    llm = MagicMock()
    llm.complete_with_retry = AsyncMock()
    return llm


@pytest.fixture
def test_case_with_nl():
    """Test case with expected_response and expected_nl_response populated."""
    return TestCase(
        id="test-001",
        nl_query="What is Apple's stock price?",
        expected_tool_calls=(
            ToolCall(
                tool_name="get_data",
                arguments={"tickers": "AAPL.O", "fields": ["P"]},
            ),
        ),
        expected_response={"AAPL.O": {"P": 185.42}},
        expected_nl_response="Apple's stock price is $185.42.",
        metadata=TestCaseMetadata(
            api_version="v1.0",
            complexity_level=1,
            tags=("price", "single_field"),
        ),
    )


@pytest.fixture
def test_case_without_nl_response():
    """Test case with expected_response but no expected_nl_response."""
    return TestCase(
        id="test-002",
        nl_query="What is Apple's stock price?",
        expected_tool_calls=(
            ToolCall(
                tool_name="get_data",
                arguments={"tickers": "AAPL.O", "fields": ["P"]},
            ),
        ),
        expected_response={"AAPL.O": {"P": 185.42}},
        expected_nl_response=None,  # Missing
        metadata=TestCaseMetadata(
            api_version="v1.0",
            complexity_level=1,
        ),
    )


@pytest.fixture
def test_case_without_expected_response():
    """Test case without expected_response (but has expected_nl_response)."""
    return TestCase(
        id="test-003",
        nl_query="What is Apple's stock price?",
        expected_tool_calls=(
            ToolCall(
                tool_name="get_data",
                arguments={"tickers": "AAPL.O", "fields": ["P"]},
            ),
        ),
        expected_response=None,  # Missing
        expected_nl_response="Apple's stock price is $185.42.",
        metadata=TestCaseMetadata(
            api_version="v1.0",
            complexity_level=1,
        ),
    )


class TestSemanticsEvaluatorBasic:
    """Basic tests for SemanticsEvaluator."""

    @pytest.mark.asyncio
    async def test_evaluate_skips_when_expected_nl_response_is_null(
        self, mock_llm, test_case_without_nl_response
    ):
        """Should skip evaluation and pass when expected_nl_response is NULL."""
        evaluator = SemanticsEvaluator(llm=mock_llm)

        result = await evaluator.evaluate(test_case_without_nl_response)

        assert result.passed is True
        assert result.score == 1.0
        assert result.stage == EvaluationStage.SEMANTICS
        assert "Skipped" in result.reason
        assert result.artifacts.get("skipped") is True
        mock_llm.complete_with_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_skips_when_expected_response_is_null_with_generator(
        self, mock_llm, test_case_without_expected_response
    ):
        """Should skip evaluation when expected_response is NULL and generator is provided."""
        evaluator = SemanticsEvaluator(llm=mock_llm)

        async def mock_generator(response_data):
            return "Generated response"

        result = await evaluator.evaluate(
            test_case_without_expected_response,
            nl_generator=mock_generator,
        )

        assert result.passed is True
        assert result.score == 1.0
        assert "Skipped" in result.reason
        mock_llm.complete_with_retry.assert_not_called()

    @pytest.mark.asyncio
    async def test_evaluate_direct_skips_when_expected_nl_response_is_null(
        self, mock_llm, test_case_without_nl_response
    ):
        """Should skip evaluate_direct when expected_nl_response is NULL."""
        evaluator = SemanticsEvaluator(llm=mock_llm)

        result = await evaluator.evaluate_direct(
            test_case_without_nl_response,
            actual_nl="Some actual response",
        )

        assert result.passed is True
        assert result.score == 1.0
        mock_llm.complete_with_retry.assert_not_called()


class TestSemanticsEvaluatorComparison:
    """Tests for semantic comparison functionality."""

    @pytest.mark.asyncio
    async def test_comparison_passes_similar_responses(self, mock_llm, test_case_with_nl):
        """Should pass when expected and actual responses are semantically similar."""
        # Mock LLM response indicating high similarity
        mock_llm.complete_with_retry.return_value = MagicMock(
            content=json.dumps(
                {
                    "meaning_match": 0.95,
                    "completeness": 0.90,
                    "accuracy": 1.0,
                    "reasoning": "Both responses convey the same Apple stock price information.",
                }
            )
        )

        evaluator = SemanticsEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_direct(
            test_case_with_nl,
            actual_nl="The current stock price of Apple is $185.42.",
        )

        assert result.passed is True
        # Score = 0.95 * 0.4 + 0.90 * 0.3 + 1.0 * 0.3 = 0.38 + 0.27 + 0.30 = 0.95
        assert result.score == pytest.approx(0.95, abs=0.01)
        assert result.stage == EvaluationStage.SEMANTICS

    @pytest.mark.asyncio
    async def test_comparison_fails_different_responses(self, mock_llm, test_case_with_nl):
        """Should fail when expected and actual responses are semantically different."""
        # Mock LLM response indicating low similarity
        mock_llm.complete_with_retry.return_value = MagicMock(
            content=json.dumps(
                {
                    "meaning_match": 0.2,
                    "completeness": 0.3,
                    "accuracy": 0.1,
                    "reasoning": "The actual response discusses Microsoft, not Apple.",
                }
            )
        )

        evaluator = SemanticsEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_direct(
            test_case_with_nl,
            actual_nl="Microsoft's stock price is $420.00.",
        )

        assert result.passed is False
        # Score = 0.2 * 0.4 + 0.3 * 0.3 + 0.1 * 0.3 = 0.08 + 0.09 + 0.03 = 0.20
        assert result.score == pytest.approx(0.20, abs=0.01)
        assert result.error_code == ErrorCode.SEMANTIC_LOW_SCORE

    @pytest.mark.asyncio
    async def test_comparison_partial_match_scores_correctly(self, mock_llm, test_case_with_nl):
        """Should return intermediate score for partial matches."""
        # Mock LLM response indicating partial similarity
        mock_llm.complete_with_retry.return_value = MagicMock(
            content=json.dumps(
                {
                    "meaning_match": 0.7,
                    "completeness": 0.6,
                    "accuracy": 0.8,
                    "reasoning": "The response mentions Apple but price is slightly different.",
                }
            )
        )

        evaluator = SemanticsEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_direct(
            test_case_with_nl,
            actual_nl="Apple stock is around $185.",
        )

        # Score = 0.7 * 0.4 + 0.6 * 0.3 + 0.8 * 0.3 = 0.28 + 0.18 + 0.24 = 0.70
        assert result.score == pytest.approx(0.70, abs=0.01)
        # Default threshold is 0.7, so this should just pass
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_score_weighted_average(self, mock_llm, test_case_with_nl):
        """Should calculate score as weighted average of criteria."""
        mock_llm.complete_with_retry.return_value = MagicMock(
            content=json.dumps(
                {
                    "meaning_match": 1.0,
                    "completeness": 0.5,
                    "accuracy": 0.5,
                    "reasoning": "Perfect meaning match but incomplete.",
                }
            )
        )

        # Custom weights
        config = LLMJudgeConfig(
            meaning_weight=0.5,
            completeness_weight=0.25,
            accuracy_weight=0.25,
        )
        evaluator = SemanticsEvaluator(config=config, llm=mock_llm)
        result = await evaluator.evaluate_direct(
            test_case_with_nl,
            actual_nl="Apple is at $185.42",
        )

        # Score = 1.0 * 0.5 + 0.5 * 0.25 + 0.5 * 0.25 = 0.5 + 0.125 + 0.125 = 0.75
        assert result.score == pytest.approx(0.75, abs=0.01)


class TestSemanticsEvaluatorErrorHandling:
    """Tests for error handling in SemanticsEvaluator."""

    @pytest.mark.asyncio
    async def test_llm_error_returns_semantic_llm_error_code(self, mock_llm, test_case_with_nl):
        """Should return SEMANTIC_LLM_ERROR when LLM call fails."""
        mock_llm.complete_with_retry.side_effect = Exception("API Error")

        evaluator = SemanticsEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_direct(
            test_case_with_nl,
            actual_nl="Apple stock price is $185.42.",
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.error_code == ErrorCode.SEMANTIC_LLM_ERROR
        assert "LLM error" in result.reason

    @pytest.mark.asyncio
    async def test_parse_invalid_json_returns_error(self, mock_llm, test_case_with_nl):
        """Should handle invalid JSON response from LLM."""
        mock_llm.complete_with_retry.return_value = MagicMock(content="This is not valid JSON")

        evaluator = SemanticsEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_direct(
            test_case_with_nl,
            actual_nl="Apple stock price is $185.42.",
        )

        assert result.passed is False
        assert result.error_code == ErrorCode.SEMANTIC_LLM_ERROR
        assert "LLM error" in result.reason

    @pytest.mark.asyncio
    async def test_timeout_returns_system_timeout_code(self, mock_llm, test_case_with_nl):
        """Should return SYSTEM_TIMEOUT when LLM call times out."""
        mock_llm.complete_with_retry.side_effect = TimeoutError()

        evaluator = SemanticsEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_direct(
            test_case_with_nl,
            actual_nl="Apple stock price is $185.42.",
        )

        assert result.passed is False
        assert result.score == 0.0
        assert result.error_code == ErrorCode.SYSTEM_TIMEOUT


class TestSemanticsEvaluatorWithGenerator:
    """Tests for SemanticsEvaluator with NL generator."""

    @pytest.mark.asyncio
    async def test_generates_nl_from_expected_response(self, mock_llm, test_case_with_nl):
        """Should call nl_generator with expected_response and compare result."""
        # Mock the generator
        generator_called_with = None

        async def mock_generator(response_data):
            nonlocal generator_called_with
            generator_called_with = response_data
            return "Generated: Apple's stock price is $185.42."

        # Mock LLM comparison
        mock_llm.complete_with_retry.return_value = MagicMock(
            content=json.dumps(
                {
                    "meaning_match": 0.95,
                    "completeness": 0.95,
                    "accuracy": 1.0,
                    "reasoning": "Semantically equivalent responses.",
                }
            )
        )

        evaluator = SemanticsEvaluator(llm=mock_llm)
        result = await evaluator.evaluate(
            test_case_with_nl,
            nl_generator=mock_generator,
        )

        # Verify generator was called with expected_response
        assert generator_called_with == {"AAPL.O": {"P": 185.42}}
        assert result.passed is True
        assert result.artifacts["actual_nl"] == "Generated: Apple's stock price is $185.42."

    @pytest.mark.asyncio
    async def test_nl_generator_receives_expected_response(self, mock_llm, test_case_with_nl):
        """Verifies nl_generator is called with correct data."""
        received_data = None

        async def capture_generator(response_data):
            nonlocal received_data
            received_data = response_data
            return "Some response"

        mock_llm.complete_with_retry.return_value = MagicMock(
            content=json.dumps(
                {
                    "meaning_match": 0.9,
                    "completeness": 0.9,
                    "accuracy": 0.9,
                    "reasoning": "Good match.",
                }
            )
        )

        evaluator = SemanticsEvaluator(llm=mock_llm)
        await evaluator.evaluate(test_case_with_nl, nl_generator=capture_generator)

        assert received_data == test_case_with_nl.expected_response


class TestSemanticsEvaluatorConfig:
    """Tests for SemanticsEvaluator configuration."""

    @pytest.mark.asyncio
    async def test_custom_pass_threshold(self, mock_llm, test_case_with_nl):
        """Should respect custom pass threshold."""
        mock_llm.complete_with_retry.return_value = MagicMock(
            content=json.dumps(
                {
                    "meaning_match": 0.8,
                    "completeness": 0.8,
                    "accuracy": 0.8,
                    "reasoning": "Good but not perfect.",
                }
            )
        )

        # Score will be 0.8 (all weights equal)
        # Default threshold 0.7 -> pass
        # Custom threshold 0.9 -> fail

        config = LLMJudgeConfig(pass_threshold=0.9)
        evaluator = SemanticsEvaluator(config=config, llm=mock_llm)
        result = await evaluator.evaluate_direct(
            test_case_with_nl,
            actual_nl="Apple's price is $185.42",
        )

        assert result.passed is False  # 0.8 < 0.9
        assert result.error_code == ErrorCode.SEMANTIC_LOW_SCORE

    @pytest.mark.asyncio
    async def test_duration_tracked(self, mock_llm, test_case_with_nl):
        """Should track duration_ms in result."""
        mock_llm.complete_with_retry.return_value = MagicMock(
            content=json.dumps(
                {
                    "meaning_match": 0.9,
                    "completeness": 0.9,
                    "accuracy": 0.9,
                    "reasoning": "Match.",
                }
            )
        )

        evaluator = SemanticsEvaluator(llm=mock_llm)
        result = await evaluator.evaluate_direct(
            test_case_with_nl,
            actual_nl="Apple stock is $185.42",
        )

        assert result.duration_ms >= 0


class TestComparisonScores:
    """Tests for ComparisonScores dataclass."""

    def test_weighted_score_calculation(self):
        """Should calculate weighted score with default weights."""
        scores = ComparisonScores(
            meaning_match=1.0,
            completeness=0.5,
            accuracy=0.5,
            reasoning="Test",
        )

        # Default weights: 0.4, 0.3, 0.3
        expected = 1.0 * 0.4 + 0.5 * 0.3 + 0.5 * 0.3
        assert scores.weighted_score == pytest.approx(expected, abs=0.001)

    def test_perfect_scores(self):
        """Should return 1.0 for perfect scores."""
        scores = ComparisonScores(
            meaning_match=1.0,
            completeness=1.0,
            accuracy=1.0,
            reasoning="Perfect match",
        )
        assert scores.weighted_score == 1.0

    def test_zero_scores(self):
        """Should return 0.0 for zero scores."""
        scores = ComparisonScores(
            meaning_match=0.0,
            completeness=0.0,
            accuracy=0.0,
            reasoning="No match",
        )
        assert scores.weighted_score == 0.0
