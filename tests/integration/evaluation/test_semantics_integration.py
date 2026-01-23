"""
Integration tests for SemanticsEvaluator with WaterfallEvaluator.

Tests cover:
- WaterfallEvaluator runs semantics when enabled
- WaterfallEvaluator skips semantics when disabled
- WaterfallEvaluator skips when expected fields are NULL
- Scorecard includes semantics_result
- BatchRunner with semantics enabled
"""

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from CONTRACTS import (
    EvaluationConfig,
    EvaluationStage,
    LLMJudgeConfig,
    Scorecard,
    SystemResponse,
    TestCase,
    TestCaseMetadata,
    ToolCall,
)
from src.evaluation.core.evaluators import WaterfallEvaluator


@pytest.fixture
def test_case_with_nl():
    """Test case with all fields populated."""
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
def test_case_without_nl():
    """Test case without expected_nl_response."""
    return TestCase(
        id="test-002",
        nl_query="What is Apple's stock price?",
        expected_tool_calls=(
            ToolCall(
                tool_name="get_data",
                arguments={"tickers": "AAPL.O", "fields": ["P"]},
            ),
        ),
        expected_response=None,
        expected_nl_response=None,
        metadata=TestCaseMetadata(
            api_version="v1.0",
            complexity_level=1,
        ),
    )


@pytest.fixture
def correct_system_response():
    """System response with correct tool calls and NL response."""
    return SystemResponse(
        raw_output=json.dumps(
            [{"tool_name": "get_data", "arguments": {"tickers": "AAPL.O", "fields": ["P"]}}]
        ),
        nl_response="Apple's current stock price is $185.42.",
        latency_ms=100,
    )


@pytest.fixture
def correct_system_response_without_nl():
    """System response without NL response."""
    return SystemResponse(
        raw_output=json.dumps(
            [{"tool_name": "get_data", "arguments": {"tickers": "AAPL.O", "fields": ["P"]}}]
        ),
        nl_response=None,
        latency_ms=100,
    )


@pytest.fixture
def mock_llm_for_semantics():
    """Mock LLM that returns high similarity scores."""
    llm = MagicMock()
    llm.complete_with_retry = AsyncMock(
        return_value=MagicMock(
            content=json.dumps(
                {
                    "meaning_match": 0.95,
                    "completeness": 0.90,
                    "accuracy": 1.0,
                    "reasoning": "Both responses convey the same stock price information.",
                }
            )
        )
    )
    return llm


class TestWaterfallEvaluatorSemanticsIntegration:
    """Integration tests for WaterfallEvaluator with semantics stage."""

    @pytest.mark.asyncio
    async def test_waterfall_runs_semantics_when_enabled(
        self, test_case_with_nl, correct_system_response, mock_llm_for_semantics
    ):
        """Should run semantics stage when enabled in config."""
        config = EvaluationConfig(semantics_stage_enabled=True)
        llm_config = LLMJudgeConfig()

        evaluator = WaterfallEvaluator(config=config, llm_judge_config=llm_config)

        # Patch the LLM initialization
        with patch.object(evaluator, "_semantics_evaluator", None):
            # Create a mock semantics evaluator
            from src.evaluation.core.semantics import SemanticsEvaluator

            mock_sem_eval = SemanticsEvaluator(config=llm_config, llm=mock_llm_for_semantics)
            evaluator._semantics_evaluator = mock_sem_eval

            scorecard = await evaluator.evaluate(
                test_case=test_case_with_nl,
                system_response=correct_system_response,
                worker_id="test-worker",
            )

        assert scorecard.semantics_result is not None
        assert scorecard.semantics_result.stage == EvaluationStage.SEMANTICS
        assert scorecard.semantics_result.passed is True
        assert scorecard.semantics_result.score > 0.7

    @pytest.mark.asyncio
    async def test_waterfall_skips_semantics_when_disabled(
        self, test_case_with_nl, correct_system_response
    ):
        """Should skip semantics stage when disabled in config."""
        config = EvaluationConfig(semantics_stage_enabled=False)

        evaluator = WaterfallEvaluator(config=config)

        scorecard = await evaluator.evaluate(
            test_case=test_case_with_nl,
            system_response=correct_system_response,
            worker_id="test-worker",
        )

        assert scorecard.semantics_result is None

    @pytest.mark.asyncio
    async def test_waterfall_skips_when_no_nl_response(
        self, test_case_with_nl, correct_system_response_without_nl
    ):
        """Should skip semantics when system response has no NL."""
        config = EvaluationConfig(semantics_stage_enabled=True)

        evaluator = WaterfallEvaluator(config=config)

        scorecard = await evaluator.evaluate(
            test_case=test_case_with_nl,
            system_response=correct_system_response_without_nl,
            worker_id="test-worker",
        )

        # Semantics should be skipped because actual_nl is None
        assert scorecard.semantics_result is None

    @pytest.mark.asyncio
    async def test_waterfall_skips_when_expected_null(
        self, test_case_without_nl, correct_system_response
    ):
        """Should skip semantics when test case has no expected_nl_response."""
        config = EvaluationConfig(semantics_stage_enabled=True)

        evaluator = WaterfallEvaluator(config=config)

        scorecard = await evaluator.evaluate(
            test_case=test_case_without_nl,
            system_response=correct_system_response,
            worker_id="test-worker",
        )

        # Semantics should be skipped because expected_nl_response is None
        assert scorecard.semantics_result is None

    @pytest.mark.asyncio
    async def test_semantics_result_in_scorecard(
        self, test_case_with_nl, correct_system_response, mock_llm_for_semantics
    ):
        """Scorecard should include semantics_result when evaluated."""
        config = EvaluationConfig(semantics_stage_enabled=True)
        llm_config = LLMJudgeConfig()

        evaluator = WaterfallEvaluator(config=config, llm_judge_config=llm_config)

        # Create and inject mock semantics evaluator
        from src.evaluation.core.semantics import SemanticsEvaluator

        mock_sem_eval = SemanticsEvaluator(config=llm_config, llm=mock_llm_for_semantics)
        evaluator._semantics_evaluator = mock_sem_eval

        scorecard = await evaluator.evaluate(
            test_case=test_case_with_nl,
            system_response=correct_system_response,
            worker_id="test-worker",
        )

        # Verify scorecard structure
        assert isinstance(scorecard, Scorecard)
        assert scorecard.syntax_result is not None
        assert scorecard.logic_result is not None
        assert scorecard.semantics_result is not None

        # Verify semantics result details
        sem_result = scorecard.semantics_result
        assert sem_result.stage == EvaluationStage.SEMANTICS
        assert "meaning_match" in sem_result.artifacts
        assert "completeness" in sem_result.artifacts
        assert "accuracy" in sem_result.artifacts

    @pytest.mark.asyncio
    async def test_overall_passed_includes_semantics(
        self, test_case_with_nl, correct_system_response
    ):
        """overall_passed should consider semantics_result."""
        config = EvaluationConfig(semantics_stage_enabled=True)
        llm_config = LLMJudgeConfig()

        evaluator = WaterfallEvaluator(config=config, llm_judge_config=llm_config)

        # Mock a failing semantics evaluator
        mock_llm_fail = MagicMock()
        mock_llm_fail.complete_with_retry = AsyncMock(
            return_value=MagicMock(
                content=json.dumps(
                    {
                        "meaning_match": 0.2,
                        "completeness": 0.2,
                        "accuracy": 0.2,
                        "reasoning": "Completely different responses.",
                    }
                )
            )
        )

        from src.evaluation.core.semantics import SemanticsEvaluator

        mock_sem_eval = SemanticsEvaluator(config=llm_config, llm=mock_llm_fail)
        evaluator._semantics_evaluator = mock_sem_eval

        scorecard = await evaluator.evaluate(
            test_case=test_case_with_nl,
            system_response=correct_system_response,
            worker_id="test-worker",
        )

        # Syntax and logic pass, but semantics fails
        assert scorecard.syntax_result.passed is True
        assert scorecard.logic_result.passed is True
        assert scorecard.semantics_result.passed is False
        # Overall should fail because semantics failed
        assert scorecard.overall_passed is False


class TestBatchRunnerWithSemantics:
    """Integration tests for BatchRunner with semantics enabled."""

    @pytest.mark.asyncio
    async def test_batch_runner_creates_evaluator_with_semantics(self):
        """BatchRunner should create WaterfallEvaluator with semantics config."""
        from src.evaluation.batch.config import BatchRunnerConfig
        from src.evaluation.batch.runner import BatchRunner

        # Mock repositories
        test_case_repo = MagicMock()
        scorecard_repo = MagicMock()
        batch_repo = MagicMock()

        config = BatchRunnerConfig(
            semantics_enabled=True,
            semantics_model="claude-3-5-haiku-20241022",
            semantics_pass_threshold=0.8,
        )

        runner = BatchRunner(
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            batch_repo=batch_repo,
            config=config,
        )

        # Verify evaluator was configured correctly
        assert runner.evaluator.config.semantics_stage_enabled is True
        assert runner.evaluator.llm_judge_config.model == "claude-3-5-haiku-20241022"
        assert runner.evaluator.llm_judge_config.pass_threshold == 0.8

    @pytest.mark.asyncio
    async def test_batch_runner_without_semantics(self):
        """BatchRunner should work with semantics disabled."""
        from src.evaluation.batch.config import BatchRunnerConfig
        from src.evaluation.batch.runner import BatchRunner

        # Mock repositories
        test_case_repo = MagicMock()
        scorecard_repo = MagicMock()
        batch_repo = MagicMock()

        config = BatchRunnerConfig(
            semantics_enabled=False,
        )

        runner = BatchRunner(
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            batch_repo=batch_repo,
            config=config,
        )

        # Verify semantics is disabled
        assert runner.evaluator.config.semantics_stage_enabled is False
