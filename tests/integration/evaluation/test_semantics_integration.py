"""
Integration tests for SemanticsEvaluator with NL2APIPack.

Tests cover:
- NL2APIPack runs semantics when enabled
- NL2APIPack skips semantics when disabled
- NL2APIPack skips when expected fields are NULL
- Scorecard includes semantics_result
- BatchRunner with semantics enabled
"""

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from CONTRACTS import (
    EvalContext,
    EvaluationStage,
    LLMJudgeConfig,
    Scorecard,
    SystemResponse,
    TestCase,
    TestCaseMetadata,
    ToolCall,
)
from src.evaluation.packs import NL2APIPack


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


def _make_system_output(response: SystemResponse) -> dict:
    """Convert SystemResponse to system_output dict for NL2APIPack."""
    return {
        "raw_output": response.raw_output,
        "nl_response": response.nl_response,
    }


class TestNL2APIPackSemanticsIntegration:
    """Integration tests for NL2APIPack with semantics stage."""

    @pytest.mark.asyncio
    async def test_pack_runs_semantics_when_enabled(
        self, test_case_with_nl, correct_system_response, mock_llm_for_semantics
    ):
        """Should run semantics stage when enabled."""
        llm_config = LLMJudgeConfig()
        pack = NL2APIPack(semantics_enabled=True)

        # Create a mock semantics evaluator and pass via context
        from src.evalkit.core.semantics import SemanticsEvaluator

        mock_sem_eval = SemanticsEvaluator(config=llm_config, llm=mock_llm_for_semantics)
        context = EvalContext(
            worker_id="test-worker",
            config={"semantics_evaluator": mock_sem_eval},
        )

        scorecard = await pack.evaluate(
            test_case=test_case_with_nl,
            system_output=_make_system_output(correct_system_response),
            context=context,
        )

        assert scorecard.semantics_result is not None
        assert scorecard.semantics_result.stage == EvaluationStage.SEMANTICS
        assert scorecard.semantics_result.passed is True
        assert scorecard.semantics_result.score > 0.7

    @pytest.mark.asyncio
    async def test_pack_skips_semantics_when_disabled(
        self, test_case_with_nl, correct_system_response
    ):
        """Should skip semantics stage when disabled."""
        pack = NL2APIPack(semantics_enabled=False)

        scorecard = await pack.evaluate(
            test_case=test_case_with_nl,
            system_output=_make_system_output(correct_system_response),
            context=EvalContext(worker_id="test-worker"),
        )

        assert scorecard.semantics_result is None

    @pytest.mark.asyncio
    async def test_pack_skips_when_no_nl_response(
        self, test_case_with_nl, correct_system_response_without_nl
    ):
        """Should skip semantics when system response has no NL."""
        pack = NL2APIPack(semantics_enabled=True)

        scorecard = await pack.evaluate(
            test_case=test_case_with_nl,
            system_output=_make_system_output(correct_system_response_without_nl),
            context=EvalContext(worker_id="test-worker"),
        )

        # Semantics should still run but pass with "skipped" reason
        assert scorecard.semantics_result is not None
        assert "Skipped" in scorecard.semantics_result.reason

    @pytest.mark.asyncio
    async def test_pack_skips_when_expected_null(
        self, test_case_without_nl, correct_system_response
    ):
        """Should skip semantics when test case has no expected_nl_response."""
        pack = NL2APIPack(semantics_enabled=True)

        scorecard = await pack.evaluate(
            test_case=test_case_without_nl,
            system_output=_make_system_output(correct_system_response),
            context=EvalContext(worker_id="test-worker"),
        )

        # Semantics should still run but pass with "skipped" reason
        assert scorecard.semantics_result is not None
        assert "Skipped" in scorecard.semantics_result.reason

    @pytest.mark.asyncio
    async def test_semantics_result_in_scorecard(
        self, test_case_with_nl, correct_system_response, mock_llm_for_semantics
    ):
        """Scorecard should include semantics_result when evaluated."""
        llm_config = LLMJudgeConfig()
        pack = NL2APIPack(semantics_enabled=True)

        # Create and inject mock semantics evaluator
        from src.evalkit.core.semantics import SemanticsEvaluator

        mock_sem_eval = SemanticsEvaluator(config=llm_config, llm=mock_llm_for_semantics)
        context = EvalContext(
            worker_id="test-worker",
            config={"semantics_evaluator": mock_sem_eval},
        )

        scorecard = await pack.evaluate(
            test_case=test_case_with_nl,
            system_output=_make_system_output(correct_system_response),
            context=context,
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
        llm_config = LLMJudgeConfig()
        pack = NL2APIPack(semantics_enabled=True)

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

        from src.evalkit.core.semantics import SemanticsEvaluator

        mock_sem_eval = SemanticsEvaluator(config=llm_config, llm=mock_llm_fail)
        context = EvalContext(
            worker_id="test-worker",
            config={"semantics_evaluator": mock_sem_eval},
        )

        scorecard = await pack.evaluate(
            test_case=test_case_with_nl,
            system_output=_make_system_output(correct_system_response),
            context=context,
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
    async def test_batch_runner_creates_pack_with_semantics(self):
        """BatchRunner should create NL2APIPack with semantics enabled."""
        from src.evalkit.batch.config import BatchRunnerConfig
        from src.evalkit.batch.runner import BatchRunner
        from src.evaluation.packs.nl2api import NL2APIPack

        # Mock repositories
        test_case_repo = MagicMock()
        scorecard_repo = MagicMock()
        batch_repo = MagicMock()

        config = BatchRunnerConfig(
            pack_name="nl2api",
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

        # Verify pack was configured correctly
        assert isinstance(runner.pack, NL2APIPack)
        assert runner.pack.semantics_enabled is True

    @pytest.mark.asyncio
    async def test_batch_runner_without_semantics(self):
        """BatchRunner should create NL2APIPack with semantics disabled."""
        from src.evalkit.batch.config import BatchRunnerConfig
        from src.evalkit.batch.runner import BatchRunner
        from src.evaluation.packs.nl2api import NL2APIPack

        # Mock repositories
        test_case_repo = MagicMock()
        scorecard_repo = MagicMock()
        batch_repo = MagicMock()

        config = BatchRunnerConfig(
            pack_name="nl2api",
            semantics_enabled=False,
        )

        runner = BatchRunner(
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            batch_repo=batch_repo,
            config=config,
        )

        # Verify pack has semantics disabled
        assert isinstance(runner.pack, NL2APIPack)
        assert runner.pack.semantics_enabled is False
