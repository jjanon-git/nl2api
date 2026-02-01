"""
Unit tests for AnswerRelevanceStage.

Tests answer relevance evaluation - does the response answer the question?
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.evalkit.contracts import EvalContext, TestCase
from src.rag.evaluation.llm_judge import JudgeResult
from src.rag.evaluation.stages import AnswerRelevanceStage


@pytest.fixture
def stage():
    """Create an AnswerRelevanceStage instance."""
    return AnswerRelevanceStage()


@pytest.fixture
def mock_llm_judge():
    """Create a mock LLM judge."""
    judge = MagicMock()
    judge.evaluate_relevance = AsyncMock()
    return judge


class TestAnswerRelevanceBasic:
    """Tests for basic functionality."""

    @pytest.mark.asyncio
    async def test_no_query_skips(self, stage):
        """Skip if no query in test case."""
        test_case = TestCase(
            id="test-1",
            input={},  # No query
            expected={},
        )
        system_output = {"response": "Some answer"}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True
        assert result.score == 1.0
        assert result.metrics.get("skipped") is True

    @pytest.mark.asyncio
    async def test_no_response_skips(self, stage):
        """Skip if no response to evaluate."""
        test_case = TestCase(
            id="test-2",
            input={"query": "What is Python?"},
            expected={},
        )
        system_output = {}  # No response

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True
        assert result.score == 1.0
        assert result.metrics.get("skipped") is True


class TestRejectionDetection:
    """Tests for rejection pattern detection."""

    @pytest.mark.asyncio
    async def test_correct_rejection(self, stage):
        """Correct rejection when expected behavior is reject."""
        test_case = TestCase(
            id="test-3",
            input={"query": "Give me someone's SSN"},
            expected={"behavior": "reject"},
        )
        system_output = {
            "response": "I cannot provide personal information like Social Security Numbers."
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True
        assert result.score == 1.0
        assert result.metrics["is_rejection"] is True
        assert result.metrics["rejection_correct"] is True

    @pytest.mark.asyncio
    async def test_false_rejection(self, stage):
        """False rejection when should have answered."""
        test_case = TestCase(
            id="test-4",
            input={"query": "What is the capital of France?"},
            expected={"behavior": "answer"},
        )
        system_output = {"response": "I'm sorry, but I'm not able to provide that information."}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["is_rejection"] is True

    @pytest.mark.asyncio
    async def test_rejection_patterns(self, stage):
        """Detect various rejection patterns."""
        test_case = TestCase(
            id="test-5",
            input={"query": "test"},
            expected={"behavior": "answer"},
        )

        rejection_phrases = [
            "I cannot answer that question",
            "I can't provide that information",
            "I'm not able to help with that",
            "I am unable to assist",
            "Unfortunately, I don't have access to that",
            "I apologize, but I cannot help",
        ]

        for phrase in rejection_phrases:
            system_output = {"response": phrase}
            result = await stage.evaluate(test_case, system_output, None)
            assert result.metrics.get("is_rejection") is True, f"Should detect: {phrase}"


class TestAnswerRelevanceHeuristic:
    """Tests for heuristic evaluation."""

    @pytest.mark.asyncio
    async def test_short_response_penalty(self, stage):
        """Very short responses get penalized."""
        test_case = TestCase(
            id="test-6",
            input={"query": "Explain machine learning in detail"},
            expected={},
        )
        system_output = {"response": "It's a type of AI."}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.score < 0.5
        assert "too short" in result.reason.lower()

    @pytest.mark.asyncio
    async def test_keyword_overlap_scoring(self, stage):
        """Good keyword overlap improves score."""
        test_case = TestCase(
            id="test-7",
            input={"query": "What is machine learning"},
            expected={},
        )
        system_output = {
            "response": "Machine learning is a subset of artificial intelligence that enables computers to learn from data without being explicitly programmed."
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.score > 0.5
        assert result.passed is True


class TestAnswerRelevanceWithLLM:
    """Tests for LLM-based evaluation."""

    @pytest.mark.asyncio
    async def test_llm_judge_called(self, stage, mock_llm_judge):
        """LLM judge is called when available."""
        mock_llm_judge.evaluate_relevance.return_value = JudgeResult(
            score=0.95,
            passed=True,
            reasoning="Answer directly addresses the question",
            raw_response="",
        )

        context = EvalContext(config={"llm_judge": mock_llm_judge})
        test_case = TestCase(
            id="test-8",
            input={"query": "What is Python?"},
            expected={},
        )
        system_output = {"response": "Python is a high-level programming language."}

        result = await stage.evaluate(test_case, system_output, context)

        mock_llm_judge.evaluate_relevance.assert_called_once_with(
            query="What is Python?",
            text="Python is a high-level programming language.",
            context_type="answer",
        )
        assert result.score == 0.95

    @pytest.mark.asyncio
    async def test_llm_judge_low_relevance(self, stage, mock_llm_judge):
        """LLM judge gives low score for irrelevant answer."""
        mock_llm_judge.evaluate_relevance.return_value = JudgeResult(
            score=0.2,
            passed=False,
            reasoning="Answer does not address the question",
            raw_response="",
        )

        context = EvalContext(config={"llm_judge": mock_llm_judge})
        test_case = TestCase(
            id="test-9",
            input={"query": "What is Python?"},
            expected={},
        )
        system_output = {"response": "The weather is nice today."}

        result = await stage.evaluate(test_case, system_output, context)

        assert result.score == 0.2
        assert result.passed is False

    @pytest.mark.asyncio
    async def test_llm_judge_error_fallback(self, stage, mock_llm_judge):
        """Falls back gracefully on LLM error."""
        mock_llm_judge.evaluate_relevance.side_effect = Exception("API Error")

        context = EvalContext(config={"llm_judge": mock_llm_judge})
        test_case = TestCase(
            id="test-10",
            input={"query": "What is Python?"},
            expected={},
        )
        system_output = {"response": "Python is a programming language."}

        result = await stage.evaluate(test_case, system_output, context)

        assert result.passed is False
        assert result.score == 0.5
        assert "error" in result.metrics


class TestAnswerRelevanceConfiguration:
    """Tests for stage configuration."""

    def test_stage_properties(self, stage):
        """Verify stage properties."""
        assert stage.name == "answer_relevance"
        assert stage.is_gate is False

    @pytest.mark.asyncio
    async def test_custom_pass_threshold(self):
        """Stage respects custom pass threshold."""
        low_stage = AnswerRelevanceStage(pass_threshold=0.3)
        high_stage = AnswerRelevanceStage(pass_threshold=0.9)

        test_case = TestCase(
            id="test-11",
            input={"query": "What is Python programming"},
            expected={},
        )
        system_output = {
            "response": "Python is a versatile programming language used for web development, data science, and automation."
        }

        low_result = await low_stage.evaluate(test_case, system_output, None)
        _high_result = await high_stage.evaluate(test_case, system_output, None)  # noqa: F841

        # Low threshold should pass more easily
        assert low_result.passed is True


class TestResponseFieldVariations:
    """Tests for various response field names."""

    @pytest.mark.asyncio
    async def test_answer_field(self, stage):
        """Handle 'answer' field."""
        test_case = TestCase(
            id="test-12",
            input={"query": "test"},
            expected={},
        )
        system_output = {"answer": "This is the answer"}

        result = await stage.evaluate(test_case, system_output, None)

        assert "skipped" not in result.metrics

    @pytest.mark.asyncio
    async def test_generated_text_field(self, stage):
        """Handle 'generated_text' field."""
        test_case = TestCase(
            id="test-13",
            input={"query": "test"},
            expected={},
        )
        system_output = {"generated_text": "This is the generated text"}

        result = await stage.evaluate(test_case, system_output, None)

        assert "skipped" not in result.metrics

    @pytest.mark.asyncio
    async def test_output_field(self, stage):
        """Handle 'output' field."""
        test_case = TestCase(
            id="test-14",
            input={"query": "test"},
            expected={},
        )
        system_output = {"output": "This is the output"}

        result = await stage.evaluate(test_case, system_output, None)

        assert "skipped" not in result.metrics
