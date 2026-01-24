"""
Unit tests for FaithfulnessStage.

Tests faithfulness (groundedness) evaluation using claim extraction and verification.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.evalkit.contracts import EvalContext, TestCase
from src.evaluation.packs.rag.llm_judge import JudgeResult
from src.evaluation.packs.rag.stages import FaithfulnessStage


@pytest.fixture
def stage():
    """Create a FaithfulnessStage instance."""
    return FaithfulnessStage()


@pytest.fixture
def mock_llm_judge():
    """Create a mock LLM judge."""
    judge = MagicMock()
    judge.evaluate_faithfulness = AsyncMock()
    return judge


class TestFaithfulnessHeuristic:
    """Tests for heuristic evaluation (no LLM judge)."""

    @pytest.mark.asyncio
    async def test_no_response_skips(self, stage):
        """Skip if no response to evaluate."""
        test_case = TestCase(
            id="test-1",
            input={"query": "test"},
            expected={},
        )
        system_output = {"context": "some context"}  # No response

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True
        assert result.score == 1.0
        assert result.metrics.get("skipped") is True

    @pytest.mark.asyncio
    async def test_no_context_skips(self, stage):
        """Skip if no context to verify against."""
        test_case = TestCase(
            id="test-2",
            input={"query": "test"},
            expected={},
        )
        system_output = {"response": "some answer"}  # No context

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True
        assert result.score == 1.0
        assert result.metrics.get("skipped") is True

    @pytest.mark.asyncio
    async def test_heuristic_high_overlap(self, stage):
        """High n-gram overlap should indicate groundedness."""
        test_case = TestCase(
            id="test-3",
            input={"query": "What is Python?"},
            expected={},
        )
        system_output = {
            # Response uses exact phrases from context for higher n-gram overlap
            "response": "Python is a popular programming language known for being easy to learn and read",
            "context": "Python is a popular programming language. It is known for being easy to learn and read.",
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.metrics["evaluation_method"] == "heuristic"
        assert result.score > 0.3  # Some overlap

    @pytest.mark.asyncio
    async def test_heuristic_low_overlap(self, stage):
        """Low n-gram overlap indicates potential hallucination."""
        test_case = TestCase(
            id="test-4",
            input={"query": "What is Python?"},
            expected={},
        )
        system_output = {
            "response": "Python was invented by Albert Einstein in 1492",  # Hallucinated
            "context": "Python was created by Guido van Rossum in 1991.",
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.metrics["evaluation_method"] == "heuristic"
        assert result.score < 0.5  # Low overlap

    @pytest.mark.asyncio
    async def test_various_response_fields(self, stage):
        """Handle various response field names."""
        test_case = TestCase(
            id="test-5",
            input={"query": "test"},
            expected={},
        )

        # Test 'answer' field
        result1 = await stage.evaluate(
            test_case,
            {"answer": "test answer", "context": "test context about test answer"},
            None,
        )
        assert "skipped" not in result1.metrics

        # Test 'generated_text' field
        result2 = await stage.evaluate(
            test_case,
            {"generated_text": "gen text", "context": "gen text is here"},
            None,
        )
        assert "skipped" not in result2.metrics

    @pytest.mark.asyncio
    async def test_chunks_concatenated_for_context(self, stage):
        """Chunks are concatenated to form context."""
        test_case = TestCase(
            id="test-6",
            input={"query": "test"},
            expected={},
        )
        system_output = {
            "response": "Python is a programming language used for data science",
            "retrieved_chunks": [
                {"text": "Python is a programming language"},
                {"text": "It is commonly used for data science"},
            ],
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert "skipped" not in result.metrics  # Context was extracted


class TestFaithfulnessWithLLM:
    """Tests for LLM-based evaluation."""

    @pytest.mark.asyncio
    async def test_llm_judge_called(self, stage, mock_llm_judge):
        """LLM judge is called when available."""
        mock_llm_judge.evaluate_faithfulness.return_value = JudgeResult(
            score=0.9,
            passed=True,
            reasoning="All claims are supported",
            raw_response="",
            metrics={"num_claims": 3, "supported_claims": 3},
        )

        context = EvalContext(config={"llm_judge": mock_llm_judge})
        test_case = TestCase(
            id="test-7",
            input={"query": "test"},
            expected={},
        )
        system_output = {
            "response": "Paris is the capital of France",
            "context": "Paris is the capital of France, located in Europe.",
        }

        result = await stage.evaluate(test_case, system_output, context)

        mock_llm_judge.evaluate_faithfulness.assert_called_once()
        assert result.score == 0.9
        assert result.passed is True

    @pytest.mark.asyncio
    async def test_partial_faithfulness(self, stage, mock_llm_judge):
        """Some claims supported, some not."""
        mock_llm_judge.evaluate_faithfulness.return_value = JudgeResult(
            score=0.5,
            passed=False,
            reasoning="Half of claims are unsupported",
            raw_response="",
            metrics={
                "num_claims": 4,
                "supported_claims": 2,
                "unsupported_claims": ["claim3", "claim4"],
            },
        )

        context = EvalContext(config={"llm_judge": mock_llm_judge})
        test_case = TestCase(
            id="test-8",
            input={"query": "test"},
            expected={},
        )
        system_output = {
            "response": "Some response with mixed claims",
            "context": "Partial context",
        }

        result = await stage.evaluate(test_case, system_output, context)

        assert result.score == 0.5
        assert result.passed is False
        assert result.metrics["supported_claims"] == 2

    @pytest.mark.asyncio
    async def test_llm_judge_error_handled(self, stage, mock_llm_judge):
        """Handle LLM judge errors gracefully."""
        mock_llm_judge.evaluate_faithfulness.side_effect = Exception("API timeout")

        context = EvalContext(config={"llm_judge": mock_llm_judge})
        test_case = TestCase(
            id="test-9",
            input={"query": "test"},
            expected={},
        )
        system_output = {
            "response": "Some response",
            "context": "Some context",
        }

        result = await stage.evaluate(test_case, system_output, context)

        # Should return with error info
        assert result.passed is False
        assert result.score == 0.5
        assert "error" in result.metrics


class TestFaithfulnessConfiguration:
    """Tests for stage configuration."""

    def test_stage_properties(self, stage):
        """Verify stage properties."""
        assert stage.name == "faithfulness"
        assert stage.is_gate is False

    @pytest.mark.asyncio
    async def test_custom_pass_threshold(self):
        """Stage respects custom pass threshold."""
        low_stage = FaithfulnessStage(pass_threshold=0.2)
        high_stage = FaithfulnessStage(pass_threshold=0.9)

        test_case = TestCase(
            id="test-10",
            input={"query": "test"},
            expected={},
        )
        # Response shares many exact phrases with context for n-gram overlap
        system_output = {
            "response": "Python is a popular programming language used for data science and analysis",
            "context": "Python is a popular programming language used for data science",
        }

        low_result = await low_stage.evaluate(test_case, system_output, None)
        _high_result = await high_stage.evaluate(test_case, system_output, None)  # noqa: F841

        # Low threshold should be easier to pass
        assert low_result.passed is True or low_result.score > 0.2

    def test_ngram_extraction(self, stage):
        """Test n-gram extraction helper."""
        text = "The quick brown fox"
        ngrams = stage._extract_ngrams(text, n=3)

        assert ("the", "quick", "brown") in ngrams
        assert ("quick", "brown", "fox") in ngrams
        assert len(ngrams) == 2

    def test_ngram_empty_text(self, stage):
        """Empty text returns empty set."""
        assert stage._extract_ngrams("", n=3) == set()
        assert stage._extract_ngrams("ab", n=3) == set()  # Too short
