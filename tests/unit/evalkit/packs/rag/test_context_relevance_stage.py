"""
Unit tests for ContextRelevanceStage.

Tests context relevance evaluation using both LLM judge and heuristic methods.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.evalkit.contracts import EvalContext, TestCase
from src.rag.evaluation.llm_judge import JudgeResult
from src.rag.evaluation.stages import ContextRelevanceStage


@pytest.fixture
def stage():
    """Create a ContextRelevanceStage instance."""
    return ContextRelevanceStage()


@pytest.fixture
def mock_llm_judge():
    """Create a mock LLM judge."""
    judge = MagicMock()
    judge.evaluate_relevance = AsyncMock()
    return judge


class TestContextRelevanceHeuristic:
    """Tests for heuristic evaluation (no LLM judge)."""

    @pytest.mark.asyncio
    async def test_no_query_skips(self, stage):
        """Skip if no query in test case."""
        test_case = TestCase(
            id="test-1",
            input={},  # No query
            expected={},
        )
        system_output = {"retrieved_chunks": ["some context"]}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True
        assert result.score == 1.0
        assert result.metrics.get("skipped") is True

    @pytest.mark.asyncio
    async def test_no_context_skips(self, stage):
        """Skip if no context retrieved."""
        test_case = TestCase(
            id="test-2",
            input={"query": "What is Python?"},
            expected={},
        )
        system_output = {}  # No retrieved chunks

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True
        assert result.score == 1.0
        assert result.metrics.get("skipped") is True

    @pytest.mark.asyncio
    async def test_heuristic_high_overlap(self, stage):
        """High keyword overlap should score well."""
        test_case = TestCase(
            id="test-3",
            input={"query": "Python programming language features"},
            expected={},
        )
        system_output = {
            "retrieved_chunks": [
                "Python is a programming language with many features",
                "Python programming is popular for its readability",
            ]
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.metrics["evaluation_method"] == "heuristic"
        # High keyword overlap should give good score
        assert result.score > 0.5

    @pytest.mark.asyncio
    async def test_heuristic_low_overlap(self, stage):
        """Low keyword overlap should score poorly."""
        test_case = TestCase(
            id="test-4",
            input={"query": "Python programming language"},
            expected={},
        )
        system_output = {
            "retrieved_chunks": [
                "JavaScript is used for web development",
                "Node.js runs JavaScript on servers",
            ]
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.metrics["evaluation_method"] == "heuristic"
        # Low overlap should give low score
        assert result.score < 0.5

    @pytest.mark.asyncio
    async def test_chunks_as_dicts(self, stage):
        """Handle chunks as list of dicts."""
        test_case = TestCase(
            id="test-5",
            input={"query": "machine learning"},
            expected={},
        )
        system_output = {
            "retrieved_chunks": [
                {"text": "Machine learning is a type of AI", "id": "1"},
                {"text": "Learning algorithms process data", "id": "2"},
            ]
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.score > 0  # Should process dict chunks

    @pytest.mark.asyncio
    async def test_context_as_string(self, stage):
        """Handle context as single string."""
        test_case = TestCase(
            id="test-6",
            input={"query": "data science"},
            expected={},
        )
        system_output = {"context": "Data science involves analyzing and interpreting data"}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.score > 0


class TestContextRelevanceWithLLM:
    """Tests for LLM-based evaluation."""

    @pytest.mark.asyncio
    async def test_llm_judge_called(self, stage, mock_llm_judge):
        """LLM judge is called when available."""
        mock_llm_judge.evaluate_relevance.return_value = JudgeResult(
            score=0.85,
            passed=True,
            reasoning="Context is highly relevant",
            raw_response="",
        )

        context = EvalContext(config={"llm_judge": mock_llm_judge})
        test_case = TestCase(
            id="test-7",
            input={"query": "test query"},
            expected={},
        )
        system_output = {"retrieved_chunks": ["relevant context"]}

        result = await stage.evaluate(test_case, system_output, context)

        mock_llm_judge.evaluate_relevance.assert_called_once()
        assert result.score == pytest.approx(0.85)

    @pytest.mark.asyncio
    async def test_multiple_chunks_evaluated(self, stage, mock_llm_judge):
        """Multiple chunks are evaluated and aggregated."""
        mock_llm_judge.evaluate_relevance.side_effect = [
            JudgeResult(score=0.9, passed=True, reasoning="Good", raw_response=""),
            JudgeResult(score=0.7, passed=True, reasoning="OK", raw_response=""),
            JudgeResult(score=0.5, passed=False, reasoning="Weak", raw_response=""),
        ]

        context = EvalContext(config={"llm_judge": mock_llm_judge})
        test_case = TestCase(
            id="test-8",
            input={"query": "test query"},
            expected={},
        )
        system_output = {"retrieved_chunks": ["chunk1", "chunk2", "chunk3"]}

        result = await stage.evaluate(test_case, system_output, context)

        assert mock_llm_judge.evaluate_relevance.call_count == 3
        assert result.metrics["num_chunks_evaluated"] == 3

    @pytest.mark.asyncio
    async def test_max_chunks_limit(self, stage, mock_llm_judge):
        """Respects max chunks limit."""
        stage.max_chunks_to_evaluate = 2

        mock_llm_judge.evaluate_relevance.return_value = JudgeResult(
            score=0.8, passed=True, reasoning="Good", raw_response=""
        )

        context = EvalContext(config={"llm_judge": mock_llm_judge})
        test_case = TestCase(
            id="test-9",
            input={"query": "test query"},
            expected={},
        )
        system_output = {"retrieved_chunks": ["chunk1", "chunk2", "chunk3", "chunk4", "chunk5"]}

        result = await stage.evaluate(test_case, system_output, context)

        # Should only evaluate 2 chunks
        assert mock_llm_judge.evaluate_relevance.call_count == 2
        assert result.metrics["num_chunks_evaluated"] == 2
        assert result.metrics["num_chunks_total"] == 5

    @pytest.mark.asyncio
    async def test_llm_judge_error_handled(self, stage, mock_llm_judge):
        """Handle LLM judge errors gracefully."""
        mock_llm_judge.evaluate_relevance.side_effect = Exception("API Error")

        context = EvalContext(config={"llm_judge": mock_llm_judge})
        test_case = TestCase(
            id="test-10",
            input={"query": "test query"},
            expected={},
        )
        system_output = {"retrieved_chunks": ["chunk1"]}

        result = await stage.evaluate(test_case, system_output, context)

        # Should continue with default score on error
        assert result.stage_name == "context_relevance"


class TestContextRelevanceConfiguration:
    """Tests for stage configuration."""

    def test_stage_properties(self, stage):
        """Verify stage properties."""
        assert stage.name == "context_relevance"
        assert stage.is_gate is False

    @pytest.mark.asyncio
    async def test_custom_pass_threshold(self):
        """Stage respects custom pass threshold."""
        low_stage = ContextRelevanceStage(pass_threshold=0.3)
        high_stage = ContextRelevanceStage(pass_threshold=0.9)

        test_case = TestCase(
            id="test-11",
            input={"query": "Python programming"},
            expected={},
        )
        system_output = {"retrieved_chunks": ["Python is a programming language"]}

        low_result = await low_stage.evaluate(test_case, system_output, None)
        _high_result = await high_stage.evaluate(test_case, system_output, None)  # noqa: F841

        # Same input should pass low threshold but may fail high
        assert low_result.passed is True
        # High threshold may or may not pass depending on score
