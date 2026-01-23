"""Tests for ambiguity detector."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from src.nl2api.clarification.detector import AmbiguityAnalysis, AmbiguityDetector
from src.nl2api.llm.protocols import LLMResponse
from src.nl2api.models import ClarificationQuestion


@dataclass
class MockLLMProvider:
    """Mock LLM provider for testing."""

    model_name: str = "mock"
    response_content: str = "CLEAR"

    async def complete(self, messages, tools=None, temperature=0.0, max_tokens=4096):
        return LLMResponse(content=self.response_content)

    async def complete_with_retry(
        self, messages, tools=None, temperature=0.0, max_tokens=4096, max_retries=3
    ):
        return LLMResponse(content=self.response_content)


class TestAmbiguityAnalysis:
    """Test suite for AmbiguityAnalysis dataclass."""

    def test_clear_query(self) -> None:
        """Test analysis for clear query."""
        analysis = AmbiguityAnalysis(
            is_ambiguous=False,
            confidence=1.0,
        )
        assert not analysis.is_ambiguous
        assert analysis.confidence == 1.0
        assert len(analysis.ambiguity_types) == 0

    def test_ambiguous_query(self) -> None:
        """Test analysis for ambiguous query."""
        questions = (ClarificationQuestion(question="Which company?", category="entity"),)
        analysis = AmbiguityAnalysis(
            is_ambiguous=True,
            ambiguity_types=("entity",),
            clarification_questions=questions,
            confidence=0.8,
        )
        assert analysis.is_ambiguous
        assert "entity" in analysis.ambiguity_types
        assert len(analysis.clarification_questions) == 1

    def test_multiple_ambiguities(self) -> None:
        """Test analysis with multiple ambiguity types."""
        analysis = AmbiguityAnalysis(
            is_ambiguous=True,
            ambiguity_types=("entity", "time_period", "metric"),
            clarification_questions=(
                ClarificationQuestion(question="Q1", category="entity"),
                ClarificationQuestion(question="Q2", category="time_period"),
            ),
        )
        assert len(analysis.ambiguity_types) == 3


class TestAmbiguityDetectorEntityAmbiguity:
    """Test suite for entity ambiguity detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.detector = AmbiguityDetector()

    @pytest.mark.asyncio
    async def test_clear_entity_query(self) -> None:
        """Test that clear entity queries are not flagged."""
        result = await self.detector.analyze(
            "What is Apple's EPS estimate?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        # Entity is resolved, should not be flagged for entity ambiguity
        assert "entity" not in result.ambiguity_types

    @pytest.mark.asyncio
    async def test_missing_entity_with_company_reference(self) -> None:
        """Test detection of missing entity with company reference."""
        result = await self.detector.analyze(
            "What is the company's revenue?",
            resolved_entities=None,
        )

        assert result.is_ambiguous
        assert "entity" in result.ambiguity_types

    @pytest.mark.asyncio
    async def test_pronoun_without_antecedent(self) -> None:
        """Test detection of pronouns without clear antecedent."""
        result = await self.detector.analyze(
            "What is their EPS?",
            resolved_entities=None,
        )

        assert result.is_ambiguous
        assert "entity" in result.ambiguity_types

    @pytest.mark.asyncio
    async def test_pronoun_with_resolved_entity(self) -> None:
        """Test that pronouns with resolved entities are not flagged."""
        result = await self.detector.analyze(
            "What is their EPS?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        # Entity is resolved, pronoun is OK
        assert "entity" not in result.ambiguity_types


class TestAmbiguityDetectorTimeAmbiguity:
    """Test suite for time period ambiguity detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.detector = AmbiguityDetector()

    @pytest.mark.asyncio
    async def test_clear_time_query(self) -> None:
        """Test that clear time queries are not flagged."""
        result = await self.detector.analyze(
            "What is Apple's Q4 2023 EPS?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        assert "time_period" not in result.ambiguity_types

    @pytest.mark.asyncio
    async def test_ambiguous_recent(self) -> None:
        """Test detection of 'recent' time reference."""
        result = await self.detector.analyze(
            "What are Apple's recent earnings?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        assert result.is_ambiguous
        assert "time_period" in result.ambiguity_types

    @pytest.mark.asyncio
    async def test_ambiguous_current(self) -> None:
        """Test detection of 'current' time reference."""
        result = await self.detector.analyze(
            "What is the current EPS estimate?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        assert result.is_ambiguous
        assert "time_period" in result.ambiguity_types

    @pytest.mark.asyncio
    async def test_ambiguous_this_year(self) -> None:
        """Test detection of 'this year' time reference."""
        result = await self.detector.analyze(
            "What is Apple's revenue this year?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        assert result.is_ambiguous
        assert "time_period" in result.ambiguity_types

    @pytest.mark.asyncio
    async def test_specific_time_not_flagged(self) -> None:
        """Test that specific time references are not flagged."""
        result = await self.detector.analyze(
            "What is Apple's FY2024 revenue?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        assert "time_period" not in result.ambiguity_types

    @pytest.mark.asyncio
    async def test_month_name_not_flagged(self) -> None:
        """Test that month names are considered specific."""
        result = await self.detector.analyze(
            "What is Apple's revenue for January 2024?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        assert "time_period" not in result.ambiguity_types


class TestAmbiguityDetectorMetricAmbiguity:
    """Test suite for metric ambiguity detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.detector = AmbiguityDetector()

    @pytest.mark.asyncio
    async def test_clear_metric_query(self) -> None:
        """Test that clear metric queries are not flagged."""
        result = await self.detector.analyze(
            "What is Apple's EPS estimate?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        assert "metric" not in result.ambiguity_types

    @pytest.mark.asyncio
    async def test_vague_performance_query(self) -> None:
        """Test detection of vague 'performance' queries."""
        result = await self.detector.analyze(
            "How is Apple performing?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        assert result.is_ambiguous
        assert "metric" in result.ambiguity_types

    @pytest.mark.asyncio
    async def test_vague_results_query(self) -> None:
        """Test detection of vague 'results' queries."""
        result = await self.detector.analyze(
            "What are Apple's results?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        assert result.is_ambiguous
        assert "metric" in result.ambiguity_types

    @pytest.mark.asyncio
    async def test_vague_how_is_query(self) -> None:
        """Test detection of vague 'how is' queries."""
        result = await self.detector.analyze(
            "How is the stock doing?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        assert result.is_ambiguous
        assert "metric" in result.ambiguity_types

    @pytest.mark.asyncio
    async def test_specific_metric_not_flagged(self) -> None:
        """Test that specific metrics are not flagged."""
        test_cases = [
            "What is Apple's EPS?",
            "What is Apple's revenue?",
            "What is Apple's stock price?",
            "What is Apple's P/E ratio?",
            "What is Apple's dividend yield?",
        ]

        for query in test_cases:
            result = await self.detector.analyze(
                query,
                resolved_entities={"Apple": "AAPL.O"},
            )
            assert "metric" not in result.ambiguity_types, f"Failed for: {query}"


class TestAmbiguityDetectorLLMIntegration:
    """Test suite for LLM-based ambiguity detection."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.mock_llm = MockLLMProvider()

    @pytest.mark.asyncio
    async def test_llm_detection_disabled_by_default(self) -> None:
        """Test that LLM detection is disabled by default."""
        detector = AmbiguityDetector(llm=self.mock_llm, use_llm_detection=False)

        # Even with LLM available, it shouldn't be used
        self.mock_llm.response_content = "AMBIGUOUS: Test question"

        result = await detector.analyze(
            "Clear query about Apple's EPS",
            resolved_entities={"Apple": "AAPL.O"},
        )

        # LLM shouldn't have been called
        assert "llm_detected" not in result.ambiguity_types

    @pytest.mark.asyncio
    async def test_llm_detection_when_enabled(self) -> None:
        """Test LLM detection when enabled."""
        self.mock_llm.response_content = "AMBIGUOUS: What specific data are you looking for?"
        detector = AmbiguityDetector(llm=self.mock_llm, use_llm_detection=True)

        result = await detector.analyze(
            "Some ambiguous query",
            resolved_entities=None,
        )

        assert "llm_detected" in result.ambiguity_types

    @pytest.mark.asyncio
    async def test_llm_clear_response(self) -> None:
        """Test LLM returning CLEAR response."""
        self.mock_llm.response_content = "CLEAR"
        detector = AmbiguityDetector(llm=self.mock_llm, use_llm_detection=True)

        result = await detector.analyze(
            "What is Apple's EPS estimate?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        # Should not add LLM-detected ambiguity
        assert "llm_detected" not in result.ambiguity_types

    @pytest.mark.asyncio
    async def test_llm_error_handling(self) -> None:
        """Test handling of LLM errors."""

        async def raise_error(*args, **kwargs):
            raise RuntimeError("LLM error")

        self.mock_llm.complete = raise_error
        detector = AmbiguityDetector(llm=self.mock_llm, use_llm_detection=True)

        # Should not raise, just log warning
        result = await detector.analyze("Query", resolved_entities=None)

        assert isinstance(result, AmbiguityAnalysis)


class TestAmbiguityDetectorCombinedAmbiguity:
    """Test suite for queries with multiple ambiguities."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.detector = AmbiguityDetector()

    @pytest.mark.asyncio
    async def test_multiple_ambiguities(self) -> None:
        """Test detection of multiple ambiguity types."""
        result = await self.detector.analyze(
            "How is the company performing recently?",
            resolved_entities=None,
        )

        assert result.is_ambiguous
        # Should have entity, time, and metric ambiguity
        assert len(result.ambiguity_types) >= 2
        assert len(result.clarification_questions) >= 2

    @pytest.mark.asyncio
    async def test_fully_ambiguous_query(self) -> None:
        """Test fully ambiguous query."""
        result = await self.detector.analyze(
            "How is it doing lately?",
            resolved_entities=None,
        )

        assert result.is_ambiguous
        assert result.confidence < 1.0


class TestClarificationQuestionGeneration:
    """Test suite for clarification question generation."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.detector = AmbiguityDetector()

    @pytest.mark.asyncio
    async def test_time_question_has_options(self) -> None:
        """Test that time clarification questions include options."""
        result = await self.detector.analyze(
            "What is Apple's recent EPS?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        time_questions = [q for q in result.clarification_questions if q.category == "time_period"]
        assert len(time_questions) > 0
        assert len(time_questions[0].options) > 0

    @pytest.mark.asyncio
    async def test_metric_question_has_options(self) -> None:
        """Test that metric clarification questions include options."""
        result = await self.detector.analyze(
            "How is Apple performing?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        metric_questions = [q for q in result.clarification_questions if q.category == "metric"]
        assert len(metric_questions) > 0
        assert len(metric_questions[0].options) > 0

    @pytest.mark.asyncio
    async def test_question_categories_set(self) -> None:
        """Test that question categories are properly set."""
        result = await self.detector.analyze(
            "How is the company doing?",
            resolved_entities=None,
        )

        for q in result.clarification_questions:
            assert q.category in ("entity", "time_period", "metric", "llm_detected")
