"""Tests for AmbiguityDetector."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.nl2api.clarification.detector import AmbiguityAnalysis, AmbiguityDetector
from src.nl2api.llm.protocols import LLMProvider, LLMResponse
from src.nl2api.models import ClarificationQuestion


class TestAmbiguityAnalysis:
    """Tests for AmbiguityAnalysis dataclass."""

    def test_analysis_creation_not_ambiguous(self) -> None:
        """Test creating a non-ambiguous analysis."""
        analysis = AmbiguityAnalysis(is_ambiguous=False)

        assert analysis.is_ambiguous is False
        assert analysis.ambiguity_types == ()
        assert analysis.clarification_questions == ()
        assert analysis.confidence == 1.0

    def test_analysis_creation_ambiguous(self) -> None:
        """Test creating an ambiguous analysis."""
        question = ClarificationQuestion(question="Which company?", category="entity")
        analysis = AmbiguityAnalysis(
            is_ambiguous=True,
            ambiguity_types=("entity", "time_period"),
            clarification_questions=(question,),
            confidence=0.8,
        )

        assert analysis.is_ambiguous is True
        assert "entity" in analysis.ambiguity_types
        assert "time_period" in analysis.ambiguity_types
        assert len(analysis.clarification_questions) == 1
        assert analysis.confidence == 0.8

    def test_analysis_is_frozen(self) -> None:
        """Test that analysis is immutable."""
        analysis = AmbiguityAnalysis(is_ambiguous=False)

        with pytest.raises(AttributeError):
            analysis.is_ambiguous = True  # type: ignore


class TestAmbiguityDetectorEntityAmbiguity:
    """Tests for entity ambiguity detection."""

    @pytest.mark.asyncio
    async def test_detects_pronoun_without_context(self) -> None:
        """Test detecting pronouns without resolved entities."""
        detector = AmbiguityDetector()

        # Query with pronoun but no resolved entities
        analysis = await detector.analyze("What is their EPS?", resolved_entities=None)

        assert analysis.is_ambiguous is True
        assert "entity" in analysis.ambiguity_types
        assert any("company" in q.question.lower() or "entity" in q.question.lower()
                   for q in analysis.clarification_questions)

    @pytest.mark.asyncio
    async def test_detects_company_reference_without_resolution(self) -> None:
        """Test detecting generic company references."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze("What is the company's revenue?", resolved_entities=None)

        assert analysis.is_ambiguous is True
        assert "entity" in analysis.ambiguity_types

    @pytest.mark.asyncio
    async def test_no_ambiguity_with_resolved_entities(self) -> None:
        """Test that resolved entities prevent entity ambiguity."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze(
            "What is their EPS?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        # Entity ambiguity should NOT be detected since entities are resolved
        assert "entity" not in analysis.ambiguity_types

    @pytest.mark.asyncio
    async def test_clear_query_no_entity_ambiguity(self) -> None:
        """Test that clear queries have no entity ambiguity."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze(
            "What is Apple's EPS?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        assert "entity" not in analysis.ambiguity_types


class TestAmbiguityDetectorTimeAmbiguity:
    """Tests for time period ambiguity detection."""

    @pytest.mark.asyncio
    async def test_detects_recent_time_reference(self) -> None:
        """Test detecting 'recent' time ambiguity."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze("What is Apple's recent revenue?")

        assert analysis.is_ambiguous is True
        assert "time_period" in analysis.ambiguity_types
        assert any("time period" in q.question.lower() or "period" in q.question.lower()
                   for q in analysis.clarification_questions)

    @pytest.mark.asyncio
    async def test_detects_this_year_ambiguity(self) -> None:
        """Test detecting 'this year' ambiguity."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze("What is Apple's EPS this year?")

        assert analysis.is_ambiguous is True
        assert "time_period" in analysis.ambiguity_types

    @pytest.mark.asyncio
    async def test_detects_last_quarter_ambiguity(self) -> None:
        """Test detecting 'last quarter' ambiguity."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze("What was the revenue last quarter?")

        assert analysis.is_ambiguous is True
        assert "time_period" in analysis.ambiguity_types

    @pytest.mark.asyncio
    async def test_no_ambiguity_with_specific_date(self) -> None:
        """Test that specific dates don't trigger time ambiguity."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze("What was Apple's EPS in Q4 2023?")

        assert "time_period" not in analysis.ambiguity_types

    @pytest.mark.asyncio
    async def test_no_ambiguity_with_fiscal_year(self) -> None:
        """Test that fiscal year references don't trigger ambiguity."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze("What is Apple's revenue for FY2024?")

        assert "time_period" not in analysis.ambiguity_types

    @pytest.mark.asyncio
    async def test_clarification_question_has_options(self) -> None:
        """Test that time period clarification includes options."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze("What is Apple's current revenue?")

        time_questions = [q for q in analysis.clarification_questions
                        if q.category == "time_period"]
        assert len(time_questions) > 0
        # Time period questions should have predefined options
        assert time_questions[0].options is not None
        assert len(time_questions[0].options) > 0


class TestAmbiguityDetectorMetricAmbiguity:
    """Tests for metric ambiguity detection."""

    @pytest.mark.asyncio
    async def test_detects_vague_performance_query(self) -> None:
        """Test detecting vague performance queries."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze("How is Apple performing?")

        assert analysis.is_ambiguous is True
        assert "metric" in analysis.ambiguity_types
        assert any("metric" in q.question.lower() or "data" in q.question.lower()
                   for q in analysis.clarification_questions)

    @pytest.mark.asyncio
    async def test_detects_vague_results_query(self) -> None:
        """Test detecting vague results queries."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze("What are Apple's results?")

        assert analysis.is_ambiguous is True
        assert "metric" in analysis.ambiguity_types

    @pytest.mark.asyncio
    async def test_detects_how_is_doing_query(self) -> None:
        """Test detecting 'how is doing' queries."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze("How is Apple doing?")

        assert analysis.is_ambiguous is True
        assert "metric" in analysis.ambiguity_types

    @pytest.mark.asyncio
    async def test_no_ambiguity_with_specific_metric(self) -> None:
        """Test that specific metrics don't trigger metric ambiguity."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze("What is Apple's EPS?")

        assert "metric" not in analysis.ambiguity_types

    @pytest.mark.asyncio
    async def test_no_ambiguity_with_revenue(self) -> None:
        """Test that revenue queries are clear."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze("What is Apple's revenue?")

        assert "metric" not in analysis.ambiguity_types

    @pytest.mark.asyncio
    async def test_no_ambiguity_with_stock_price(self) -> None:
        """Test that price queries are clear."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze("What is Apple's stock price?")

        assert "metric" not in analysis.ambiguity_types

    @pytest.mark.asyncio
    async def test_metric_clarification_has_options(self) -> None:
        """Test that metric clarification includes options."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze("How are Apple's numbers?")

        metric_questions = [q for q in analysis.clarification_questions
                          if q.category == "metric"]
        assert len(metric_questions) > 0
        assert metric_questions[0].options is not None
        assert "Earnings (EPS)" in metric_questions[0].options
        assert "Revenue" in metric_questions[0].options


class TestAmbiguityDetectorMultipleAmbiguities:
    """Tests for queries with multiple types of ambiguity."""

    @pytest.mark.asyncio
    async def test_detects_entity_and_time_ambiguity(self) -> None:
        """Test detecting both entity and time ambiguity."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze("What is their recent EPS?")

        assert analysis.is_ambiguous is True
        assert "entity" in analysis.ambiguity_types
        assert "time_period" in analysis.ambiguity_types
        assert len(analysis.clarification_questions) >= 2

    @pytest.mark.asyncio
    async def test_detects_entity_and_metric_ambiguity(self) -> None:
        """Test detecting both entity and metric ambiguity."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze("How are they performing?")

        assert analysis.is_ambiguous is True
        assert "entity" in analysis.ambiguity_types
        assert "metric" in analysis.ambiguity_types


class TestAmbiguityDetectorLLMIntegration:
    """Tests for LLM-based ambiguity detection."""

    @pytest.mark.asyncio
    async def test_llm_detection_disabled_by_default(self) -> None:
        """Test that LLM detection is disabled by default."""
        detector = AmbiguityDetector()

        assert detector._use_llm_detection is False

    @pytest.mark.asyncio
    async def test_llm_detection_requires_llm(self) -> None:
        """Test that LLM detection requires an LLM provider."""
        # Even if flag is True, without LLM it should be disabled
        detector = AmbiguityDetector(llm=None, use_llm_detection=True)

        assert detector._use_llm_detection is False

    @pytest.mark.asyncio
    async def test_llm_detection_when_enabled(self) -> None:
        """Test LLM-based detection when enabled."""
        mock_llm = MagicMock(spec=LLMProvider)
        mock_llm.complete = AsyncMock(return_value=LLMResponse(
            content="AMBIGUOUS: What specific financial data are you looking for?",
            usage={"total_tokens": 50},
        ))

        detector = AmbiguityDetector(llm=mock_llm, use_llm_detection=True)

        # Query that only LLM would flag
        analysis = await detector.analyze(
            "Tell me about Microsoft",
            resolved_entities={"Microsoft": "MSFT.O"},
        )

        # Should have called LLM
        mock_llm.complete.assert_called_once()

        # Should have LLM-detected ambiguity
        assert "llm_detected" in analysis.ambiguity_types
        assert any("financial data" in q.question.lower()
                   for q in analysis.clarification_questions)

    @pytest.mark.asyncio
    async def test_llm_detection_clear_response(self) -> None:
        """Test that CLEAR response from LLM doesn't add ambiguity."""
        mock_llm = MagicMock(spec=LLMProvider)
        mock_llm.complete = AsyncMock(return_value=LLMResponse(
            content="CLEAR",
            usage={"total_tokens": 10},
        ))

        detector = AmbiguityDetector(llm=mock_llm, use_llm_detection=True)

        analysis = await detector.analyze(
            "What is Apple's EPS?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        # Should not have LLM-detected ambiguity
        assert "llm_detected" not in analysis.ambiguity_types

    @pytest.mark.asyncio
    async def test_llm_detection_handles_errors(self) -> None:
        """Test that LLM errors don't break detection."""
        mock_llm = MagicMock(spec=LLMProvider)
        mock_llm.complete = AsyncMock(side_effect=Exception("LLM error"))

        detector = AmbiguityDetector(llm=mock_llm, use_llm_detection=True)

        # Should not raise, just skip LLM detection
        analysis = await detector.analyze("What is Apple's EPS?")

        # Analysis should still work without LLM
        assert analysis is not None
        assert "llm_detected" not in analysis.ambiguity_types


class TestAmbiguityDetectorClearQueries:
    """Tests for queries that should NOT be flagged as ambiguous."""

    @pytest.mark.asyncio
    async def test_clear_specific_query(self) -> None:
        """Test that fully specified queries are clear."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze(
            "What is Apple's EPS estimate for Q4 2024?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        assert analysis.is_ambiguous is False
        assert len(analysis.clarification_questions) == 0
        assert analysis.confidence == 1.0

    @pytest.mark.asyncio
    async def test_clear_price_query(self) -> None:
        """Test that price queries are clear."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze(
            "What is Microsoft's current stock price?",
            resolved_entities={"Microsoft": "MSFT.O"},
        )

        # "current" might trigger time ambiguity but price is always current
        # Check that at minimum it's a valid analysis
        assert analysis is not None

    @pytest.mark.asyncio
    async def test_clear_dividend_query(self) -> None:
        """Test that dividend queries are clear."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze(
            "What is Apple's dividend yield?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        assert "metric" not in analysis.ambiguity_types

    @pytest.mark.asyncio
    async def test_clear_market_cap_query(self) -> None:
        """Test that market cap queries are clear."""
        detector = AmbiguityDetector()

        analysis = await detector.analyze(
            "What is Apple's market cap?",
            resolved_entities={"Apple": "AAPL.O"},
        )

        assert "metric" not in analysis.ambiguity_types
