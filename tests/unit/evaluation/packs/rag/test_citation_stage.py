"""
Unit tests for CitationStage.

Tests citation presence, validity, accuracy, and coverage.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.evalkit.contracts import EvalContext, TestCase
from src.evaluation.packs.rag.llm_judge import ClaimVerificationResult
from src.evaluation.packs.rag.stages import CitationStage


@pytest.fixture
def stage():
    """Create a CitationStage instance."""
    return CitationStage()


@pytest.fixture
def mock_llm_judge():
    """Create a mock LLM judge."""
    judge = MagicMock()
    judge.verify_claim = AsyncMock()
    return judge


class TestCitationExtraction:
    """Tests for citation pattern extraction."""

    def test_extract_bracketed_numbers(self, stage):
        """Extract [1], [2] style citations."""
        response = "Paris is the capital of France [1]. It has many museums [2]."
        citations = stage._extract_citations(response)

        assert "1" in citations
        assert "2" in citations

    def test_extract_source_n_format(self, stage):
        """Extract [Source N] style citations - the primary RAG format."""
        response = "The data shows growth [Source 1]. Revenue increased [Source 2]."
        citations = stage._extract_citations(response)

        assert "1" in citations
        assert "2" in citations

    def test_extract_source_parenthetical(self, stage):
        """Extract (Source N) style citations."""
        response = "The data shows growth (Source 1)."
        citations = stage._extract_citations(response)

        assert "1" in citations

    def test_extract_double_bracket(self, stage):
        """Extract [[N]] style citations."""
        response = "This is documented in [[1]] and confirmed [[2]]."
        citations = stage._extract_citations(response)

        assert "1" in citations
        assert "2" in citations

    def test_no_citations(self, stage):
        """No citations returns empty set."""
        response = "This is a statement without any citations."
        citations = stage._extract_citations(response)

        assert len(citations) == 0


class TestCitationValidation:
    """Tests for citation validity checking."""

    def test_validate_direct_match(self, stage):
        """Citations matching source IDs are valid."""
        citations = {"1", "2", "3"}
        sources = {
            "1": {"text": "source 1"},
            "2": {"text": "source 2"},
            "3": {"text": "source 3"},
        }

        valid = stage._validate_citations(citations, sources)

        assert valid == citations

    def test_validate_numeric_index(self, stage):
        """Numeric citations map to source indices."""
        citations = {"1", "2"}
        sources = {
            "0": {"text": "first"},
            "1": {"text": "second"},
        }

        valid = stage._validate_citations(citations, sources)

        assert "1" in valid
        # "2" is within range of 2 sources

    def test_validate_source_n_format(self, stage):
        """Source N format validated against source count."""
        citations = {"Source 1", "Source 2"}
        sources = {
            "1": {"text": "source 1"},
            "2": {"text": "source 2"},
        }

        valid = stage._validate_citations(citations, sources)

        assert "Source 1" in valid
        assert "Source 2" in valid

    def test_validate_no_match(self, stage):
        """Unmatched citations are invalid."""
        citations = {"999", "nonexistent"}
        sources = {"1": {"text": "only source"}}

        valid = stage._validate_citations(citations, sources)

        assert len(valid) == 0


class TestCitationEvaluation:
    """Tests for the evaluate() method."""

    @pytest.mark.asyncio
    async def test_no_citations_required(self, stage):
        """Skip when citations not required."""
        test_case = TestCase(
            id="test-1",
            input={"query": "test"},
            expected={"requires_citations": False},
        )
        system_output = {"response": "Answer without citations"}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True
        assert result.metrics.get("skipped") is True

    @pytest.mark.asyncio
    async def test_no_response_skips(self, stage):
        """Skip when no response to evaluate."""
        test_case = TestCase(
            id="test-2",
            input={"query": "test"},
            expected={},
        )
        system_output = {}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True
        assert result.metrics.get("skipped") is True

    @pytest.mark.asyncio
    async def test_no_citations_fails(self, stage):
        """Fail when no citations and they're required."""
        test_case = TestCase(
            id="test-3",
            input={"query": "What is Python?"},
            expected={},  # citations required by default
        )
        system_output = {
            "response": "Python is a programming language.",
            "sources": [{"id": "1", "text": "Python info"}],
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is False
        assert result.score == 0.0
        assert result.metrics["citation_count"] == 0

    @pytest.mark.asyncio
    async def test_valid_citations_pass(self, stage):
        """Pass when valid citations present."""
        test_case = TestCase(
            id="test-4",
            input={"query": "What is Python?"},
            expected={},
        )
        system_output = {
            "response": "Python is a programming language [1]. It's widely used [2].",
            "sources": [
                {"id": "1", "text": "Python info"},
                {"id": "2", "text": "Usage info"},
            ],
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.metrics["presence"] == 1.0
        assert result.metrics["citation_count"] == 2
        assert result.metrics["validity"] == 1.0

    @pytest.mark.asyncio
    async def test_invalid_citations_penalty(self, stage):
        """Invalid citations reduce validity score."""
        test_case = TestCase(
            id="test-5",
            input={"query": "test"},
            expected={},
        )
        system_output = {
            "response": "Statement [1] and another [99].",  # [99] doesn't exist
            "sources": [{"id": "1", "text": "source 1"}],
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.metrics["citation_count"] == 2
        assert result.metrics["validity"] < 1.0
        assert len(result.metrics["invalid_citations"]) > 0

    @pytest.mark.asyncio
    async def test_coverage_metric(self, stage):
        """Coverage measures citation density."""
        test_case = TestCase(
            id="test-6",
            input={"query": "test"},
            expected={},
        )
        # Multiple sentences (each > 20 chars), one citation
        system_output = {
            "response": "This is the first important statement about the topic. Here is another statement that provides more context. The final statement has the citation [1].",
            "sources": [{"id": "1", "text": "source"}],
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.metrics["coverage"] < 1.0  # Not all statements cited


class TestCitationWithLLM:
    """Tests for LLM-based accuracy evaluation."""

    @pytest.mark.asyncio
    async def test_accuracy_evaluation(self, stage, mock_llm_judge):
        """LLM judge verifies citation accuracy."""
        mock_llm_judge.verify_claim.return_value = ClaimVerificationResult(
            claim="Paris is the capital",
            supported=True,
            evidence="Paris, the capital of France",
        )

        context = EvalContext(config={"llm_judge": mock_llm_judge})
        test_case = TestCase(
            id="test-7",
            input={"query": "What is the capital of France?"},
            expected={},
        )
        system_output = {
            "response": "Paris is the capital [1].",
            "sources": [{"id": "1", "text": "Paris is the capital of France."}],
        }

        result = await stage.evaluate(test_case, system_output, context)

        assert result.metrics["accuracy"] == 1.0


class TestCitationConfiguration:
    """Tests for stage configuration."""

    def test_stage_properties(self, stage):
        """Verify stage properties."""
        assert stage.name == "citation"
        assert stage.is_gate is False

    @pytest.mark.asyncio
    async def test_custom_weights(self):
        """Custom weights affect score calculation."""
        custom_stage = CitationStage(
            presence_weight=0.5,
            validity_weight=0.2,
            accuracy_weight=0.2,
            coverage_weight=0.1,
        )

        test_case = TestCase(
            id="test-8",
            input={"query": "test"},
            expected={},
        )
        system_output = {
            "response": "Statement [1].",
            "sources": [{"id": "1", "text": "source"}],
        }

        result = await custom_stage.evaluate(test_case, system_output, None)

        # With custom weights, presence (0.5) has more impact
        assert result.score > 0.5


class TestStatementEstimation:
    """Tests for statement count estimation."""

    def test_estimate_statements(self, stage):
        """Estimate number of statements in text."""
        # Each sentence must be > 20 chars to be counted
        text = "This is the first longer sentence. Here comes the second statement now! And finally there is the third one?"
        count = stage._estimate_statement_count(text)

        assert count == 3

    def test_short_sentences_excluded(self, stage):
        """Very short sentences not counted."""
        text = "Yes. No. Maybe. This is a substantially longer statement that should be counted."
        count = stage._estimate_statement_count(text)

        assert count == 1  # Only the longer one

    def test_empty_text(self, stage):
        """Empty text returns 0."""
        assert stage._estimate_statement_count("") == 0
