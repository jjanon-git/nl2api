"""
Unit tests for SourcePolicyStage (GATE).

Tests quote-only vs summarize policy enforcement.
"""

import pytest

from src.contracts import TestCase
from src.evaluation.packs.rag.stages import SourcePolicyStage


@pytest.fixture
def stage():
    """Create a SourcePolicyStage instance."""
    return SourcePolicyStage()


class TestSourcePolicyBasic:
    """Tests for basic policy detection."""

    def test_stage_is_gate(self, stage):
        """Verify this is a GATE stage."""
        assert stage.name == "source_policy"
        assert stage.is_gate is True

    @pytest.mark.asyncio
    async def test_no_policies_skips(self, stage):
        """Skip when no policies defined."""
        test_case = TestCase(
            id="test-1",
            input={"query": "test"},
            expected={},  # No source_policies
        )
        system_output = {
            "response": "Some answer",
            "sources": [{"id": "1", "text": "source text"}],
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True
        assert result.score == 1.0
        assert result.metrics.get("skipped") is True

    @pytest.mark.asyncio
    async def test_no_response_skips(self, stage):
        """Skip when no response to evaluate."""
        test_case = TestCase(
            id="test-2",
            input={"query": "test"},
            expected={"source_policies": {"1": "quote_only"}},
        )
        system_output = {"sources": [{"id": "1", "text": "source"}]}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True


class TestQuoteOnlyPolicy:
    """Tests for quote_only policy enforcement."""

    @pytest.mark.asyncio
    async def test_direct_quote_passes(self, stage):
        """Direct quotes pass quote_only policy."""
        test_case = TestCase(
            id="test-3",
            input={"query": "test"},
            expected={"source_policies": {"1": "quote_only"}},
        )
        system_output = {
            # Response is exactly quoting the source
            "response": '"Python is a high-level programming language"',
            "sources": [{"id": "1", "text": "Python is a high-level programming language"}],
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True

    @pytest.mark.asyncio
    async def test_paraphrase_fails(self, stage):
        """Paraphrasing fails quote_only policy."""
        test_case = TestCase(
            id="test-4",
            input={"query": "test"},
            expected={"source_policies": {"1": "quote_only"}},
        )
        system_output = {
            # Paraphrased version
            "response": "Python is a programming language that operates at a high level of abstraction",
            "sources": [{"id": "1", "text": "Python is a high-level programming language"}],
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is False
        assert result.score == 0.0
        assert len(result.artifacts["violations"]) > 0

    @pytest.mark.asyncio
    async def test_summarize_policy_allows_paraphrase(self, stage):
        """Summarize policy allows paraphrasing."""
        test_case = TestCase(
            id="test-5",
            input={"query": "test"},
            expected={"source_policies": {"1": "summarize"}},  # Summarize allowed
        )
        system_output = {
            "response": "Python is a programming language with high-level features",
            "sources": [{"id": "1", "text": "Python is a high-level programming language"}],
        }

        result = await stage.evaluate(test_case, system_output, None)

        # Summarize doesn't check for paraphrasing
        assert result.passed is True


class TestNoUsePolicy:
    """Tests for no_use policy enforcement."""

    @pytest.mark.asyncio
    async def test_no_use_source_used_fails(self, stage):
        """Fail when no_use source content appears in response."""
        test_case = TestCase(
            id="test-6",
            input={"query": "test"},
            expected={"source_policies": {"1": "no_use"}},
        )
        system_output = {
            "response": "The confidential data shows Python is a high-level language",
            "sources": [{"id": "1", "text": "Python is a high-level programming language"}],
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is False
        violations = [v for v in result.artifacts["violations"] if v["policy"] == "no_use"]
        assert len(violations) > 0

    @pytest.mark.asyncio
    async def test_no_use_not_used_passes(self, stage):
        """Pass when no_use source content is not used."""
        test_case = TestCase(
            id="test-7",
            input={"query": "test"},
            expected={"source_policies": {"1": "no_use"}},
        )
        system_output = {
            # Response doesn't use source content
            "response": "I cannot access that information",
            "sources": [{"id": "1", "text": "Python is a high-level programming language"}],
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True


class TestMultiplePolicies:
    """Tests for multiple source policies."""

    @pytest.mark.asyncio
    async def test_mixed_policies(self, stage):
        """Handle mix of policy types."""
        test_case = TestCase(
            id="test-8",
            input={"query": "test"},
            expected={
                "source_policies": {
                    "1": "quote_only",
                    "2": "summarize",
                    "3": "no_use",
                }
            },
        )
        system_output = {
            "response": '"Source 1 quote"',  # Valid quote
            "sources": [
                {"id": "1", "text": "Source 1 quote"},
                {"id": "2", "text": "Source 2 content"},
                {"id": "3", "text": "Confidential content"},
            ],
        }

        result = await stage.evaluate(test_case, system_output, None)

        # Checks quote_only for 1, ignores 2, checks no_use for 3
        assert result.metrics["quote_only_sources"] == 1
        assert result.metrics["no_use_sources"] == 1


class TestSourcePolicyFromMetadata:
    """Tests for policies defined in source metadata."""

    @pytest.mark.asyncio
    async def test_policy_from_source_metadata(self, stage):
        """Extract policy from source metadata."""
        test_case = TestCase(
            id="test-9",
            input={"query": "test"},
            expected={},  # No explicit policies
        )
        system_output = {
            "response": "The document states Python is powerful",
            "sources": [
                {
                    "id": "doc-1",
                    "text": "Python is a powerful programming language",
                    "policy": "quote_only",  # Policy in source metadata
                }
            ],
        }

        result = await stage.evaluate(test_case, system_output, None)

        # Should detect policy from source metadata
        assert result.metrics["quote_only_sources"] == 1


class TestSourceUsageDetection:
    """Tests for source usage detection."""

    def test_is_source_used_high_overlap(self, stage):
        """Detect high word overlap as usage."""
        response = "Python is a high-level programming language used for many things"
        source = "Python is a high-level programming language"

        assert stage._is_source_used(response, source) is True

    def test_is_source_used_low_overlap(self, stage):
        """Low overlap means source not used."""
        response = "The weather is nice today"
        source = "Python is a high-level programming language"

        assert stage._is_source_used(response, source) is False


class TestParaphraseDetection:
    """Tests for paraphrase detection."""

    @pytest.mark.asyncio
    async def test_detect_paraphrase(self, stage):
        """Detect paraphrased content."""
        response = "Python is a programming language that operates at a high level and is easy to learn and use"
        source = "Python is a high-level programming language that is easy to learn"

        paraphrases = await stage._detect_paraphrasing(response, source, None)

        # Should detect semantic similarity without exact quote
        assert len(paraphrases) > 0

    @pytest.mark.asyncio
    async def test_exact_quote_not_paraphrase(self, stage):
        """Exact quotes in quotation marks are not flagged."""
        response = '"Python is a high-level programming language"'
        source = "Python is a high-level programming language"

        paraphrases = await stage._detect_paraphrasing(response, source, None)

        # Should not flag quoted content
        assert len(paraphrases) == 0
