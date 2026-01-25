"""
Unit tests for PolicyComplianceStage (GATE).

Tests content policy violation detection.
"""

import pytest

from src.evalkit.contracts import TestCase
from src.evaluation.packs.rag.stages import PolicyComplianceStage


@pytest.fixture
def stage():
    """Create a PolicyComplianceStage instance."""
    return PolicyComplianceStage()


class TestPolicyComplianceBasic:
    """Tests for basic policy detection."""

    def test_stage_is_gate(self, stage):
        """Verify this is a GATE stage."""
        assert stage.name == "policy_compliance"
        assert stage.is_gate is True

    @pytest.mark.asyncio
    async def test_no_response_skips(self, stage):
        """Skip when no response to evaluate."""
        test_case = TestCase(
            id="test-1",
            input={"query": "test"},
            expected={},
        )
        system_output = {}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True
        assert result.metrics.get("skipped") is True

    @pytest.mark.asyncio
    async def test_clean_response_passes(self, stage):
        """Clean response with no violations passes."""
        test_case = TestCase(
            id="test-2",
            input={"query": "What is Python?"},
            expected={},
        )
        system_output = {
            "response": "Python is a versatile programming language used for web development, data science, and automation."
        }

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is True
        assert result.score == 1.0
        assert result.metrics["violation_count"] == 0


class TestPIIDetection:
    """Tests for PII (Personally Identifiable Information) detection."""

    @pytest.mark.asyncio
    async def test_detect_ssn(self, stage):
        """Detect Social Security Number patterns."""
        test_case = TestCase(
            id="test-3",
            input={"query": "test"},
            expected={},
        )
        system_output = {"response": "The customer's SSN is 123-45-6789."}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is False
        assert result.metrics["pii_detected"] is True
        assert any(v["policy"] == "pii_ssn" for v in result.artifacts["violations"])

    @pytest.mark.asyncio
    async def test_detect_email(self, stage):
        """Detect email address patterns."""
        test_case = TestCase(
            id="test-4",
            input={"query": "test"},
            expected={},
        )
        system_output = {"response": "Contact john.doe@example.com for more information."}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is False
        assert result.metrics["pii_detected"] is True

    @pytest.mark.asyncio
    async def test_detect_phone_number(self, stage):
        """Detect phone number patterns."""
        test_case = TestCase(
            id="test-5",
            input={"query": "test"},
            expected={},
        )
        system_output = {"response": "Call us at 555-123-4567 for support."}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is False
        assert any(v["policy"] == "pii_phone" for v in result.artifacts["violations"])

    @pytest.mark.asyncio
    async def test_detect_credit_card(self, stage):
        """Detect credit card number patterns."""
        test_case = TestCase(
            id="test-6",
            input={"query": "test"},
            expected={},
        )
        system_output = {"response": "Card number: 1234567890123456"}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is False
        assert any(v["policy"] == "pii_credit_card" for v in result.artifacts["violations"])

    @pytest.mark.asyncio
    async def test_multiple_pii_types(self, stage):
        """Detect multiple PII types in same response."""
        test_case = TestCase(
            id="test-7",
            input={"query": "test"},
            expected={},
        )
        system_output = {"response": "Contact john@example.com or call 555-123-4567."}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.metrics["violation_count"] >= 2


class TestProhibitedContent:
    """Tests for prohibited content detection."""

    @pytest.mark.asyncio
    async def test_detect_violence(self, stage):
        """Detect violent content patterns."""
        test_case = TestCase(
            id="test-8",
            input={"query": "test"},
            expected={},
        )
        system_output = {"response": "The instructions show how to kill him silently."}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is False
        assert result.metrics["critical_count"] > 0

    @pytest.mark.asyncio
    async def test_detect_security_threats(self, stage):
        """Detect security threat content."""
        test_case = TestCase(
            id="test-9",
            input={"query": "test"},
            expected={},
        )
        system_output = {"response": "Here's how to hack the database and breach the system."}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is False


class TestCustomPolicies:
    """Tests for custom policy rules."""

    @pytest.mark.asyncio
    async def test_custom_pattern_policy(self, stage):
        """Custom pattern-based policy."""
        test_case = TestCase(
            id="test-10",
            input={"query": "test"},
            expected={
                "policies": [
                    {
                        "name": "no_competitor",
                        "pattern": r"\b(competitor|rival)\s+product\b",
                        "severity": "high",
                        "description": "Cannot mention competitor products",
                    }
                ]
            },
        )
        system_output = {"response": "Our competitor product has similar features."}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is False
        assert any(v["policy"] == "no_competitor" for v in result.artifacts["violations"])

    @pytest.mark.asyncio
    async def test_must_contain_policy(self, stage):
        """Policy requiring certain content."""
        test_case = TestCase(
            id="test-11",
            input={"query": "test"},
            expected={
                "policies": [
                    {
                        "name": "disclaimer_required",
                        "must_contain": "consult a professional",
                        "severity": "medium",
                    }
                ]
            },
        )
        system_output = {"response": "Here's the financial advice you requested."}

        result = await stage.evaluate(test_case, system_output, None)

        # Should fail because required content is missing
        assert result.metrics["medium_count"] > 0

    @pytest.mark.asyncio
    async def test_must_not_contain_policy(self, stage):
        """Policy prohibiting certain content."""
        test_case = TestCase(
            id="test-12",
            input={"query": "test"},
            expected={
                "policies": [
                    {
                        "name": "no_guarantees",
                        "must_not_contain": "guarantee",
                        "severity": "high",
                    }
                ]
            },
        )
        system_output = {"response": "We guarantee this will work perfectly."}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is False


class TestSeverityHandling:
    """Tests for severity-based pass/fail logic."""

    @pytest.mark.asyncio
    async def test_low_severity_passes(self, stage):
        """Low severity violations don't fail the gate."""
        test_case = TestCase(
            id="test-13",
            input={"query": "test"},
            expected={
                "policies": [
                    {
                        "name": "style_warning",
                        "pattern": r"\bvery\b",
                        "severity": "low",
                    }
                ]
            },
        )
        system_output = {"response": "This is a very interesting topic."}

        result = await stage.evaluate(test_case, system_output, None)

        # Low severity should not fail the gate
        assert result.passed is True
        assert result.metrics["low_count"] > 0

    @pytest.mark.asyncio
    async def test_medium_severity_passes(self, stage):
        """Medium severity violations don't fail the gate."""
        test_case = TestCase(
            id="test-14",
            input={"query": "test"},
            expected={
                "policies": [
                    {
                        "name": "informal_language",
                        "pattern": r"\bawesome\b",
                        "severity": "medium",
                    }
                ]
            },
        )
        system_output = {"response": "Python is an awesome language."}

        result = await stage.evaluate(test_case, system_output, None)

        # Medium severity passes but with warning
        assert result.passed is True
        assert result.metrics["medium_count"] > 0

    @pytest.mark.asyncio
    async def test_high_severity_fails(self, stage):
        """High severity violations fail the gate."""
        test_case = TestCase(
            id="test-15",
            input={"query": "test"},
            expected={
                "policies": [
                    {
                        "name": "critical_info",
                        "pattern": r"\bsecret\s+key\b",
                        "severity": "high",
                    }
                ]
            },
        )
        system_output = {"response": "The secret key is ABC123."}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.passed is False
        assert result.metrics["high_count"] > 0


class TestScoreCalculation:
    """Tests for violation-based score calculation."""

    @pytest.mark.asyncio
    async def test_no_violations_full_score(self, stage):
        """No violations gives score of 1.0."""
        test_case = TestCase(
            id="test-16",
            input={"query": "test"},
            expected={},
        )
        system_output = {"response": "Clean response with no issues."}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.score == 1.0

    @pytest.mark.asyncio
    async def test_violations_reduce_score(self, stage):
        """Violations reduce the score."""
        test_case = TestCase(
            id="test-17",
            input={"query": "test"},
            expected={
                "policies": [
                    {"name": "p1", "pattern": r"\btest\b", "severity": "low"},
                    {"name": "p2", "pattern": r"\bword\b", "severity": "low"},
                ]
            },
        )
        system_output = {"response": "test word in response"}

        result = await stage.evaluate(test_case, system_output, None)

        assert result.score < 1.0
