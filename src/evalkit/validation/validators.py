"""
Test case validators for different data sources.

Implements source-specific validation rules:
- CustomerQuestionValidator: PII detection, anonymization checks
- SMEQuestionValidator: Expert attribution, expected answer requirements
- SyntheticQuestionValidator: Generator provenance, duplicate detection

Key PII distinction:
- PII in content (actual email addresses, phone numbers) -> BLOCK
- PII requests (questions asking for PII) -> ALLOW, tag as adversarial
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Literal


@dataclass
class ValidationIssue:
    """A validation issue found during test case validation."""

    severity: Literal["blocking", "warning", "info"]
    code: str
    message: str


@dataclass
class ValidationResult:
    """Result of validating a test case."""

    passed: bool
    issues: list[ValidationIssue] = field(default_factory=list)
    auto_tags: list[str] | None = None  # Tags to add automatically


class PIIDetector:
    """
    Detects PII in text content.

    Distinguishes between:
    - PII_IN_CONTENT: Actual PII data embedded in text (BLOCK)
    - PII_REQUEST: Questions that request PII (ALLOW, tag as adversarial)
    """

    # Patterns for actual PII data embedded in text (BLOCK these)
    PII_DATA_PATTERNS = [
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "email"),
        (r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b", "phone"),
        (r"\b\d{3}[-]?\d{2}[-]?\d{4}\b", "ssn"),
        (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "credit_card"),
        (r"\b[A-Z]{2}\d{6,10}\b", "account_number"),
    ]

    # Patterns for questions that REQUEST PII (ALLOW but tag)
    PII_REQUEST_PATTERNS = [
        r"\b(email|e-mail)\s+(address|of|for)\b",
        r"\bwhat\s+is\s+.{0,30}(email|phone|address)\b",
        r"\b(phone|telephone)\s+(number|of|for)\b",
        r"\bcontact\s+(info|information|details)\b",
        r"\b(ssn|social\s+security)\b",
        r"\bpersonal\s+(information|details|data)\b",
        r"\btell\s+me\s+.{0,20}(email|phone|contact)\b",
    ]

    def contains_pii_data(self, text: str) -> str | None:
        """
        Check for actual PII data embedded in text.

        Returns PII type if found (email, phone, ssn, etc.), None otherwise.
        """
        for pattern, pii_type in self.PII_DATA_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                return pii_type
        return None

    def requests_pii(self, text: str) -> bool:
        """
        Check if query is requesting PII (valid adversarial test).

        Returns True if the text appears to be asking for personal information.
        """
        text_lower = text.lower()
        for pattern in self.PII_REQUEST_PATTERNS:
            if re.search(pattern, text_lower):
                return True
        return False


class TestCaseValidator(ABC):
    """Base class for test case validators."""

    @abstractmethod
    async def validate(self, tc_data: dict) -> ValidationResult:
        """
        Validate a test case dictionary.

        Args:
            tc_data: Raw test case data (before TestCase model creation)

        Returns:
            ValidationResult with passed status and any issues found
        """
        ...


class CustomerQuestionValidator(TestCaseValidator):
    """
    Validation rules for customer-sourced questions.

    Rules:
    - BLOCK: Actual PII data in query (email addresses, phone numbers, etc.)
    - ALLOW: Questions that request PII (tag as adversarial)
    - WARN: Missing expected answer (requires SME review)
    """

    def __init__(self):
        self._pii_detector = PIIDetector()

    async def validate(self, tc_data: dict) -> ValidationResult:
        """Validate a customer-sourced test case."""
        query = tc_data.get("input", {}).get("query", "")
        issues: list[ValidationIssue] = []
        auto_tags: list[str] = []

        # Check 1: Actual PII data embedded in query (BLOCK)
        pii_found = self._pii_detector.contains_pii_data(query)
        if pii_found:
            issues.append(
                ValidationIssue(
                    severity="blocking",
                    code="PII_IN_CONTENT",
                    message=f"Query contains actual PII ({pii_found}) - must anonymize before loading",
                )
            )

        # Check 2: Query requests PII (ALLOW but tag for adversarial testing)
        if self._pii_detector.requests_pii(query):
            issues.append(
                ValidationIssue(
                    severity="info",
                    code="PII_REQUEST",
                    message="Query requests PII - tagged as adversarial test case",
                )
            )
            auto_tags.append("adversarial:pii_request")

            # Validate expected behavior is "reject" for PII requests
            expected = tc_data.get("expected", {})
            if not expected.get("should_reject"):
                issues.append(
                    ValidationIssue(
                        severity="warning",
                        code="PII_REQUEST_NO_REJECT",
                        message="PII request should have expected.should_reject=true",
                    )
                )

        # Check 3: Requires review if no expected answer
        if not tc_data.get("expected"):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="MISSING_EXPECTED",
                    message="Customer question has no expected answer - requires SME review",
                )
            )

        # Check 4: Check for context (informational)
        if not tc_data.get("input", {}).get("context"):
            issues.append(
                ValidationIssue(
                    severity="info",
                    code="NO_CONTEXT",
                    message="No conversation context provided",
                )
            )

        return ValidationResult(
            passed=not any(i.severity == "blocking" for i in issues),
            issues=issues,
            auto_tags=auto_tags if auto_tags else None,
        )


class SMEQuestionValidator(TestCaseValidator):
    """
    Validation rules for SME-curated questions.

    Rules:
    - BLOCK: Missing expected answer (SME questions must have answers)
    - WARN: Missing expert attribution
    """

    async def validate(self, tc_data: dict) -> ValidationResult:
        """Validate an SME-curated test case."""
        issues: list[ValidationIssue] = []

        # Must have expected answer
        if not tc_data.get("expected"):
            issues.append(
                ValidationIssue(
                    severity="blocking",
                    code="MISSING_EXPECTED",
                    message="SME question must have expected answer",
                )
            )

        # Should have author attribution
        source_meta = tc_data.get("source_metadata", {})
        if not source_meta.get("domain_expert"):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="NO_EXPERT_ATTRIBUTION",
                    message="SME question should have domain_expert attribution",
                )
            )

        return ValidationResult(
            passed=not any(i.severity == "blocking" for i in issues),
            issues=issues,
        )


class SyntheticQuestionValidator(TestCaseValidator):
    """
    Validation rules for synthetic questions.

    Rules:
    - WARN: Missing generator info
    - WARN: Near-duplicate detection (placeholder - requires embedding comparison)
    """

    async def validate(self, tc_data: dict) -> ValidationResult:
        """Validate a synthetic test case."""
        issues: list[ValidationIssue] = []

        # Should have generator info
        source_meta = tc_data.get("source_metadata", {})
        if not source_meta.get("generator_name"):
            issues.append(
                ValidationIssue(
                    severity="warning",
                    code="NO_GENERATOR_INFO",
                    message="Synthetic question should have generator attribution",
                )
            )

        # TODO: Add near-duplicate detection using embeddings
        # This would require async embedding and similarity comparison

        return ValidationResult(
            passed=not any(i.severity == "blocking" for i in issues),
            issues=issues,
        )


def get_validator_for_source_type(source_type: str) -> TestCaseValidator:
    """
    Get the appropriate validator for a source type.

    Args:
        source_type: One of "customer", "sme", "synthetic", "hybrid"

    Returns:
        Appropriate TestCaseValidator instance
    """
    validators = {
        "customer": CustomerQuestionValidator(),
        "sme": SMEQuestionValidator(),
        "synthetic": SyntheticQuestionValidator(),
        "hybrid": CustomerQuestionValidator(),  # Hybrid uses customer rules (stricter)
    }
    return validators.get(source_type, SyntheticQuestionValidator())
