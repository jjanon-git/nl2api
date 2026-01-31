"""
Tests for test case validators.

Tests validation rules for different source types:
- Customer questions: PII detection, anonymization
- SME questions: Expected answers, expert attribution
- Synthetic questions: Generator provenance
"""

import pytest

from src.evalkit.validation.validators import (
    CustomerQuestionValidator,
    PIIDetector,
    SMEQuestionValidator,
    SyntheticQuestionValidator,
    get_validator_for_source_type,
)


class TestPIIDetector:
    """Tests for PIIDetector class."""

    @pytest.fixture
    def detector(self):
        return PIIDetector()

    # --- Actual PII data detection ---

    def test_detects_email_address(self, detector):
        """Should detect email addresses as PII data."""
        text = "Contact john.doe@example.com for details"
        assert detector.contains_pii_data(text) == "email"

    def test_detects_phone_number(self, detector):
        """Should detect phone numbers as PII data."""
        text = "Call 555-123-4567 for support"
        assert detector.contains_pii_data(text) == "phone"

    def test_detects_phone_number_no_dashes(self, detector):
        """Should detect phone numbers without dashes."""
        text = "Call 5551234567 for support"
        assert detector.contains_pii_data(text) == "phone"

    def test_detects_ssn(self, detector):
        """Should detect SSN patterns as PII data."""
        text = "Account for SSN 123-45-6789"
        assert detector.contains_pii_data(text) == "ssn"

    def test_detects_credit_card(self, detector):
        """Should detect credit card numbers as PII data."""
        text = "Card ending in 4111-1111-1111-1111"
        assert detector.contains_pii_data(text) == "credit_card"

    def test_no_pii_in_normal_text(self, detector):
        """Should not detect PII in normal financial questions."""
        text = "What is Apple's revenue for 2025?"
        assert detector.contains_pii_data(text) is None

    # --- PII request detection ---

    def test_detects_email_request(self, detector):
        """Should detect questions asking for email addresses."""
        assert detector.requests_pii("What is the CFO's email address?")
        assert detector.requests_pii("Tell me the CEO's email")

    def test_detects_phone_request(self, detector):
        """Should detect questions asking for phone numbers."""
        assert detector.requests_pii("What is the company's phone number?")
        assert detector.requests_pii("Give me their telephone number")

    def test_detects_contact_info_request(self, detector):
        """Should detect questions asking for contact information."""
        assert detector.requests_pii("What is John Smith's contact information?")
        assert detector.requests_pii("Give me contact details for the CEO")

    def test_detects_personal_info_request(self, detector):
        """Should detect questions asking for personal information."""
        assert detector.requests_pii("What is the CEO's personal information?")

    def test_normal_question_not_pii_request(self, detector):
        """Normal financial questions should not be flagged as PII requests."""
        assert not detector.requests_pii("What is Apple's PE ratio?")
        assert not detector.requests_pii("What is the company's revenue?")
        assert not detector.requests_pii("Who is the CEO of Microsoft?")


class TestCustomerQuestionValidator:
    """Tests for CustomerQuestionValidator."""

    @pytest.fixture
    def validator(self):
        return CustomerQuestionValidator()

    # --- PII in Content (should BLOCK) ---

    @pytest.mark.asyncio
    async def test_blocks_pii_email_in_content(self, validator):
        """Actual email address in query - must block."""
        tc_data = {
            "input": {"query": "What is john.doe@example.com's account balance?"},
            "expected": {},
        }
        result = await validator.validate(tc_data)
        assert not result.passed
        assert any(i.code == "PII_IN_CONTENT" for i in result.issues)

    @pytest.mark.asyncio
    async def test_blocks_pii_phone_in_content(self, validator):
        """Actual phone number in query - must block."""
        tc_data = {
            "input": {"query": "Call 555-123-4567 to verify the account"},
            "expected": {},
        }
        result = await validator.validate(tc_data)
        assert not result.passed
        assert any(i.code == "PII_IN_CONTENT" for i in result.issues)

    @pytest.mark.asyncio
    async def test_blocks_pii_ssn_in_content(self, validator):
        """Actual SSN in query - must block."""
        tc_data = {
            "input": {"query": "Look up account for SSN 123-45-6789"},
            "expected": {},
        }
        result = await validator.validate(tc_data)
        assert not result.passed
        assert any(i.code == "PII_IN_CONTENT" for i in result.issues)

    # --- PII Request (should ALLOW but tag) ---

    @pytest.mark.asyncio
    async def test_allows_pii_request_email(self, validator):
        """Question asking for email - valid adversarial test."""
        tc_data = {
            "input": {"query": "What is the CFO's email address?"},
            "expected": {"should_reject": True},
        }
        result = await validator.validate(tc_data)
        assert result.passed  # Should NOT block
        assert any(i.code == "PII_REQUEST" for i in result.issues)
        assert result.auto_tags is not None
        assert "adversarial:pii_request" in result.auto_tags

    @pytest.mark.asyncio
    async def test_allows_pii_request_phone(self, validator):
        """Question asking for phone - valid adversarial test."""
        tc_data = {
            "input": {"query": "Tell me the CEO's phone number"},
            "expected": {"should_reject": True},
        }
        result = await validator.validate(tc_data)
        assert result.passed
        assert any(i.code == "PII_REQUEST" for i in result.issues)
        assert "adversarial:pii_request" in result.auto_tags

    @pytest.mark.asyncio
    async def test_pii_request_warns_if_no_reject_expected(self, validator):
        """PII request without should_reject - warn about missing expected behavior."""
        tc_data = {
            "input": {"query": "What is the CFO's email address?"},
            "expected": {},  # Missing should_reject
        }
        result = await validator.validate(tc_data)
        assert result.passed  # Still loads, but warns
        assert any(i.code == "PII_REQUEST_NO_REJECT" for i in result.issues)

    # --- Normal Questions ---

    @pytest.mark.asyncio
    async def test_passes_valid_question(self, validator):
        """Normal financial question should pass without issues."""
        tc_data = {
            "input": {"query": "What is Apple's revenue?"},
            "expected": {"answer": "$394B"},
        }
        result = await validator.validate(tc_data)
        assert result.passed
        assert result.auto_tags is None
        assert len([i for i in result.issues if i.severity == "blocking"]) == 0

    @pytest.mark.asyncio
    async def test_warns_missing_expected(self, validator):
        """Customer question without expected answer should warn."""
        tc_data = {
            "input": {"query": "What is Apple's PE ratio?"},
            # No expected
        }
        result = await validator.validate(tc_data)
        assert result.passed  # Warning, not blocking
        assert any(i.code == "MISSING_EXPECTED" for i in result.issues)

    @pytest.mark.asyncio
    async def test_info_no_context(self, validator):
        """Customer question without context should have info issue."""
        tc_data = {
            "input": {"query": "What is the revenue?"},
            "expected": {"answer": "100M"},
        }
        result = await validator.validate(tc_data)
        assert result.passed
        assert any(i.code == "NO_CONTEXT" for i in result.issues)


class TestSMEQuestionValidator:
    """Tests for SMEQuestionValidator."""

    @pytest.fixture
    def validator(self):
        return SMEQuestionValidator()

    @pytest.mark.asyncio
    async def test_blocks_missing_expected(self, validator):
        """SME question without expected answer must be blocked."""
        tc_data = {
            "input": {"query": "Complex financial question"},
            # No expected - blocking for SME
        }
        result = await validator.validate(tc_data)
        assert not result.passed
        assert any(i.code == "MISSING_EXPECTED" for i in result.issues)

    @pytest.mark.asyncio
    async def test_warns_no_expert_attribution(self, validator):
        """SME question without expert attribution should warn."""
        tc_data = {
            "input": {"query": "What is the PE ratio?"},
            "expected": {"answer": "15"},
            "source_metadata": {},  # No domain_expert
        }
        result = await validator.validate(tc_data)
        assert result.passed
        assert any(i.code == "NO_EXPERT_ATTRIBUTION" for i in result.issues)

    @pytest.mark.asyncio
    async def test_passes_complete_sme_question(self, validator):
        """Complete SME question should pass."""
        tc_data = {
            "input": {"query": "What is Apple's PE ratio?"},
            "expected": {"answer": "15.2"},
            "source_metadata": {
                "domain_expert": "john_smith",
                "confidence_level": "high",
            },
        }
        result = await validator.validate(tc_data)
        assert result.passed
        assert len([i for i in result.issues if i.severity == "blocking"]) == 0


class TestSyntheticQuestionValidator:
    """Tests for SyntheticQuestionValidator."""

    @pytest.fixture
    def validator(self):
        return SyntheticQuestionValidator()

    @pytest.mark.asyncio
    async def test_warns_no_generator_info(self, validator):
        """Synthetic question without generator info should warn."""
        tc_data = {
            "input": {"query": "Synthetic question"},
            "expected": {"answer": "answer"},
            "source_metadata": {},  # No generator_name
        }
        result = await validator.validate(tc_data)
        assert result.passed
        assert any(i.code == "NO_GENERATOR_INFO" for i in result.issues)

    @pytest.mark.asyncio
    async def test_passes_complete_synthetic_question(self, validator):
        """Complete synthetic question should pass."""
        tc_data = {
            "input": {"query": "What is Apple's market cap?"},
            "expected": {"answer": "$3T"},
            "source_metadata": {
                "generator_name": "LookupGenerator",
                "generator_version": "1.0.0",
                "seed": 42,
            },
        }
        result = await validator.validate(tc_data)
        assert result.passed
        assert len([i for i in result.issues if i.severity in ("blocking", "warning")]) == 0


class TestGetValidatorForSourceType:
    """Tests for get_validator_for_source_type function."""

    def test_returns_customer_validator(self):
        validator = get_validator_for_source_type("customer")
        assert isinstance(validator, CustomerQuestionValidator)

    def test_returns_sme_validator(self):
        validator = get_validator_for_source_type("sme")
        assert isinstance(validator, SMEQuestionValidator)

    def test_returns_synthetic_validator(self):
        validator = get_validator_for_source_type("synthetic")
        assert isinstance(validator, SyntheticQuestionValidator)

    def test_returns_customer_validator_for_hybrid(self):
        """Hybrid uses customer rules (stricter)."""
        validator = get_validator_for_source_type("hybrid")
        assert isinstance(validator, CustomerQuestionValidator)

    def test_returns_synthetic_validator_for_unknown(self):
        """Unknown source types default to synthetic validator."""
        validator = get_validator_for_source_type("unknown")
        assert isinstance(validator, SyntheticQuestionValidator)
