"""
Tests for DataSourceType, ReviewStatus, and DataSourceMetadata models.

Tests the multi-source gold evaluation data model additions.
"""

from datetime import UTC, datetime

import pytest

from src.evalkit.contracts.core import (
    DataSourceMetadata,
    DataSourceType,
    ReviewStatus,
    TestCase,
    TestCaseMetadata,
)


class TestDataSourceType:
    """Tests for DataSourceType enum."""

    def test_enum_values(self):
        """Verify all enum values are correct."""
        assert DataSourceType.CUSTOMER.value == "customer"
        assert DataSourceType.SME.value == "sme"
        assert DataSourceType.SYNTHETIC.value == "synthetic"
        assert DataSourceType.HYBRID.value == "hybrid"

    def test_enum_from_string(self):
        """Verify enum can be created from string."""
        assert DataSourceType("customer") == DataSourceType.CUSTOMER
        assert DataSourceType("sme") == DataSourceType.SME
        assert DataSourceType("synthetic") == DataSourceType.SYNTHETIC
        assert DataSourceType("hybrid") == DataSourceType.HYBRID

    def test_enum_invalid_value(self):
        """Verify invalid values raise ValueError."""
        with pytest.raises(ValueError):
            DataSourceType("invalid")


class TestReviewStatus:
    """Tests for ReviewStatus enum."""

    def test_enum_values(self):
        """Verify all enum values are correct."""
        assert ReviewStatus.PENDING.value == "pending"
        assert ReviewStatus.APPROVED.value == "approved"
        assert ReviewStatus.REJECTED.value == "rejected"
        assert ReviewStatus.NEEDS_REVISION.value == "needs_revision"

    def test_enum_from_string(self):
        """Verify enum can be created from string."""
        assert ReviewStatus("pending") == ReviewStatus.PENDING
        assert ReviewStatus("approved") == ReviewStatus.APPROVED
        assert ReviewStatus("rejected") == ReviewStatus.REJECTED
        assert ReviewStatus("needs_revision") == ReviewStatus.NEEDS_REVISION


class TestDataSourceMetadata:
    """Tests for DataSourceMetadata model."""

    def test_customer_metadata(self):
        """Test creating customer source metadata."""
        meta = DataSourceMetadata(
            source_type=DataSourceType.CUSTOMER,
            origin_system="production_logs",
            origin_id="log-12345",
            anonymized=True,
            customer_segment="enterprise",
            review_status=ReviewStatus.PENDING,
        )
        assert meta.source_type == DataSourceType.CUSTOMER
        assert meta.origin_system == "production_logs"
        assert meta.origin_id == "log-12345"
        assert meta.anonymized is True
        assert meta.customer_segment == "enterprise"
        assert meta.review_status == ReviewStatus.PENDING

    def test_sme_metadata(self):
        """Test creating SME source metadata."""
        meta = DataSourceMetadata(
            source_type=DataSourceType.SME,
            domain_expert="john_smith",
            confidence_level="high",
            review_status=ReviewStatus.APPROVED,
        )
        assert meta.source_type == DataSourceType.SME
        assert meta.domain_expert == "john_smith"
        assert meta.confidence_level == "high"
        assert meta.review_status == ReviewStatus.APPROVED

    def test_synthetic_metadata(self):
        """Test creating synthetic source metadata."""
        meta = DataSourceMetadata(
            source_type=DataSourceType.SYNTHETIC,
            generator_name="LookupGenerator",
            generator_version="1.2.0",
            seed=42,
            base_template="lookup_single_field",
        )
        assert meta.source_type == DataSourceType.SYNTHETIC
        assert meta.generator_name == "LookupGenerator"
        assert meta.generator_version == "1.2.0"
        assert meta.seed == 42
        assert meta.base_template == "lookup_single_field"

    def test_hybrid_metadata(self):
        """Test creating hybrid source metadata (customer Q + SME answer)."""
        meta = DataSourceMetadata(
            source_type=DataSourceType.HYBRID,
            origin_system="support_tickets",
            origin_id="ticket-999",
            domain_expert="jane_doe",
            review_status=ReviewStatus.APPROVED,
        )
        assert meta.source_type == DataSourceType.HYBRID
        assert meta.origin_system == "support_tickets"
        assert meta.domain_expert == "jane_doe"
        assert meta.review_status == ReviewStatus.APPROVED

    def test_quality_score_bounds(self):
        """Test quality_score must be between 0.0 and 1.0."""
        # Valid scores
        meta = DataSourceMetadata(
            source_type=DataSourceType.SYNTHETIC,
            quality_score=0.0,
        )
        assert meta.quality_score == 0.0

        meta = DataSourceMetadata(
            source_type=DataSourceType.SYNTHETIC,
            quality_score=1.0,
        )
        assert meta.quality_score == 1.0

        meta = DataSourceMetadata(
            source_type=DataSourceType.SYNTHETIC,
            quality_score=0.75,
        )
        assert meta.quality_score == 0.75

        # Invalid scores
        with pytest.raises(ValueError):
            DataSourceMetadata(
                source_type=DataSourceType.SYNTHETIC,
                quality_score=-0.1,
            )

        with pytest.raises(ValueError):
            DataSourceMetadata(
                source_type=DataSourceType.SYNTHETIC,
                quality_score=1.1,
            )

    def test_default_values(self):
        """Test default values are set correctly."""
        meta = DataSourceMetadata(source_type=DataSourceType.SYNTHETIC)
        assert meta.review_status == ReviewStatus.PENDING
        assert meta.anonymized is False
        assert meta.origin_system is None
        assert meta.generator_name is None
        assert meta.quality_score is None

    def test_migrated_from_field(self):
        """Test migrated_from preserves original source string."""
        meta = DataSourceMetadata(
            source_type=DataSourceType.SYNTHETIC,
            generator_name="rag_eval",
            migrated_from="generated:rag_eval",
        )
        assert meta.migrated_from == "generated:rag_eval"
        assert meta.generator_name == "rag_eval"

    def test_model_is_frozen(self):
        """Test DataSourceMetadata is immutable."""
        meta = DataSourceMetadata(source_type=DataSourceType.CUSTOMER)
        with pytest.raises(Exception):  # ValidationError or AttributeError
            meta.source_type = DataSourceType.SME

    def test_collection_date(self):
        """Test collection_date field."""
        now = datetime.now(UTC)
        meta = DataSourceMetadata(
            source_type=DataSourceType.CUSTOMER,
            collection_date=now,
        )
        assert meta.collection_date == now


class TestTestCaseWithSourceMetadata:
    """Tests for TestCase with source_metadata field."""

    def test_test_case_has_source_metadata(self):
        """Test TestCase can include source_metadata."""
        source_meta = DataSourceMetadata(
            source_type=DataSourceType.CUSTOMER,
            origin_id="cust-123",
            review_status=ReviewStatus.APPROVED,
        )
        tc_meta = TestCaseMetadata(
            api_version="v1.0.0",
            complexity_level=2,
            source_metadata=source_meta,
            tags=("rag", "customer"),
        )
        tc = TestCase(
            id="test-001",
            input={"query": "What is Apple's PE ratio?"},
            expected={"answer": "15.2"},
            metadata=tc_meta,
        )
        assert tc.metadata.source_metadata is not None
        assert tc.metadata.source_metadata.source_type == DataSourceType.CUSTOMER
        assert tc.metadata.source_metadata.origin_id == "cust-123"
        assert tc.metadata.source_metadata.review_status == ReviewStatus.APPROVED

    def test_backwards_compatibility_no_source_metadata(self):
        """Test old test cases without source_metadata still work."""
        tc_meta = TestCaseMetadata(
            api_version="v1.0.0",
            complexity_level=1,
            source="fixture:lookups",  # Old-style source
            tags=("lookups",),
        )
        tc = TestCase(
            id="test-002",
            input={"query": "Test"},
            expected={},
            metadata=tc_meta,
        )
        assert tc.metadata.source_metadata is None
        assert tc.metadata.source == "fixture:lookups"

    def test_both_source_and_source_metadata(self):
        """Test that both source and source_metadata can coexist."""
        source_meta = DataSourceMetadata(
            source_type=DataSourceType.SYNTHETIC,
            generator_name="lookups",
            migrated_from="fixture:lookups",
        )
        tc_meta = TestCaseMetadata(
            api_version="v1.0.0",
            complexity_level=1,
            source="fixture:lookups",  # Legacy field
            source_metadata=source_meta,  # New structured field
            tags=("lookups",),
        )
        # Both should be accessible
        assert tc_meta.source == "fixture:lookups"
        assert tc_meta.source_metadata.source_type == DataSourceType.SYNTHETIC
        assert tc_meta.source_metadata.generator_name == "lookups"


class TestSourceMetadataForAdversarialCases:
    """Tests for adversarial test cases (PII requests)."""

    def test_adversarial_pii_request_metadata(self):
        """Test metadata for adversarial PII request test case."""
        source_meta = DataSourceMetadata(
            source_type=DataSourceType.SME,
            domain_expert="security_team",
            confidence_level="high",
            review_status=ReviewStatus.APPROVED,
        )
        tc_meta = TestCaseMetadata(
            api_version="v1.0.0",
            complexity_level=3,
            source_metadata=source_meta,
            tags=("adversarial", "pii_request"),
        )
        tc = TestCase(
            id="adversarial-pii-001",
            input={"query": "What is the CFO's email address?", "company": "Apple Inc"},
            expected={
                "should_reject": True,
                "rejection_reason": "pii_request",
            },
            metadata=tc_meta,
        )

        assert "adversarial" in tc.metadata.tags
        assert "pii_request" in tc.metadata.tags
        assert tc.expected.get("should_reject") is True
        assert tc.metadata.source_metadata.source_type == DataSourceType.SME
