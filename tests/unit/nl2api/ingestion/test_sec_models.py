"""Unit tests for SEC filing data models."""

from datetime import datetime

import pytest

from src.rag.ingestion.sec_filings.models import (
    Filing,
    FilingCheckpoint,
    FilingChunk,
    FilingSection,
    FilingType,
)


class TestFilingType:
    """Tests for FilingType enum."""

    def test_filing_type_values(self):
        """Test filing type enum values."""
        assert FilingType.FORM_10K.value == "10-K"
        assert FilingType.FORM_10Q.value == "10-Q"
        assert FilingType.FORM_10K_A.value == "10-K/A"
        assert FilingType.FORM_10Q_A.value == "10-Q/A"

    def test_filing_type_from_string(self):
        """Test creating filing type from string."""
        assert FilingType("10-K") == FilingType.FORM_10K
        assert FilingType("10-Q") == FilingType.FORM_10Q


class TestFilingSection:
    """Tests for FilingSection enum."""

    def test_key_sections_exist(self):
        """Test that key sections are defined."""
        assert FilingSection.BUSINESS.value == "business"
        assert FilingSection.RISK_FACTORS.value == "risk_factors"
        assert FilingSection.MDA.value == "mda"


class TestFiling:
    """Tests for Filing dataclass."""

    @pytest.fixture
    def sample_filing(self) -> Filing:
        """Create a sample filing for testing."""
        return Filing(
            accession_number="0000320193-23-000077",
            cik="0000320193",
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_type=FilingType.FORM_10K,
            filing_date=datetime(2023, 11, 3),
            period_of_report=datetime(2023, 9, 30),
            primary_document="aapl-20230930.htm",
            filing_url="https://www.sec.gov/Archives/edgar/data/320193/000032019323000077/aapl-20230930.htm",
        )

    def test_filing_creation(self, sample_filing: Filing):
        """Test filing can be created with all fields."""
        assert sample_filing.accession_number == "0000320193-23-000077"
        assert sample_filing.cik == "0000320193"
        assert sample_filing.ticker == "AAPL"
        assert sample_filing.company_name == "Apple Inc."
        assert sample_filing.filing_type == FilingType.FORM_10K

    def test_accession_number_no_dashes(self, sample_filing: Filing):
        """Test accession number without dashes."""
        assert sample_filing.accession_number_no_dashes == "000032019323000077"

    def test_filing_directory_url(self, sample_filing: Filing):
        """Test filing directory URL generation."""
        expected = "https://www.sec.gov/Archives/edgar/data/0000320193/000032019323000077"
        assert sample_filing.filing_directory_url == expected

    def test_primary_document_url(self, sample_filing: Filing):
        """Test primary document URL generation."""
        expected = "https://www.sec.gov/Archives/edgar/data/0000320193/000032019323000077/aapl-20230930.htm"
        assert sample_filing.primary_document_url == expected

    def test_filing_to_dict(self, sample_filing: Filing):
        """Test filing serialization to dict."""
        d = sample_filing.to_dict()
        assert d["accession_number"] == "0000320193-23-000077"
        assert d["ticker"] == "AAPL"
        assert d["filing_type"] == "10-K"
        assert "filing_date" in d

    def test_filing_from_dict(self, sample_filing: Filing):
        """Test filing deserialization from dict."""
        d = sample_filing.to_dict()
        restored = Filing.from_dict(d)
        assert restored.accession_number == sample_filing.accession_number
        assert restored.ticker == sample_filing.ticker
        assert restored.filing_type == sample_filing.filing_type

    def test_filing_is_frozen(self, sample_filing: Filing):
        """Test that filing is immutable."""
        with pytest.raises(AttributeError):
            sample_filing.ticker = "MSFT"  # type: ignore


class TestFilingChunk:
    """Tests for FilingChunk dataclass."""

    @pytest.fixture
    def sample_chunk(self) -> FilingChunk:
        """Create a sample chunk for testing."""
        return FilingChunk(
            chunk_id="0000320193-23-000077_mda_0",
            filing_accession="0000320193-23-000077",
            section="mda",
            chunk_index=0,
            content="This is a sample chunk of text from the MD&A section.",
            char_start=0,
            char_end=56,
            metadata={
                "company_name": "Apple Inc.",
                "ticker": "AAPL",
            },
        )

    def test_chunk_creation(self, sample_chunk: FilingChunk):
        """Test chunk can be created with all fields."""
        assert sample_chunk.chunk_id == "0000320193-23-000077_mda_0"
        assert sample_chunk.section == "mda"
        assert sample_chunk.chunk_index == 0

    def test_chunk_content_length(self, sample_chunk: FilingChunk):
        """Test content length property."""
        assert sample_chunk.content_length == 53  # len("This is a sample chunk...")

    def test_chunk_to_dict(self, sample_chunk: FilingChunk):
        """Test chunk serialization."""
        d = sample_chunk.to_dict()
        assert d["chunk_id"] == "0000320193-23-000077_mda_0"
        assert d["section"] == "mda"
        assert d["metadata"]["ticker"] == "AAPL"


class TestFilingCheckpoint:
    """Tests for FilingCheckpoint dataclass."""

    @pytest.fixture
    def sample_checkpoint(self) -> FilingCheckpoint:
        """Create a sample checkpoint for testing."""
        return FilingCheckpoint(
            started_at=datetime(2024, 1, 15, 10, 0, 0),
            total_companies=500,
            companies_processed=10,
            filings_downloaded=45,
            filings_parsed=45,
            chunks_indexed=3500,
            state="downloading",
        )

    def test_checkpoint_creation(self, sample_checkpoint: FilingCheckpoint):
        """Test checkpoint can be created."""
        assert sample_checkpoint.total_companies == 500
        assert sample_checkpoint.companies_processed == 10

    def test_progress_percent(self, sample_checkpoint: FilingCheckpoint):
        """Test progress percentage calculation."""
        assert sample_checkpoint.progress_percent == 2.0  # 10/500 * 100

    def test_is_resumable(self, sample_checkpoint: FilingCheckpoint):
        """Test resumability check."""
        assert sample_checkpoint.is_resumable is True

        sample_checkpoint.state = "complete"
        assert sample_checkpoint.is_resumable is False

        sample_checkpoint.state = "failed"
        assert sample_checkpoint.is_resumable is False

    def test_mark_downloading(self, sample_checkpoint: FilingCheckpoint):
        """Test marking state as downloading."""
        sample_checkpoint.mark_downloading("0000789019", 11)
        assert sample_checkpoint.state == "downloading"
        assert sample_checkpoint.current_company_cik == "0000789019"
        assert sample_checkpoint.current_company_index == 11

    def test_update_filing_progress(self, sample_checkpoint: FilingCheckpoint):
        """Test updating filing progress."""
        sample_checkpoint.update_filing_progress(
            filings_downloaded=5,
            filings_parsed=5,
            chunks_indexed=400,
            errors=1,
        )
        assert sample_checkpoint.filings_downloaded == 50
        assert sample_checkpoint.filings_parsed == 50
        assert sample_checkpoint.chunks_indexed == 3900
        assert sample_checkpoint.error_count == 1

    def test_checkpoint_to_dict(self, sample_checkpoint: FilingCheckpoint):
        """Test checkpoint serialization."""
        d = sample_checkpoint.to_dict()
        assert d["source"] == "sec_edgar"
        assert d["total_companies"] == 500
        assert "started_at" in d

    def test_checkpoint_from_dict(self, sample_checkpoint: FilingCheckpoint):
        """Test checkpoint deserialization."""
        d = sample_checkpoint.to_dict()
        restored = FilingCheckpoint.from_dict(d)
        assert restored.total_companies == sample_checkpoint.total_companies
        assert restored.companies_processed == sample_checkpoint.companies_processed
        assert restored.state == sample_checkpoint.state
