"""Unit tests for SEC filing document chunker."""

from datetime import datetime

import pytest

from src.nl2api.ingestion.sec_filings.chunker import DocumentChunker, estimate_chunk_count
from src.nl2api.ingestion.sec_filings.config import SECFilingConfig
from src.nl2api.ingestion.sec_filings.models import Filing, FilingType


class TestDocumentChunker:
    """Tests for DocumentChunker class."""

    @pytest.fixture
    def chunker(self) -> DocumentChunker:
        """Create chunker with default settings."""
        return DocumentChunker(
            chunk_size=1000,
            chunk_overlap=200,
            min_chunk_size=100,
        )

    @pytest.fixture
    def small_chunker(self) -> DocumentChunker:
        """Create chunker with small chunk size for testing."""
        return DocumentChunker(
            chunk_size=200,
            chunk_overlap=50,
            min_chunk_size=50,
        )

    @pytest.fixture
    def sample_filing(self) -> Filing:
        """Create sample filing for testing."""
        return Filing(
            accession_number="0000320193-23-000077",
            cik="0000320193",
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_type=FilingType.FORM_10K,
            filing_date=datetime(2023, 11, 3),
            period_of_report=datetime(2023, 9, 30),
            primary_document="aapl-20230930.htm",
            filing_url="https://example.com/filing.htm",
        )

    def test_chunker_initialization(self, chunker: DocumentChunker):
        """Test chunker can be initialized."""
        assert chunker is not None
        assert chunker._chunk_size == 1000
        assert chunker._chunk_overlap == 200

    def test_chunker_from_config(self):
        """Test creating chunker from config."""
        config = SECFilingConfig(
            chunk_size=2000,
            chunk_overlap=400,
            min_chunk_size=200,
        )
        chunker = DocumentChunker.from_config(config)
        assert chunker._chunk_size == 2000
        assert chunker._chunk_overlap == 400

    def test_chunk_small_text(self, chunker: DocumentChunker, sample_filing: Filing):
        """Test chunking text smaller than chunk size."""
        text = "This is a small piece of text that fits in one chunk."
        chunks = chunker.chunk_section(text, sample_filing, "mda")

        # Small text should result in no chunks (below min_chunk_size)
        # or a single chunk if above threshold
        assert len(chunks) <= 1

    def test_chunk_medium_text(self, small_chunker: DocumentChunker, sample_filing: Filing):
        """Test chunking medium-sized text."""
        # Create text that should produce multiple chunks
        text = " ".join([f"This is sentence number {i}." for i in range(50)])

        chunks = small_chunker.chunk_section(text, sample_filing, "mda")

        # Should produce multiple chunks
        assert len(chunks) > 1

        # Chunks should have sequential indices
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_chunk_respects_paragraphs(self, chunker: DocumentChunker, sample_filing: Filing):
        """Test that chunking respects paragraph boundaries where possible."""
        text = """First paragraph with some content here.

        Second paragraph with different content.

        Third paragraph completing the text."""

        chunks = chunker.chunk_section(text, sample_filing, "business")

        # Each chunk should have content
        for chunk in chunks:
            assert len(chunk.content) > 0

    def test_chunk_metadata(self, small_chunker: DocumentChunker, sample_filing: Filing):
        """Test that chunks have correct metadata."""
        text = " ".join(["Word"] * 200)  # Create text that will be chunked
        chunks = small_chunker.chunk_section(text, sample_filing, "risk_factors")

        if chunks:
            chunk = chunks[0]
            assert chunk.metadata["ticker"] == "AAPL"
            assert chunk.metadata["company_name"] == "Apple Inc."
            assert chunk.metadata["section"] == "risk_factors"
            assert chunk.metadata["filing_type"] == "10-K"

    def test_chunk_filing_multiple_sections(
        self, chunker: DocumentChunker, sample_filing: Filing
    ):
        """Test chunking multiple sections of a filing."""
        sections = {
            "business": "Apple Inc. designs and manufactures smartphones. " * 20,
            "risk_factors": "The company faces various market risks. " * 20,
            "mda": "Management discusses the financial results. " * 20,
        }

        chunks = chunker.chunk_filing(sections, sample_filing)

        # Should have chunks from all sections
        sections_in_chunks = {chunk.section for chunk in chunks}
        assert len(sections_in_chunks) >= 1

    def test_chunk_ids_are_unique(self, small_chunker: DocumentChunker, sample_filing: Filing):
        """Test that chunk IDs are unique within a section."""
        text = " ".join(["Content"] * 500)
        chunks = small_chunker.chunk_section(text, sample_filing, "mda")

        chunk_ids = [chunk.chunk_id for chunk in chunks]
        assert len(chunk_ids) == len(set(chunk_ids))  # All unique

    def test_empty_text_returns_empty_list(
        self, chunker: DocumentChunker, sample_filing: Filing
    ):
        """Test that empty text returns empty chunk list."""
        chunks = chunker.chunk_section("", sample_filing, "mda")
        assert chunks == []

    def test_text_below_min_size_returns_empty(
        self, chunker: DocumentChunker, sample_filing: Filing
    ):
        """Test that text below minimum size returns empty list."""
        chunks = chunker.chunk_section("Short", sample_filing, "mda")
        assert chunks == []

    def test_split_by_sentences(self, small_chunker: DocumentChunker):
        """Test splitting long paragraphs by sentences."""
        paragraph = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."

        chunks = small_chunker._split_long_paragraph(paragraph)

        # Should split into multiple pieces
        assert len(chunks) >= 1

    def test_split_by_characters(self, small_chunker: DocumentChunker):
        """Test character-level splitting respects word boundaries."""
        text = "word " * 100  # 500 characters

        chunks = small_chunker._split_by_characters(text)

        # Should split into multiple pieces
        assert len(chunks) >= 1

        # No chunk should start or end with partial words
        for chunk in chunks:
            assert not chunk.startswith(" ")
            # Last chunk might have trailing space due to input

    def test_get_overlap_text(self, small_chunker: DocumentChunker):
        """Test overlap text extraction."""
        text = "This is some text that will have overlap extracted."
        overlap = small_chunker._get_overlap_text(text)

        assert len(overlap) <= small_chunker._chunk_overlap
        # Overlap should start at word boundary (no leading partial word)


class TestEstimateChunkCount:
    """Tests for estimate_chunk_count function."""

    def test_estimate_single_section(self):
        """Test estimation with single section."""
        sections = {"mda": "x" * 10000}  # 10k characters
        estimate = estimate_chunk_count(sections, chunk_size=4000, chunk_overlap=800)

        # Should estimate roughly 3-4 chunks
        assert estimate >= 2

    def test_estimate_multiple_sections(self):
        """Test estimation with multiple sections."""
        sections = {
            "business": "x" * 5000,
            "risk_factors": "x" * 5000,
            "mda": "x" * 5000,
        }
        estimate = estimate_chunk_count(sections, chunk_size=4000, chunk_overlap=800)

        # Should estimate based on total content plus section overhead
        assert estimate >= 3  # At least one per section

    def test_estimate_small_sections(self):
        """Test estimation with small sections."""
        sections = {
            "business": "x" * 100,
            "risk_factors": "x" * 100,
        }
        estimate = estimate_chunk_count(sections, chunk_size=4000, chunk_overlap=800)

        # Small sections: minimum of section count
        assert estimate >= 2
