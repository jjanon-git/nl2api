"""Unit tests for SEC filing parser."""

from datetime import datetime

import pytest

from src.rag.ingestion.sec_filings.models import Filing, FilingType
from src.rag.ingestion.sec_filings.parser import (
    FilingParser,
    get_section_metadata,
)


class TestFilingParser:
    """Tests for FilingParser class."""

    @pytest.fixture
    def parser(self) -> FilingParser:
        """Create parser instance."""
        return FilingParser(extract_all_sections=False)

    @pytest.fixture
    def parser_all_sections(self) -> FilingParser:
        """Create parser that extracts all sections."""
        return FilingParser(extract_all_sections=True)

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

    def test_parser_initialization(self, parser: FilingParser):
        """Test parser can be initialized."""
        assert parser is not None

    def test_clean_html_removes_tags(self, parser: FilingParser):
        """Test that HTML tags are removed."""
        html = "<p>This is <b>bold</b> and <i>italic</i> text.</p>"
        clean = parser._clean_html(html)
        assert "<p>" not in clean
        assert "<b>" not in clean
        assert "bold" in clean
        assert "italic" in clean

    def test_clean_html_removes_script_tags(self, parser: FilingParser):
        """Test that script tags and content are removed."""
        html = """
        <p>Text before</p>
        <script>alert('hello');</script>
        <p>Text after</p>
        """
        clean = parser._clean_html(html)
        assert "alert" not in clean
        assert "Text before" in clean
        assert "Text after" in clean

    def test_clean_html_removes_style_tags(self, parser: FilingParser):
        """Test that style tags and content are removed."""
        html = """
        <p>Visible text</p>
        <style>.hidden { display: none; }</style>
        """
        clean = parser._clean_html(html)
        assert ".hidden" not in clean
        assert "display" not in clean
        assert "Visible" in clean

    def test_clean_html_handles_entities(self, parser: FilingParser):
        """Test that HTML entities are decoded."""
        html = "<p>AT&amp;T &ndash; Revenue &gt; $100M</p>"
        clean = parser._clean_html(html)
        assert "&amp;" not in clean
        assert "&ndash;" not in clean
        assert "&gt;" not in clean

    def test_regex_clean_html_fallback(self, parser: FilingParser):
        """Test regex-based HTML cleaning (fallback method)."""
        html = "<p>Simple <b>text</b></p>"
        clean = parser._regex_clean_html(html)
        assert "<p>" not in clean
        assert "<b>" not in clean
        assert "Simple" in clean
        assert "text" in clean

    def test_find_section_positions_10k(self, parser: FilingParser):
        """Test finding section positions in 10-K text."""
        text = """
        Item 1. Business

        Apple Inc. designs, manufactures, and markets smartphones...

        Item 1A. Risk Factors

        The Company is subject to various risks...

        Item 7. Management's Discussion and Analysis

        The following discussion should be read...
        """

        from src.rag.ingestion.sec_filings.parser import (
            KEY_SECTIONS_10K,
            SECTION_PATTERNS_10K,
        )

        positions = parser._find_section_positions(
            text.lower(),
            SECTION_PATTERNS_10K,
            KEY_SECTIONS_10K,
        )

        assert len(positions) >= 2  # Should find at least business, risk_factors

    def test_clean_section_text(self, parser: FilingParser):
        """Test section text cleaning."""
        text = """
        42
        Table of Contents

        Item 7. Management's Discussion and Analysis

        This is the actual content of the section.
        With multiple paragraphs.



        And some extra whitespace.
        """
        clean = parser._clean_section_text(text)

        # Should not have excessive whitespace
        assert "   " not in clean
        assert "\n\n\n" not in clean

    def test_parse_sections_only_10k(self, parser: FilingParser):
        """Test parsing 10-K HTML content directly."""
        html = """
        <html>
        <body>
        <h1>UNITED STATES SECURITIES AND EXCHANGE COMMISSION</h1>

        <h2>Item 1. Business</h2>
        <p>Apple Inc. designs, manufactures, and markets smartphones, personal computers,
        tablets, wearables and accessories, and sells a variety of related services.
        The Company's fiscal year is the 52 or 53-week period that ends on the last
        Saturday of September.</p>

        <h2>Item 1A. Risk Factors</h2>
        <p>The Company's business, reputation, results of operations, financial condition
        and stock price can be affected by a number of factors. The following discussion
        identifies some of the most significant risk factors.</p>

        <h2>Item 7. Management's Discussion and Analysis</h2>
        <p>The following discussion should be read in conjunction with the Company's
        Consolidated Financial Statements and accompanying Notes.</p>
        </body>
        </html>
        """

        sections = parser.parse_sections_only(html, FilingType.FORM_10K)

        # Key sections parser extracts: business, risk_factors, mda
        assert len(sections) >= 1  # At least one section should be extracted


class TestGetSectionMetadata:
    """Tests for get_section_metadata function."""

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

    def test_metadata_includes_all_fields(self, sample_filing: Filing):
        """Test that metadata includes all required fields."""
        metadata = get_section_metadata("mda", sample_filing)

        assert metadata["source"] == "sec_edgar"
        assert metadata["document_type"] == "sec_filing"
        assert metadata["filing_type"] == "10-K"
        assert metadata["accession_number"] == "0000320193-23-000077"
        assert metadata["cik"] == "0000320193"
        assert metadata["ticker"] == "AAPL"
        assert metadata["company_name"] == "Apple Inc."
        assert metadata["section"] == "mda"
        assert "filing_date" in metadata
        assert "period_of_report" in metadata
