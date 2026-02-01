"""Tests for SEC filing parser."""

from src.rag.ingestion.sec_filings.models import FilingType
from src.rag.ingestion.sec_filings.parser import FilingParser


class TestFilingParser:
    """Tests for FilingParser class."""

    def test_skips_table_of_contents_entries(self):
        """Parser should skip TOC entries and find actual section content."""
        # HTML with TOC followed by actual sections
        test_html = """
        <html>
        <body>
        <h1>Table of Contents</h1>
        <p>Item 1. Financial Statements 5</p>
        <p>Item 2. Management's Discussion and Analysis 22</p>
        <p>Item 3. Quantitative and Qualitative Disclosures 33</p>

        <h2>Item 1. Financial Statements</h2>
        <p>This is the actual financial statements section with substantial content.</p>

        <h2>Item 2. Management's Discussion and Analysis</h2>
        <p>This is the actual MDA section with substantial content about business operations,
        results of operations, and liquidity. Revenue increased by 15% compared to prior year.</p>

        <h2>Item 3. Quantitative and Qualitative Disclosures About Market Risk</h2>
        <p>Market risk disclosures here with enough content to pass the minimum threshold.</p>
        </body>
        </html>
        """

        parser = FilingParser(extract_all_sections=False)
        sections = parser.parse_sections_only(test_html, FilingType.FORM_10Q)

        # Should find MDA section
        assert "mda" in sections

        # Content should be from actual section, not TOC
        mda_content = sections["mda"]
        assert "actual MDA section" in mda_content
        assert "business operations" in mda_content
        # Should NOT contain TOC page numbers
        assert "22 Item 3" not in mda_content

    def test_handles_ixbrl_format(self):
        """Parser should extract text from iXBRL tags, not remove it."""
        test_html = """
        <html>
        <body>
        <ix:header>XBRL metadata to remove</ix:header>
        <p>Item 2. Management's Discussion and Analysis</p>
        <ix:nonnumeric name="us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax">
        Revenue was <ix:nonfraction>$1.5 billion</ix:nonfraction> for the quarter.
        </ix:nonnumeric>
        <p>The company continued to invest in technology and innovation during the period.</p>
        </body>
        </html>
        """

        parser = FilingParser(extract_all_sections=False)
        sections = parser.parse_sections_only(test_html, FilingType.FORM_10Q)

        assert "mda" in sections
        mda_content = sections["mda"]
        # Text inside ix:nonnumeric should be preserved
        assert "Revenue was" in mda_content
        assert "$1.5 billion" in mda_content
        # ix:header content should be removed
        assert "XBRL metadata" not in mda_content

    def test_decodes_html_entities(self):
        """Parser should decode HTML entities like &#160; and &#8220;."""
        test_html = """
        <html>
        <body>
        <p>Item 2. Management&#8217;s Discussion and Analysis</p>
        <p>Revenue&#160;increased by &#8220;significant&#8221; amount during the quarter.</p>
        <p>More content here to ensure section passes minimum length threshold for extraction.</p>
        </body>
        </html>
        """

        parser = FilingParser(extract_all_sections=False)
        sections = parser.parse_sections_only(test_html, FilingType.FORM_10Q)

        assert "mda" in sections
        mda_content = sections["mda"]
        # Entities should be decoded
        assert "increased by" in mda_content
        # Should not contain raw entity codes
        assert "&#160;" not in mda_content
        assert "&#8220;" not in mda_content

    def test_handles_unicode_apostrophe_in_section_headers(self):
        """Parser should match section headers with curly apostrophe (U+2019)."""
        # Use the actual Unicode character
        test_html = """
        <html>
        <body>
        <p>Item 2. Management\u2019s Discussion and Analysis</p>
        <p>This is the MDA content with detailed analysis of operations and financial results
        for the quarter. The company reported strong performance across all segments.</p>
        </body>
        </html>
        """

        parser = FilingParser(extract_all_sections=False)
        sections = parser.parse_sections_only(test_html, FilingType.FORM_10Q)

        # Should still find MDA despite curly apostrophe
        assert "mda" in sections

    def test_is_toc_entry_detection(self):
        """Test the TOC entry detection helper method."""
        parser = FilingParser()

        # TOC entry: item followed by page number then another item
        toc_text = "item 2. management's discussion 22 item 3. quantitative"
        assert parser._is_toc_entry(toc_text, 0) is True

        # Actual section: no page number pattern
        section_text = (
            "item 2. management's discussion and analysis overview this section provides "
            "detailed information about operations and financial condition of the company."
        )
        assert parser._is_toc_entry(section_text, 0) is False

        # Another TOC pattern: items very close together
        short_toc = "item 2. mda overview item 3. risk"
        assert parser._is_toc_entry(short_toc, 0) is True
