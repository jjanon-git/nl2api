"""
SEC Filing Parser

Parses 10-K and 10-Q HTML filings to extract text sections.
"""

import logging
import re
from pathlib import Path
from typing import Any

from src.rag.ingestion.sec_filings.models import Filing, FilingSection, FilingType

logger = logging.getLogger(__name__)


# Section patterns for 10-K filings
# These patterns match the beginning of each section header
SECTION_PATTERNS_10K: dict[FilingSection, list[str]] = {
    FilingSection.BUSINESS: [
        r"item\s*1\.?\s*[-–—]?\s*business",
        r"item\s*1\b[^0-9a-z]",  # Just "Item 1" followed by non-alphanumeric
    ],
    FilingSection.RISK_FACTORS: [
        r"item\s*1a\.?\s*[-–—]?\s*risk\s*factors",
        r"item\s*1a\b[^0-9]",
    ],
    FilingSection.UNRESOLVED_STAFF_COMMENTS: [
        r"item\s*1b\.?\s*[-–—]?\s*unresolved\s*staff",
        r"item\s*1b\b[^0-9]",
    ],
    FilingSection.PROPERTIES: [
        r"item\s*2\.?\s*[-–—]?\s*properties",
        r"item\s*2\b[^0-9]",
    ],
    FilingSection.LEGAL_PROCEEDINGS: [
        r"item\s*3\.?\s*[-–—]?\s*legal\s*proceedings",
        r"item\s*3\b[^0-9]",
    ],
    FilingSection.MINE_SAFETY: [
        r"item\s*4\.?\s*[-–—]?\s*mine\s*safety",
        r"item\s*4\b[^0-9]",
    ],
    FilingSection.MARKET_INFO: [
        r"item\s*5\.?\s*[-–—]?\s*market\s*for",
        r"item\s*5\b[^0-9]",
    ],
    FilingSection.SELECTED_FINANCIAL: [
        r"item\s*6\.?\s*[-–—]?\s*selected\s*financial",
        r"\[?reserved\]?",  # Item 6 is often reserved
        r"item\s*6\b[^0-9]",
    ],
    FilingSection.MDA: [
        r"item\s*7\.?\s*[-–—]?\s*management['''\u2019]?s?\s*discussion",
        r"item\s*7\b[^0-9a-z]",
        r"management['''\u2019]?s?\s*discussion\s*and\s*analysis",
    ],
    FilingSection.QUANTITATIVE_QUALITATIVE: [
        r"item\s*7a\.?\s*[-–—]?\s*quantitative",
        r"item\s*7a\b[^0-9]",
    ],
    FilingSection.FINANCIAL_STATEMENTS: [
        r"item\s*8\.?\s*[-–—]?\s*financial\s*statements",
        r"item\s*8\b[^0-9]",
    ],
    FilingSection.CHANGES_ACCOUNTANTS: [
        r"item\s*9\.?\s*[-–—]?\s*changes\s*in\s*and\s*disagreements",
        r"item\s*9\b[^0-9a-z]",
    ],
    FilingSection.CONTROLS_PROCEDURES: [
        r"item\s*9a\.?\s*[-–—]?\s*controls?\s*and\s*procedures",
        r"item\s*9a\b[^0-9]",
    ],
    FilingSection.OTHER: [
        r"item\s*9b\.?\s*[-–—]?\s*other",
        r"item\s*9b\b[^0-9]",
    ],
}

# Section patterns for 10-Q filings (Part I - Financial Information)
SECTION_PATTERNS_10Q: dict[FilingSection, list[str]] = {
    FilingSection.FINANCIAL_STATEMENTS: [
        r"item\s*1\.?\s*[-–—]?\s*financial\s*statements",
        r"part\s*i[^v].*item\s*1\b",
    ],
    FilingSection.MDA: [
        r"item\s*2\.?\s*[-–—]?\s*management['''\u2019]?s?\s*discussion",
        r"management['''\u2019]?s?\s*discussion\s*and\s*analysis",
    ],
    FilingSection.QUANTITATIVE_QUALITATIVE: [
        r"item\s*3\.?\s*[-–—]?\s*quantitative",
    ],
    FilingSection.CONTROLS_PROCEDURES: [
        r"item\s*4\.?\s*[-–—]?\s*controls?\s*and\s*procedures",
    ],
    # Part II items
    FilingSection.LEGAL_PROCEEDINGS_Q: [
        r"item\s*1\.?\s*[-–—]?\s*legal\s*proceedings",
        r"part\s*ii.*item\s*1\b",
    ],
    FilingSection.RISK_FACTORS_Q: [
        r"item\s*1a\.?\s*[-–—]?\s*risk\s*factors",
    ],
}

# Order of sections for determining section boundaries
SECTION_ORDER_10K = [
    FilingSection.BUSINESS,
    FilingSection.RISK_FACTORS,
    FilingSection.UNRESOLVED_STAFF_COMMENTS,
    FilingSection.PROPERTIES,
    FilingSection.LEGAL_PROCEEDINGS,
    FilingSection.MINE_SAFETY,
    FilingSection.MARKET_INFO,
    FilingSection.SELECTED_FINANCIAL,
    FilingSection.MDA,
    FilingSection.QUANTITATIVE_QUALITATIVE,
    FilingSection.FINANCIAL_STATEMENTS,
    FilingSection.CHANGES_ACCOUNTANTS,
    FilingSection.CONTROLS_PROCEDURES,
    FilingSection.OTHER,
]

SECTION_ORDER_10Q = [
    FilingSection.FINANCIAL_STATEMENTS,
    FilingSection.MDA,
    FilingSection.QUANTITATIVE_QUALITATIVE,
    FilingSection.CONTROLS_PROCEDURES,
    FilingSection.LEGAL_PROCEEDINGS_Q,
    FilingSection.RISK_FACTORS_Q,
]

# Key sections to extract (most valuable for RAG)
KEY_SECTIONS_10K = {
    FilingSection.BUSINESS,
    FilingSection.RISK_FACTORS,
    FilingSection.MDA,
}

KEY_SECTIONS_10Q = {
    FilingSection.MDA,
    FilingSection.RISK_FACTORS_Q,
}


class FilingParser:
    """
    Parser for SEC 10-K and 10-Q filings.

    Extracts text sections from HTML filings for RAG indexing.
    """

    def __init__(self, extract_all_sections: bool = False):
        """
        Initialize filing parser.

        Args:
            extract_all_sections: If True, extract all sections.
                                  If False, only extract key sections (MDA, Risk Factors, Business).
        """
        self._extract_all_sections = extract_all_sections

    def parse(
        self,
        html_path: Path,
        filing: Filing,
    ) -> dict[str, str]:
        """
        Parse a filing HTML file and extract sections.

        Args:
            html_path: Path to the downloaded HTML file
            filing: Filing metadata

        Returns:
            Dict mapping section names to extracted text content
        """
        logger.info(f"Parsing filing: {filing.accession_number} ({filing.filing_type.value})")

        # Read HTML content
        html_content = html_path.read_text(encoding="utf-8", errors="replace")

        # Clean HTML
        clean_text = self._clean_html(html_content)

        # Get section patterns based on filing type
        if filing.filing_type in (FilingType.FORM_10K, FilingType.FORM_10K_A):
            patterns = SECTION_PATTERNS_10K
            section_order = SECTION_ORDER_10K
            key_sections = KEY_SECTIONS_10K
        else:
            patterns = SECTION_PATTERNS_10Q
            section_order = SECTION_ORDER_10Q
            key_sections = KEY_SECTIONS_10Q

        # Determine which sections to extract
        if self._extract_all_sections:
            sections_to_extract = set(section_order)
        else:
            sections_to_extract = key_sections

        # Find section boundaries
        section_positions = self._find_section_positions(clean_text, patterns, sections_to_extract)

        # Extract section content
        sections = {}
        sorted_positions = sorted(section_positions.items(), key=lambda x: x[1])

        for i, (section, start_pos) in enumerate(sorted_positions):
            # End position is start of next section or end of document
            if i + 1 < len(sorted_positions):
                end_pos = sorted_positions[i + 1][1]
            else:
                end_pos = len(clean_text)

            section_text = clean_text[start_pos:end_pos].strip()

            # Clean up section text
            section_text = self._clean_section_text(section_text)

            if section_text and len(section_text) > 100:  # Minimum content threshold
                sections[section.value] = section_text
                logger.debug(f"Extracted section {section.value}: {len(section_text)} chars")

        logger.info(f"Extracted {len(sections)} sections from filing")
        return sections

    def _clean_html(self, html: str) -> str:
        """
        Remove HTML tags and clean text content.

        Args:
            html: Raw HTML content

        Returns:
            Cleaned plain text
        """
        # Try to use BeautifulSoup if available
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(html, "lxml")

            # Remove script, style, and hidden XBRL header (metadata we don't need)
            for element in soup(["script", "style", "ix:header"]):
                element.decompose()

            # IMPORTANT: Unwrap (not decompose!) ix:nonnumeric and ix:nonfraction tags
            # These tags CONTAIN the actual text content in iXBRL filings
            # decompose() removes content, unwrap() keeps text and removes only the tag
            for tag_name in ["ix:nonnumeric", "ix:nonfraction", "ix:continuation"]:
                for element in soup.find_all(tag_name):
                    element.unwrap()

            # Get text
            text = soup.get_text(separator=" ", strip=True)

        except ImportError:
            # Fallback to regex-based cleaning
            text = self._regex_clean_html(html)

        # Decode any remaining HTML entities (&#160;, &#8220;, etc.)
        import html as html_module

        text = html_module.unescape(text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text)
        text = re.sub(r"\n\s*\n", "\n\n", text)

        return text.strip()

    def _regex_clean_html(self, html: str) -> str:
        """
        Clean HTML using regex (fallback when BeautifulSoup not available).

        Args:
            html: Raw HTML content

        Returns:
            Cleaned text
        """
        # Remove script and style blocks
        html = re.sub(r"<script[^>]*>.*?</script>", " ", html, flags=re.DOTALL | re.IGNORECASE)
        html = re.sub(r"<style[^>]*>.*?</style>", " ", html, flags=re.DOTALL | re.IGNORECASE)

        # Remove XBRL inline elements
        html = re.sub(r"<ix:[^>]*>", " ", html, flags=re.IGNORECASE)
        html = re.sub(r"</ix:[^>]*>", " ", html, flags=re.IGNORECASE)

        # Remove HTML comments
        html = re.sub(r"<!--.*?-->", " ", html, flags=re.DOTALL)

        # Remove all HTML tags
        html = re.sub(r"<[^>]+>", " ", html)

        # Decode HTML entities
        html = html.replace("&nbsp;", " ")
        html = html.replace("&amp;", "&")
        html = html.replace("&lt;", "<")
        html = html.replace("&gt;", ">")
        html = html.replace("&quot;", '"')
        html = html.replace("&#39;", "'")
        html = html.replace("&rsquo;", "'")
        html = html.replace("&ldquo;", '"')
        html = html.replace("&rdquo;", '"')
        html = html.replace("&mdash;", "—")
        html = html.replace("&ndash;", "–")

        return html

    def _find_section_positions(
        self,
        text: str,
        patterns: dict[FilingSection, list[str]],
        sections_to_find: set[FilingSection],
    ) -> dict[FilingSection, int]:
        """
        Find starting positions of sections in the text.

        Handles table of contents by finding all matches and selecting
        the one that appears to be the actual section start (not a TOC entry).

        Args:
            text: Cleaned text content
            patterns: Section patterns to search for
            sections_to_find: Set of sections to look for

        Returns:
            Dict mapping sections to their start positions
        """
        positions: dict[FilingSection, int] = {}
        text_lower = text.lower()

        for section in sections_to_find:
            if section not in patterns:
                continue

            best_pos = None
            for pattern in patterns[section]:
                # Find ALL matches, not just the first one
                for match in re.finditer(pattern, text_lower):
                    pos = match.start()

                    # Skip if this looks like a table of contents entry
                    # TOC entries are typically followed by page numbers within 100 chars
                    if self._is_toc_entry(text_lower, pos):
                        continue

                    # Use this match if it's the first valid one we found
                    if best_pos is None:
                        best_pos = pos
                        break  # Found a valid match for this pattern

                if best_pos is not None:
                    break  # Found a valid match, don't try other patterns

            if best_pos is not None:
                positions[section] = best_pos

        return positions

    def _is_toc_entry(self, text: str, pos: int) -> bool:
        """
        Check if a match position appears to be a table of contents entry.

        TOC entries typically have:
        - A page number shortly after the section name
        - Multiple "Item X" references in close succession
        - Limited content before the next "Item" reference

        Args:
            text: The full text (lowercase)
            pos: Position of the match

        Returns:
            True if this appears to be a TOC entry, False otherwise
        """
        # Look at the next 200 characters after the match
        lookahead = text[pos : pos + 200]

        # TOC pattern: section name followed by page number pattern
        # e.g., "item 2. management's discussion... 22 item 3"
        # The key indicator is a bare number followed by another "item"
        toc_pattern = r"^[^0-9]*\d{1,3}\s+item\s*\d"
        if re.search(toc_pattern, lookahead):
            return True

        # Another TOC indicator: very short distance to next item reference
        # In actual content, items are separated by substantial text
        next_item = re.search(r"item\s*\d", lookahead[20:])  # Skip the current item
        if next_item and next_item.start() < 100:
            # Next item reference is within ~100 chars - likely TOC
            return True

        return False

    def _clean_section_text(self, text: str) -> str:
        """
        Clean extracted section text.

        Args:
            text: Raw section text

        Returns:
            Cleaned text
        """
        # Remove page numbers and headers/footers patterns
        text = re.sub(r"\d+\s*Table of Contents", "", text, flags=re.IGNORECASE)
        text = re.sub(r"Table of Contents\s*\d+", "", text, flags=re.IGNORECASE)

        # Remove standalone page numbers
        text = re.sub(r"(?:^|\n)\s*\d{1,3}\s*(?:\n|$)", "\n", text)

        # Remove excessive whitespace while preserving paragraphs
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)

        # Remove common SEC filing boilerplate at start
        text = re.sub(
            r"^.*?(?:item\s*\d+[a-z]?\.?\s*[-–—]?\s*)",
            "",
            text,
            count=1,
            flags=re.IGNORECASE | re.DOTALL,
        )

        return text.strip()

    def parse_sections_only(
        self,
        html_content: str,
        filing_type: FilingType,
    ) -> dict[str, str]:
        """
        Parse HTML content directly without file I/O.

        Useful for testing or when content is already in memory.

        Args:
            html_content: HTML content as string
            filing_type: Type of filing

        Returns:
            Dict mapping section names to extracted text content
        """
        clean_text = self._clean_html(html_content)

        if filing_type in (FilingType.FORM_10K, FilingType.FORM_10K_A):
            patterns = SECTION_PATTERNS_10K
            section_order = SECTION_ORDER_10K
            key_sections = KEY_SECTIONS_10K
        else:
            patterns = SECTION_PATTERNS_10Q
            section_order = SECTION_ORDER_10Q
            key_sections = KEY_SECTIONS_10Q

        if self._extract_all_sections:
            sections_to_extract = set(section_order)
        else:
            sections_to_extract = key_sections

        section_positions = self._find_section_positions(clean_text, patterns, sections_to_extract)

        sections = {}
        sorted_positions = sorted(section_positions.items(), key=lambda x: x[1])

        for i, (section, start_pos) in enumerate(sorted_positions):
            if i + 1 < len(sorted_positions):
                end_pos = sorted_positions[i + 1][1]
            else:
                end_pos = len(clean_text)

            section_text = clean_text[start_pos:end_pos].strip()
            section_text = self._clean_section_text(section_text)

            if section_text and len(section_text) > 100:
                sections[section.value] = section_text

        return sections


def get_section_metadata(section: str, filing: Filing) -> dict[str, Any]:
    """
    Generate metadata for a filing section.

    Args:
        section: Section name
        filing: Filing object

    Returns:
        Metadata dictionary for RAG indexing
    """
    return {
        "source": "sec_edgar",
        "document_type": "sec_filing",
        "filing_type": filing.filing_type.value,
        "accession_number": filing.accession_number,
        "cik": filing.cik,
        "ticker": filing.ticker,
        "company_name": filing.company_name,
        "filing_date": filing.filing_date.isoformat(),
        "period_of_report": filing.period_of_report.isoformat(),
        "section": section,
    }
