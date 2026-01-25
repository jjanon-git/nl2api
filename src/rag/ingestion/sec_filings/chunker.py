"""
Document Chunker for SEC Filings

Chunks filing sections into smaller pieces suitable for RAG indexing.
Respects semantic boundaries (paragraphs, sentences) where possible.
"""

import logging
import re
from datetime import datetime

from src.rag.ingestion.sec_filings.config import SECFilingConfig
from src.rag.ingestion.sec_filings.models import Filing, FilingChunk

logger = logging.getLogger(__name__)


def derive_fiscal_quarter(period_of_report: datetime, filing_type: str) -> int | None:
    """
    Derive fiscal quarter from period of report date.

    For 10-Q filings, determine which quarter based on the period end month.
    Most companies follow calendar year, so:
    - Q1: Jan-Mar (month 1-3)
    - Q2: Apr-Jun (month 4-6)
    - Q3: Jul-Sep (month 7-9)
    - Q4: Oct-Dec (month 10-12) - typically covered by 10-K

    Args:
        period_of_report: Period end date
        filing_type: Filing type (10-K, 10-Q, etc.)

    Returns:
        Quarter number (1-4) or None for 10-K filings
    """
    # 10-K covers the full year, not a specific quarter
    if "10-K" in filing_type:
        return None

    month = period_of_report.month
    if month <= 3:
        return 1
    elif month <= 6:
        return 2
    elif month <= 9:
        return 3
    else:
        return 4


class DocumentChunker:
    """
    Chunks SEC filing sections for RAG indexing.

    Features:
    - Respects section boundaries (doesn't split across sections)
    - Splits on paragraph boundaries when possible
    - Falls back to sentence splitting for long paragraphs
    - Maintains configurable overlap between chunks
    - **Contextual chunking**: Prepends document context to improve embeddings
    """

    # Section name mappings for human-readable context
    SECTION_LABELS = {
        "business": "Business Description",
        "mda": "Management's Discussion and Analysis",
        "risk_factors": "Risk Factors",
        "properties": "Properties",
        "legal_proceedings": "Legal Proceedings",
        "controls_procedures": "Controls and Procedures",
        "financial_statements": "Financial Statements",
        "exhibits": "Exhibits and Financial Statement Schedules",
    }

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 800,
        min_chunk_size: int = 200,
        contextual_chunking: bool = True,
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size (smaller chunks are merged or dropped)
            contextual_chunking: If True, prepend document context to each chunk
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size
        self._contextual_chunking = contextual_chunking

        # Sentence boundary pattern
        self._sentence_pattern = re.compile(
            r"(?<=[.!?])\s+(?=[A-Z])"  # Split after sentence-ending punctuation before capital
        )

        # Paragraph boundary pattern (two or more newlines)
        self._paragraph_pattern = re.compile(r"\n\s*\n")

    @classmethod
    def from_config(cls, config: SECFilingConfig) -> "DocumentChunker":
        """Create chunker from configuration."""
        return cls(
            chunk_size=config.chunk_size,
            chunk_overlap=config.chunk_overlap,
            min_chunk_size=config.min_chunk_size,
            contextual_chunking=getattr(config, "contextual_chunking", True),
        )

    def _generate_context_prefix(
        self,
        filing: Filing,
        section_name: str,
    ) -> str:
        """
        Generate a context prefix for chunks.

        This implements contextual chunking (also called "late chunking context").
        The prefix provides document-level context that helps embeddings
        better capture the semantic meaning of each chunk.

        Research shows this can improve retrieval accuracy by 15-35%.

        Args:
            filing: Filing metadata
            section_name: Section being chunked

        Returns:
            Context prefix string to prepend to chunk content
        """
        section_label = self.SECTION_LABELS.get(section_name, section_name.title())

        # Format the fiscal period
        period = filing.period_of_report
        if "10-K" in filing.filing_type.value:
            period_str = f"Fiscal Year {period.year}"
        else:
            quarter = derive_fiscal_quarter(period, filing.filing_type.value)
            period_str = f"Q{quarter} {period.year}" if quarter else f"{period.strftime('%B %Y')}"

        # Build context prefix
        context_lines = [
            f"Company: {filing.company_name} ({filing.ticker})",
            f"Filing: {filing.filing_type.value}, {period_str}",
            f"Section: {section_label}",
            "",  # Blank line before content
        ]

        return "\n".join(context_lines)

    def chunk_filing(
        self,
        sections: dict[str, str],
        filing: Filing,
    ) -> list[FilingChunk]:
        """
        Chunk all sections of a filing.

        Args:
            sections: Dict mapping section names to text content
            filing: Filing metadata

        Returns:
            List of FilingChunk objects
        """
        all_chunks = []

        for section_name, section_text in sections.items():
            section_chunks = self.chunk_section(
                text=section_text,
                filing=filing,
                section_name=section_name,
            )
            all_chunks.extend(section_chunks)

        logger.info(
            f"Chunked filing {filing.accession_number}: "
            f"{len(sections)} sections → {len(all_chunks)} chunks"
        )
        return all_chunks

    def chunk_section(
        self,
        text: str,
        filing: Filing,
        section_name: str,
    ) -> list[FilingChunk]:
        """
        Chunk a single section.

        Args:
            text: Section text content
            filing: Filing metadata
            section_name: Name of the section

        Returns:
            List of FilingChunk objects
        """
        if not text or len(text) < self._min_chunk_size:
            return []

        # Generate context prefix if contextual chunking is enabled
        context_prefix = ""
        if self._contextual_chunking:
            context_prefix = self._generate_context_prefix(filing, section_name)

        # Adjust chunk size to account for context prefix
        effective_chunk_size = self._chunk_size
        if context_prefix:
            effective_chunk_size = max(
                self._chunk_size - len(context_prefix),
                self._min_chunk_size * 2,  # Ensure we still have room for content
            )

        # Split into chunks (using effective size if contextual)
        original_chunk_size = self._chunk_size
        self._chunk_size = effective_chunk_size
        text_chunks = self._split_text(text)
        self._chunk_size = original_chunk_size

        # Create FilingChunk objects
        chunks = []
        char_offset = 0

        for i, chunk_text in enumerate(text_chunks):
            chunk_id = f"{filing.accession_number}_{section_name}_{i}"

            # Find actual position in original text
            chunk_start = text.find(chunk_text[:100], char_offset)
            if chunk_start == -1:
                chunk_start = char_offset
            chunk_end = chunk_start + len(chunk_text)

            # Derive additional metadata
            filing_type_str = filing.filing_type.value
            fiscal_quarter = derive_fiscal_quarter(filing.period_of_report, filing_type_str)
            is_amendment = "/A" in filing_type_str

            # Prepend context to chunk content if enabled
            final_content = context_prefix + chunk_text if context_prefix else chunk_text

            chunk = FilingChunk(
                chunk_id=chunk_id,
                filing_accession=filing.accession_number,
                section=section_name,
                chunk_index=i,
                content=final_content,
                char_start=chunk_start,
                char_end=chunk_end,
                metadata={
                    "source": "sec_edgar",
                    "document_type": "sec_filing",
                    "filing_type": filing_type_str,
                    "cik": filing.cik,
                    "ticker": filing.ticker,
                    "company_name": filing.company_name,
                    "filing_date": filing.filing_date.isoformat(),
                    "period_of_report": filing.period_of_report.isoformat(),
                    "fiscal_year": filing.period_of_report.year,
                    "fiscal_quarter": fiscal_quarter,  # None for 10-K, 1-4 for 10-Q
                    "is_amendment": is_amendment,
                    "section": section_name,
                    "chunk_index": i,
                    "total_chunks_in_section": len(text_chunks),
                    "contextual_chunking": self._contextual_chunking,
                },
            )
            chunks.append(chunk)

            # Update offset for next chunk (account for overlap)
            char_offset = max(char_offset, chunk_end - self._chunk_overlap)

        return chunks

    def _split_text(self, text: str) -> list[str]:
        """
        Split text into chunks.

        Strategy:
        1. First try splitting by paragraphs
        2. If paragraphs are too long, split by sentences
        3. If sentences are too long, split by character with word boundaries

        Args:
            text: Text to split

        Returns:
            List of chunk strings
        """
        # If text is small enough, return as single chunk
        if len(text) <= self._chunk_size:
            return [text.strip()] if text.strip() else []

        # Split into paragraphs
        paragraphs = self._paragraph_pattern.split(text)
        paragraphs = [p.strip() for p in paragraphs if p.strip()]

        chunks = []
        current_chunk = ""

        for para in paragraphs:
            # If paragraph itself is too long, split it
            if len(para) > self._chunk_size:
                # Flush current chunk first
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""

                # Split the long paragraph
                para_chunks = self._split_long_paragraph(para)
                chunks.extend(para_chunks)
                continue

            # Check if adding this paragraph would exceed chunk size
            test_chunk = current_chunk + "\n\n" + para if current_chunk else para

            if len(test_chunk) <= self._chunk_size:
                current_chunk = test_chunk
            else:
                # Save current chunk and start new one
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Start new chunk, possibly with overlap
                if self._chunk_overlap > 0 and current_chunk:
                    # Get overlap from end of previous chunk
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + "\n\n" + para if overlap_text else para
                else:
                    current_chunk = para

        # Don't forget the last chunk
        if current_chunk and len(current_chunk.strip()) >= self._min_chunk_size:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_long_paragraph(self, para: str) -> list[str]:
        """
        Split a long paragraph by sentences.

        Args:
            para: Paragraph text

        Returns:
            List of chunks
        """
        sentences = self._sentence_pattern.split(para)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            # Fallback: split by character with word boundaries
            return self._split_by_characters(para)

        chunks = []
        current_chunk = ""

        for sentence in sentences:
            # If single sentence is too long, split by characters
            if len(sentence) > self._chunk_size:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                    current_chunk = ""
                chunks.extend(self._split_by_characters(sentence))
                continue

            test_chunk = current_chunk + " " + sentence if current_chunk else sentence

            if len(test_chunk) <= self._chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                if self._chunk_overlap > 0 and current_chunk:
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    current_chunk = sentence

        if current_chunk and len(current_chunk.strip()) >= self._min_chunk_size:
            chunks.append(current_chunk.strip())

        return chunks

    def _split_by_characters(self, text: str) -> list[str]:
        """
        Split text by character count, respecting word boundaries.

        Args:
            text: Text to split

        Returns:
            List of chunks
        """
        chunks = []
        words = text.split()
        current_chunk = ""

        for word in words:
            test_chunk = current_chunk + " " + word if current_chunk else word

            if len(test_chunk) <= self._chunk_size:
                current_chunk = test_chunk
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())

                # Handle overlap
                if self._chunk_overlap > 0 and current_chunk:
                    overlap_text = self._get_overlap_text(current_chunk)
                    current_chunk = overlap_text + " " + word if overlap_text else word
                else:
                    current_chunk = word

        if current_chunk and len(current_chunk.strip()) >= self._min_chunk_size:
            chunks.append(current_chunk.strip())

        return chunks

    def _get_overlap_text(self, text: str) -> str:
        """
        Get overlap text from end of a chunk.

        Args:
            text: Previous chunk text

        Returns:
            Overlap text (up to chunk_overlap characters)
        """
        if len(text) <= self._chunk_overlap:
            return text

        # Get last N characters, then find word boundary
        overlap = text[-self._chunk_overlap :]

        # Find first word boundary
        space_idx = overlap.find(" ")
        if space_idx > 0 and space_idx < len(overlap) // 2:
            overlap = overlap[space_idx + 1 :]

        return overlap

    def chunk_section_hierarchical(
        self,
        text: str,
        filing: Filing,
        section_name: str,
        parent_chunk_size: int = 4000,
        child_chunk_size: int = 512,
        child_overlap: int = 64,
    ) -> list[FilingChunk]:
        """
        Chunk a section using small-to-big retrieval strategy.

        Creates a two-level hierarchy:
        - Parent chunks (chunk_level=0): Large context windows for retrieval output
        - Child chunks (chunk_level=1): Small precise chunks for search matching

        Search is performed on children, but parents are returned for context.

        Args:
            text: Section text content
            filing: Filing metadata
            section_name: Name of the section
            parent_chunk_size: Size of parent chunks (default 4000)
            child_chunk_size: Size of child chunks (default 512)
            child_overlap: Overlap for child chunks (default 64)

        Returns:
            List of FilingChunk objects (both parents and children)
        """
        if not text or len(text) < self._min_chunk_size:
            return []

        # Generate context prefix if contextual chunking is enabled
        context_prefix = ""
        if self._contextual_chunking:
            context_prefix = self._generate_context_prefix(filing, section_name)

        all_chunks: list[FilingChunk] = []

        # First, create parent chunks (large context)
        # Temporarily adjust chunk size
        original_size = self._chunk_size
        original_overlap = self._chunk_overlap
        self._chunk_size = parent_chunk_size
        self._chunk_overlap = 0  # No overlap for parents
        parent_texts = self._split_text(text)
        self._chunk_size = original_size
        self._chunk_overlap = original_overlap

        parent_char_offset = 0

        for parent_idx, parent_text in enumerate(parent_texts):
            parent_chunk_id = f"{filing.accession_number}_{section_name}_p{parent_idx}"

            # Find position in original text
            parent_start = text.find(parent_text[:100], parent_char_offset)
            if parent_start == -1:
                parent_start = parent_char_offset
            parent_end = parent_start + len(parent_text)

            # Derive additional metadata
            filing_type_str = filing.filing_type.value
            fiscal_quarter = derive_fiscal_quarter(filing.period_of_report, filing_type_str)

            # Create parent chunk
            final_content = context_prefix + parent_text if context_prefix else parent_text
            parent_chunk = FilingChunk(
                chunk_id=parent_chunk_id,
                filing_accession=filing.accession_number,
                section=section_name,
                chunk_index=parent_idx,
                content=final_content,
                char_start=parent_start,
                char_end=parent_end,
                parent_chunk_id=None,  # Parents have no parent
                chunk_level=0,  # Parent level
                metadata={
                    "source": "sec_edgar",
                    "document_type": "sec_filing",
                    "filing_type": filing_type_str,
                    "cik": filing.cik,
                    "ticker": filing.ticker,
                    "company_name": filing.company_name,
                    "filing_date": filing.filing_date.isoformat(),
                    "period_of_report": filing.period_of_report.isoformat(),
                    "fiscal_year": filing.period_of_report.year,
                    "fiscal_quarter": fiscal_quarter,
                    "section": section_name,
                    "chunk_level": 0,
                    "chunk_type": "parent",
                },
            )
            all_chunks.append(parent_chunk)

            # Now create child chunks from this parent
            child_texts = self._split_into_children(parent_text, child_chunk_size, child_overlap)

            for child_idx, child_text in enumerate(child_texts):
                child_chunk_id = (
                    f"{filing.accession_number}_{section_name}_p{parent_idx}_c{child_idx}"
                )

                # Create child chunk (no context prefix - context comes from parent)
                child_chunk = FilingChunk(
                    chunk_id=child_chunk_id,
                    filing_accession=filing.accession_number,
                    section=section_name,
                    chunk_index=child_idx,
                    content=child_text,
                    char_start=parent_start,  # Approximate - within parent
                    char_end=parent_start + len(child_text),
                    parent_chunk_id=parent_chunk_id,  # Link to parent
                    chunk_level=1,  # Child level
                    metadata={
                        "source": "sec_edgar",
                        "document_type": "sec_filing",
                        "filing_type": filing_type_str,
                        "cik": filing.cik,
                        "ticker": filing.ticker,
                        "company_name": filing.company_name,
                        "filing_date": filing.filing_date.isoformat(),
                        "period_of_report": filing.period_of_report.isoformat(),
                        "fiscal_year": filing.period_of_report.year,
                        "fiscal_quarter": fiscal_quarter,
                        "section": section_name,
                        "chunk_level": 1,
                        "chunk_type": "child",
                        "parent_chunk_id": parent_chunk_id,
                    },
                )
                all_chunks.append(child_chunk)

            parent_char_offset = parent_end

        logger.info(
            f"Hierarchical chunking {section_name}: "
            f"{len(parent_texts)} parents, {len(all_chunks) - len(parent_texts)} children"
        )
        return all_chunks

    def _split_into_children(
        self,
        text: str,
        child_size: int,
        overlap: int,
    ) -> list[str]:
        """
        Split text into small child chunks.

        Args:
            text: Parent chunk text
            child_size: Target size for children
            overlap: Overlap between children

        Returns:
            List of child chunk texts
        """
        if len(text) <= child_size:
            return [text.strip()] if text.strip() else []

        # Try to split on sentence boundaries first
        sentences = self._sentence_pattern.split(text)
        sentences = [s.strip() for s in sentences if s.strip()]

        children = []
        current = ""

        for sentence in sentences:
            test = current + " " + sentence if current else sentence

            if len(test) <= child_size:
                current = test
            else:
                if current:
                    children.append(current.strip())

                # Add overlap from previous
                if overlap > 0 and current:
                    overlap_text = current[-overlap:] if len(current) > overlap else current
                    # Find word boundary
                    space_idx = overlap_text.find(" ")
                    if space_idx > 0:
                        overlap_text = overlap_text[space_idx + 1 :]
                    current = overlap_text + " " + sentence if overlap_text else sentence
                else:
                    current = sentence

        if current and len(current.strip()) >= 50:  # Min child size
            children.append(current.strip())

        return children

    def chunk_filing_hierarchical(
        self,
        sections: dict[str, str],
        filing: Filing,
        parent_chunk_size: int = 4000,
        child_chunk_size: int = 512,
        child_overlap: int = 64,
    ) -> list[FilingChunk]:
        """
        Chunk all sections of a filing using hierarchical small-to-big strategy.

        Args:
            sections: Dict mapping section names to text content
            filing: Filing metadata
            parent_chunk_size: Size of parent chunks (default 4000)
            child_chunk_size: Size of child chunks (default 512)
            child_overlap: Overlap for child chunks (default 64)

        Returns:
            List of FilingChunk objects (both parents and children)
        """
        all_chunks = []

        for section_name, section_text in sections.items():
            section_chunks = self.chunk_section_hierarchical(
                text=section_text,
                filing=filing,
                section_name=section_name,
                parent_chunk_size=parent_chunk_size,
                child_chunk_size=child_chunk_size,
                child_overlap=child_overlap,
            )
            all_chunks.extend(section_chunks)

        parent_count = sum(1 for c in all_chunks if c.chunk_level == 0)
        child_count = sum(1 for c in all_chunks if c.chunk_level == 1)

        logger.info(
            f"Hierarchical chunking {filing.accession_number}: "
            f"{len(sections)} sections → {parent_count} parents, {child_count} children"
        )
        return all_chunks


def estimate_chunk_count(
    sections: dict[str, str],
    chunk_size: int = 4000,
    chunk_overlap: int = 800,
) -> int:
    """
    Estimate number of chunks for a filing without actually chunking.

    Useful for progress estimation.

    Args:
        sections: Section texts
        chunk_size: Target chunk size
        chunk_overlap: Overlap between chunks

    Returns:
        Estimated chunk count
    """
    total_chars = sum(len(text) for text in sections.values())

    if total_chars <= chunk_size:
        return len(sections)  # One chunk per section at minimum

    # Estimate chunks accounting for overlap
    effective_chunk_size = chunk_size - chunk_overlap
    if effective_chunk_size <= 0:
        effective_chunk_size = chunk_size // 2

    estimated = total_chars // effective_chunk_size

    # Add some buffer for section boundaries
    estimated += len(sections)

    return max(estimated, len(sections))
