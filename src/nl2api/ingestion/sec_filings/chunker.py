"""
Document Chunker for SEC Filings

Chunks filing sections into smaller pieces suitable for RAG indexing.
Respects semantic boundaries (paragraphs, sentences) where possible.
"""

import logging
import re

from src.nl2api.ingestion.sec_filings.config import SECFilingConfig
from src.nl2api.ingestion.sec_filings.models import Filing, FilingChunk

logger = logging.getLogger(__name__)


class DocumentChunker:
    """
    Chunks SEC filing sections for RAG indexing.

    Features:
    - Respects section boundaries (doesn't split across sections)
    - Splits on paragraph boundaries when possible
    - Falls back to sentence splitting for long paragraphs
    - Maintains configurable overlap between chunks
    """

    def __init__(
        self,
        chunk_size: int = 4000,
        chunk_overlap: int = 800,
        min_chunk_size: int = 200,
    ):
        """
        Initialize chunker.

        Args:
            chunk_size: Target chunk size in characters
            chunk_overlap: Overlap between consecutive chunks
            min_chunk_size: Minimum chunk size (smaller chunks are merged or dropped)
        """
        self._chunk_size = chunk_size
        self._chunk_overlap = chunk_overlap
        self._min_chunk_size = min_chunk_size

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
        )

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

        # Split into chunks
        text_chunks = self._split_text(text)

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

            chunk = FilingChunk(
                chunk_id=chunk_id,
                filing_accession=filing.accession_number,
                section=section_name,
                chunk_index=i,
                content=chunk_text,
                char_start=chunk_start,
                char_end=chunk_end,
                metadata={
                    "source": "sec_edgar",
                    "document_type": "sec_filing",
                    "filing_type": filing.filing_type.value,
                    "cik": filing.cik,
                    "ticker": filing.ticker,
                    "company_name": filing.company_name,
                    "filing_date": filing.filing_date.isoformat(),
                    "period_of_report": filing.period_of_report.isoformat(),
                    "section": section_name,
                    "chunk_index": i,
                    "total_chunks_in_section": len(text_chunks),
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
        overlap = text[-self._chunk_overlap:]

        # Find first word boundary
        space_idx = overlap.find(" ")
        if space_idx > 0 and space_idx < len(overlap) // 2:
            overlap = overlap[space_idx + 1 :]

        return overlap


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
