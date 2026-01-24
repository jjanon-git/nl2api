"""
RAG Ingestion Module

Handles ingestion of documents for RAG systems.
"""

from src.rag.ingestion.sec_filings import (
    DocumentChunker,
    Filing,
    FilingCheckpoint,
    FilingChunk,
    FilingMetadataRepo,
    FilingParser,
    FilingRAGIndexer,
    FilingSection,
    FilingType,
    SECEdgarClient,
    SECFilingConfig,
    estimate_chunk_count,
    filter_filings_by_date,
    get_section_metadata,
    load_sp500_companies,
)

__all__ = [
    # Config
    "SECFilingConfig",
    # Models
    "Filing",
    "FilingChunk",
    "FilingCheckpoint",
    "FilingType",
    "FilingSection",
    # Client
    "SECEdgarClient",
    "load_sp500_companies",
    "filter_filings_by_date",
    # Parser
    "FilingParser",
    "get_section_metadata",
    # Chunker
    "DocumentChunker",
    "estimate_chunk_count",
    # Indexer
    "FilingRAGIndexer",
    "FilingMetadataRepo",
]
