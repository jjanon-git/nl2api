"""
SEC Filing Ingestion Pipeline

Downloads, parses, chunks, and indexes SEC 10-K and 10-Q filings for RAG.

Usage:
    python scripts/ingest_sec_filings.py --tickers AAPL,MSFT,GOOGL

Components:
    - SECEdgarClient: Rate-limited client for SEC EDGAR API
    - FilingParser: Extracts sections from 10-K/10-Q HTML
    - DocumentChunker: Splits sections into chunks for RAG
    - FilingRAGIndexer: Indexes chunks with local embeddings
"""

from src.rag.ingestion.sec_filings.chunker import DocumentChunker, estimate_chunk_count
from src.rag.ingestion.sec_filings.client import (
    SECEdgarClient,
    filter_filings_by_date,
    load_sp500_companies,
)
from src.rag.ingestion.sec_filings.config import SECFilingConfig
from src.rag.ingestion.sec_filings.indexer import FilingMetadataRepo, FilingRAGIndexer
from src.rag.ingestion.sec_filings.models import (
    Filing,
    FilingCheckpoint,
    FilingChunk,
    FilingSection,
    FilingType,
)
from src.rag.ingestion.sec_filings.parser import FilingParser, get_section_metadata

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
