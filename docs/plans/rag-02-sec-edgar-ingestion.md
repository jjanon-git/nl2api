# Plan: SEC EDGAR Filing Ingestion for RAG Evaluation

**Status:** Ready for Full Ingestion
**Created:** 2026-01-22
**Last Updated:** 2026-01-23 (fixed URL: use www.sec.gov for archives, not data.sec.gov)
**Estimated Storage:** ~16 GB (12.5 GB downloads + 3.6 GB database)

---

## Progress Tracking

| Step | Description | Status |
|------|-------------|--------|
| 1 | Data models and config | ✅ Complete |
| 2 | SEC EDGAR client | ✅ Complete |
| 3 | Filing parser | ✅ Complete |
| 4 | Document chunker | ✅ Complete |
| 5 | Database migration | ✅ Complete |
| 6 | RAG indexer integration | ✅ Complete |
| 7 | CLI script | ✅ Complete |
| 8 | S&P 500 data file | ✅ Complete |
| 9 | Unit tests | ✅ Complete (55 tests) |
| 10 | Small-scale test (3 companies) | ✅ Complete (838 chunks indexed) |
| 11 | Full overnight ingestion | 🔲 Not Started |

---

## Overview

Build a checkpointed ingestion pipeline to download, parse, chunk, and index SEC 10-K and 10-Q filings for S&P 500 companies (2 years of data) into the existing RAG infrastructure.

**Key insight:** The codebase already has local embeddings (sentence-transformers, 384 dims) and a mature checkpoint system. This plan leverages existing infrastructure.

---

## Scope

| In Scope | Out of Scope |
|----------|--------------|
| SEC EDGAR 10-K/10-Q download | RAGPack evaluation stages (separate agent) |
| Filing HTML parsing | Earnings transcripts (phase 2) |
| Chunking for RAG | Financial Modeling Prep API (phase 2) |
| Local embeddings (384 dims) | OpenAI embeddings |
| PostgreSQL/pgvector indexing | Azure migration |
| Checkpointed/resumable | Real-time ingestion |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    SEC Filing Ingestion Pipeline                 │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  1. DOWNLOAD (rate-limited, checkpointed)                       │
│     SECEdgarClient → data/sec_filings/downloads/                │
│                                                                  │
│  2. PARSE (10-K/10-Q HTML extraction)                           │
│     FilingParser → sections dict (risk_factors, mda, etc.)      │
│                                                                  │
│  3. CHUNK (semantic boundaries, 4K chars)                       │
│     DocumentChunker → FilingChunk list with metadata            │
│                                                                  │
│  4. EMBED (local sentence-transformers, 384 dims)               │
│     LocalEmbedder → vectors (existing infrastructure)           │
│                                                                  │
│  5. INDEX (bulk COPY protocol)                                  │
│     RAGIndexer → rag_documents table                            │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

---

## Files to Create

```
src/nl2api/ingestion/sec_filings/
├── __init__.py              # Public exports
├── config.py                # SECFilingConfig (pydantic-settings)
├── models.py                # Filing, FilingChunk, FilingCheckpoint
├── client.py                # SECEdgarClient with rate limiting
├── parser.py                # FilingParser (10-K/10-Q HTML)
├── chunker.py               # DocumentChunker
└── indexer.py               # FilingRAGIndexer (wraps existing)

scripts/
└── ingest_sec_filings.py    # CLI entry point

data/tickers/
└── sp500.json               # S&P 500 ticker → CIK mapping

src/common/storage/postgres/migrations/
└── 012_sec_filings.sql      # Filing metadata table
```

---

## Implementation Steps

### Step 1: Data Models and Config (~2 hours)

**File:** `src/nl2api/ingestion/sec_filings/models.py`

```python
@dataclass(frozen=True)
class Filing:
    accession_number: str      # "0001234567-23-000001"
    cik: str                   # 10-digit
    ticker: str | None
    company_name: str
    filing_type: FilingType    # 10-K or 10-Q
    filing_date: datetime
    period_of_report: datetime
    primary_document: str
    filing_url: str

@dataclass
class FilingChunk:
    chunk_id: str
    filing_accession: str
    section: str               # "risk_factors", "mda", etc.
    chunk_index: int
    content: str
    metadata: dict
```

**File:** `src/nl2api/ingestion/sec_filings/config.py`

Follow pattern from `src/nl2api/ingestion/config.py`:
- `SECFilingConfig(BaseSettings)` with `batch_size`, `max_concurrent`, `checkpoint_interval`
- Environment variables: `SEC_USER_AGENT`, `SEC_RATE_LIMIT`

### Step 2: SEC EDGAR Client (~4 hours)

**File:** `src/nl2api/ingestion/sec_filings/client.py`

```python
class SECEdgarClient:
    """Rate-limited client for SEC EDGAR API (10 req/sec)."""

    BASE_URL = "https://data.sec.gov"

    async def get_company_filings(self, cik: str, filing_types: list, after_date: datetime) -> list[Filing]
    async def download_filing(self, filing: Filing, output_dir: Path) -> Path
    async def get_sp500_ciks(self) -> list[dict]  # Load from data/tickers/sp500.json
```

**Key patterns from GLEIF ingestion (`scripts/ingest_gleif.py`):**
- `AsyncRateLimiter` with 100ms between requests
- User-Agent header required by SEC
- Streaming download with `httpx.AsyncClient.stream()`
- Retry with exponential backoff on 429/5xx

### Step 3: Filing Parser (~6 hours)

**File:** `src/nl2api/ingestion/sec_filings/parser.py`

```python
class FilingParser:
    """Parse 10-K/10-Q HTML into sections."""

    SECTION_PATTERNS_10K = {
        "business": r"item\s*1[.\s:]+business",
        "risk_factors": r"item\s*1a[.\s:]+risk\s*factors",
        "mda": r"item\s*7[.\s:]+management",
        # ... more sections
    }

    def parse(self, html_path: Path, filing_type: FilingType) -> dict[str, str]
    def _clean_html(self, html: str) -> str  # Remove XBRL, scripts, styles
    def _extract_section(self, soup, start_pattern, end_patterns) -> str
```

**Libraries:** `beautifulsoup4`, `lxml`

### Step 4: Document Chunker (~3 hours)

**File:** `src/nl2api/ingestion/sec_filings/chunker.py`

```python
class DocumentChunker:
    """Chunk filing sections for RAG."""

    def __init__(self, chunk_size=4000, chunk_overlap=800, min_chunk_size=800):
        ...

    def chunk_filing(self, sections: dict[str, str], filing: Filing) -> list[FilingChunk]
    def _chunk_section(self, text: str, section_name: str) -> list[str]
```

**Strategy:**
1. Respect section boundaries (don't split across sections)
2. Split large sections by paragraphs
3. Split large paragraphs by sentences with overlap
4. Preserve metadata (filing info, section name, chunk index)

### Step 5: Database Migration (~1 hour)

**File:** `src/common/storage/postgres/migrations/012_sec_filings.sql`

```sql
CREATE TABLE IF NOT EXISTS sec_filings (
    accession_number VARCHAR(30) PRIMARY KEY,
    cik VARCHAR(10) NOT NULL,
    ticker VARCHAR(20),
    company_name TEXT NOT NULL,
    filing_type VARCHAR(10) NOT NULL,
    filing_date DATE NOT NULL,
    period_of_report DATE NOT NULL,
    status VARCHAR(20) DEFAULT 'pending',
    chunks_count INTEGER DEFAULT 0,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    indexed_at TIMESTAMPTZ
);

CREATE INDEX idx_sec_filings_cik ON sec_filings(cik);
CREATE INDEX idx_sec_filings_ticker ON sec_filings(ticker);
CREATE INDEX idx_sec_filings_date ON sec_filings(filing_date);
```

### Step 6: RAG Indexer Integration (~4 hours)

**File:** `src/nl2api/ingestion/sec_filings/indexer.py`

```python
class FilingRAGIndexer:
    """Index filing chunks into RAG system."""

    def __init__(self, pool, embedder=None, batch_size=50):
        self._embedder = embedder or create_embedder("local")  # Uses existing LocalEmbedder
        self._rag_indexer = RAGIndexer(pool, embedder=self._embedder)

    async def index_filing_chunks(self, chunks: list[FilingChunk], checkpoint_id: str | None = None) -> list[str]
```

**Integration points:**
- Uses existing `LocalEmbedder` (384 dims, sentence-transformers)
- Uses existing COPY protocol pattern from `RAGIndexer._bulk_insert_field_codes`
- Document type: `"sec_filing"` (add to `DocumentType` enum in `src/nl2api/rag/protocols.py`)

### Step 7: CLI Script (~4 hours)

**File:** `scripts/ingest_sec_filings.py`

Follow pattern from `scripts/ingest_gleif.py`:

```bash
# Full S&P 500 ingestion (2 years)
python scripts/ingest_sec_filings.py

# Specific tickers
python scripts/ingest_sec_filings.py --tickers AAPL,MSFT,GOOGL

# Resume from checkpoint
python scripts/ingest_sec_filings.py --resume

# Dry run (count filings only)
python scripts/ingest_sec_filings.py --dry-run
```

**Workflow:**
1. Load S&P 500 CIKs from `data/tickers/sp500.json`
2. For each company: fetch filing index → filter 10-K/10-Q → download → parse → chunk → index
3. Update checkpoint after each company
4. Report summary statistics

### Step 8: S&P 500 Data File (~2 hours)

**File:** `data/tickers/sp500.json`

```json
{
  "updated_at": "2026-01-22",
  "source": "SEC + Wikipedia",
  "companies": [
    {"ticker": "AAPL", "cik": "0000320193", "name": "Apple Inc."},
    {"ticker": "MSFT", "cik": "0000789019", "name": "Microsoft Corporation"},
    ...
  ]
}
```

Curate manually from SEC company_tickers.json + Wikipedia S&P 500 list.

### Step 9: Unit Tests (~6 hours)

**Files:**
- `tests/unit/nl2api/ingestion/test_sec_client.py` - Mock SEC API responses
- `tests/unit/nl2api/ingestion/test_filing_parser.py` - Real HTML samples
- `tests/unit/nl2api/ingestion/test_chunker.py` - Chunking logic

### Step 10: Integration Test (~4 hours)

**File:** `tests/integration/ingestion/test_sec_ingestion.py`

Test against real PostgreSQL with small sample (3 companies, 1 filing each).

---

## Resource Estimates

| Metric | Estimate |
|--------|----------|
| Companies | 500 (S&P 500) |
| Filings per company | ~10 (2 years × 5 filings/year) |
| Total filings | ~5,000 |
| Total download | ~12.5 GB (avg 2.5 MB/filing) |
| Download time | ~8.5 hours (10 req/sec limit) |
| Chunks per filing | ~75 average |
| Total chunks | ~375,000 |
| Embedding time | ~3 hours (local, 384 dims) |
| Storage (vectors) | ~600 MB |
| Storage (text) | ~3 GB |

**Recommendation:** Run overnight with checkpointing. Can stop/resume safely.

---

## Verification

### Unit Tests
```bash
pytest tests/unit/nl2api/ingestion/test_sec_*.py -v
```

### Integration Test (requires PostgreSQL)
```bash
docker compose up -d
pytest tests/integration/ingestion/test_sec_ingestion.py -v
```

### Manual Verification
```bash
# Dry run to verify API access
python scripts/ingest_sec_filings.py --tickers AAPL --dry-run

# Small test run (3 companies)
python scripts/ingest_sec_filings.py --tickers AAPL,MSFT,GOOGL --years 1

# Verify data in database
psql -U nl2api -d nl2api -c "SELECT COUNT(*) FROM rag_documents WHERE document_type = 'sec_filing';"

# Test retrieval
python -c "
from src.nl2api.rag.retriever import HybridRAGRetriever
# ... test query against SEC filings
"
```

---

## Dependencies

**New packages to add to requirements.txt:**
- `beautifulsoup4>=4.12.0` - HTML parsing
- `lxml>=5.0.0` - Fast HTML parser backend

**Existing packages used:**
- `httpx` - Async HTTP client
- `asyncpg` - PostgreSQL
- `sentence-transformers` - Local embeddings (already configured)

---

## Phase 2: Financial Modeling Prep (Future)

After SEC filings are working, add earnings transcripts:

```python
class FMPClient:
    """Financial Modeling Prep API client (250 free calls/day)."""

    async def get_earnings_transcript(self, ticker: str, year: int, quarter: int) -> str
```

This is out of scope for the current implementation.
