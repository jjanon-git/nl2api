# Plan: SEC EDGAR Filing Ingestion for RAG Evaluation

**Status:** Phase 1 Complete (10-K only), Phase 2 Pending (10-Q + Transcripts)
**Created:** 2026-01-22
**Last Updated:** 2026-02-01 (documented 10-Q gap and earnings transcript requirements)
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
| 11 | Full 10-K ingestion | ✅ Complete (1.4M chunks, 400+ companies) |
| 12 | **10-Q quarterly filings** | 🔲 Not Started (Phase 2) |
| 13 | **Earnings call transcripts** | 🔲 Not Started (Phase 2) |

### Current Index State (2026-02-01)

| Filing Type | Companies | Chunks | Date Range |
|-------------|-----------|--------|------------|
| 10-K (annual) | 400+ | 1,445,154 | FY2023-FY2024 |
| 10-Q (quarterly) | 0 | 0 | - |
| Earnings transcripts | 0 | 0 | - |

**Gap identified:** User queries for "last earnings call" or "recent quarterly results" return annual report data because 10-Q and transcript data is missing.

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

## Phase 2: 10-Q Quarterly Filings

**Priority:** HIGH - Required for temporal relevance on quarterly queries

### Why 10-Q Matters

| User Query | Current Behavior | Expected Behavior |
|------------|------------------|-------------------|
| "Amazon's last quarter results" | Returns FY2024 10-K guidance | Should return Q3 2025 10-Q data |
| "Recent revenue growth" | Annual data only | Quarterly trends |
| "Q2 earnings highlights" | No data / wrong data | Specific quarter data |

### Implementation

The existing pipeline already supports 10-Q - it's just not being ingested:

```bash
# Current (10-K only)
python scripts/ingest_sec_filings.py --filing-types 10-K

# Add 10-Q support
python scripts/ingest_sec_filings.py --filing-types 10-K,10-Q
```

### Resource Estimates (10-Q Addition)

| Metric | Estimate |
|--------|----------|
| Filings per company/year | 4 (quarterly) |
| Total new filings | ~4,000 (500 companies × 2 years × 4 quarters) |
| Download size | ~4 GB (10-Q avg ~1 MB, smaller than 10-K) |
| New chunks | ~200,000 |
| Additional storage | ~1.5 GB |
| Download time | ~7 hours (rate limited) |

### Parser Changes Needed

The `FilingParser` needs 10-Q section patterns (different from 10-K):

```python
SECTION_PATTERNS_10Q = {
    "financial_statements": r"item\s*1[.\s:]+financial\s*statements",
    "mda": r"item\s*2[.\s:]+management",  # Same as 10-K Item 7
    "quantitative_disclosures": r"item\s*3[.\s:]+quantitative",
    "controls": r"item\s*4[.\s:]+controls",
}
```

---

## Phase 3: Earnings Call Transcripts

**Priority:** MEDIUM - Valuable for conversational queries but requires external data source

### Why Transcripts Matter

| User Query | 10-K/10-Q Response | Transcript Response |
|------------|-------------------|---------------------|
| "What did the CEO say about AI?" | Legal boilerplate | Direct quotes, tone, emphasis |
| "Management outlook on margins" | Risk factors section | Forward-looking commentary |
| "Analyst concerns" | Not available | Q&A section with analyst questions |

### Data Sources

| Source | Cost | Coverage | Quality |
|--------|------|----------|---------|
| **Financial Modeling Prep** | Free tier: 250 calls/day | Good (S&P 500) | Structured JSON |
| **Seeking Alpha** | Scraping TOS issues | Excellent | Raw text |
| **FactSet** | Enterprise pricing | Comprehensive | Professional |
| **Bloomberg** | Terminal required | Best | Professional |
| **Earnings Call APIs** | ~$50-200/mo | Varies | Structured |

### Recommended: Financial Modeling Prep

```python
class FMPClient:
    """Financial Modeling Prep API client."""

    BASE_URL = "https://financialmodelingprep.com/api/v3"

    async def get_earnings_transcript(
        self,
        ticker: str,
        year: int,
        quarter: int
    ) -> EarningsTranscript:
        """
        Fetch earnings call transcript.

        Free tier: 250 calls/day
        Returns: Full transcript with speaker labels
        """
        ...

    async def list_available_transcripts(
        self,
        ticker: str
    ) -> list[TranscriptMetadata]:
        """List all available transcripts for a ticker."""
        ...
```

### Transcript Chunking Strategy

Earnings calls have different structure than SEC filings:

```
1. Opening remarks (CEO/CFO prepared statements)
2. Financial highlights
3. Business segment updates
4. Guidance
5. Q&A session (analyst questions + management answers)
```

**Recommended approach:**
- Chunk by speaker turn (preserve who said what)
- Keep Q&A pairs together (question + answer as one chunk)
- Tag chunks with speaker role (CEO, CFO, Analyst)
- Include timestamp metadata for ordering

### Resource Estimates (Transcripts)

| Metric | Estimate |
|--------|----------|
| Transcripts per company/year | 4 (quarterly) |
| Total transcripts | ~4,000 |
| Avg transcript length | ~15,000 words |
| Chunks per transcript | ~30 |
| Total new chunks | ~120,000 |
| API calls needed | ~8,000 (list + fetch) |
| Time to ingest | ~32 days at 250/day free tier |

**Note:** Free tier would take ~1 month. Paid tier ($29/mo for 750/day) reduces to ~11 days.

### Schema Addition

```sql
-- Add to rag_documents metadata for transcripts
{
    "document_type": "earnings_transcript",
    "ticker": "AMZN",
    "company_name": "Amazon.com Inc.",
    "fiscal_year": 2025,
    "fiscal_quarter": 3,
    "call_date": "2025-10-26",
    "speaker": "Andy Jassy",
    "speaker_role": "CEO",
    "section": "prepared_remarks",  -- or "qa"
    "chunk_index": 5
}
```

---

## Phase 2/3 Prerequisites

Before starting Phase 2 or 3:

1. **Disk space:** Current index is ~3.6 GB. Adding 10-Q + transcripts adds ~3 GB.
2. **API keys:** FMP requires free account registration
3. **Rate limit planning:** SEC (10 req/sec) + FMP (250/day free) need coordination
