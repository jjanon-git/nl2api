# Entity Resolution Improvement Plan

**Status:** ✅ Completed (design superseded by nl2api-01)
**Priority:** P0.2 (Critical)
**Author:** Mostly Claude, with some minor assistance from Sid
**Last Updated:** 2026-01-21

---

## Executive Summary

The entity resolution system has a solid foundation with multi-level caching, fuzzy matching, and external API integration. However, analysis reveals significant gaps in **coverage** (109 companies in a JSON file), **accuracy** (false positives from extraction), and **architecture** (no database backing).

This plan proposes a comprehensive overhaul to achieve:
- **2M+ entity coverage** via canonical database
- **95%+ resolution accuracy** for financial queries
- **<5ms P95 latency** for cached entities
- **Private company support** (P2)

---

## Current State Analysis

### Critical Gap: No Database Backing

```
Current Architecture (PROBLEMATIC):
┌─────────────────────────────────────────────────────────────────┐
│  Entity Resolution                                               │
├─────────────────────────────────────────────────────────────────┤
│  src/nl2api/resolution/data/company_mappings.json               │
│  ├── 109 companies (static JSON file)                           │
│  ├── 178 aliases                                                │
│  ├── 109 ticker mappings                                        │
│  └── NO database table, NO scalability                          │
├─────────────────────────────────────────────────────────────────┤
│  OpenFIGI API (fallback)                                        │
│  └── 250ms+ latency per call                                    │
└─────────────────────────────────────────────────────────────────┘
```

### What Exists (Implemented)

| Component | Status | Notes |
|-----------|--------|-------|
| Static mappings | 109 companies | JSON file, ~20% S&P 500 coverage |
| Ticker resolution | 109 tickers | Direct ticker → RIC |
| Fuzzy matching | rapidfuzz | 75-85 threshold, handles typos |
| Multi-level caching | L1 (memory) + L2 (Redis) | 0.03ms cached, 0.3ms cold |
| Circuit breaker | Implemented | Protects against API failures |
| OpenFIGI integration | Working | 250ms latency per call |
| Test coverage | 42 tests | Normalization, fuzzy, extraction |

### Coverage Gaps (Current)

```
Current Coverage:
├── US Public Companies: 100 companies
│   ├── S&P 500: ~20% coverage (100/500)
│   ├── NASDAQ 100: ~60% coverage (est. 60/100)
│   ├── Russell 2000: <1% coverage
│   └── Total US public: ~5,000 companies exist
├── International Public: 9 companies
│   ├── FTSE 100: ~3% coverage
│   ├── DAX 40: 0% coverage
│   ├── Nikkei 225: ~1% coverage
│   └── Total global public: ~60,000 companies exist
├── Private Companies: 0 coverage
│   └── Total US private: ~30M+ companies
└── Total known entities: 280 (including aliases)
```

---

## Target Architecture

```
Target Architecture (DATABASE-BACKED):
┌─────────────────────────────────────────────────────────────────┐
│  Entity Resolution System                                        │
├─────────────────────────────────────────────────────────────────┤
│  PostgreSQL: entities table                                      │
│  ├── 2M+ entities (public companies globally)                   │
│  ├── Full-text search (GIN index)                               │
│  ├── Fuzzy matching (pg_trgm extension)                         │
│  └── Vector embeddings for semantic search (pgvector)           │
├─────────────────────────────────────────────────────────────────┤
│  Data Sources (Ingestion Pipeline)                               │
│  ├── GLEIF (LEI): 2M+ legal entities                            │
│  ├── SEC EDGAR: All US public filers                            │
│  ├── OpenFIGI: Security identifier resolution                   │
│  ├── Refinitiv PermID: RIC mappings                             │
│  └── OpenCorporates: Private companies (P2)                     │
├─────────────────────────────────────────────────────────────────┤
│  Caching Layer                                                   │
│  ├── L1: In-memory (hot entities)                               │
│  ├── L2: Redis (warm entities)                                  │
│  └── L3: PostgreSQL (cold storage)                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Goals

| Goal | Metric | Current | Target |
|------|--------|---------|--------|
| Public company coverage | Companies in DB | 109 | 2M+ (via GLEIF) |
| US public coverage | % of all US public | ~2% | 100% |
| Global public coverage | % of global public | <0.2% | 95%+ |
| Resolution accuracy | Benchmark test | ~70% | 95%+ |
| False positive rate | Incorrect extractions | ~10% | <2% |
| P95 latency (cached) | Resolution time | 2ms | <5ms |
| P95 latency (DB lookup) | Resolution time | N/A | <20ms |
| Private companies (P2) | Coverage | 0 | 100K+ |

---

## Data Sources Analysis

### Tier 1: Free, High-Quality Sources (P0)

#### 1. GLEIF - Legal Entity Identifier (LEI)

**Coverage:** 2M+ legal entities globally
**Format:** CSV/JSON bulk download (daily updates)
**Cost:** Free
**Data Quality:** Excellent (ISO 17442 standard)

```
GLEIF Data Structure:
├── LEI (20-char identifier)
├── Legal Name
├── Legal Address (Country, City, Postal)
├── Headquarters Address
├── Registration Status (ISSUED, LAPSED, etc.)
├── Entity Category (GENERAL, FUND, BRANCH)
├── Registration Date
└── Next Renewal Date

Key Advantage:
- Standardized global identifier
- Links to parent/child relationships
- Updated daily
- Free bulk download: https://www.gleif.org/en/lei-data/gleif-golden-copy
```

#### 2. SEC EDGAR - US Public Companies

**Coverage:** ~8,500 active US filers
**Format:** JSON API + bulk files
**Cost:** Free
**Data Quality:** Authoritative (SEC regulated)

```
SEC EDGAR Data Structure:
├── CIK (Central Index Key)
├── Company Name
├── Ticker Symbol(s)
├── Exchange
├── SIC Code (Industry)
├── State of Incorporation
├── Fiscal Year End
└── Filing History

Key API Endpoints:
- Company search: https://efts.sec.gov/LATEST/search-index
- Company facts: https://data.sec.gov/api/xbrl/companyfacts/CIK{cik}.json
- Ticker-CIK mapping: https://www.sec.gov/files/company_tickers.json
```

#### 3. OpenFIGI (Already Integrated)

**Coverage:** Global securities
**Format:** REST API
**Cost:** Free (1,000 req/day), paid for bulk
**Data Quality:** Good (Bloomberg-backed)

```
Already integrated at src/nl2api/resolution/openfigi.py
Can be used to enrich entities with FIGI identifiers
```

### Tier 2: Commercial Data Sources (P1)

#### 4. Refinitiv PermID

**Coverage:** 200M+ entities
**Format:** REST API
**Cost:** Part of Refinitiv license
**Data Quality:** Excellent (source of RICs)

```
PermID Provides:
├── Permanent Identifier
├── Organization Name(s)
├── RIC mappings (direct!)
├── Industry classification
├── Geographic presence
└── Corporate hierarchy
```

#### 5. EODHD (End of Day Historical Data)

**Coverage:** 150K+ securities globally
**Format:** REST API
**Cost:** $80/month (Fundamentals)
**Data Quality:** Good

### Tier 3: Private Companies (P2)

#### 6. OpenCorporates

**Coverage:** 200M+ companies worldwide
**Format:** REST API
**Cost:** Free tier (limited), paid for bulk
**Data Quality:** Varies by jurisdiction

```
Includes:
├── Company registration number
├── Jurisdiction
├── Company type
├── Status (active, dissolved, etc.)
├── Registered address
└── Officers (some jurisdictions)

Limitation: No stock identifiers (private companies)
```

#### 7. LinkedIn Company Data (via DataAxle/ZoomInfo)

**Coverage:** 100M+ companies
**Format:** API integration
**Cost:** High ($$$)
**Data Quality:** Good for private companies

---

## Database Schema

### Migration: 007_entities.sql

```sql
-- Entity Resolution Schema
-- Canonical database for company/entity resolution

-- Enable trigram extension for fuzzy matching
CREATE EXTENSION IF NOT EXISTS pg_trgm;

-- =============================================================================
-- Entities Table (Canonical Store)
-- =============================================================================

CREATE TABLE IF NOT EXISTS entities (
    -- Primary key (UUID for flexibility)
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),

    -- Core identifiers
    lei VARCHAR(20) UNIQUE,              -- Legal Entity Identifier (GLEIF)
    cik VARCHAR(10),                     -- SEC Central Index Key
    permid VARCHAR(20),                  -- Refinitiv PermID
    figi VARCHAR(12),                    -- OpenFIGI identifier

    -- Primary identifiers for resolution
    primary_name TEXT NOT NULL,          -- Official legal name
    display_name TEXT,                   -- Common display name

    -- Stock identifiers (for public companies)
    ticker VARCHAR(20),                  -- Primary ticker symbol
    ric VARCHAR(50),                     -- Reuters Instrument Code
    exchange VARCHAR(20),                -- Primary exchange (NYSE, NASDAQ, LSE, etc.)

    -- Classification
    entity_type VARCHAR(50) NOT NULL DEFAULT 'company',  -- company, fund, government, etc.
    entity_status VARCHAR(20) NOT NULL DEFAULT 'active', -- active, inactive, merged, dissolved
    is_public BOOLEAN NOT NULL DEFAULT false,

    -- Geographic
    country_code CHAR(2),                -- ISO 3166-1 alpha-2
    region VARCHAR(100),                 -- State/Province
    city VARCHAR(100),

    -- Industry
    sic_code VARCHAR(4),                 -- Standard Industrial Classification
    naics_code VARCHAR(6),               -- North American Industry Classification
    gics_sector VARCHAR(100),            -- Global Industry Classification Standard

    -- Corporate hierarchy
    parent_entity_id UUID REFERENCES entities(id),
    ultimate_parent_id UUID REFERENCES entities(id),

    -- Data quality
    data_source VARCHAR(50) NOT NULL,    -- gleif, sec_edgar, permid, manual
    confidence_score FLOAT DEFAULT 1.0,
    last_verified_at TIMESTAMPTZ,

    -- Vector embedding for semantic search
    name_embedding vector(1536),

    -- Timestamps
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- =============================================================================
-- Entity Aliases Table (for fuzzy matching)
-- =============================================================================

CREATE TABLE IF NOT EXISTS entity_aliases (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    entity_id UUID NOT NULL REFERENCES entities(id) ON DELETE CASCADE,

    alias TEXT NOT NULL,                 -- Normalized alias (lowercase, stripped)
    alias_type VARCHAR(50) NOT NULL,     -- ticker, legal_name, trade_name, abbreviation
    is_primary BOOLEAN DEFAULT false,

    -- Dedup
    UNIQUE(entity_id, alias)
);

-- =============================================================================
-- Indexes for Entity Resolution
-- =============================================================================

-- Primary lookups
CREATE INDEX IF NOT EXISTS idx_entities_lei ON entities(lei);
CREATE INDEX IF NOT EXISTS idx_entities_cik ON entities(cik);
CREATE INDEX IF NOT EXISTS idx_entities_ticker ON entities(ticker);
CREATE INDEX IF NOT EXISTS idx_entities_ric ON entities(ric);
CREATE INDEX IF NOT EXISTS idx_entities_permid ON entities(permid);

-- Full-text search on names
CREATE INDEX IF NOT EXISTS idx_entities_primary_name_fts ON entities
    USING GIN(to_tsvector('english', primary_name));
CREATE INDEX IF NOT EXISTS idx_entities_display_name_fts ON entities
    USING GIN(to_tsvector('english', COALESCE(display_name, '')));

-- Trigram index for fuzzy matching
CREATE INDEX IF NOT EXISTS idx_entities_primary_name_trgm ON entities
    USING GIN(lower(primary_name) gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_entities_display_name_trgm ON entities
    USING GIN(lower(COALESCE(display_name, '')) gin_trgm_ops);

-- Alias lookups (critical for resolution performance)
CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias ON entity_aliases(lower(alias));
CREATE INDEX IF NOT EXISTS idx_entity_aliases_alias_trgm ON entity_aliases
    USING GIN(lower(alias) gin_trgm_ops);
CREATE INDEX IF NOT EXISTS idx_entity_aliases_entity ON entity_aliases(entity_id);

-- Classification filters
CREATE INDEX IF NOT EXISTS idx_entities_country ON entities(country_code);
CREATE INDEX IF NOT EXISTS idx_entities_exchange ON entities(exchange);
CREATE INDEX IF NOT EXISTS idx_entities_is_public ON entities(is_public);
CREATE INDEX IF NOT EXISTS idx_entities_status ON entities(entity_status);

-- Vector similarity (for semantic entity matching)
CREATE INDEX IF NOT EXISTS idx_entities_embedding ON entities
    USING ivfflat (name_embedding vector_cosine_ops) WITH (lists = 1000);

-- =============================================================================
-- Helper Functions
-- =============================================================================

-- Fast entity lookup by alias (with fuzzy matching fallback)
CREATE OR REPLACE FUNCTION resolve_entity(
    p_query TEXT,
    p_fuzzy_threshold FLOAT DEFAULT 0.3
)
RETURNS TABLE (
    entity_id UUID,
    primary_name TEXT,
    ric VARCHAR(50),
    ticker VARCHAR(20),
    match_type TEXT,
    similarity FLOAT
) AS $$
BEGIN
    -- 1. Exact alias match (fastest)
    RETURN QUERY
    SELECT e.id, e.primary_name, e.ric, e.ticker, 'exact'::TEXT, 1.0::FLOAT
    FROM entities e
    JOIN entity_aliases a ON e.id = a.entity_id
    WHERE lower(a.alias) = lower(p_query)
    LIMIT 1;

    IF FOUND THEN RETURN; END IF;

    -- 2. Ticker match
    RETURN QUERY
    SELECT e.id, e.primary_name, e.ric, e.ticker, 'ticker'::TEXT, 1.0::FLOAT
    FROM entities e
    WHERE upper(e.ticker) = upper(p_query)
    LIMIT 1;

    IF FOUND THEN RETURN; END IF;

    -- 3. Fuzzy match on aliases
    RETURN QUERY
    SELECT e.id, e.primary_name, e.ric, e.ticker, 'fuzzy'::TEXT,
           similarity(lower(a.alias), lower(p_query))::FLOAT AS sim
    FROM entities e
    JOIN entity_aliases a ON e.id = a.entity_id
    WHERE similarity(lower(a.alias), lower(p_query)) > p_fuzzy_threshold
    ORDER BY sim DESC
    LIMIT 5;

END;
$$ LANGUAGE plpgsql;

-- Trigger for updated_at
CREATE TRIGGER update_entities_updated_at
    BEFORE UPDATE ON entities
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();
```

---

## Data Ingestion Pipeline

### Phase 1: GLEIF Bulk Import (P0)

**File:** `scripts/ingest_gleif.py`

```python
"""
GLEIF LEI Data Ingestion Pipeline

Downloads and imports 2M+ legal entities from GLEIF golden copy.
Runs as a scheduled job (daily/weekly).

Data source: https://www.gleif.org/en/lei-data/gleif-golden-copy
"""

import asyncio
import csv
import gzip
import httpx
from pathlib import Path

GLEIF_DOWNLOAD_URL = "https://leidata-preview.gleif.org/storage/golden-copy-files/2024/01/20/lei2.csv.gz"

async def download_gleif_data(output_path: Path) -> None:
    """Download GLEIF golden copy (compressed CSV)."""
    async with httpx.AsyncClient() as client:
        response = await client.get(GLEIF_DOWNLOAD_URL, follow_redirects=True)
        output_path.write_bytes(response.content)

async def import_gleif_entities(pool, csv_path: Path) -> int:
    """Bulk import GLEIF entities using COPY protocol."""
    imported = 0

    with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        async with pool.acquire() as conn:
            # Use COPY for bulk insert (fastest)
            await conn.copy_records_to_table(
                'entities',
                records=transform_gleif_records(reader),
                columns=['lei', 'primary_name', 'country_code', 'entity_status',
                         'entity_type', 'data_source']
            )

    return imported

def transform_gleif_records(reader):
    """Transform GLEIF CSV to entity records."""
    for row in reader:
        if row.get('Entity.EntityStatus') != 'ACTIVE':
            continue

        yield (
            row['LEI'],                           # lei
            row['Entity.LegalName'],              # primary_name
            row['Entity.LegalAddress.Country'],   # country_code
            'active',                             # entity_status
            row.get('Entity.EntityCategory', 'company').lower(),  # entity_type
            'gleif',                              # data_source
        )
```

### Phase 2: SEC EDGAR Import (P0)

**File:** `scripts/ingest_sec_edgar.py`

```python
"""
SEC EDGAR Company Data Ingestion

Imports all US public company filers with CIK-ticker mappings.
Updates RIC mappings for US exchanges.
"""

import httpx

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
SEC_EXCHANGES_URL = "https://www.sec.gov/files/company_tickers_exchange.json"

async def import_sec_companies(pool) -> int:
    """Import SEC EDGAR company data."""
    async with httpx.AsyncClient() as client:
        # Get ticker mappings
        resp = await client.get(SEC_TICKERS_URL)
        tickers = resp.json()

        # Get exchange mappings
        resp = await client.get(SEC_EXCHANGES_URL)
        exchanges = resp.json()

    async with pool.acquire() as conn:
        for entry in tickers.values():
            cik = str(entry['cik_str']).zfill(10)
            ticker = entry['ticker']
            name = entry['title']
            exchange = exchanges.get(ticker, {}).get('exchange', 'O')

            # Generate RIC from ticker + exchange
            ric = generate_ric(ticker, exchange)

            await conn.execute("""
                INSERT INTO entities (cik, primary_name, ticker, ric, exchange,
                                      is_public, country_code, data_source)
                VALUES ($1, $2, $3, $4, $5, true, 'US', 'sec_edgar')
                ON CONFLICT (cik) DO UPDATE SET
                    ticker = EXCLUDED.ticker,
                    ric = EXCLUDED.ric,
                    updated_at = NOW()
            """, cik, name, ticker, ric, exchange)

    return len(tickers)

def generate_ric(ticker: str, exchange: str) -> str:
    """Generate RIC from ticker and exchange."""
    suffix_map = {
        'NYSE': '.N',
        'NASDAQ': '.O',
        'AMEX': '.A',
        'BATS': '.Z',
        'N': '.N',
        'O': '.O',
        'A': '.A',
    }
    return f"{ticker}{suffix_map.get(exchange, '.O')}"
```

### Phase 3: Alias Generation (P0)

**File:** `scripts/generate_aliases.py`

```python
"""
Generate entity aliases for fuzzy matching.

Creates normalized aliases from:
- Legal name variations
- Trade names
- Ticker symbols
- Common abbreviations
"""

import re

COMPANY_SUFFIXES = [
    'Inc', 'Inc.', 'Incorporated',
    'Corp', 'Corp.', 'Corporation',
    'Ltd', 'Ltd.', 'Limited',
    'LLC', 'L.L.C.',
    'PLC', 'P.L.C.',
    'Co', 'Co.', 'Company',
    'Holdings', 'Group', 'International',
    'SA', 'S.A.', 'AG', 'GmbH', 'NV', 'BV',
]

async def generate_aliases_for_entity(conn, entity_id: str, name: str, ticker: str = None):
    """Generate and insert aliases for an entity."""
    aliases = set()

    # Original name (normalized)
    aliases.add(normalize(name))

    # Without suffixes
    for suffix in COMPANY_SUFFIXES:
        pattern = rf'\s*[,&]?\s*{re.escape(suffix)}\.?$'
        stripped = re.sub(pattern, '', name, flags=re.I).strip()
        if stripped != name:
            aliases.add(normalize(stripped))

    # Ticker as alias
    if ticker:
        aliases.add(normalize(ticker))

    # Without punctuation
    clean = re.sub(r'[^\w\s]', '', name)
    aliases.add(normalize(clean))

    # Common abbreviations (e.g., "International Business Machines" -> "IBM")
    if len(name.split()) >= 3:
        initials = ''.join(word[0] for word in name.split() if word[0].isupper())
        if len(initials) >= 2:
            aliases.add(normalize(initials))

    # Insert aliases
    for alias in aliases:
        if len(alias) >= 2:
            await conn.execute("""
                INSERT INTO entity_aliases (entity_id, alias, alias_type)
                VALUES ($1, $2, 'generated')
                ON CONFLICT (entity_id, alias) DO NOTHING
            """, entity_id, alias)

def normalize(s: str) -> str:
    """Normalize string for matching."""
    return s.lower().strip()
```

---

## Resolver Updates

### Updated Entity Resolver

**File:** `src/nl2api/resolution/resolver.py` (modifications)

```python
class ExternalEntityResolver:
    """
    Entity resolver using PostgreSQL canonical database.

    Resolution order:
    1. L1 Cache (in-memory) - 0.03ms
    2. L2 Cache (Redis) - 1ms
    3. L3 Database (PostgreSQL) - 5-20ms
    4. External API (OpenFIGI) - 250ms (fallback only)
    """

    def __init__(
        self,
        db_pool: asyncpg.Pool = None,  # NEW: Database connection
        ...
    ):
        self._db_pool = db_pool
        ...

    async def resolve_single(self, entity: str, ...) -> ResolvedEntity | None:
        # ... L1/L2 cache checks ...

        # L3: Database lookup (NEW)
        if self._db_pool:
            result = await self._resolve_from_db(entity)
            if result:
                await self._cache_result(normalized, result)
                return result

        # L4: Fallback to existing static mappings + OpenFIGI
        ...

    async def _resolve_from_db(self, entity: str) -> ResolvedEntity | None:
        """Resolve entity from PostgreSQL database."""
        async with self._db_pool.acquire() as conn:
            rows = await conn.fetch("""
                SELECT entity_id, primary_name, ric, ticker, match_type, similarity
                FROM resolve_entity($1, $2)
                LIMIT 1
            """, entity, self._fuzzy_threshold / 100)

            if rows:
                row = rows[0]
                return ResolvedEntity(
                    original=entity,
                    identifier=row['ric'] or f"{row['ticker']}.O",
                    entity_type='company',
                    confidence=row['similarity'],
                )

        return None
```

---

## Implementation Phases

### Phase 0: Database Schema (Week 1)

| Task | Effort | Owner |
|------|--------|-------|
| Create migration 007_entities.sql | 1 day | - |
| Add pg_trgm extension to docker-compose | 0.5 day | - |
| Create EntityRepository protocol | 0.5 day | - |
| Implement PostgresEntityRepository | 1 day | - |
| Unit tests for repository | 1 day | - |

### Phase 1: GLEIF Import (Week 2)

| Task | Effort | Owner |
|------|--------|-------|
| GLEIF download script | 1 day | - |
| GLEIF parsing + transformation | 1 day | - |
| Bulk import with COPY protocol | 1 day | - |
| Alias generation for GLEIF entities | 1 day | - |
| Integration tests | 1 day | - |

**Outcome:** 2M+ entities in database

### Phase 2: SEC EDGAR Import (Week 2)

| Task | Effort | Owner |
|------|--------|-------|
| SEC ticker/CIK mapping import | 0.5 day | - |
| RIC generation for US companies | 0.5 day | - |
| Merge with GLEIF data (dedup by LEI/CIK) | 1 day | - |

**Outcome:** All US public companies with RICs

### Phase 3: Resolver Migration (Week 3)

| Task | Effort | Owner |
|------|--------|-------|
| Update ExternalEntityResolver for DB | 2 days | - |
| Fallback to JSON mappings (backward compat) | 0.5 day | - |
| Performance benchmarking | 1 day | - |
| Update all unit tests | 1 day | - |

**Outcome:** Resolver uses database as primary source

### Phase 4: Accuracy & Quality (Week 4)

| Task | Effort | Owner |
|------|--------|-------|
| False positive reduction (Phase 1 of original plan) | 3 days | - |
| Accuracy benchmark suite | 2 days | - |

### Phase 5: Private Companies (P2 - Future)

| Task | Effort | Notes |
|------|--------|-------|
| OpenCorporates integration | 1 week | Free tier limited |
| Dun & Bradstreet evaluation | 1 week | Commercial |
| LinkedIn Company API evaluation | 1 week | Commercial |

---

## Success Criteria

| Criterion | Measurement | Target |
|-----------|-------------|--------|
| Total entities in DB | COUNT(*) | 2M+ |
| US public company coverage | vs SEC filers | 100% |
| Global public company coverage | vs major indices | 95%+ |
| Resolution accuracy | Benchmark test | 95%+ |
| False positive rate | Benchmark test | <2% |
| P95 latency (DB lookup) | Performance test | <20ms |
| P95 latency (cached) | Performance test | <5ms |
| Test coverage | pytest-cov | 95%+ |

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| GLEIF data quality issues | Medium | Validate LEI format, filter inactive |
| Database size (2M+ rows) | Low | Proper indexing, partitioning if needed |
| RIC mapping accuracy | High | Cross-reference with OpenFIGI, manual review |
| Migration breaks existing tests | High | Maintain JSON fallback, incremental rollout |
| Performance regression | Medium | Benchmark before/after, caching layers |

---

## Files to Create/Modify

### New Files

| File | Purpose | Phase |
|------|---------|-------|
| `src/common/storage/postgres/migrations/007_entities.sql` | Database schema | P0 |
| `src/common/storage/postgres/entity_repo.py` | PostgresEntityRepository | P0 |
| `scripts/ingest_gleif.py` | GLEIF bulk import (~2M entities) | P1 |
| `scripts/ingest_sec_edgar.py` | SEC EDGAR import (~8.5K US filers) | P1 |
| `scripts/generate_aliases.py` | Alias generation for fuzzy matching | P1 |
| `scripts/refresh_entities.py` | Manual refresh trigger (optional) | P1 |
| `tests/unit/common/storage/test_entity_repo.py` | Repository unit tests | P0 |
| `tests/accuracy/resolution/test_entity_accuracy.py` | Accuracy benchmark suite | P4 |

### Modified Files

| File | Changes | Phase |
|------|---------|-------|
| `src/nl2api/resolution/resolver.py` | Add DB lookup in resolution chain, fallback to JSON | P3 |
| `docker-compose.yml` | Add pg_trgm extension | P0 |
| `src/common/storage/postgres/__init__.py` | Export EntityRepository | P0 |
| `src/nl2api/config.py` | Add entity refresh config options | P3 |

### Not In Scope (P2 - Backlog)

| File | Purpose |
|------|---------|
| `scripts/ingest_opencorporates.py` | Private company ingestion |
| `src/nl2api/resolution/private_resolver.py` | Private company resolution logic |

---

## Appendix: Data Source Comparison

| Source | Coverage | Cost | Update Freq | RIC Support |
|--------|----------|------|-------------|-------------|
| GLEIF | 2M+ entities | Free | Daily | No (LEI only) |
| SEC EDGAR | 8.5K US filers | Free | Daily | Via mapping |
| OpenFIGI | Global securities | Free/Paid | Real-time | Via FIGI |
| Refinitiv PermID | 200M+ entities | Licensed | Real-time | Direct |
| OpenCorporates | 200M+ companies | Free/Paid | Varies | No |

---

## Design Decisions

### 1. Database Hosting

**Decision:** Same PostgreSQL instance as other tables.

**Rationale:** Keep infrastructure simple until scale requires separation. The entities table will have ~2M rows which PostgreSQL handles easily. Migration to a separate instance is straightforward if needed later (just change connection string).

### 2. Data Refresh Strategy

**Decision:** One-off initial ingestion with configurable refresh interval (disabled by default).

**Rationale:**
- Entity data changes slowly (new companies, delistings, mergers)
- Daily sync is overkill for most use cases
- Provide `ENTITY_REFRESH_ENABLED=false` and `ENTITY_REFRESH_INTERVAL_DAYS=30` config options
- Manual refresh via `scripts/refresh_entities.py` for on-demand updates

### 3. Private Companies (P2)

**Decision:** OpenCorporates (free tier) as first evaluation target, but deferred to P2.

**Rationale:**
- Free tier allows evaluation without commercial commitment
- 200M+ companies provides good coverage
- No stock identifiers (expected for private companies)
- **Status:** Added to backlog, not in current implementation scope

### 4. RIC Validation

**Decision:** Validate RICs via OpenFIGI during ingestion/update, not at inference time.

**Rationale:**
- Ingestion is batch operation - can afford 250ms per entity
- Inference must be fast (<20ms) - no external API calls
- Store validated RIC in database, trust it at query time
- Flag entities with unvalidated RICs for later batch validation

---

## Directory Structure

### Design Principle: Separation of Concerns

The entity resolution system spans multiple concerns. Rather than creating a monolithic `entities/` module, we follow the existing codebase patterns:

```
src/
├── nl2api/resolution/           # BUSINESS LOGIC: How to resolve entities
│   ├── resolver.py              # ExternalEntityResolver (orchestrates resolution)
│   ├── mappings.py              # Fallback JSON mappings (backward compat)
│   ├── openfigi.py              # OpenFIGI client (external API)
│   └── protocols.py             # ResolvedEntity dataclass
│
├── common/storage/postgres/     # DATA ACCESS: How to store/retrieve entities
│   ├── entity_repo.py           # PostgresEntityRepository (NEW)
│   └── migrations/
│       └── 007_entities.sql     # Entity schema (NEW)

scripts/                         # OPERATIONAL: One-off data tasks
├── ingest_gleif.py              # GLEIF bulk import (NEW)
├── ingest_sec_edgar.py          # SEC EDGAR import (NEW)
├── generate_aliases.py          # Alias generation (NEW)
└── refresh_entities.py          # Manual refresh trigger (NEW)

tests/
├── unit/
│   ├── nl2api/resolution/       # Resolution logic tests
│   └── storage/                 # Repository tests
│       └── test_entity_repo.py  # NEW
└── accuracy/
    └── resolution/              # Accuracy benchmarks (NEW)
```

### Why This Structure?

| Concern | Location | Rationale |
|---------|----------|-----------|
| Resolution logic | `src/nl2api/resolution/` | Business logic stays with NL2API domain |
| Data access | `src/common/storage/postgres/` | Follows existing repository pattern (TestCaseRepo, ScorecardRepo) |
| Ingestion scripts | `scripts/` | One-off operational tasks, not runtime code |
| Database schema | `migrations/` | Consistent with existing migrations |

### Future Refactoring

If entity management grows significantly (e.g., entity lifecycle tracking, merger/acquisition handling, real-time updates), consider:
- Dedicated `src/entities/` module with its own domain model
- Event-driven entity updates via message queue
- Separate microservice for entity resolution

For now, the simpler structure is preferred.

---

## Phase 1: Data Ingestion - Detailed Implementation Plan

### Design Goals

The ingestion pipeline must be:

| Goal | Requirement |
|------|-------------|
| **Reliable** | Resume from failure, handle bad records gracefully |
| **Scalable** | Process 2M+ records without memory issues |
| **Repeatable** | Same input = same output, safe to re-run |
| **Observable** | Clear progress, logging, metrics |

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Entity Ingestion Pipeline                             │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Download   │───▶│   Validate   │───▶│  Transform   │───▶│   Load    │ │
│  │   (Stream)   │    │   (Filter)   │    │  (Normalize) │    │  (COPY)   │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│         │                   │                   │                   │       │
│         ▼                   ▼                   ▼                   ▼       │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      Checkpoint Store (JSON)                          │  │
│  │  { "source": "gleif", "last_offset": 1500000, "state": "loading" }   │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐  │
│  │                      Progress Reporter                                │  │
│  │  [████████████████████░░░░░░░░░░] 65% (1.3M / 2M) ETA: 5m            │  │
│  └──────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Reliability Patterns

#### 1. Checkpointing (Resume from Failure)

```python
@dataclass
class IngestionCheckpoint:
    """Tracks ingestion progress for resume capability."""
    source: str                    # gleif, sec_edgar
    started_at: datetime
    last_offset: int               # Rows processed
    last_entity_id: str | None     # Last successfully imported entity
    state: str                     # downloading, validating, loading, complete, failed
    error_message: str | None
    stats: dict                    # Counts, timings

CHECKPOINT_FILE = "data/checkpoints/ingestion_{source}.json"

class CheckpointManager:
    """Manages ingestion checkpoints for reliability."""

    def save(self, checkpoint: IngestionCheckpoint) -> None:
        """Atomically save checkpoint (write to temp, then rename)."""
        temp_path = f"{self.path}.tmp"
        with open(temp_path, 'w') as f:
            json.dump(asdict(checkpoint), f)
        os.rename(temp_path, self.path)  # Atomic on POSIX

    def load(self) -> IngestionCheckpoint | None:
        """Load existing checkpoint if present."""
        if not os.path.exists(self.path):
            return None
        with open(self.path) as f:
            return IngestionCheckpoint(**json.load(f))

    def should_resume(self) -> bool:
        """Check if we should resume from checkpoint."""
        cp = self.load()
        return cp is not None and cp.state not in ('complete', 'failed')
```

#### 2. Idempotency (Safe to Re-run)

```sql
-- Database uses ON CONFLICT for upsert - re-running is safe
INSERT INTO entities (lei, primary_name, ...)
VALUES ($1, $2, ...)
ON CONFLICT (lei) DO UPDATE SET
    primary_name = EXCLUDED.primary_name,
    updated_at = NOW()
WHERE entities.updated_at < EXCLUDED.updated_at;  -- Only update if newer

-- Aliases use ON CONFLICT DO NOTHING - duplicates ignored
INSERT INTO entity_aliases (entity_id, alias, alias_type)
VALUES ($1, $2, $3)
ON CONFLICT (entity_id, alias) DO NOTHING;
```

#### 3. Validation (Data Quality)

```python
@dataclass
class ValidationResult:
    is_valid: bool
    errors: list[str]
    warnings: list[str]

class EntityValidator:
    """Validates entity records before import."""

    def validate(self, record: dict) -> ValidationResult:
        errors = []
        warnings = []

        # Required fields
        if not record.get('primary_name'):
            errors.append("Missing primary_name")

        # LEI format (20 alphanumeric chars)
        lei = record.get('lei')
        if lei and not re.match(r'^[A-Z0-9]{20}$', lei):
            errors.append(f"Invalid LEI format: {lei}")

        # Country code (ISO 3166-1 alpha-2)
        country = record.get('country_code')
        if country and country not in VALID_COUNTRY_CODES:
            warnings.append(f"Unknown country code: {country}")

        return ValidationResult(is_valid=len(errors) == 0, errors=errors, warnings=warnings)
```

#### 4. Error Handling (Bad Records Don't Kill Job)

```python
class IngestionErrorHandler:
    """Handles errors during ingestion without killing the job."""

    def __init__(self, max_errors: int = 1000, error_log_path: str = None):
        self.max_errors = max_errors
        self.error_count = 0
        self.error_log = open(error_log_path, 'w') if error_log_path else None

    def handle_error(self, record: dict, error: Exception) -> bool:
        """Handle a record error. Returns True to continue, False to abort."""
        self.error_count += 1
        if self.error_log:
            self.error_log.write(json.dumps({
                "record_id": record.get('lei') or record.get('cik'),
                "error": str(error),
            }) + '\n')

        if self.error_count >= self.max_errors:
            raise IngestionAbortError(f"Too many errors: {self.error_count}")
        return True
```

### Scalability Patterns

#### 1. Streaming Downloads (Memory Efficient)

```python
async def download_gleif_streaming(output_path: Path) -> int:
    """Download GLEIF data with streaming (don't load 500MB into memory)."""
    url = "https://leidata.gleif.org/api/v1/concatenated-files/lei2/..."

    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        async with client.stream('GET', url) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0

            with open(output_path, 'wb') as f:
                async for chunk in response.aiter_bytes(chunk_size=65536):
                    f.write(chunk)
                    downloaded += len(chunk)
                    if total_size:
                        logger.info(f"Download: {downloaded/total_size*100:.1f}%")

    return downloaded
```

#### 2. COPY Protocol (100K+ rows/sec)

```python
async def bulk_load_entities(pool: asyncpg.Pool, records: Iterator[tuple], batch_size: int = 50000) -> int:
    """Bulk load using PostgreSQL COPY protocol (10-100x faster than INSERT)."""
    columns = ['id', 'lei', 'cik', 'primary_name', 'data_source', ...]
    total_loaded = 0
    batch = []

    async with pool.acquire() as conn:
        for record in records:
            batch.append(record)
            if len(batch) >= batch_size:
                result = await conn.copy_records_to_table('entities', records=batch, columns=columns)
                total_loaded += int(result.split()[-1])
                batch = []

        if batch:
            result = await conn.copy_records_to_table('entities', records=batch, columns=columns)
            total_loaded += int(result.split()[-1])

    return total_loaded
```

#### 3. Streaming CSV Parser (Generator)

```python
def parse_gleif_csv_streaming(csv_path: Path, skip_rows: int = 0) -> Iterator[dict]:
    """Parse GLEIF CSV as generator - never loads full file into memory."""
    with gzip.open(csv_path, 'rt', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for i, row in enumerate(reader):
            if i < skip_rows:
                continue
            if row.get('Entity.EntityStatus') != 'ACTIVE':
                continue
            yield {
                'lei': row['LEI'],
                'primary_name': row['Entity.LegalName'],
                'country_code': row.get('Entity.LegalAddress.Country'),
                'data_source': 'gleif',
            }
```

#### 4. Progress Tracking

```python
class ProgressTracker:
    """Tracks and reports ingestion progress."""

    def __init__(self, total: int, report_interval: int = 10000):
        self.total = total
        self.processed = 0
        self.started_at = time.time()

    def update(self, count: int = 1) -> None:
        self.processed += count
        if self.processed % 10000 == 0:
            elapsed = time.time() - self.started_at
            rate = self.processed / elapsed
            remaining = (self.total - self.processed) / rate
            logger.info(f"Progress: {self.processed/self.total*100:.1f}% Rate: {rate:.0f}/sec ETA: {remaining/60:.1f}min")
```

### Repeatability Patterns

#### 1. CLI Interface

```python
# scripts/ingest_entities.py

@click.group()
def cli():
    """Entity ingestion commands."""

@cli.command()
@click.option('--source', type=click.Choice(['gleif', 'sec_edgar', 'all']), default='all')
@click.option('--dry-run', is_flag=True, help='Validate without loading')
@click.option('--resume', is_flag=True, help='Resume from checkpoint')
@click.option('--batch-size', default=50000)
@click.option('--max-errors', default=1000)
def ingest(source, dry_run, resume, batch_size, max_errors):
    """Ingest entity data from external sources."""
    asyncio.run(_ingest(source, dry_run, resume, batch_size, max_errors))

@cli.command()
def status():
    """Show ingestion status and statistics."""

@cli.command()
@click.option('--source', required=True)
@click.option('--confirm', is_flag=True)
def reset(source, confirm):
    """Reset checkpoint for a source."""
```

#### 2. Configuration

```python
class EntityIngestionConfig(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="ENTITY_")

    gleif_download_url: str = "https://leidata.gleif.org/..."
    sec_tickers_url: str = "https://www.sec.gov/files/company_tickers.json"
    batch_size: int = 50000
    max_errors: int = 1000
    checkpoint_interval: int = 100000
    data_dir: Path = Path("data/entity_ingestion")
    refresh_enabled: bool = False
    refresh_interval_days: int = 30
```

### CLI Usage Examples

```bash
# Full GLEIF ingestion
python scripts/ingest_entities.py ingest --source=gleif
# INFO  downloading_gleif url=https://leidata.gleif.org/...
# INFO  batch_loaded loaded=50000 total=50000 progress_pct=2.4
# INFO  ingestion_complete loaded_entities=1950000 rate_per_sec=10803

# Dry run (validate only)
python scripts/ingest_entities.py ingest --source=gleif --dry-run

# Resume after failure
python scripts/ingest_entities.py ingest --source=gleif --resume

# Check status
python scripts/ingest_entities.py status
# GLEIF: 1,950,000 entities
# SEC EDGAR: 8,500 entities
# Total: 1,958,500 entities

# SEC EDGAR with RIC validation
python scripts/ingest_sec_edgar.py ingest --validate-rics
```

### P1 Deliverables Checklist

| Deliverable | Description | Status |
|-------------|-------------|--------|
| `scripts/ingest_entities.py` | Main CLI with ingest/status/reset | Pending |
| `scripts/ingest_gleif.py` | GLEIF-specific ingestion | Pending |
| `scripts/ingest_sec_edgar.py` | SEC EDGAR-specific ingestion | Pending |
| `src/nl2api/ingestion/checkpoint.py` | Checkpoint manager | Pending |
| `src/nl2api/ingestion/validation.py` | Entity validation | Pending |
| `tests/unit/scripts/test_ingest_*.py` | Unit tests | Pending |
| `tests/integration/test_entity_ingestion.py` | Integration tests | Pending |
