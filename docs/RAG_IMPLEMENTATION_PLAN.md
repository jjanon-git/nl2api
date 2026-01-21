# RAG Implementation Plan

## Overview

The RAG infrastructure is **architecturally complete** but **functionally empty** - no data is loaded into the `rag_documents` table.

## Current State

### What Exists (Complete Infrastructure)

| Component | File | Status |
|-----------|------|--------|
| Database schema (pgvector) | `migrations/002_rag_documents.sql` | ✅ Complete |
| RAG Indexer with batch/checkpoint | `src/nl2api/rag/indexer.py` | ✅ Complete |
| Hybrid Retriever (vector + keyword) | `src/nl2api/rag/retriever.py` | ✅ Complete |
| Orchestrator integration | `src/nl2api/orchestrator.py` | ✅ Complete |
| Agent context injection | `src/nl2api/agents/base.py:153-158` | ✅ Complete |
| Reference documents | `*_REFERENCE.md` files | ✅ Available |

### What's Missing

| Gap | Impact |
|-----|--------|
| No data in `rag_documents` table | RAG returns empty results |
| No embedder initialized | Can't generate embeddings |
| No indexing script | No way to load reference docs |
| Parsers only for Estimates domain | Other domains not parsed |

## Implementation Plan

### Phase 1: Index Field Codes

**1.1 Create indexing script**
```
scripts/index_field_codes.py
```

Parse all 5 reference documents:
- `DATASTREAM_REFERENCE.md` (600+ field codes)
- `FUNDAMENTALS_REFERENCE.md` (400+ field codes)
- `ESTIMATES_REFERENCE.md`
- `OFFICERS_DIRECTORS_REFERENCE.md`
- `SCREENING_REFERENCE.md`

**1.2 Add domain-specific parsers**

Extend `parse_estimates_reference()` pattern in `indexer.py`:
- `parse_datastream_reference()`
- `parse_fundamentals_reference()`
- `parse_officers_reference()`
- `parse_screening_reference()`

**1.3 Initialize embedder**

Configure `OpenAIEmbedder` with API key from `.env`:
```python
from src.nl2api.rag.retriever import OpenAIEmbedder

embedder = OpenAIEmbedder(api_key=os.environ["OPENAI_API_KEY"])
indexer.set_embedder(embedder)
```

### Phase 2: Wire Into Evaluation

**Option A: Full orchestrator flow**
```
Query → Orchestrator → RAG Retriever → AgentContext.field_codes → Agent → Tool Call
```

**Option B: Lightweight RAG injection**
```
Query → RAG Retriever → Inject into prompt → Agent → Tool Call
```

### Phase 3: Validate & Measure

**3.1 Integration tests**
```python
# tests/integration/test_rag_retrieval.py
async def test_bid_price_retrieves_pb_field():
    results = await retriever.retrieve_field_codes("bid price", domain="datastream")
    codes = [r.field_code for r in results]
    assert "PB" in codes
```

**3.2 Metrics**
- Retrieval hit rate
- Pass rate comparison with/without RAG
- Field code coverage per domain

## Files to Create/Modify

| File | Action |
|------|--------|
| `scripts/index_field_codes.py` | Create - Parse & index all reference docs |
| `src/nl2api/rag/indexer.py` | Modify - Add parsers for all domains |
| `src/evaluation/cli/commands/api_batch.py` | Modify - Add RAG retrieval option |
| `tests/integration/test_rag_retrieval.py` | Create - End-to-end RAG tests |

## How Agent Uses Retrieved Field Codes

From `src/nl2api/agents/base.py:153-158`:

```python
if context.field_codes:
    field_codes_text = "\n".join(
        f"- {fc.get('code', '')}: {fc.get('description', '')}"
        for fc in context.field_codes
    )
    system_prompt += f"\n\nAvailable field codes:\n{field_codes_text}"
```

The retrieved field codes are dynamically appended to the system prompt, keeping the base prompt small while injecting query-relevant context.

## Database Schema

From `migrations/002_rag_documents.sql`:

```sql
CREATE TABLE rag_documents (
    id UUID PRIMARY KEY,
    domain VARCHAR(50) NOT NULL,           -- 'datastream', 'estimates', etc.
    document_type VARCHAR(50) NOT NULL,    -- 'field_code', 'query_example'
    field_code VARCHAR(100),               -- 'PB', 'TR.EPSMean', etc.
    content TEXT NOT NULL,                 -- Searchable text
    embedding vector(1536),                -- OpenAI embedding
    content_tsv tsvector,                  -- Full-text search
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

-- Indexes
CREATE INDEX idx_rag_embedding ON rag_documents USING hnsw (embedding vector_cosine_ops);
CREATE INDEX idx_rag_fts ON rag_documents USING GIN (content_tsv);
```

## Priority Order

1. **Must Have**: Index reference documents, initialize embedder
2. **Should Have**: All domain parsers, validation layer, fallback to static mappings
3. **Nice to Have**: Redis caching, query expansion, observability
