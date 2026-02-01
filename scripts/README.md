# Scripts Directory

Utility scripts for data generation, ingestion, evaluation, and CI/CD.

## Naming Convention

Scripts use prefixes to indicate their category:

| Prefix | Purpose | Example |
|--------|---------|---------|
| `db-` | Database operations | `db-migrate.py` |
| `ingest-` | External data ingestion | `ingest-gleif.py` |
| `gen-` | Test case/data generation | `gen-test-cases.py` |
| `load-` | Load data to DB for evaluation | `load-nl2api-fixtures.py` |
| `eval-` | Run evaluations | `eval-nl2api.py` |
| `backfill-` | Data maintenance/backfill | `backfill-client-type.py` |
| `test-` | Testing utilities | `test-dashboard-metrics.py` |
| `ci-` | CI/CD utilities | `ci-local.sh` |
| `run-` | Start services | `run-rag-ui.py` |
| `analyze-` | Data analysis | `analyze-metrics.py` |

## Script Categories

### Database (`db-`)

| Script | Purpose |
|--------|---------|
| `db-migrate.py` | Run PostgreSQL migrations |

### Ingestion (`ingest-`)

External data → PostgreSQL. Run in order for first-time setup.

| Script | Purpose | Prerequisites |
|--------|---------|---------------|
| `ingest-gleif.py` | Download 2M+ GLEIF legal entities | PostgreSQL |
| `ingest-sec-edgar.py` | SEC company tickers + RIC mappings | `ingest-gleif.py` |
| `ingest-sec-filings.py` | SEC 10-K/10-Q filings (RAG) | PostgreSQL, OpenAI API |
| `ingest-entities.py` | CLI wrapper for above | - |
| `ingest-field-codes.py` | Index field code reference docs | PostgreSQL |

### Generation (`gen-`)

Generate test fixtures and evaluation data.

| Script | Purpose | Prerequisites |
|--------|---------|---------------|
| `gen-test-cases.py` | Generate 11K+ NL2API test cases | `generators/` modules |
| `gen-nl-responses.py` | Generate NL responses (~$5) | `gen-test-cases.py` output, Claude API |
| `gen-rag-eval-dataset.py` | Generate RAG eval dataset | SEC filings |
| `gen-sec-rag-answers.py` | Generate RAG reference answers | SEC filings, Claude API |
| `gen-enrich-rag-questions.py` | Enrich questions with metadata | Claude API |
| `gen-expand-sp500.py` | Expand S&P 500 list with CIKs | SEC API |
| `gen-economic-indicators.py` | Generate economic indicators | None |
| `gen-eval-data.py` | Generic eval data generation | - |

### Loading (`load-`)

Load fixtures to PostgreSQL for batch evaluation.

| Script | Purpose | Prerequisites |
|--------|---------|---------------|
| `load-nl2api-fixtures.py` | Load NL2API fixtures | PostgreSQL, `gen-test-cases.py` output |
| `load-rag-fixtures.py` | Load RAG fixtures | PostgreSQL |
| `load-test-cases.py` | Generic test case loader | PostgreSQL |

### Evaluation (`eval-`)

Run evaluation pipelines.

| Script | Purpose | Prerequisites |
|--------|---------|---------------|
| `eval-nl2api.py` | NL2API eval with Claude Haiku | Fixtures in DB, Claude API |
| `eval-nl2api-batch.py` | NL2API eval with Batch API (50% cheaper) | Fixtures in DB, Claude API |
| `eval-estimates.py` | EstimatesAgent evaluation | Fixtures, LLM API |
| `eval-routing.py` | Routing accuracy evaluation | Accuracy test framework |
| `eval-rag-baseline-DEPRECATED.py` | **DEPRECATED** - use batch framework | - |

### Backfill (`backfill-`)

Data maintenance for existing records.

| Script | Purpose |
|--------|---------|
| `backfill-client-type.py` | Backfill NULL client_type in scorecards |
| `backfill-sec-metadata.py` | Add SEC metadata to existing chunks |
| `backfill-contextual-prefixes.py` | Add context prefixes to RAG chunks |
| `backfill-reembed-openai.py` | Re-embed with OpenAI embeddings |
| `backfill-reindex-small-to-big.py` | Reindex with small-to-big chunking |

### Testing (`test-`)

Testing and verification utilities.

| Script | Purpose |
|--------|---------|
| `test-dashboard-metrics.py` | Push synthetic metrics to verify OTEL stack |
| `test-rag-dashboard.py` | Test RAG dashboard |
| `test-distributed-e2e.py` | E2E test for distributed evaluation |
| `test-metrics-export.py` | Test metrics export pipeline |

### CI/CD (`ci-`)

| Script | Purpose |
|--------|---------|
| `ci-local.sh` | Run full CI locally (lint, test, build) |
| `ci-test-changed.sh` | Run tests for changed files only |
| `verify-podman.sh` | Verify Podman stack setup |

### Infrastructure

| Script | Purpose |
|--------|---------|
| `run-rag-ui.py` | Launch RAG Streamlit UI |
| `analyze-metrics.py` | Analyze request metrics from JSONL |

## First-Time Setup Order

```bash
# 1. Database
python scripts/db-migrate.py

# 2. Ingestion (order matters)
python scripts/ingest-gleif.py --mode full    # ~2M entities
python scripts/ingest-sec-edgar.py            # Depends on GLEIF
python scripts/ingest-sec-filings.py          # RAG documents

# 3. Generation
python scripts/gen-test-cases.py --all        # ~11K test cases
python scripts/gen-nl-responses.py --all      # Optional, costs ~$5

# 4. Loading
python scripts/load-nl2api-fixtures.py --all
python scripts/load-rag-fixtures.py

# 5. Evaluation
python -m src.evalkit.cli.main batch run --pack nl2api --tag entity_resolution
```

## Subdirectories

| Directory | Purpose |
|-----------|---------|
| `generators/` | Test case generator modules (used by `gen-test-cases.py`) |
| `data/` | Static data files for generation |
