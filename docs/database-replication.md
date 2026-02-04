# Database Replication Guide

How to replicate the evalkit database on another machine using GitHub Releases.

## Quick Start

### On Source Machine (export & upload)

```bash
# 1. Export database
python scripts/db-export.py                     # All tables (~2-5 GB)
python scripts/db-export.py --tables rag        # Just RAG data (~300-500 MB)
python scripts/db-export.py --tables entities   # Just entity data (~1-2 GB)

# 2. Upload to GitHub Release
python scripts/db-upload.py exports/evalkit_all_20260203_120000.dump.gz
```

### On Target Machine (download & restore)

```bash
# 1. Start infrastructure
docker compose up -d

# 2. Download and restore
python scripts/db-restore.py --download data-all-20260203

# Or if you have the file locally:
python scripts/db-restore.py exports/evalkit_all_20260203.dump.gz
```

## Table Groups

| Group | Tables | Size | Use Case |
|-------|--------|------|----------|
| `all` | Everything | ~2-5 GB | Full replica |
| `rag` | rag_documents, sec_filings, etc. | ~300-500 MB | RAG evaluation only |
| `entities` | entities, entity_aliases | ~1-2 GB | Entity resolution only |
| `fixtures` | test_cases, batch_jobs, scorecards | ~50-200 MB | Test fixtures + results |
| `minimal` | test_cases, batch_jobs | ~10-50 MB | Minimal for quick tests |

## Workflow Examples

### Full Replication

```bash
# Source machine
python scripts/db-export.py --tables all
python scripts/db-upload.py exports/evalkit_all_*.dump.gz

# Target machine
python scripts/db-restore.py --download data-all-20260203
```

### RAG-Only Setup

If you only need RAG evaluation (no NL2API entity resolution):

```bash
# Source
python scripts/db-export.py --tables rag
python scripts/db-upload.py exports/evalkit_rag_*.dump.gz

# Target
python scripts/db-restore.py --download data-rag-20260203
python scripts/load-nl2api-fixtures.py --all  # Fixtures from git
```

### Fresh Entities + Cached RAG

If you want fresh GLEIF data but want to skip re-embedding SEC filings:

```bash
# Source (export just RAG embeddings)
python scripts/db-export.py --tables rag
python scripts/db-upload.py exports/evalkit_rag_*.dump.gz

# Target
python scripts/db-restore.py --download data-rag-20260203
python scripts/ingest-gleif.py --mode full    # Fresh entity data
python scripts/ingest-sec-edgar.py            # Fresh SEC mappings
python scripts/load-nl2api-fixtures.py --all
```

## Listing Available Releases

```bash
gh release list | grep data-
```

## Troubleshooting

### "relation already exists" errors

Use `--clean` to drop tables before restore:

```bash
python scripts/db-restore.py --clean exports/evalkit_all.dump.gz
```

### Restore specific tables only

```bash
python scripts/db-restore.py exports/evalkit_all.dump.gz --tables rag_documents entities
```

### Check what's in a dump file

```bash
pg_restore -l exports/evalkit_all.dump | head -50
```

## Data Freshness

| Data Type | Source | Update Frequency | Re-fetch Recommendation |
|-----------|--------|------------------|-------------------------|
| GLEIF entities | GLEIF Golden Copy | Daily deltas | Monthly for fresh data |
| SEC mappings | SEC EDGAR | Real-time | Weekly |
| RAG embeddings | OpenAI API | Static | Only when model changes |
| Test fixtures | Git repo | On commit | Always use git version |
