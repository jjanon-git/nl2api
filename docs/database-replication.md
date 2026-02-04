# Database Replication Guide

How to replicate the nl2api database on another machine using GitHub Releases.

## Quick Start (RAG Data)

The RAG database (~12GB with embeddings) is available as a GitHub Release.

### Download and Restore

```bash
# 1. Start infrastructure
docker compose up -d

# 2. Download all parts
gh release download data-rag-20260203 -p "*.part-*" -D exports/

# 3. Reassemble split files
cd exports
cat evalkit_rag_20260203.dump.gz.part-* > evalkit_rag_20260203.dump.gz

# 4. Decompress
gunzip -k evalkit_rag_20260203.dump.gz

# 5. Restore into database
docker exec -i evalkit-db pg_restore -U nl2api -d nl2api -v --no-owner --no-acl < evalkit_rag_20260203.dump

# 6. Clean up
rm evalkit_rag_20260203.dump.gz.part-* evalkit_rag_20260203.dump evalkit_rag_20260203.dump.gz

# 7. Load test fixtures from Git (references chunk IDs in rag_documents)
python scripts/load-rag-fixtures.py
```

**Note:** RAG test fixtures are stored in Git (`tests/fixtures/rag/*.json`), not the database dump. They reference chunk IDs from `rag_documents`, so restore the dump first.

## Creating a New Export

Large exports (>2GB) need to be split for GitHub Release upload.

### Export and Upload

```bash
# 1. Export from docker container (pg_dump not installed locally)
docker exec evalkit-db pg_dump -U nl2api -d nl2api -Fc -v \
  -t rag_documents \
  -t sec_filings \
  -t sec_filing_ingestion_jobs \
  -t indexing_checkpoint \
  > exports/evalkit_rag_$(date +%Y%m%d).dump.gz

# 2. Split into <2GB chunks for GitHub
cd exports
split -b 1500m evalkit_rag_*.dump.gz evalkit_rag_$(date +%Y%m%d).dump.gz.part-

# 3. Create release and upload
gh release create data-rag-$(date +%Y%m%d) \
  --title "RAG Database Snapshot ($(date +%Y-%m-%d))" \
  --notes "RAG database with vector embeddings" \
  evalkit_rag_*.dump.gz.part-*
```

## Table Groups

| Group | Tables | Actual Size | Use Case |
|-------|--------|-------------|----------|
| `rag` | rag_documents, sec_filings, etc. | ~12 GB | RAG evaluation (includes embeddings) |
| `entities` | entities, entity_aliases | ~1-2 GB | Entity resolution only |
| `fixtures` | test_cases, batch_jobs, scorecards | ~50-200 MB | Test fixtures + results |
| `minimal` | test_cases, batch_jobs | ~10-50 MB | Minimal for quick tests |

## Available Releases

```bash
gh release list | grep data-
```

Current releases:
- `data-rag-20260203` - RAG tables with SEC filing embeddings (~12GB, 9 parts)

## Troubleshooting

### "relation already exists" errors

Drop tables before restore:

```bash
docker exec -i evalkit-db pg_restore -U nl2api -d nl2api -v --clean --if-exists --no-owner --no-acl < evalkit_rag.dump
```

### Check what's in a dump file

```bash
docker exec evalkit-db pg_restore -l /dev/stdin < exports/evalkit_rag.dump | head -50
```

### Verify restore worked

```bash
docker exec evalkit-db psql -U nl2api -d nl2api -c "SELECT COUNT(*) FROM rag_documents;"
```

## Data Freshness

| Data Type | Source | Update Frequency | Re-fetch Recommendation |
|-----------|--------|------------------|-------------------------|
| GLEIF entities | GLEIF Golden Copy | Daily deltas | Monthly for fresh data |
| SEC mappings | SEC EDGAR | Real-time | Weekly |
| RAG embeddings | OpenAI API | Static | Only when model changes |
| Test fixtures | Git repo | On commit | Always use git version |
