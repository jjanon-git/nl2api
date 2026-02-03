# Troubleshooting Guide

Common issues and their solutions for the Evalkit project.

## Known Gotchas

| Issue | Solution |
|-------|----------|
| FastAPI returns 422 for valid JSON | Don't use `from __future__ import annotations` in FastAPI files |
| Grafana shows no data | Check metric names have `evalkit_` prefix and `_total` suffix for counters |
| Batch eval fails silently | Run `python scripts/load-nl2api-fixtures.py --all` first |
| Orchestrator fails with "API key not set" | Pass router explicitly to avoid hidden NL2APIConfig dependency |
| "Unexpected keyword argument" after refactor | Kill old server process and restart (code changes not picked up) |

## Grafana Issues

### No Data in Dashboards

1. **Check metric names in Prometheus**
   - Go to http://localhost:9090
   - Search for your metric (e.g., `evalkit_eval_batch_tests_total`)
   - Verify data exists

2. **Check datasource UID**
   - Dashboard JSON files reference datasources by `uid`
   - Datasource config in `config/grafana/provisioning/datasources/` must specify matching `uid`
   - Common issue: dashboard has `"uid": "prometheus"` but datasource config doesn't set `uid`

3. **Check metric naming**
   - OTEL Collector adds `evalkit_` prefix (configured in `config/otel-collector-config.yaml`)
   - Counters get `_total` suffix automatically
   - Example: `eval_batch_tests_passed` → `evalkit_eval_batch_tests_passed_total`

### Metrics Not Appearing

1. Verify OTEL Collector is running: `docker compose ps`
2. Check Prometheus is scraping: http://localhost:9090/targets
3. Verify application has `EVALKIT_TELEMETRY_ENABLED=true`

### Counters Show Wrong Values / Don't Accumulate

**Symptom:** Grafana shows counters that reset or show only recent increments instead of cumulative totals.

**Root cause:** OTEL Python SDK defaults to delta temporality for OTLP exports, but Prometheus requires cumulative temporality.

**Diagnosis:**
```bash
# Check Prometheus metrics - counters should grow over time
curl localhost:8889/metrics | grep evalkit_eval_batch_tests_total

# If values stay small or reset, it's a temporality issue
```

**Fix:** Ensure `src/evalkit/common/telemetry/setup.py` sets cumulative temporality:
```python
from opentelemetry.sdk.metrics.export import AggregationTemporality

preferred_temporality = {
    metrics.Counter: AggregationTemporality.CUMULATIVE,
    metrics.Histogram: AggregationTemporality.CUMULATIVE,
    # ... all metric types
}

metric_exporter = OTLPMetricExporter(
    endpoint=_config.otlp_endpoint,
    preferred_temporality=preferred_temporality,  # Required!
)
```

**Key insight:** OTLP defaults work for stateful collectors (like OTEL Collector with memory), but Prometheus scrapes expect cumulative values. Always configure cumulative temporality for Prometheus integration.

## Batch Evaluation Issues

### Batch Runs But Results Empty

1. **Check you're looking at the right batch**
   ```sql
   SELECT batch_id, COUNT(*) FROM scorecards GROUP BY batch_id ORDER BY batch_id DESC LIMIT 5;
   ```

2. **Verify fixtures are loaded**
   ```bash
   python scripts/load-nl2api-fixtures.py --all
   ```

3. **Check generated_nl_response field**
   ```sql
   SELECT id, generated_nl_response IS NOT NULL as has_response
   FROM scorecards WHERE batch_id = 'YOUR_BATCH_ID' LIMIT 10;
   ```

### Fix Applied But Batch Still Broken

When fixing code that affects a running batch:

1. **Kill the old process first**: `pkill -f "batch run"`
2. **Verify the fix is in the code**: `grep` for the change
3. **Restart with fresh environment**: Ensure env vars are set
4. **Verify new process picks up changes**: Check scorecards from the NEW batch_id

### Verification Query

```sql
-- Quick check for batch health
SELECT
    COUNT(*) as total,
    COUNT(CASE WHEN generated_nl_response IS NOT NULL AND generated_nl_response != '' THEN 1 END) as with_response,
    COUNT(CASE WHEN overall_passed THEN 1 END) as passed
FROM scorecards
WHERE batch_id = 'YOUR_BATCH_ID';
```

## Database Issues

### Connection Refused

1. Start Docker services: `docker compose up -d`
2. Wait for PostgreSQL to be ready: `docker compose logs postgres`
3. Verify port 5432 is accessible: `pg_isready -h localhost -p 5432`

### Migration Issues

1. Check migrations folder: `ls migrations/`
2. Migrations follow naming: `NNN_description.sql`
3. Run pending migrations manually if needed

## LLM API Issues

### "API key not set" Error

1. Check environment variable is set: `echo $NL2API_ANTHROPIC_API_KEY`
2. If using .env file, ensure it's loaded
3. Pass router explicitly to avoid hidden NL2APIConfig dependency

### Rate Limiting

1. Use Batch API for high-volume operations
2. Enable exponential backoff (configured in `tests/accuracy/core/config.py`)
3. Consider tier1 tests first before running full suite

## Server/Process Issues

### "Unexpected keyword argument" After Code Changes

When refactoring code that adds new parameters to method signatures:

1. **Root cause**: Long-running servers (Streamlit, FastAPI) keep old code in memory
2. **Check running processes**: `ps aux | grep -E "(streamlit|uvicorn)" | grep -v grep`
3. **Kill old processes**: `kill <PID>` for each stale process
4. **Restart with fresh code**: `.venv/bin/python3 -m streamlit run src/rag/ui/app.py`
5. **Verify new process**: Check the process start time matches expectation

**Prevention**: After significant refactors, always restart long-running servers before testing.

## RAG Evaluation Issues

### Retrieval Stage Always Fails

**Symptom:** All tests fail retrieval evaluation with "no relevant docs retrieved" even when the response is good.

**Root cause:** Fixture files contain stale chunk UUIDs from a previous ingestion. When documents are re-ingested, they get new UUIDs.

**Quick fix:** Clear relevant_docs to skip retrieval evaluation:
```sql
UPDATE test_cases
SET expected_json = jsonb_set(expected_json, '{relevant_docs}', '[]'::jsonb)
WHERE tags @> ARRAY['rag']
  AND expected_json->'relevant_docs' IS NOT NULL;
```

**Proper fix:** Update fixtures with current chunk IDs:
```bash
python scripts/update-rag-fixture-chunk-ids.py \
  --fixture tests/fixtures/rag/sec_evaluation_set_verified.json \
  --dry-run  # Preview changes first

# Then apply:
python scripts/update-rag-fixture-chunk-ids.py \
  --fixture tests/fixtures/rag/sec_evaluation_set_verified.json
```

**Prevention:** After re-ingesting documents, always run the fixture update script before evaluation.

## Entity Resolution Issues

### "Entity not found" Errors

1. Ensure entity resolution service is running
2. Check `RAG_UI_ENTITY_RESOLUTION_ENDPOINT` is set correctly
3. Use factory: `create_entity_resolver(config)` not direct instantiation

## Where to Document New Issues

| Issue Type | Where to Document |
|------------|-------------------|
| Process/standards gap | CLAUDE.md |
| Code pattern/gotcha | Comment in affected file |
| Config/integration issue | This file (`docs/troubleshooting.md`) |
| Regression risk | Add a test |

**Every debugging session should result in documentation or a test.**
