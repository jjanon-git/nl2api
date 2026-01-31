# RAG OpenAI Embeddings Fix

**Date:** 2026-01-31
**Status:** COMPLETED

## Summary

Fixed RAG tests to use OpenAI embeddings (1536 dims) instead of local embeddings (384 dims). The database schema expects 1536-dim vectors (migration 013).

## Completed Work

### 1. Field Code Indexing with OpenAI Embeddings

All 463 field codes now indexed with OpenAI embeddings:

| Domain | Count | With Embeddings |
|--------|-------|-----------------|
| datastream | 101 | 101 |
| estimates | 79 | 79 |
| fundamentals | 202 | 202 |
| officers | 33 | 33 |
| screening | 48 | 48 |

Command used:
```bash
source .env && \
OPENAI_API_KEY="${NL2API_OPENAI_API_KEY}" \
EMBEDDING_PROVIDER=openai \
.venv/bin/python scripts/index_field_codes.py --clear
```

### 2. Fixed Multi-Turn Integration Tests

**Issue:** `TypeError: unsupported operand type(s) for +: 'coroutine' and 'coroutine'`
**Location:** `src/nl2api/observability/metrics.py:232`
**Root cause:** `create_mock_llm()` only mocked `complete` but `BaseDomainAgent.process` calls `complete_with_retry`. Also, mock `usage` dict used wrong key names.

**Fix:** Updated `tests/integration/nl2api/test_multi_turn.py`:
```python
def create_mock_llm() -> LLMProvider:
    mock_response = LLMResponse(
        content="Test response",
        tool_calls=(),
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
    )
    mock_llm = MagicMock(spec=LLMProvider)
    mock_llm.complete = AsyncMock(return_value=mock_response)
    mock_llm.complete_with_retry = AsyncMock(return_value=mock_response)
    return mock_llm
```

### 3. Test Verification

| Test Suite | Count | Status |
|------------|-------|--------|
| RAG Integration tests | 7 | PASS |
| Multi-turn integration tests | 18 | PASS |
| All integration tests | 173 | PASS (7 skipped for SEC ingestion) |

## Related Prior Changes

1. **Updated routing accuracy tests** (`tests/accuracy/routing/test_routing_accuracy.py`)
   - Changed tier1/tier2 to use realtime API (fast, ~30s for 50 samples)
   - Changed tier3 to use batch API (50% cheaper, but ~8 hours)
   - Verified: tier2 passed with 98.5% accuracy in 4min 37s

## Environment Variables Required

```bash
# In .env file
NL2API_OPENAI_API_KEY=sk-proj-...  # OpenAI API key for embeddings
DATABASE_URL=postgresql://nl2api:nl2api@localhost:5432/nl2api
```

## Files Modified

- `tests/integration/nl2api/test_multi_turn.py` - Fixed mock LLM with proper usage keys and complete_with_retry
