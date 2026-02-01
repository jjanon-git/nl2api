# Infra-03: Trace-to-Scorecard Linking

**Priority:** P1 (High)
**Effort:** 1-2 days
**Status:** Complete (Core Implementation)

---

## Problem Statement

When evaluation failures occur, operators cannot easily link a failed scorecard to its execution trace in Jaeger. The current workflow requires:

1. See failure in Grafana dashboard
2. Note the test_case_id
3. Open Jaeger UI separately
4. Manually search by test_case_id attribute
5. Hope the trace hasn't expired

This friction slows debugging and makes root cause analysis tedious.

---

## Goals

1. Store trace_id in scorecards when they are created
2. Enable direct lookup from scorecard → Jaeger trace
3. Add Grafana data links for one-click trace navigation
4. Maintain backwards compatibility with existing scorecards

---

## Current State Analysis

### What Exists

| Component | Status |
|-----------|--------|
| Span creation in evaluator | ✅ Works (`src/evalkit/core/evaluator.py:185-221`) |
| Exception recording | ✅ Works (stack traces captured) |
| Jaeger datasource in Grafana | ✅ Configured |
| trace_id in Scorecard model | ❌ Missing |
| Grafana → Jaeger links | ❌ Missing |

### Span Hierarchy (Current)

```
evaluator.evaluate (root span)
├── evaluator.stage.syntax
├── evaluator.stage.logic
├── evaluator.stage.execution
└── evaluator.stage.semantics
```

Each span has attributes: `test_case.id`, `result.passed`, `result.score`

---

## Implementation Plan

### Phase 1: Add trace_id to Scorecard Model

**File:** `src/evalkit/contracts/evaluation.py`

Add field to Scorecard:

```python
class Scorecard(BaseModel):
    # ... existing fields ...

    trace_id: str | None = Field(
        default=None,
        description="OpenTelemetry trace ID for Jaeger lookup"
    )
    span_id: str | None = Field(
        default=None,
        description="Root span ID for this evaluation"
    )
```

### Phase 2: Capture trace_id in Evaluator

**File:** `src/evalkit/core/evaluator.py`

Modify `evaluate()` to capture and return trace context:

```python
from opentelemetry import trace

async def evaluate(self, test_case: TestCase, ...) -> Scorecard:
    with tracer.start_as_current_span("evaluator.evaluate") as span:
        # Capture trace context
        span_context = span.get_span_context()
        trace_id = format(span_context.trace_id, '032x')
        span_id = format(span_context.span_id, '016x')

        # ... existing evaluation logic ...

        # Include in scorecard
        return Scorecard(
            ...,
            trace_id=trace_id,
            span_id=span_id,
        )
```

### Phase 3: Database Migration

**File:** `migrations/015_add_trace_id_to_scorecards.sql`

```sql
-- Add trace_id column to scorecards table
ALTER TABLE scorecards
ADD COLUMN IF NOT EXISTS trace_id VARCHAR(32);

ALTER TABLE scorecards
ADD COLUMN IF NOT EXISTS span_id VARCHAR(16);

-- Index for trace lookups
CREATE INDEX IF NOT EXISTS idx_scorecards_trace_id
ON scorecards(trace_id)
WHERE trace_id IS NOT NULL;
```

### Phase 4: Update Repository

**File:** `src/evalkit/common/storage/postgres/scorecard_repo.py`

Update insert/select queries to include trace_id:

```python
async def create(self, scorecard: Scorecard) -> Scorecard:
    # Add trace_id to INSERT columns
    ...

async def get_by_id(self, scorecard_id: str) -> Scorecard | None:
    # Include trace_id in SELECT
    ...
```

### Phase 5: Add Grafana Data Links

**File:** `config/grafana/provisioning/dashboards/json/nl2api/nl2api-evaluation.json`

Add data link to failure panels:

```json
{
  "fieldConfig": {
    "defaults": {
      "links": [
        {
          "title": "View Trace in Jaeger",
          "url": "http://localhost:16686/trace/${__data.fields.trace_id}",
          "targetBlank": true
        }
      ]
    }
  }
}
```

### Phase 6: CLI Enhancement

**File:** `src/evalkit/cli/commands/batch.py`

Add trace link to failure output:

```python
def format_failure(scorecard: Scorecard) -> str:
    output = f"FAILED: {scorecard.test_case_id}"
    if scorecard.trace_id:
        output += f"\n  Trace: http://localhost:16686/trace/{scorecard.trace_id}"
    return output
```

---

## Files to Create

| File | Purpose |
|------|---------|
| `migrations/015_add_trace_id_to_scorecards.sql` | Database migration |
| `tests/unit/evalkit/test_trace_capture.py` | Unit tests for trace capture |

## Files to Modify

| File | Changes |
|------|---------|
| `src/evalkit/contracts/evaluation.py` | Add trace_id, span_id fields |
| `src/evalkit/core/evaluator.py` | Capture trace context |
| `src/evalkit/common/storage/postgres/scorecard_repo.py` | Persist trace_id |
| `config/grafana/provisioning/dashboards/json/nl2api/nl2api-evaluation.json` | Add data links |
| `src/evalkit/cli/commands/batch.py` | Show trace links in CLI |

---

## Testing Plan

1. **Unit Tests**
   - Verify trace_id is captured correctly
   - Verify trace_id format (32 hex chars)
   - Verify scorecard serialization includes trace_id

2. **Integration Tests**
   - Run evaluation with telemetry enabled
   - Verify trace_id stored in database
   - Verify trace exists in Jaeger

3. **Manual Verification**
   - Run batch evaluation
   - Click data link in Grafana
   - Verify correct trace opens in Jaeger

---

## Success Criteria

- [x] trace_id field added to Scorecard model
- [x] trace_id captured during evaluation
- [x] trace_id persisted to database (migration created)
- [ ] Grafana panels link to Jaeger traces (future enhancement)
- [ ] CLI shows trace links for failures (future enhancement)
- [x] Backwards compatible (null trace_id for old scorecards)

---

## Rollback Plan

1. trace_id field is nullable - no impact on existing code
2. If issues arise, revert evaluator changes
3. Old scorecards continue to work (trace_id = null)

---

## Future Enhancements

After this is working:
- Add batch-level span to group all evaluations
- Consider Tempo migration for exemplars (see infra-04-tempo-exemplars.md)
- Add trace-to-logs correlation if Loki is added
