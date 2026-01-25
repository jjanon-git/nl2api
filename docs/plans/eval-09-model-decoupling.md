# Evalkit Model Decoupling Plan

**Date:** 2026-01-24
**Status:** Complete (Phase 1)
**Priority:** P1 (Cleaner API for non-NL2API packs - RAG works by ignoring fields)
**Author:** Mostly Claude, with some minor assistance from Sid

---

## Implementation Summary (2026-01-24)

**Approach taken:** Backwards-compatible enhancement rather than breaking separation.

**What was implemented:**
1. **TestCase now supports both generic and NL2API fields**
   - Generic fields: `input`, `expected` (dict)
   - NL2API fields: `nl_query`, `expected_tool_calls`, `expected_response`, `expected_nl_response` (kept for backwards compatibility)
   - `content_hash` uses NL2API fields if present, falls back to generic fields

2. **NL2APITestCase is an alias for TestCase** (not a subclass)
   - Maintains backwards compatibility
   - Allows explicit intent signaling

3. **Exports updated:**
   - `NL2APITestCase` added to CONTRACTS.py
   - `NL2APITestCase` added to `src/evalkit/contracts/__init__.py`
   - `NL2APITestCase` added to `src/contracts/__init__.py` (shim)

4. **Deprecation warnings added:**
   - NL2API fields marked with `deprecated=True` in Pydantic Field()
   - `DeprecationWarning` emitted when deprecated fields are accessed
   - Field descriptions updated to point to generic alternatives

5. **Documentation created:**
   - `docs/evaluation-test-case-patterns.md` - recommended patterns for new vs existing packs

6. **Test coverage:** 2083 unit tests pass, 49 evaluation integration tests pass

**What was NOT implemented (deferred to Phase 2):**
- Scorecard subclass separation (already has `stage_results` dict, NL2API fields kept as accessors)
- Database schema migration
- True subclass separation (would break 80+ tests)

**Why this approach:**
- Clean separation would have broken 80+ existing tests
- Current approach achieves the goal (RAG pack works with generic fields) without breaking changes
- Full decoupling can be done incrementally as a future phase

---

## Problem Statement

The evalkit platform's core models (`TestCase` and `Scorecard`) contain NL2API-specific fields in the base classes, making them unsuitable for other evaluation packs (RAG, code-gen, etc.).

### Current State

**TestCase** (`src/evalkit/contracts/core.py:414-433`) has NL2API-specific fields:
```python
# These fields are in the BASE TestCase, not a subclass:
nl_query: str
expected_tool_calls: tuple[ToolCall, ...]
expected_response: dict | None
expected_nl_response: str | None
```

**Scorecard** (`src/evalkit/contracts/evaluation.py:195-210`) has hardcoded NL2API stage results:
```python
# Base Scorecard has NL2API-specific stage results:
syntax_result: StageResult | None
logic_result: StageResult | None
execution_result: StageResult | None
semantics_result: StageResult | None
```

### Impact

1. **RAG, code-gen, and other packs must ignore these fields** - Creates confusion
2. **`get_all_stage_results()` has fragile merge logic** - Combines hardcoded + dynamic fields
3. **Database schema has NL2API columns as first-class** - Not JSON, making schema changes hard
4. **Unit tests are coupled to NL2API structure** - Can't test generic evaluation behavior

---

## Proposed Solution

### Phase 1: Create Generic Base Classes

**New `TestCase` base with only generic fields:**
```python
class TestCase(BaseModel):
    """Generic test case for any evaluation pack."""
    model_config = ConfigDict(frozen=True)

    # Identity
    id: str = Field(default_factory=lambda: str(uuid4()))

    # Generic input/output
    input: dict[str, Any]           # Pack-specific input data
    expected: dict[str, Any]        # Pack-specific expected output

    # Metadata
    metadata: TestCaseMetadata = Field(default_factory=TestCaseMetadata)
    tags: tuple[str, ...] = ()
    category: str | None = None
    subcategory: str | None = None
    complexity: int = Field(default=1, ge=1, le=5)
```

**New `NL2APITestCase` subclass:**
```python
class NL2APITestCase(TestCase):
    """Test case for NL2API evaluation pack."""

    # NL2API-specific fields
    nl_query: str
    expected_tool_calls: tuple[ToolCall, ...]
    expected_response: dict[str, Any] | None = None
    expected_nl_response: str | None = None

    @property
    def input(self) -> dict[str, Any]:
        """Map to generic interface."""
        return {"nl_query": self.nl_query}

    @property
    def expected(self) -> dict[str, Any]:
        """Map to generic interface."""
        return {
            "tool_calls": [tc.model_dump() for tc in self.expected_tool_calls],
            "response": self.expected_response,
            "nl_response": self.expected_nl_response,
        }
```

### Phase 2: Create Generic Scorecard

**New `Scorecard` base:**
```python
class Scorecard(BaseModel):
    """Generic scorecard for any evaluation pack."""
    model_config = ConfigDict(frozen=True)

    # Identity
    scorecard_id: str = Field(default_factory=lambda: str(uuid4()))
    test_case_id: str
    batch_id: str | None = None

    # Results (generic)
    stage_results: dict[str, StageResult] = Field(default_factory=dict)
    overall_passed: bool
    overall_score: float = Field(ge=0.0, le=1.0)

    # Timing
    latency_ms: int
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))

    # Client tracking
    client_type: str = "unknown"
    client_version: str | None = None
    eval_mode: str = "unknown"

    def get_stage_result(self, stage_name: str) -> StageResult | None:
        """Get result for a specific stage."""
        return self.stage_results.get(stage_name)

    def get_all_stage_results(self) -> dict[str, StageResult]:
        """Get all stage results."""
        return self.stage_results
```

**New `NL2APIScorecard` subclass (backwards compatible):**
```python
class NL2APIScorecard(Scorecard):
    """Scorecard for NL2API evaluation pack with convenience accessors."""

    @property
    def syntax_result(self) -> StageResult | None:
        return self.stage_results.get("syntax")

    @property
    def logic_result(self) -> StageResult | None:
        return self.stage_results.get("logic")

    @property
    def execution_result(self) -> StageResult | None:
        return self.stage_results.get("execution")

    @property
    def semantics_result(self) -> StageResult | None:
        return self.stage_results.get("semantics")
```

### Phase 3: Database Schema Migration

**Current schema (NL2API-coupled):**
```sql
CREATE TABLE test_cases (
    id TEXT PRIMARY KEY,
    nl_query TEXT NOT NULL,           -- NL2API-specific
    expected_tool_calls JSONB,        -- NL2API-specific
    expected_response JSONB,          -- NL2API-specific
    expected_nl_response TEXT,        -- NL2API-specific
    ...
);

CREATE TABLE scorecards (
    scorecard_id TEXT PRIMARY KEY,
    syntax_result JSONB,              -- NL2API-specific
    logic_result JSONB,               -- NL2API-specific
    execution_result JSONB,           -- NL2API-specific
    semantics_result JSONB,           -- NL2API-specific
    ...
);
```

**New schema (generic):**
```sql
CREATE TABLE test_cases (
    id TEXT PRIMARY KEY,
    pack_type TEXT NOT NULL,          -- 'nl2api', 'rag', etc.
    input JSONB NOT NULL,             -- Pack-specific input
    expected JSONB NOT NULL,          -- Pack-specific expected output
    tags TEXT[],
    category TEXT,
    subcategory TEXT,
    complexity INTEGER DEFAULT 1,
    metadata JSONB,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE scorecards (
    scorecard_id TEXT PRIMARY KEY,
    test_case_id TEXT REFERENCES test_cases(id),
    batch_id TEXT,
    pack_type TEXT NOT NULL,          -- 'nl2api', 'rag', etc.
    stage_results JSONB NOT NULL,     -- All stages in one column
    overall_passed BOOLEAN NOT NULL,
    overall_score REAL NOT NULL,
    latency_ms INTEGER,
    client_type TEXT DEFAULT 'unknown',
    client_version TEXT,
    eval_mode TEXT DEFAULT 'unknown',
    created_at TIMESTAMPTZ DEFAULT NOW()
);
```

**Migration strategy:**
1. Add new columns (`pack_type`, `input`, `expected`, `stage_results`)
2. Backfill from existing columns
3. Update repository code to use new columns
4. Drop old columns after validation

---

## Implementation Steps

### Step 1: Define New Base Classes (Non-breaking)
- [ ] Create `TestCaseBase` with generic fields
- [ ] Create `ScorecardBase` with generic fields
- [ ] Keep existing `TestCase` and `Scorecard` as aliases initially

### Step 2: Update Packs to Use Generic Interface
- [ ] Update NL2API pack to work with both old and new interfaces
- [ ] Update RAG pack to use generic interface
- [ ] Add type guards for pack-specific handling

### Step 3: Database Migration
- [ ] Create migration script for new schema
- [ ] Add pack_type column with default 'nl2api'
- [ ] Create backfill script for existing data
- [ ] Update repository implementations

### Step 4: Repository Updates
- [ ] Update `TestCaseRepository` to handle generic interface
- [ ] Update `ScorecardRepository` to handle generic interface
- [ ] Add pack-specific query methods where needed

### Step 5: Remove Legacy Fields
- [ ] Deprecate direct field access on base classes
- [ ] Add deprecation warnings
- [ ] Remove in next major version

---

## Backwards Compatibility

### Python API Compatibility

To avoid breaking existing code, use a compatibility layer:

```python
# In src/evalkit/contracts/core.py

class TestCase(TestCaseBase):
    """Backwards-compatible TestCase with NL2API fields."""

    # These become computed properties that delegate to input/expected
    @property
    def nl_query(self) -> str:
        return self.input.get("nl_query", "")

    @property
    def expected_tool_calls(self) -> tuple[ToolCall, ...]:
        # Parse from expected dict
        ...
```

### Database Compatibility

During migration:
1. Write to both old and new columns
2. Read from new columns with fallback to old
3. After validation period, drop old columns

---

## Risks and Mitigations

| Risk | Impact | Mitigation |
|------|--------|------------|
| Breaking existing tests | HIGH | Compatibility layer, gradual deprecation |
| Database migration failure | HIGH | Reversible migration, backups, dual-write |
| Pack incompatibility | MEDIUM | Integration tests per pack |
| Performance regression | LOW | Benchmark before/after |

---

## Success Criteria

1. **RAG pack works without NL2API fields** - No dummy values needed
2. **New packs can be added** without touching base models
3. **Existing tests pass** with no changes
4. **Database queries work** for all pack types
5. **`get_all_stage_results()` is clean** - No merge logic needed

---

## Verification Approach

### Unit Tests (run after each phase)

```bash
# All existing tests must pass
.venv/bin/python -m pytest tests/unit/ -v --tb=short -x

# Specific contract tests
.venv/bin/python -m pytest tests/unit/evaluation/ -v -k "test_case or scorecard"
```

### Integration Tests (after Phase 3: Database Migration)

```bash
# Test both packs against real database
.venv/bin/python -m pytest tests/integration/evaluation/ -v

# Verify data roundtrip
.venv/bin/python -m src.evaluation.cli.main batch run --pack nl2api --limit 5
.venv/bin/python -m src.evaluation.cli.main batch run --pack rag --limit 5
.venv/bin/python -m src.evaluation.cli.main batch list  # Both should appear
```

### Migration Verification (Phase 3)

```sql
-- Verify backfill completed
SELECT pack_type, COUNT(*) FROM test_cases GROUP BY pack_type;
SELECT pack_type, COUNT(*) FROM scorecards GROUP BY pack_type;

-- Verify no NULL in new columns
SELECT COUNT(*) FROM test_cases WHERE input IS NULL;
SELECT COUNT(*) FROM scorecards WHERE stage_results IS NULL;

-- Verify data integrity (NL2API)
SELECT COUNT(*) FROM test_cases
WHERE pack_type = 'nl2api'
AND input->>'nl_query' != nl_query;  -- Should be 0
```

### Backwards Compatibility Verification

```python
# Test that old code patterns still work
from src.evalkit.contracts.core import TestCase

# Old pattern (should still work via compatibility layer)
tc = TestCase(
    nl_query="What is Apple's PE ratio?",
    expected_tool_calls=(ToolCall(...),),
)
assert tc.nl_query == "What is Apple's PE ratio?"

# New pattern
assert tc.input["nl_query"] == "What is Apple's PE ratio?"
```

### Performance Verification

```bash
# Benchmark before migration
time .venv/bin/python -m src.evaluation.cli.main batch run --pack nl2api --limit 100

# Benchmark after migration (should be within 10%)
time .venv/bin/python -m src.evaluation.cli.main batch run --pack nl2api --limit 100
```

### Checklist Per Phase

| Phase | Verification |
|-------|--------------|
| Phase 1 | Unit tests pass, RAG pack imports without errors |
| Phase 2 | Both packs can create scorecards, `get_all_stage_results()` works |
| Phase 3 | Migration script runs, backfill SQL returns expected counts |
| Phase 4 | Integration tests pass, batch runs complete for both packs |
| Phase 5 | Deprecation warnings appear, old code still works |

---

## Estimated Effort

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: Base classes | 3 days | None |
| Phase 2: Scorecard refactor | 2 days | Phase 1 |
| Phase 3: Database migration | 5 days | Phase 1, 2 |
| Phase 4: Repository updates | 3 days | Phase 3 |
| Phase 5: Legacy removal | 2 days | Phase 4 + validation |
| **Total** | **~3 weeks** | |

---

## Related Files

| File | Changes Needed |
|------|----------------|
| `src/evalkit/contracts/core.py` | New TestCaseBase, update TestCase |
| `src/evalkit/contracts/evaluation.py` | New ScorecardBase, update Scorecard |
| `src/evalkit/common/storage/postgres/test_case_repo.py` | Generic queries |
| `src/evalkit/common/storage/postgres/scorecard_repo.py` | Generic queries |
| `src/evaluation/packs/nl2api.py` | Use NL2APITestCase |
| `src/evaluation/packs/rag/pack.py` | Use generic TestCase |
| `migrations/` | Schema migration scripts |

---

## Notes

- This is a prerequisite for making evalkit a standalone package
- RAG pack is already working around this by ignoring NL2API fields
- The `get_all_stage_results()` method has fragile merge logic that this fixes
