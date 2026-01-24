# Public Repository Release Audit

**Date:** 2026-01-21
**Updated:** 2026-01-24
**Status:** P0 Complete, Evalkit Platform Assessment Added
**Author:** Mostly Claude, with some minor assistance from Sid

---

## Executive Summary

The nl2api codebase has good test coverage (1964+ tests). P0 blockers (LICENSE, CONTRIBUTING, etc.) have been resolved. However, a critical assessment of the **evaluation platform (evalkit)** reveals significant architectural issues that block enterprise adoption: NL2API coupling in core models, incomplete distributed system, and missing security features.

**Evalkit Platform Grade: C+ (2.5/5)** - Works for basic single-process runs, not enterprise-ready.

---

## P0 - BLOCKERS ✅ COMPLETE

| Item | Status | Date |
|------|--------|------|
| 1. Exposed API Key in .env | ✅ Resolved | 2026-01-21 |
| 2. Missing LICENSE File | ✅ MIT License added | 2026-01-21 |
| 3. Missing CONTRIBUTING.md | ✅ Added | 2026-01-21 |
| 4. Missing CODE_OF_CONDUCT.md | ✅ Contributor Covenant added | 2026-01-21 |
| 5. Missing SECURITY.md | ✅ Added | 2026-01-21 |
| 6. Missing GitHub Templates | ✅ Bug, feature, PR templates added | 2026-01-21 |

---

## P1 - IMPORTANT (Original Items)

| Item | Status | Notes |
|------|--------|-------|
| 7. Incomplete pyproject.toml Metadata | Partial | Authors added, URLs pending repo decision |
| 8. MCP Client Stub Documentation | ✅ Done | Experimental notice added |
| 9. Evaluation Stages 3 & 4 Documentation | Pending | See evalkit assessment below |
| 10. Azure Backend Documentation | Pending | NotImplementedError still present |
| 11. Deprecation Warnings | ✅ Done | `can_handle()` has warning |
| 12. Module Exports Completeness | Pending | |

---

## EVALKIT PLATFORM ASSESSMENT (NEW)

This section documents a critical architectural review of the evaluation platform, focusing **only** on the generic framework—not NL2API or RAG pack implementations.

### Overall Assessment

| Aspect | Grade | Summary |
|--------|-------|---------|
| Core Abstractions | D | NL2API fields hardcoded in "generic" models |
| Batch Runner | C | Works but has race conditions, no checkpoint |
| Storage Layer | B- | Clean protocols, but singleton anti-pattern |
| Distributed System | D | Incomplete, missing production guarantees |
| Enterprise Features | F | No auth, no audit, no tenant isolation |
| **Overall** | **C+** | Basic single-process works, distributed broken |

---

### Critical Finding 1: NL2API Coupling in Core Models

**Severity: HIGH** - Blocks all non-NL2API evaluation packs

**TestCase has NL2API-specific fields** (`src/evalkit/contracts/core.py:414-433`):
```python
# These fields are in the BASE TestCase, not a subclass:
nl_query: str
expected_tool_calls: tuple[ToolCall, ...]
expected_response: dict | None
expected_nl_response: str | None
```

**Scorecard has hardcoded NL2API stages** (`src/evalkit/contracts/evaluation.py:195-210`):
```python
# Base Scorecard has NL2API-specific stage results:
syntax_result: StageResult | None
logic_result: StageResult | None
execution_result: StageResult | None
semantics_result: StageResult | None
```

**Impact:**
- RAG, code-gen, and other packs must ignore these fields
- `get_all_stage_results()` has fragile merge logic (lines 263-277)
- Database schema has NL2API columns as first-class, not JSON

**Remediation:**
- [ ] Create base `TestCase` with only generic fields (`input`, `expected`, `tags`)
- [ ] Create `NL2APITestCase` subclass with `nl_query`, `expected_tool_calls`
- [ ] Same for `Scorecard` → `NL2APIScorecard`
- [ ] Migrate database schema to use generic JSON columns

---

### Critical Finding 2: Race Conditions in Batch Runner

**Severity: HIGH** - Progress tracking is wrong under concurrency

**Location:** `src/evalkit/batch/runner.py:229-241`
```python
async def evaluate_with_progress(tc: TestCase) -> Scorecard:
    nonlocal passed_count, failed_count  # ← Shared mutable state
    scorecard = await self._evaluate_one(tc, ...)
    if scorecard.overall_passed:
        passed_count += 1  # ← NOT ATOMIC
    else:
        failed_count += 1  # ← NOT ATOMIC
```

**Impact:**
- Multiple async tasks increment counters without locks
- Progress bar shows wrong numbers
- Final counts may be incorrect

**Remediation:**
- [ ] Use `asyncio.Lock()` for counter updates
- [ ] Or use `atomics` library for atomic integers
- [ ] Add test for concurrent correctness

---

### Critical Finding 3: No Checkpoint/Resume

**Severity: HIGH** - CLAUDE.md claims feature that doesn't exist

**Location:** `src/evalkit/batch/runner.py:128-312`

The `run()` method:
1. Loads ALL test cases into memory
2. Runs them in a single pass
3. No progress persistence
4. If interrupted, must restart from zero

**Impact:**
- Large batches (>10k tests) risk OOM
- Network interruption loses all progress
- No resume capability despite documentation claims

**Remediation:**
- [ ] Save progress every N test cases to database
- [ ] Add `--resume <batch_id>` CLI option
- [ ] Skip already-evaluated test cases on resume
- [ ] Or remove claim from CLAUDE.md

---

### Critical Finding 4: Distributed System Incomplete

**Severity: CRITICAL** - Not production-ready

| Component | Issue | Location |
|-----------|-------|----------|
| **Worker heartbeat** | TODO comment, not implemented | `worker.py:332` |
| **Exactly-once delivery** | No idempotency checks | `redis_stream.py` |
| **Leader election** | None - multiple coordinators conflict | `coordinator.py` |
| **Nack logic** | `is_retriable` property undefined | `redis_stream.py:297` |
| **Dead worker detection** | Relies only on message timeout | `worker.py` |

**Worker heartbeat TODO:**
```python
# TODO: Publish to Redis for coordinator monitoring
```

**Impact:**
- Messages can be lost or duplicated
- Dead workers not detected
- Multiple coordinators cause race conditions
- Stalled tasks not recovered properly

**Remediation:**
- [ ] Implement worker heartbeat (Redis HSET every 30s)
- [ ] Add idempotency key to WorkerTask
- [ ] Implement coordinator leader election (SETNX/Redlock)
- [ ] Fix `is_retriable` property
- [ ] Add exactly-once delivery with transactional outbox

---

### Critical Finding 5: Enterprise Features Missing

**Severity: HIGH** - Blocks enterprise adoption

| Feature | Status | Issue |
|---------|--------|-------|
| **Authentication** | Missing | No API keys, no JWT, no auth layer |
| **Authorization** | Missing | No RBAC, anyone can do anything |
| **Tenant Isolation** | Schema only | `client_type` column exists but queries don't filter |
| **Audit Logging** | Missing | No "who did what when" |
| **Secrets Management** | Weak | API keys in env vars, no SecretStr |
| **Data Retention** | Missing | Scorecards never deleted |

**Tenant isolation is fiction** (`src/evalkit/common/storage/postgres/scorecard_repo.py`):
- `Client` model exists with `tenant_id`
- But `get_by_batch()` returns ALL scorecards, no tenant filter
- No row-level security in PostgreSQL

**Remediation:**
- [ ] Add `tenant_id` parameter to all repository queries
- [ ] Implement PostgreSQL row-level security
- [ ] Add API key authentication layer
- [ ] Add audit log table and logging
- [ ] Use Pydantic `SecretStr` for sensitive fields

---

### Critical Finding 6: Silent Exception Handling

**Severity: MEDIUM** - Makes debugging impossible

**Locations with bare `pass` or silent failures:**
- `batch/runner.py:340` - bare `pass`
- `distributed/worker.py:176, 202` - bare `pass`
- `queue/redis_stream.py:665` - silent `pass` on exception
- `continuous/scheduler.py:174-175` - continues on any error
- `batch/response_generators.py:121-123` - returns empty on exception

**Impact:**
- Errors disappear silently
- No logging for debugging
- Production issues hard to diagnose

**Remediation:**
- [ ] Replace all `pass` with `logger.exception()`
- [ ] Add structured error codes
- [ ] Create exception hierarchy (`EvalKitError`, `StageError`, etc.)

---

### Critical Finding 7: Incomplete Implementations

| Feature | Location | Issue |
|---------|----------|-------|
| **Regression detector** | `continuous/regression.py:159-168` | Returns `None`, disabled |
| **Email alerting** | `continuous/alerts.py:289-298` | Stub only |
| **Azure storage** | `storage/factory.py:79-83` | `NotImplementedError` |
| **Batch summary** | `scorecard_repo.py` | Incomplete implementation |

**Regression detector is disabled:**
```python
async def _get_previous_batch_metrics(...) -> dict[str, Any] | None:
    # This is a simplified implementation...
    # For now, we return None to indicate no comparison available
    return None
```

---

### Critical Finding 8: Global Singleton Anti-pattern

**Severity: MEDIUM** - Makes testing and multi-tenancy difficult

**Location:** `src/evalkit/common/storage/factory.py:25-28`
```python
_test_case_repo: TestCaseRepository | None = None
_scorecard_repo: ScorecardRepository | None = None
_batch_repo: BatchJobRepository | None = None
_pool: asyncpg.Pool | None = None
```

Uses `global` keyword for state management.

**Remediation:**
- [ ] Create `RepositoryProvider` class
- [ ] Use dependency injection
- [ ] Add `async with provider.session()` context manager

---

## Evalkit Remediation Plan

### Phase 1: Core Stability (4-6 weeks)

| Task | Effort | Priority |
|------|--------|----------|
| Decouple TestCase from NL2API | 2 weeks | P0 |
| Decouple Scorecard from NL2API | 1 week | P0 |
| Fix race condition in progress tracking | 2 days | P0 |
| Add exception hierarchy | 2 days | P1 |
| Replace global singletons | 3 days | P1 |

### Phase 2: Distributed Reliability (4-5 weeks)

| Task | Effort | Priority |
|------|--------|----------|
| Implement exactly-once delivery | 2 weeks | P0 |
| Implement worker heartbeat | 1 week | P0 |
| Add coordinator leader election | 1 week | P1 |
| Fix silent exception handling | 3 days | P1 |
| Add checkpoint/resume | 1 week | P1 |

### Phase 3: Enterprise Features (4-5 weeks)

| Task | Effort | Priority |
|------|--------|----------|
| Implement tenant isolation | 2 weeks | P0 |
| Add authentication layer | 2 weeks | P1 |
| Implement audit logging | 1 week | P2 |
| Complete regression detection | 3 days | P2 |

**Total Remediation Effort: 12-16 weeks**

---

## P2 - NICE TO HAVE (Original Items)

| Item | Status |
|------|--------|
| 13. README Enhancements (badges) | Pending |
| 14. Silent Exception Handlers | See evalkit findings above |
| 15. Type Hint Completeness | Pending |
| 16. CHANGELOG.md | Pending |

---

## Summary

**What's Complete:**
- All P0 open-source files (LICENSE, CONTRIBUTING, etc.)
- Basic single-process batch evaluation works
- Storage protocols are well-designed
- OTEL integration exists

**What's Broken:**
- Core models are NL2API-coupled, not generic
- Distributed mode has critical bugs
- No enterprise features (auth, audit, tenancy)
- Race conditions in batch runner

**Recommendation:** Do not market evalkit as "enterprise-ready" or "production distributed evaluation platform" until Phase 1-2 remediation is complete.
