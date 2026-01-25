# Evalkit Release Audit

**Date:** 2026-01-21
**Updated:** 2026-01-25
**Status:** Phase 1 Complete, Continuing Remediation
**Author:** Mostly Claude, with some minor assistance from Sid

---

## Executive Summary

**Evalkit** is a generic evaluation framework designed to support multiple evaluation packs (NL2API, RAG, code-gen, etc.). This audit focuses on the framework's readiness for public release.

| Aspect | Grade | Status |
|--------|-------|--------|
| **Single-process batch evaluation** | B+ | Production-ready |
| **Generic abstraction** | B+ | Core models support any pack via `input`/`expected` fields |
| **Distributed multi-worker mode** | D | Critical gaps, not production-ready |
| **Enterprise features** | F | Missing auth, audit, tenant isolation |

**Overall: B (3.2/5)** - Single-process is production-ready. Distributed mode has critical bugs.

### What Works
- Single-process `BatchRunner` with checkpoint/resume
- Generic `TestCase.input`/`expected` and `Scorecard.stage_results` fields (NL2API fields are deprecated legacy)
- Proper concurrency handling (asyncio.Lock)
- Repository provider pattern (no singletons)
- OTEL telemetry integration
- PostgreSQL and in-memory backends
- RAG and NL2API packs both working

### What's Broken
- Distributed workers: no heartbeat, no exactly-once delivery, no leader election
- No authentication, authorization, or audit logging

---

## Current Focus: Remaining Work

### HIGH PRIORITY - Architectural Issues

#### 1. Distributed System Incomplete
**Severity: CRITICAL** - Not production-ready for multi-worker deployment

| Component | Issue | Location |
|-----------|-------|----------|
| **Worker heartbeat** | TODO comment, not implemented | `worker.py:332` |
| **Exactly-once delivery** | No idempotency checks | `redis_stream.py` |
| **Leader election** | None - multiple coordinators conflict | `coordinator.py` |
| **Nack logic** | `is_retriable` property undefined | `redis_stream.py:297` |
| **Dead worker detection** | Relies only on message timeout | `worker.py` |

**Remediation:**
- [ ] Implement worker heartbeat (Redis HSET every 30s)
- [ ] Add idempotency key to WorkerTask
- [ ] Implement coordinator leader election (SETNX/Redlock)
- [ ] Fix `is_retriable` property
- [ ] Add exactly-once delivery with transactional outbox

---

#### 2. Enterprise Features Missing
**Severity: HIGH** - Blocks enterprise adoption

| Feature | Status | Issue |
|---------|--------|-------|
| **Authentication** | Missing | No API keys, no JWT, no auth layer |
| **Authorization** | Missing | No RBAC, anyone can do anything |
| **Tenant Isolation** | Schema only | `client_type` column exists but queries don't filter |
| **Audit Logging** | Missing | No "who did what when" |
| **Secrets Management** | Weak | API keys in env vars, no SecretStr |
| **Data Retention** | Missing | Scorecards never deleted |

**Remediation:**
- [ ] Add `tenant_id` parameter to all repository queries
- [ ] Implement PostgreSQL row-level security
- [ ] Add API key authentication layer
- [ ] Add audit log table and logging
- [ ] Use Pydantic `SecretStr` for sensitive fields

---

### MEDIUM PRIORITY - Incomplete Implementations

| Feature | Location | Issue |
|---------|----------|-------|
| **Email alerting** | `continuous/alerts.py:289-298` | Stub only |
| **Azure storage** | `storage/factory.py:79-83` | `NotImplementedError` |
| **Batch summary** | `scorecard_repo.py` | Incomplete implementation |
| **Remaining silent handlers** | `distributed/worker.py:176, 202`, `continuous/scheduler.py:174-175` | Bare `pass` needs logging |

---

### LOW PRIORITY - Nice to Have

| Item | Status |
|------|--------|
| README badges | Pending |
| Type hint completeness | Pending |
| CHANGELOG.md | Pending |
| pyproject.toml URLs | Pending (repo decision) |
| Evaluation Stages 3 & 4 docs | Pending |
| Azure backend docs | Pending |
| Module exports completeness | Pending |

---

## Remediation Plan (Updated)

### Phase 1: Core Stability - ✅ COMPLETE

| Task | Status |
|------|--------|
| Fix race condition in progress tracking | ✅ Done (2026-01-24) |
| Add exception hierarchy | ✅ Done (2026-01-24) |
| Replace global singletons | ✅ Done (2026-01-24) |
| Add checkpoint/resume | ✅ Done (2026-01-25) |
| Generic TestCase/Scorecard fields | ✅ Done (previously) |

### Phase 2: Distributed Reliability (~4 weeks) - NOT STARTED

| Task | Effort | Priority | Status |
|------|--------|----------|--------|
| Implement exactly-once delivery | 2 weeks | P0 | Pending |
| Implement worker heartbeat | 1 week | P0 | Pending |
| Add coordinator leader election | 1 week | P1 | Pending |

### Phase 3: Enterprise Features (4-5 weeks)

| Task | Effort | Priority | Status |
|------|--------|----------|--------|
| Implement tenant isolation | 2 weeks | P0 | Pending |
| Add authentication layer | 2 weeks | P1 | Pending |
| Implement audit logging | 1 week | P2 | Pending |

**Remaining Effort: ~8 weeks** (Phase 2: 4 weeks + Phase 3: 4 weeks)

---

## Recommendation

Single-process batch evaluation is now **production-ready** with:
- Generic `TestCase`/`Scorecard` models (supports any pack)
- Checkpoint/resume for interrupted runs
- Proper concurrency handling with locks
- Clean exception hierarchy
- Repository provider pattern (no singletons)

Both NL2API and RAG packs are working. New packs can be added using the generic `input`/`expected` and `stage_results` fields.

**Do not** deploy distributed mode (multi-worker) until Phase 2 is complete.

---

---

# Completed Work History

## P0 - BLOCKERS ✅ COMPLETE (2026-01-21)

| Item | Status | Date |
|------|--------|------|
| 1. Exposed API Key in .env | ✅ Resolved | 2026-01-21 |
| 2. Missing LICENSE File | ✅ MIT License added | 2026-01-21 |
| 3. Missing CONTRIBUTING.md | ✅ Added | 2026-01-21 |
| 4. Missing CODE_OF_CONDUCT.md | ✅ Contributor Covenant added | 2026-01-21 |
| 5. Missing SECURITY.md | ✅ Added | 2026-01-21 |
| 6. Missing GitHub Templates | ✅ Bug, feature, PR templates added | 2026-01-21 |

---

## Completed Critical Findings

### Finding: Race Conditions in Batch Runner ✅ FIXED (2026-01-24)

**Original Issue:** Progress tracking was wrong under concurrency - multiple async tasks incremented counters without locks.

**Resolution:** Added `asyncio.Lock()` for counter updates in `src/evalkit/batch/runner.py`.

---

### Finding: No Checkpoint/Resume ✅ FIXED (2026-01-25)

**Original Issue:** Large batches risked OOM, network interruption lost all progress, no resume capability.

**Resolution:**
- Added `get_evaluated_test_case_ids()` to scorecard repository
- Added `update_progress()` to batch repository for periodic checkpoints
- Added `checkpoint_interval` config (default: 10)
- Implemented resume logic in BatchRunner with `--resume <batch_id>` CLI option
- 13 unit tests + 3 integration tests

---

### Finding: NL2API Coupling in Core Models ✅ ADDRESSED (previously)

**Original Issue:** Core `TestCase` and `Scorecard` models had NL2API-specific fields that would block other packs.

**Resolution:** Models now have generic fields with NL2API fields as deprecated legacy:
- `TestCase.input: dict` and `TestCase.expected: dict` for any pack
- `Scorecard.stage_results: dict[str, StageResult]` for any stages
- `Scorecard.generated_output: dict` for any output format
- NL2API fields (`nl_query`, `expected_tool_calls`, `syntax_result`, etc.) marked deprecated, optional
- RAG pack works using generic fields

---

### Finding: Silent Exception Handling ✅ PARTIALLY ADDRESSED (2026-01-24)

**What was done:**
- Created exception hierarchy (`EvalKitError` base, domain-specific subclasses)
- Added structured error codes to all exceptions
- Exceptions now importable from `src/evalkit/exceptions.py`
- Backward compatibility maintained via re-exports

**Remaining:** Replace bare `pass` in `distributed/worker.py:176, 202` and `continuous/scheduler.py:174-175` with `logger.exception()`.

---

### Finding: Global Singleton Anti-pattern ✅ ADDRESSED (2026-01-24)

**Resolution:**
- Created `RepositoryProvider` class with proper lifecycle management
- Supports multiple concurrent providers (for multi-tenancy)
- Uses async context manager pattern
- Backward compatible - legacy `create_repositories()` still works
- 18 unit tests added

**New usage:**
```python
async with RepositoryProvider(StorageConfig(backend="memory")) as provider:
    test_case = await provider.test_cases.get("test-id")
```

---

### Finding: Regression Detector Incomplete ✅ FIXED (2026-01-24)

**Resolution:**
- Implemented `_get_previous_batch_metrics()` to find previous batch by client_type
- Queries database for most recent batch before current one
- Added 5 new unit tests
- Graceful error handling - returns `None` on database errors

---

## P1 Items Status

| Item | Status | Notes |
|------|--------|-------|
| 7. Incomplete pyproject.toml Metadata | Partial | Authors added, URLs pending repo decision |
| 8. MCP Client Stub Documentation | ✅ Done | Experimental notice added |
| 11. Deprecation Warnings | ✅ Done | `can_handle()` has warning |
