# Evalkit Release Audit

**Date:** 2026-01-21
**Updated:** 2026-01-31 (Comprehensive Audit)
**Status:** Phase 1 Complete, Phase 2-4 Identified
**Author:** Mostly Claude, with some minor assistance from Sid

---

## Executive Summary

**Evalkit** is a generic evaluation framework designed to support multiple evaluation packs (NL2API, RAG, code-gen, etc.). This audit focuses on the framework's readiness for public release.

| Aspect | Grade | Status |
|--------|-------|--------|
| **Single-process batch evaluation** | B+ | Production-ready |
| **Generic abstraction** | B+ | Core models support any pack via `input`/`expected` fields |
| **Performance** | C+ | Several optimization opportunities identified |
| **Code Quality** | C | Significant over-engineering and duplication |
| **Security** | B- | SQL injection safe, but error disclosure issues |
| **Test Coverage** | B | 700+ evalkit tests, some gaps in services |
| **Distributed multi-worker mode** | D | Critical gaps, not production-ready |
| **Enterprise features** | F | Missing auth, audit, tenant isolation |

**Overall: B- (2.9/5)** - Single-process is production-ready but codebase needs cleanup.

### What Works
- Single-process `BatchRunner` with checkpoint/resume
- Generic `TestCase.input`/`expected` and `Scorecard.stage_results` fields (NL2API fields are deprecated legacy)
- Proper concurrency handling (asyncio.Lock)
- Repository provider pattern (no singletons)
- OTEL telemetry integration
- PostgreSQL and in-memory backends
- RAG and NL2API packs both working
- SQL injection prevention (parameterized queries throughout)

### What's Broken/Needs Work
- Distributed workers: no heartbeat, no exactly-once delivery, no leader election
- No authentication, authorization, or audit logging
- Significant code duplication from incomplete migrations
- Performance bottlenecks in RAG retriever and batch evaluation
- Error messages leak internal details in HTTP endpoints

---

## Codebase Overview

**Total Codebase:**
- Source: 228 Python files (~58,300 lines)
- Tests: 212 files (~60,500 lines)
- Scripts: 45 files (~12,200 lines)
- Fixtures: 28 MB (generated test data)

---

## NEW FINDINGS (2026-01-31 Audit)

### Performance Issues (12 Found)

| Issue | Severity | Location | Impact |
|-------|----------|----------|--------|
| **SQL string interpolation in RAG** | ~~CRITICAL~~ | `rag/retriever/retriever.py` + `nl2api/rag/retriever.py` | ✅ FIXED (2026-01-31) - Now uses parameterized queries |
| **Missing scorecard indexes** | HIGH | `scorecard_repo.py` | Slow resume/batch queries |
| **In-memory test case filtering** | HIGH | `batch/runner.py:197` | Loads 10k records to filter in Python |
| **JSON serialization overhead** | MEDIUM | `test_case_repo.py:113-118` | 10-20% batch eval time on JSON encoding |
| **Vector string conversion** | MEDIUM | `retriever.py:185-187` | 1536 str() calls per embedding |
| **Metrics aggregation in memory** | MEDIUM | `runner.py:617-624` | Holds all scorecards in RAM |
| **Connection pool undersized** | MEDIUM | `postgres/client.py:27-28` | max_size=10 too low for concurrency |
| **Missing composite indexes** | MEDIUM | `migrations/002_rag_documents.sql` | Need (document_type, domain) index |
| **Checkpoint inside lock** | LOW | `runner.py:303-317` | DB write inside counter_lock |
| **Duplicate metadata parsing** | LOW | `retriever.py:274-285` (3x) | Same logic repeated 3 times |
| **Embedding before cache check** | LOW | `retriever.py:185` | Generates embedding before checking cache |
| **Entity resolver per-query** | LOW | `orchestrator.py:261` | No batch resolution |

**Recommended Fixes (Priority Order):**
1. ✅ Parameterize RAG query filters (affects every retrieval) - DONE 2026-01-31
2. ✅ Add scorecard composite indexes - DONE 2026-02-01 (migration 018)
3. Add `exclude_ids` parameter to repository list() method
4. Use asyncpg native JSONB handling

---

### Over-Engineering Issues (5 Remaining, ~550 lines removable)

| Issue | Scope | Lines | Location |
|-------|-------|-------|----------|
| ~~**src/evaluation → src/evalkit duplication**~~ | ~~CRITICAL~~ | ~~~2000~~ | ✅ FIXED (2026-01-31) - Removed 10,700 lines |
| ~~**src/rag_ui → src/rag/ui duplication**~~ | ~~CRITICAL~~ | ~~~300~~ | ✅ FIXED (2026-02-01) - Deleted src/rag_ui/, updated scripts |
| ~~**Duplicate circuit breaker**~~ | ~~HIGH~~ | ~~400~~ | ✅ FIXED (2026-02-01) - Deleted services/resilience.py, using evalkit |
| ~~**Exception hierarchy (28 classes)**~~ | ~~HIGH~~ | ~~394~~ | ✅ FIXED (2026-02-01) - Reduced to 15 classes, removed 12 unused |
| ~~**Duplicate RAG protocols**~~ | ~~MEDIUM~~ | ~~154~~ | ✅ FIXED (2026-02-01) - Re-export from canonical src/rag/retriever/protocols.py |
| ~~**Entity resolver multiplicity**~~ | ~~MEDIUM~~ | ~~400~~ | ✅ FIXED (2026-02-01) - Shared module in evalkit/common/entity_resolution/ |
| **Unused config options (~10)** | MEDIUM | ~100 | `routing_tier3_model`, `cohere_api_key`, `mcp_mode=hybrid`, etc. |
| **NoOp telemetry classes** | LOW | ~45 | Custom when OTEL SDK provides no-ops |
| **Overly complex factory** | LOW | ~50 | `rag/retriever/factory.py` - 105 lines for 2 class choices |
| **Entity resolver multiplicity** | MEDIUM | ~400 | 3 implementations, 1 used in practice |
| **Protocol over-specification** | LOW | ~250 | 36 methods for 3 implementations |
| **Compatibility shims (src/common, src/contracts)** | MEDIUM | ~100 | Empty directories with re-exports |

**Consolidation Plan:**
- Phase 1 (1-2 weeks): ✅ Complete src/evaluation and src/rag_ui migrations, delete old paths
- Phase 2 (1 week): Eliminate duplicate circuit breaker, consolidate protocols
- Phase 3 (2-3 weeks): Reduce exception hierarchy, remove dead config, simplify factories

---

### Security Issues (10 Found)

| Issue | Severity | Location | Status |
|-------|----------|----------|--------|
| **Error message information disclosure** | HIGH | `services/entity_resolution/transports/http/app.py` + `mcp_servers/entity_resolution/transports/sse.py` | ✅ FIXED (2026-01-31) |
| **Rate limiting fails open** | MEDIUM | `mcp_servers/entity_resolution/middleware.py:156-158` | Allows all requests if Redis down |
| ~~**Broad exception handling**~~ | ~~MEDIUM~~ | `rag/evaluation/stages/citation.py` | ✅ FIXED (2026-02-01) - Now catches specific exceptions with logging |
| **Unsanitized error strings in metrics** | MEDIUM | `evalkit/core/semantics/evaluator.py:259,270` + 8 others | Exception strings in telemetry |
| **Hardcoded test credentials** | MEDIUM | `mcp_servers/entity_resolution/__main__.py:66` + 2 | `postgresql://nl2api:nl2api@...` |
| **Missing rate limit on /stats** | LOW | `services/entity_resolution/transports/http/app.py:262` | Endpoint not rate-limited |
| **Exception strings in JSON responses** | LOW | `rag/evaluation/llm_judge.py:319,404` | Potential log/JSON injection |

**Positive Findings (No Issues):**
- SQL injection: All queries use parameterized `$1, $2` syntax
- No eval()/exec(): No dynamic code execution found
- No shell injection: subprocess calls use list format
- Configuration: Uses Pydantic BaseSettings with environment variables

**Immediate Actions:**
1. ✅ Replace `detail=str(e)` with generic error messages in HTTP endpoints (DONE 2026-01-31)
2. ✅ Parameterize RAG SQL queries to enable query plan caching (DONE 2026-01-31)
3. ✅ Document rate limiter fail-open behavior (DONE 2026-02-01) - Added security note in middleware.py
4. ✅ Replace broad `except Exception:` with specific exceptions (DONE 2026-02-01) - Fixed citation.py

---

### Test Coverage Assessment

**Test Distribution:**
- Unit tests: 163 files (evalkit: 700+, nl2api: 1000+, rag: 200+)
- Integration tests: 29 files
- Accuracy tests: 13 files (tier1/2/3)
- E2E tests: 7 files

**Coverage Gaps Identified:**
| Module | Status | Gap |
|--------|--------|-----|
| `src/evalkit/` | Good (700+ tests) | distributed/worker needs more edge cases |
| `src/nl2api/` | Good (1000+ tests) | conversation/manager has thin coverage |
| `src/rag/` | Moderate (200+ tests) | UI and ingestion modules need more tests |
| `src/services/` | Low | HTTP transport error paths undertested |
| `src/mcp_servers/` | Low | Only basic tool tests, no transport tests |

**Untested Files:**
- `src/services/entity_resolution/transports/sse.py` - No unit tests
- `src/mcp_servers/entity_resolution/transports/` - Minimal coverage
- `src/rag/ui/app.py` - Streamlit apps hard to unit test (manual verification required)

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
| **Secrets Management** | Weak | API keys in env vars, no SecretStr consistently |
| **Data Retention** | Missing | Scorecards never deleted |

**Remediation:**
- [ ] Add `tenant_id` parameter to all repository queries
- [ ] Implement PostgreSQL row-level security
- [ ] Add API key authentication layer
- [ ] Add audit log table and logging
- [ ] Use Pydantic `SecretStr` for all sensitive fields

---

#### 3. Code Consolidation Required (NEW)
**Severity: HIGH** - Technical debt impacting maintainability

| Migration | Files Affected | Action |
|-----------|----------------|--------|
| `src/evaluation/` → `src/evalkit/` | 15 shim files | ✅ DONE (2026-01-31) |
| `src/rag_ui/` → `src/rag/ui/` | 4 duplicate files | ✅ DONE (2026-02-01) |
| `src/common/` → `src/evalkit/common/` | Empty shim | Delete |
| `src/contracts/` → `src/evalkit/contracts/` | Re-export shim | Delete |
| Duplicate circuit breaker | 2 implementations | Use evalkit version only |
| Duplicate RAG protocols | 2 identical files | Keep one, import from there |

---

### MEDIUM PRIORITY - Incomplete Implementations

| Feature | Location | Issue |
|---------|----------|-------|
| **Email alerting** | `continuous/alerts.py:289-298` | Stub only |
| **Azure storage** | `storage/factory.py:79-83` | `NotImplementedError` |
| **Batch summary** | `scorecard_repo.py` | Incomplete implementation |
| **Remaining silent handlers** | `distributed/worker.py:176, 202`, `continuous/scheduler.py:174-175` | Bare `pass` needs logging |
| **Near-duplicate detection** | `validation/validators.py:253` | TODO comment |
| **RAG context in MCP** | `mcp_servers/entity_resolution/nl2api_tools.py:284` | TODO: Add RAG context |

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

### Phase 2: Code Consolidation (~2 weeks) - IN PROGRESS
| Task | Effort | Priority | Status |
|------|--------|----------|--------|
| Complete src/evaluation migration | 3 days | P0 | ✅ Done (2026-01-31) |
| Complete src/rag_ui migration | 1 day | P0 | ✅ Done (2026-02-01) |
| Consolidate circuit breaker | 1 day | P1 | Pending |
| Consolidate RAG protocols | 0.5 days | P1 | Pending |
| Remove unused config options | 1 day | P2 | Pending |

### Phase 3: Performance Optimization (~2 weeks) - NEW
| Task | Effort | Priority | Status |
|------|--------|----------|--------|
| Parameterize RAG query filters | 2 days | P0 | Pending |
| Add scorecard composite indexes | 1 day | P0 | Pending |
| Add exclude_ids to repository | 1 day | P1 | Pending |
| Optimize JSON serialization | 2 days | P2 | Pending |

### Phase 4: Security Hardening (~1 week) - NEW
| Task | Effort | Priority | Status |
|------|--------|----------|--------|
| Fix error message disclosure | 1 day | P0 | Pending |
| Document rate limiter fail-open | 0.5 days | P1 | Pending |
| Replace broad exception handlers | 2 days | P2 | Pending |

### Phase 5: Distributed Reliability (~4 weeks)
| Task | Effort | Priority | Status |
|------|--------|----------|--------|
| Implement exactly-once delivery | 2 weeks | P0 | Pending |
| Implement worker heartbeat | 1 week | P0 | Pending |
| Add coordinator leader election | 1 week | P1 | Pending |

### Phase 6: Enterprise Features (4-5 weeks)
| Task | Effort | Priority | Status |
|------|--------|----------|--------|
| Implement tenant isolation | 2 weeks | P0 | Pending |
| Add authentication layer | 2 weeks | P1 | Pending |
| Implement audit logging | 1 week | P2 | Pending |

**Remaining Effort: ~14 weeks** (Phase 2: 2w + Phase 3: 2w + Phase 4: 1w + Phase 5: 4w + Phase 6: 5w)

---

## Recommendation

Single-process batch evaluation is **production-ready** with:
- Generic `TestCase`/`Scorecard` models (supports any pack)
- Checkpoint/resume for interrupted runs
- Proper concurrency handling with locks
- Clean exception hierarchy
- Repository provider pattern (no singletons)

Both NL2API and RAG packs are working. New packs can be added using the generic `input`/`expected` and `stage_results` fields.

**Before public release, complete:**
1. Phase 2 (Code Consolidation) - Eliminate duplication, reduce maintenance burden
2. Phase 3 (Performance) - Fix critical performance issues
3. Phase 4 (Security) - Fix error disclosure

**Do not** deploy distributed mode (multi-worker) until Phase 5 is complete.

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

---

## Appendix: Detailed Findings

### Performance: RAG Query String Interpolation

**Location:** `src/rag/retriever/retriever.py` lines 197, 400, 568

**Problem:**
```python
domain_filter = f"AND domain = '{domain}'"
type_filter = f"AND document_type IN ({types})"

sql = f"""
    WITH vector_search AS (
        ...
        {type_filter}
        {domain_filter}
```

**Impact:** Every query misses PostgreSQL query plan cache. Should use parameterized filters.

---

### Over-Engineering: Exception Hierarchy ✅ FIXED (2026-02-01)

**Location:** `src/evalkit/exceptions.py` (originally 394 lines, 28 exception classes)

**Problem:** Extensive hierarchy but grep shows no code catches specific exceptions. All caught generically.

**Resolution:** Reduced from 28 to 15 exception classes by removing 12 unused:
- `InvalidConfigError`, `MissingConfigError` (use `ConfigurationError`)
- `EntityNotFoundError` (use `StorageError`)
- `EvaluationTimeoutError`, `StageEvaluationError`, `LLMJudgeError` (use `EvaluationError`)
- `WorkerError`, `CoordinatorError`, `MessageProcessingError` (use `DistributedError`)
- `RetryExhaustedError`, `MetricsError`, `TracingError` (not used anywhere)

Remaining hierarchy provides sufficient granularity via category exceptions + `error.code` attribute.

---

### Security: Error Message Disclosure

**Location:** `src/services/entity_resolution/transports/http/app.py` lines 225, 242, 260

**Problem:**
```python
except Exception as e:
    logger.exception(f"Error resolving entity: {e}")
    raise HTTPException(status_code=500, detail=str(e))  # Exposes internals!
```

**Fix:** Use `detail="An error occurred"` and log details server-side only.
