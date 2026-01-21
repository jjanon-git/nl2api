# Codebase Standards Audit Report

**Date:** 2026-01-21
**Audited Against:** CLAUDE.md standards (commit b218d2f)

## Executive Summary

| Area | Status | Critical | High | Medium | Low |
|------|--------|----------|------|--------|-----|
| Integration Tests | FAIL | 3 | 8 | 4 | 3 |
| Security | WARN | 1 | 0 | 0 | 1 |
| Error Handling | WARN | 1 | 5 | 2 | 1 |
| Telemetry/OTEL | FAIL | 4 | 8 | 5 | 0 |
| Evaluation Standards | FAIL | 3 | 2 | 1 | 0 |

---

## 1. Integration Test Coverage Gaps

### P0 (Critical) - Database Repositories

| Component | Location | Current State | Gap |
|-----------|----------|---------------|-----|
| PostgresTestCaseRepository | `src/common/storage/postgres/test_case_repo.py` | Unit only (mocked pool) | Full CRUD integration |
| PostgresScorecardRepository | `src/common/storage/postgres/scorecard_repo.py` | Unit only (mocked pool) | Full CRUD integration |
| PostgresBatchJobRepository | `src/common/storage/postgres/batch_repo.py` | Unit only (mocked pool) | Full CRUD integration |

### P0 (Critical) - Orchestration

| Component | Location | Current State | Gap |
|-----------|----------|---------------|-----|
| NL2APIOrchestrator | `src/nl2api/orchestrator.py` | API contract only | Full E2E with DB |

### P1 (High) - Core Components

| Component | Location | Current State | Gap |
|-----------|----------|---------------|-----|
| RAG Indexer | `src/nl2api/rag/indexer.py` | **No tests** | Bulk indexing, checkpoint |
| Batch Runner | `src/evaluation/batch/runner.py` | Unit only | DB persistence |
| Entity Resolver pipeline | `src/nl2api/resolution/` | Entity repo only | Full resolution flow |
| Domain Agents (5) | `src/nl2api/agents/*.py` | API contract only | Full E2E |

### P2 (Medium) - External Services

| Component | Location | Current State | Gap |
|-----------|----------|---------------|-----|
| Redis Cache | `src/common/cache/redis_cache.py` | MemoryCache only | Real Redis |
| Config Loading | `src/nl2api/config.py` | Partial | Env var validation |
| DB Connection Pool | `src/common/storage/postgres/client.py` | Indirect | Pool lifecycle |
| Conversation Storage | `src/nl2api/conversation/storage.py` | Unit only | DB persistence |

### P3 (Low) - Resilience & Observability

| Component | Location | Current State | Gap |
|-----------|----------|---------------|-----|
| Circuit Breaker | `src/common/resilience/circuit_breaker.py` | Mocked failures | Real failure scenarios |
| Retry with Backoff | `src/common/resilience/retry.py` | Mocked | Real delays |
| OTEL Collector | `src/common/telemetry/` | Unit only | Real collector |

---

## 2. Security Violations

### P0 (Critical) - SQL Injection Pattern

Dynamic WHERE clause construction via f-strings. While parameters use `$1, $2` binding, the WHERE structure is built dynamically.

| File | Line | Code Pattern |
|------|------|--------------|
| `src/common/storage/postgres/test_case_repo.py` | 174-181 | `f"WHERE {where_clause}"` in list() |
| `src/common/storage/postgres/test_case_repo.py` | 250 | `f"WHERE {where_clause}"` in count() |
| `src/common/storage/postgres/entity_repo.py` | 699-708 | `f"WHERE {' AND '.join(conditions)}"` in search() |

**Risk:** Medium-High. Conditions are currently hardcoded, but pattern is vulnerable to future misuse.

**Fix:** Refactor to use query builder or validate conditions against whitelist.

### P3 (Low) - Development Defaults

| File | Line | Issue |
|------|------|-------|
| `src/nl2api/config.py` | 146 | Default DB URL with dev credentials |

**Status:** Acceptable for development, needs documentation.

### Positive Findings

- No hardcoded API keys or secrets in source code
- Proper use of environment variables via pydantic-settings
- No secrets logged
- Comprehensive input validation in `src/nl2api/ingestion/validation.py`

---

## 3. Error Handling Violations

### P0 (Critical) - Bare Except

| File | Line | Code |
|------|------|------|
| `tests/integration/test_estimates_full_eval.py` | 152 | `except:` without type |

### P1 (High) - Swallowed Exceptions

| File | Line | Pattern |
|------|------|---------|
| `src/evaluation/cli/commands/batch.py` | 145 | `except Exception: pass` |
| `src/evaluation/cli/commands/batch.py` | 220 | `except Exception: pass` |
| `src/evaluation/cli/commands/batch.py` | 363 | `except Exception: pass` |
| `src/evaluation/cli/commands/batch.py` | 437 | `except Exception: pass` |
| `src/evaluation/cli/commands/run.py` | 187 | `except Exception: pass` |

**Context:** All in cleanup/finally blocks for `close_repositories()`. Should log at warning level.

### P2 (Medium) - Anti-patterns

| File | Line | Issue |
|------|------|-------|
| `src/nl2api/mcp/client.py` | 137-138 | Log-and-raise (logs error then raises) |
| `src/nl2api/llm/claude.py` | 214 | Missing `from e` exception chain |

### P3 (Low) - Health Check Patterns

| File | Line | Issue |
|------|------|-------|
| `src/common/storage/postgres/client.py` | 142 | Broad `except Exception` for health check |
| `src/nl2api/mcp/client.py` | 431 | `except Exception: return False` |

**Context:** Acceptable for health checks but could be more specific.

---

## 4. Telemetry/OTEL Coverage Gaps

### P0 (Critical) - External API Calls

| Component | File | Missing |
|-----------|------|---------|
| Claude Provider | `src/nl2api/llm/claude.py` | No spans for LLM API calls |
| OpenAI Provider | `src/nl2api/llm/openai.py` | No spans for LLM API calls |
| OpenFIGI Resolver | `src/nl2api/resolution/openfigi.py` | No spans for external API |
| Entity Resolver | `src/nl2api/resolution/resolver.py` | No spans for resolution |

### P1 (High) - Multi-step Orchestration

| Component | File | Missing |
|-----------|------|---------|
| DatastreamAgent | `src/nl2api/agents/datastream.py` | No spans |
| EstimatesAgent | `src/nl2api/agents/estimates.py` | No spans |
| FundamentalsAgent | `src/nl2api/agents/fundamentals.py` | No spans |
| OfficersAgent | `src/nl2api/agents/officers.py` | No spans |
| ScreeningAgent | `src/nl2api/agents/screening.py` | No spans |
| RAG Retriever | `src/nl2api/rag/retriever.py` | No DB query spans |
| RAG Indexer | `src/nl2api/rag/indexer.py` | No bulk insert spans |
| Evaluators | `src/evaluation/core/evaluators.py` | No stage spans |

### P2 (Medium) - Database & Cache

| Component | File | Missing |
|-----------|------|---------|
| TestCaseRepository | `src/common/storage/postgres/test_case_repo.py` | No DB operation spans |
| ScorecardRepository | `src/common/storage/postgres/scorecard_repo.py` | No DB operation spans |
| BatchJobRepository | `src/common/storage/postgres/batch_repo.py` | No DB operation spans |
| Redis Cache | `src/common/cache/redis_cache.py` | No cache hit/miss spans |
| Batch Runner | `src/evaluation/batch/runner.py` | Metrics only, no tracing |

### Existing Coverage (Reference)

- `src/nl2api/orchestrator.py` - Comprehensive tracing
- `src/nl2api/routing/llm_router.py` - Has `trace_span()`
- `src/evaluation/batch/metrics.py` - Has metrics (not tracing)

---

## 5. Evaluation Standards Violations

### P0 (Critical) - Test Fixture Contract Violations

All 12,887 generated fixtures in `tests/fixtures/lseg/generated/` violate the TestCase contract:

| Field | Expected | Actual |
|-------|----------|--------|
| `expected_nl_response` | Required (min_length=1) | **MISSING** |
| `metadata.api_version` | Required (pattern `v1.0.0`) | **MISSING** |
| `metadata.complexity_level` | Required (1-5) | At root as `complexity` |
| `metadata.tags` | Required array | At root level |
| Tool call `tool_name` | Required field | Uses `function` instead |

**Impact:** Fixtures cannot deserialize to TestCase objects per CONTRACTS.py.

### P1 (High) - Evaluator OTEL Integration

| Issue | Location |
|-------|----------|
| No OTEL spans in evaluators | `src/evaluation/core/evaluators.py` (entire file) |
| No stage-level metrics | All evaluator classes |

### P2 (Medium) - Minor Issues

| Issue | Location |
|-------|----------|
| Missing explicit `error_code=None` | `evaluators.py:302-322` (stub stages) |

### Positive Findings

- Scorecard immutability respected (`model_copy()` used correctly)
- StageResult frozen models working
- BatchMetrics properly used in runner.py

---

## Remediation Plan

### Phase 1: P0 Security (Immediate) - COMPLETED

- [x] Refactor SQL query building in `test_case_repo.py` - Added whitelist-based `_build_filter_query()` helper
- [x] Refactor SQL query building in `entity_repo.py` - Added whitelist-based `_build_search_query()` helper

### Phase 2: P1 OTEL Coverage - PARTIAL

- [x] Add spans to LLM providers (claude.py, openai.py) - Full tracing with token metrics
- [x] Add spans to domain agents (via base.py) - All 5 agents now traced
- [x] Add spans to evaluators - Pipeline, syntax, and logic stages traced
- [ ] Add spans to RAG retriever/indexer
- [ ] Add spans to entity resolver

### Phase 3: P1 Integration Tests

- [ ] Repository integration tests (3 repos)
- [ ] Batch runner integration test
- [ ] Entity resolver pipeline test

### Phase 4: P2 Error Handling - COMPLETED

- [x] Fix bare except in test file (`test_estimates_full_eval.py:152`)
- [x] Add logging to swallowed exceptions in cleanup blocks (batch.py, run.py)
- [x] Fix missing exception chains (claude.py, openai.py)

### Phase 5: P2 OTEL & Integration

- [ ] Add spans to DB repositories
- [ ] Add spans to Redis cache
- [ ] Add Redis integration tests
- [ ] Add config loading tests

### Phase 6: P0 Evaluation (Requires Discussion)

- [ ] Fixture contract migration strategy
- [ ] Fixture regeneration or migration script
- [ ] Backward compatibility approach

---

## Appendix: Files Requiring Changes

### Security Fixes
- `src/common/storage/postgres/test_case_repo.py`
- `src/common/storage/postgres/entity_repo.py`

### OTEL Additions
- `src/nl2api/llm/claude.py`
- `src/nl2api/llm/openai.py`
- `src/nl2api/agents/base.py`
- `src/nl2api/agents/datastream.py`
- `src/nl2api/agents/estimates.py`
- `src/nl2api/agents/fundamentals.py`
- `src/nl2api/agents/officers.py`
- `src/nl2api/agents/screening.py`
- `src/nl2api/rag/retriever.py`
- `src/nl2api/rag/indexer.py`
- `src/nl2api/resolution/resolver.py`
- `src/nl2api/resolution/openfigi.py`
- `src/evaluation/core/evaluators.py`
- `src/common/storage/postgres/test_case_repo.py`
- `src/common/storage/postgres/scorecard_repo.py`
- `src/common/storage/postgres/batch_repo.py`
- `src/common/cache/redis_cache.py`

### Error Handling Fixes
- `tests/integration/test_estimates_full_eval.py`
- `src/evaluation/cli/commands/batch.py`
- `src/evaluation/cli/commands/run.py`
- `src/nl2api/mcp/client.py`
- `src/nl2api/llm/claude.py`

### New Integration Tests
- `tests/integration/storage/test_test_case_repo.py`
- `tests/integration/storage/test_scorecard_repo.py`
- `tests/integration/storage/test_batch_repo.py`
- `tests/integration/nl2api/test_orchestrator_e2e.py`
- `tests/integration/nl2api/test_batch_runner.py`
