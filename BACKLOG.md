# Project Backlog

This file tracks all planned work, technical debt, and in-flight items for the NL2API project.

**Last Updated:** 2026-01-24 (Evalkit extraction Stage 1 complete, distributed workers Phases 1-3 complete)

---

## How to Use This File

1. **All planned work should be tracked here** - Before starting significant work, add it to the backlog
2. **Update status as work progresses** - Move items between sections as they're completed
3. **Mark items complete when done** - Move to "Completed" section with date
4. **Reference plan docs** - Link to detailed plans in `docs/plans/` for complex items

---

## Capabilities Evaluation Matrix

**Every capability needs evaluation.** This matrix tracks what we can measure.

| Capability | Description | Fixtures | Eval Mode | Baseline | Status |
|------------|-------------|----------|-----------|----------|--------|
| **Entity Resolution** | Map company names → RICs | 3,109 | `resolver` | **99.5%** | ✅ Evaluated |
| **Query Routing** | Route query → correct domain agent | 270 | `routing` | **88.9%** (95.2% excl. out-of-domain) | ✅ Evaluated |
| **Tool Selection** | Select correct tool within agent | 2,288 (complex) | `orchestrator` | — | ❌ Not loaded/run |
| **Entity Extraction** | Extract company names from NL query | 1,605 | `orchestrator` | — | ❌ Not loaded/run |
| **DatastreamAgent** | Price, time series, calculated fields | 3,715 (lookups) + 2,500 (temporal) | `orchestrator` | — | ❌ Not loaded/run |
| **EstimatesAgent** | I/B/E/S forecasts, recommendations | 0 | `orchestrator` | — | ❌ No fixtures |
| **FundamentalsAgent** | WC codes, TR codes, financials | 0 | `orchestrator` | — | ❌ No fixtures |
| **OfficersAgent** | Executives, compensation, governance | 0 | `orchestrator` | — | ❌ No fixtures |
| **ScreeningAgent** | SCREEN expressions, rankings | 274 | `orchestrator` | — | ❌ Not loaded/run |
| **Comparison Queries** | Compare multiple entities | 3,658 | `orchestrator` | — | ❌ Not loaded/run |
| **NL Response Gen** | Generate human-readable response | 0 | `orchestrator` | — | ❌ No fixtures (blocked by temporal) |
| **RAG Retrieval** | Retrieve relevant context | 0 | — | — | ❌ No fixtures |
| **Clarification Flow** | Handle ambiguous queries | 0 | — | — | ❌ No fixtures |
| **Tool Composition** | Multi-tool query chaining | 0 | `orchestrator` | — | ❌ No fixtures |
| **Multi-Turn Context** | Conversation context accumulation | 0 | `orchestrator` | — | ❌ No fixtures |
| **Cross-Client Comparison** | Claude vs ChatGPT vs Custom | — | `mcp_passthrough` | — | ❌ Not implemented |

### Legend
- ✅ **Evaluated** - Has fixtures, baseline established, tracking over time
- ⚠️ **Needs improvement** - Evaluated but accuracy below target
- ❌ **No fixtures** - Capability exists but no evaluation data
- ❌ **Not loaded/run** - Fixtures exist but not in database / not evaluated

### Priority Actions
1. **Load all fixtures** into database (not just entity_resolution)
2. **Run orchestrator evaluation** to get baselines for tool selection, agents
3. **Create fixtures** for missing capabilities (routing, extraction, RAG, per-agent)

---

## Architecture Overview

```
User Query
    │
    ▼
┌─────────────────────┐
│  Entity Resolution  │ ──► Database (2.9M entities) + fuzzy matching
│      ~10ms          │     99.5% accuracy on test fixtures
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│   FM-First Router   │ ──► LLM call with agents as tools
│     ~300ms          │     Cache: Redis L1 + pgvector L2
│    ~100 tokens      │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│ Ambiguity Detection │ ──► Rule-based patterns
│      ~5ms           │
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│  Context Retrieval  │ ──► RAG (pgvector) or MCP
│     ~50ms           │     Dual-mode: local/mcp/hybrid
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│    Domain Agent     │ ──► Rule-based OR LLM tool-calling
│    ~500ms           │     5 agents: datastream, estimates,
│   ~1000 tokens      │     fundamentals, officers, screening
└─────────────────────┘
    │
    ▼
┌─────────────────────┐
│     Tool Calls      │ ──► Returned to caller
└─────────────────────┘

Total: ~850ms, ~1100 tokens per request
```

---

## Up Next (Recommended Order)

| # | Item | Rationale |
|---|------|-----------|
| 1 | ~~**Evaluation Framework Generalization (Phases 1-2)**~~ | ✅ Stage 1 Complete - evalkit namespace created |
| 2 | ~~**NL2API Codebase Separation**~~ | ✅ Stage 1 Complete - merged with evalkit extraction |
| 3 | **RAG Evaluation Pack** | Unblocked - plan ready at docs/plans/rag-03-evaluation-design.md |
| 4 | **Distributed Evaluation Infrastructure (Phase 4+)** | Phases 1-3 complete, coordinator & batch API remaining |
| 5 | **Test Quality Improvements** | Critical - current tests don't validate correctness |

---

## Critical Priority (P0) - Production & Quality Blockers

### Hardcoded Credentials in Docker Compose ✅ COMPLETE
**Created:** 2026-01-21
**Completed:** 2026-01-22
**Severity:** CRITICAL

**Solution implemented:**
- [x] Moved all credentials to `.env.example` with placeholder values
- [x] Updated docker-compose.yml to use `${VAR:-default}` syntax
- [x] Added Redis AUTH configuration (optional via `REDIS_PASSWORD`)
- [x] All ports bound to `127.0.0.1` instead of `0.0.0.0`

**Files changed:**
- `.env.example` - Added Docker infrastructure credentials section
- `docker-compose.yml` - Environment variables with defaults, localhost binding

---

### Input Validation & Rate Limiting ✅ COMPLETE
**Created:** 2026-01-21
**Completed:** 2026-01-22
**Severity:** CRITICAL

**Solution implemented:**
- [x] Rate limiting middleware (in-memory and Redis-backed options)
- [x] Pydantic Field validation with max_length constraints
- [x] Request body size limits (1MB default)
- [x] Input validation for suspicious patterns (XSS, script injection)
- [x] 32 unit tests for security middleware

**Files created/changed:**
- `src/mcp_servers/entity_resolution/middleware.py` - NEW: Rate limiter + validator
- `src/mcp_servers/entity_resolution/transports/sse.py` - Added middleware integration
- `tests/unit/mcp_servers/test_security_middleware.py` - NEW: 32 tests

**Defaults:**
- 100 requests per minute per client
- 500 char max entity name
- 100 max batch size
- 2000 char max query length

---

### Test Quality: Tests Don't Validate Correctness
**Created:** 2026-01-21
**Status:** Not Started
**Severity:** CRITICAL
**Docs:** [docs/plans/eval-09-test-quality.md](docs/plans/eval-09-test-quality.md)

The test suite has structural issues that allow broken agents to pass:

**Problems:**
1. Fixture tests only check `can_handle()`, not `process()` - agents that claim they can handle queries but generate garbage pass
2. 121k fixtures but only ~100 sampled (0.08% coverage)
3. Coverage thresholds are 10-40% (absurdly low)
4. Mock LLM returns static response - can't test real behavior
5. Assertions like `len(result.tool_calls) >= 0` are always true
6. No execution stage - tool calls never validated against APIs

**Fix:**
- [ ] Add `process()` tests that validate tool call structure
- [ ] Increase fixture sampling to at least 1% (1,200 samples)
- [ ] Raise coverage thresholds to 70%+ per category
- [ ] Add semantic validation of tool call arguments
- [ ] Implement mock LLM that validates prompt/response pairs

---

### Health Checks: Liveness vs Readiness Separation ✅ COMPLETE
**Created:** 2026-01-21
**Completed:** 2026-01-22
**Severity:** HIGH

**Solution implemented:**
- [x] `/health` endpoint (liveness) - simple "alive" check
- [x] `/ready` endpoint (readiness) - checks server, database, Redis
- [x] Proper Kubernetes probe compatibility

**Files changed:**
- `src/mcp_servers/entity_resolution/transports/sse.py` - Added `/ready` endpoint, updated `/health`

**Readiness checks:**
- Server initialization
- Database connection via health resource
- Redis connection (if enabled)

---

### PII/Secrets Redaction in Logs ✅ COMPLETE
**Created:** 2026-01-21
**Completed:** 2026-01-22
**Severity:** HIGH

**Solution implemented:**
- [x] Created `src/common/logging/sanitizer.py` with `SanitizingFilter`
- [x] Redacts API keys (Anthropic, OpenAI, generic), passwords, Bearer tokens
- [x] Redacts connection strings (PostgreSQL, Redis, Azure)
- [x] Provides `configure_sanitized_logging()` for global setup
- [x] Provides `get_sanitized_logger()` for per-logger setup
- [x] 16 unit tests covering all redaction patterns

**Usage:**
```python
from src.common.logging import configure_sanitized_logging, get_sanitized_logger

# Global configuration
configure_sanitized_logging()

# Per-logger configuration
logger = get_sanitized_logger(__name__)
```

---

## High Priority (P0)

### Evaluation Framework Generalization (evalframework)
**Created:** 2026-01-22
**Status:** ✅ Stage 1 Complete
**Docs:** [docs/plans/eval-06-evalkit-extraction.md](docs/plans/eval-06-evalkit-extraction.md)

Transform the evaluation platform from an NL2API-specific tool into a general-purpose ML evaluation framework (`evalframework`).

**Problem:** NL2API assumptions baked into core layers block RAG, classification, and other evaluation use cases.

**Phase 1: Extract Generic Core (~1 week)**
- [ ] Genericize `TestCase` - `input: dict`, `expected: dict` instead of NL2API fields
- [ ] Genericize `Scorecard` - `stage_results: dict[str, StageResult]` instead of fixed 4 fields
- [ ] Create `EvaluationPack` protocol - interface for domain-specific evaluation
- [ ] Create `NL2APIPack` - refactor existing NL2API stages into a pack
- [ ] Update storage - store generic JSON, not NL2API-specific columns
- [ ] Database migration for generic columns
- [ ] Unit tests for generic core
- [ ] Integration tests for NL2APIPack backwards compatibility

**Phase 2: Simple API Facade (~3 days)**
- [ ] Create `Evaluator` facade class with sensible defaults
- [ ] Support `Evaluator(pack=NL2APIPack())` pattern
- [ ] Add `results.to_json()`, `results.to_dataframe()` exports
- [ ] Consolidate config classes into single `EvaluatorConfig`
- [ ] Unit tests for facade API
- [ ] End-to-end test: load fixtures → evaluate → export results

**Usage after completion:**
```python
from evalframework import Evaluator
from evalframework.packs import NL2APIPack

evaluator = Evaluator(pack=NL2APIPack())
results = await evaluator.evaluate(test_cases, my_system)
print(f"Accuracy: {results.accuracy:.2%}")
results.to_json("results.json")
```

**Depends on:** Nothing
**Blocks:** RAG Evaluation Pack (Phase 3)

---

### NL2API Codebase Separation
**Created:** 2026-01-22
**Status:** ✅ Stage 1 Complete (merged with evalkit extraction)
**Docs:** [docs/plans/eval-06-evalkit-extraction.md](docs/plans/eval-06-evalkit-extraction.md)

NL2API was the first concrete use case for the evaluation framework. The codebase should be reorganized to clearly separate:

1. **evalframework** - General-purpose evaluation infrastructure
   - Generic TestCase, Scorecard, EvaluationPack protocol
   - Batch runner, storage protocols, telemetry
   - CLI with `--pack` option

2. **nl2api** - Domain-specific implementation
   - NL2APIPack with 4 stages (Syntax, Logic, Execution, Semantics)
   - Domain agents, orchestrator, entity resolution
   - NL2API-specific fixtures and tests

**Tasks:**
- [ ] Create `src/evalframework/` directory structure
- [ ] Move generic evaluation code from `src/evaluation/` to `src/evalframework/`
- [ ] Create `src/evalframework/packs/nl2api/` for NL2API-specific code
- [ ] Update imports throughout codebase
- [ ] Update CLI entry points
- [ ] Consider separate package publishing (evalframework vs nl2api)

**Depends on:** Evaluation Framework Generalization (Phases 1-2)

---

### Distributed Evaluation Infrastructure
**Created:** 2026-01-21
**Status:** In Progress - Phases 1-3 Complete
**Docs:** [docs/plans/eval-05-distributed-workers.md](docs/plans/eval-05-distributed-workers.md)

Current evaluation runner (`BatchRunner`) uses `asyncio.Semaphore` for local concurrency but lacks true distributed processing capabilities.
- No task queue (Celery/BullMQ/Redis Streams)
- No worker nodes (single monolithic process)
- No distributed locking or state management

**Fix:**
- [ ] Implement `AzureBatchJobRepository` for cloud state storage
- [ ] Add Redis-based job queue (BullMQ or custom via Redis Streams)
- [ ] Create worker service entry point (`src.evaluation.worker`)
- [ ] Add `worker` service to `docker-compose.yml`
- [ ] Update `BatchRunner` to dispatch jobs to queue instead of local execution

---

### Multi-Client Evaluation Platform (Eval Matrix)
**Created:** 2026-01-21
**Status:** In Progress - Phase 0 Complete
**Docs:** [docs/plans/eval-03-multi-client-platform.md](docs/plans/eval-03-multi-client-platform.md)

Multi-dimensional evaluation framework for comparing components × LLMs × configs.

**Phase 0 (Foundation) - COMPLETE:**
- [x] Add client tracking to Scorecard (`client_type`, `client_version`, `eval_mode`)
- [x] Token tracking through orchestrator (`NL2APIResponse.input_tokens/output_tokens`)
- [x] Model-aware cost calculation (`src/evaluation/batch/pricing.py`)
- [x] Agent factory for component-level testing (`get_agent_by_name()`)
- [x] `eval matrix run` CLI command with component selection
- [x] `eval matrix compare` for side-by-side comparison
- [x] `--mode tool_only` and `--agent` options in batch run
- [x] Backfill script for historical scorecards

**Usage:**
```bash
# Compare agents across LLMs
eval matrix run --component datastream --llm claude-3-5-haiku-20241022 --limit 50
eval matrix run --component datastream --llm claude-3-5-sonnet-20241022 --limit 50
eval matrix compare --runs <id1>,<id2>
```

**Phase 1 (Continuous + MCP Passthrough):**
- [ ] Background daemon for scheduled evaluations
- [ ] MCP tool call logging middleware (external client tracking)
- [ ] MCP passthrough evaluator (evaluate logged tool calls)
- [ ] Regression detection with Slack alerting
- [ ] Model version change detection + auto-re-evaluation

**Plan:** [docs/plans/jiggly-cuddling-cosmos.md](~/.claude/plans/jiggly-cuddling-cosmos.md) (ready, ~10-15 days)

**Blocking:** Distributed Evaluation Infrastructure - Phase 1 daemon design will be replaced by workers + task queue

**Phase 2 (Multi-API):**
- [ ] API abstraction layer (support multiple data providers)
- [ ] Mock execution with recorded responses
- [ ] Cost tracking per API call

---

### Codebase Audit: Remove Early Development Hacks ✅ COMPLETE
**Created:** 2026-01-21
**Completed:** 2026-01-21

**Results:**
- Deleted `scripts/generate_company_mappings.py` (dead code - generated removed static mappings)
- Removed 32 unused imports across `src/` via `ruff --fix`
- Added logging to silent exception handler in `mcp/context.py`

**Kept (acceptable patterns):**
- Field code mappings in agents (API reference data, small & stable)
- `_classify_query_legacy` fallback (resilience for router failures)
- `can_handle()` deprecated methods (still used by legacy fallback)

---

### Temporal Evaluation Data Handling ✅ COMPLETE
**Created:** 2026-01-21
**Completed:** 2026-01-22
**Docs:** [docs/plans/eval-08-temporal-data.md](docs/plans/eval-08-temporal-data.md)

**Solution implemented:**
- [x] `DateResolver` - resolves relative dates (-1D, -1M, -1Y, FY0, FQ0) to absolute
- [x] `TemporalComparator` - wraps ASTComparator with 3 validation modes:
  - BEHAVIORAL: Accept any valid temporal expression
  - STRUCTURAL: Normalize to absolute dates before comparison (default)
  - DATA: Exact match required
- [x] Integrated into `LogicEvaluator` (uses TemporalComparator when not in DATA mode)
- [x] CLI options: `--temporal-mode`, `--evaluation-date`
- [x] 21 unit tests covering all modes and edge cases

**Files:**
- `src/evaluation/core/temporal/date_resolver.py`
- `src/evaluation/core/temporal/comparator.py`
- `tests/unit/evaluation/test_temporal_comparator.py`

**Usage:**
```bash
# Default (structural mode - normalizes dates)
eval batch run --limit 10

# Behavioral mode (accepts any valid temporal expression)
eval batch run --temporal-mode behavioral --limit 10

# With specific evaluation date
eval batch run --evaluation-date 2026-01-15 --limit 10
```

---

### Live API Integration for Evaluation
**Created:** 2026-01-21
**Status:** Unblocked - Ready to start
**Depends on:** ~~Temporal Evaluation Data Handling~~ ✅

Connect evaluation pipeline to real LSEG APIs to:
- Populate `expected_response` with real data
- Enable Stage 3 (Execution) evaluation
- Generate accurate `expected_nl_response` values

**Prerequisites complete:**
- [x] Temporal data handling (relative dates normalized before comparison)

**Remaining work:**
- [ ] LSEG API credentials and authentication
- [ ] API client implementation for Datastream, Eikon
- [ ] Response caching to reduce API costs
- [ ] Rate limiting compliance

---

## Medium Priority (P1)

### Architecture: CONTRACTS.py Decomposition ✅ COMPLETE
**Created:** 2026-01-21
**Completed:** 2026-01-22
**Severity:** MEDIUM

**Solution implemented:**
Split 1,428-line monolith into focused modules under `src/contracts/`:

| Module | Lines | Contents |
|--------|-------|----------|
| `core.py` | ~530 | FrozenDict, enums, ToolRegistry, ToolCall, TestCase, SystemResponse |
| `evaluation.py` | ~310 | StageResult, Scorecard, EvaluationConfig, Evaluator ABC |
| `worker.py` | ~120 | WorkerTask, BatchJob, WorkerConfig |
| `tenant.py` | ~220 | LLMProvider, TargetSystemConfig, TestSuite, Client, EvaluationRun |
| `storage.py` | ~70 | TableStorageEntity, IdempotencyRecord |
| `__init__.py` | ~70 | Re-exports all for backward compatibility |

**Backward compatibility preserved:**
- `CONTRACTS.py` (118 lines) now re-exports from `src/contracts/`
- All existing imports continue to work: `from CONTRACTS import TestCase`

**All 1370 unit tests pass after split.**

---

### Architecture: Orchestrator God Class (803 lines)
**Created:** 2026-01-21
**Status:** Not Started
**Severity:** MEDIUM
**Docs:** [docs/plans/nl2api-05-orchestrator-refactor.md](docs/plans/nl2api-05-orchestrator-refactor.md)

Orchestrator has several problems:
1. Hidden lazy initialization (`_router_initialized` flag)
2. 11-parameter `__init__` with complex conditionals
3. Error handling conflates failures with clarification requests
4. `_RAGContextAdapter` is an adapter that shouldn't exist

**Fix:**
- [ ] Extract router initialization to factory/builder
- [ ] Separate error handling from clarification flow
- [ ] Have RAGRetriever implement ContextProvider directly
- [ ] Split into NL2APIOrchestrator + OrchestratorBuilder

---

### Architecture: Protocols Defined But Not Enforced ✅ COMPLETE
**Created:** 2026-01-21
**Completed:** 2026-01-22
**Severity:** MEDIUM

**Solution implemented:**
- [x] All protocols now have `@runtime_checkable` decorator
- [x] `NL2APIOrchestrator.__init__` validates all protocol parameters with isinstance checks
- [x] `register_agent()` validates agent implements DomainAgent protocol
- [x] TypeError raised with helpful message if protocol not implemented

**Files changed:**
- `src/nl2api/routing/cache.py` - Added `@runtime_checkable` to RedisClient, PostgresPool
- `src/nl2api/orchestrator.py` - Added isinstance checks in `__init__` and `register_agent`
- Test mocks updated to implement full protocol interfaces

**Remaining:**
- [ ] Have BaseDomainAgent explicitly implement DomainAgent (minor)
- [ ] Remove deprecated `can_handle()` from protocol (breaking change, deferred)

---

### Architecture: God Classes > 500 Lines
**Created:** 2026-01-21
**Status:** Not Started
**Severity:** LOW

Six files exceed 500 lines:
- RAG Indexer: 1024 lines (should split field code vs query example indexing)
- Entity Repo: 887 lines (should split into EntityRepository, EntityAliasRepository, EntityMatcher)
- Orchestrator: 803 lines (see above)
- Batch Commands: 783 lines
- Metrics: 667 lines
- Scorecard Repo: 655 lines

**Fix:** Refactor incrementally as these files are touched.

---

### No Deployment Configuration (K8s/IaC)
**Created:** 2026-01-21
**Status:** Not Started
**Severity:** HIGH (for production)
**Docs:** [docs/plans/infra-02-deployment.md](docs/plans/infra-02-deployment.md)

No deployment infrastructure:
- No Kubernetes manifests
- No Terraform/CloudFormation templates
- No Helm charts
- No scaling guidelines
- No monitoring/alerting as code

**Fix:**
- [ ] Create `deploy/` directory with Kubernetes manifests or Helm charts
- [ ] Document deployment prerequisites
- [ ] Create deployment runbook
- [ ] Add resource requests/limits
- [ ] Add HPA policies

---

### No Alerting Rules Defined
**Created:** 2026-01-21
**Status:** Not Started
**Severity:** MEDIUM

Prometheus/Grafana exist but no alerting:
- No SLO/SLI definitions
- No alert rules (circuit breaker open, response time >5s)
- No trace sampling strategy

**Fix:**
- [ ] Define alerting rules (e.g., circuit breaker open, latency >5s)
- [ ] Add SLO dashboards (e.g., "99% of queries <2s")
- [ ] Implement trace sampling for production
- [ ] Add business metrics (entity resolution accuracy, query acceptance rate)

---

### Tool Composition Testing
**Created:** 2026-01-21
**Status:** Not Started
**Related:** Multi-Client Evaluation Platform

When queries require multiple tools in sequence (e.g., "Which S&P 500 company has highest P/E and what do analysts recommend?"):
- Test tool ordering (screening → estimates)
- Validate data flow between tools
- Handle partial failures (tool 1 succeeds, tool 2 fails)

**Requires:** `ToolChain` evaluation model, expected sequence matching, data flow validation.

---

### Ground Truth for Routing (Acceptable Routes)
**Created:** 2026-01-21
**Status:** Not Started
**Related:** Multi-Client Evaluation Platform

Multiple valid routes often exist for a query:
- "What's Apple's stock price?" → datastream OR fundamentals
- Both are "correct" but may differ in freshness/detail

**Task:** Define "acceptable route set" instead of single correct answer. Score routing as pass if any acceptable route is chosen.

---

### Semantic Equivalence Detection
**Created:** 2026-01-21
**Status:** Not Started
**Related:** Multi-Client Evaluation Platform

Different tool calls can produce equivalent results:
- `fundamentals_query(fields=["TR.PERatio"])` vs
- `fundamentals_query(fields=["TR.PriceClose", "TR.EPSActValue"])` + calculate

**Task:** Define canonical forms for common queries. Score efficiency separately from correctness.

---

### Multi-Turn Conversation Evaluation
**Created:** 2026-01-21
**Status:** Not Started

Current tests are single-turn. Real usage involves context:
- "Show me Apple's financials"
- "Compare to Microsoft" (requires context)
- "What about growth rates?" (requires context from turns 1 & 2)

**Task:** Create `ConversationTestCase` with multiple turns. Evaluate context accumulation and clarification flows.

---

### Cost & Token Attribution
**Created:** 2026-01-21
**Status:** Not Started
**Related:** Multi-Client Evaluation Platform

Track per-evaluation costs for orchestrator comparison:
- Input/output tokens per LLM call
- API calls made to data providers
- Estimated cost in USD using provider pricing
- Cost-efficiency metric: accuracy per dollar

---

### Latency SLOs & Budgets
**Created:** 2026-01-21
**Status:** Not Started
**Related:** Multi-Client Evaluation Platform

Different query types have different acceptable latencies:
- Interactive: < 2 seconds
- Batch: < 30 seconds
- Reports: < 5 minutes

**Task:** Define latency SLOs per query category. Track P50/P95/P99. Alert on violations.

---

### Prompt/Schema Versioning
**Created:** 2026-01-21
**Status:** Not Started

Tool schemas evolve (new parameters, renames). Test cases may fail on new schemas.

**Task:**
- Version tool schemas explicitly
- Migrate test cases when schema changes
- Support multiple schema versions simultaneously
- Track which test cases work with which versions

---

### Model Upgrade Detection
**Created:** 2026-01-21
**Status:** Not Started
**Related:** Multi-Client Evaluation Platform

When Claude/ChatGPT update models, accuracy may change without warning.

**Task:**
- Query model version at evaluation time
- Detect version changes automatically
- Trigger re-evaluation on version change
- Compare before/after metrics

---

### Haiku Routing Spike (P1.1) ✅ COMPLETE
**Created:** 2026-01-21
**Completed:** 2026-01-21
**Result:** ✅ **Haiku outperforms Sonnet at 1/10th cost**

| Model | Pass Rate | Cost/query | Notes |
|-------|-----------|------------|-------|
| **Sonnet** | 88.9% (240/270) | ~$0.003 | Baseline |
| **Haiku** | 94.1% (254/270) | ~$0.0003 | **Winner** |

**Recommendation:** Switch routing to Claude 3.5 Haiku for production.
- 5.2% accuracy improvement
- 10x cost reduction
- Slightly higher latency acceptable for routing

```bash
# Run routing evaluation with model override
.venv/bin/python -m src.evaluation.cli.main batch run --tag routing --mode routing --model claude-3-5-haiku-20241022
```

**Sonnet subcategory breakdown (for reference):**
| Category | Pass Rate | Notes |
|----------|-----------|-------|
| fundamentals_clear | 100% (40/40) | |
| officers_clear | 100% (25/25) | |
| screening_clear | 100% (25/25) | |
| estimates_clear | 97.5% (39/40) | |
| temporal_ambiguous | 96.7% (29/30) | |
| edge_cases | 96% (24/25) | |
| datastream_clear | 87.5% (35/40) | Some price queries routed to fundamentals |
| domain_boundary | 84% (21/25) | Ambiguous cross-domain queries |
| negative_cases | 10% (2/20) | Expected - model tries to help |

---

### Token Optimization (P1.3)
**Created:** 2026-01-20
**Status:** Deferred (Low ROI)
**Analyzed:** 2026-01-21

**Analysis Results:**
- Average agent prompt: ~720 tokens (range: 600-880)
- Total across 5 agents: ~3,620 tokens
- With Haiku at $0.25/1M input tokens: ~$0.0009/request for all agents
- Potential savings from compression: ~$0.0001/request

**Decision:** Not worth the complexity. Haiku's low cost makes token optimization low-priority. May revisit if traffic increases significantly or if we switch to more expensive models for agent processing.

**Original goal:** Compress prompts dynamically based on query type.

---

### Smart RAG Context Selection (P1.4)
**Created:** 2026-01-20
**Status:** Unblocked - Ready to start
**Updated:** 2026-01-21

**Prerequisites Complete:**
- [x] Embeddings generated: 463 field codes with 384-dim local embeddings
- [x] Local embedder implemented (sentence-transformers, no API key needed)
- [x] Hybrid search now functional (vector + keyword)

**Next steps:**
1. Establish baseline retrieval quality metrics
2. Implement query-type-aware weighting
3. Add reranking if needed

**Original goal:** Implement query-type-aware weighting and reranking.

---

### Full Synthetic Dataset for Pipeline Testing
**Created:** 2026-01-21
**Status:** Planned

Create small (100-500 cases) fully-hydrated synthetic dataset:
- Populated `expected_response` (synthetic but structurally valid)
- Populated `expected_nl_response` (LLM-generated)
- Used to pressure-test full evaluation pipeline (all 4 stages)

**Not for accuracy tracking** - only for infrastructure validation.

---

### OTEL Coverage Gaps ✅ COMPLETE
**Created:** 2026-01-21
**Completed:** 2026-01-21
**Docs:** [docs/plans/eval-07-standards-audit.md](docs/plans/eval-07-standards-audit.md)

**All components instrumented:**
- [x] LLM providers (claude.py, openai.py)
- [x] Domain agents (via base.py)
- [x] Evaluators
- [x] RAG retriever/indexer (`rag.retrieve`, `rag.retrieve_by_keyword`, `rag.index_field_codes_batch`)
- [x] Entity resolver (`entity.resolve`, `entity.resolve_single`)
- [x] DB repositories:
  - test_case_repo: `db.test_case.get`, `db.test_case.get_many`, `db.test_case.save`
  - scorecard_repo: `db.scorecard.get`, `db.scorecard.save`
  - batch_repo: `db.batch_job.create`, `db.batch_job.get`, `db.batch_job.update`, `db.batch_job.list_recent`
  - entity_repo: `db.entity.resolve`, `db.entity.resolve_batch`, `db.entity.search`, `db.entity.bulk_import`, `db.entity.bulk_import_aliases`
- [x] Redis cache (`cache.get`, `cache.set`)

---

### Integration Test Gaps ✅ COMPLETE
**Created:** 2026-01-21
**Reviewed:** 2026-01-21
**Status:** Already covered
**Docs:** [docs/plans/eval-07-standards-audit.md](docs/plans/eval-07-standards-audit.md)

**Existing coverage found:**
- [x] Repository CRUD (test_case, scorecard, batch): `tests/unit/storage/test_postgres_repos.py` (runs against real DB)
- [x] Entity repo: `tests/integration/test_entity_repo.py` (comprehensive CRUD, resolution, batch, search)
- [x] Batch runner: `tests/unit/test_batch_runner.py`
- [x] Entity resolver pipeline: `tests/integration/test_entity_repo.py` (resolve, resolve_batch)
- [x] Orchestrator E2E: `tests/integration/test_api_contract.py` (test_orchestrator_end_to_end)
- [x] RAG retrieval: `tests/integration/test_rag_retrieval.py`
- [x] Estimates queries: `tests/integration/test_estimates_queries.py`

**Note:** Some "unit" tests in `tests/unit/storage/` are actually integration tests (use real DB via db_pool fixture).

---

## Fast-Follow

### MCP Elicitation for Clarifications
**Created:** 2026-01-21
**Status:** Not Started
**Depends on:** NL2API MCP tools (in progress)

Currently, when a query is ambiguous, we return clarification questions in the response for Claude to present to the user. MCP supports server-initiated elicitation where the server can directly prompt the user for input.

**Task:** Implement MCP elicitation protocol to handle clarifications natively:
- Server sends elicitation request when query is ambiguous
- User responds directly to server
- Server continues processing with clarified input

**Benefit:** Tighter UX - clarifications handled at protocol level rather than requiring Claude to mediate.

---

### Domain MCP Tools - Live API Integration
**Created:** 2026-01-21
**Status:** Partially Complete
**Docs:** [docs/plans/nl2api-04-mcp-dual-mode.md](docs/plans/nl2api-04-mcp-dual-mode.md)

**Already implemented:**
- Entity resolution MCP server with `resolve_entity`, `resolve_entities_batch`, `extract_and_resolve`
- NL2API tools exposed via MCP: `nl2api_query`, `query_datastream`, `query_estimates`, `query_fundamentals`, `query_officers`, `query_screening`
- Stdio transport for Claude Desktop, SSE for HTTP

**Current limitation:** Tools return **simulated data** (LLM-generated placeholders), not live LSEG API responses.

**Remaining work:**
1. Connect domain agents to live LSEG APIs (Datastream, Eikon, etc.)
2. Add proper authentication/credential handling for LSEG APIs
3. Handle rate limiting and quotas
4. Add response caching to reduce API costs

**Testing:**
- Integration tests with live APIs (requires LSEG credentials)
- Compare simulated vs live response quality
- Measure latency impact of live API calls

**Benefit:** Enables real financial data queries through Claude Desktop.

---

## Low Priority (P2)

### Rule-Based Coverage Expansion (P2.0)
**Created:** 2026-01-20
**Status:** Not Started
**Demoted from:** P1.3

Only 15-50% of queries are handled by rules. Expand pattern coverage for common query types.

**Why demoted:** Lower ROI than routing optimization. Rules are fallback; LLM handles most queries anyway.

---

### pyproject.toml Repository URL
**Created:** 2026-01-21
**Status:** Pending (needs repo URL decision)

`pyproject.toml` and `CONTRIBUTING.md` still have `YOUR_USERNAME` placeholder. Update when public repo URL is finalized.

---

### CHANGELOG.md
**Created:** 2026-01-21
**Status:** Not Started

Add changelog following Keep a Changelog format. Start from current state or document all phases.

---

### Type Hint Completeness
**Created:** 2026-01-21
**Status:** Not Started

mypy runs with `continue-on-error: true` in CI. Fix mypy errors incrementally and remove the flag.

---

### Silent Exception Handlers
**Created:** 2026-01-21
**Status:** Not Started

19 occurrences of `except: pass` or `except Exception: pass` in the codebase. Add `logger.debug()` or `logger.warning()` for observability.

**Locations:** Primarily in `src/evaluation/cli/commands/batch.py` and `run.py` cleanup blocks.

---

### Streaming Support (P2.1)
**Created:** 2026-01-20
**Status:** Not Started

Users wait for full response. Implement streaming for better UX.

---

### A/B Testing Infrastructure (P2.2)
**Created:** 2026-01-20
**Updated:** 2026-01-21
**Status:** Not Started
**Related:** Multi-Client Evaluation Platform

No variant comparison capability. Add infrastructure to compare different approaches.

**Extended scope:**
- Calculate statistical significance (p-values, confidence intervals)
- Stratified sampling by query difficulty/category
- Control for confounding variables (query distribution, time of day)
- Report confidence intervals, not just point estimates
- Minimum sample sizes for significance

---

### User Feedback Loop (P2.3)
**Created:** 2026-01-20
**Updated:** 2026-01-21
**Status:** Not Started

No accuracy feedback mechanism from users. Add feedback collection and analysis.

**Extended scope:**
- Log all production queries + tool calls
- Flag failed/corrected queries automatically
- Auto-generate test cases from production errors
- Human review queue for edge cases
- Close the loop: production errors → test cases → improved accuracy

---

### Query Expansion/Rewriting (P2.4)
**Created:** 2026-01-20
**Status:** Not Started

Ambiguous queries reduce accuracy. Implement query rewriting for clarity.

---

### MCP Tool Execution
**Created:** 2026-01-20
**Status:** Experimental
**Docs:** [docs/plans/nl2api-04-mcp-dual-mode.md](docs/plans/nl2api-04-mcp-dual-mode.md)

MCP client has tool discovery implemented. Tool execution (`tools/call`) not yet implemented.

**Current state:** Marked experimental in code. See `src/nl2api/mcp/client.py`.

---

### Entity Resolution Test Coverage
**Created:** 2026-01-21
**Status:** Known gap

`tests/unit/nl2api/test_entity_resolution_fixtures.py` failing due to coverage threshold.
Need to either:
- Generate more entity resolution test cases
- Adjust coverage thresholds

---

### Azure AI Search Migration
**Created:** 2026-01-20
**Status:** Not Started

Migrate from pgvector to Azure AI Search for production scale.

---

## Research Spikes

### Fine-Tuning Feasibility
**Duration:** 2-3 weeks
**Hypothesis:** Domain-specific model outperforms general LLM
**Status:** Not Started

---

### Prompt Optimization (DSPy)
**Duration:** 2 weeks
**Hypothesis:** Automated optimization improves accuracy
**Status:** Not Started

---

### Embedding Model Comparison
**Duration:** 1 week
**Hypothesis:** Better embeddings improve RAG retrieval
**Status:** Not Started

---

### Custom Orchestrator vs Vanilla MCP Evaluation
**Duration:** 1-2 weeks
**Created:** 2026-01-21
**Status:** Not Started

**Question:** Is our custom NL2API orchestrator (routing → entity resolution → agent dispatch → tool generation) providing measurable value over a simpler "vanilla" MCP implementation where Claude directly uses tools?

**Hypothesis:** The orchestrator adds value through:
1. Domain-specific routing (vs Claude choosing tools ad-hoc)
2. Entity resolution accuracy (2.9M entity database)
3. Context injection (field codes, examples via RAG)
4. Structured tool call generation (consistent output format)

**How to measure:**
1. **Accuracy comparison:** Same test queries through orchestrator vs direct Claude tool use
2. **Cost comparison:** Token usage per query (orchestrator has multiple LLM calls)
3. **Latency comparison:** End-to-end response time
4. **Consistency:** Variance in tool call quality across repeated runs

**Vanilla MCP baseline:**
- Expose raw LSEG API tools directly to Claude
- Let Claude handle entity resolution, field selection, API construction
- Compare output quality against orchestrator

**Success criteria:** Define threshold where orchestrator justifies complexity (e.g., >10% accuracy improvement, or <2x cost for same accuracy)

---

## Completed

### 2026-01-22

- [x] **Temporal Data Handling** - `DateResolver` + `TemporalComparator` with 3 validation modes (behavioral, structural, data), integrated into LogicEvaluator, 21 tests
- [x] **Canonical Tool Names** - Standardized tool names (get_data vs datastream_get_data) and RIC ticker format, AST comparator with tool name normalization
- [x] **Protocol Enforcement** - All protocols have `@runtime_checkable`, NL2APIOrchestrator validates protocol conformance with isinstance checks
- [x] **PII/Secrets Redaction** - Created `src/common/logging/sanitizer.py` with SanitizingFilter for API keys, passwords, tokens, connection strings
- [x] **Health Checks: Liveness vs Readiness** - Added `/health` (liveness) and `/ready` (readiness) endpoints with server, database, and Redis checks for Kubernetes compatibility
- [x] **CONTRACTS.py Decomposition** - Split 1,428-line monolith into 5 focused modules under `src/contracts/` with backward-compatible re-exports

### 2026-01-21

- [x] **Haiku Routing Spike: Haiku wins** - Haiku achieves **94.1%** vs Sonnet's 88.9% on 270 routing cases at 1/10th cost. Recommendation: use Haiku for production routing.
- [x] **Routing Validation Benchmark (P1.1): 88.9% baseline** - 270 routing test cases, Claude Sonnet 4. By category: fundamentals/officers/screening 100%, estimates 97.5%, edge cases 96%, datastream 87.5%, negative (out-of-domain) 10%. Excluding out-of-domain: **95.2%**. Enables Haiku cost comparison spike.
- [x] **Entity Resolution (P0.2): 99.5% accuracy** - Database-backed resolution with 2.9M entities, pg_trgm fuzzy matching, multi-stage lookup (aliases → primary_name → ticker → fuzzy). See [edge cases doc](docs/plans/nl2api-03-entity-resolution-edge-cases.md) for remaining 0.5%.
- [x] Public Release Audit (P0): LICENSE, CONTRIBUTING, CODE_OF_CONDUCT, SECURITY, GitHub templates
- [x] pyproject.toml metadata for PyPI
- [x] MCP experimental notice + tracking in docs/status.md
- [x] Deprecation warning for `can_handle()` in base.py
- [x] Documentation reorganization (kebab-case naming in docs/plans/)
- [x] GLEIF/SEC API compatibility fixes (User-Agent headers, ZIP support)
- [x] Fixture contract alignment: Generators aligned with CONTRACTS.py, `_meta` blocks, `tool_name` field
- [x] Security fixes: SQL query building refactored (whitelist-based)
- [x] Error handling fixes: bare except, swallowed exceptions, exception chains
- [x] Tool selection eval data: 15,760 test cases loaded into PostgreSQL
- [x] OTEL metrics fix: Switched to Prometheus scrape pattern (port 8889)
- [x] Observability verification: Traces → Jaeger ✓, Metrics → Prometheus ✓, Scorecards → PostgreSQL ✓
- [x] Removed static mappings: Deleted company_mappings.json (109 companies), mappings.py, and fuzzy matching against static list - simplifies resolver to use database + OpenFIGI only
- [x] **Codebase Audit**: Deleted dead `generate_company_mappings.py`, removed 32 unused imports, added logging to silent exception handlers
- [x] **Batch API for Accuracy Tests**: Added Anthropic Batch API support (50% cheaper), retry with jitter, configurable real-time fallback

### 2026-01-20

- [x] P0.1 End-to-End Observability: OTEL stack, Grafana dashboards, tracing spans
- [x] P0.3 Request-Level Metrics: RequestMetrics, emitters, OTEL integration
- [x] Entity Resolution Expansion: 100+ static mappings, fuzzy matching, OpenFIGI integration
- [x] Economic Indicators Indexing: Bulk indexing support for 8,700+ synthetic indicators

### Earlier

- [x] Phase 1: Infrastructure (LLM abstraction, RAG retriever, orchestrator)
- [x] Phase 2: EstimatesAgent with full evaluation integration
- [x] Phase 3: Multi-turn conversations, domain agents
- [x] Phase 4: All 5 domain agents with fixture-based tests
- [x] Phase 5: Scale & Production (resilience, bulk indexing, Redis caching)

---

## Notes

### Priority Definitions

| Priority | Meaning |
|----------|---------|
| **P0** | Critical - blocks other work or has significant impact |
| **P1** | High - should be done in current/next sprint |
| **P2** | Medium - backlog for when capacity allows |
| **P3** | Low - nice to have, research items |

### Status Definitions

| Status | Meaning |
|--------|---------|
| **Not Started** | Planned but no work begun |
| **In Progress** | Actively being worked on |
| **Blocked** | Waiting on dependency |
| **Partially Complete** | Some work done, more remaining |
| **Planned** | Scheduled for specific timeframe |
| **Experimental** | Proof of concept, not production-ready |
