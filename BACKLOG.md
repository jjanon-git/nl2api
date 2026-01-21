# Project Backlog

This file tracks all planned work, technical debt, and in-flight items for the NL2API project.

**Last Updated:** 2026-01-21

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
| **Query Routing** | Route query → correct domain agent | 75 | `routing` | **94.7%** | ✅ Evaluated |
| **Tool Selection** | Select correct tool within agent | 2,288 (complex) | `orchestrator` | — | ❌ Not loaded/run |
| **Entity Extraction** | Extract company names from NL query | 0 | — | — | ❌ No fixtures |
| **DatastreamAgent** | Price, time series, calculated fields | 3,715 (lookups) + 2,500 (temporal) | `orchestrator` | — | ❌ Not loaded/run |
| **EstimatesAgent** | I/B/E/S forecasts, recommendations | 0 | `orchestrator` | — | ❌ No fixtures |
| **FundamentalsAgent** | WC codes, TR codes, financials | 0 | `orchestrator` | — | ❌ No fixtures |
| **OfficersAgent** | Executives, compensation, governance | 0 | `orchestrator` | — | ❌ No fixtures |
| **ScreeningAgent** | SCREEN expressions, rankings | 274 | `orchestrator` | — | ❌ Not loaded/run |
| **Comparison Queries** | Compare multiple entities | 3,658 | `orchestrator` | — | ❌ Not loaded/run |
| **NL Response Gen** | Generate human-readable response | 0 | `orchestrator` | — | ❌ No fixtures (blocked by temporal) |
| **RAG Retrieval** | Retrieve relevant context | 0 | — | — | ❌ No fixtures |
| **Clarification Flow** | Handle ambiguous queries | 0 | — | — | ❌ No fixtures |

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
| 1 | **Routing Validation Benchmark** | Can't improve what we can't measure. Must establish routing accuracy baseline. |
| 2 | **Haiku Routing Spike** | Potential 10x cost reduction if Haiku matches Sonnet for routing. Low effort to test. |
| 3 | **Temporal Data Handling** | Blocks live API integration which blocks NL response generation testing. |

---

## High Priority (P0)

### Codebase Audit: Remove Early Development Hacks
**Created:** 2026-01-21
**Status:** Not Started

Audit the codebase for patterns similar to the static mappings that were just removed:
- Small hardcoded lists that should be database-backed
- Fallback logic that masks real issues
- Redundant code paths that add complexity without value
- Dead code or unused parameters

**Example just removed:** Static company_mappings.json (109 companies) that was:
- Redundant with 2.9M entity database
- Causing fuzzy matching to only search 280 names
- Adding complexity with ticker_mappings + common_mappings code paths

**Areas to audit:**
- [ ] `src/nl2api/agents/` - Hardcoded patterns, fallback lists
- [ ] `src/nl2api/rag/` - Static retrieval configs
- [ ] `src/evaluation/` - Hardcoded thresholds or test data
- [ ] `scripts/` - One-off hacks that became permanent

---

### Temporal Evaluation Data Handling
**Created:** 2026-01-21
**Status:** Planning Required
**Docs:** [docs/plans/temporal-eval-data.md](docs/plans/temporal-eval-data.md)

Financial data changes constantly. Before live API integration, we need a strategy for:
- Handling stale expected values (stock prices change)
- Point-in-time vs current data queries
- Validation approaches (exact match, range-based, semantic)

**Blocking:** Live API integration for evaluation

**Next steps:**
1. Decide on validation approach (see options in plan doc)
2. Design schema changes if needed (temporal_type, valid_as_of, tolerance fields)
3. Implement temporal-aware validation logic

---

### Live API Integration for Evaluation
**Created:** 2026-01-21
**Status:** Blocked by temporal data handling
**Depends on:** Temporal Evaluation Data Handling

Connect evaluation pipeline to real LSEG APIs to:
- Populate `expected_response` with real data
- Enable Stage 3 (Execution) evaluation
- Generate accurate `expected_nl_response` values

---

## Medium Priority (P1)

### Haiku Routing Spike (P1.1)
**Created:** 2026-01-21
**Status:** Ready to Start
**Duration:** 1 week
**Hypothesis:** Haiku may be sufficient at 1/10th cost

Test if Claude 3.5 Haiku performs comparably to Sonnet for query routing.

**Baseline established:** Sonnet achieves **94.7%** on 75 routing test cases. Run same fixtures with Haiku to compare.

```bash
# Run routing evaluation (uses configured model)
.venv/bin/python -m src.evaluation.cli.main batch run --tag routing --mode routing
```

**Why promoted from Research:** High potential ROI, low implementation risk. Baseline now established.

---

### Token Optimization (P1.3)
**Created:** 2026-01-20
**Status:** Not Started

Static prompts waste 800-1200 tokens per agent. Compress prompts dynamically based on query type.

---

### Smart RAG Context Selection (P1.4)
**Created:** 2026-01-20
**Status:** Not Started

Fixed retrieval parameters don't adapt to query type. Implement query-type-aware weighting and reranking.

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

### OTEL Coverage Gaps
**Created:** 2026-01-21
**Status:** Partially Complete
**Docs:** [docs/plans/codebase-standards-audit.md](docs/plans/codebase-standards-audit.md)

**Completed:**
- [x] LLM providers (claude.py, openai.py)
- [x] Domain agents (via base.py)
- [x] Evaluators

**Remaining:**
- [ ] RAG retriever/indexer
- [ ] Entity resolver
- [ ] DB repositories
- [ ] Redis cache

---

### Integration Test Gaps
**Created:** 2026-01-21
**Status:** Not Started
**Docs:** [docs/plans/codebase-standards-audit.md](docs/plans/codebase-standards-audit.md)

Missing integration tests for:
- Repository CRUD operations (3 repos)
- Batch runner
- Entity resolver pipeline
- Orchestrator E2E

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
**Status:** Not Started

No variant comparison capability. Add infrastructure to compare different approaches.

---

### User Feedback Loop (P2.3)
**Created:** 2026-01-20
**Status:** Not Started

No accuracy feedback mechanism from users. Add feedback collection and analysis.

---

### Query Expansion/Rewriting (P2.4)
**Created:** 2026-01-20
**Status:** Not Started

Ambiguous queries reduce accuracy. Implement query rewriting for clarity.

---

### MCP Tool Execution
**Created:** 2026-01-20
**Status:** Experimental
**Docs:** [docs/plans/mcp-migration.md](docs/plans/mcp-migration.md)

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

## Completed

### 2026-01-21

- [x] **Routing Validation Benchmark (P1.1): 94.7% baseline** - 75 routing test cases, Claude Sonnet 4. Failures: 3 ambiguous cases + 1 PE ratio mis-route. Enables Haiku cost comparison spike.
- [x] **Entity Resolution (P0.2): 99.5% accuracy** - Database-backed resolution with 2.9M entities, pg_trgm fuzzy matching, multi-stage lookup (aliases → primary_name → ticker → fuzzy). See [edge cases doc](docs/plans/entity-resolution-edge-cases.md) for remaining 0.5%.
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
