# Project Status

**Last Updated:** 2026-01-20

## Overview

NL2API is a Natural Language to API translation system for LSEG financial data APIs. Translates natural language queries into structured API calls for Datastream, Estimates, Fundamentals, and other LSEG data services. Includes an evaluation framework for testing at scale.

---

## Current Phase: Phase 5 Complete - Scale & Production

### Implementation Progress

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1: Infrastructure** | ✅ Complete | LLM abstraction, RAG retriever, orchestrator skeleton |
| **Phase 2: EstimatesAgent** | ✅ Complete | First domain agent with full evaluation integration |
| **Phase 3: Multi-turn + Clarification** | ✅ Complete | Conversation sessions, query expansion, entity resolution |
| **Phase 4: Remaining Agents** | ✅ Complete | All 5 domain agents with comprehensive fixture-based tests |
| **Phase 5: Scale & Production** | ✅ Complete | Resilience patterns, bulk indexing, Redis caching |

---

## What's Implemented

### NL2API System Components

| Component | Status | Notes |
|-----------|--------|-------|
| **Orchestrator** | ✅ | Query classification, agent routing, entity resolution |
| **LLM Abstraction** | ✅ | Claude + OpenAI providers with tool-calling |
| **RAG Retriever** | ✅ | Hybrid vector + keyword search (pgvector) |
| **Entity Resolution** | ✅ | Pattern-based + static mappings, circuit breaker, Redis cache |
| **Clarification Flow** | ✅ | Ambiguity detection and question generation |
| **Multi-turn Conversations** | ✅ | Session management, query expansion |
| **Evaluation Adapter** | ✅ | Integrates with WaterfallEvaluator |

### Domain Agents

| Agent | Status | Capabilities |
|-------|--------|--------------|
| **EstimatesAgent** | ✅ | EPS, revenue, EBITDA forecasts, analyst recommendations |
| **DatastreamAgent** | ✅ | Price, volume, PE, market cap, historical time series |
| **FundamentalsAgent** | ✅ | Balance sheet, income statement, financial ratios |
| **OfficersAgent** | ✅ | Executives, compensation, board members, governance |
| **ScreeningAgent** | ✅ | SCREEN expressions, index constituents, TOP-N rankings |

---

### Phase 5 Components (Scale & Production)

| Component | File(s) | Description |
|-----------|---------|-------------|
| **Circuit Breaker** | `src/common/resilience/circuit_breaker.py` | Fail-fast pattern for external services |
| **Retry with Backoff** | `src/common/resilience/retry.py` | Exponential backoff for transient failures |
| **Redis Cache** | `src/common/cache/redis_cache.py` | L1/L2 caching with in-memory fallback |
| **Bulk Indexing** | `src/nl2api/rag/indexer.py` | COPY protocol for 10-50x faster inserts |
| **Checkpoint/Resume** | `src/nl2api/rag/checkpoint.py` | Resumable indexing for large jobs |
| **Rate Limiting** | `src/nl2api/rag/retriever.py` | OpenAI embedding rate limits |
| **Batch Saves** | `src/common/storage/postgres/scorecard_repo.py` | Efficient batch scorecard persistence |
| **Pool Health** | `src/common/storage/postgres/client.py` | Connection pool monitoring |
| **IVFFlat Index** | `src/common/storage/postgres/migrations/005_ivfflat_index.sql` | Vector index for 1M+ documents |

---

## Test Coverage

### Summary

```
Total Unit Tests:     606 (601 passed, 5 skipped)
├── NL2API Tests:     502
│   ├── LLM & Providers:      41
│   ├── RAG Retriever:        14
│   ├── Entity Resolver:      32
│   ├── Clarification:        27
│   ├── Orchestrator:         19
│   ├── Conversation:         45
│   ├── Eval Adapter:         18
│   ├── EstimatesAgent:       51
│   ├── DatastreamAgent:      36 + 26 fixture-based
│   ├── FundamentalsAgent:    49
│   ├── OfficersAgent:        41
│   ├── ScreeningAgent:       47 + 22 fixture-based
│   └── Fixture Coverage:     ~30 dynamic tests
├── Common Tests:      33 passing
│   ├── Resilience:           15 (circuit breaker, retry)
│   └── Cache:                18 (Redis, memory cache)
└── Evaluation Tests:  71 passing
```

### Dynamic Fixture-Based Testing

The test suite uses **programmatic fixture expansion** - tests automatically scale as test data grows.

```
Generated Test Fixtures: 12,887 total
├── lookups/       3,745 cases
├── temporal/      2,727 cases
├── comparisons/   3,658 cases
├── screening/       265 cases
├── complex/       2,277 cases
└── errors/          215 cases
```

**How It Works:**

1. `FixtureLoader` discovers all fixture categories at runtime
2. `CoverageRegistry` defines minimum coverage thresholds per category
3. Parameterized tests auto-generate from fixture structure
4. Tests fail if agent coverage drops below thresholds
5. Growth detection alerts on fixture count changes

**Key Test Files:**

| File | Purpose |
|------|---------|
| `fixture_loader.py` | Loads fixtures from `tests/fixtures/lseg/generated/` |
| `test_fixture_coverage.py` | Dynamic coverage enforcement |
| `test_datastream_fixtures.py` | DatastreamAgent against 6,000+ fixtures |
| `test_screening_fixtures.py` | ScreeningAgent against 265 fixtures |

---

## File Structure

```
nl2api/
├── CONTRACTS.py                 # Shared data models
├── docs/status.md               # This file
├── README.md                    # Project overview
│
├── src/
│   ├── nl2api/                  # NL2API System
│   │   ├── orchestrator.py      # Main entry point
│   │   ├── config.py            # Configuration
│   │   ├── models.py            # Response models
│   │   ├── llm/                 # LLM providers
│   │   │   ├── protocols.py     # LLMProvider protocol
│   │   │   ├── claude.py        # ClaudeProvider
│   │   │   ├── openai.py        # OpenAIProvider
│   │   │   └── factory.py       # Provider factory
│   │   ├── agents/              # Domain agents
│   │   │   ├── protocols.py     # DomainAgent protocol
│   │   │   ├── base.py          # BaseDomainAgent
│   │   │   ├── datastream.py    # DatastreamAgent ✅
│   │   │   ├── estimates.py     # EstimatesAgent ✅
│   │   │   ├── fundamentals.py  # FundamentalsAgent ✅
│   │   │   ├── officers.py      # OfficersAgent ✅
│   │   │   └── screening.py     # ScreeningAgent ✅
│   │   ├── rag/                 # RAG retrieval
│   │   │   ├── protocols.py
│   │   │   ├── retriever.py     # + Redis cache, rate limiting
│   │   │   ├── indexer.py       # + Bulk insert, progress
│   │   │   └── checkpoint.py    # NEW: Resumable indexing
│   │   ├── resolution/          # Entity resolution
│   │   │   ├── protocols.py
│   │   │   └── resolver.py
│   │   ├── clarification/       # Ambiguity handling
│   │   │   └── detector.py
│   │   ├── conversation/        # Multi-turn support ✅
│   │   │   ├── manager.py
│   │   │   ├── expander.py
│   │   │   ├── storage.py
│   │   │   └── models.py
│   │   └── evaluation/          # Eval integration
│   │       └── adapter.py
│   │
│   ├── common/
│   │   ├── resilience/          # Phase 5: Resilience patterns
│   │   │   ├── circuit_breaker.py
│   │   │   └── retry.py
│   │   ├── cache/               # Phase 5: Caching layer
│   │   │   └── redis_cache.py
│   │   └── storage/             # Shared storage layer
│   │
│   └── evaluation/              # Evaluation pipeline
│       ├── core/                # Evaluators
│       └── batch/               # Batch runner
│
├── tests/
│   ├── unit/
│   │   ├── nl2api/              # 426 NL2API unit tests
│   │   │   ├── fixture_loader.py
│   │   │   ├── test_fixture_coverage.py
│   │   │   ├── test_datastream_fixtures.py
│   │   │   └── test_screening_fixtures.py
│   │   └── common/              # 33 common tests (resilience, cache)
│   │       ├── test_resilience.py
│   │       └── test_cache.py
│   └── fixtures/lseg/generated/ # 12,887 test fixtures
│       ├── lookups/
│       ├── temporal/
│       ├── comparisons/
│       ├── screening/
│       ├── complex/
│       └── errors/
```

---

## Running Tests

```bash
# All unit tests
.venv/bin/python -m pytest tests/unit/ -v

# NL2API tests only
.venv/bin/python -m pytest tests/unit/nl2api/ -v

# Fixture coverage tests
.venv/bin/python -m pytest tests/unit/nl2api/test_fixture_coverage.py -v

# With coverage report
.venv/bin/python -m pytest tests/unit/nl2api/ -v --tb=short 2>&1 | tail -20
```

---

## Running Real LLM Evaluation

```bash
# Set API key
export NL2API_ANTHROPIC_API_KEY="sk-ant-..."

# Run evaluation (default: 50 test cases)
python scripts/run_estimates_eval.py --limit 50

# Use OpenAI instead
export NL2API_LLM_PROVIDER="openai"
export NL2API_OPENAI_API_KEY="sk-..."
python scripts/run_estimates_eval.py --limit 50
```

---

### Recently Implemented (Jan 20, 2026)

| Task | Status | Description |
|------|--------|-------------|
| **Entity Resolution Expansion** | ✅ | 100+ static mappings, fuzzy matching (rapidfuzz), OpenFIGI integration |
| **Economic Indicators Indexing** | ✅ | Bulk indexing support for 8,700+ synthetic indicators (scale-ready) |
| **Request Metrics (P0.3)** | ✅ | Accuracy measurement, emitters, OTEL integration |

### Experimental Features

| Feature | Status | Description |
|---------|--------|-------------|
| **MCP Client** | Experimental | Model Context Protocol client for future MCP server integration. Tool discovery implemented, tool execution pending. See `src/nl2api/mcp/`. |
| **Evaluation Stage 3** | Planned | Live API execution verification. Currently passes through. |
| **Evaluation Stage 4** | Planned | LLM-as-Judge semantic comparison. Currently passes through. |

### Next Steps (TODO)

### Production Integration

| Task | Priority | Description |
|------|----------|-------------|
| **Azure AI Search** | MEDIUM | Migrate from pgvector to Azure AI Search for production scale |
| **Production Deployment** | MEDIUM | Azure infrastructure, monitoring, alerting |
| **MCP Tool Execution** | LOW | Complete MCP client implementation for tool execution |

### Agent Improvements

| Task | Priority | Description |
|------|----------|-------------|
| **Increase Agent Coverage** | HIGH | Current coverage 15-50% - expand rule-based detection for common queries |
| **Improve SCREEN Expressions** | MEDIUM | More complex filter combinations |
| **Real LLM Evaluation** | MEDIUM | Run against Claude/GPT-4 for accuracy metrics |

---

## Known Limitations

1. **Entity Resolution**: Uses static company→RIC mappings (~30 companies). External API integration ready (circuit breaker, retry, Redis cache in place) - just needs endpoint configuration.

2. **Rule-Based Coverage**: Agents cover ~15-50% of queries via rules. Complex queries fall back to LLM.

3. **Zero Coverage Categories**: `complex/` and `errors/` categories have expected zero coverage (advanced features).

4. **Local RAG Only**: Current RAG uses local pgvector with IVFFlat index. Azure AI Search planned for production.
