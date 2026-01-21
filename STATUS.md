# Project Status

**Last Updated:** 2026-01-20

## Overview

EvalPlatform is a distributed evaluation framework for testing LLM tool-calling, with an embedded NL2API system for translating natural language queries into LSEG financial API calls.

---

## Current Phase: Phase 4 Complete - All Domain Agents Implemented

### Implementation Progress

| Phase | Status | Description |
|-------|--------|-------------|
| **Phase 1: Infrastructure** | ✅ Complete | LLM abstraction, RAG retriever, orchestrator skeleton |
| **Phase 2: EstimatesAgent** | ✅ Complete | First domain agent with full evaluation integration |
| **Phase 3: Multi-turn + Clarification** | ✅ Complete | Conversation sessions, query expansion, entity resolution |
| **Phase 4: Remaining Agents** | ✅ Complete | All 5 domain agents with comprehensive fixture-based tests |
| **Phase 5: Scale & Production** | ⏳ Not Started | Index economic indicators, production optimization |

---

## What's Implemented

### NL2API System Components

| Component | Status | Notes |
|-----------|--------|-------|
| **Orchestrator** | ✅ | Query classification, agent routing, entity resolution |
| **LLM Abstraction** | ✅ | Claude + OpenAI providers with tool-calling |
| **RAG Retriever** | ✅ | Hybrid vector + keyword search (pgvector) |
| **Entity Resolution** | ✅ | Pattern-based + static mappings (~30 companies) |
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

## Test Coverage

### Summary

```
Total Unit Tests:     497 passing
├── NL2API Tests:     426 passing
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
evalPlatform/
├── CONTRACTS.py                 # Shared data models
├── STATUS.md                    # This file
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
│   │   │   ├── retriever.py
│   │   │   └── indexer.py
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
│   ├── common/storage/          # Shared storage layer
│   └── evaluation/              # Evaluation pipeline
│       ├── core/                # Evaluators
│       └── batch/               # Batch runner
│
├── tests/
│   ├── unit/nl2api/             # 426 unit tests
│   │   ├── fixture_loader.py    # Fixture loading utility
│   │   ├── test_fixture_coverage.py    # Dynamic coverage tests
│   │   ├── test_datastream_fixtures.py # DatastreamAgent fixtures
│   │   └── test_screening_fixtures.py  # ScreeningAgent fixtures
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

## Next Steps

### Phase 5: Scale & Production

1. **Index Economic Indicators** - ~1M indicators for RAG
2. **Performance Optimization** - Caching, batch processing
3. **External Entity Resolution** - Integrate company→RIC API
4. **Production Deployment** - Azure infrastructure, monitoring

### Improvements

1. **Increase Agent Coverage** - Current coverage 15-50% depending on category
2. **Add More Field Patterns** - Expand rule-based detection
3. **Improve SCREEN Expressions** - More complex filter combinations
4. **Real LLM Evaluation** - Run against Claude/GPT-4 for accuracy metrics

---

## Known Limitations

1. **Entity Resolution**: Uses static company→RIC mappings (~30 companies). External API planned.

2. **Rule-Based Coverage**: Agents cover ~15-50% of queries via rules. Complex queries fall back to LLM.

3. **Zero Coverage Categories**: `complex/` and `errors/` categories have expected zero coverage (advanced features).

4. **No Production RAG**: Current RAG uses local pgvector. Azure AI Search planned for scale.
