# Project Status

**Last Updated:** 2026-01-20

## Overview

EvalPlatform is a distributed evaluation framework for testing LLM tool-calling, with an embedded NL2API system for translating natural language queries into LSEG financial API calls.

---

## Current Phase: Phase 2 (EstimatesAgent) - IN PROGRESS

### What's Implemented

| Component | Status | Notes |
|-----------|--------|-------|
| **Phase 1: Infrastructure** | ✅ Complete | |
| └─ LLM Abstraction | ✅ | Claude + OpenAI providers with tool-calling |
| └─ RAG Retriever | ✅ | Hybrid vector + keyword search (pgvector) |
| └─ NL2APIOrchestrator | ✅ | Query classification, entity resolution, agent routing |
| └─ Configuration | ✅ | pydantic-settings with env var support |
| └─ Unit Tests | ✅ | 183 tests for NL2API modules |
| **Phase 2: EstimatesAgent** | 🟡 In Progress | |
| └─ EstimatesAgent Implementation | ✅ | Rule-based + LLM fallback |
| └─ Field Code Mappings | ✅ | EPS, revenue, EBITDA, recommendations, etc. |
| └─ Period Detection | ✅ | FY1, FQ1, FY2, etc. |
| └─ Entity Resolution | ✅ | Pattern-based extraction + static mappings |
| └─ Evaluation Adapter | ✅ | Integrates with WaterfallEvaluator |
| └─ Integration Tests | ✅ | 18 tests |
| └─ **Real LLM Evaluation** | 🟡 Ready | Script created, needs API key to run |
| **Phase 3: Multi-turn + Clarification** | ⏳ Not Started | |
| **Phase 4: Remaining Agents** | ⏳ Not Started | DatastreamAgent, FundamentalsAgent, etc. |
| **Phase 5: Scale & Production** | ⏳ Not Started | |

---

## Test Coverage

```
Total Unit Tests:     254 passing
├── NL2API Tests:     183 passing
│   ├── LLM Protocols:     24
│   ├── LLM Providers:     17
│   ├── RAG Retriever:     14
│   ├── Entity Resolver:   17
│   ├── Clarification:     27
│   ├── Orchestrator:      19
│   ├── EstimatesAgent:    51
│   └── Eval Adapter:      15
└── Evaluation Tests:  71 passing

Integration Tests:     18 passing
├── Full Eval:         3
├── Sample Queries:    14
└── Multi-Company:     1
```

---

## EstimatesAgent Accuracy (Mock LLM - Rule-Based Only)

Last evaluation run against 589 estimates-tagged test cases with **mock LLM** (only measures rule-based extraction):

| Metric | Value |
|--------|-------|
| Exact Match | 21.9% |
| Partial Match | 28.0% |
| No Match | 50.1% |
| Average Precision | 49.79% |
| Average Recall | 30.41% |
| Rule-Based Coverage | 71.1% |
| Rule-Based Recall | 42.12% |

**Important:** These metrics only reflect rule-based pattern matching. LLM fallback cases return mock responses and are not properly evaluated.

---

## Running Real LLM Evaluation

To measure actual accuracy with a real LLM:

```bash
# 1. Set API key
export NL2API_ANTHROPIC_API_KEY="sk-ant-..."

# 2. Run evaluation (default: 50 test cases)
python scripts/run_estimates_eval.py --limit 50

# 3. Full evaluation (takes longer, costs more)
python scripts/run_estimates_eval.py --limit 500

# 4. Use OpenAI instead
export NL2API_LLM_PROVIDER="openai"
export NL2API_OPENAI_API_KEY="sk-..."
python scripts/run_estimates_eval.py --limit 50
```

Results are saved to `estimates_eval_<provider>_<count>.json`.

---

## File Structure

```
evalPlatform/
├── CONTRACTS.py                 # Shared data models
├── STATUS.md                    # This file
├── .env.example                 # Environment variable template
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
│   │   │   └── estimates.py     # EstimatesAgent ✅
│   │   ├── rag/                 # RAG retrieval
│   │   │   ├── protocols.py     # RAGRetriever protocol
│   │   │   ├── retriever.py     # HybridRAGRetriever
│   │   │   └── indexer.py       # Document indexer
│   │   ├── resolution/          # Entity resolution
│   │   │   ├── protocols.py     # EntityResolver protocol
│   │   │   └── resolver.py      # ExternalEntityResolver
│   │   ├── clarification/       # Ambiguity handling
│   │   │   └── detector.py      # AmbiguityDetector
│   │   └── evaluation/          # Eval integration
│   │       └── adapter.py       # NL2APITargetAdapter
│   │
│   ├── common/storage/          # Shared storage layer
│   └── evaluation/              # Evaluation pipeline
│       ├── core/                # Evaluators
│       └── batch/               # Batch runner
│
├── scripts/
│   └── run_estimates_eval.py    # Real LLM evaluation script
│
└── tests/
    ├── unit/nl2api/             # 183 unit tests
    └── integration/             # 18 integration tests
```

---

## Next Steps

### Immediate (to complete Phase 2)
1. **Run real LLM evaluation** - Get actual accuracy metrics with Claude/OpenAI
2. **Analyze failures** - Identify patterns in LLM failures
3. **Improve prompts** - Enhance system prompt based on failure analysis

### Phase 3 (Multi-turn + Clarification)
1. Implement conversation storage (PostgreSQL)
2. Add query expansion for follow-up queries
3. Integrate external entity resolution API
4. Implement clarification question flow

### Phase 4 (Remaining Agents)
1. DatastreamAgent - price, time series, calculated fields
2. FundamentalsAgent - WC codes, TR codes, financials
3. OfficersAgent - executives, compensation, governance
4. ScreeningAgent - SCREEN expressions, rankings

---

## Environment Setup

```bash
# Install dependencies
pip install -e .

# Start PostgreSQL (for RAG storage)
docker compose up -d

# Set API keys
cp .env.example .env
# Edit .env with your API keys

# Run tests
.venv/bin/python -m pytest tests/unit/ -v
.venv/bin/python -m pytest tests/integration/ -v
```

---

## Known Limitations

1. **Entity Resolution**: Uses static company→RIC mappings (~30 companies). External API integration planned for Phase 3.

2. **Rule-Based Coverage**: Only covers common field patterns (EPS, revenue, ratings). Complex queries fall back to LLM.

3. **No RAG in Evaluation**: Current evaluation doesn't use RAG retrieval. LLM relies on system prompt examples only.

4. **Single Domain**: Only EstimatesAgent implemented. Other LSEG domains (Datastream, Fundamentals, etc.) pending.
