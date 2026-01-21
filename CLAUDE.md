# CLAUDE.md - Project Context for Claude Code

## Project Overview

**EvalPlatform** is a distributed evaluation framework for testing LLM tool-calling at scale (~400k test cases), with an embedded **NL2API system** for translating natural language queries into LSEG financial API calls.

## Quick Commands

```bash
# Start PostgreSQL (required for storage)
docker compose up -d

# Run all tests
.venv/bin/python -m pytest tests/unit/ -v

# Run NL2API tests only (497 tests)
.venv/bin/python -m pytest tests/unit/nl2api/ -v

# Run fixture coverage tests
.venv/bin/python -m pytest tests/unit/nl2api/test_fixture_coverage.py -v

# Run single test case evaluation
.venv/bin/python -m src.cli.main run tests/fixtures/search_products.json

# Run batch evaluation
.venv/bin/python -m src.cli.main batch run --limit 10

# View batch results
.venv/bin/python -m src.cli.main batch list
```

## Architecture

### NL2API System

```
┌─────────────────────────────────────────────────────────────────┐
│                         NL2API System                            │
├─────────────────────────────────────────────────────────────────┤
│  NL2APIOrchestrator                                              │
│  ├─ Query classification (route to domain agent)                │
│  ├─ Entity resolution (Company → RIC via resolver)              │
│  └─ Ambiguity detection → Clarification flow                    │
├─────────────────────────────────────────────────────────────────┤
│  Domain Agents (5 implemented)                                   │
│  ├─ DatastreamAgent     (price, time series, calculated fields) │
│  ├─ EstimatesAgent      (I/B/E/S forecasts, recommendations)    │
│  ├─ FundamentalsAgent   (WC codes, TR codes, financials)        │
│  ├─ OfficersAgent       (executives, compensation, governance)  │
│  └─ ScreeningAgent      (SCREEN expressions, rankings)          │
├─────────────────────────────────────────────────────────────────┤
│  Support Components                                              │
│  ├─ LLM Abstraction (Claude + OpenAI providers)                 │
│  ├─ RAG Retriever (hybrid vector + keyword, pgvector)           │
│  ├─ Conversation Manager (multi-turn, query expansion)          │
│  └─ Entity Resolver (pattern-based + static mappings)           │
└─────────────────────────────────────────────────────────────────┘
```

### Evaluation Pipeline (Waterfall)

```
Raw Output → [Stage 1: Syntax] → Parsed ToolCalls → [Stage 2: Logic] → Scorecard
                  │                                       │
                  │ FAIL: Hard stop                       │ FAIL: Soft continue
                  ▼                                       ▼
              Scorecard                              Scorecard
```

- **Stage 1 (Syntax)**: Validates JSON structure and schema - GATE (hard stop on failure)
- **Stage 2 (Logic)**: AST-based tool call comparison - HIGH priority (soft continue)
- **Stage 3 (Execution)**: Live API verification - CRITICAL (configurable)
- **Stage 4 (Semantics)**: LLM-as-Judge NL comparison - LOW priority (configurable)

### Key Data Models (CONTRACTS.py)

```python
TestCase     # The "Gold Standard" - nl_query, expected_tool_calls, expected_nl_response
ToolCall     # Tool name + frozen arguments (hashable for set comparison)
Scorecard    # Evaluation result with per-stage results and overall pass/fail
BatchJob     # Batch tracking with status and progress
```

### Storage Layer

Protocol-based abstraction with implementations for:
- **PostgreSQL** (`src/common/storage/postgres/`) - Local development
- **In-Memory** (`src/common/storage/memory/`) - Unit tests
- **Azure** (not yet implemented) - Production

Factory pattern: `create_repositories(config) -> (test_case_repo, scorecard_repo, batch_repo)`

## Directory Structure

```
evalPlatform/
├── CONTRACTS.py              # Pydantic v2 data models
├── src/
│   ├── nl2api/               # NL2API System
│   │   ├── orchestrator.py   # Main entry point
│   │   ├── config.py         # Configuration (pydantic-settings)
│   │   ├── llm/              # LLM providers (Claude, OpenAI)
│   │   ├── agents/           # Domain agents (5 implemented)
│   │   ├── rag/              # RAG retrieval (pgvector)
│   │   ├── resolution/       # Entity resolution
│   │   ├── clarification/    # Ambiguity detection
│   │   ├── conversation/     # Multi-turn support
│   │   └── evaluation/       # Eval adapter
│   ├── common/storage/       # Shared storage layer
│   └── evaluation/           # Evaluation pipeline
│       ├── core/             # Evaluators
│       └── batch/            # Batch runner
├── tests/
│   ├── unit/nl2api/          # 497 NL2API tests
│   │   ├── fixture_loader.py # Fixture loading utility
│   │   ├── test_fixture_coverage.py  # Dynamic coverage tests
│   │   └── test_*_fixtures.py        # Agent fixture tests
│   └── fixtures/lseg/generated/      # 12,887 test fixtures
└── docker-compose.yml        # PostgreSQL + pgvector
```

## Dynamic Fixture-Based Testing

Tests automatically scale with test data using programmatic fixture expansion:

### How It Works

1. **FixtureLoader** discovers all categories from `tests/fixtures/lseg/generated/`
2. **CoverageRegistry** defines minimum coverage thresholds per category/agent
3. **Parameterized tests** auto-generate from fixture structure
4. Tests **fail if coverage drops** below thresholds (prevents regressions)
5. **Growth detection** alerts when fixture counts change

### Key Files

| File | Purpose |
|------|---------|
| `tests/unit/nl2api/fixture_loader.py` | `FixtureLoader` class, `GeneratedTestCase` dataclass |
| `tests/unit/nl2api/test_fixture_coverage.py` | `CoverageRegistry`, dynamic parameterized tests |
| `tests/unit/nl2api/test_datastream_fixtures.py` | DatastreamAgent against 6,000+ fixtures |
| `tests/unit/nl2api/test_screening_fixtures.py` | ScreeningAgent against 265 fixtures |

### Coverage Registry Example

```python
class CoverageRegistry:
    REQUIRED_COVERAGE = [
        # (category, subcategory, min_rate, agent_class)
        ("lookups", "single_field", 0.3, DatastreamAgent),
        ("lookups", "multi_field", 0.15, DatastreamAgent),
        ("screening", "top_n", 0.5, ScreeningAgent),
    ]
```

### Adding New Test Data

1. Add JSON to `tests/fixtures/lseg/generated/<category>/`
2. Tests auto-discover and include new data
3. Add coverage requirements to `CoverageRegistry` if needed

## Key Patterns

### 1. Protocol-Based Design
```python
@runtime_checkable
class DomainAgent(Protocol):
    async def can_handle(self, query: str) -> float: ...
    async def process(self, context: AgentContext) -> AgentResult: ...
```

### 2. Frozen Pydantic Models
All data models use `model_config = ConfigDict(frozen=True)` for immutability.

### 3. Order-Independent Tool Call Comparison
`ToolCall.arguments` uses `FrozenDict` for hashability, enabling set-based comparison.

### 4. Async-First Design
All repository operations, agents, and evaluators are async.

### 5. Rule-Based + LLM Fallback
Agents use pattern matching first, fall back to LLM for complex queries.

## Development Notes

### Running Tests
```bash
# All unit tests
.venv/bin/python -m pytest tests/unit/ -v

# NL2API tests only
.venv/bin/python -m pytest tests/unit/nl2api/ -v

# Specific agent tests
.venv/bin/python -m pytest tests/unit/nl2api/test_datastream.py -v
```

### Database
- PostgreSQL 16 with pgvector extension
- Tables: `test_cases`, `scorecards`, `batch_jobs`

### Environment Variables
- `EVAL_BACKEND`: "postgres" | "memory" | "azure"
- `NL2API_LLM_PROVIDER`: "anthropic" | "openai"
- `NL2API_ANTHROPIC_API_KEY`: Claude API key
- `NL2API_OPENAI_API_KEY`: OpenAI API key

## Current Status

**Phase 4 Complete** - All domain agents implemented with comprehensive fixture-based tests.

| Component | Tests |
|-----------|-------|
| DatastreamAgent | 36 + 26 fixture-based |
| ScreeningAgent | 47 + 22 fixture-based |
| EstimatesAgent | 51 |
| FundamentalsAgent | 49 |
| OfficersAgent | 41 |
| Conversation | 45 |
| **Total NL2API** | **497 passing** |

## Important Files to Review

| File | Purpose |
|------|---------|
| `CONTRACTS.py` | All data models - read this first |
| `src/nl2api/orchestrator.py` | NL2API main entry point |
| `src/nl2api/agents/*.py` | Domain agent implementations |
| `tests/unit/nl2api/test_fixture_coverage.py` | Dynamic test infrastructure |
| `tests/unit/nl2api/fixture_loader.py` | Fixture loading utility |
