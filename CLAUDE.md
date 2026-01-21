# CLAUDE.md - Project Context for Claude Code

## CRITICAL: Testing Requirements

**ALWAYS run tests after making code changes.** This is non-negotiable.

### After ANY code modification:
```bash
# 1. Run unit tests (REQUIRED before committing)
.venv/bin/python -m pytest tests/unit/ -v --tb=short -x

# 2. If tests fail: FIX THE ISSUES before proceeding
# 3. Only commit after all tests pass
```

### When to run accuracy tests:
- **tier1**: Quick sanity check (~50 samples, ~2 min) - Run when modifying routing/agents
- **tier2**: Standard evaluation (~200 samples, ~10 min) - Run before merging significant changes
- **tier3**: Comprehensive (~2000+ samples, ~60 min) - Run weekly or for major releases

```bash
# Only run accuracy tests when specifically needed (they cost API credits)
.venv/bin/python -m pytest tests/accuracy/ -m tier1 -v  # Quick check
```

### Pre-commit checklist:
1. ✅ Unit tests pass: `pytest tests/unit/ -v --tb=short -x`
2. ✅ Linting passes: `ruff check .`
3. ✅ No regressions in changed areas

---

## Project Overview

**NL2API** is a Natural Language to API translation system for LSEG financial data APIs. It translates natural language queries into structured API calls for Datastream, Estimates, Fundamentals, and other LSEG data services. Includes an evaluation framework for testing at scale (~400k test cases).

## Quick Commands

```bash
# Start PostgreSQL (required for storage)
docker compose up -d

# Run all tests
.venv/bin/python -m pytest tests/unit/ -v

# Run NL2API tests only
.venv/bin/python -m pytest tests/unit/nl2api/ -v

# Run fixture coverage tests
.venv/bin/python -m pytest tests/unit/nl2api/test_fixture_coverage.py -v

# Run accuracy tests (requires ANTHROPIC_API_KEY)
.venv/bin/python -m pytest tests/accuracy/ -m tier1 -v   # Quick (~50 samples)
.venv/bin/python -m pytest tests/accuracy/ -m tier2 -v   # Standard (~200 samples)
.venv/bin/python -m pytest tests/accuracy/ -m tier3 -v   # Comprehensive (all)

# Lint
ruff check .

# Run single test case evaluation
.venv/bin/python -m src.evaluation.cli.main run tests/fixtures/search_products.json

# Run batch evaluation
.venv/bin/python -m src.evaluation.cli.main batch run --limit 10

# View batch results
.venv/bin/python -m src.evaluation.cli.main batch list
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
nl2api/
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
│   ├── common/
│   │   ├── storage/          # Storage layer (postgres, memory)
│   │   ├── telemetry/        # OTEL metrics + tracing
│   │   ├── cache/            # Redis caching
│   │   └── resilience/       # Circuit breaker, retry
│   └── evaluation/           # Evaluation pipeline
│       ├── core/             # Evaluators
│       └── batch/            # Batch runner
├── tests/
│   ├── unit/                         # Unit tests (mocked LLM)
│   │   ├── nl2api/                   # NL2API unit tests
│   │   └── common/                   # Resilience + cache tests
│   ├── accuracy/                     # Accuracy tests (real LLM)
│   │   ├── core/                     # Evaluator, config, thresholds
│   │   ├── agents/                   # Per-agent accuracy tests
│   │   └── domains/                  # Per-domain accuracy tests
│   └── fixtures/lseg/generated/      # 12,887 test fixtures
└── docker-compose.yml                # PostgreSQL + pgvector + Redis + OTEL
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

## Accuracy Testing

Accuracy tests use **real LLM calls** to measure system output quality (vs unit tests which mock LLMs).

### Key Distinction

| Aspect | Unit Tests | Accuracy Tests |
|--------|-----------|----------------|
| LLM Calls | Mocked | Real |
| Purpose | Test code behavior | Measure output quality |
| Assertions | Exact matches | Threshold-based (≥80%) |
| Speed | Fast (ms) | Slower (seconds/query) |

### Test Tiers

| Tier | Samples | Use Case |
|------|---------|----------|
| tier1 | ~50 | PR checks, quick feedback |
| tier2 | ~200 | Daily CI |
| tier3 | All | Weekly comprehensive |

### Thresholds

- **Global**: 80% minimum accuracy
- **Per-category**: lookups (85%), temporal (80%), comparisons (75%), screening (75%), complex (70%), errors (90%)

### Running Accuracy Tests

```bash
# Requires ANTHROPIC_API_KEY
pytest tests/accuracy/ -m tier1    # Quick
pytest tests/accuracy/ -m tier2    # Standard
pytest tests/accuracy/ -m tier3    # Comprehensive
```

See [docs/accuracy-testing.md](docs/accuracy-testing.md) for full documentation.

## CI/CD

GitHub Actions workflows in `.github/workflows/`:

| Workflow | Trigger | Purpose |
|----------|---------|---------|
| `ci.yml` | Push/PR to main | Lint + unit tests |
| `accuracy.yml` | PR (tier1), daily (tier2), weekly (tier3) | Accuracy tests with real LLM |

Manual trigger: Go to Actions → select workflow → "Run workflow" → choose tier.

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

**Phase 5+ Complete** - Scale, Production, and Observability features implemented.

| Component | Description |
|-----------|-------------|
| Circuit Breaker | `src/common/resilience/circuit_breaker.py` - Fail-fast for external services |
| Retry with Backoff | `src/common/resilience/retry.py` - Exponential backoff |
| Redis Cache | `src/common/cache/redis_cache.py` - L1/L2 caching |
| Bulk Indexing | COPY protocol for fast RAG inserts |
| Checkpoint/Resume | Resumable indexing for large jobs |
| OTEL Telemetry | `src/common/telemetry/` - Metrics, tracing via OTEL Collector |
| Accuracy Testing | `tests/accuracy/` - Real LLM accuracy evaluation framework |

**Total Tests: 606+ (unit) + accuracy tests**

## Important Files to Review

| File | Purpose |
|------|---------|
| `CONTRACTS.py` | All data models - read this first |
| `src/nl2api/orchestrator.py` | NL2API main entry point |
| `src/nl2api/agents/*.py` | Domain agent implementations |
| `tests/unit/nl2api/test_fixture_coverage.py` | Dynamic test infrastructure |
| `tests/unit/nl2api/fixture_loader.py` | Fixture loading utility |
| `tests/accuracy/core/evaluator.py` | Accuracy testing evaluator |
| `docs/accuracy-testing.md` | Accuracy testing pattern documentation |
