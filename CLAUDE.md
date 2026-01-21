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

### Integration Testing Requirements

**ALWAYS write integration tests when code involves:**
- Database operations (repository methods against real PostgreSQL)
- Multi-component flows (orchestrator → agent → storage)
- API endpoint handlers
- Configuration loading and validation

**DO NOT write automated integration tests for external third-party APIs:**
- GLEIF, SEC EDGAR, OpenFIGI, and similar public data APIs
- These APIs change, have rate limits, and require network access
- Instead: **manually verify** the integration works, then document what you tested

```bash
# Run integration tests (requires docker compose up -d)
.venv/bin/python -m pytest tests/integration/ -v --tb=short -x
```

**When to write integration tests vs unit tests vs manual verification:**

| Scenario | Unit Test | Integration Test | Manual Verify |
|----------|-----------|------------------|---------------|
| Pure function logic | ✅ | | |
| Class with injected dependencies | ✅ | | |
| Repository CRUD operations | | ✅ | |
| Multi-component orchestration | | ✅ | |
| Database migrations | | ✅ | |
| End-to-end query processing | | ✅ | |
| External API integrations (GLEIF, SEC, etc.) | ✅ (mock) | | ✅ |

**Manual verification for external APIs:**
When implementing external API integrations, before marking work complete:
1. Run the script/code manually with real credentials
2. Verify data is fetched and parsed correctly
3. Document what you tested in the PR/commit (e.g., "Manually verified: fetched 10 GLEIF records, parsed LEI/company name correctly")

### Pre-commit checklist:
1. ✅ Unit tests pass: `pytest tests/unit/ -v --tb=short -x`
2. ✅ Integration tests pass: `pytest tests/integration/ -v --tb=short -x`
3. ✅ Linting passes: `ruff check .`
4. ✅ Security checklist reviewed (see Security Standards section)
5. ✅ Telemetry added for external calls/DB operations
6. ✅ No regressions in changed areas

---

## Project Overview

**NL2API** is a Natural Language to API translation system for LSEG financial data APIs. It translates natural language queries into structured API calls for Datastream, Estimates, Fundamentals, and other LSEG data services. Includes an evaluation framework for testing at scale (~400k test cases).

## Quick Commands

```bash
# Start PostgreSQL (required for storage)
docker compose up -d

# Run all unit tests
.venv/bin/python -m pytest tests/unit/ -v

# Run all integration tests (requires docker compose up -d)
.venv/bin/python -m pytest tests/integration/ -v

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
│   ├── unit/                         # Unit tests (mocked dependencies)
│   │   ├── nl2api/                   # NL2API unit tests
│   │   └── common/                   # Resilience + cache tests
│   ├── integration/                  # Integration tests (real DB, multi-component)
│   │   ├── storage/                  # Repository integration tests
│   │   └── nl2api/                   # Orchestrator + agent integration tests
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

## Evaluation Standards

**The evaluation pipeline is critical infrastructure.** Changes must be carefully validated.

### Evaluation Pipeline Architecture

```
TestCase → Evaluator Pipeline → Scorecard
              │
              ├─ Stage 1: SyntaxEvaluator (GATE - hard stop on failure)
              ├─ Stage 2: LogicEvaluator (AST comparison)
              ├─ Stage 3: ExecutionEvaluator (live API verification)
              └─ Stage 4: SemanticsEvaluator (LLM-as-Judge)
```

### When to Modify Evaluation Code

| Scenario | Action |
|----------|--------|
| New tool/API added | Add test fixtures + update LogicEvaluator if needed |
| New evaluation metric | Add to `CONTRACTS.py` first, then implement evaluator |
| Scoring logic change | Requires tier2 accuracy test before/after comparison |
| New evaluator stage | Must integrate with OTEL (see below) |

### Test Fixture Requirements

When adding or modifying test fixtures in `tests/fixtures/`:

1. **Follow the TestCase contract** (defined in `CONTRACTS.py`):
   ```python
   TestCase(
       id="unique-id",
       nl_query="What is Apple's PE ratio?",
       expected_tool_calls=(ToolCall(...),),
       expected_nl_response="Apple's PE ratio is...",
       tags=["fundamentals", "single-field"],
       category="lookups",
       subcategory="single_field",
   )
   ```

2. **Include required metadata**:
   - `category` and `subcategory` for coverage tracking
   - `tags` for filtering in accuracy tests
   - Unique `id` (format: `{category}-{subcategory}-{sequence}`)

3. **Validate fixtures load correctly**:
   ```bash
   pytest tests/unit/nl2api/test_fixture_coverage.py -v -k "test_fixtures_load"
   ```

### Evaluator Implementation Rules

1. **All evaluators must return `StageResult`** with:
   - `passed`: bool
   - `score`: float (0.0-1.0)
   - `error_code`: ErrorCode enum if failed
   - `reason`: Human-readable explanation
   - `duration_ms`: Execution time

2. **GATE stages stop the pipeline** - If Stage 1 (Syntax) fails, subsequent stages don't run

3. **All evaluators must integrate with OTEL** (see Telemetry Requirements):
   ```python
   from src.common.telemetry import get_tracer

   tracer = get_tracer(__name__)

   class MyEvaluator:
       def evaluate(self, test_case: TestCase, response: SystemResponse) -> StageResult:
           with tracer.start_as_current_span("evaluator.my_stage") as span:
               span.set_attribute("test_case.id", test_case.id)
               span.set_attribute("test_case.category", test_case.category)
               # ... evaluation logic
               span.set_attribute("result.passed", result.passed)
               span.set_attribute("result.score", result.score)
               return result
   ```

4. **Use `BatchMetrics` for aggregate metrics**:
   ```python
   from src.evaluation.batch.metrics import get_metrics

   metrics = get_metrics()
   metrics.record_test_result(scorecard, batch_id, tags)
   ```

### Scorecard Immutability

Scorecards are **immutable** once created. Never modify a scorecard after evaluation:

```python
# CORRECT - Create new scorecard with updated fields
new_scorecard = scorecard.model_copy(update={"notes": "reprocessed"})

# WRONG - Scorecards are frozen
scorecard.notes = "reprocessed"  # Raises error
```

### Batch Evaluation Changes

When modifying `src/evaluation/batch/`:

1. **Checkpoint/resume must be preserved** - Don't break resumable batch runs
2. **Metrics must be recorded** - Use `BatchMetrics` for all test results
3. **Test with small batches first**: `batch run --limit 10`

---

## Evaluation Data Generation Standards

**Follow these guidelines when working with test fixtures and evaluation data.**

### Key Decisions (Documented)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Git strategy for fixtures | **Commit to git** | Reproducibility, review visibility, no CI API costs |
| NL response generation model | **Claude 3.5 Haiku** | Better quality than 3 Haiku; negligible cost difference |
| `expected_response` field | **Leave null** | Until execution stage is implemented (deferred) |
| Fixture schema | **Align with CONTRACTS.py** | Generator dataclass must match `TestCase` contract |

See `docs/plans/evaluation-data-contract-plan.md` for full rationale.

### Test Case Field Definitions

```python
# In CONTRACTS.py
class TestCase:
    expected_response: dict | None      # Raw API data (e.g., {"AAPL.O": {"P": 246.02}})
                                        # Currently NULL - populate when execution stage added
    expected_nl_response: str | None    # Human-readable sentence (e.g., "Apple's price is $246.02")
                                        # Generated by Claude 3.5 Haiku
```

**Do NOT confuse these fields:**
- `expected_response` = structured data from API
- `expected_nl_response` = natural language summary for user

### Regenerating Fixtures

```bash
# Regenerate all fixtures (12,887 test cases)
python scripts/generate_test_cases.py --all

# Regenerate specific category
python scripts/generate_test_cases.py --category lookups

# Validate generated output against CONTRACTS.py
python scripts/generate_test_cases.py --validate

# Generate NL responses (uses Claude 3.5 Haiku, ~$5 cost)
python scripts/generate_nl_responses.py --all
```

### When Adding New Evaluation Data

1. **Add to source data** in `data/field_codes/` or `data/tickers/`
2. **Regenerate fixtures** using the generator scripts
3. **Commit generated fixtures** to git (don't generate in CI)
4. **Run validation**: `python scripts/generate_test_cases.py --validate`
5. **Run tests**: `pytest tests/unit/nl2api/test_fixture_coverage.py -v`

### Synthetic Data Caveats

All evaluation data is synthetic. When documenting or using:

- `expected_nl_response` values are **LLM-generated** (Claude 3.5 Haiku)
- `expected_response` values (when populated) are **based on API specs**, not live calls
- API specifications were **reverse-engineered** from public documentation
- Ticker/company data is **point-in-time** and may become stale

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

---

## Security Standards

**ALWAYS review code changes for security vulnerabilities.** This project handles financial data queries.

### Security Checklist (verify before committing):

1. **Input Validation**
   - [ ] All user input is validated before use
   - [ ] Query parameters are sanitized (no raw string interpolation)
   - [ ] File paths are validated and constrained to expected directories

2. **SQL Injection Prevention**
   - [ ] Use parameterized queries ONLY (never f-strings or `.format()` with SQL)
   - [ ] Use `asyncpg` parameter binding: `$1, $2` syntax
   ```python
   # CORRECT
   await conn.fetch("SELECT * FROM users WHERE id = $1", user_id)

   # WRONG - SQL injection vulnerable
   await conn.fetch(f"SELECT * FROM users WHERE id = {user_id}")
   ```

3. **Secrets Management**
   - [ ] No secrets in code (use environment variables)
   - [ ] No secrets in logs (redact API keys, tokens)
   - [ ] `.env` files are in `.gitignore`

4. **API Security**
   - [ ] External API keys loaded from environment only
   - [ ] Rate limiting considered for external calls
   - [ ] Error messages don't leak internal details

5. **Dependency Security**
   - [ ] New dependencies reviewed for known vulnerabilities
   - [ ] Pin dependency versions in requirements

### Common Patterns in This Codebase

```python
# Safe database query
async def get_by_id(self, id: str) -> Optional[Model]:
    row = await self.conn.fetchrow(
        "SELECT * FROM table WHERE id = $1", id
    )
    return Model(**row) if row else None

# Safe configuration loading
class Config(BaseSettings):
    api_key: SecretStr  # Never logs the actual value

    model_config = SettingsConfigDict(env_prefix="NL2API_")
```

---

## Error Handling Patterns

**Follow consistent error handling patterns across the codebase.**

### Exception Hierarchy

```python
# Base exception for all NL2API errors
class NL2APIError(Exception):
    """Base exception for NL2API system."""
    pass

# Domain-specific exceptions
class EntityResolutionError(NL2APIError):
    """Failed to resolve entity (company, RIC, etc.)."""
    pass

class AgentProcessingError(NL2APIError):
    """Agent failed to process query."""
    pass

class StorageError(NL2APIError):
    """Database or storage operation failed."""
    pass
```

### Error Handling Rules

1. **Catch specific exceptions, not bare `except:`**
   ```python
   # CORRECT
   try:
       result = await external_api.call()
   except httpx.TimeoutException:
       logger.warning("API timeout, using fallback")
       return fallback_result
   except httpx.HTTPStatusError as e:
       logger.error(f"API error: {e.response.status_code}")
       raise AgentProcessingError("External API failed") from e

   # WRONG
   try:
       result = await external_api.call()
   except:
       pass  # Swallows all errors silently
   ```

2. **Preserve exception chains with `from`**
   ```python
   except SomeError as e:
       raise OurError("Context about what failed") from e
   ```

3. **Log at appropriate levels**
   - `DEBUG`: Detailed diagnostic info
   - `INFO`: Normal operations (query processed, batch started)
   - `WARNING`: Recoverable issues (fallback used, retry triggered)
   - `ERROR`: Failures requiring attention (API down, invalid data)

4. **Don't log and raise** - do one or the other at each level
   ```python
   # Let caller decide whether to log
   raise StorageError("Connection failed") from e

   # OR handle locally and don't propagate
   logger.error("Connection failed, returning empty result")
   return []
   ```

5. **Fail fast at boundaries, recover internally**
   - External inputs: Validate early, fail with clear errors
   - Internal operations: Use fallbacks and retries where sensible

---

## Telemetry Requirements

**Add observability to new code that involves external calls or significant operations.**

### When to Add Tracing Spans

Add OTEL spans for:
- External API calls (LLM providers, LSEG APIs)
- Database operations (queries, bulk inserts)
- Multi-step orchestration flows
- Cache operations (hits/misses)
- Retry/circuit breaker events
- **Evaluation code** (evaluators, batch runners, scorers)

### Evaluation Telemetry (REQUIRED)

**All evaluation code MUST integrate with OTEL.** The evaluation pipeline is critical for measuring system quality, and observability is essential for debugging and monitoring.

| Component | Required Telemetry |
|-----------|-------------------|
| Evaluators | Spans with `test_case.id`, `result.passed`, `result.score` |
| Batch runner | Spans for batch lifecycle + `BatchMetrics` for aggregates |
| Scorers | Spans with scoring details and `duration_ms` |

Use `src.evaluation.batch.metrics.BatchMetrics` for aggregate metrics:
```python
from src.evaluation.batch.metrics import get_metrics

metrics = get_metrics()
metrics.record_test_result(scorecard, batch_id, tags)
metrics.record_batch_complete(batch_job, duration_seconds)
```

### Span Implementation Pattern

```python
from src.common.telemetry import get_tracer

tracer = get_tracer(__name__)

async def process_query(self, query: str) -> Result:
    with tracer.start_as_current_span("process_query") as span:
        span.set_attribute("query.length", len(query))

        # Child span for LLM call
        with tracer.start_as_current_span("llm_call") as llm_span:
            llm_span.set_attribute("llm.provider", self.provider)
            response = await self.llm.complete(query)
            llm_span.set_attribute("llm.tokens", response.usage.total_tokens)

        span.set_attribute("result.status", "success")
        return result
```

### Required Span Attributes

| Operation Type | Required Attributes |
|---------------|---------------------|
| LLM calls | `llm.provider`, `llm.model`, `llm.tokens` |
| Database | `db.operation`, `db.table`, `db.rows_affected` |
| External API | `http.method`, `http.url`, `http.status_code` |
| Cache | `cache.hit`, `cache.key_prefix` |
| Agent | `agent.name`, `agent.confidence`, `query.category` |
| Evaluator | `test_case.id`, `test_case.category`, `result.passed`, `result.score` |
| Batch | `batch.id`, `batch.total_tests`, `batch.passed_count`, `batch.duration_ms` |

### Metrics to Track

Record metrics for:
- Request latency (histograms)
- Error rates (counters by error type)
- Cache hit rates
- LLM token usage
- Query throughput

```python
from src.common.telemetry import get_meter

meter = get_meter(__name__)
request_counter = meter.create_counter("nl2api.requests")
latency_histogram = meter.create_histogram("nl2api.latency_ms")

async def handle_request(self, query: str):
    start = time.time()
    request_counter.add(1, {"agent": self.name})
    try:
        result = await self.process(query)
        latency_histogram.record((time.time() - start) * 1000)
        return result
    except Exception:
        request_counter.add(1, {"agent": self.name, "error": "true"})
        raise
```

---

## Known Gaps

The following areas need documented standards but are not yet fully specified:

| Gap | Description | Workaround |
|-----|-------------|------------|
| **Migration guidance** | No documented process for DB schema changes | Always create migrations in `migrations/` for schema changes. Follow existing migration file naming: `NNN_description.sql` |
| **Backwards compatibility** | No policy for API versioning or deprecation | Avoid breaking changes to public interfaces. If unavoidable, discuss with team first |
| **Performance testing** | No load/stress testing framework | Manual testing with `ab` or `wrk` for critical paths |
| **Documentation requirements** | No standard for when to update docs | Update docs when adding new features or changing behavior |

These gaps should be addressed as the project matures. If you encounter a situation requiring guidance in these areas, ask the user for clarification.

---

## Development Notes

### Running Tests
```bash
# All unit tests
.venv/bin/python -m pytest tests/unit/ -v

# All integration tests (requires docker compose up -d)
.venv/bin/python -m pytest tests/integration/ -v

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
| `src/evaluation/core/evaluators.py` | Evaluation pipeline stages (Syntax, Logic, etc.) |
| `src/evaluation/batch/runner.py` | Batch evaluation runner |
| `src/evaluation/batch/metrics.py` | OTEL metrics for evaluation |
| `tests/unit/nl2api/test_fixture_coverage.py` | Dynamic test infrastructure |
| `tests/unit/nl2api/fixture_loader.py` | Fixture loading utility |
| `tests/accuracy/core/evaluator.py` | Accuracy testing evaluator |
| `docs/accuracy-testing.md` | Accuracy testing pattern documentation |
