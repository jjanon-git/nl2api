# CLAUDE.md - Project Context for Claude Code

> **Note**: This project inherits shared development standards from `../CLAUDE.md` (self-improvement loop, testing requirements, security standards, error handling patterns).

## Project-Specific Documentation Locations

When applying the self-improvement loop (see parent), use these project-specific locations:

| Issue Type | Where to Document |
|------------|-------------------|
| NL2API process/standards | This file |
| Config/integration issue | `docs/troubleshooting.md` |
| Evaluation pipeline gotcha | Comment in `src/evaluation/` |
| Agent behavior issue | Comment in `src/nl2api/agents/` |

**Examples from this project:**

| Situation | Root Cause | Action |
|-----------|------------|--------|
| Grafana shows no data | Datasource UID mismatch | Add to `docs/troubleshooting.md` |
| Metrics not appearing | Missing `_total` suffix | Add to `docs/troubleshooting.md` |
| Batch eval fails silently | No fixtures in DB | Add prerequisite to this file |
| FastAPI returns 422 for valid JSON | `from __future__ import annotations` breaks introspection | Add comment in affected file |

---

## CRITICAL: Every Capability Needs Evaluation

**No capability is complete without evaluation data and metrics.** This is non-negotiable.

When building or modifying a capability:
1. **Define what "correct" means** - What is the expected output for a given input?
2. **Create evaluation fixtures** - Test cases with inputs and expected outputs
3. **Establish baseline metrics** - Run evaluation and record accuracy
4. **Track over time** - Ensure changes don't regress accuracy

A capability without evaluation is untested code. See **BACKLOG.md → Capabilities Evaluation Matrix** for current status.

---

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

**When to write integration tests vs unit tests vs manual verification** (extends parent):

| Scenario | Unit Test | Integration Test | Manual Verify |
|----------|-----------|------------------|---------------|
| Pure function logic | ✅ | | |
| Class with injected dependencies | ✅ | | |
| **Config defaults and behavior** | ✅ | | |
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

### Testing Configuration Changes

**When modifying config defaults or adding config-driven behavior, ALWAYS add tests for:**

| Change Type | Required Tests |
|-------------|----------------|
| New config field | Test default value is correct |
| Config affects runtime behavior | Test behavior uses config value |
| Config switches between implementations | Test both code paths |
| Config with environment override | Test env var is respected |

**Example: Routing model configuration**
```python
# 1. Test config default
def test_config_defaults_to_haiku_for_routing():
    cfg = NL2APIConfig()
    assert cfg.routing_model == "claude-3-5-haiku-20241022"

# 2. Test behavior uses config
def test_orchestrator_creates_separate_routing_llm():
    # Verify orchestrator creates different LLM when routing_model differs
    ...

# 3. Test matching case (no separate LLM needed)
def test_orchestrator_reuses_main_llm_when_models_match():
    # Verify no extra LLM created when models are same
    ...
```

**Why this matters:** Config changes silently affect runtime behavior. Without tests, regressions go unnoticed until production. The routing model switch (Sonnet → Haiku) required 3 tests to properly verify.

### Test Coverage Requirements

**After completing any new module or significant code addition, assess test coverage.**

```bash
# Check coverage for changed files
.venv/bin/python -m pytest tests/unit/ --cov=src/path/to/module --cov-report=term-missing -v

# Example: Check MCP server coverage
.venv/bin/python -m pytest tests/unit/mcp_servers/ --cov=src/mcp_servers --cov-report=term-missing -v
```

**Coverage targets:**
| Component Type | Minimum Coverage |
|----------------|------------------|
| Core business logic (agents, orchestrator) | 80% |
| Utilities and helpers | 70% |
| New modules (MCP servers, CLI) | 60% |
| Transport/IO code | 40% (harder to test without integration tests) |

**What to test for coverage gaps:**
- All public methods should have at least one test
- Error handling paths (exceptions, edge cases)
- Configuration options and their effects
- Protocol handlers and message routing

**When to skip coverage:**
- `__main__.py` entry points (tested via integration/manual)
- I/O-heavy code that requires real connections (mark for integration tests instead)

### Manual Testing Requirements

**Before committing new servers, CLIs, or API endpoints, manually verify they work.**

This applies to:
- HTTP/API servers (MCP servers, REST APIs)
- CLI tools
- Any code that unit tests can't fully cover

**Manual testing checklist:**
1. Start the server/CLI with default configuration
2. Verify health/status endpoints respond
3. Test at least one happy-path request per endpoint
4. Test at least one error case (invalid input, missing resource)
5. Document what you tested in the commit message

**Example: MCP Server manual testing**
```bash
# Start server
.venv/bin/python -m src.mcp_servers.entity_resolution --port 8085 --no-redis &

# Test endpoints
curl http://localhost:8085/health
curl -X POST http://localhost:8085/api/resolve -H "Content-Type: application/json" -d '{"entity": "Apple"}'
curl -X POST http://localhost:8085/mcp -H "Content-Type: application/json" -d '{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}'

# Kill server
pkill -f "entity_resolution.*8085"
```

**Why this matters:** Unit tests with mocks can pass while real code fails (wrong imports, missing dependencies, I/O issues). Manual testing catches these gaps.

### Pre-commit checklist:
1. ✅ Unit tests pass: `pytest tests/unit/ -v --tb=short -x`
2. ✅ Integration tests pass: `pytest tests/integration/ -v --tb=short -x`
3. ✅ Linting passes: `ruff check .`
4. ✅ **Test coverage assessed for new code** (see above)
5. ✅ **Manual testing completed for servers/CLIs** (see above)
6. ✅ Security checklist reviewed (see Security Standards section)
7. ✅ Telemetry added for external calls/DB operations
8. ✅ No regressions in changed areas

---

## CRITICAL: Backlog Tracking

**All planned work must be tracked in [BACKLOG.md](BACKLOG.md).**

### Before starting significant work:
1. Check if the work is already in the backlog
2. If not, add it with priority, status, and description
3. Link to detailed plan docs in `docs/plans/` for complex items

### While working:
1. Update status in BACKLOG.md as work progresses
2. Move items between sections (In Progress, High Priority, etc.)
3. Add sub-tasks for partially complete work

### When completing work:
1. Move item to "Completed" section with date
2. Update any related plan docs (roadmap.md, status.md)
3. Mark checkboxes for sub-tasks

### What to track:
- New features or capabilities
- Bug fixes that span multiple files
- Refactoring efforts
- Technical debt items
- Research spikes

### What NOT to track:
- Simple typo fixes
- Single-file documentation updates
- Minor code cleanup

**The backlog is the source of truth for planned work.** Keep it updated so anyone can see what's in progress and what's next.

---

## CRITICAL: LLM Prompt Review Requirement

**Before running any script or code that calls an LLM API, ALWAYS review the prompt with the user first.**

This applies to:
- New scripts that generate content using LLMs (e.g., `generate_nl_responses.py`)
- Modifications to existing LLM prompts in agents or evaluators
- Any batch processing that will make multiple LLM API calls

**Required review process:**
1. Show the **full system prompt** and **example user prompt** to the user
2. Explain what the LLM will be asked to generate
3. Get explicit confirmation before running
4. Document the approved prompt in the script or a referenced file

**Why this matters:**
- LLM calls cost money (especially at scale)
- Prompt quality directly impacts output quality
- Bad prompts can generate thousands of low-quality test cases
- Prompts should be reviewed like code before "deploying"

**Required review format:**
```
SYSTEM PROMPT:
[full system prompt text]

USER PROMPT (template):
[example user prompt with placeholders]

EXAMPLE OUTPUT:
[what the LLM is expected to return]

Scope: N test cases
Estimated cost: $X.XX (show calculation: input tokens × rate + output tokens × rate)
Model: [model name]
```

**Cost estimation is mandatory** - always calculate and show the estimated cost before getting approval.

---

## Project Overview

**NL2API** is a Natural Language to API translation system for LSEG financial data APIs. It translates natural language queries into structured API calls for Datastream, Estimates, Fundamentals, and other LSEG data services. Includes an evaluation framework for testing at scale (~19k test cases).

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

# Load fixtures to database (REQUIRED before batch evaluation)
.venv/bin/python scripts/load_fixtures_to_db.py --all

# Run batch evaluation
.venv/bin/python -m src.evaluation.cli.main batch run --limit 10

# View batch results
.venv/bin/python -m src.evaluation.cli.main batch list

# View metrics in Grafana (after docker compose up -d)
# Open http://localhost:3000 (default: admin/admin)
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
│   └── fixtures/lseg/generated/      # ~19k test fixtures
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

### Adding New Test Case Generators

When adding a new fixture generator, follow this checklist:

1. [ ] Generator created in `scripts/generators/` (extend `BaseGenerator` or create standalone)
2. [ ] Output includes `_meta` block with `TestCaseSetConfig` fields
3. [ ] Generator registered in `scripts/generators/__init__.py`
4. [ ] Generator added to `scripts/generate_test_cases.py`
5. [ ] Category added to `FixtureLoader.CATEGORIES` in `tests/unit/nl2api/fixture_loader.py`
6. [ ] Coverage thresholds added to `CoverageRegistry.REQUIRED_COVERAGE` in `test_fixture_coverage.py`
7. [ ] Test file created in `tests/unit/nl2api/test_{category}_fixtures.py`
8. [ ] **Documentation updated in `docs/evaluation-data.md`**

This checklist prevents documentation and test infrastructure from being overlooked.

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

### Evaluation Modes

| Mode | Flag | When to Use |
|------|------|-------------|
| Batch API | `use_batch_api=True` (default) | CI runs, cost-sensitive |
| Real-time | `use_batch_api=False` | Debugging, immediate feedback |

```bash
# Quick debugging with real-time API:
pytest tests/accuracy/routing/ -k realtime
```

Config options in `tests/accuracy/core/config.py`.

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

### Batch Evaluation vs Unit Tests (CRITICAL DECISION)

**Use the right tool for the right purpose:**

| Purpose | Tool | Persisted? | When to Use |
|---------|------|------------|-------------|
| **Accuracy tracking** | `batch run --mode resolver` | ✅ Yes | Measure real system accuracy over time |
| **End-to-end accuracy** | `batch run --mode orchestrator` | ✅ Yes | Full pipeline accuracy (costs API credits) |
| **Pipeline testing** | `batch run --mode simulated` | ❌ No* | Test evaluation infrastructure only |
| **Code correctness** | Unit tests (`pytest`) | ❌ No | Verify code behavior with mocks |

*Simulated results ARE persisted but SHOULD NOT be used for accuracy tracking.

**The key insight:**
- **Batch evaluation** is for measuring **real system performance** against test cases
- **Unit tests** are for verifying **code correctness** with mocked dependencies
- **Simulated responses** produce 100% pass rates and are meaningless for tracking improvement

**Default behavior:**
```bash
# Default mode is 'resolver' - uses real EntityResolver for meaningful accuracy
.venv/bin/python -m src.evaluation.cli.main batch run --tag entity_resolution --limit 100

# Explicit modes
batch run --mode resolver       # Real accuracy (DEFAULT)
batch run --mode orchestrator   # Full pipeline (requires LLM API key)
batch run --mode simulated      # Pipeline testing only (NOT for tracking)
```

**What gets persisted to track over time:**
- Scorecards with pass/fail for each test case
- Batch job metadata (total, passed, failed, duration)
- Results visible in Grafana dashboards

**What should stay in unit tests:**
- Pipeline infrastructure validation (use mocked responses)
- Evaluator logic correctness
- Repository CRUD operations

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
| Batch eval default mode | **`resolver` (real)** | Persist meaningful accuracy, not simulated 100% |
| Simulated responses | **Unit tests only** | Pipeline infra testing doesn't need persistence |

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

### Test Case Set Configuration

Different capabilities have different field requirements. Each fixture file declares what's required via `_meta`:

```json
{
  "_meta": {
    "name": "entity_extraction_us_equities",
    "capability": "entity_extraction",
    "requires_nl_response": false,
    "requires_expected_response": false,
    "schema_version": "1.0"
  },
  "test_cases": [...]
}
```

| Capability | `requires_nl_response` | `requires_expected_response` |
|------------|------------------------|------------------------------|
| `nl2api` | `true` | `false` (until execution stage) |
| `entity_extraction` | `false` | `false` |
| `tool_generation` | `false` | `false` |

**When creating new test case sets:**
1. Always include a `_meta` block
2. Set `capability` to identify what's being evaluated
3. Set `requires_*` flags based on what the evaluation needs
4. The fixture loader validates test cases against these requirements

### Regenerating Fixtures

```bash
# Regenerate all fixtures (~19k test cases)
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

### 6. Rate Limit Resilience
For high-volume API calls (accuracy tests, batch evaluation):
- Prefer Batch API when immediate results aren't needed
- Use exponential backoff with jitter on retries
- Config in `tests/accuracy/core/config.py`

---

## Security Standards

See parent `../CLAUDE.md` for the full security checklist. This project handles **financial data queries**, so security is critical.

### Project-Specific: SQL Injection Prevention

This project uses `asyncpg` with PostgreSQL. Always use parameterized queries:

```python
# CORRECT - Use $1, $2 parameter binding
await conn.fetch("SELECT * FROM users WHERE id = $1", user_id)

# WRONG - SQL injection vulnerable
await conn.fetch(f"SELECT * FROM users WHERE id = {user_id}")
```

### Project-Specific: Safe Configuration Pattern

```python
class Config(BaseSettings):
    api_key: SecretStr  # Never logs the actual value

    model_config = SettingsConfigDict(env_prefix="NL2API_")
```

---

## Error Handling Patterns

See parent `../CLAUDE.md` for general error handling rules. Project-specific exceptions:

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

---

## Telemetry Requirements

See parent `../CLAUDE.md` for general observability principles. This project uses **OpenTelemetry (OTEL)** for tracing and metrics.

### When to Add OTEL Spans

Add spans for:
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

### MCP Server Telemetry (Project-Specific)

**MCP servers in this project MUST include client differentiation in traces.** This enables debugging multi-tenant deployments and understanding usage patterns per client.

| Component | Required Telemetry |
|-----------|-------------------|
| Server | Spans with `server.name`, `jsonrpc.method`, `jsonrpc.id` |
| Client Context | `client.session_id`, `client.transport`, `client.id` (if provided), `client.name` (if provided) |
| Tool calls | `tool.name`, `tool.success` |
| Resource reads | `resource.uri`, `resource.success` |

**Client differentiation implementation:**
1. Create a `ClientContext` dataclass with session ID, client ID, transport type
2. Use Python `contextvars` to propagate context through async call stack
3. For HTTP/SSE: Extract client info from headers (`X-Client-ID`, `X-Client-Name`, `User-Agent`)
4. For stdio: Generate session ID at startup (single client per process)
5. Add context attributes to all spans via helper method

```python
from src.mcp_servers.entity_resolution.context import (
    ClientContext,
    get_client_context,
    set_client_context,
)

# In middleware or startup
ctx = create_sse_context(client_id=request.headers.get("X-Client-ID"))
set_client_context(ctx)

# In span-creating methods
def _add_client_context_to_span(self, span: Any) -> None:
    ctx = get_client_context()
    if ctx:
        for key, value in ctx.to_span_attributes().items():
            span.set_attribute(key, value)
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
| MCP Server | `server.name`, `jsonrpc.method`, `client.session_id`, `client.transport` |
| MCP Tool | `tool.name`, `tool.success`, `client.id` (if provided) |
| MCP Resource | `resource.uri`, `resource.success` |

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

See parent `../CLAUDE.md` for general gaps. Project-specific gaps:

| Gap | Description | Workaround |
|-----|-------------|------------|
| **Migration guidance** | No documented process for DB schema changes | Always create migrations in `migrations/` for schema changes. Follow existing migration file naming: `NNN_description.sql` |
| **Backwards compatibility** | No policy for API versioning or deprecation | Avoid breaking changes to public interfaces. If unavoidable, discuss with team first |

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
- `NL2API_TELEMETRY_ENABLED`: "true" to enable OTEL metrics/tracing

### Observability Stack

The observability stack runs via `docker compose up -d`:

| Service | Port | Purpose |
|---------|------|---------|
| PostgreSQL | 5432 | Primary database |
| Redis | 6379 | Caching |
| OTEL Collector | 4317 (gRPC), 4318 (HTTP) | Receives telemetry |
| Prometheus | 9090 | Metrics storage, queries |
| Grafana | 3000 | Dashboards (admin/admin) |
| Jaeger | 16686 | Distributed tracing |

**Metrics flow:**
```
Application (OTLP) → OTEL Collector (4317) → Prometheus Exporter (8889) → Prometheus (9090) → Grafana
```

**IMPORTANT: Metric naming convention**
- OTEL Collector adds `nl2api_` prefix to all metrics (configured in `config/otel-collector-config.yaml`)
- Dashboard queries must use prefixed names: `nl2api_eval_batch_tests_total`, not `eval_batch_tests_total`
- OTEL adds `_total` suffix to counters: `eval_batch_tests_passed` becomes `nl2api_eval_batch_tests_passed_total`
- If Grafana shows no data, check metric names match what's in Prometheus

**IMPORTANT: Grafana datasource UID**
- Dashboard JSON files reference datasources by `uid` (e.g., `"uid": "prometheus"`)
- Datasource config in `config/grafana/provisioning/datasources/` MUST specify matching `uid`
- If dashboards show "No data", verify datasource UID matches between dashboard and config

**Prerequisite for batch evaluation metrics:**
1. Load fixtures: `python scripts/load_fixtures_to_db.py --all`
2. Run batch: `python -m src.evaluation.cli.main batch run --limit 10`
3. View in Grafana: http://localhost:3000 → "NL2API Evaluation & Accuracy" dashboard

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
| `src/nl2api/resolution/resolver.py` | Entity resolution implementation |
| `src/evaluation/core/evaluators.py` | Evaluation pipeline stages (Syntax, Logic, etc.) |
| `src/evaluation/batch/runner.py` | Batch evaluation runner |
| `src/evaluation/batch/metrics.py` | OTEL metrics for evaluation |
| `scripts/generators/entity_resolution_generator.py` | Entity resolution test case generator |
| `tests/unit/nl2api/test_fixture_coverage.py` | Dynamic test infrastructure |
| `tests/unit/nl2api/fixture_loader.py` | Fixture loading utility |
| `tests/accuracy/core/evaluator.py` | Accuracy testing evaluator |
| `docs/accuracy-testing.md` | Accuracy testing pattern documentation |
| `docs/evaluation-data.md` | Evaluation data and fixture documentation |
