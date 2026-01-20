# CLAUDE.md - Project Context for Claude Code

## Project Overview

**EvalPlatform** is a distributed evaluation framework for testing LLM tool-calling at scale (~400k test cases). It validates that LLM orchestrators correctly parse queries, call the right APIs with correct arguments, and return appropriate responses.

## Quick Commands

```bash
# Start PostgreSQL (required for storage)
docker compose up -d

# Run tests
.venv/bin/python -m pytest tests/unit/ -v

# Run single test case evaluation
.venv/bin/python -m src.cli.main run tests/fixtures/search_products.json

# Run batch evaluation
.venv/bin/python -m src.cli.main batch run --limit 10

# Check batch status
.venv/bin/python -m src.cli.main batch list

# View batch results
.venv/bin/python -m src.cli.main batch results <batch-id>
```

## Architecture

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
- **Stage 3 (Execution)**: Live API verification - CRITICAL (not yet implemented)
- **Stage 4 (Semantics)**: LLM-as-Judge NL comparison - LOW priority (not yet implemented)

### Key Data Models (CONTRACTS.py)

```python
TestCase     # The "Gold Standard" - nl_query, expected_tool_calls, expected_nl_response
ToolCall     # Tool name + frozen arguments (hashable for set comparison)
Scorecard    # Evaluation result with per-stage results and overall pass/fail
BatchJob     # Batch tracking with status and progress
```

### Storage Layer

Protocol-based abstraction with implementations for:
- **PostgreSQL** (`src/storage/postgres/`) - Local development
- **In-Memory** (`src/storage/memory/`) - Unit tests
- **Azure** (not yet implemented) - Production

Factory pattern: `create_repositories(config) -> (test_case_repo, scorecard_repo, batch_repo)`

## Directory Structure

```
evalPlatform/
├── CONTRACTS.py              # Pydantic v2 data models (TestCase, Scorecard, etc.)
├── src/
│   ├── batch/                # Batch runner with concurrency control
│   │   ├── runner.py         # BatchRunner class
│   │   ├── config.py         # BatchRunnerConfig
│   │   └── metrics.py        # OpenTelemetry metrics
│   ├── cli/
│   │   ├── main.py           # CLI entry point
│   │   └── commands/
│   │       ├── run.py        # Single evaluation command
│   │       └── batch.py      # Batch commands (run, status, results, list)
│   ├── core/
│   │   ├── evaluators.py     # SyntaxEvaluator, LogicEvaluator, WaterfallEvaluator
│   │   └── ast_comparator.py # Tool call comparison (order-independent, type-coercing)
│   └── storage/
│       ├── protocols.py      # Repository protocols (duck-typed interfaces)
│       ├── factory.py        # Repository factory
│       ├── postgres/         # PostgreSQL implementations
│       └── memory/           # In-memory implementations
├── tests/
│   ├── unit/                 # Unit tests (71 tests, ~1s)
│   └── fixtures/             # Sample test cases
└── docker-compose.yml        # PostgreSQL + pgvector
```

## Key Patterns

### 1. Protocol-Based Repositories
```python
@runtime_checkable
class TestCaseRepository(Protocol):
    async def get(self, test_case_id: str) -> TestCase | None: ...
    async def save(self, test_case: TestCase) -> None: ...
    async def list(...) -> list[TestCase]: ...
```

### 2. Frozen Pydantic Models
All data models use `model_config = ConfigDict(frozen=True)` for immutability.

### 3. Order-Independent Tool Call Comparison
`ToolCall.arguments` uses `FrozenDict` for hashability, enabling set-based comparison.

### 4. Async-First Design
All repository operations and evaluators are async.

### 5. Semaphore-Based Concurrency
`BatchRunner` uses `asyncio.Semaphore(max_concurrency)` for controlled parallelism.

## Development Notes

### Running Tests
```bash
# All unit tests
.venv/bin/python -m pytest tests/unit/ -v

# Specific test file
.venv/bin/python -m pytest tests/unit/test_batch_runner.py -v
```

### Database
- Uses PostgreSQL 16 with pgvector extension
- Schema in `src/storage/postgres/migrations/001_initial.sql`
- Tables: `test_cases`, `scorecards`, `batch_jobs`

### Environment Variables
- `EVAL_BACKEND`: "postgres" | "memory" | "azure"
- `EVAL_POSTGRES_URL`: PostgreSQL connection string (default: localhost)

## Current Sprint Status

Sprint 3 implemented: Batch Runner with local concurrency
- Batch run/status/results/list commands
- PostgreSQL batch job persistence
- Progress tracking with Rich
- OpenTelemetry metrics (optional)

## Important Files to Review

| File | Purpose |
|------|---------|
| `CONTRACTS.py` | All data models - read this first |
| `src/core/evaluators.py` | Evaluation pipeline implementation |
| `src/storage/protocols.py` | Repository interfaces |
| `src/batch/runner.py` | Batch execution logic |
| `src/cli/commands/batch.py` | CLI command implementations |
