# EvalPlatform

Distributed evaluation framework for testing LLM tool-calling at scale (~400k test cases), with an embedded **NL2API system** for translating natural language queries into LSEG financial API calls.

## Quick Start

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install with dev dependencies
pip install -e ".[dev]"

# Start PostgreSQL (required for storage)
docker compose up -d

# Run tests
.venv/bin/python -m pytest tests/unit/ -v

# Run single test case evaluation
.venv/bin/python -m src.cli.main run tests/fixtures/search_products.json

# Run batch evaluation
.venv/bin/python -m src.cli.main batch run --limit 10

# View batch results
.venv/bin/python -m src.cli.main batch list
```

---

## Features

### NL2API System

A complete natural language to API translation system for LSEG financial data services:

| Component | Description |
|-----------|-------------|
| **NL2APIOrchestrator** | Main entry point - routes queries to appropriate domain agents |
| **5 Domain Agents** | Specialized agents for different API domains (see below) |
| **Multi-turn Conversations** | Session-based conversation support with query expansion |
| **Entity Resolution** | Company name → RIC code resolution |
| **Clarification Flow** | Ambiguity detection and clarifying questions |

**Domain Agents:**

| Agent | Domain | Capabilities |
|-------|--------|--------------|
| `DatastreamAgent` | Price & Time Series | Current price, historical data, technical indicators |
| `ScreeningAgent` | Stock Screening | SCREEN expressions, index constituents, rankings |
| `EstimatesAgent` | Analyst Estimates | EPS, revenue forecasts, recommendations |
| `FundamentalsAgent` | Financial Data | Balance sheet, income statement, ratios |
| `OfficersAgent` | Corporate Governance | Executives, compensation, board members |

### Evaluation Pipeline

- **Stage 1 (Syntax)**: JSON structure validation - hard stop on failure
- **Stage 2 (Logic)**: AST-based tool call comparison with order-independence and type coercion
- **Stage 3 (Execution)**: Live API verification (configurable)
- **Stage 4 (Semantics)**: LLM-as-Judge NL comparison (configurable)

### Batch Processing

- Concurrent evaluation with configurable parallelism
- Progress tracking with Rich
- Batch status and results commands
- Optional OpenTelemetry metrics

### Storage

- PostgreSQL backend with pgvector
- In-memory repositories for unit tests
- Protocol-based abstraction for future Azure integration

---

## Test Coverage

### Dynamic Fixture-Based Testing

The test suite uses **programmatic fixture expansion** - tests automatically scale as test data grows.

```
Total Test Cases:     12,887 generated fixtures
Total Unit Tests:     497 passing (+ evaluation tests)

Fixture Categories:
├── lookups/       3,745 cases (single/multi-field queries)
├── temporal/      2,727 cases (historical, time series)
├── comparisons/   3,658 cases (multi-stock comparisons)
├── screening/       265 cases (SCREEN expressions)
├── complex/       2,277 cases (multi-step queries)
└── errors/          215 cases (error handling)
```

### How Fixture-Based Testing Works

Test files in `tests/unit/nl2api/` automatically discover and test against all fixtures:

1. **Dynamic Discovery**: Tests discover categories, subcategories, and tags at runtime
2. **Parameterized Tests**: pytest generates test cases from fixture structure
3. **Coverage Enforcement**: Minimum coverage thresholds fail tests if agents regress
4. **Growth Detection**: Tests alert when fixture counts change significantly

**Key Files:**

| File | Purpose |
|------|---------|
| `tests/unit/nl2api/fixture_loader.py` | Utility for loading generated fixtures |
| `tests/unit/nl2api/test_fixture_coverage.py` | Dynamic coverage tests that auto-expand |
| `tests/unit/nl2api/test_datastream_fixtures.py` | DatastreamAgent fixture-based tests |
| `tests/unit/nl2api/test_screening_fixtures.py` | ScreeningAgent fixture-based tests |

### Coverage Registry

The `CoverageRegistry` class defines minimum coverage thresholds:

```python
class CoverageRegistry:
    REQUIRED_COVERAGE = [
        ("lookups", "single_field", 0.3, DatastreamAgent),
        ("lookups", "multi_field", 0.15, DatastreamAgent),
        ("temporal", "historical_price", 0.4, DatastreamAgent),
        ("screening", "index_constituents", 0.3, ScreeningAgent),
        ("screening", "top_n", 0.5, ScreeningAgent),
        # ... more thresholds
    ]
```

When coverage drops below these thresholds, tests fail - preventing regressions.

### Adding New Test Data

To add new test fixtures:

1. Add JSON files to `tests/fixtures/lseg/generated/<category>/`
2. Tests automatically discover and include the new data
3. If coverage is zero for new subcategories, tests will warn (but not fail for expected-zero categories like `complex/` or `errors/`)

---

## Documentation

- **ARCHITECTURE.md** - Full system design and API specifications
- **STATUS.md** - Current implementation status and next steps
- **CLAUDE.md** - Quick reference for AI assistants
- **\*_REFERENCE.md** - LSEG API reference docs (Datastream, Estimates, Fundamentals, Officers, Screening)
