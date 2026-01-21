# Accuracy Testing Pattern for NL2API

**Status:** In Progress (Phase 1 Complete)

---

## Overview

This document defines the pattern for **accuracy tests** in NL2API - tests that measure the quality and correctness of system outputs using real LLM calls.

## Key Distinction: Unit Tests vs Accuracy Tests

| Aspect | Unit Tests | Accuracy Tests |
|--------|-----------|----------------|
| **Location** | `tests/unit/` | `tests/accuracy/` |
| **LLM Calls** | Mocked | Real |
| **Purpose** | Test code behavior | Measure output quality |
| **Assertions** | Exact matches | Threshold-based (≥80%) |
| **Speed** | Fast (ms) | Slower (seconds per query) |
| **Output** | Pass/Fail | Metrics report + Pass/Fail |
| **Question Answered** | "Does the code work correctly?" | "Does the system produce correct outputs?" |

Both test types can use the same fixture data from `tests/fixtures/`:

```
tests/fixtures/lseg/generated/
           │
           ├──► Unit Tests (mocked LLM)
           │    - Test rule-based pattern matching
           │    - Test code paths
           │    - Fast, deterministic
           │
           └──► Accuracy Tests (real LLM)
                - Measure end-to-end correctness
                - Report accuracy metrics
                - Track quality over time
```

---

## Directory Structure

```
tests/
├── unit/                           # Unit tests (existing)
│   └── nl2api/
│       ├── test_*.py              # Agent/component tests
│       └── test_fixture_coverage.py  # Fixture-based unit tests
│
├── integration/                    # API contract tests (existing)
│   └── test_api_contract.py
│
├── accuracy/                       # Accuracy tests (NEW)
│   ├── __init__.py
│   ├── conftest.py                # Shared fixtures, LLM setup
│   │
│   ├── core/                      # Shared infrastructure
│   │   ├── __init__.py
│   │   ├── config.py             # Tiers, thresholds
│   │   └── evaluator.py          # AccuracyEvaluator, AccuracyReport
│   │
│   ├── agents/                    # Per-agent accuracy
│   │   ├── test_datastream_accuracy.py
│   │   ├── test_estimates_accuracy.py
│   │   ├── test_fundamentals_accuracy.py
│   │   ├── test_officers_accuracy.py
│   │   └── test_screening_accuracy.py
│   │
│   └── domains/                   # Per-domain accuracy
│       ├── test_lookups_accuracy.py
│       ├── test_temporal_accuracy.py
│       ├── test_comparisons_accuracy.py
│       ├── test_screening_accuracy.py
│       └── test_complex_accuracy.py
│
└── fixtures/lseg/generated/       # Shared test data (existing)
    ├── lookups/        (3,745 cases)
    ├── temporal/       (2,727 cases)
    ├── comparisons/    (3,658 cases)
    ├── screening/        (265 cases)
    ├── complex/        (2,277 cases)
    └── errors/           (215 cases)
```

---

## Test Tiers

Accuracy tests support tiered execution for different use cases:

| Tier | Samples | Categories | Threshold | Time | Use Case |
|------|---------|------------|-----------|------|----------|
| **Tier 1** | 50 | lookups, temporal | 75% | <5 min | PR checks, quick feedback |
| **Tier 2** | 200 | lookups, temporal, comparisons, screening | 80% | ~15 min | Daily CI |
| **Tier 3** | All | All categories | 80% | ~1 hr+ | Weekly comprehensive |

### Running Tiers

```bash
# Quick sanity check (PR checks)
pytest tests/accuracy/ -m tier1

# Standard evaluation (daily CI)
pytest tests/accuracy/ -m tier2

# Comprehensive evaluation (weekly)
pytest tests/accuracy/ -m tier3
```

---

## Accuracy Thresholds

### Global Threshold

**80% minimum accuracy** (TODO: Evaluate and adjust based on baseline runs)

### Category-Specific Thresholds

| Category | Threshold | Rationale |
|----------|-----------|-----------|
| `lookups` | 85% | Simple queries, should be highly accurate |
| `temporal` | 80% | Time series queries |
| `comparisons` | 75% | Multi-stock comparisons are harder |
| `screening` | 75% | SCREEN expressions are complex |
| `complex` | 70% | Multi-step queries, expected to be hardest |
| `errors` | 90% | Error detection should be reliable |

---

## Metrics and Observability

Accuracy test results are emitted to the same OTEL stack as request metrics:

```
Accuracy Tests
     │
     ▼ (OTLP)
OTEL Collector
     │
     ├──► Prometheus (metrics)
     │         │
     │         ▼
     │     Grafana Dashboard
     │         - Accuracy trends over time
     │         - Per-category breakdown
     │         - Degradation alerts
     │
     └──► Jaeger (traces)
              - Per-query trace for debugging failures
```

### Metrics Emitted

```python
# Counter: Total accuracy test runs
accuracy_tests_total{tier, category, agent}

# Counter: Passed/failed tests
accuracy_tests_passed{tier, category, agent}
accuracy_tests_failed{tier, category, agent}

# Histogram: Accuracy score distribution
accuracy_score{tier, category, agent}

# Gauge: Current accuracy rate
accuracy_rate{tier, category, agent}
```

---

## Implementation Pattern

### AccuracyEvaluator

```python
class AccuracyEvaluator:
    """Evaluates NL2API accuracy using real LLM calls."""

    async def evaluate_query(
        self,
        query: str,
        expected_tool_calls: list[ToolCall],
    ) -> AccuracyResult:
        """
        Evaluate a single query against expected output.

        Returns:
            AccuracyResult with correct/incorrect and details
        """

    async def evaluate_batch(
        self,
        test_cases: list[TestCase],
        parallel: int = 5,
    ) -> AccuracyReport:
        """
        Evaluate a batch of test cases.

        Returns:
            AccuracyReport with overall accuracy and breakdown
        """
```

### Accuracy Test Structure

```python
# tests/accuracy/agents/test_datastream_accuracy.py

import pytest
from tests.accuracy.core import AccuracyEvaluator, AccuracyConfig, Tier

class TestDatastreamAccuracy:
    """Accuracy tests for DatastreamAgent."""

    @pytest.mark.tier1
    @pytest.mark.asyncio
    async def test_price_queries_accuracy(self, evaluator, fixture_loader):
        """Measure accuracy on price lookup queries."""
        cases = fixture_loader.load_by_subcategory("lookups", "single_field", limit=50)

        report = await evaluator.evaluate_batch(cases)

        assert report.accuracy >= 0.80, (
            f"Price query accuracy {report.accuracy:.1%} below 80% threshold. "
            f"Failed: {report.failed_count}/{report.total_count}"
        )

    @pytest.mark.tier2
    @pytest.mark.asyncio
    async def test_temporal_queries_accuracy(self, evaluator, fixture_loader):
        """Measure accuracy on time series queries."""
        cases = fixture_loader.load_category("temporal", limit=100)

        report = await evaluator.evaluate_batch(cases)

        assert report.accuracy >= 0.80
```

---

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/accuracy.yml
name: Accuracy Tests

on:
  pull_request:
    branches: [main]
  schedule:
    - cron: '0 6 * * *'  # Daily at 6 AM
    - cron: '0 0 * * 0'  # Weekly on Sunday

jobs:
  tier1:
    name: Quick Accuracy Check
    if: github.event_name == 'pull_request'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pytest tests/accuracy/ -m tier1

  tier2:
    name: Standard Accuracy Eval
    if: github.event_name == 'schedule' || github.ref == 'refs/heads/main'
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pytest tests/accuracy/ -m tier2

  tier3:
    name: Comprehensive Accuracy
    if: github.event.schedule == '0 0 * * 0'  # Weekly only
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: pytest tests/accuracy/ -m tier3
```

---

## Environment Requirements

```bash
# Required environment variables
NL2API_ANTHROPIC_API_KEY=sk-ant-...   # For Claude
# OR
NL2API_OPENAI_API_KEY=sk-...          # For OpenAI

# Optional
NL2API_TELEMETRY_ENABLED=true         # Emit to OTEL
NL2API_TELEMETRY_OTLP_ENDPOINT=http://localhost:4317
```

---

## Current Status

### Completed ✅

- [x] `tests/accuracy/core/` - AccuracyEvaluator, RoutingAccuracyEvaluator, config, thresholds
- [x] `tests/accuracy/routing/test_routing_accuracy.py` - Routing accuracy with tier1/tier2/tier3
- [x] `tests/accuracy/agents/test_datastream_accuracy.py`
- [x] `tests/accuracy/agents/test_estimates_accuracy.py`
- [x] `tests/accuracy/domains/test_lookups_accuracy.py`
- [x] `tests/accuracy/domains/test_screening_accuracy.py`
- [x] `scripts/eval_routing.py` - CLI for routing evaluation
- [x] Updated routing prompt for temporal ambiguity (confidence scoring)
- [x] `.github/workflows/accuracy.yml` - CI/CD integration

### In Progress 🔄

Coverage gaps to fill:

```
tests/accuracy/agents/
  ✅ test_datastream_accuracy.py
  ✅ test_estimates_accuracy.py
  ❌ test_fundamentals_accuracy.py    # TODO
  ❌ test_officers_accuracy.py        # TODO
  ❌ test_screening_accuracy.py       # TODO (agent-specific)

tests/accuracy/domains/
  ✅ test_lookups_accuracy.py
  ✅ test_screening_accuracy.py
  ❌ test_temporal_accuracy.py        # TODO
  ❌ test_comparisons_accuracy.py     # TODO
  ❌ test_complex_accuracy.py         # TODO
  ❌ test_errors_accuracy.py          # TODO
```

---

## Next Steps

### Phase 2: Complete Test Coverage

1. **Agent Accuracy Tests** (Priority: High)
   - [ ] `test_fundamentals_accuracy.py` - Balance sheet, income statement, ratios
   - [ ] `test_officers_accuracy.py` - CEO/CFO info, board members, compensation
   - [ ] `test_screening_accuracy.py` - SCREEN expressions, top-N rankings

2. **Domain Accuracy Tests** (Priority: Medium)
   - [ ] `test_temporal_accuracy.py` - Time series, date ranges, historical data
   - [ ] `test_comparisons_accuracy.py` - Multi-stock comparisons
   - [ ] `test_complex_accuracy.py` - Multi-step queries, combined operations
   - [ ] `test_errors_accuracy.py` - Error detection, invalid query handling

3. **End-to-End Tests** (Priority: Low)
   - [ ] `tests/accuracy/e2e/test_orchestrator_accuracy.py` - Full pipeline

### Phase 3: Evaluation & Tuning

4. **Baseline Establishment**
   - [ ] Run tier3 comprehensive evaluation
   - [ ] Document baseline accuracy per category
   - [ ] Calibrate thresholds based on actual results

5. **Performance Benchmarking**
   - [ ] Compare keyword router vs LLM router accuracy
   - [ ] Measure latency impact of routing cache
   - [ ] Document cost per evaluation run

---

## Open Questions / Design Decisions

### Threshold Calibration
Current 80% global threshold is a starting point. After baseline runs:
- Adjust per-category thresholds based on observed accuracy
- Consider tighter thresholds for production-critical categories

### Flaky Test Handling
LLM outputs can vary. Current mitigations:
- Retries (max 2)
- Confidence scoring for ambiguous queries
- Consider: semantic similarity matching instead of exact match

### Cost Management
Real LLM calls have cost. Current approach:
- Use Haiku for tier1/tier2 (fast, cheap)
- Sonnet for tier3 comprehensive (higher quality)
- Future: Track cost per evaluation run, budget alerts

### Historical Tracking
Store accuracy results over time to detect degradation:
- Results saved to `routing_eval_*.json` (gitignored)
- Future: Push to OTEL for Grafana dashboards

---

## References

- [CLAUDE.md](../CLAUDE.md) - Project overview and test commands
- [Fixture Loader](../tests/unit/nl2api/fixture_loader.py) - Existing fixture infrastructure
- [Evaluation Pipeline](../src/evaluation/) - Existing evaluation framework
- [OTEL Setup](../src/common/telemetry/) - Telemetry integration
