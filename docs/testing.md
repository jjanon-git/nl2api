# Testing Guide

This document covers testing standards, patterns, and commands for the Evalkit project.

## Quick Reference

```bash
# Run tests for changed modules only (fast feedback)
./scripts/ci-test-changed.sh

# Run all unit tests (required before committing)
pytest tests/unit/ -v --tb=short -x

# Run integration tests (requires docker compose up -d)
pytest tests/integration/ -v --tb=short -x

# Run accuracy tests (requires ANTHROPIC_API_KEY)
pytest tests/accuracy/ -m tier1 -v   # Quick (~50 samples)
pytest tests/accuracy/ -m tier2 -v   # Standard (~200 samples)
pytest tests/accuracy/ -m tier3 -v   # Comprehensive (all)
```

## Module-to-Test Mapping

With 2900+ unit tests, run targeted tests during development for fast feedback:

| If you changed... | Run these tests |
|-------------------|-----------------|
| `src/nl2api/` | `pytest tests/unit/nl2api/ -x` |
| `src/evalkit/` | `pytest tests/unit/evalkit/ -x` |
| `src/rag/` | `pytest tests/unit/rag/ -x` |
| `src/common/` | `pytest tests/unit/common/ -x` |
| `src/mcp_servers/` | `pytest tests/unit/mcp_servers/ -x` |
| Multiple modules | `./scripts/ci-test-changed.sh` or full suite |

**Before committing:** Always run the full unit test suite to catch cross-module regressions.

## When to Run Accuracy Tests

- **tier1**: Quick sanity check (~50 samples, ~2 min) - Run when modifying routing/agents
- **tier2**: Standard evaluation (~200 samples, ~10 min) - Run before merging significant changes
- **tier3**: Comprehensive (~2000+ samples, ~60 min) - Run weekly or for major releases

## Integration Testing Requirements

**ALWAYS write integration tests when code involves:**
- Database operations (repository methods against real PostgreSQL)
- Multi-component flows (orchestrator → agent → storage)
- API endpoint handlers
- Configuration loading and validation

**DO NOT write automated integration tests for external third-party APIs:**
- GLEIF, SEC EDGAR, OpenFIGI, and similar public data APIs
- These APIs change, have rate limits, and require network access
- Instead: **manually verify** the integration works, then document what you tested

### When to Write Each Test Type

| Scenario | Unit Test | Integration Test | Manual Verify |
|----------|-----------|------------------|---------------|
| Pure function logic | ✅ | | |
| Class with injected dependencies | ✅ | | |
| Config defaults and behavior | ✅ | | |
| Repository CRUD operations | | ✅ | |
| Multi-component orchestration | | ✅ | |
| Database migrations | | ✅ | |
| End-to-end query processing | | ✅ | |
| External API integrations (GLEIF, SEC, etc.) | ✅ (mock) | | ✅ |

### Manual Verification for External APIs

When implementing external API integrations, before marking work complete:
1. Run the script/code manually with real credentials
2. Verify data is fetched and parsed correctly
3. Document what you tested in the PR/commit (e.g., "Manually verified: fetched 10 GLEIF records, parsed LEI/company name correctly")

## Testing Configuration Changes

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

**Why this matters:** Config changes silently affect runtime behavior. Without tests, regressions go unnoticed until production.

## Test Coverage Requirements

**After completing any new module or significant code addition, assess test coverage.**

```bash
# Check coverage for changed files
pytest tests/unit/ --cov=src/path/to/module --cov-report=term-missing -v

# Example: Check MCP server coverage
pytest tests/unit/mcp_servers/ --cov=src/mcp_servers --cov-report=term-missing -v
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

## Manual Testing Requirements

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

## Dynamic Fixture-Based Testing

Tests automatically scale with test data using programmatic fixture expansion.

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
4. [ ] Generator added to `scripts/gen-test-cases.py`
5. [ ] Category added to `FixtureLoader.CATEGORIES` in `tests/unit/nl2api/fixture_loader.py`
6. [ ] Coverage thresholds added to `CoverageRegistry.REQUIRED_COVERAGE` in `test_fixture_coverage.py`
7. [ ] Test file created in `tests/unit/nl2api/test_{category}_fixtures.py`
8. [ ] **Documentation updated in `docs/evaluation-data.md`**

## RAG Evaluation Testing

**To properly measure RAG system performance, run the full 560-case test set.**

The 400-item `sec_evaluation_set_verified.json` alone is insufficient - it only covers retrieval/generation scenarios. The additional 160 cases test policy compliance, rejection calibration, and edge cases.

### RAG Test Fixture Breakdown

| File | Count | Purpose |
|------|-------|---------|
| `sec_evaluation_set_verified.json` | 400 | Main SEC retrieval + generation |
| `adversarial_test_set.json` | 50 | Adversarial/edge cases |
| `should_reject_policy.json` | 30 | Policy rejection tests |
| `sec_evaluation_set_new.json` | 20 | New unverified cases |
| `should_answer_complete.json` | 20 | Should-answer scenarios |
| `citation_required.json` | 10 | Citation requirement tests |
| `quote_only_sources.json` | 10 | Quote-only policy tests |
| `rag_test_cases.json` | 10 | Basic RAG tests |
| `should_reject_no_context.json` | 10 | No-context rejection tests |
| **Total** | **560** | |

### Running RAG Evaluation

**Prerequisites:**
1. SEC filing data must be indexed (run `scripts/ingest-sec-filings.py` first)
2. All RAG fixtures loaded to database

```bash
# Load all RAG fixtures (560 test cases)
for f in tests/fixtures/rag/*.json; do
  python scripts/load-rag-fixtures.py --fixture "$f"
done

# Run full RAG evaluation with OpenAI stack
EVAL_LLM_PROVIDER=openai \
EVALKIT_TELEMETRY_ENABLED=true \
python -m src.evalkit.cli.main batch run \
  --pack rag \
  --tag rag \
  --label "full-rag-eval" \
  --mode generation \
  --parallel-stages \
  --concurrency 5 \
  -v
```

### What Each Stage Measures

| Stage | Measures | Threshold |
|-------|----------|-----------|
| `retrieval` | Recall@5 of expected chunks | 0.5 |
| `context_relevance` | Retrieved context quality | 0.25 (OpenAI) / 0.35 (Anthropic) |
| `faithfulness` | Response grounded in context | 0.4 |
| `answer_relevance` | Response addresses query | 0.5 |
| `citation` | Citation presence/accuracy | 0.6 |
| `source_policy` | Quote-only enforcement | GATE (must pass) |
| `policy_compliance` | Content policy violations | GATE (must pass) |
| `rejection_calibration` | Correct rejection behavior | 0.5 |

**Important:** Without the full 560 cases, you'll miss:
- Policy rejection testing (should model refuse?)
- Citation requirement validation
- Edge case handling (adversarial queries)
- Rejection calibration (false positives/negatives)

## Pre-commit Checklist

1. ✅ Unit tests pass: `pytest tests/unit/ -v --tb=short -x`
2. ✅ Integration tests pass: `pytest tests/integration/ -v --tb=short -x`
3. ✅ Linting passes: `ruff check .`
4. ✅ Test coverage assessed for new code
5. ✅ Manual testing completed for servers/CLIs
6. ✅ Security checklist reviewed
7. ✅ Telemetry added for external calls/DB operations
8. ✅ No regressions in changed areas
