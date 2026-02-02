# Evaluation Pipeline Guide

This document covers the Evalkit evaluation pipeline architecture, standards, and best practices.

**The evaluation pipeline is critical infrastructure.** Changes must be carefully validated.

## Pipeline Architecture

```
TestCase → NL2APIPack → Scorecard
              │
              ├─ Stage 1: SyntaxStage (GATE - hard stop on failure)
              ├─ Stage 2: LogicStage (AST comparison)
              ├─ Stage 3: ExecutionStage (live API verification, deferred)
              └─ Stage 4: SemanticsStage (LLM-as-Judge)
```

See `src/evalkit/packs/nl2api.py` for stage implementations.

## When to Modify Evaluation Code

| Scenario | Action |
|----------|--------|
| New tool/API added | Add test fixtures + update pack if needed |
| New evaluation metric | Add to `src/contracts/` first, then implement in pack |
| Scoring logic change | Requires tier2 accuracy test before/after comparison |
| New evaluation stage | Must integrate with OTEL (see telemetry.md) |

## Test Fixture Requirements

When adding or modifying test fixtures in `tests/fixtures/`:

1. **Follow the TestCase contract** (defined in `src/contracts/core.py`):
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

## Evaluator Implementation Rules

1. **All evaluators must return `StageResult`** with:
   - `passed`: bool
   - `score`: float (0.0-1.0)
   - `error_code`: ErrorCode enum if failed
   - `reason`: Human-readable explanation
   - `duration_ms`: Execution time

2. **GATE stages stop the pipeline** - If Stage 1 (Syntax) fails, subsequent stages don't run

3. **All evaluators must integrate with OTEL** (see [telemetry.md](telemetry.md)):
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
   from src.evalkit.batch.metrics import get_metrics

   metrics = get_metrics()
   metrics.record_test_result(scorecard, batch_id, tags)
   ```

## Scorecard Immutability

Scorecards are **immutable** once created. Never modify a scorecard after evaluation:

```python
# CORRECT - Create new scorecard with updated fields
new_scorecard = scorecard.model_copy(update={"notes": "reprocessed"})

# WRONG - Scorecards are frozen
scorecard.notes = "reprocessed"  # Raises error
```

## Batch Evaluation

### Modifying Batch Code

When modifying `src/evalkit/batch/`:

1. **Checkpoint/resume must be preserved** - Don't break resumable batch runs
2. **Metrics must be recorded** - Use `BatchMetrics` for all test results
3. **Test with small batches first**: `batch run --pack nl2api --limit 10`

### Batch Evaluation vs Unit Tests

**Use the right tool for the right purpose:**

| Purpose | Tool | Persisted? | When to Use |
|---------|------|------------|-------------|
| **Accuracy tracking** | `batch run --pack nl2api --mode resolver` | ✅ Yes | Measure real system accuracy over time |
| **End-to-end accuracy** | `batch run --pack nl2api --mode orchestrator` | ✅ Yes | Full pipeline accuracy (costs API credits) |
| **Pipeline testing** | `batch run --pack nl2api --mode simulated` | ❌ No* | Test evaluation infrastructure only |
| **RAG evaluation** | `batch run --pack rag` | ✅ Yes | Evaluate RAG systems (retrieval, faithfulness) |
| **Code correctness** | Unit tests (`pytest`) | ❌ No | Verify code behavior with mocks |

*Simulated results ARE persisted but SHOULD NOT be used for accuracy tracking.

**The key insight:**
- **Batch evaluation** is for measuring **real system performance** against test cases
- **Unit tests** are for verifying **code correctness** with mocked dependencies
- **Simulated responses** produce 100% pass rates and are meaningless for tracking improvement

### Running Batch Evaluation

**Pack AND Tag selection is REQUIRED:**
```bash
# --pack and --tag are REQUIRED - ensures reproducible, trackable evaluation runs
python -m src.evalkit.cli.main batch run --pack nl2api --tag entity_resolution --label my-test

# Available packs, tags, and modes
batch run --pack nl2api --tag entity_resolution --label <label>  # Entity resolution accuracy
batch run --pack nl2api --tag lookups --label <label>            # Single/multi-field queries
batch run --pack nl2api --tag temporal --label <label>           # Date-based queries
batch run --pack nl2api --tag screening --label <label>          # Screening/filtering
batch run --pack rag --tag rag --label <label>                   # RAG retrieval evaluation

# Additional mode options for NL2API
--mode resolver      # (DEFAULT) Entity resolution only
--mode orchestrator  # Full pipeline (requires LLM API key)
```

### RAG Evaluation Prerequisites

```bash
# 1. Load RAG fixtures to database (one-time setup)
python scripts/load-rag-fixtures.py

# 2. Run RAG evaluation with proper tracking
python -m src.evalkit.cli.main batch run --pack rag --tag rag --label my-experiment

# 3. View results in Grafana at http://localhost:3000
```

**IMPORTANT:** Never use `scripts/run_rag_baseline.py` directly - it's deprecated and doesn't integrate with the observability stack. Always use the batch framework for tracked, reproducible evaluation runs.

### What Gets Persisted

**What gets persisted to track over time:**
- Scorecards with pass/fail for each test case
- Batch job metadata (total, passed, failed, duration)
- Results visible in Grafana dashboards

**What should stay in unit tests:**
- Pipeline infrastructure validation (use mocked responses)
- Evaluator logic correctness
- Repository CRUD operations

## Creating New Evaluation Packs

**When creating a new evaluation pack (e.g., RAG, code-gen, etc.), follow this checklist:**

1. **Create the pack implementation** in `src/evalkit/packs/`:
   ```python
   from src.contracts.evaluation import EvaluationPack, Stage

   class MyPack:
       @property
       def name(self) -> str:
           return "my_pack"  # Used in metrics and dashboards

       def get_stages(self) -> list[Stage]:
           return [MyStage1(), MyStage2()]
   ```

2. **Implement pack stages** that implement the `Stage` protocol:
   - Each stage has `name`, `is_gate`, and `evaluate()` method
   - First stage should typically be a GATE (stops pipeline on failure)
   - Use descriptive stage names (they appear in metrics)

3. **Add pack-specific unit tests** in `tests/unit/evalkit/test_packs.py`

4. **Update dashboards for the new pack**:
   - The evaluation dashboard (`config/grafana/provisioning/dashboards/json/evaluation-dashboard.json`) uses `pack_name` label for filtering
   - **If your pack has custom stages**, add stage-specific panels similar to the NL2API ones
   - The `Failures by Stage` chart automatically shows all stage names

5. **Verify metrics flow**:
   - `pack_name` label is automatically added to all metrics by `EvalMetrics`
   - Stage names are recorded dynamically from `scorecard.get_all_stage_results()`
   - Test by running a small batch and checking Prometheus/Grafana

### Dashboard Patterns for New Packs

| Panel Type | How It Works |
|------------|--------------|
| Overall Pass Rate | Filters by `pack_name=~"$pack_name"` - works automatically |
| Per-Stage Pass Rate | Requires stage name in query (e.g., `stage='retrieval'`) |
| Failures by Stage | Groups by `stage` label - shows all stages automatically |

**Example: Adding RAG pack panels:**

If your RAG pack has stages `retrieval` and `faithfulness`, add panels like:
```json
{
  "title": "Retrieval Pass Rate",
  "expr": "100 * sum(evalkit_eval_stage_passed_total{stage='retrieval', pack_name='rag'}) / clamp_min(sum(evalkit_eval_stage_passed_total{stage='retrieval', pack_name='rag'}) + sum(evalkit_eval_stage_failed_total{stage='retrieval', pack_name='rag'}), 1)"
}
```

## Batch Evaluation Verification

**Never declare a batch evaluation "working" without verifying actual results.**

### Before Declaring Success

1. **Wait for actual results** - At least 5 scorecards must be completed
2. **Verify data quality** - Check that:
   - `generated_nl_response` is NOT empty (for generation mode)
   - Stage results have meaningful scores (not all 0.00)
   - No "LLM call failed" errors in stage reasons
3. **Check the right batch** - Verify you're looking at the current batch_id, not old data

### After Code Fixes

When fixing code that affects a running batch:

1. **Kill the old process first** - `pkill -f "batch run"`
2. **Verify the fix is in the code** - `grep` for the change
3. **Restart with fresh environment** - Ensure env vars are set
4. **Verify new process picks up changes** - Check scorecards from the NEW batch_id

### End-to-End Field Tracing

When adding/fixing fields that flow through the pipeline:

1. **Trace the full path**: response generator → runner → pack → Scorecard creation
2. **Don't stop at the first file** - Follow the data all the way to where it's saved
3. **Verify in database** - Query the actual stored data, not just code inspection

### Verification Query

```sql
-- Quick check for batch health
SELECT
    COUNT(*) as total,
    COUNT(CASE WHEN generated_nl_response IS NOT NULL AND generated_nl_response != '' THEN 1 END) as with_response,
    COUNT(CASE WHEN overall_passed THEN 1 END) as passed
FROM scorecards
WHERE batch_id = 'YOUR_BATCH_ID';
```

**Root cause of past failures:** Shallow verification (checking "is it running?" instead of "are results correct?"), and not following data flow end-to-end through the codebase.

## Evaluation Data Generation Standards

**Follow these guidelines when working with test fixtures and evaluation data.**

### Key Decisions (Documented)

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Git strategy for fixtures | **Commit to git** | Reproducibility, review visibility, no CI API costs |
| NL response generation model | **Claude 3.5 Haiku** | Better quality than 3 Haiku; negligible cost difference |
| `expected_response` field | **Leave null** | Until execution stage is implemented (deferred) |
| Fixture schema | **Align with src/contracts/** | Generator dataclass must match `TestCase` contract |
| Batch eval default mode | **`resolver` (real)** | Persist meaningful accuracy, not simulated 100% |
| Simulated responses | **Unit tests only** | Pipeline infra testing doesn't need persistence |

See `docs/plans/evaluation-data-contract-plan.md` for full rationale.

### Test Case Field Definitions

```python
# In src/contracts/core.py
class TestCase:
    expected_response: dict | None      # Raw API data (e.g., {"AAPL.O": {"P": 246.02}})
                                        # Currently NULL - populate when execution stage added
    expected_nl_response: str | None    # Human-readable sentence (e.g., "Apple's PE ratio is $246.02")
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
python scripts/gen-test-cases.py --all

# Regenerate specific category
python scripts/gen-test-cases.py --category lookups

# Validate generated output against src/contracts/ schemas
python scripts/gen-test-cases.py --validate

# Generate NL responses (uses Claude 3.5 Haiku, ~$5 cost)
python scripts/generate_nl_responses.py --all
```

### When Adding New Evaluation Data

1. **Add to source data** in `data/field_codes/` or `data/tickers/`
2. **Regenerate fixtures** using the generator scripts
3. **Commit generated fixtures** to git (don't generate in CI)
4. **Run validation**: `python scripts/gen-test-cases.py --validate`
5. **Run tests**: `pytest tests/unit/nl2api/test_fixture_coverage.py -v`

### Synthetic Data Caveats

All evaluation data is synthetic. When documenting or using:

- `expected_nl_response` values are **LLM-generated** (Claude 3.5 Haiku)
- `expected_response` values (when populated) are **based on API specs**, not live calls
- API specifications were **reverse-engineered** from public documentation
- Ticker/company data is **point-in-time** and may become stale
