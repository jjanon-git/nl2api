# Evaluation Data Documentation

This document describes the test case fixtures used for evaluation in the NL2API system.

## Overview

The evaluation system uses generated test fixtures stored in `tests/fixtures/lseg/generated/`.
Each category has its own directory with a JSON file containing test cases.

## Running Evaluations

### Batch Evaluation (for accuracy tracking)

```bash
# Run with real EntityResolver (default) - results are persisted
.venv/bin/python -m src.evaluation.cli.main batch run --tag entity_resolution --limit 100

# Run with full orchestrator (requires LLM API key)
batch run --mode orchestrator --tag entity_resolution --limit 100

# View results
batch run list
batch run results <batch-id>
```

### Evaluation Modes

| Mode | Command | Purpose | Persist? |
|------|---------|---------|----------|
| `resolver` | `--mode resolver` (default) | Real EntityResolver accuracy | ✅ Yes |
| `orchestrator` | `--mode orchestrator` | Full NL2API pipeline | ✅ Yes |
| `simulated` | `--mode simulated` | Pipeline testing only | ⚠️ Not for tracking |

**Important:** The default mode (`resolver`) uses the real system and persists results for tracking accuracy over time. Use `--mode simulated` only for testing the evaluation pipeline infrastructure.

### Unit Tests vs Batch Evaluation

- **Unit tests** (`pytest`): Verify code correctness with mocked dependencies
- **Batch evaluation** (`batch run`): Measure real system accuracy, persisted for tracking

See `CLAUDE.md` section "Batch Evaluation vs Unit Tests" for the full decision rationale.

## Test Case Categories

| Category | Count | Description |
|----------|-------|-------------|
| `lookups` | ~3,700 | Single/multi-field lookup queries |
| `temporal` | ~2,700 | Time series and temporal variants |
| `comparisons` | ~3,600 | Multi-ticker comparison queries |
| `screening` | ~260 | Stock screening queries |
| `complex` | ~2,200 | Multi-step workflows |
| `errors` | varies | Error scenario test cases |
| `entity_resolution` | ~5,000 | Entity name/ticker to RIC resolution |

---

## Entity Resolution Test Cases

### Overview

The entity resolution test set evaluates the system's ability to map company names,
tickers, and identifiers to canonical RICs (Reuters Instrument Codes).

**Capability:** `entity_resolution`
**Test Cases:** ~5,000
**Source Data:** GLEIF (2.95M entities) + SEC EDGAR (8K entities)

### Test Case Structure

Entity resolution uses a different structure than NL2API test cases:

```json
{
  "_meta": {
    "name": "entity_resolution",
    "capability": "entity_resolution",
    "description": "Entity name/ticker to RIC resolution tests",
    "requires_nl_response": false,
    "requires_expected_response": false,
    "schema_version": "1.0"
  },
  "test_cases": [
    {
      "id": "entity_resolution_abc123",
      "nl_query": "What is Apple's revenue?",
      "expected_tool_calls": [{
        "tool_name": "get_data",
        "arguments": {"tickers": ["AAPL.O"], "fields": ["TR.Revenue"]}
      }],
      "expected_response": null,
      "expected_nl_response": null,
      "category": "entity_resolution",
      "subcategory": "exact_match",
      "metadata": {
        "input_entity": "Apple",
        "expected_ric": "AAPL.O",
        "expected_lei": "HWUPKR0MPOU8FGXBT394",
        "match_type": "exact",
        "confidence_threshold": 0.95
      }
    }
  ]
}
```

**Key fields:**
- `expected_nl_response`: Always `null` (entity resolution is a structured lookup)
- `expected_response`: Always `null` (resolution result is in `metadata.expected_ric`)
- `metadata.input_entity`: The entity string to resolve
- `metadata.expected_ric`: The expected RIC output
- `metadata.match_type`: Type of matching (exact, fuzzy, alias, etc.)

### Subcategories

| Subcategory | Count | Description |
|-------------|-------|-------------|
| `exact_match` | 800 | Exact name/ticker matches from database |
| `ticker_lookup` | 500 | Ticker symbol resolution (incl. short: GE, T) |
| `alias_match` | 600 | Trade names, former names (Google→Alphabet) |
| `suffix_variations` | 400 | Legal suffix handling (Inc, AG, GmbH, N.V.) |
| `fuzzy_misspellings` | 500 | Typo tolerance (Mircosoft, Aple, Googgle) |
| `abbreviations` | 300 | Common abbreviations (IBM, J&J, P&G) |
| `international` | 600 | Non-US companies by region |
| `ambiguous` | 400 | Same name, different entities |
| `ticker_collisions` | 200 | Same ticker, multiple exchanges |
| `edge_case_names` | 300 | Numbers (3M), hyphens (Coca-Cola), special chars |
| `negative_cases` | 400 | Should NOT resolve (common words, fictional) |

### Evaluation Logic

Test passes if:
- For positive cases: `resolver.resolve_single(metadata.input_entity).identifier == metadata.expected_ric`
- For negative cases: `resolver.resolve_single(metadata.input_entity)` returns `null`

### Expected Baseline Results (Current Resolver)

The current resolver has known weaknesses. Expected baseline accuracy:

| Subcategory | Expected | Reason |
|-------------|----------|--------|
| exact_match | ~15% | Only 109 static mappings |
| ticker_lookup | ~10% | Short ticker regex broken |
| alias_match | ~10% | Limited alias coverage |
| suffix_variations | ~5% | Missing SE, AG, GmbH |
| fuzzy_misspellings | ~5% | No DB fuzzy matching |
| abbreviations | ~20% | Some hardcoded |
| international | ~2% | Minimal coverage |
| ambiguous | ~0% | Non-deterministic |
| ticker_collisions | ~0% | No exchange context |
| edge_case_names | ~5% | Pattern issues |
| negative_cases | ~40% | Over-matches common words |

### Generation

```bash
# Generate entity resolution fixtures (requires database)
export DATABASE_URL="postgresql://nl2api:nl2api@localhost:5432/nl2api"
python scripts/gen-test-cases.py --category entity_resolution

# Or use the generator directly
python -m scripts.generators.entity_resolution_generator \
  --output tests/fixtures/lseg/generated/entity_resolution/entity_resolution.json
```

### Data Sources

- **GLEIF Golden Copy**: 2,950,189 legal entities with LEIs
- **SEC EDGAR**: 8,036 US public companies with tickers and RICs

---

## Other Test Case Categories

### lookups

Single and multi-field data lookups.

**Subcategories:**
- `single_field`: Single metric queries (price, PE, market cap)
- `multi_field`: Multiple metrics in one query
- `category_lookup`: Full financial statement queries

### temporal

Time series and date-based queries.

**Subcategories:**
- `historical_price`: Historical price data
- `date_range`: Specific date ranges
- `relative_dates`: "last week", "past month"

### comparisons

Multi-ticker comparison queries.

**Subcategories:**
- `two_stock`: Compare two companies
- `multi_stock`: Compare 3+ companies
- `sector_comparison`: Sector-level comparisons

### screening

Stock screening and filtering queries.

**Subcategories:**
- `top_n`: Top N by metric
- `filter`: Filter by criteria
- `index_constituents`: Index membership queries

### complex

Multi-step and compound queries.

**Subcategories:**
- `multi_step`: Queries requiring multiple API calls
- `conditional`: Conditional logic queries

---

## Adding New Test Case Categories

When adding a new fixture generator, ensure:

1. [ ] Generator extends `BaseGenerator` in `scripts/generators/`
2. [ ] Output includes `_meta` block with `TestCaseSetConfig` fields
3. [ ] Generator registered in `scripts/generators/__init__.py`
4. [ ] Generator added to `scripts/gen-test-cases.py`
5. [ ] Category added to `FixtureLoader.CATEGORIES` in `tests/unit/nl2api/fixture_loader.py`
6. [ ] Coverage thresholds added to `CoverageRegistry.REQUIRED_COVERAGE`
7. [ ] Test file created in `tests/unit/nl2api/test_{category}_fixtures.py`
8. [ ] **Documentation updated in this file (`docs/evaluation-data.md`)**

---

## Synthetic Data Generation

### ⚠️ IMPORTANT: All Test Data is Synthetically Generated

**All `expected_response` and `expected_nl_response` values in test fixtures are synthetically generated by Claude 3.5 Haiku, NOT from real API calls.**

This means:
- Stock prices, financial metrics, and data values are **realistic but fictional**
- The data is useful for testing NL generation quality, **not API accuracy**
- Never use this data for financial decisions

### What Gets Generated

| Field | Source | Purpose |
|-------|--------|---------|
| `expected_response` | LLM-generated | Synthetic API response data (JSON) |
| `expected_nl_response` | LLM-generated | Natural language summary of the response |
| `expected_tool_calls` | Rule-based generators | Deterministic from query patterns |
| `nl_query` | Rule-based generators | Deterministic from templates |

### Generation Process

The `expected_response` and `expected_nl_response` fields are generated using:

```bash
# Preview prompts and cost estimate
python scripts/generate_eval_data.py --all --dry-run

# Generate for all categories (~$4-5 for ~18K test cases)
python scripts/generate_eval_data.py --all

# Generate for specific category
python scripts/generate_eval_data.py --category lookups --limit 100
```

**Generation Model:** Claude 3.5 Haiku (`claude-3-5-haiku-20241022`)
**Concurrency:** 5 concurrent requests with retry on rate limits
**Cost:** ~$0.00024 per test case

### Example Generated Data

**Input (from rule-based generator):**
```json
{
  "nl_query": "What is Apple's stock price?",
  "expected_tool_calls": [{"tool_name": "get_data", "arguments": {"tickers": "AAPL.O", "fields": ["P"]}}]
}
```

**Output (LLM-generated):**
```json
{
  "expected_response": {"AAPL.O": {"P": 185.42, "currency": "USD", "timestamp": "2024-02-16"}},
  "expected_nl_response": "Apple's stock price is currently $185.42."
}
```

### How the Data is Used

1. **Semantics Evaluation (Stage 4)**: Compares system-generated NL against `expected_nl_response`
2. **NL Generation Testing**: Tests if system produces semantically equivalent responses
3. **Quality Assurance**: Ensures NL output is coherent and informative

**NOT used for:**
- Verifying API correctness (use live API tests for that)
- Training models (data is for evaluation only)
- Financial analysis (values are fictional)

---

## Synthetic Data Caveats

All evaluation data is synthetic. When using or documenting:

- `expected_nl_response` values are **LLM-generated** (Claude 3.5 Haiku)
- `expected_response` values are **LLM-generated**, not from live API calls
- Stock prices, financial metrics are **realistic but fictional**
- API specifications were **reverse-engineered** from public documentation
- Ticker/company data is **point-in-time** and may become stale
- Entity data from GLEIF/SEC EDGAR may have corporate actions (mergers, delistings)

**Last generation run:** See `tests/fixtures/lseg/generated/*/` file timestamps

---

## File Locations

| File | Purpose |
|------|---------|
| `tests/fixtures/lseg/generated/` | Generated test fixtures |
| `scripts/generators/` | Test case generators |
| `scripts/gen-test-cases.py` | Main generation orchestrator |
| `tests/unit/nl2api/fixture_loader.py` | Fixture loading utility |
| `tests/unit/nl2api/test_fixture_coverage.py` | Coverage tracking |
