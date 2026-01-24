# Evaluation Data Contract Improvement Plan

**Date:** 2026-01-21
**Status:** ✅ Complete

> **Completed:** 2026-01-21
> **Implementation:** `TestCaseSetConfig` in `src/contracts/core.py`, `_meta` blocks in fixture files, validation in `fixture_loader.py`. `expected_response` field remains null (deferred until execution stage).

---

## Decisions Made

| Question | Decision | Rationale |
|----------|----------|-----------|
| Rename `expected_raw_data`? | ✅ Yes → `expected_response` | Clearer semantics |
| Populate `expected_response`? | ⏸️ Leave null | Until execution stage is implemented (tracked as future work) |
| Haiku model version? | ✅ Claude 3.5 Haiku | ~$3 more but better quality; one-time cost |
| Git strategy for fixtures? | ✅ Commit to git | Reproducibility, review visibility, no CI API calls |
| Field nullability? | ✅ Test case set metadata | Each dataset declares which fields are required |

### Tracking: Future Work

- [ ] **DEFERRED**: Populate `expected_response` with realistic data when execution stage is implemented

---

## Test Case Set Configuration

### Problem

Different evaluation capabilities have different field requirements:

| Capability | `expected_nl_response` | `expected_response` |
|------------|------------------------|---------------------|
| NL2API (full) | Required | Optional (until execution stage) |
| Entity extraction | Not applicable | Not applicable |
| Tool call generation | Not applicable | Optional |
| Response formatting | Required | Required |

Making fields globally nullable loses validation - we can't detect incomplete data.

### Solution: Test Case Set Metadata

Each test case set (fixture file) declares which fields are required via a `_meta` block.

### Interface Definition

**Add to `CONTRACTS.py`:**

```python
class TestCaseSetConfig(BaseModel):
    """
    Configuration for a test case set defining required fields and metadata.

    Embedded in fixture files as the '_meta' key. The fixture loader validates
    each test case against this configuration.
    """

    model_config = ConfigDict(frozen=True)

    # Identity
    name: str = Field(
        description="Human-readable name for this test case set",
        examples=["lookups", "entity_extraction_us_equities"],
    )
    capability: str = Field(
        description="The capability this set evaluates",
        examples=["nl2api", "entity_extraction", "tool_generation"],
    )
    description: str | None = Field(
        default=None,
        description="Optional description of what this test set covers",
    )

    # Field requirements - which fields must be present
    requires_nl_response: bool = Field(
        default=True,
        description="Whether expected_nl_response is required for test cases in this set",
    )
    requires_expected_response: bool = Field(
        default=False,
        description="Whether expected_response is required for test cases in this set",
    )

    # Generation metadata
    schema_version: str = Field(
        default="1.0",
        description="Schema version for forward compatibility",
    )
    generated_at: datetime | None = Field(
        default=None,
        description="When this fixture set was generated",
    )
    generator: str | None = Field(
        default=None,
        description="Script/tool that generated this set",
        examples=["scripts/generate_test_cases.py"],
    )
```

### Fixture File Format

```json
{
  "_meta": {
    "name": "entity_extraction_us_equities",
    "capability": "entity_extraction",
    "description": "Entity extraction tests for US equity tickers",
    "requires_nl_response": false,
    "requires_expected_response": false,
    "schema_version": "1.0",
    "generated_at": "2026-01-21T10:00:00Z",
    "generator": "scripts/generate_entity_tests.py"
  },
  "test_cases": [
    {
      "id": "entity_001",
      "nl_query": "What is Apple's stock price?",
      "expected_tool_calls": [...],
      "expected_response": null,
      "expected_nl_response": null
    }
  ]
}
```

**NL2API example (requires NL response):**

```json
{
  "_meta": {
    "name": "lookups",
    "capability": "nl2api",
    "requires_nl_response": true,
    "requires_expected_response": false,
    "schema_version": "1.0"
  },
  "test_cases": [
    {
      "id": "lookups_001",
      "nl_query": "What is Apple's stock price?",
      "expected_tool_calls": [...],
      "expected_nl_response": "Apple's stock price is $246.02."
    }
  ]
}
```

### Fixture Loader Validation

**Update `tests/unit/nl2api/fixture_loader.py`:**

```python
class FixtureLoader:
    def load(self, path: Path) -> tuple[TestCaseSetConfig, list[TestCase]]:
        """Load fixture file and validate against its metadata."""
        data = json.loads(path.read_text())

        # Parse metadata (use defaults if missing for backward compat)
        meta_data = data.get("_meta", {})
        config = TestCaseSetConfig(
            name=meta_data.get("name", path.stem),
            capability=meta_data.get("capability", "nl2api"),
            **{k: v for k, v in meta_data.items() if k in TestCaseSetConfig.model_fields}
        )

        # Load and validate test cases
        test_cases = []
        for tc_data in data.get("test_cases", data):  # Support old format
            tc = TestCase(**tc_data)
            self._validate_against_config(tc, config)
            test_cases.append(tc)

        return config, test_cases

    def _validate_against_config(self, tc: TestCase, config: TestCaseSetConfig) -> None:
        """Validate test case has required fields per set config."""
        if config.requires_nl_response and not tc.expected_nl_response:
            raise ValueError(
                f"Test case {tc.id} missing expected_nl_response "
                f"(required by {config.name})"
            )
        if config.requires_expected_response and not tc.expected_response:
            raise ValueError(
                f"Test case {tc.id} missing expected_response "
                f"(required by {config.name})"
            )
```

### Migration Path

1. **Phase 1**: Add `TestCaseSetConfig` to `CONTRACTS.py`
2. **Phase 2**: Update fixture loader to parse `_meta` (with backward compat)
3. **Phase 3**: Add `_meta` blocks to existing fixtures during regeneration
4. **Phase 4**: Enable validation (initially as warnings, then errors)

---

## Executive Summary

This plan addresses four key issues with the evaluation data infrastructure:

1. **Response field expansion** - Add `expected_response` (raw data) alongside `expected_nl_response` (natural language)
2. **Fixture generation process** - Validate the pattern and ensure updateability
3. **Documentation gaps** - Document synthetic data generation and API reverse engineering
4. **NL response population** - Use Haiku to generate `expected_nl_response` values

---

## Issue 1: Expected Response Fields

### Current State

The `TestCase` contract in `CONTRACTS.py` has:
```python
expected_raw_data: dict[str, Any] | None  # Mock return data from tool execution
expected_nl_response: str                  # Expected natural language response
```

**Problem:** The semantics are unclear, and `expected_raw_data` isn't being used consistently.

### Proposed Change

Rename and clarify the fields to distinguish:

| Field | Purpose | Example |
|-------|---------|---------|
| `expected_response` | The **data value** returned by the API | `{"AAPL.O": {"P": 246.02}}` |
| `expected_nl_response` | A **human-readable sentence** summarizing the result | `"Apple's stock price is $246.02."` |

### Implementation

1. **Update CONTRACTS.py:**
```python
class TestCase(BaseModel):
    # ... existing fields ...

    expected_response: dict[str, Any] | None = Field(
        default=None,
        description="Expected structured data response from API execution",
        examples=[{"AAPL.O": {"P": 246.02, "MV": 3850000000000}}],
    )
    expected_nl_response: str | None = Field(
        default=None,  # Make optional during migration
        description="Expected natural language summary of the response",
        examples=["Apple's stock price is $246.02 with a market cap of $3.85 trillion."],
    )
```

2. **Update database schema** (`migrations/008_expected_response.sql`):
```sql
-- Rename for clarity
ALTER TABLE test_cases RENAME COLUMN expected_raw_data TO expected_response;

-- Ensure expected_nl_response allows NULL during migration
ALTER TABLE test_cases ALTER COLUMN expected_nl_response DROP NOT NULL;
```

3. **Update generators** to produce `expected_response` where deterministic (e.g., static field mappings)

4. **Backward compatibility**: The fixture loader already handles missing fields gracefully

### Decisions (Confirmed)

- [x] Rename `expected_raw_data` → `expected_response` ✅
- [x] Make `expected_nl_response` optional (nullable) during migration ✅
- [x] Leave `expected_response` as null until execution stage is implemented ✅

---

## Issue 2: Fixture Generation Pattern Assessment

### Current Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Data Sources                                  │
├─────────────────────────────────────────────────────────────────┤
│  data/field_codes/     data/tickers/       data/               │
│  ├─ datastream.json    ├─ us_mega_caps     ├─ nl_templates     │
│  ├─ fundamentals_wc    ├─ us_by_sector     ├─ temporal_patterns│
│  ├─ fundamentals_tr    ├─ international    ├─ comparison_pairs │
│  ├─ estimates_tr       └─ indices          └─ screening_criteria│
│  └─ officers_tr                                                 │
├─────────────────────────────────────────────────────────────────┤
│                    Generators                                    │
├─────────────────────────────────────────────────────────────────┤
│  scripts/generators/                                             │
│  ├─ base_generator.py      (shared utilities, TestCase dataclass)│
│  ├─ lookup_generator.py    (~3,745 cases)                       │
│  ├─ temporal_generator.py  (~2,727 cases)                       │
│  ├─ comparison_generator.py(~3,658 cases)                       │
│  ├─ screening_generator.py (~265 cases)                         │
│  ├─ error_generator.py     (~215 cases)                         │
│  └─ complex_generator.py   (~2,277 cases)                       │
├─────────────────────────────────────────────────────────────────┤
│                    Output                                        │
├─────────────────────────────────────────────────────────────────┤
│  tests/fixtures/lseg/generated/  (12,887 total test cases)      │
│  ├─ lookups/lookups.json                                        │
│  ├─ temporal/temporal.json                                      │
│  ├─ comparisons/comparisons.json                                │
│  ├─ screening/screening.json                                    │
│  ├─ errors/errors.json                                          │
│  └─ complex/complex.json                                        │
└─────────────────────────────────────────────────────────────────┘
```

### Assessment: Is This a Good Pattern?

**Strengths:**
- ✅ Modular generators (easy to add new categories)
- ✅ Combinatorial coverage (300 fields × 200 tickers × 13 time patterns)
- ✅ Deterministic IDs (MD5 hash allows deduplication)
- ✅ Separation of data (field codes, tickers) from logic (generators)
- ✅ Single command regeneration (`python scripts/generate_test_cases.py`)

**Weaknesses:**
- ⚠️ Generator TestCase differs from CONTRACTS.py TestCase (schema drift)
- ⚠️ No versioning of generated fixtures
- ⚠️ Missing `expected_nl_response` field entirely
- ⚠️ Uses `function` instead of `tool_name` in tool calls

**Verdict:** The pattern is sound but needs alignment with the contract.

### Update Process Verification

**Current process exists:**
```bash
# Regenerate all fixtures
python scripts/generate_test_cases.py --all

# Regenerate specific category
python scripts/generate_test_cases.py --category lookups

# Load into database
python scripts/load_test_cases.py
```

**Recommended enhancements:**
1. Add `--version` flag to tag generated fixtures with schema version
2. Add `--validate` flag to check generated output against CONTRACTS.py
3. Add CI job to detect fixture staleness when API docs change

### Action Items

- [x] Confirm regeneration process exists (`generate_test_cases.py`)
- [ ] Add schema version tracking to generated fixtures
- [ ] Add validation step in generation pipeline
- [ ] Align generator TestCase dataclass with CONTRACTS.py

---

## Issue 3: Documentation Requirements

### Current Documentation Gaps

| Topic | Current State | Required |
|-------|---------------|----------|
| Synthetic data generation | None | Full documentation |
| API reverse engineering | Implicit in docs/api-reference/ | Explicit documentation |
| Field code sources | Scattered in data/field_codes/ | Consolidated reference |
| Fixture update process | In scripts only | User-facing docs |

### Proposed Documentation

Create `docs/evaluation-data.md`:

```markdown
# Evaluation Data Generation

## Overview

The NL2API evaluation suite uses **synthetically generated test cases** to ensure
comprehensive coverage across API domains, field codes, and query patterns.

## Data Sources

### API Reference Documentation
The API specifications in `docs/api-reference/` were created by:
1. Reviewing LSEG/Refinitiv public API documentation
2. Extracting field codes, parameters, and response formats
3. Documenting natural language mappings

**Important:** These specifications are derived from public documentation and may
not reflect all API capabilities or recent changes. They should be validated
against actual API behavior periodically.

### Field Code Catalogs
Located in `data/field_codes/`:
- `datastream_fields.json` - Price, volume, and fundamental fields
- `fundamentals_wc.json` - Worldscope codes (WC01001, etc.)
- `fundamentals_tr.json` - Refinitiv TR codes
- `estimates_tr.json` - I/B/E/S estimate fields
- `officers_tr.json` - Executive and board member fields

### Ticker Universe
Located in `data/tickers/`:
- 200+ US mega-cap stocks
- International tickers by region
- Index constituents
- Edge cases (ADRs, delisted, special characters)

## Generation Process

### Running the Generator
```bash
# Generate all categories (12,887 test cases)
python scripts/generate_test_cases.py --all

# Generate specific category
python scripts/generate_test_cases.py --category lookups

# Validate generated output
python scripts/generate_test_cases.py --validate
```

### Generator Architecture
Six specialized generators produce test cases:
1. **LookupGenerator** - Single/multi-field data retrieval
2. **TemporalGenerator** - Time series queries
3. **ComparisonGenerator** - Multi-ticker comparisons
4. **ScreeningGenerator** - Filtering and ranking
5. **ErrorGenerator** - Invalid input handling
6. **ComplexGenerator** - Multi-step workflows

### Fixture Schema
Generated fixtures follow this structure:
```json
{
  "id": "lookups_abc123",
  "nl_query": "What is Apple's stock price?",
  "expected_tool_calls": [...],
  "expected_response": {"AAPL.O": {"P": 246.02}},
  "expected_nl_response": "Apple's stock price is $246.02.",
  "complexity": 1,
  "category": "lookups",
  "tags": ["price", "datastream"]
}
```

## Synthetic Data Caveats

⚠️ **Important Limitations:**

1. **Expected responses are synthetic** - The `expected_response` values are
   generated based on field code specifications, not live API calls.

2. **NL responses are LLM-generated** - The `expected_nl_response` values were
   generated using Claude Haiku and may contain inaccuracies.

3. **API specifications may drift** - The underlying API documentation was
   reverse-engineered from public sources and may not reflect current behavior.

4. **Ticker data is point-in-time** - Company names, tickers, and RICs may
   change due to corporate actions.

## Updating Fixtures

When API specifications change:

1. Update the relevant `docs/api-reference/*.md` file
2. Update `data/field_codes/*.json` if field codes changed
3. Regenerate affected fixtures: `python scripts/generate_test_cases.py --category <name>`
4. Run validation: `python scripts/generate_test_cases.py --validate`
5. Re-run accuracy tests: `pytest tests/accuracy/ -m tier1`
```

---

## Issue 4: Populating expected_nl_response with Claude 3.5 Haiku

### Approach

Use **Claude 3.5 Haiku** (`claude-3-5-haiku-20241022`) to generate natural language responses for all 12,887 test cases.

**Why 3.5 Haiku over 3 Haiku:**
- Better reasoning quality for natural language generation
- ~$3 additional cost is negligible for one-time generation
- Evaluation data quality matters more than generation cost

### Implementation Plan

1. **Create generation script** (`scripts/generate_nl_responses.py`):
```python
"""
Generate expected_nl_response values using Claude Haiku.

This script:
1. Loads existing test cases from fixtures
2. Generates natural language responses using Haiku
3. Updates fixtures with the generated responses
4. Tracks generation metadata (model, timestamp, cost)
"""

SYSTEM_PROMPT = """
You are generating expected natural language responses for an NL2API evaluation system.

Given:
- A natural language query
- The expected API tool calls
- The expected data response (if available)

Generate a concise, natural language response that a user would expect.

Rules:
- Be factual and specific
- Include relevant numbers/values from the response
- Keep responses under 100 words
- Use natural phrasing (not robotic)
"""
```

2. **Batch processing with rate limiting:**
   - Process in batches of 100
   - Checkpoint progress to allow resume
   - Estimate: ~$5 for 12,887 cases at Claude 3.5 Haiku rates

3. **Update fixtures with metadata:**
```json
{
  "expected_nl_response": "Apple's stock price is $246.02.",
  "nl_response_metadata": {
    "generated_by": "claude-3-5-haiku-20241022",
    "generated_at": "2026-01-21T10:30:00Z",
    "synthetic": true
  }
}
```

4. **Store in database:**
   - Add column: `nl_response_generated_by`
   - Add column: `nl_response_generated_at`
   - Flag: `nl_response_synthetic: boolean`

### Cost Estimate (Claude 3.5 Haiku)

| Item | Count | Tokens/Item | Rate | Cost |
|------|-------|-------------|------|------|
| Input (query + context) | 12,887 | ~200 | $0.80/1M | ~$2.06 |
| Output (response) | 12,887 | ~50 | $4.00/1M | ~$2.58 |
| **Total** | | | | **~$4.64** |

*Note: Actual cost may vary based on prompt complexity and response length.*

### Caveats to Document

Add to generated fixtures and documentation:

```
⚠️ SYNTHETIC DATA NOTICE

The `expected_nl_response` values in this dataset were generated using
Claude 3.5 Haiku (claude-3-5-haiku-20241022) on 2026-01-21.

These responses are synthetic approximations and may:
- Contain factual inaccuracies
- Not match actual API response formatting
- Require updates when API behavior changes

For production evaluation, consider validating against live API responses.
```

---

## Implementation Phases

### Phase 1: Contract & Schema Updates (1-2 days)
- [ ] Update CONTRACTS.py with clarified field names
- [ ] Create database migration
- [ ] Update fixture loader for backward compatibility

### Phase 2: Generator Alignment (1 day)
- [ ] Align generator TestCase with CONTRACTS.py
- [ ] Add `expected_response` generation where deterministic
- [ ] Add schema validation to generation pipeline

### Phase 3: Documentation (0.5 day)
- [ ] Create `docs/evaluation-data.md`
- [ ] Add caveats to existing fixture metadata
- [ ] Update CLAUDE.md with eval data section

### Phase 4: NL Response Generation (1 day)
- [ ] Create `scripts/generate_nl_responses.py`
- [ ] Run Haiku generation with checkpointing
- [ ] Update fixtures with responses and metadata
- [ ] Load updated fixtures to database

### Phase 5: Validation & Testing (0.5 day)
- [ ] Run full test suite
- [ ] Verify fixture loading
- [ ] Run tier1 accuracy tests

---

## Git Strategy for Generated Fixtures

**Decision: Commit generated fixtures to version control.**

### Rationale

| Approach | Pros | Cons |
|----------|------|------|
| **Commit to git** ✅ | Reproducible, reviewable, no CI API costs | Larger repo, must remember to regenerate |
| Generate in CI | Always fresh, smaller repo | Needs API credentials, slower CI, non-deterministic |

### Implementation

1. **Mark as generated** in `.gitattributes`:
   ```
   tests/fixtures/lseg/generated/** linguist-generated=true
   ```

2. **Add header to generated JSON files:**
   ```json
   {
     "_meta": {
       "generated": true,
       "generator": "scripts/generate_test_cases.py",
       "generated_at": "2026-01-21T10:00:00Z",
       "schema_version": "1.0"
     },
     "test_cases": [...]
   }
   ```

3. **Document regeneration process** in `docs/evaluation-data.md`

4. **Add CI check** to warn if fixtures are stale (optional future enhancement)

---

## Open Questions (Resolved)

| Question | Resolution |
|----------|------------|
| `expected_response` values | Leave null until execution stage is implemented |
| Haiku model version | Use Claude 3.5 Haiku for better quality |
| Regeneration frequency | On API doc changes; document process |
| Version control | Commit generated fixtures to git |

---

## Appendix: Files to Modify

| File | Changes |
|------|---------|
| `CONTRACTS.py` | Rename field, make nl_response optional, add `TestCaseSetConfig` |
| `migrations/008_*.sql` | Schema migration |
| `scripts/generators/base_generator.py` | Align TestCase dataclass, add `_meta` generation |
| `scripts/load_test_cases.py` | Handle new fields |
| `scripts/generate_nl_responses.py` | New script |
| `tests/unit/nl2api/fixture_loader.py` | Add `_meta` parsing and validation |
| `docs/evaluation-data.md` | New documentation |
| `CLAUDE.md` | Add eval data section ✅ |
| `tests/fixtures/lseg/generated/*.json` | Add `_meta` blocks during regeneration |
