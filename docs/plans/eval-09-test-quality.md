# Test Quality Improvements Plan

**Status:** Not Started
**Priority:** P0 (Critical)
**Author:** Mostly Claude, with some minor assistance from Sid
**Created:** 2026-01-24

---

## Problem Statement

The test suite has structural issues that allow broken agents to pass. Current tests provide false confidence - agents that generate garbage tool calls pass because tests only verify superficial properties.

### Current State (Problems)

| Issue | Impact | Example |
|-------|--------|---------|
| Tests only check `can_handle()`, not `process()` | Agents that claim to handle queries but generate garbage pass | Agent returns wrong tool, tests pass |
| 121k fixtures but only ~100 sampled (0.08%) | Regressions go undetected | New bug affects 1000 fixtures, caught by 0 samples |
| Coverage thresholds at 10-40% | Absurdly low bar | Agent handles 15% of queries correctly, tests pass |
| Mock LLM returns static response | Can't test real behavior | LLM prompt changes, tests still pass |
| Tautological assertions | Always true | `len(result.tool_calls) >= 0` |
| No execution stage | Tool calls never validated | `{"field": "INVALID"}` passes |

### Files Affected

- `tests/unit/nl2api/test_fixture_coverage.py` - Coverage registry and thresholds
- `tests/unit/nl2api/test_datastream_fixtures.py` - Only tests `can_handle()`
- `tests/unit/nl2api/test_screening_fixtures.py` - Only tests `can_handle()`
- `tests/accuracy/core/evaluator.py` - Real LLM evaluation (separate from unit tests)

---

## Goals

1. **Tests catch broken agents** - If an agent generates invalid tool calls, tests fail
2. **Meaningful coverage** - Sample enough fixtures to catch regressions
3. **Realistic behavior** - Mock LLM validates prompt/response structure
4. **Semantic validation** - Tool call arguments are checked for validity

---

## Phases

### Phase 1: Add `process()` Tests (3 days)

Currently, fixture tests only verify `can_handle()` returns non-zero confidence. This doesn't test actual tool generation.

**Tasks:**
- [ ] Create `test_agent_process.py` for each agent
- [ ] For each fixture, call `agent.process()` and verify:
  - Tool call count matches expected
  - Tool names are valid (exist in agent's tool registry)
  - Required arguments are present
  - Argument types are correct
- [ ] Use mock LLM that returns fixture's expected tool calls
- [ ] Add to CI pipeline

**Test Structure:**
```python
@pytest.mark.parametrize("fixture", load_fixtures("datastream"))
async def test_process_generates_valid_tool_calls(fixture, mock_llm):
    agent = DatastreamAgent(llm=mock_llm)
    result = await agent.process(AgentContext(query=fixture.nl_query, ...))

    assert len(result.tool_calls) > 0, "Should generate at least one tool call"
    for tc in result.tool_calls:
        assert tc.tool_name in VALID_TOOL_NAMES
        validate_tool_arguments(tc.tool_name, tc.arguments)
```

### Phase 2: Increase Sampling Coverage (2 days)

Current: ~100 samples from 121k fixtures (0.08%)
Target: 1,200+ samples (1%+)

**Tasks:**
- [ ] Update `CoverageRegistry.REQUIRED_COVERAGE` thresholds
- [ ] Implement stratified sampling (by category/subcategory)
- [ ] Ensure each subcategory has minimum representation
- [ ] Add sampling seed for reproducibility

**Thresholds by Category:**
| Category | Current | Target | Rationale |
|----------|---------|--------|-----------|
| lookups/single_field | 30% | 70% | Core functionality |
| lookups/multi_field | 15% | 70% | Core functionality |
| temporal | 10% | 60% | Complex date handling |
| comparisons | 10% | 50% | Multi-entity queries |
| screening | 50% | 80% | High business value |
| complex | 10% | 40% | Edge cases |

### Phase 3: Smart Mock LLM (3 days)

Current mock returns static response regardless of prompt. This can't catch:
- Prompt template regressions
- Missing context injection
- Incorrect tool schemas

**Tasks:**
- [ ] Create `ValidatingMockLLM` class
- [ ] Verify prompt contains required sections:
  - System prompt with tool definitions
  - User query
  - Context (if RAG enabled)
- [ ] Verify tool call format matches schema
- [ ] Return fixture-appropriate responses (not static)

**Implementation:**
```python
class ValidatingMockLLM:
    def __init__(self, expected_fixtures: dict[str, ToolCall]):
        self.expected = expected_fixtures

    async def complete(self, messages: list[Message]) -> LLMResponse:
        # Validate prompt structure
        assert any("tools" in m.content for m in messages), "Missing tool definitions"

        # Extract query and return appropriate fixture response
        query = extract_user_query(messages)
        if query in self.expected:
            return make_tool_call_response(self.expected[query])
        raise ValueError(f"No fixture for query: {query}")
```

### Phase 4: Semantic Validation (2 days)

Validate that tool call arguments make semantic sense, not just structural correctness.

**Tasks:**
- [ ] Create argument validators per tool:
  - `get_data`: fields must be valid Datastream codes
  - `screen`: expression must parse correctly
  - `get_estimates`: metrics must be valid I/B/E/S codes
- [ ] Add field code validation against known registry
- [ ] Add date format validation
- [ ] Add entity/RIC format validation

**Validators:**
```python
FIELD_CODE_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]{1,20}$")
RIC_PATTERN = re.compile(r"^[A-Z0-9.]+$")

def validate_get_data_args(args: dict) -> list[str]:
    errors = []
    for field in args.get("fields", []):
        if not FIELD_CODE_PATTERN.match(field):
            errors.append(f"Invalid field code: {field}")
    for ric in args.get("instruments", []):
        if not RIC_PATTERN.match(ric):
            errors.append(f"Invalid RIC: {ric}")
    return errors
```

### Phase 5: Tautology Removal (1 day)

Find and fix assertions that are always true.

**Tasks:**
- [ ] Grep for tautological patterns:
  - `>= 0` on length/count
  - `is not None` after assignment
  - `in [...]` with full enum
- [ ] Replace with meaningful assertions:
  - `> 0` for "must have results"
  - Specific value checks
  - Range validations

**Example Fixes:**
```python
# BAD: Always true
assert len(result.tool_calls) >= 0

# GOOD: Actually validates
assert len(result.tool_calls) > 0, "Should generate at least one tool call"
assert len(result.tool_calls) == len(fixture.expected_tool_calls)
```

---

## Success Criteria

| Metric | Current | Target |
|--------|---------|--------|
| Fixture sampling rate | 0.08% | 1%+ |
| Per-category thresholds | 10-40% | 50-80% |
| `process()` test coverage | 0% | 100% of agents |
| Tautological assertions | ~20 | 0 |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Tests become slow with more samples | Parallel execution, tiered sampling |
| Mock LLM too strict | Start permissive, tighten gradually |
| Field code registry incomplete | Build from existing fixtures first |

---

## Dependencies

- None - can start immediately

---

## Effort Estimate

| Phase | Effort |
|-------|--------|
| Phase 1: `process()` tests | 3 days |
| Phase 2: Sampling coverage | 2 days |
| Phase 3: Smart mock LLM | 3 days |
| Phase 4: Semantic validation | 2 days |
| Phase 5: Tautology removal | 1 day |
| **Total** | **11 days** |
