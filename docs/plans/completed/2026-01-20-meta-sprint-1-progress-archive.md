# Sprint 1: Minimal End-to-End Implementation

> **Goal:** `python -m src.evaluation.cli.main run test.json` → pass/fail in terminal
> **Started:** 2026-01-19
> **Status:** COMPLETE
>
> *Note: This is a historical record of Sprint 1. Files were later reorganized from `src/core` and `src/cli` into `src/evaluation/`. See CLAUDE.md for current commands.*

---

## Deliverables

```bash
# Load test case from file, mock LLM response, evaluate
python -m src.evaluation.cli.main run tests/fixtures/search_products.json

# Output:
# Test: search_query
# Stage 1 (Syntax): PASS ✓
# Stage 2 (Logic):  PASS ✓ (score: 1.0)
# Overall: PASS
```

---

## Tasks

### 1. Project Scaffolding
- [x] Create directory structure
- [x] Create `pyproject.toml` with dependencies
- [x] Create `src/` package structure

### 2. Core Components
- [x] `src/evaluation/core/ast_comparator.py` - AST-based tool call comparison
- [x] `src/evaluation/core/evaluators.py` - SyntaxEvaluator & LogicEvaluator implementations
- [x] `src/nl2api/config.py` - Configuration loading

### 3. CLI
- [x] `src/evaluation/cli/main.py` - Typer CLI entry point
- [x] `src/evaluation/cli/commands/run.py` - Run command implementation

### 4. Test Fixtures
- [x] `tests/fixtures/` - Sample test cases in JSON format
- [x] Basic happy path test case
- [x] Edge case test cases

### 5. Unit Tests
- [x] Tests for AST comparator
- [x] Tests for syntax evaluator
- [x] Tests for logic evaluator
- [x] Tests for CLI

### 6. Validation
- [x] All unit tests pass (44 tests in core evaluation)
- [x] CLI runs end-to-end with sample fixture
- [x] Argument permutation handled correctly

---

## Exit Criteria

- [x] Can evaluate a test case locally
- [x] AST comparison handles argument permutation
- [x] Unit tests pass (44/44)

---

## Implementation Log

### 2026-01-19 - Session Start

**Created:**
- Directory structure scaffolded
- `pyproject.toml` with Pydantic, Typer dependencies
- `src/evaluation/core/ast_comparator.py` - Deep comparison with type coercion
- `src/evaluation/core/evaluators.py` - Syntax & Logic evaluators
- `src/evaluation/cli/main.py` - Typer CLI
- `tests/fixtures/` - Sample test cases
- Unit tests for all components

### 2026-01-19 - Sprint Complete

**Validated:**
- All 44 unit tests passing
- CLI end-to-end working with all test fixtures
- Order-independent matching working (multi_tool_call.json)
- Type coercion working (type_coercion.json - string "456" matches int 456)
- Syntax errors properly halt pipeline (syntax_error.json)
- Logic mismatches properly detected with detailed diffs (logic_mismatch.json)

**Test Results:**
```
tests/unit/test_ast_comparator.py - 19 tests PASSED
tests/unit/test_evaluators.py - 25 tests PASSED
Total: 44 tests in 0.25s
```

**CLI Demo:**
```bash
.venv/bin/python -m src.evaluation.cli.main run tests/fixtures/search_products.json
# Output: Overall: PASS (score: 1.00)

.venv/bin/python -m src.evaluation.cli.main run tests/fixtures/logic_mismatch.json -v
# Output: Overall: FAIL (score: 0.25) with detailed argument diff
```

---

## Files Created (Original Locations)

| File | Purpose | Current Location |
|------|---------|------------------|
| `pyproject.toml` | Project configuration & dependencies | `pyproject.toml` |
| `src/__init__.py` | Package root | `src/__init__.py` |
| `src/core/__init__.py` | Core module | `src/evaluation/core/__init__.py` |
| `src/core/ast_comparator.py` | Tool call comparison logic | `src/evaluation/core/ast_comparator.py` |
| `src/core/evaluators.py` | Stage 1 & 2 evaluators | `src/evaluation/core/evaluators.py` |
| `src/core/config.py` | Configuration management | `src/nl2api/config.py` |
| `src/cli/__init__.py` | CLI module | `src/evaluation/cli/__init__.py` |
| `src/cli/main.py` | CLI entry point | `src/evaluation/cli/main.py` |
| `src/cli/commands/__init__.py` | Commands module | `src/evaluation/cli/commands/__init__.py` |
| `src/cli/commands/run.py` | Run command | `src/evaluation/cli/commands/run.py` |
| `tests/__init__.py` | Tests package | `tests/__init__.py` |
| `tests/unit/__init__.py` | Unit tests | `tests/unit/__init__.py` |
| `tests/unit/test_ast_comparator.py` | AST comparator tests | `tests/unit/test_ast_comparator.py` |
| `tests/unit/test_evaluators.py` | Evaluator tests | `tests/unit/test_evaluators.py` |
| `tests/fixtures/search_products.json` | Sample test case | `tests/fixtures/search_products.json` |

---

## Architecture Notes

### AST Comparator Design

The `ASTComparator` handles:
1. **Order-independent comparison** - Tool calls are compared as sets
2. **Type coercion** - `"5"` and `5` are considered equal
3. **Nested object comparison** - Deep equality with tolerance
4. **Argument permutation** - Same args in different order match

### Evaluator Pipeline (Sprint 1)

```
Raw Output → [Stage 1: Syntax] → Parsed ToolCalls → [Stage 2: Logic] → Scorecard
                  │                                       │
                  │ FAIL: Pipeline halts                  │ FAIL: Logged, continues
                  ▼                                       ▼
              Scorecard                              Scorecard
```

Stage 3 (Execution) and Stage 4 (Semantics) are deferred to Sprint 4.
