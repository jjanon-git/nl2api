# TestCase Patterns for Evaluation Packs

This document describes the recommended patterns for creating test cases in the evalkit framework.

## Quick Reference

| Pack Type | Use This | Example |
|-----------|----------|---------|
| **New packs** (RAG, code-gen, etc.) | Generic `input`/`expected` fields | See [Generic Pattern](#generic-pattern-recommended) |
| **NL2API pack** | `NL2APITestCase` with NL2API fields | See [NL2API Pattern](#nl2api-pattern-backwards-compatible) |

---

## Generic Pattern (Recommended)

**For new evaluation packs**, use the generic `input` and `expected` fields:

```python
from src.evalkit.contracts import TestCase

# RAG evaluation test case
test_case = TestCase(
    id="rag-001",
    input={
        "query": "What was Apple's revenue in Q4 2024?",
        "context_sources": ["10-K", "earnings_call"],
    },
    expected={
        "answer": "Apple reported $119.6B in revenue for Q4 2024.",
        "citations": ["10-K page 42"],
        "faithfulness_score": 0.95,
    },
)

# Code generation test case
test_case = TestCase(
    id="codegen-001",
    input={
        "prompt": "Write a function to reverse a string",
        "language": "python",
    },
    expected={
        "code": "def reverse_string(s): return s[::-1]",
        "test_cases_pass": True,
    },
)
```

### Benefits of Generic Pattern

1. **Pack-agnostic** - Same TestCase model works for any evaluation pack
2. **Flexible schema** - `input` and `expected` are dicts, so pack defines its own schema
3. **Future-proof** - No need to modify core contracts when adding new packs

---

## NL2API Pattern (Backwards Compatible)

**For NL2API evaluation**, use `NL2APITestCase` with the NL2API-specific fields:

```python
from src.evalkit.contracts import NL2APITestCase, ToolCall

# NL2API test case (explicit type)
test_case = NL2APITestCase(
    id="nl2api-001",
    nl_query="What is Apple's PE ratio?",
    expected_tool_calls=(
        ToolCall(tool_name="get_fundamentals", arguments={"ticker": "AAPL", "field": "PE"}),
    ),
    expected_nl_response="Apple's PE ratio is 28.5.",
)
```

> **Note:** `NL2APITestCase` is an alias for `TestCase`. The NL2API-specific fields (`nl_query`, `expected_tool_calls`, etc.) exist on the base TestCase for backwards compatibility. Using `NL2APITestCase` explicitly signals intent.

### NL2API Fields (Deprecated for New Packs)

These fields are marked as deprecated for new packs:

| Field | Deprecated | Use Instead |
|-------|------------|-------------|
| `nl_query` | Yes | `input["nl_query"]` |
| `expected_tool_calls` | Yes | `expected["tool_calls"]` |
| `expected_response` | Yes | `expected["response"]` |
| `expected_nl_response` | Yes | `expected["nl_response"]` |

---

## Converting Between Patterns

### NL2API → Generic

Use `to_generic()` to convert NL2API test cases to the generic format:

```python
nl2api_tc = NL2APITestCase(
    id="nl2api-001",
    nl_query="Get Apple price",
    expected_tool_calls=(ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
)

generic_tc = nl2api_tc.to_generic()
# generic_tc.input == {"nl_query": "Get Apple price"}
# generic_tc.expected == {"tool_calls": [{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}]}
```

### Generic → NL2API

Use `from_generic()` to create NL2API test cases from generic dicts:

```python
tc = NL2APITestCase.from_generic(
    id="nl2api-001",
    input={"nl_query": "Get Apple price"},
    expected={"tool_calls": [{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}]},
)

# tc.nl_query == "Get Apple price"
# tc.expected_tool_calls == (ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),)
```

---

## Pack Implementation Guidelines

When creating a new evaluation pack:

1. **Define your input/expected schema** in documentation
2. **Use generic fields** - Read from `test_case.input` and `test_case.expected`
3. **Validate at pack level** - Check required fields exist in the dicts

```python
class MyPack:
    @property
    def name(self) -> str:
        return "my_pack"

    def validate_test_case(self, test_case: TestCase) -> list[str]:
        """Validate test case has required fields for this pack."""
        errors = []
        if "query" not in test_case.input:
            errors.append("input.query is required")
        if "expected_output" not in test_case.expected:
            errors.append("expected.expected_output is required")
        return errors
```

---

## Migration Guide

If you have existing code using NL2API fields directly:

1. **No immediate action required** - The fields still work
2. **For new packs** - Use generic pattern from the start
3. **Future migration** - Phase 2 of decoupling plan may convert NL2API to use generic fields internally

See `docs/plans/eval-09-model-decoupling.md` for the full decoupling roadmap.
