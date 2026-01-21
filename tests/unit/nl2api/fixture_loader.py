"""Fixture loader utility for loading generated test cases."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator


@dataclass(frozen=True)
class GeneratedTestCase:
    """A test case loaded from the generated fixtures."""

    id: str
    nl_query: str
    expected_tool_calls: tuple[dict[str, Any], ...]
    complexity: int
    category: str
    subcategory: str
    tags: tuple[str, ...]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GeneratedTestCase:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            nl_query=data["nl_query"],
            expected_tool_calls=tuple(data.get("expected_tool_calls", [])),
            complexity=data.get("complexity", 1),
            category=data.get("category", ""),
            subcategory=data.get("subcategory", ""),
            tags=tuple(data.get("tags", [])),
            metadata=data.get("metadata", {}),
        )


class FixtureLoader:
    """Loads generated test fixtures from the filesystem."""

    FIXTURE_DIR = Path(__file__).parent.parent.parent / "fixtures" / "lseg" / "generated"

    CATEGORIES = [
        "lookups",
        "temporal",
        "comparisons",
        "complex",
        "screening",
        "errors",
    ]

    def __init__(self, fixture_dir: Path | None = None):
        """Initialize the fixture loader."""
        self.fixture_dir = fixture_dir or self.FIXTURE_DIR

    def load_category(self, category: str) -> list[GeneratedTestCase]:
        """Load all test cases from a category."""
        file_path = self.fixture_dir / category / f"{category}.json"

        if not file_path.exists():
            return []

        with open(file_path) as f:
            data = json.load(f)

        return [GeneratedTestCase.from_dict(tc) for tc in data.get("test_cases", [])]

    def load_all(self) -> dict[str, list[GeneratedTestCase]]:
        """Load all test cases from all categories."""
        result = {}
        for category in self.CATEGORIES:
            cases = self.load_category(category)
            if cases:
                result[category] = cases
        return result

    def iterate_all(self) -> Iterator[GeneratedTestCase]:
        """Iterate over all test cases from all categories."""
        for category in self.CATEGORIES:
            for case in self.load_category(category):
                yield case

    def load_by_tag(self, tag: str) -> list[GeneratedTestCase]:
        """Load test cases with a specific tag."""
        return [tc for tc in self.iterate_all() if tag in tc.tags]

    def load_by_subcategory(self, subcategory: str) -> list[GeneratedTestCase]:
        """Load test cases with a specific subcategory."""
        return [tc for tc in self.iterate_all() if tc.subcategory == subcategory]

    def get_summary(self) -> dict[str, int]:
        """Get a summary of test case counts by category."""
        return {cat: len(self.load_category(cat)) for cat in self.CATEGORIES}


def normalize_tool_call(tool_call: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize a tool call for comparison.

    Handles differences like:
    - datastream.get_data vs get_data
    - Different ticker formats
    - Optional fields
    """
    result = dict(tool_call)

    # Normalize function name
    func_name = result.get("function", result.get("tool_name", ""))
    if "." in func_name:
        func_name = func_name.split(".")[-1]
    result["function"] = func_name

    # Remove tool_name if present (use function instead)
    result.pop("tool_name", None)

    return result


def compare_tool_calls(
    actual: dict[str, Any],
    expected: dict[str, Any],
    strict: bool = False,
) -> tuple[bool, str]:
    """
    Compare actual and expected tool calls.

    Args:
        actual: The tool call produced by the agent
        expected: The expected tool call from fixtures
        strict: If True, require exact match. If False, allow partial matches.

    Returns:
        Tuple of (match, reason)
    """
    actual_norm = normalize_tool_call(actual)
    expected_norm = normalize_tool_call(expected)

    # Check function name
    if actual_norm.get("function") != expected_norm.get("function"):
        return False, f"Function mismatch: {actual_norm.get('function')} != {expected_norm.get('function')}"

    actual_args = actual_norm.get("arguments", {})
    expected_args = expected_norm.get("arguments", {})

    if strict:
        if actual_args != expected_args:
            return False, f"Arguments mismatch: {actual_args} != {expected_args}"
        return True, "Exact match"

    # Non-strict comparison - check key fields
    issues = []

    # Check fields
    actual_fields = set(actual_args.get("fields", []))
    expected_fields = set(expected_args.get("fields", []))
    if actual_fields != expected_fields:
        missing = expected_fields - actual_fields
        extra = actual_fields - expected_fields
        if missing:
            issues.append(f"Missing fields: {missing}")
        if extra:
            issues.append(f"Extra fields: {extra}")

    # Check tickers (normalize format)
    actual_tickers = _normalize_tickers(actual_args.get("tickers", ""))
    expected_tickers = _normalize_tickers(expected_args.get("tickers", ""))
    if actual_tickers != expected_tickers:
        issues.append(f"Ticker mismatch: {actual_tickers} != {expected_tickers}")

    if issues:
        return False, "; ".join(issues)

    return True, "Match (non-strict)"


def _normalize_tickers(tickers: str | list[str]) -> set[str]:
    """Normalize tickers for comparison."""
    if isinstance(tickers, list):
        return set(tickers)

    if not tickers:
        return set()

    # Split by comma and strip
    return {t.strip() for t in tickers.split(",")}


def extract_ticker_symbol(ticker: str) -> str:
    """
    Extract the base symbol from a ticker in various formats.

    Examples:
        @AAPL -> AAPL
        U:MSFT -> MSFT
        AAPL.O -> AAPL
        J:6758 -> 6758
    """
    # Remove common prefixes
    prefixes = ["@", "U:", "C:", "D:", "J:", "K:", "H:"]
    for prefix in prefixes:
        if ticker.startswith(prefix):
            ticker = ticker[len(prefix):]
            break

    # Remove common suffixes (.O, .N, .L, etc.)
    if "." in ticker:
        ticker = ticker.split(".")[0]

    return ticker
