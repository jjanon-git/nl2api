"""Fixture loader utility for loading generated test cases."""

from __future__ import annotations

import json
import logging
import os
import random
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from CONTRACTS import TestCaseSetConfig

logger = logging.getLogger(__name__)

# Environment variable to limit fixtures for fast unit tests
# Set FIXTURE_SAMPLE_SIZE=50 to load only 50 fixtures per category
FIXTURE_SAMPLE_SIZE = int(os.environ.get("FIXTURE_SAMPLE_SIZE", "0"))


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
    expected_response: dict[str, Any] | None = None
    expected_nl_response: str | None = None

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
            expected_response=data.get("expected_response"),
            expected_nl_response=data.get("expected_nl_response"),
        )


@dataclass
class LoadedFixtureSet:
    """A loaded fixture set with its configuration and test cases."""

    config: TestCaseSetConfig
    test_cases: list[GeneratedTestCase]
    validation_errors: list[str] = field(default_factory=list)


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
        "entity_resolution",
    ]

    def __init__(self, fixture_dir: Path | None = None):
        """Initialize the fixture loader."""
        self.fixture_dir = fixture_dir or self.FIXTURE_DIR

    def load_category(
        self, category: str, sample_size: int | None = None
    ) -> list[GeneratedTestCase]:
        """
        Load test cases from a category.

        Args:
            category: The category name to load
            sample_size: If set, randomly sample this many fixtures.
                         If None, uses FIXTURE_SAMPLE_SIZE env var (0 = all).
        """
        result = self.load_category_with_config(category)
        if not result:
            return []

        test_cases = result.test_cases

        # Apply sampling if requested
        limit = sample_size if sample_size is not None else FIXTURE_SAMPLE_SIZE
        if limit > 0 and len(test_cases) > limit:
            # Use consistent seed for reproducibility
            rng = random.Random(42)
            test_cases = rng.sample(test_cases, limit)

        return test_cases

    def load_category_with_config(
        self, category: str, validate: bool = False
    ) -> LoadedFixtureSet | None:
        """
        Load all test cases from a category with configuration.

        Args:
            category: The category name to load
            validate: If True, validate test cases against config requirements

        Returns:
            LoadedFixtureSet with config, test cases, and any validation errors
        """
        file_path = self.fixture_dir / category / f"{category}.json"

        if not file_path.exists():
            return None

        with open(file_path) as f:
            data = json.load(f)

        # Parse _meta block (with defaults for backward compatibility)
        meta_data = data.get("_meta", {})
        config = self._parse_config(meta_data, category)

        # Load test cases
        test_cases = [
            GeneratedTestCase.from_dict(tc)
            for tc in data.get("test_cases", data if isinstance(data, list) else [])
        ]

        # Validate if requested
        validation_errors = []
        if validate:
            validation_errors = self._validate_test_cases(test_cases, config)

        return LoadedFixtureSet(
            config=config,
            test_cases=test_cases,
            validation_errors=validation_errors,
        )

    def _parse_config(self, meta_data: dict[str, Any], category: str) -> TestCaseSetConfig:
        """Parse _meta block into TestCaseSetConfig with defaults."""
        # Parse datetime if present
        generated_at = None
        if meta_data.get("generated_at"):
            try:
                generated_at = datetime.fromisoformat(
                    meta_data["generated_at"].replace("Z", "+00:00")
                )
            except (ValueError, TypeError):
                pass

        return TestCaseSetConfig(
            name=meta_data.get("name", category),
            capability=meta_data.get("capability", "nl2api"),
            description=meta_data.get("description"),
            requires_nl_response=meta_data.get("requires_nl_response", True),
            requires_expected_response=meta_data.get("requires_expected_response", False),
            schema_version=meta_data.get("schema_version", "1.0"),
            generated_at=generated_at,
            generator=meta_data.get("generator"),
        )

    def _validate_test_cases(
        self, test_cases: list[GeneratedTestCase], config: TestCaseSetConfig
    ) -> list[str]:
        """Validate test cases against set configuration."""
        errors = []

        for tc in test_cases:
            if config.requires_nl_response and not tc.expected_nl_response:
                errors.append(
                    f"Test case {tc.id} missing expected_nl_response (required by {config.name})"
                )
            if config.requires_expected_response and not tc.expected_response:
                errors.append(
                    f"Test case {tc.id} missing expected_response (required by {config.name})"
                )

        if errors:
            logger.warning(f"Validation errors in {config.name}: {len(errors)} issues found")

        return errors

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
            yield from self.load_category(category)

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
        return (
            False,
            f"Function mismatch: {actual_norm.get('function')} != {expected_norm.get('function')}",
        )

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
            ticker = ticker[len(prefix) :]
            break

    # Remove common suffixes (.O, .N, .L, etc.)
    if "." in ticker:
        ticker = ticker.split(".")[0]

    return ticker
