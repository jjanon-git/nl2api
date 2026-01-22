"""
AST-based Tool Call Comparator

Provides deep comparison of tool calls with:
- Order-independent comparison (set semantics)
- Type-aware argument comparison ("5" vs 5)
- Nested object deep comparison
- Argument permutation tolerance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from CONTRACTS import ToolCall, ToolRegistry


@dataclass
class ComparisonResult:
    """Result of comparing two sets of tool calls."""

    matches: bool
    score: float  # 0.0 to 1.0
    matched_calls: list[tuple[ToolCall, ToolCall]] = field(default_factory=list)
    missing_calls: list[ToolCall] = field(default_factory=list)  # Expected but not found
    extra_calls: list[ToolCall] = field(default_factory=list)  # Found but not expected
    argument_diffs: list[dict[str, Any]] = field(default_factory=list)

    @property
    def summary(self) -> str:
        """Human-readable summary of comparison."""
        if self.matches:
            return f"MATCH (score: {self.score:.2f})"

        parts = []
        if self.missing_calls:
            parts.append(f"{len(self.missing_calls)} missing call(s)")
        if self.extra_calls:
            parts.append(f"{len(self.extra_calls)} extra call(s)")
        if self.argument_diffs:
            parts.append(f"{len(self.argument_diffs)} argument diff(s)")

        return f"MISMATCH: {', '.join(parts)} (score: {self.score:.2f})"


class ASTComparator:
    """
    Compares tool calls using AST-like semantic comparison.

    Handles:
    - Order-independent matching
    - Type coercion (string "5" matches int 5)
    - Numeric tolerance for float comparison
    - Nested object/array comparison
    - Tool name normalization (datastream_get_data -> get_data)
    """

    def __init__(
        self,
        numeric_tolerance: float = 0.0001,
        normalize_tool_names: bool = True,
    ):
        """
        Initialize comparator.

        Args:
            numeric_tolerance: Relative tolerance for numeric comparisons (default 0.01%)
            normalize_tool_names: Whether to normalize tool names before comparison
        """
        self.numeric_tolerance = numeric_tolerance
        self.normalize_tool_names = normalize_tool_names

    def _get_normalized_tool_name(self, tool_name: str) -> str:
        """Normalize tool name for comparison."""
        if not self.normalize_tool_names:
            return tool_name
        return ToolRegistry.normalize(tool_name)

    def compare(
        self,
        expected: tuple[ToolCall, ...] | list[ToolCall],
        actual: tuple[ToolCall, ...] | list[ToolCall],
    ) -> ComparisonResult:
        """
        Compare expected vs actual tool calls.

        Uses set-based matching to handle order independence.
        Each expected call is matched to at most one actual call.

        Args:
            expected: Expected tool calls from test case
            actual: Actual tool calls from system response

        Returns:
            ComparisonResult with detailed diff information
        """
        expected_list = list(expected)
        actual_list = list(actual)

        matched_pairs: list[tuple[ToolCall, ToolCall]] = []
        argument_diffs: list[dict[str, Any]] = []

        # Track which actual calls have been matched
        matched_actual_indices: set[int] = set()

        # Try to match each expected call
        for exp_call in expected_list:
            best_match_idx: int | None = None
            best_match_diff: dict[str, Any] | None = None

            for i, act_call in enumerate(actual_list):
                if i in matched_actual_indices:
                    continue

                # Check tool name match (with optional normalization)
                exp_name = self._get_normalized_tool_name(exp_call.tool_name)
                act_name = self._get_normalized_tool_name(act_call.tool_name)
                if exp_name != act_name:
                    continue

                # Compare arguments
                args_match, diff = self._compare_arguments(
                    dict(exp_call.arguments),
                    dict(act_call.arguments),
                )

                if args_match:
                    # Perfect match found
                    best_match_idx = i
                    best_match_diff = None
                    break
                elif best_match_idx is None:
                    # Store as potential partial match
                    best_match_idx = i
                    best_match_diff = diff

            if best_match_idx is not None:
                matched_actual_indices.add(best_match_idx)
                matched_pairs.append((exp_call, actual_list[best_match_idx]))
                if best_match_diff:
                    argument_diffs.append({
                        "expected_call": exp_call.tool_name,
                        "diff": best_match_diff,
                    })

        # Determine missing and extra calls
        missing_calls = [
            exp for i, exp in enumerate(expected_list)
            if not any(exp == pair[0] for pair in matched_pairs)
        ]
        extra_calls = [
            act for i, act in enumerate(actual_list)
            if i not in matched_actual_indices
        ]

        # Calculate score
        total_expected = len(expected_list)
        total_actual = len(actual_list)

        if total_expected == 0 and total_actual == 0:
            score = 1.0
        elif total_expected == 0:
            score = 0.0  # Extra calls with nothing expected
        else:
            # Score based on matched calls and argument accuracy
            matched_count = len(matched_pairs) - len(argument_diffs)
            score = matched_count / max(total_expected, total_actual)

        matches = (
            len(missing_calls) == 0
            and len(extra_calls) == 0
            and len(argument_diffs) == 0
        )

        return ComparisonResult(
            matches=matches,
            score=max(0.0, min(1.0, score)),
            matched_calls=matched_pairs,
            missing_calls=missing_calls,
            extra_calls=extra_calls,
            argument_diffs=argument_diffs,
        )

    def _compare_arguments(
        self,
        expected: dict[str, Any],
        actual: dict[str, Any],
    ) -> tuple[bool, dict[str, Any] | None]:
        """
        Compare argument dictionaries.

        Returns:
            Tuple of (matches, diff_details)
        """
        diff: dict[str, Any] = {}

        # Check for missing keys
        missing_keys = set(expected.keys()) - set(actual.keys())
        if missing_keys:
            diff["missing_keys"] = list(missing_keys)

        # Check for extra keys
        extra_keys = set(actual.keys()) - set(expected.keys())
        if extra_keys:
            diff["extra_keys"] = list(extra_keys)

        # Compare common keys
        common_keys = set(expected.keys()) & set(actual.keys())
        value_diffs: dict[str, dict[str, Any]] = {}

        for key in common_keys:
            if not self._values_equal(expected[key], actual[key]):
                value_diffs[key] = {
                    "expected": expected[key],
                    "actual": actual[key],
                }

        if value_diffs:
            diff["value_diffs"] = value_diffs

        if diff:
            return False, diff
        return True, None

    def _values_equal(self, expected: Any, actual: Any) -> bool:
        """
        Compare two values with type coercion and tolerance.

        Handles:
        - Numeric tolerance
        - String/number coercion
        - Nested dicts and lists
        - None handling
        """
        # None handling
        if expected is None and actual is None:
            return True
        if expected is None or actual is None:
            return False

        # Same type, direct comparison
        if type(expected) is type(actual):
            if isinstance(expected, dict):
                return self._dicts_equal(expected, actual)
            if isinstance(expected, (list, tuple)):
                return self._lists_equal(expected, actual)
            if isinstance(expected, float):
                return self._floats_equal(expected, actual)
            return expected == actual

        # Type coercion: string <-> number
        if isinstance(expected, str) and isinstance(actual, (int, float)):
            try:
                return self._values_equal(float(expected), float(actual))
            except ValueError:
                return False

        if isinstance(expected, (int, float)) and isinstance(actual, str):
            try:
                return self._values_equal(float(expected), float(actual))
            except ValueError:
                return False

        # int <-> float
        if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
            return self._floats_equal(float(expected), float(actual))

        # list <-> tuple
        if isinstance(expected, (list, tuple)) and isinstance(actual, (list, tuple)):
            return self._lists_equal(list(expected), list(actual))

        return False

    def _floats_equal(self, expected: float, actual: float) -> bool:
        """Compare floats with tolerance."""
        if expected == 0:
            return abs(actual) <= self.numeric_tolerance
        return abs(expected - actual) / abs(expected) <= self.numeric_tolerance

    def _dicts_equal(self, expected: dict, actual: dict) -> bool:
        """Compare dictionaries recursively."""
        if set(expected.keys()) != set(actual.keys()):
            return False
        return all(
            self._values_equal(expected[k], actual[k])
            for k in expected.keys()
        )

    def _lists_equal(self, expected: list, actual: list) -> bool:
        """Compare lists element by element."""
        if len(expected) != len(actual):
            return False
        return all(
            self._values_equal(e, a)
            for e, a in zip(expected, actual)
        )


def compare_tool_calls(
    expected: tuple[ToolCall, ...] | list[ToolCall],
    actual: tuple[ToolCall, ...] | list[ToolCall],
    numeric_tolerance: float = 0.0001,
) -> ComparisonResult:
    """
    Convenience function for comparing tool calls.

    Args:
        expected: Expected tool calls
        actual: Actual tool calls
        numeric_tolerance: Tolerance for numeric comparisons

    Returns:
        ComparisonResult
    """
    comparator = ASTComparator(numeric_tolerance=numeric_tolerance)
    return comparator.compare(expected, actual)
