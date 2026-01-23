"""Tests for AST comparator."""

from CONTRACTS import ToolCall
from src.evaluation.core.ast_comparator import ASTComparator


class TestASTComparator:
    """Test suite for ASTComparator."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.comparator = ASTComparator()

    def test_exact_match_single_call(self) -> None:
        """Test exact match with single tool call."""
        expected = [ToolCall(tool_name="search", arguments={"query": "test"})]
        actual = [ToolCall(tool_name="search", arguments={"query": "test"})]

        result = self.comparator.compare(expected, actual)

        assert result.matches is True
        assert result.score == 1.0
        assert len(result.matched_calls) == 1
        assert len(result.missing_calls) == 0
        assert len(result.extra_calls) == 0

    def test_exact_match_multiple_calls(self) -> None:
        """Test exact match with multiple tool calls."""
        expected = [
            ToolCall(tool_name="get_user", arguments={"id": 1}),
            ToolCall(tool_name="get_orders", arguments={"user_id": 1}),
        ]
        actual = [
            ToolCall(tool_name="get_user", arguments={"id": 1}),
            ToolCall(tool_name="get_orders", arguments={"user_id": 1}),
        ]

        result = self.comparator.compare(expected, actual)

        assert result.matches is True
        assert result.score == 1.0
        assert len(result.matched_calls) == 2

    def test_order_independent_matching(self) -> None:
        """Test that order doesn't matter for matching."""
        expected = [
            ToolCall(tool_name="first", arguments={"a": 1}),
            ToolCall(tool_name="second", arguments={"b": 2}),
        ]
        actual = [
            ToolCall(tool_name="second", arguments={"b": 2}),
            ToolCall(tool_name="first", arguments={"a": 1}),
        ]

        result = self.comparator.compare(expected, actual)

        assert result.matches is True
        assert result.score == 1.0

    def test_missing_call(self) -> None:
        """Test detection of missing call."""
        expected = [
            ToolCall(tool_name="call_a", arguments={}),
            ToolCall(tool_name="call_b", arguments={}),
        ]
        actual = [
            ToolCall(tool_name="call_a", arguments={}),
        ]

        result = self.comparator.compare(expected, actual)

        assert result.matches is False
        assert len(result.missing_calls) == 1
        assert result.missing_calls[0].tool_name == "call_b"

    def test_extra_call(self) -> None:
        """Test detection of extra call."""
        expected = [
            ToolCall(tool_name="call_a", arguments={}),
        ]
        actual = [
            ToolCall(tool_name="call_a", arguments={}),
            ToolCall(tool_name="call_extra", arguments={}),
        ]

        result = self.comparator.compare(expected, actual)

        assert result.matches is False
        assert len(result.extra_calls) == 1
        assert result.extra_calls[0].tool_name == "call_extra"

    def test_argument_mismatch(self) -> None:
        """Test detection of argument mismatch."""
        expected = [
            ToolCall(tool_name="search", arguments={"limit": 10}),
        ]
        actual = [
            ToolCall(tool_name="search", arguments={"limit": 20}),
        ]

        result = self.comparator.compare(expected, actual)

        assert result.matches is False
        assert len(result.argument_diffs) == 1

    def test_type_coercion_string_to_int(self) -> None:
        """Test that string '5' matches int 5."""
        expected = [
            ToolCall(tool_name="get_item", arguments={"id": 5}),
        ]
        actual = [
            ToolCall(tool_name="get_item", arguments={"id": "5"}),
        ]

        result = self.comparator.compare(expected, actual)

        assert result.matches is True
        assert result.score == 1.0

    def test_type_coercion_int_to_string(self) -> None:
        """Test that int 5 matches string '5'."""
        expected = [
            ToolCall(tool_name="get_item", arguments={"id": "123"}),
        ]
        actual = [
            ToolCall(tool_name="get_item", arguments={"id": 123}),
        ]

        result = self.comparator.compare(expected, actual)

        assert result.matches is True
        assert result.score == 1.0

    def test_numeric_tolerance(self) -> None:
        """Test numeric comparison with tolerance."""
        comparator = ASTComparator(numeric_tolerance=0.01)  # 1% tolerance

        expected = [
            ToolCall(tool_name="calc", arguments={"value": 100.0}),
        ]
        actual = [
            ToolCall(tool_name="calc", arguments={"value": 100.5}),  # 0.5% diff
        ]

        result = comparator.compare(expected, actual)

        assert result.matches is True

    def test_numeric_tolerance_exceeded(self) -> None:
        """Test numeric comparison exceeding tolerance."""
        comparator = ASTComparator(numeric_tolerance=0.001)  # 0.1% tolerance

        expected = [
            ToolCall(tool_name="calc", arguments={"value": 100.0}),
        ]
        actual = [
            ToolCall(tool_name="calc", arguments={"value": 102.0}),  # 2% diff
        ]

        result = comparator.compare(expected, actual)

        assert result.matches is False

    def test_nested_object_comparison(self) -> None:
        """Test comparison of nested objects in arguments."""
        expected = [
            ToolCall(
                tool_name="create_order",
                arguments={
                    "items": [
                        {"product_id": 1, "quantity": 2},
                        {"product_id": 2, "quantity": 1},
                    ],
                    "shipping": {"method": "express", "address": "123 Main St"},
                },
            ),
        ]
        actual = [
            ToolCall(
                tool_name="create_order",
                arguments={
                    "items": [
                        {"product_id": 1, "quantity": 2},
                        {"product_id": 2, "quantity": 1},
                    ],
                    "shipping": {"method": "express", "address": "123 Main St"},
                },
            ),
        ]

        result = self.comparator.compare(expected, actual)

        assert result.matches is True

    def test_nested_object_mismatch(self) -> None:
        """Test detection of mismatch in nested objects."""
        expected = [
            ToolCall(
                tool_name="search",
                arguments={"filters": {"category": "electronics"}},
            ),
        ]
        actual = [
            ToolCall(
                tool_name="search",
                arguments={"filters": {"category": "books"}},
            ),
        ]

        result = self.comparator.compare(expected, actual)

        assert result.matches is False

    def test_empty_calls(self) -> None:
        """Test comparison with empty call lists."""
        result = self.comparator.compare([], [])

        assert result.matches is True
        assert result.score == 1.0

    def test_extra_argument_key(self) -> None:
        """Test detection of extra argument key."""
        expected = [
            ToolCall(tool_name="func", arguments={"a": 1}),
        ]
        actual = [
            ToolCall(tool_name="func", arguments={"a": 1, "b": 2}),
        ]

        result = self.comparator.compare(expected, actual)

        assert result.matches is False
        assert len(result.argument_diffs) == 1

    def test_missing_argument_key(self) -> None:
        """Test detection of missing argument key."""
        expected = [
            ToolCall(tool_name="func", arguments={"a": 1, "b": 2}),
        ]
        actual = [
            ToolCall(tool_name="func", arguments={"a": 1}),
        ]

        result = self.comparator.compare(expected, actual)

        assert result.matches is False
        assert len(result.argument_diffs) == 1

    def test_score_calculation(self) -> None:
        """Test score calculation for partial matches."""
        expected = [
            ToolCall(tool_name="call_a", arguments={}),
            ToolCall(tool_name="call_b", arguments={}),
            ToolCall(tool_name="call_c", arguments={}),
        ]
        actual = [
            ToolCall(tool_name="call_a", arguments={}),
            ToolCall(tool_name="call_b", arguments={}),
        ]

        result = self.comparator.compare(expected, actual)

        assert result.matches is False
        assert 0.0 < result.score < 1.0  # Partial score

    def test_boolean_comparison(self) -> None:
        """Test boolean value comparison."""
        expected = [
            ToolCall(tool_name="toggle", arguments={"enabled": True}),
        ]
        actual = [
            ToolCall(tool_name="toggle", arguments={"enabled": True}),
        ]

        result = self.comparator.compare(expected, actual)

        assert result.matches is True

    def test_none_value_comparison(self) -> None:
        """Test comparison with None values."""
        expected = [
            ToolCall(tool_name="func", arguments={"value": None}),
        ]
        actual = [
            ToolCall(tool_name="func", arguments={"value": None}),
        ]

        result = self.comparator.compare(expected, actual)

        assert result.matches is True

    def test_result_summary(self) -> None:
        """Test human-readable result summary."""
        expected = [ToolCall(tool_name="a", arguments={})]
        actual = [ToolCall(tool_name="a", arguments={})]

        result = self.comparator.compare(expected, actual)

        assert "MATCH" in result.summary
        assert "1.00" in result.summary
