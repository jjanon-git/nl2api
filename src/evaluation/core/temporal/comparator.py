"""
Temporal-Aware Tool Call Comparator.

Wraps ASTComparator with temporal awareness for date fields.
Normalizes relative date expressions before comparison.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

from CONTRACTS import TemporalValidationMode, ToolCall
from src.evaluation.core.ast_comparator import ASTComparator, ComparisonResult
from src.evaluation.core.temporal.date_resolver import DateResolver


@dataclass
class TemporalComparisonResult(ComparisonResult):
    """Extended comparison result with temporal metadata."""

    date_normalization_applied: bool = False
    normalized_dates: dict[str, dict[str, str]] = field(default_factory=dict)
    temporal_validation_errors: list[str] = field(default_factory=list)


class TemporalComparator:
    """
    Temporal-aware tool call comparator.

    Wraps ASTComparator with temporal awareness for date fields.
    Normalizes relative date expressions (like "-1D") before comparison.

    Modes:
    - BEHAVIORAL: Only validate that both have valid temporal expressions
    - STRUCTURAL: Normalize dates to absolute, then compare with ASTComparator
    - DATA: Exact match required (delegates directly to ASTComparator)
    """

    DEFAULT_DATE_FIELDS = ("start", "end", "SDate", "EDate", "Period")

    def __init__(
        self,
        date_resolver: DateResolver,
        validation_mode: TemporalValidationMode = TemporalValidationMode.STRUCTURAL,
        relative_date_fields: tuple[str, ...] | None = None,
        base_comparator: ASTComparator | None = None,
    ):
        """
        Initialize the temporal comparator.

        Args:
            date_resolver: Resolver for date expressions
            validation_mode: What aspect of temporal handling to validate
            relative_date_fields: Field names that may contain date expressions
            base_comparator: Optional base comparator (creates default if not provided)
        """
        self.date_resolver = date_resolver
        self.validation_mode = validation_mode
        self.relative_date_fields = relative_date_fields or self.DEFAULT_DATE_FIELDS
        self.base_comparator = base_comparator or ASTComparator()

    def compare(
        self,
        expected: tuple[ToolCall, ...] | list[ToolCall],
        actual: tuple[ToolCall, ...] | list[ToolCall],
    ) -> TemporalComparisonResult:
        """
        Compare expected vs actual tool calls with temporal awareness.

        Args:
            expected: Expected tool calls from test case
            actual: Actual tool calls from system response

        Returns:
            TemporalComparisonResult with detailed diff and temporal metadata
        """
        if self.validation_mode == TemporalValidationMode.BEHAVIORAL:
            return self._compare_behavioral(expected, actual)
        elif self.validation_mode == TemporalValidationMode.STRUCTURAL:
            return self._compare_structural(expected, actual)
        else:  # DATA mode - exact match
            return self._compare_data(expected, actual)

    def _compare_behavioral(
        self,
        expected: tuple[ToolCall, ...] | list[ToolCall],
        actual: tuple[ToolCall, ...] | list[ToolCall],
    ) -> TemporalComparisonResult:
        """
        Behavioral validation: only check that date fields have valid expressions.

        Both relative and absolute dates are accepted as long as they're valid.
        """
        validation_errors: list[str] = []

        # Validate expected tool calls
        for i, tc in enumerate(expected):
            errors = self._validate_temporal_fields(tc, f"expected[{i}]")
            validation_errors.extend(errors)

        # Validate actual tool calls
        for i, tc in enumerate(actual):
            errors = self._validate_temporal_fields(tc, f"actual[{i}]")
            validation_errors.extend(errors)

        # For behavioral mode, if both have valid temporal expressions, pass
        # We still need to check tool names and non-date arguments match
        if validation_errors:
            return TemporalComparisonResult(
                matches=False,
                score=0.0,
                temporal_validation_errors=validation_errors,
            )

        # Compare non-date fields using base comparator
        # First, strip date fields from comparison
        expected_stripped = [self._strip_date_fields(tc) for tc in expected]
        actual_stripped = [self._strip_date_fields(tc) for tc in actual]

        base_result = self.base_comparator.compare(
            tuple(expected_stripped),
            tuple(actual_stripped),
        )

        return TemporalComparisonResult(
            matches=base_result.matches,
            score=base_result.score,
            matched_calls=base_result.matched_calls,
            missing_calls=base_result.missing_calls,
            extra_calls=base_result.extra_calls,
            argument_diffs=base_result.argument_diffs,
            date_normalization_applied=False,
        )

    def _compare_structural(
        self,
        expected: tuple[ToolCall, ...] | list[ToolCall],
        actual: tuple[ToolCall, ...] | list[ToolCall],
    ) -> TemporalComparisonResult:
        """
        Structural validation: normalize dates to absolute, then compare.

        This is the main use case - comparing "-1D" to "2026-01-20" should pass
        if they resolve to the same date.
        """
        normalized_dates: dict[str, dict[str, str]] = {}

        # Normalize expected tool calls
        expected_normalized = []
        for i, tc in enumerate(expected):
            normalized_tc, normalizations = self._normalize_date_fields(tc)
            expected_normalized.append(normalized_tc)
            if normalizations:
                normalized_dates[f"expected[{i}]"] = normalizations

        # Normalize actual tool calls
        actual_normalized = []
        for i, tc in enumerate(actual):
            normalized_tc, normalizations = self._normalize_date_fields(tc)
            actual_normalized.append(normalized_tc)
            if normalizations:
                normalized_dates[f"actual[{i}]"] = normalizations

        # Compare normalized tool calls
        base_result = self.base_comparator.compare(
            tuple(expected_normalized),
            tuple(actual_normalized),
        )

        return TemporalComparisonResult(
            matches=base_result.matches,
            score=base_result.score,
            matched_calls=base_result.matched_calls,
            missing_calls=base_result.missing_calls,
            extra_calls=base_result.extra_calls,
            argument_diffs=base_result.argument_diffs,
            date_normalization_applied=bool(normalized_dates),
            normalized_dates=normalized_dates,
        )

    def _compare_data(
        self,
        expected: tuple[ToolCall, ...] | list[ToolCall],
        actual: tuple[ToolCall, ...] | list[ToolCall],
    ) -> TemporalComparisonResult:
        """
        Data validation: exact match required, no normalization.

        This mode is for point-in-time validation where exact dates matter.
        """
        base_result = self.base_comparator.compare(
            tuple(expected),
            tuple(actual),
        )

        return TemporalComparisonResult(
            matches=base_result.matches,
            score=base_result.score,
            matched_calls=base_result.matched_calls,
            missing_calls=base_result.missing_calls,
            extra_calls=base_result.extra_calls,
            argument_diffs=base_result.argument_diffs,
            date_normalization_applied=False,
        )

    def _validate_temporal_fields(self, tc: ToolCall, context: str) -> list[str]:
        """
        Validate that date fields contain valid temporal expressions.

        Args:
            tc: Tool call to validate
            context: Context string for error messages

        Returns:
            List of validation error messages
        """
        errors: list[str] = []
        args = dict(tc.arguments)

        for field_name in self.relative_date_fields:
            if field_name in args:
                value = args[field_name]
                if isinstance(value, str):
                    if not self.date_resolver.is_valid_temporal_expr(value):
                        errors.append(
                            f"{context}.{tc.tool_name}.{field_name}: "
                            f"'{value}' is not a valid temporal expression"
                        )

        return errors

    def _normalize_date_fields(self, tc: ToolCall) -> tuple[ToolCall, dict[str, str]]:
        """
        Normalize date fields to absolute dates.

        Args:
            tc: Tool call to normalize

        Returns:
            Tuple of (normalized ToolCall, dict of normalizations performed)
        """
        args = dict(tc.arguments)
        normalizations: dict[str, str] = {}

        for field_name in self.relative_date_fields:
            if field_name in args:
                value = args[field_name]
                if isinstance(value, str):
                    normalized = self.date_resolver.normalize(value)
                    if normalized != value:
                        normalizations[field_name] = f"{value} -> {normalized}"
                        args[field_name] = normalized

        if normalizations:
            return ToolCall(tool_name=tc.tool_name, arguments=args), normalizations
        return tc, normalizations

    def _strip_date_fields(self, tc: ToolCall) -> ToolCall:
        """
        Remove date fields from a tool call for behavioral comparison.

        Args:
            tc: Tool call to strip

        Returns:
            New ToolCall without date fields
        """
        args = {k: v for k, v in tc.arguments.items() if k not in self.relative_date_fields}
        return ToolCall(tool_name=tc.tool_name, arguments=args)


def compare_tool_calls_temporal(
    expected: tuple[ToolCall, ...] | list[ToolCall],
    actual: tuple[ToolCall, ...] | list[ToolCall],
    reference_date: date | None = None,
    validation_mode: TemporalValidationMode = TemporalValidationMode.STRUCTURAL,
    relative_date_fields: tuple[str, ...] | None = None,
    fiscal_year_end_month: int = 12,
) -> TemporalComparisonResult:
    """
    Convenience function for temporal-aware tool call comparison.

    Args:
        expected: Expected tool calls
        actual: Actual tool calls
        reference_date: Reference date for resolving relative expressions
        validation_mode: What aspect of temporal handling to validate
        relative_date_fields: Field names that may contain date expressions
        fiscal_year_end_month: Month when fiscal year ends (for FY/FQ resolution)

    Returns:
        TemporalComparisonResult
    """
    resolver = DateResolver(
        reference_date=reference_date or date.today(),
        fiscal_year_end_month=fiscal_year_end_month,
    )
    comparator = TemporalComparator(
        date_resolver=resolver,
        validation_mode=validation_mode,
        relative_date_fields=relative_date_fields,
    )
    return comparator.compare(expected, actual)
