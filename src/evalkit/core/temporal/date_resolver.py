"""
Date Resolver for Temporal Evaluation.

Resolves relative date expressions (like "-1D", "FQ0") to absolute dates
given a reference date. Used for temporally-aware evaluation where
relative and absolute dates should be treated as equivalent.
"""

from __future__ import annotations

import re
from datetime import date, timedelta
from typing import Final

# Regex patterns for different temporal expressions
RELATIVE_DAY_PATTERN: Final = re.compile(r"^(-?\d+)D$")  # -1D, 0D, 7D
RELATIVE_MONTH_PATTERN: Final = re.compile(r"^(-?\d+)M$")  # -1M, -3M, 12M
RELATIVE_YEAR_PATTERN: Final = re.compile(r"^(-?\d+)Y$")  # -1Y, -5Y, 10Y
FISCAL_YEAR_PATTERN: Final = re.compile(r"^FY(-?\d+|\d{4})$")  # FY0, FY-1, FY2024
FISCAL_QUARTER_PATTERN: Final = re.compile(r"^FQ(-?\d+|\d{1}\d{4})$")  # FQ0, FQ-1, FQ12024
ABSOLUTE_DATE_PATTERN: Final = re.compile(r"^\d{4}-\d{2}-\d{2}$")  # 2024-01-15


class DateResolver:
    """
    Resolves relative date expressions to absolute dates.

    Supports:
    - Relative days: -1D, 0D, 7D
    - Relative months: -1M, -3M, 12M
    - Relative years: -1Y, -5Y, 10Y
    - Fiscal years: FY0 (current), FY-1 (prior), FY1 (next), FY2024 (absolute)
    - Fiscal quarters: FQ0 (current), FQ-1 (prior), FQ1 (next), FQ12024 (Q1 2024)
    - Absolute dates: 2024-01-15 (passed through unchanged)

    Example:
        >>> resolver = DateResolver(reference_date=date(2026, 1, 21))
        >>> resolver.resolve("-1D")
        datetime.date(2026, 1, 20)
        >>> resolver.are_equivalent("-1D", "2026-01-20")
        True
    """

    def __init__(
        self,
        reference_date: date | None = None,
        fiscal_year_end_month: int = 12,
    ):
        """
        Initialize the DateResolver.

        Args:
            reference_date: The reference date for relative expressions.
                           Defaults to today if not provided.
            fiscal_year_end_month: Month when fiscal year ends (1-12).
                                  Defaults to 12 (December).
        """
        self.reference_date = reference_date or date.today()
        self.fiscal_year_end_month = fiscal_year_end_month

    def resolve(self, expr: str) -> date | None:
        """
        Resolve a date expression to an absolute date.

        Args:
            expr: The date expression to resolve (e.g., "-1D", "FY0", "2024-01-15")

        Returns:
            The resolved date, or None if the expression is invalid.
        """
        if not expr:
            return None

        expr = expr.strip()

        # Try absolute date first
        if ABSOLUTE_DATE_PATTERN.match(expr):
            try:
                return date.fromisoformat(expr)
            except ValueError:
                return None

        # Relative days: -1D, 0D, 7D
        match = RELATIVE_DAY_PATTERN.match(expr)
        if match:
            days = int(match.group(1))
            return self.reference_date + timedelta(days=days)

        # Relative months: -1M, -3M, 12M
        match = RELATIVE_MONTH_PATTERN.match(expr)
        if match:
            months = int(match.group(1))
            return self._add_months(self.reference_date, months)

        # Relative years: -1Y, -5Y, 10Y
        match = RELATIVE_YEAR_PATTERN.match(expr)
        if match:
            years = int(match.group(1))
            return self._add_years(self.reference_date, years)

        # Fiscal years: FY0, FY-1, FY1, FY2024
        match = FISCAL_YEAR_PATTERN.match(expr)
        if match:
            fy_value = match.group(1)
            return self._resolve_fiscal_year(fy_value)

        # Fiscal quarters: FQ0, FQ-1, FQ1, FQ12024
        match = FISCAL_QUARTER_PATTERN.match(expr)
        if match:
            fq_value = match.group(1)
            return self._resolve_fiscal_quarter(fq_value)

        return None

    def normalize(self, expr: str) -> str:
        """
        Normalize a date expression to ISO format (YYYY-MM-DD).

        If the expression is already absolute, returns it unchanged.
        If relative, resolves and returns as ISO format.
        If invalid, returns the original expression.

        Args:
            expr: The date expression to normalize

        Returns:
            Normalized date string in YYYY-MM-DD format, or original if invalid.
        """
        if not expr:
            return expr

        resolved = self.resolve(expr)
        if resolved:
            return resolved.isoformat()
        return expr

    def are_equivalent(self, a: str, b: str) -> bool:
        """
        Check if two date expressions are equivalent.

        Args:
            a: First date expression
            b: Second date expression

        Returns:
            True if both resolve to the same date, False otherwise.
        """
        resolved_a = self.resolve(a)
        resolved_b = self.resolve(b)

        if resolved_a is None or resolved_b is None:
            return False

        return resolved_a == resolved_b

    def is_valid_temporal_expr(self, expr: str) -> bool:
        """
        Check if a string is a valid temporal expression.

        Args:
            expr: The expression to validate

        Returns:
            True if the expression is a recognized temporal pattern.
        """
        if not expr:
            return False

        expr = expr.strip()

        return any(
            [
                ABSOLUTE_DATE_PATTERN.match(expr),
                RELATIVE_DAY_PATTERN.match(expr),
                RELATIVE_MONTH_PATTERN.match(expr),
                RELATIVE_YEAR_PATTERN.match(expr),
                FISCAL_YEAR_PATTERN.match(expr),
                FISCAL_QUARTER_PATTERN.match(expr),
            ]
        )

    def _add_months(self, d: date, months: int) -> date:
        """Add months to a date, handling month/year overflow."""
        year = d.year
        month = d.month + months

        # Handle year overflow
        while month > 12:
            year += 1
            month -= 12
        while month < 1:
            year -= 1
            month += 12

        # Handle day overflow (e.g., Jan 31 + 1 month -> Feb 28)
        day = min(d.day, self._days_in_month(year, month))

        return date(year, month, day)

    def _add_years(self, d: date, years: int) -> date:
        """Add years to a date, handling leap year edge cases."""
        new_year = d.year + years

        # Handle Feb 29 in non-leap years
        if d.month == 2 and d.day == 29:
            if not self._is_leap_year(new_year):
                return date(new_year, 2, 28)

        return date(new_year, d.month, d.day)

    def _resolve_fiscal_year(self, fy_value: str) -> date:
        """
        Resolve a fiscal year expression.

        - FY0: Current fiscal year end date
        - FY-1: Prior fiscal year end date
        - FY1: Next fiscal year end date
        - FY2024: Fiscal year 2024 end date
        """
        current_fy = self._get_current_fiscal_year()

        if len(fy_value) == 4:  # Absolute: FY2024
            fy_year = int(fy_value)
        else:  # Relative: FY0, FY-1, FY1
            offset = int(fy_value)
            fy_year = current_fy + offset

        # Return the fiscal year end date
        return self._fiscal_year_end_date(fy_year)

    def _resolve_fiscal_quarter(self, fq_value: str) -> date:
        """
        Resolve a fiscal quarter expression.

        - FQ0: Current fiscal quarter end date
        - FQ-1: Prior fiscal quarter end date
        - FQ1: Next fiscal quarter end date
        - FQ12024: Q1 of fiscal year 2024 end date
        """
        if len(fq_value) == 5:  # Absolute: FQ12024 (quarter + year)
            quarter = int(fq_value[0])
            fy_year = int(fq_value[1:])
        else:  # Relative: FQ0, FQ-1, FQ1
            current_fq, current_fy = self._get_current_fiscal_quarter()
            offset = int(fq_value)

            # Calculate target quarter and year
            total_quarters = (current_fy * 4 + current_fq - 1) + offset
            fy_year = total_quarters // 4
            quarter = (total_quarters % 4) + 1

        return self._fiscal_quarter_end_date(fy_year, quarter)

    def _get_current_fiscal_year(self) -> int:
        """Get the current fiscal year based on reference date."""
        if self.reference_date.month > self.fiscal_year_end_month:
            return self.reference_date.year + 1
        return self.reference_date.year

    def _get_current_fiscal_quarter(self) -> tuple[int, int]:
        """Get the current fiscal quarter (1-4) and fiscal year."""
        fy = self._get_current_fiscal_year()

        # Calculate which quarter we're in based on fiscal year end
        # For December year-end: Q1=Jan-Mar, Q2=Apr-Jun, Q3=Jul-Sep, Q4=Oct-Dec
        # For June year-end: Q1=Jul-Sep, Q2=Oct-Dec, Q3=Jan-Mar, Q4=Apr-Jun

        # Months from fiscal year start
        fiscal_start_month = (self.fiscal_year_end_month % 12) + 1
        months_from_start = (self.reference_date.month - fiscal_start_month) % 12
        quarter = (months_from_start // 3) + 1

        return quarter, fy

    def _fiscal_year_end_date(self, fy_year: int) -> date:
        """Get the end date of a fiscal year."""
        # Fiscal year 2024 ends on the last day of fiscal_year_end_month in 2024
        # (or 2023 if fiscal year end is after December - not typical)
        year = fy_year if self.fiscal_year_end_month == 12 else fy_year
        month = self.fiscal_year_end_month
        day = self._days_in_month(year, month)
        return date(year, month, day)

    def _fiscal_quarter_end_date(self, fy_year: int, quarter: int) -> date:
        """Get the end date of a fiscal quarter."""
        # Calculate the end month for the quarter
        # Q1 ends 3 months after fiscal year start
        # Q2 ends 6 months after fiscal year start, etc.
        fiscal_start_month = (self.fiscal_year_end_month % 12) + 1
        end_month = (fiscal_start_month + (quarter * 3) - 1 - 1) % 12 + 1

        # Determine the year
        if end_month > self.fiscal_year_end_month and self.fiscal_year_end_month != 12:
            year = fy_year - 1
        else:
            year = fy_year

        day = self._days_in_month(year, end_month)
        return date(year, end_month, day)

    @staticmethod
    def _days_in_month(year: int, month: int) -> int:
        """Get the number of days in a month."""
        if month in (4, 6, 9, 11):
            return 30
        if month == 2:
            if DateResolver._is_leap_year(year):
                return 29
            return 28
        return 31

    @staticmethod
    def _is_leap_year(year: int) -> bool:
        """Check if a year is a leap year."""
        return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)
