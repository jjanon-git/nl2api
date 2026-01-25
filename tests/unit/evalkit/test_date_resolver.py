"""
Unit tests for DateResolver.

Tests temporal expression resolution including:
- Relative days, months, years
- Fiscal years and quarters
- Equivalence checking
- Validation
- Edge cases (leap years, month boundaries)
"""

from datetime import date

from src.evalkit.core.temporal.date_resolver import DateResolver


class TestRelativeDays:
    """Tests for relative day expressions (-1D, 0D, 7D)."""

    def test_resolve_today(self):
        """0D should resolve to reference date."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.resolve("0D") == date(2026, 1, 21)

    def test_resolve_yesterday(self):
        """-1D should resolve to day before reference."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.resolve("-1D") == date(2026, 1, 20)

    def test_resolve_week_ago(self):
        """-7D should resolve to 7 days before reference."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.resolve("-7D") == date(2026, 1, 14)

    def test_resolve_future_day(self):
        """7D should resolve to 7 days after reference."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.resolve("7D") == date(2026, 1, 28)

    def test_resolve_cross_month_boundary(self):
        """-30D from Jan 21 should cross into December."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.resolve("-30D") == date(2025, 12, 22)

    def test_resolve_cross_year_boundary(self):
        """-365D should go back roughly a year."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.resolve("-365D") == date(2025, 1, 21)


class TestRelativeMonths:
    """Tests for relative month expressions (-1M, -3M, 12M)."""

    def test_resolve_last_month(self):
        """-1M should go back one month."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.resolve("-1M") == date(2025, 12, 21)

    def test_resolve_three_months_ago(self):
        """-3M should go back three months."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.resolve("-3M") == date(2025, 10, 21)

    def test_resolve_six_months_ago(self):
        """-6M should go back six months."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.resolve("-6M") == date(2025, 7, 21)

    def test_resolve_year_in_months(self):
        """-12M should go back one year."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.resolve("-12M") == date(2025, 1, 21)

    def test_resolve_future_months(self):
        """3M should go forward three months."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.resolve("3M") == date(2026, 4, 21)

    def test_month_end_handling_jan31_to_feb(self):
        """Jan 31 - 1M should give Feb 28/29, not Feb 31."""
        resolver = DateResolver(reference_date=date(2026, 3, 31))
        # March 31 - 1M = Feb 28 (not leap year in 2026)
        assert resolver.resolve("-1M") == date(2026, 2, 28)

    def test_month_end_handling_leap_year(self):
        """Month handling should respect leap years."""
        # 2024 is a leap year
        resolver = DateResolver(reference_date=date(2024, 3, 31))
        assert resolver.resolve("-1M") == date(2024, 2, 29)


class TestRelativeYears:
    """Tests for relative year expressions (-1Y, -5Y, 10Y)."""

    def test_resolve_last_year(self):
        """-1Y should go back one year."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.resolve("-1Y") == date(2025, 1, 21)

    def test_resolve_five_years_ago(self):
        """-5Y should go back five years."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.resolve("-5Y") == date(2021, 1, 21)

    def test_resolve_ten_years_future(self):
        """10Y should go forward ten years."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.resolve("10Y") == date(2036, 1, 21)

    def test_leap_year_feb29_to_non_leap(self):
        """Feb 29 - 1Y should give Feb 28 in non-leap year."""
        resolver = DateResolver(reference_date=date(2024, 2, 29))
        assert resolver.resolve("-1Y") == date(2023, 2, 28)

    def test_leap_year_feb29_to_leap(self):
        """Feb 29 - 4Y should give Feb 29 in another leap year."""
        resolver = DateResolver(reference_date=date(2024, 2, 29))
        assert resolver.resolve("-4Y") == date(2020, 2, 29)


class TestFiscalYears:
    """Tests for fiscal year expressions (FY0, FY-1, FY2024)."""

    def test_resolve_fy0_december_end(self):
        """FY0 with Dec year-end should give Dec 31 of current FY."""
        # Jan 21, 2026 is in FY2026 (Dec year-end)
        resolver = DateResolver(reference_date=date(2026, 1, 21), fiscal_year_end_month=12)
        assert resolver.resolve("FY0") == date(2026, 12, 31)

    def test_resolve_fy_minus_1(self):
        """FY-1 should give prior fiscal year end."""
        resolver = DateResolver(reference_date=date(2026, 1, 21), fiscal_year_end_month=12)
        assert resolver.resolve("FY-1") == date(2025, 12, 31)

    def test_resolve_fy1_next_year(self):
        """FY1 should give next fiscal year end."""
        resolver = DateResolver(reference_date=date(2026, 1, 21), fiscal_year_end_month=12)
        assert resolver.resolve("FY1") == date(2027, 12, 31)

    def test_resolve_fy_absolute(self):
        """FY2024 should give FY2024 end date."""
        resolver = DateResolver(reference_date=date(2026, 1, 21), fiscal_year_end_month=12)
        assert resolver.resolve("FY2024") == date(2024, 12, 31)


class TestFiscalQuarters:
    """Tests for fiscal quarter expressions (FQ0, FQ-1, FQ12024)."""

    def test_resolve_fq0(self):
        """FQ0 should give current quarter end."""
        # Jan 21, 2026 is in Q1 (Dec year-end)
        resolver = DateResolver(reference_date=date(2026, 1, 21), fiscal_year_end_month=12)
        result = resolver.resolve("FQ0")
        # Q1 ends March 31
        assert result == date(2026, 3, 31)

    def test_resolve_fq_minus_1(self):
        """FQ-1 should give prior quarter end."""
        resolver = DateResolver(reference_date=date(2026, 1, 21), fiscal_year_end_month=12)
        result = resolver.resolve("FQ-1")
        # Prior quarter is Q4 of prior year, ends Dec 31
        assert result == date(2025, 12, 31)

    def test_resolve_fq1_next_quarter(self):
        """FQ1 should give next quarter end."""
        resolver = DateResolver(reference_date=date(2026, 1, 21), fiscal_year_end_month=12)
        result = resolver.resolve("FQ1")
        # Next quarter is Q2, ends June 30
        assert result == date(2026, 6, 30)

    def test_resolve_fq_absolute(self):
        """FQ12024 should give Q1 2024 end date."""
        resolver = DateResolver(reference_date=date(2026, 1, 21), fiscal_year_end_month=12)
        result = resolver.resolve("FQ12024")
        assert result == date(2024, 3, 31)


class TestAbsoluteDates:
    """Tests for absolute date expressions (YYYY-MM-DD)."""

    def test_resolve_absolute_date(self):
        """Absolute date should be parsed correctly."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.resolve("2024-06-15") == date(2024, 6, 15)

    def test_resolve_absolute_date_unchanged(self):
        """Absolute date resolution shouldn't depend on reference."""
        resolver1 = DateResolver(reference_date=date(2026, 1, 21))
        resolver2 = DateResolver(reference_date=date(2020, 6, 15))
        assert resolver1.resolve("2024-06-15") == resolver2.resolve("2024-06-15")

    def test_resolve_invalid_date(self):
        """Invalid date should return None."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.resolve("2024-02-30") is None  # Feb 30 doesn't exist


class TestNormalize:
    """Tests for the normalize() method."""

    def test_normalize_relative_to_absolute(self):
        """-1D should normalize to YYYY-MM-DD format."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.normalize("-1D") == "2026-01-20"

    def test_normalize_absolute_unchanged(self):
        """Absolute date should remain unchanged."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.normalize("2024-06-15") == "2024-06-15"

    def test_normalize_invalid_unchanged(self):
        """Invalid expression should return original."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.normalize("hello") == "hello"


class TestEquivalence:
    """Tests for the are_equivalent() method."""

    def test_equivalent_relative_and_absolute(self):
        """-1D should be equivalent to the resolved absolute date."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.are_equivalent("-1D", "2026-01-20") is True

    def test_not_equivalent_different_days(self):
        """-1D should not be equivalent to a different day."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.are_equivalent("-1D", "2026-01-19") is False

    def test_equivalent_same_expression(self):
        """Same expression should be equivalent to itself."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.are_equivalent("-1D", "-1D") is True

    def test_not_equivalent_invalid(self):
        """Invalid expressions should not be equivalent."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.are_equivalent("-1D", "hello") is False
        assert resolver.are_equivalent("hello", "-1D") is False


class TestValidation:
    """Tests for the is_valid_temporal_expr() method."""

    def test_valid_relative_day(self):
        """-1D should be valid."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.is_valid_temporal_expr("-1D") is True

    def test_valid_relative_month(self):
        """-3M should be valid."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.is_valid_temporal_expr("-3M") is True

    def test_valid_relative_year(self):
        """-1Y should be valid."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.is_valid_temporal_expr("-1Y") is True

    def test_valid_fiscal_year(self):
        """FY0 should be valid."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.is_valid_temporal_expr("FY0") is True

    def test_valid_fiscal_quarter(self):
        """FQ1 should be valid."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.is_valid_temporal_expr("FQ1") is True

    def test_valid_absolute_date(self):
        """2024-06-15 should be valid."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.is_valid_temporal_expr("2024-06-15") is True

    def test_invalid_text(self):
        """Plain text should be invalid."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.is_valid_temporal_expr("hello") is False

    def test_invalid_empty(self):
        """Empty string should be invalid."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.is_valid_temporal_expr("") is False

    def test_invalid_partial_pattern(self):
        """Partial patterns should be invalid."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.is_valid_temporal_expr("-1") is False
        assert resolver.is_valid_temporal_expr("D") is False
        assert resolver.is_valid_temporal_expr("FY") is False


class TestDefaultReferenceDate:
    """Tests for default reference date behavior."""

    def test_default_uses_today(self):
        """Without reference date, should use today."""
        resolver = DateResolver()
        assert resolver.reference_date == date.today()

    def test_default_resolves_from_today(self):
        """0D with default reference should be today."""
        resolver = DateResolver()
        assert resolver.resolve("0D") == date.today()


class TestEdgeCases:
    """Tests for edge cases and boundary conditions."""

    def test_leap_year_detection(self):
        """Leap year logic should be correct."""
        # 2024 is a leap year (divisible by 4, not 100)
        assert DateResolver._is_leap_year(2024) is True
        # 2023 is not
        assert DateResolver._is_leap_year(2023) is False
        # 2000 is (divisible by 400)
        assert DateResolver._is_leap_year(2000) is True
        # 1900 is not (divisible by 100 but not 400)
        assert DateResolver._is_leap_year(1900) is False

    def test_days_in_month(self):
        """Days in month calculation should be correct."""
        # Regular months
        assert DateResolver._days_in_month(2024, 1) == 31  # Jan
        assert DateResolver._days_in_month(2024, 4) == 30  # Apr
        # February in leap/non-leap
        assert DateResolver._days_in_month(2024, 2) == 29  # Leap
        assert DateResolver._days_in_month(2023, 2) == 28  # Non-leap

    def test_whitespace_handling(self):
        """Expressions with whitespace should be handled."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.resolve(" -1D ") == date(2026, 1, 20)
        assert resolver.is_valid_temporal_expr(" -1D ") is True

    def test_none_input(self):
        """None/empty input should be handled gracefully."""
        resolver = DateResolver(reference_date=date(2026, 1, 21))
        assert resolver.resolve("") is None
        assert resolver.resolve(None) is None
        assert resolver.normalize("") == ""
        assert resolver.is_valid_temporal_expr("") is False
