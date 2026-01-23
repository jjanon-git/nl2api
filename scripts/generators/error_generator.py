"""
Error Generator - Generates error scenario test cases.

Target: ~500 test cases
- Invalid ticker errors (5 APIs × 20 tickers)
- Invalid field errors
- Missing data scenarios
- Future date errors
- Permission/access errors
"""

import random
from pathlib import Path

from .base_generator import BaseGenerator, TestCase


class ErrorGenerator(BaseGenerator):
    """Generator for error scenario test cases."""

    def __init__(self, data_dir: Path):
        super().__init__(data_dir)
        self.category = "errors"

    def _generate_invalid_ticker_errors(self) -> list[TestCase]:
        """Generate invalid ticker error scenarios."""
        test_cases = []

        invalid_tickers = [
            {"ticker": "INVALID123", "type": "nonexistent"},
            {"ticker": "U:XXXXX", "type": "nonexistent_us"},
            {"ticker": "ZZZZZZ", "type": "nonexistent_uk"},
            {"ticker": "12345", "type": "numeric"},
            {"ticker": "U:", "type": "empty_prefix"},
            {"ticker": "", "type": "empty"},
            {"ticker": "A A A", "type": "spaces"},
            {"ticker": "AAPL@", "type": "special_char"},
            {"ticker": "U:ENRN", "type": "delisted_bankrupt"},
            {"ticker": "U:LEH", "type": "delisted_bankrupt"},
            {"ticker": "ABCDEFG", "type": "too_long"},
            {"ticker": "X", "type": "single_char"},
            {"ticker": "D:XXXXX", "type": "nonexistent_de"},
            {"ticker": "J:9999", "type": "nonexistent_jp"},
            {"ticker": "H:0000", "type": "nonexistent_hk"},
            {"ticker": "F:ZZZZZ", "type": "nonexistent_fr"},
            {"ticker": "AAPL;DROP", "type": "sql_injection"},
            {"ticker": "<script>", "type": "xss_attempt"},
            {"ticker": "NULL", "type": "reserved_word"},
            {"ticker": "undefined", "type": "reserved_word2"},
        ]

        fields = ["P", "MV", "WC01001", "TR.Revenue", "PE", "DY", "WC08301", "EPS"]

        templates = [
            "What is {ticker}'s stock price?",
            "Get {ticker} revenue",
            "Show me {ticker} market cap",
            "Get data for {ticker}",
        ]

        for invalid in invalid_tickers:
            for field in fields:
                template = random.choice(templates)
                nl_query = template.format(ticker=invalid["ticker"])

                tool_call = {
                    "tool_name": "get_data",
                    "arguments": {
                        "tickers": invalid["ticker"],
                        "fields": [field],
                        "start": "0D",
                        "end": "0D",
                        "kind": 0,
                    },
                }

                expected_error = {
                    "error_type": "invalid_ticker",
                    "error_subtype": invalid["type"],
                    "message_pattern": "Instrument not found|Invalid ticker|No data available",
                }

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=[tool_call],
                    category=self.category,
                    subcategory="invalid_ticker",
                    complexity=1,
                    tags=["error", "invalid_ticker", invalid["type"]],
                    metadata={
                        "ticker": invalid["ticker"],
                        "error_type": invalid["type"],
                        "expected_error": expected_error,
                    },
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_invalid_field_errors(self) -> list[TestCase]:
        """Generate invalid field code error scenarios."""
        test_cases = []

        valid_tickers = ["@AAPL", "U:MSFT", "U:GOOGL", "U:AMZN", "U:META", "U:NVDA"]

        invalid_fields = [
            {"field": "INVALID_FIELD", "type": "nonexistent"},
            {"field": "WC99999", "type": "invalid_wc"},
            {"field": "TR.InvalidField", "type": "invalid_tr"},
            {"field": "123456", "type": "numeric"},
            {"field": "", "type": "empty"},
            {"field": "P E", "type": "spaces"},
            {"field": "PE@RATIO", "type": "special_char"},
            {"field": "WC0", "type": "short_wc"},
            {"field": "WC1234567890", "type": "long_wc"},
            {"field": "TR.", "type": "incomplete_tr"},
            {"field": "TR.123", "type": "numeric_tr"},
            {"field": "NULL", "type": "reserved_word"},
            {"field": "SELECT", "type": "sql_keyword"},
            {"field": "DROP TABLE", "type": "injection"},
            {"field": "*", "type": "wildcard"},
            {"field": "P,MV", "type": "multiple_fields"},
            {"field": "WC01001.Q", "type": "period_suffix"},
            {"field": "TR.Revenue.FY", "type": "extra_period"},
        ]

        templates = [
            "Get {company}'s {field}",
            "What is {company}'s {field}?",
            "Show me {field} for {company}",
        ]

        for ticker in valid_tickers:
            for invalid in invalid_fields:
                template = random.choice(templates)
                nl_query = template.format(company=ticker, field=invalid["field"])

                tool_call = {
                    "tool_name": "get_data",
                    "arguments": {
                        "tickers": ticker,
                        "fields": [invalid["field"]],
                        "start": "0D",
                        "end": "0D",
                        "kind": 0,
                    },
                }

                expected_error = {
                    "error_type": "invalid_field",
                    "error_subtype": invalid["type"],
                    "message_pattern": "Data type not recognized|Invalid field|Unknown field code",
                }

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=[tool_call],
                    category=self.category,
                    subcategory="invalid_field",
                    complexity=1,
                    tags=["error", "invalid_field", invalid["type"]],
                    metadata={
                        "ticker": ticker,
                        "field": invalid["field"],
                        "error_type": invalid["type"],
                        "expected_error": expected_error,
                    },
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_future_date_errors(self) -> list[TestCase]:
        """Generate future date error scenarios."""
        test_cases = []

        tickers = ["@AAPL", "U:MSFT", "U:GOOGL", "U:AMZN"]

        future_dates = [
            {"date": "2030-01-01", "nl": "January 2030"},
            {"date": "2050-12-31", "nl": "December 2050"},
            {"date": "2100-06-15", "nl": "June 2100"},
        ]

        fields = ["P", "WC01001", "MV"]

        for ticker in tickers:
            company_name = ticker.replace("@", "").replace("U:", "")

            for future in future_dates:
                for field in fields:
                    nl_query = f"What was {company_name}'s stock price on {future['nl']}?"

                    tool_call = {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": ticker,
                            "fields": [field],
                            "start": future["date"],
                            "end": future["date"],
                            "kind": 0,
                        },
                    }

                    expected_error = {
                        "error_type": "future_date",
                        "message_pattern": "Date in future|No data available|Invalid date range",
                    }

                    test_case = self._create_test_case(
                        nl_query=nl_query,
                        tool_calls=[tool_call],
                        category=self.category,
                        subcategory="future_date",
                        complexity=1,
                        tags=["error", "future_date"],
                        metadata={
                            "ticker": ticker,
                            "date": future["date"],
                            "expected_error": expected_error,
                        },
                    )

                    if test_case:
                        test_cases.append(test_case)

        return test_cases

    def _generate_missing_data_errors(self) -> list[TestCase]:
        """Generate missing data scenarios."""
        test_cases = []

        # Industry-specific field mismatches
        mismatches = [
            {
                "ticker": "U:JPM",
                "name": "JPMorgan",
                "field": "WC02101",
                "field_name": "inventory",
                "reason": "Banks don't have inventory",
            },
            {
                "ticker": "U:TSLA",
                "name": "Tesla",
                "field": "WC01002",
                "field_name": "premiums earned",
                "reason": "Tesla is not an insurance company",
            },
            {
                "ticker": "U:PG",
                "name": "Procter & Gamble",
                "field": "WC01076",
                "field_name": "net interest income",
                "reason": "P&G is not a bank",
            },
        ]

        for mismatch in mismatches:
            nl_query = f"What is {mismatch['name']}'s {mismatch['field_name']}?"

            tool_call = {
                "tool_name": "get_data",
                "arguments": {
                    "tickers": mismatch["ticker"],
                    "fields": [mismatch["field"]],
                    "start": "0D",
                    "end": "0D",
                    "kind": 0,
                },
            }

            expected_behavior = {
                "behavior": "null_or_warning",
                "reason": mismatch["reason"],
                "message_pattern": "Not applicable|No data|N/A",
            }

            test_case = self._create_test_case(
                nl_query=nl_query,
                tool_calls=[tool_call],
                category=self.category,
                subcategory="missing_data",
                complexity=2,
                tags=["error", "missing_data", "industry_mismatch"],
                metadata={
                    "ticker": mismatch["ticker"],
                    "field": mismatch["field"],
                    "reason": mismatch["reason"],
                    "expected_behavior": expected_behavior,
                },
            )

            if test_case:
                test_cases.append(test_case)

        return test_cases

    def _generate_private_company_errors(self) -> list[TestCase]:
        """Generate private company query errors."""
        test_cases = []

        private_companies = [
            {"name": "SpaceX", "ticker": "SPACEX"},
            {"name": "Stripe", "ticker": "STRIPE"},
            {"name": "Databricks", "ticker": "DATABRICKS"},
        ]

        fields = ["P", "WC01001", "MV", "TR.Revenue"]

        for company in private_companies:
            for field in fields:
                nl_query = f"What is {company['name']}'s stock price?"

                tool_call = {
                    "tool_name": "get_data",
                    "arguments": {
                        "tickers": company["ticker"],
                        "fields": [field],
                        "start": "0D",
                        "end": "0D",
                        "kind": 0,
                    },
                }

                expected_error = {
                    "error_type": "private_company",
                    "message_pattern": "No public filings|Instrument not found|Private company",
                }

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=[tool_call],
                    category=self.category,
                    subcategory="private_company",
                    complexity=1,
                    tags=["error", "private_company"],
                    metadata={
                        "company": company["name"],
                        "ticker": company["ticker"],
                        "expected_error": expected_error,
                    },
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_date_range_errors(self) -> list[TestCase]:
        """Generate invalid date range errors."""
        test_cases = []

        tickers = ["@AAPL", "U:MSFT"]

        invalid_ranges = [
            {"start": "2024-12-31", "end": "2024-01-01", "type": "reversed"},
            {"start": "1800-01-01", "end": "1800-12-31", "type": "too_old"},
            {"start": "invalid", "end": "0D", "type": "invalid_format"},
        ]

        for ticker in tickers:
            for date_range in invalid_ranges:
                nl_query = f"Get {ticker.replace('U:', '').replace('@', '')} price from {date_range['start']} to {date_range['end']}"

                tool_call = {
                    "tool_name": "get_data",
                    "arguments": {
                        "tickers": ticker,
                        "fields": ["P"],
                        "start": date_range["start"],
                        "end": date_range["end"],
                    },
                }

                expected_error = {
                    "error_type": "invalid_date_range",
                    "error_subtype": date_range["type"],
                    "message_pattern": "Invalid date|Start date after end|No data available",
                }

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=[tool_call],
                    category=self.category,
                    subcategory="invalid_date_range",
                    complexity=1,
                    tags=["error", "date_range", date_range["type"]],
                    metadata={
                        "ticker": ticker,
                        "start": date_range["start"],
                        "end": date_range["end"],
                        "expected_error": expected_error,
                    },
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_delisted_stock_errors(self) -> list[TestCase]:
        """Generate delisted stock query scenarios."""
        test_cases = []

        delisted = self.edge_cases.get("delisted", {}).get("tickers", [])

        for stock in delisted:
            nl_query = f"What is {stock['name']}'s current stock price?"

            tool_call = {
                "tool_name": "get_data",
                "arguments": {
                    "tickers": stock["symbol"],
                    "fields": ["P"],
                    "start": "0D",
                    "end": "0D",
                    "kind": 0,
                },
            }

            expected_behavior = {
                "behavior": "historical_only_or_error",
                "delisting_date": stock.get("delisting_date"),
                "reason": stock.get("reason"),
                "message_pattern": "Delisted|No current data|Historical data only",
            }

            test_case = self._create_test_case(
                nl_query=nl_query,
                tool_calls=[tool_call],
                category=self.category,
                subcategory="delisted_stock",
                complexity=2,
                tags=["error", "delisted", stock.get("reason", "unknown")],
                metadata={
                    "company": stock["name"],
                    "ticker": stock["symbol"],
                    "delisting_date": stock.get("delisting_date"),
                    "reason": stock.get("reason"),
                    "expected_behavior": expected_behavior,
                },
            )

            if test_case:
                test_cases.append(test_case)

        return test_cases

    def generate(self) -> list[TestCase]:
        """Generate all error test cases."""
        test_cases = []

        # Invalid ticker errors
        test_cases.extend(self._generate_invalid_ticker_errors())

        # Invalid field errors
        test_cases.extend(self._generate_invalid_field_errors())

        # Future date errors
        test_cases.extend(self._generate_future_date_errors())

        # Missing data scenarios
        test_cases.extend(self._generate_missing_data_errors())

        # Private company errors
        test_cases.extend(self._generate_private_company_errors())

        # Date range errors
        test_cases.extend(self._generate_date_range_errors())

        # Delisted stock scenarios
        test_cases.extend(self._generate_delisted_stock_errors())

        return test_cases
