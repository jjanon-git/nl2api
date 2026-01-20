"""
Temporal Generator - Generates time series and temporal variant test cases.

Target: ~2,500 test cases
- Core queries × temporal patterns = 100 queries × 25 patterns = 2,500
"""

import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from .base_generator import BaseGenerator, TestCase


class TemporalGenerator(BaseGenerator):
    """Generator for temporal variant test cases."""

    def __init__(self, data_dir: Path):
        super().__init__(data_dir)
        self.category = "temporal"

    def _build_tool_call(self, ticker: str, fields: List[str],
                         start: str, end: str,
                         freq: Optional[str] = None) -> Dict:
        """Build a tool call with temporal parameters."""
        call = {
            "function": "get_data",
            "arguments": {
                "tickers": ticker,
                "fields": fields,
                "start": start,
                "end": end
            }
        }
        if freq:
            call["arguments"]["freq"] = freq
        return call

    def _generate_historical_price_data(self) -> List[TestCase]:
        """Generate historical price time series queries."""
        test_cases = []

        tickers = self._get_all_tickers(count=30)

        # Time ranges with natural language
        time_ranges = [
            {"start": "-1M", "end": "0D", "freq": "D", "nl": "for the last month"},
            {"start": "-3M", "end": "0D", "freq": "D", "nl": "for the past 3 months"},
            {"start": "-6M", "end": "0D", "freq": "D", "nl": "for the past 6 months"},
            {"start": "-1Y", "end": "0D", "freq": "D", "nl": "for the past year"},
            {"start": "-1Y", "end": "0D", "freq": "W", "nl": "weekly for the past year"},
            {"start": "-5Y", "end": "0D", "freq": "W", "nl": "weekly for the past 5 years"},
            {"start": "-5Y", "end": "0D", "freq": "M", "nl": "monthly for the past 5 years"},
            {"start": "-10Y", "end": "0D", "freq": "M", "nl": "monthly for the past decade"},
            {"start": "-10Y", "end": "0D", "freq": "Y", "nl": "annually for the last 10 years"},
            {"start": "2024-01-01", "end": "2024-12-31", "freq": "D", "nl": "for all of 2024"},
            {"start": "2023-01-01", "end": "2023-12-31", "freq": "D", "nl": "for all of 2023"},
            {"start": "2020-01-01", "end": "0D", "freq": "W", "nl": "weekly since 2020"},
            {"start": "-0Y", "end": "0D", "freq": "D", "nl": "year to date"},
        ]

        query_templates = [
            "Get {company}'s stock price {time_range}",
            "Show {company}'s daily prices {time_range}",
            "{company} {frequency} prices {time_range}",
            "Get {company}'s closing price history {time_range}",
        ]

        for ticker_info in tickers:
            ticker_symbol = ticker_info.get("symbol", "")
            company_name = ticker_info.get("name", "Company")

            for time_range in time_ranges:
                template = random.choice(query_templates)
                freq_name = {"D": "daily", "W": "weekly", "M": "monthly", "Y": "annual"}.get(
                    time_range["freq"], "daily"
                )

                nl_query = template.format(
                    company=company_name,
                    time_range=time_range["nl"],
                    frequency=freq_name
                )

                tool_call = self._build_tool_call(
                    ticker=ticker_symbol,
                    fields=["P"],
                    start=time_range["start"],
                    end=time_range["end"],
                    freq=time_range["freq"]
                )

                complexity = self._calculate_complexity(
                    num_tickers=1,
                    num_fields=1,
                    is_time_series=True
                )

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=[tool_call],
                    category=self.category,
                    subcategory="historical_price",
                    complexity=complexity,
                    tags=["price", "time_series", time_range["freq"]],
                    metadata={
                        "ticker": ticker_symbol,
                        "time_range": time_range,
                        "frequency": time_range["freq"]
                    }
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_ohlcv_time_series(self) -> List[TestCase]:
        """Generate OHLCV time series queries."""
        test_cases = []

        tickers = self._get_all_tickers(count=20)

        time_ranges = [
            {"start": "-1W", "end": "0D", "freq": "D", "nl": "for last week"},
            {"start": "-1M", "end": "0D", "freq": "D", "nl": "for the last month"},
            {"start": "-3M", "end": "0D", "freq": "D", "nl": "for the past 3 months"},
            {"start": "2024-01-01", "end": "2024-01-31", "freq": "D", "nl": "for January 2024"},
        ]

        queries = [
            {"nl": "Get {company}'s OHLC data {time_range}", "fields": ["PO", "PH", "PL", "P"]},
            {"nl": "Show {company}'s daily OHLCV {time_range}", "fields": ["PO", "PH", "PL", "P", "VO"]},
            {"nl": "{company} open, high, low, close, volume {time_range}", "fields": ["PO", "PH", "PL", "P", "VO"]},
        ]

        for ticker_info in tickers:
            ticker_symbol = ticker_info.get("symbol", "")
            company_name = ticker_info.get("name", "Company")

            for time_range in time_ranges:
                for query in queries:
                    nl_query = query["nl"].format(
                        company=company_name,
                        time_range=time_range["nl"]
                    )

                    tool_call = self._build_tool_call(
                        ticker=ticker_symbol,
                        fields=query["fields"],
                        start=time_range["start"],
                        end=time_range["end"],
                        freq=time_range["freq"]
                    )

                    complexity = self._calculate_complexity(
                        num_tickers=1,
                        num_fields=len(query["fields"]),
                        is_time_series=True
                    )

                    test_case = self._create_test_case(
                        nl_query=nl_query,
                        tool_calls=[tool_call],
                        category=self.category,
                        subcategory="ohlcv",
                        complexity=complexity,
                        tags=["ohlcv", "time_series"],
                        metadata={
                            "ticker": ticker_symbol,
                            "fields": query["fields"]
                        }
                    )

                    if test_case:
                        test_cases.append(test_case)

        return test_cases

    def _generate_fundamental_time_series(self) -> List[TestCase]:
        """Generate fundamental data time series queries."""
        test_cases = []

        tickers = self._get_all_tickers(count=25)

        # Fundamental metrics with appropriate frequencies
        metrics = [
            {"field": "WC01001", "name": "revenue", "freq": ["Y", "Q"]},
            {"field": "WC01751", "name": "net income", "freq": ["Y", "Q"]},
            {"field": "WC18198", "name": "EBITDA", "freq": ["Y", "Q"]},
            {"field": "WC05201", "name": "EPS", "freq": ["Y", "Q"]},
            {"field": "WC08301", "name": "ROE", "freq": ["Y"]},
            {"field": "WC08326", "name": "ROA", "freq": ["Y"]},
            {"field": "WC02999", "name": "total assets", "freq": ["Y", "Q"]},
            {"field": "WC03255", "name": "total debt", "freq": ["Y", "Q"]},
            {"field": "WC04860", "name": "operating cash flow", "freq": ["Y", "Q"]},
            {"field": "WC05101", "name": "dividends per share", "freq": ["Y"]},
        ]

        time_ranges = [
            {"start": "-5Y", "end": "0D", "nl": "for the last 5 years"},
            {"start": "-10Y", "end": "0D", "nl": "for the past decade"},
            {"start": "2020-01-01", "end": "0D", "nl": "since 2020"},
            {"start": "-3Y", "end": "0D", "nl": "for the last 3 years"},
        ]

        for ticker_info in tickers:
            ticker_symbol = ticker_info.get("symbol", "")
            company_name = ticker_info.get("name", "Company")

            for metric in metrics:
                for time_range in time_ranges:
                    for freq in metric["freq"]:
                        freq_name = "annual" if freq == "Y" else "quarterly"

                        nl_query = f"Get {company_name}'s {freq_name} {metric['name']} {time_range['nl']}"

                        tool_call = self._build_tool_call(
                            ticker=ticker_symbol,
                            fields=[metric["field"]],
                            start=time_range["start"],
                            end=time_range["end"],
                            freq=freq
                        )

                        complexity = self._calculate_complexity(
                            num_tickers=1,
                            num_fields=1,
                            is_time_series=True
                        )

                        test_case = self._create_test_case(
                            nl_query=nl_query,
                            tool_calls=[tool_call],
                            category=self.category,
                            subcategory="fundamental_history",
                            complexity=complexity,
                            tags=["fundamentals", "time_series", freq_name],
                            metadata={
                                "ticker": ticker_symbol,
                                "metric": metric["name"],
                                "field": metric["field"],
                                "frequency": freq
                            }
                        )

                        if test_case:
                            test_cases.append(test_case)

        return test_cases

    def _generate_fiscal_period_queries(self) -> List[TestCase]:
        """Generate queries with fiscal period parameters."""
        test_cases = []

        tickers = self._get_all_tickers(count=20)

        fiscal_periods = [
            {"period": "FY2024", "nl": "for fiscal 2024"},
            {"period": "FY2023", "nl": "for fiscal 2023"},
            {"period": "FY2022", "nl": "for fiscal 2022"},
            {"period": "FQ12024", "nl": "for Q1 2024"},
            {"period": "FQ42024", "nl": "for Q4 2024"},
        ]

        metrics = [
            {"field": "WC01001", "name": "revenue"},
            {"field": "WC01751", "name": "net income"},
            {"field": "WC05201", "name": "EPS"},
        ]

        for ticker_info in tickers:
            ticker_symbol = ticker_info.get("symbol", "")
            company_name = ticker_info.get("name", "Company")

            for period in fiscal_periods:
                for metric in metrics:
                    nl_query = f"What was {company_name}'s {metric['name']} {period['nl']}?"

                    # This would map to a period-specific query
                    tool_call = {
                        "function": "get_data",
                        "arguments": {
                            "tickers": ticker_symbol,
                            "fields": [metric["field"]],
                            "period": period["period"]
                        }
                    }

                    complexity = self._calculate_complexity(num_tickers=1, num_fields=1)

                    test_case = self._create_test_case(
                        nl_query=nl_query,
                        tool_calls=[tool_call],
                        category=self.category,
                        subcategory="fiscal_period",
                        complexity=complexity,
                        tags=["fiscal", "period_specific"],
                        metadata={
                            "ticker": ticker_symbol,
                            "period": period["period"],
                            "metric": metric["name"]
                        }
                    )

                    if test_case:
                        test_cases.append(test_case)

        return test_cases

    def _generate_ratio_time_series(self) -> List[TestCase]:
        """Generate financial ratio time series queries."""
        test_cases = []

        tickers = self._get_all_tickers(count=15)

        ratio_sets = [
            {
                "fields": ["WC08301", "WC08326", "WC08376"],
                "name": "profitability ratios (ROE, ROA, ROIC)"
            },
            {
                "fields": ["WC08316", "WC08366"],
                "name": "margin profile (operating and net margins)"
            },
            {
                "fields": ["WC08106", "WC08101"],
                "name": "liquidity ratios (current and quick ratio)"
            },
            {
                "fields": ["WC08221", "WC08224"],
                "name": "leverage ratios (debt to capital, debt to equity)"
            },
        ]

        time_ranges = [
            {"start": "-5Y", "end": "0D", "freq": "Y", "nl": "over the last 5 years"},
            {"start": "-10Y", "end": "0D", "freq": "Y", "nl": "over the past decade"},
        ]

        for ticker_info in tickers:
            ticker_symbol = ticker_info.get("symbol", "")
            company_name = ticker_info.get("name", "Company")

            for ratio_set in ratio_sets:
                for time_range in time_ranges:
                    nl_query = f"Show {company_name}'s {ratio_set['name']} {time_range['nl']}"

                    tool_call = self._build_tool_call(
                        ticker=ticker_symbol,
                        fields=ratio_set["fields"],
                        start=time_range["start"],
                        end=time_range["end"],
                        freq=time_range["freq"]
                    )

                    complexity = self._calculate_complexity(
                        num_tickers=1,
                        num_fields=len(ratio_set["fields"]),
                        is_time_series=True
                    )

                    test_case = self._create_test_case(
                        nl_query=nl_query,
                        tool_calls=[tool_call],
                        category=self.category,
                        subcategory="ratio_history",
                        complexity=complexity,
                        tags=["ratios", "time_series"],
                        metadata={
                            "ticker": ticker_symbol,
                            "ratio_set": ratio_set["name"]
                        }
                    )

                    if test_case:
                        test_cases.append(test_case)

        return test_cases

    def generate(self) -> List[TestCase]:
        """Generate all temporal test cases."""
        test_cases = []

        # Historical price data
        test_cases.extend(self._generate_historical_price_data())

        # OHLCV time series
        test_cases.extend(self._generate_ohlcv_time_series())

        # Fundamental time series
        test_cases.extend(self._generate_fundamental_time_series())

        # Fiscal period queries
        test_cases.extend(self._generate_fiscal_period_queries())

        # Ratio time series
        test_cases.extend(self._generate_ratio_time_series())

        return test_cases
