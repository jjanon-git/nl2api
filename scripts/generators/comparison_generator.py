"""
Comparison Generator - Generates multi-ticker comparison test cases.

Target: ~3,000 test cases
- 150 pairs × 20 metrics = 3,000
"""

import random
from pathlib import Path

from .base_generator import BaseGenerator, TestCase


class ComparisonGenerator(BaseGenerator):
    """Generator for multi-ticker comparison test cases."""

    def __init__(self, data_dir: Path):
        super().__init__(data_dir)
        self.category = "comparisons"

    def _build_tool_call(
        self, tickers: list[str], fields: list[str], start: str = "0D", end: str = "0D"
    ) -> dict:
        """Build a tool call for comparison query."""
        return {
            "tool_name": "get_data",
            "arguments": {
                "tickers": ",".join(tickers),
                "fields": fields,
                "start": start,
                "end": end,
                "kind": 0,
            },
        }

    def _generate_two_stock_comparisons(self) -> list[TestCase]:
        """Generate 2-stock comparison queries."""
        test_cases = []

        # Get comparison pairs from data
        pair_categories = [
            "tech_giants",
            "financials",
            "healthcare",
            "consumer",
            "energy",
            "industrials",
            "auto",
            "international",
        ]

        metrics = [
            {"field": "P", "name": "stock price"},
            {"field": "MV", "name": "market cap"},
            {"field": "PE", "name": "PE ratio"},
            {"field": "DY", "name": "dividend yield"},
            {"field": "WC01001", "name": "revenue"},
            {"field": "WC01751", "name": "net income"},
            {"field": "WC08301", "name": "ROE"},
            {"field": "WC08326", "name": "ROA"},
            {"field": "WC05201", "name": "EPS"},
            {"field": "WC18198", "name": "EBITDA"},
            {"field": "WC08316", "name": "operating margin"},
            {"field": "WC08366", "name": "net margin"},
            {"field": "WC03255", "name": "total debt"},
            {"field": "WC08224", "name": "debt to equity"},
            {"field": "WC04860", "name": "operating cash flow"},
        ]

        templates = [
            "Compare {metric} for {company_a} and {company_b}",
            "{company_a} vs {company_b} {metric}",
            "Which has higher {metric}: {company_a} or {company_b}?",
            "Get {metric} for {company_a} vs {company_b}",
            "{metric} comparison: {company_a} versus {company_b}",
        ]

        for category in pair_categories:
            pairs = self.comparison_pairs.get(category, {}).get("pairs", [])

            for pair in pairs:
                ticker_a = pair.get("a", "")
                ticker_b = pair.get("b", "")
                pair_name = pair.get("name", f"{ticker_a} vs {ticker_b}")

                # Get company names from tickers
                name_a = pair_name.split(" vs ")[0] if " vs " in pair_name else ticker_a
                name_b = pair_name.split(" vs ")[1] if " vs " in pair_name else ticker_b

                for metric in metrics:
                    template = random.choice(templates)
                    nl_query = template.format(
                        company_a=name_a, company_b=name_b, metric=metric["name"]
                    )

                    tool_call = self._build_tool_call(
                        tickers=[ticker_a, ticker_b], fields=[metric["field"]]
                    )

                    complexity = self._calculate_complexity(num_tickers=2, num_fields=1)

                    test_case = self._create_test_case(
                        nl_query=nl_query,
                        tool_calls=[tool_call],
                        category=self.category,
                        subcategory="two_stock",
                        complexity=complexity,
                        tags=["comparison", "two_stock", category],
                        metadata={
                            "tickers": [ticker_a, ticker_b],
                            "metric": metric["name"],
                            "sector": category,
                        },
                    )

                    if test_case:
                        test_cases.append(test_case)

        return test_cases

    def _generate_group_comparisons(self) -> list[TestCase]:
        """Generate multi-stock group comparison queries."""
        test_cases = []

        pair_categories = ["tech_giants", "financials", "healthcare", "consumer", "energy"]

        metrics = [
            {"field": "MV", "name": "market cap"},
            {"field": "PE", "name": "PE ratio"},
            {"field": "WC01001", "name": "revenue"},
            {"field": "WC01751", "name": "net income"},
            {"field": "WC08301", "name": "ROE"},
            {"field": "DY", "name": "dividend yield"},
            {"field": "WC05201", "name": "EPS"},
        ]

        for category in pair_categories:
            groups = self.comparison_pairs.get(category, {}).get("group_comparisons", [])

            for group in groups:
                companies = group.get("companies", [])
                group_name = group.get("name", "companies")

                if len(companies) < 3:
                    continue

                for metric in metrics:
                    nl_query = f"Compare {metric['name']} for {group_name}"

                    tool_call = self._build_tool_call(tickers=companies, fields=[metric["field"]])

                    complexity = self._calculate_complexity(
                        num_tickers=len(companies), num_fields=1
                    )

                    test_case = self._create_test_case(
                        nl_query=nl_query,
                        tool_calls=[tool_call],
                        category=self.category,
                        subcategory="group_comparison",
                        complexity=complexity,
                        tags=["comparison", "group", category],
                        metadata={
                            "tickers": companies,
                            "group_name": group_name,
                            "metric": metric["name"],
                        },
                    )

                    if test_case:
                        test_cases.append(test_case)

        return test_cases

    def _generate_sector_comparisons(self) -> list[TestCase]:
        """Generate sector-based comparison queries."""
        test_cases = []

        metrics = [
            {"field": "MV", "name": "market cap"},
            {"field": "WC01001", "name": "revenue"},
            {"field": "WC01751", "name": "net income"},
            {"field": "WC08301", "name": "ROE"},
            {"field": "PE", "name": "PE ratio"},
        ]

        for sector, tickers in self.us_by_sector.get("sectors", {}).items():
            if len(tickers) < 2:
                continue

            ticker_symbols = [t.get("symbol", "") for t in tickers]

            for metric in metrics:
                nl_query = f"Compare {metric['name']} for {sector} companies"

                tool_call = self._build_tool_call(tickers=ticker_symbols, fields=[metric["field"]])

                complexity = self._calculate_complexity(
                    num_tickers=len(ticker_symbols), num_fields=1
                )

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=[tool_call],
                    category=self.category,
                    subcategory="sector_comparison",
                    complexity=complexity,
                    tags=["comparison", "sector", sector.lower().replace(" ", "_")],
                    metadata={
                        "sector": sector,
                        "tickers": ticker_symbols,
                        "metric": metric["name"],
                    },
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_index_comparisons(self) -> list[TestCase]:
        """Generate index comparison queries."""
        test_cases = []

        index_pairs = self.comparison_pairs.get("index_comparisons", {}).get("pairs", [])

        metrics = [
            {"field": "PI", "name": "index level"},
            {"field": "P", "name": "price"},
        ]

        time_ranges = [
            {"start": "-1Y", "end": "0D", "freq": "D", "nl": "over the past year"},
            {"start": "-5Y", "end": "0D", "freq": "W", "nl": "over the past 5 years"},
            {"start": "-0Y", "end": "0D", "freq": "D", "nl": "year to date"},
        ]

        for pair in index_pairs:
            index_a = pair.get("a", "")
            index_b = pair.get("b", "")
            pair_name = pair.get("name", f"{index_a} vs {index_b}")

            for time_range in time_ranges:
                for metric in metrics:
                    nl_query = f"Compare {pair_name} performance {time_range['nl']}"

                    tool_call = {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": f"{index_a},{index_b}",
                            "fields": [metric["field"]],
                            "start": time_range["start"],
                            "end": time_range["end"],
                            "freq": time_range["freq"],
                        },
                    }

                    complexity = self._calculate_complexity(
                        num_tickers=2, num_fields=1, is_time_series=True
                    )

                    test_case = self._create_test_case(
                        nl_query=nl_query,
                        tool_calls=[tool_call],
                        category=self.category,
                        subcategory="index_comparison",
                        complexity=complexity,
                        tags=["comparison", "index", "time_series"],
                        metadata={
                            "indices": [index_a, index_b],
                            "pair_name": pair_name,
                            "time_range": time_range["nl"],
                        },
                    )

                    if test_case:
                        test_cases.append(test_case)

        return test_cases

    def _generate_historical_comparisons(self) -> list[TestCase]:
        """Generate historical comparison queries."""
        test_cases = []

        pair_categories = ["tech_giants", "financials"]

        metrics = [
            {"field": "WC01001", "name": "revenue"},
            {"field": "P", "name": "stock price"},
            {"field": "WC08301", "name": "ROE"},
        ]

        time_ranges = [
            {"start": "-5Y", "end": "0D", "freq": "Y", "nl": "over the last 5 years"},
            {"start": "-10Y", "end": "0D", "freq": "Y", "nl": "over the past decade"},
            {"start": "-3Y", "end": "0D", "freq": "Q", "nl": "quarterly for the past 3 years"},
        ]

        for category in pair_categories:
            pairs = self.comparison_pairs.get(category, {}).get("pairs", [])[:5]

            for pair in pairs:
                ticker_a = pair.get("a", "")
                ticker_b = pair.get("b", "")
                pair_name = pair.get("name", f"{ticker_a} vs {ticker_b}")

                for metric in metrics:
                    for time_range in time_ranges:
                        nl_query = (
                            f"Compare {metric['name']} growth for {pair_name} {time_range['nl']}"
                        )

                        tool_call = {
                            "tool_name": "get_data",
                            "arguments": {
                                "tickers": f"{ticker_a},{ticker_b}",
                                "fields": [metric["field"]],
                                "start": time_range["start"],
                                "end": time_range["end"],
                                "freq": time_range["freq"],
                            },
                        }

                        complexity = self._calculate_complexity(
                            num_tickers=2, num_fields=1, is_time_series=True
                        )

                        test_case = self._create_test_case(
                            nl_query=nl_query,
                            tool_calls=[tool_call],
                            category=self.category,
                            subcategory="historical_comparison",
                            complexity=complexity,
                            tags=["comparison", "historical", category],
                            metadata={
                                "tickers": [ticker_a, ticker_b],
                                "metric": metric["name"],
                                "time_range": time_range["nl"],
                            },
                        )

                        if test_case:
                            test_cases.append(test_case)

        return test_cases

    def _generate_multi_metric_comparisons(self) -> list[TestCase]:
        """Generate comparisons with multiple metrics."""
        test_cases = []

        metric_combos = [
            {"fields": ["WC01001", "WC01751", "WC08301"], "name": "revenue, net income, and ROE"},
            {"fields": ["PE", "PTBV", "DY"], "name": "PE, price-to-book, and dividend yield"},
            {
                "fields": ["WC08316", "WC08366", "WC08301"],
                "name": "operating margin, net margin, and ROE",
            },
        ]

        pairs = [
            {"a": "@AAPL", "b": "U:MSFT", "name": "Apple vs Microsoft"},
            {"a": "U:GOOGL", "b": "U:META", "name": "Google vs Meta"},
            {"a": "U:JPM", "b": "U:BAC", "name": "JPMorgan vs Bank of America"},
        ]

        for pair in pairs:
            for combo in metric_combos:
                nl_query = f"Compare {combo['name']} for {pair['name']}"

                tool_call = self._build_tool_call(
                    tickers=[pair["a"], pair["b"]], fields=combo["fields"]
                )

                complexity = self._calculate_complexity(
                    num_tickers=2, num_fields=len(combo["fields"])
                )

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=[tool_call],
                    category=self.category,
                    subcategory="multi_metric",
                    complexity=complexity,
                    tags=["comparison", "multi_metric"],
                    metadata={"tickers": [pair["a"], pair["b"]], "metrics": combo["name"]},
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def generate(self) -> list[TestCase]:
        """Generate all comparison test cases."""
        test_cases = []

        # Two-stock comparisons
        test_cases.extend(self._generate_two_stock_comparisons())

        # Group comparisons
        test_cases.extend(self._generate_group_comparisons())

        # Sector comparisons
        test_cases.extend(self._generate_sector_comparisons())

        # Index comparisons
        test_cases.extend(self._generate_index_comparisons())

        # Historical comparisons
        test_cases.extend(self._generate_historical_comparisons())

        # Multi-metric comparisons
        test_cases.extend(self._generate_multi_metric_comparisons())

        return test_cases
