"""
Lookup Generator - Generates single-field and multi-field lookup test cases.

Target: ~4,000 test cases
- Single-field lookups: 300 fields × 10 tickers = 3,000
- Multi-field lookups: 50 field combos × 20 tickers = 1,000
"""

import random
from typing import List, Dict, Any, Optional
from pathlib import Path
from .base_generator import BaseGenerator, TestCase


class LookupGenerator(BaseGenerator):
    """Generator for lookup test cases (single and multi-field)."""

    def __init__(self, data_dir: Path):
        super().__init__(data_dir)
        self.category = "lookups"

    def _get_nl_variant(self, field_info: Dict) -> str:
        """Get a natural language variant for a field."""
        variants = field_info.get("nl_variants", [])
        if variants:
            return random.choice(variants)
        return field_info.get("description", "value")

    def _get_company_name(self, ticker_info: Dict) -> str:
        """Get company name or formatted ticker."""
        return ticker_info.get("name", ticker_info.get("symbol", "Company"))

    def _build_tool_call(self, ticker: str, fields: List[str],
                         start: str = "0D", end: str = "0D",
                         freq: Optional[str] = None, kind: int = 0) -> Dict:
        """Build a tool call dictionary."""
        call = {
            "function": "get_data",
            "arguments": {
                "tickers": ticker,
                "fields": fields,
                "start": start,
                "end": end,
                "kind": kind
            }
        }
        if freq:
            call["arguments"]["freq"] = freq
        return call

    def _generate_single_field_lookups(self) -> List[TestCase]:
        """Generate single-field lookup test cases."""
        test_cases = []
        templates = self.nl_templates.get("lookup_single", {}).get("templates", [
            "What is {{company}}'s {{field_name}}?",
            "Get {{company}}'s {{field_name}}",
            "Show me {{field_name}} for {{company}}"
        ])

        # Get all fields
        all_fields = []

        # Datastream fields
        for category, fields in self.datastream_fields.items():
            if isinstance(fields, dict):
                for code, info in fields.items():
                    if isinstance(info, dict):
                        all_fields.append({
                            "code": code,
                            "api": "datastream",
                            "category": category,
                            **info
                        })

        # TR fields (fundamentals)
        for category, fields in self.fundamentals_tr.items():
            if isinstance(fields, dict):
                for code, info in fields.items():
                    if isinstance(info, dict):
                        all_fields.append({
                            "code": code,
                            "api": "refinitiv",
                            "category": category,
                            **info
                        })

        # Worldscope fields
        for category, fields in self.fundamentals_wc.items():
            if isinstance(fields, dict):
                for code, info in fields.items():
                    if isinstance(info, dict):
                        all_fields.append({
                            "code": code,
                            "api": "datastream",
                            "category": category,
                            **info
                        })

        # Estimates fields
        for category, fields in self.estimates_tr.items():
            if isinstance(fields, dict):
                for code, info in fields.items():
                    if isinstance(info, dict):
                        all_fields.append({
                            "code": code,
                            "api": "refinitiv",
                            "category": f"estimates_{category}",
                            **info
                        })

        # Officers/Directors fields
        for category, fields in self.officers_tr.items():
            if isinstance(fields, dict):
                for code, info in fields.items():
                    if isinstance(info, dict):
                        all_fields.append({
                            "code": code,
                            "api": "refinitiv",
                            "category": f"officers_{category}",
                            **info
                        })

        # Get ALL tickers for full field coverage
        tickers = self._get_all_tickers(count=200)

        # Generate combinations - ensure 100% field coverage
        for field in all_fields:
            # Sample 12 tickers per field to ensure all fields are tested
            sampled_tickers = random.sample(tickers, min(12, len(tickers)))

            for ticker_info in sampled_tickers:
                ticker_symbol = ticker_info.get("symbol", "")
                company_name = self._get_company_name(ticker_info)
                field_name = self._get_nl_variant(field)

                # Select template
                template = random.choice(templates)
                nl_query = template.replace("{{company}}", company_name).replace("{{field_name}}", field_name)

                # Build tool call
                tool_call = self._build_tool_call(
                    ticker=ticker_symbol,
                    fields=[field["code"]]
                )

                complexity = self._calculate_complexity(num_tickers=1, num_fields=1)

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=[tool_call],
                    category=self.category,
                    subcategory="single_field",
                    complexity=complexity,
                    tags=[field.get("category", "unknown"), field["api"]],
                    metadata={
                        "field_code": field["code"],
                        "ticker": ticker_symbol,
                        "field_category": field.get("category")
                    }
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_multi_field_lookups(self) -> List[TestCase]:
        """Generate multi-field lookup test cases."""
        test_cases = []
        templates = self.nl_templates.get("lookup_multi_field", {}).get("templates", [
            "Get {{company}}'s {{field1}} and {{field2}}",
            "Show me {{company}}'s {{field1}}, {{field2}}, and {{field3}}"
        ])

        # Define field combinations (logical groupings)
        field_combos = [
            # Income statement package
            {
                "name": "income_summary",
                "fields": ["WC01001", "WC01100", "WC01250", "WC01751"],
                "nl": "revenue, gross profit, operating income, and net income"
            },
            # Balance sheet summary
            {
                "name": "balance_sheet_summary",
                "fields": ["WC02999", "WC03351", "WC03501"],
                "nl": "total assets, liabilities, and equity"
            },
            # Cash flow summary
            {
                "name": "cash_flow_summary",
                "fields": ["WC04860", "WC04870", "WC04890"],
                "nl": "operating, investing, and financing cash flows"
            },
            # Per share metrics
            {
                "name": "per_share",
                "fields": ["WC05201", "WC05101", "WC05476"],
                "nl": "EPS, dividends per share, and book value per share"
            },
            # Profitability ratios
            {
                "name": "profitability",
                "fields": ["WC08301", "WC08326", "WC08376"],
                "nl": "ROE, ROA, and ROIC"
            },
            # Valuation metrics
            {
                "name": "valuation",
                "fields": ["PE", "PTBV", "EV", "DY"],
                "nl": "PE ratio, price to book, enterprise value, and dividend yield"
            },
            # OHLC data
            {
                "name": "ohlc",
                "fields": ["PO", "PH", "PL", "P"],
                "nl": "open, high, low, and close prices"
            },
            # OHLCV data
            {
                "name": "ohlcv",
                "fields": ["PO", "PH", "PL", "P", "VO"],
                "nl": "open, high, low, close, and volume"
            },
            # Margin profile
            {
                "name": "margins",
                "fields": ["WC08316", "WC08321", "WC08366"],
                "nl": "operating, pretax, and net margins"
            },
            # Liquidity metrics
            {
                "name": "liquidity",
                "fields": ["WC02001", "WC08106", "WC08101"],
                "nl": "cash, current ratio, and quick ratio"
            },
            # Debt metrics
            {
                "name": "debt",
                "fields": ["WC03255", "WC08221", "WC08224"],
                "nl": "total debt, debt to capital, and debt to equity"
            },
            # Price and market cap
            {
                "name": "price_and_cap",
                "fields": ["P", "MV"],
                "nl": "stock price and market cap"
            },
            # Dividend data
            {
                "name": "dividend",
                "fields": ["DY", "DPS", "POUT"],
                "nl": "dividend yield, DPS, and payout ratio"
            },
            # Company info
            {
                "name": "company_info",
                "fields": ["NAME", "SECTOR", "CURR", "EXCH"],
                "nl": "name, sector, currency, and exchange"
            }
        ]

        tickers = self._get_all_tickers(count=30)

        for combo in field_combos:
            # Sample tickers for each combo
            sampled_tickers = random.sample(tickers, min(20, len(tickers)))

            for ticker_info in sampled_tickers:
                ticker_symbol = ticker_info.get("symbol", "")
                company_name = self._get_company_name(ticker_info)

                # Build query
                nl_query = f"Get {company_name}'s {combo['nl']}"

                tool_call = self._build_tool_call(
                    ticker=ticker_symbol,
                    fields=combo["fields"]
                )

                complexity = self._calculate_complexity(
                    num_tickers=1,
                    num_fields=len(combo["fields"])
                )

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=[tool_call],
                    category=self.category,
                    subcategory="multi_field",
                    complexity=complexity,
                    tags=["multi_field", combo["name"]],
                    metadata={
                        "field_combo": combo["name"],
                        "fields": combo["fields"],
                        "ticker": ticker_symbol
                    }
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_field_category_lookups(self) -> List[TestCase]:
        """Generate lookups for entire field categories."""
        test_cases = []

        category_queries = [
            {
                "query_template": "Get {company}'s full income statement",
                "fields": ["WC01001", "WC01051", "WC01100", "WC01101", "WC01201",
                          "WC01151", "WC01250", "WC01251", "WC01401", "WC01451", "WC01751"],
                "tags": ["income_statement", "full_statement"]
            },
            {
                "query_template": "Show {company}'s complete balance sheet",
                "fields": ["WC02001", "WC02051", "WC02101", "WC02201", "WC02501",
                          "WC02999", "WC03040", "WC03051", "WC03101", "WC03251",
                          "WC03351", "WC03501", "WC03999"],
                "tags": ["balance_sheet", "full_statement"]
            },
            {
                "query_template": "Get {company}'s cash flow statement",
                "fields": ["WC04201", "WC04049", "WC04860", "WC04601", "WC04870",
                          "WC04551", "WC04890", "WC04851"],
                "tags": ["cash_flow", "full_statement"]
            }
        ]

        tickers = self._get_all_tickers(count=20)

        for query_def in category_queries:
            for ticker_info in tickers:
                ticker_symbol = ticker_info.get("symbol", "")
                company_name = self._get_company_name(ticker_info)

                nl_query = query_def["query_template"].replace("{company}", company_name)

                tool_call = self._build_tool_call(
                    ticker=ticker_symbol,
                    fields=query_def["fields"]
                )

                complexity = self._calculate_complexity(
                    num_tickers=1,
                    num_fields=len(query_def["fields"])
                )

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=[tool_call],
                    category=self.category,
                    subcategory="category_lookup",
                    complexity=complexity,
                    tags=query_def["tags"],
                    metadata={
                        "fields": query_def["fields"],
                        "ticker": ticker_symbol
                    }
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def generate(self) -> List[TestCase]:
        """Generate all lookup test cases."""
        test_cases = []

        # Single field lookups (~3000)
        test_cases.extend(self._generate_single_field_lookups())

        # Multi-field lookups (~1000)
        test_cases.extend(self._generate_multi_field_lookups())

        # Category lookups (~100)
        test_cases.extend(self._generate_field_category_lookups())

        return test_cases
