"""
Screening Generator - Generates stock screening and filtering test cases.

Target: ~500 test cases
- Index constituent queries
- Top N by metric
- Multi-criteria screens
- Sector-specific screens
"""

import random
from pathlib import Path

from .base_generator import BaseGenerator, TestCase


class ScreeningGenerator(BaseGenerator):
    """Generator for screening and filtering test cases."""

    def __init__(self, data_dir: Path):
        super().__init__(data_dir)
        self.category = "screening"

    def _generate_index_constituent_queries(self) -> list[TestCase]:
        """Generate index constituent list queries."""
        test_cases = []

        indices_data = self.indices.get("indices", {})

        templates = [
            "What companies are in the {index}?",
            "List all {index} constituents",
            "Show {index} members",
            "Get the components of the {index}",
            "Which stocks are in the {index}?",
        ]

        for region, indices in indices_data.items():
            if region == "Sector_Indices":
                continue

            for index_info in indices:
                index_name = index_info.get("name", "")
                index_code = index_info.get("symbol", "")
                constituent_code = index_info.get("constituent_code")

                if not constituent_code:
                    continue

                template = random.choice(templates)
                nl_query = template.format(index=index_name)

                tool_call = {
                    "tool_name": "get_data",
                    "arguments": {
                        "tickers": f"{constituent_code}|L",
                        "fields": ["MNEM", "NAME"],
                        "kind": 0,
                    },
                }

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=[tool_call],
                    category=self.category,
                    subcategory="index_constituents",
                    complexity=3,
                    tags=["screening", "index", "constituents"],
                    metadata={
                        "index_name": index_name,
                        "index_code": index_code,
                        "constituent_code": constituent_code,
                    },
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_top_n_queries(self) -> list[TestCase]:
        """Generate top N by metric queries."""
        test_cases = []

        ranking_metrics = self.screening_criteria.get("ranking_metrics", [])

        top_ns = [5, 10, 20, 50]

        universes = [
            {"name": "companies", "filter": None},
            {"name": "US stocks", "filter": 'IN(TR.ExchangeCountryCode,"US")'},
            {"name": "NYSE stocks", "filter": 'IN(TR.ExchangeMarketIdCode,"XNYS")'},
            {"name": "NASDAQ stocks", "filter": 'IN(TR.ExchangeMarketIdCode,"XNAS")'},
        ]

        templates = [
            "What are the top {n} companies by {metric}?",
            "Show the {n} largest companies by {metric}",
            "Get top {n} stocks by {metric}",
            "List the {n} highest {metric} stocks",
        ]

        for metric in ranking_metrics:
            for n in top_ns:
                for universe in universes:
                    template = random.choice(templates)
                    nl_query = template.format(n=n, metric=metric["name"])

                    if universe["name"] != "companies":
                        nl_query = nl_query.replace("companies", universe["name"])

                    # Build SCREEN expression
                    screen_parts = ["U(IN(Equity(active,public,primary)))"]
                    if universe["filter"]:
                        screen_parts.append(universe["filter"])
                    screen_parts.append(f"TOP({metric['field']}, {n}, nnumber)")
                    screen_parts.append("CURN=USD")

                    screen_exp = f"SCREEN({', '.join(screen_parts)})"

                    tool_call = {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": screen_exp,
                            "fields": ["TR.CommonName", metric["field"]],
                        },
                    }

                    test_case = self._create_test_case(
                        nl_query=nl_query,
                        tool_calls=[tool_call],
                        category=self.category,
                        subcategory="top_n",
                        complexity=4,
                        tags=["screening", "top_n", metric["name"].replace(" ", "_")],
                        metadata={"metric": metric["name"], "n": n, "universe": universe["name"]},
                    )

                    if test_case:
                        test_cases.append(test_case)

        return test_cases

    def _generate_single_criteria_screens(self) -> list[TestCase]:
        """Generate single-criteria filter queries."""
        test_cases = []

        single_criteria = self.screening_criteria.get("single_criteria", {})

        templates = [
            "Find stocks with {criteria}",
            "Show companies where {criteria}",
            "Get stocks with {criteria}",
            "Which stocks have {criteria}?",
        ]

        for category, criteria_list in single_criteria.items():
            for criteria in criteria_list:
                template = random.choice(templates)
                nl_query = template.format(criteria=criteria["nl"])

                screen_exp = (
                    f"SCREEN(U(IN(Equity(active,public,primary))), {criteria['filter']}, CURN=USD)"
                )

                tool_call = {
                    "tool_name": "get_data",
                    "arguments": {
                        "tickers": screen_exp,
                        "fields": ["TR.CommonName", "TR.CompanyMarketCap"],
                    },
                }

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=[tool_call],
                    category=self.category,
                    subcategory="single_criteria",
                    complexity=4,
                    tags=["screening", "filter", category],
                    metadata={
                        "criteria_name": criteria["name"],
                        "filter": criteria["filter"],
                        "category": category,
                    },
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_multi_criteria_screens(self) -> list[TestCase]:
        """Generate multi-criteria filter queries."""
        test_cases = []

        multi_screens = self.screening_criteria.get("multi_criteria_screens", {})

        for screen_key, screen_def in multi_screens.items():
            nl_query = f"Find {screen_def['nl']}"

            filters = screen_def["filters"]
            screen_exp = (
                f"SCREEN(U(IN(Equity(active,public,primary))), {', '.join(filters)}, CURN=USD)"
            )

            tool_call = {
                "tool_name": "get_data",
                "arguments": {
                    "tickers": screen_exp,
                    "fields": ["TR.CommonName", "TR.CompanyMarketCap"],
                },
            }

            test_case = self._create_test_case(
                nl_query=nl_query,
                tool_calls=[tool_call],
                category=self.category,
                subcategory="multi_criteria",
                complexity=6,
                tags=["screening", "multi_criteria", screen_key],
                metadata={"screen_name": screen_def["name"], "filters": filters},
            )

            if test_case:
                test_cases.append(test_case)

        return test_cases

    def _generate_sector_screens(self) -> list[TestCase]:
        """Generate sector-specific screening queries."""
        test_cases = []

        self.screening_criteria.get("sector_screens", {})

        # Sector codes
        sector_codes = {
            "technology": "57",
            "healthcare": "55",
            "financials": "55",
            "energy": "50",
            "consumer discretionary": "54",
            "consumer staples": "53",
            "industrials": "52",
        }

        metrics = [
            {"field": "TR.CompanyMarketCap", "name": "market cap"},
            {"field": "TR.Revenue", "name": "revenue"},
            {"field": "TR.NetIncome", "name": "net income"},
            {"field": "TR.ROE", "name": "ROE"},
        ]

        for sector, code in sector_codes.items():
            for metric in metrics:
                nl_query = f"What are the top {sector} companies by {metric['name']}?"

                screen_exp = f'SCREEN(U(IN(Equity(active,public,primary))), IN(TR.TRBCEconSectorCode,"{code}"), TOP({metric["field"]}, 10, nnumber), CURN=USD)'

                tool_call = {
                    "tool_name": "get_data",
                    "arguments": {
                        "tickers": screen_exp,
                        "fields": ["TR.CommonName", metric["field"]],
                    },
                }

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=[tool_call],
                    category=self.category,
                    subcategory="sector_screen",
                    complexity=5,
                    tags=["screening", "sector", sector.replace(" ", "_")],
                    metadata={"sector": sector, "sector_code": code, "metric": metric["name"]},
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_index_ranked_queries(self) -> list[TestCase]:
        """Generate index constituent ranked by metric queries."""
        test_cases = []

        indices = [
            {"code": "LS&PCOMP", "name": "S&P 500"},
            {"code": "LFTSE100", "name": "FTSE 100"},
            {"code": "LDAXINDX", "name": "DAX"},
        ]

        metrics = [
            {"field": "DY", "name": "dividend yield"},
            {"field": "PE", "name": "PE ratio"},
            {"field": "MV", "name": "market cap"},
            {"field": "WC08301", "name": "ROE"},
        ]

        for index in indices:
            for metric in metrics:
                nl_query = f"Which {index['name']} stocks have the highest {metric['name']}?"

                # This requires a two-step approach in practice
                tool_calls = [
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": f"{index['code']}|L",
                            "fields": ["RIC"],
                            "kind": 0,
                        },
                    },
                    {
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": "{{constituents}}",
                            "fields": ["TR.CommonName", metric["field"]],
                        },
                    },
                ]

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=tool_calls,
                    category=self.category,
                    subcategory="index_ranked",
                    complexity=7,
                    tags=["screening", "index", "ranked"],
                    metadata={
                        "index_name": index["name"],
                        "index_code": index["code"],
                        "metric": metric["name"],
                    },
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def _generate_historical_constituent_queries(self) -> list[TestCase]:
        """Generate historical index constituent queries."""
        test_cases = []

        historical_dates = [
            {"code": "0120", "nl": "January 2020"},
            {"code": "1219", "nl": "December 2019"},
            {"code": "0618", "nl": "June 2018"},
            {"code": "0115", "nl": "January 2015"},
        ]

        indices = [
            {"base": "LS&PCOMP", "name": "S&P 500"},
            {"base": "LFTSE100", "name": "FTSE 100"},
        ]

        for index in indices:
            for date in historical_dates:
                nl_query = f"What companies were in the {index['name']} in {date['nl']}?"

                tool_call = {
                    "tool_name": "get_data",
                    "arguments": {
                        "tickers": f"{index['base']}{date['code']}|L",
                        "fields": ["MNEM", "NAME"],
                        "kind": 0,
                    },
                }

                test_case = self._create_test_case(
                    nl_query=nl_query,
                    tool_calls=[tool_call],
                    category=self.category,
                    subcategory="historical_constituents",
                    complexity=5,
                    tags=["screening", "index", "historical"],
                    metadata={"index_name": index["name"], "date": date["nl"]},
                )

                if test_case:
                    test_cases.append(test_case)

        return test_cases

    def generate(self) -> list[TestCase]:
        """Generate all screening test cases."""
        test_cases = []

        # Index constituent queries
        test_cases.extend(self._generate_index_constituent_queries())

        # Top N queries
        test_cases.extend(self._generate_top_n_queries())

        # Single criteria screens
        test_cases.extend(self._generate_single_criteria_screens())

        # Multi-criteria screens
        test_cases.extend(self._generate_multi_criteria_screens())

        # Sector screens
        test_cases.extend(self._generate_sector_screens())

        # Index ranked queries
        test_cases.extend(self._generate_index_ranked_queries())

        # Historical constituent queries
        test_cases.extend(self._generate_historical_constituent_queries())

        return test_cases
