#!/usr/bin/env python3
"""
Routing Test Case Generator

Generates comprehensive routing evaluation fixtures for testing
query → domain routing accuracy.
"""

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

# Target counts per subcategory
TARGETS = {
    "datastream_clear": 40,
    "estimates_clear": 40,
    "fundamentals_clear": 40,
    "officers_clear": 25,
    "screening_clear": 25,
    "temporal_ambiguous": 30,
    "domain_boundary": 25,
    "edge_cases": 25,
    "negative_cases": 20,
}

# Company names for variation
COMPANIES = [
    "Apple",
    "Microsoft",
    "Google",
    "Amazon",
    "Tesla",
    "Meta",
    "NVIDIA",
    "Netflix",
    "Adobe",
    "Salesforce",
    "Intel",
    "AMD",
    "Cisco",
    "Oracle",
    "IBM",
    "SAP",
    "Qualcomm",
    "Broadcom",
    "Texas Instruments",
    "Micron",
    "JPMorgan",
    "Goldman Sachs",
    "Morgan Stanley",
    "Bank of America",
    "Wells Fargo",
    "Coca-Cola",
    "PepsiCo",
    "Procter & Gamble",
    "Johnson & Johnson",
    "Pfizer",
    "ExxonMobil",
    "Chevron",
    "Shell",
    "BP",
    "ConocoPhillips",
    "Boeing",
    "Lockheed Martin",
    "Raytheon",
    "General Dynamics",
    "Northrop Grumman",
    "Walmart",
    "Target",
    "Costco",
    "Home Depot",
    "Lowe's",
    "Disney",
    "Comcast",
    "AT&T",
    "Verizon",
    "T-Mobile",
]

# Tickers for variation
TICKERS = [
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "TSLA",
    "META",
    "NVDA",
    "NFLX",
    "ADBE",
    "CRM",
    "INTC",
    "AMD",
    "CSCO",
    "ORCL",
    "IBM",
    "QCOM",
    "JPM",
    "GS",
    "MS",
    "BAC",
    "WFC",
    "C",
    "USB",
    "PNC",
    "KO",
    "PEP",
    "PG",
    "JNJ",
    "PFE",
    "MRK",
    "ABBV",
    "LLY",
]


def generate_id(text: str) -> str:
    """Generate unique ID from text."""
    return f"routing_{hashlib.md5(text.encode()).hexdigest()[:12]}"


def generate_datastream_clear() -> list[dict]:
    """Generate clear datastream routing cases."""
    templates = [
        "What is {company}'s current stock price?",
        "Get {company}'s market cap",
        "Show me {company}'s trading volume today",
        "What is the PE ratio for {company}?",
        "Get the dividend yield for {company}",
        "Show {company}'s stock price history for the past month",
        "What's the 52-week high for {company}?",
        "Get {company}'s OHLC data for yesterday",
        "What is {company}'s beta?",
        "Get the average daily volume for {company}",
        "Show {company}'s market capitalization",
        "What was {company}'s closing price yesterday?",
        "Get {company}'s shares outstanding",
        "What is {company}'s price-to-book ratio?",
        "Show the bid-ask spread for {company}",
        "Get {company}'s enterprise value",
        "What is the current price of {ticker}?",
        "Show me {ticker} stock performance",
        "Get {ticker}'s intraday price chart",
        "What is {ticker} trading at?",
        "Show {company}'s YTD return",
        "Get the volatility for {company}",
        "What is {company}'s relative strength index?",
        "Show {company}'s moving average",
        "Get {company}'s price momentum",
        "What is the current S&P 500 index value?",
        "Show me the Dow Jones Industrial Average",
        "Get the NASDAQ composite index",
        "What is the current price of gold?",
        "Show me crude oil prices",
        "Get the 10-year Treasury yield",
        "What is {company}'s free float?",
        "Show {company}'s short interest",
        "Get the put/call ratio for {company}",
        "What is {company}'s implied volatility?",
        "Show {company}'s options chain",
        "Get {company}'s historical volatility",
        "What is {company}'s Sharpe ratio?",
        "Show {company}'s correlation with the market",
        "Get {company}'s alpha and beta",
    ]

    cases = []
    for i, template in enumerate(templates[: TARGETS["datastream_clear"]]):
        company = COMPANIES[i % len(COMPANIES)]
        ticker = TICKERS[i % len(TICKERS)]
        query = template.format(company=company, ticker=ticker)
        cases.append(
            {
                "id": generate_id(query),
                "nl_query": query,
                "expected_tool_calls": [{"tool_name": "route_to_datastream", "arguments": {}}],
                "expected_response": None,
                "expected_nl_response": None,
                "complexity": 1,
                "category": "routing",
                "subcategory": "datastream_clear",
                "tags": ["routing", "datastream"],
                "metadata": {"expected_domain": "datastream", "min_confidence": 0.8},
            }
        )
    return cases


def generate_estimates_clear() -> list[dict]:
    """Generate clear estimates routing cases."""
    templates = [
        "What do analysts expect {company}'s EPS to be next quarter?",
        "What is the analyst consensus revenue forecast for {company}?",
        "Show me analyst recommendations for {company}",
        "What is the target price for {company} according to analysts?",
        "How many analysts have a buy rating on {company}?",
        "What is the expected earnings growth for {company} next year?",
        "Show the EPS estimate revisions for {company}",
        "What is the projected EBITDA for {company}?",
        "Get the forward PE ratio for {company}",
        "What are the expected earnings per share for {company} in FY2025?",
        "Show analyst upgrades and downgrades for {company}",
        "What is the consensus estimate for {company}'s revenue next quarter?",
        "How many analysts cover {company}?",
        "What is the expected dividend for {company} next year?",
        "Get the earnings surprise history for {company}",
        "What is the mean analyst price target for {company}?",
        "Show {company}'s EPS forecast trend",
        "What is the expected revenue growth rate for {company}?",
        "Get the analyst sentiment score for {company}",
        "What is the projected net income for {company}?",
        "Show {company}'s forward revenue estimates",
        "What is the consensus EBITDA margin forecast for {company}?",
        "Get the number of buy/hold/sell ratings for {company}",
        "What is the highest analyst price target for {company}?",
        "Show the lowest analyst price target for {company}",
        "What is {company}'s projected free cash flow?",
        "Get the PEG ratio forecast for {company}",
        "What is the expected operating margin for {company}?",
        "Show {company}'s earnings calendar and estimates",
        "What do analysts predict for {company}'s gross margin?",
        "Get the long-term growth estimate for {company}",
        "What is the 5-year earnings growth forecast for {company}?",
        "Show analyst estimate dispersion for {company}",
        "What is the standard deviation of EPS estimates for {company}?",
        "Get the estimate momentum for {company}",
        "What is {company}'s expected book value next year?",
        "Show {company}'s projected return on equity",
        "What is the forecasted debt-to-equity ratio for {company}?",
        "Get the expected capex for {company}",
        "What is the projected dividend growth for {company}?",
    ]

    cases = []
    for i, template in enumerate(templates[: TARGETS["estimates_clear"]]):
        company = COMPANIES[i % len(COMPANIES)]
        query = template.format(company=company)
        cases.append(
            {
                "id": generate_id(query),
                "nl_query": query,
                "expected_tool_calls": [{"tool_name": "route_to_estimates", "arguments": {}}],
                "expected_response": None,
                "expected_nl_response": None,
                "complexity": 1,
                "category": "routing",
                "subcategory": "estimates_clear",
                "tags": ["routing", "estimates", "forecast"],
                "metadata": {"expected_domain": "estimates", "min_confidence": 0.8},
            }
        )
    return cases


def generate_fundamentals_clear() -> list[dict]:
    """Generate clear fundamentals routing cases."""
    templates = [
        "What was {company}'s revenue in 2023?",
        "Show me {company}'s net income for the last fiscal year",
        "What is {company}'s total debt?",
        "Get {company}'s operating margin from the last annual report",
        "What was {company}'s reported earnings per share last quarter?",
        "Show {company}'s balance sheet data",
        "What is {company}'s cash flow from operations?",
        "Get {company}'s return on equity for fiscal 2023",
        "What was {company}'s gross profit in the last annual report?",
        "Show me {company}'s income statement for 2022",
        "What is {company}'s current ratio?",
        "Get {company}'s total assets",
        "What was {company}'s net interest income last year?",
        "Show {company}'s dividend payout ratio",
        "What is {company}'s research and development expense?",
        "Get {company}'s working capital",
        "What was {company}'s EBITDA last quarter?",
        "Show {company}'s inventory turnover ratio",
        "What is {company}'s debt-to-equity ratio?",
        "Get {company}'s quick ratio",
        "What was {company}'s operating cash flow in 2023?",
        "Show {company}'s retained earnings",
        "What is {company}'s asset turnover ratio?",
        "Get {company}'s interest coverage ratio",
        "What was {company}'s gross margin last year?",
        "Show {company}'s accounts receivable turnover",
        "What is {company}'s return on assets?",
        "Get {company}'s book value per share",
        "What was {company}'s free cash flow in 2022?",
        "Show {company}'s capital expenditure history",
        "What is {company}'s tangible book value?",
        "Get {company}'s goodwill and intangibles",
        "What was {company}'s SG&A expense last fiscal year?",
        "Show {company}'s depreciation and amortization",
        "What is {company}'s long-term debt?",
        "Get {company}'s shareholders' equity",
        "What was {company}'s cost of goods sold in 2023?",
        "Show {company}'s operating expenses breakdown",
        "What is {company}'s profit margin history?",
        "Get {company}'s financial statements for Q3 2023",
    ]

    cases = []
    for i, template in enumerate(templates[: TARGETS["fundamentals_clear"]]):
        company = COMPANIES[i % len(COMPANIES)]
        query = template.format(company=company)
        cases.append(
            {
                "id": generate_id(query),
                "nl_query": query,
                "expected_tool_calls": [{"tool_name": "route_to_fundamentals", "arguments": {}}],
                "expected_response": None,
                "expected_nl_response": None,
                "complexity": 1,
                "category": "routing",
                "subcategory": "fundamentals_clear",
                "tags": ["routing", "fundamentals", "historical"],
                "metadata": {"expected_domain": "fundamentals", "min_confidence": 0.8},
            }
        )
    return cases


def generate_officers_clear() -> list[dict]:
    """Generate clear officers routing cases."""
    templates = [
        "Who is the CEO of {company}?",
        "Show me the board of directors for {company}",
        "What is the CEO compensation at {company}?",
        "Who is {company}'s CFO?",
        "Show the executive team at {company}",
        "What is the total executive compensation at {company}?",
        "Who are the independent directors at {company}?",
        "Get the governance structure for {company}",
        "How long has the CEO been at {company}?",
        "Who is on the audit committee at {company}?",
        "What is the background of {company}'s CEO?",
        "Show {company}'s C-suite executives",
        "Who is the Chairman of {company}?",
        "Get the compensation committee members at {company}",
        "What is {company}'s CEO-to-worker pay ratio?",
        "Show the insider ownership at {company}",
        "Who is {company}'s Chief Operating Officer?",
        "Get the board tenure information for {company}",
        "What is the average director age at {company}?",
        "Show {company}'s executive stock options",
        "Who is {company}'s General Counsel?",
        "Get the nominating committee at {company}",
        "What bonuses did {company}'s executives receive?",
        "Show the board diversity at {company}",
        "Who is {company}'s Chief Technology Officer?",
    ]

    cases = []
    for i, template in enumerate(templates[: TARGETS["officers_clear"]]):
        company = COMPANIES[i % len(COMPANIES)]
        query = template.format(company=company)
        cases.append(
            {
                "id": generate_id(query),
                "nl_query": query,
                "expected_tool_calls": [{"tool_name": "route_to_officers", "arguments": {}}],
                "expected_response": None,
                "expected_nl_response": None,
                "complexity": 1,
                "category": "routing",
                "subcategory": "officers_clear",
                "tags": ["routing", "officers", "governance"],
                "metadata": {"expected_domain": "officers", "min_confidence": 0.8},
            }
        )
    return cases


def generate_screening_clear() -> list[dict]:
    """Generate clear screening routing cases."""
    templates = [
        "Find the top 10 companies by market cap in the technology sector",
        "Show me stocks with PE ratio under 15 and dividend yield over 3%",
        "Which companies in the S&P 500 have the highest revenue growth?",
        "List the bottom 5 performing stocks in the Dow Jones",
        "Find all healthcare stocks with market cap over $50 billion",
        "Rank energy companies by dividend yield",
        "What are the constituents of the NASDAQ 100?",
        "Screen for stocks with ROE above 20% and debt-to-equity below 0.5",
        "Find the best performing small cap stocks this year",
        "Show financial stocks trading below book value",
        "List the top 20 companies by revenue in the retail sector",
        "Find stocks with market cap between $1B and $10B",
        "Screen for companies with positive earnings growth last 5 years",
        "Show me undervalued stocks in the consumer sector",
        "Rank pharmaceutical companies by R&D spending",
        "Find stocks with the lowest volatility in the S&P 500",
        "List companies with the highest free cash flow yield",
        "Screen for stocks hitting 52-week highs",
        "Find dividend aristocrats with yield above 4%",
        "Show me growth stocks with PEG ratio under 1",
        "Rank tech companies by profit margin",
        "Find the most shorted stocks in the market",
        "Screen for stocks with insider buying activity",
        "List companies with the strongest balance sheets",
        "Find value stocks with low price-to-sales ratio",
    ]

    cases = []
    for i, template in enumerate(templates[: TARGETS["screening_clear"]]):
        query = template
        cases.append(
            {
                "id": generate_id(query),
                "nl_query": query,
                "expected_tool_calls": [{"tool_name": "route_to_screening", "arguments": {}}],
                "expected_response": None,
                "expected_nl_response": None,
                "complexity": 2,
                "category": "routing",
                "subcategory": "screening_clear",
                "tags": ["routing", "screening", "ranking"],
                "metadata": {"expected_domain": "screening", "min_confidence": 0.8},
            }
        )
    return cases


def generate_temporal_ambiguous() -> list[dict]:
    """Generate temporally ambiguous cases (estimates vs fundamentals)."""
    templates = [
        ("What is {company}'s EPS?", ["fundamentals", "estimates"]),
        ("Show me earnings for {company}", ["fundamentals", "estimates"]),
        ("What is {company}'s revenue?", ["fundamentals", "estimates"]),
        ("Get {company}'s earnings per share", ["fundamentals", "estimates"]),
        ("What is {company}'s profit margin?", ["fundamentals", "estimates"]),
        ("Show {company}'s net income", ["fundamentals", "estimates"]),
        ("What is {company}'s EBITDA?", ["fundamentals", "estimates"]),
        ("Get {company}'s operating margin", ["fundamentals", "estimates"]),
        ("What is {company}'s gross margin?", ["fundamentals", "estimates"]),
        ("Show {company}'s cash flow", ["fundamentals", "estimates"]),
        ("What is {company}'s ROE?", ["fundamentals", "estimates"]),
        ("Get {company}'s return on assets", ["fundamentals", "estimates"]),
        ("What is {company}'s debt ratio?", ["fundamentals", "estimates"]),
        ("Show {company}'s leverage", ["fundamentals", "estimates"]),
        ("What is {company}'s interest coverage?", ["fundamentals", "estimates"]),
        ("Get {company}'s dividend", ["fundamentals", "estimates", "datastream"]),
        ("What is {company}'s book value?", ["fundamentals", "estimates"]),
        ("Show {company}'s working capital", ["fundamentals", "estimates"]),
        ("What is {company}'s capex?", ["fundamentals", "estimates"]),
        ("Get {company}'s free cash flow", ["fundamentals", "estimates"]),
        ("What is {company}'s growth rate?", ["fundamentals", "estimates"]),
        ("Show {company}'s margins", ["fundamentals", "estimates"]),
        ("What is {company}'s profitability?", ["fundamentals", "estimates"]),
        ("Get {company}'s financial performance", ["fundamentals", "estimates"]),
        ("What is {company}'s earnings?", ["fundamentals", "estimates"]),
        ("Show {company}'s income", ["fundamentals", "estimates"]),
        ("What is {company}'s profit?", ["fundamentals", "estimates"]),
        ("Get {company}'s sales", ["fundamentals", "estimates"]),
        ("What is {company}'s top line?", ["fundamentals", "estimates"]),
        ("Show {company}'s bottom line", ["fundamentals", "estimates"]),
    ]

    cases = []
    for i, (template, acceptable) in enumerate(templates[: TARGETS["temporal_ambiguous"]]):
        company = COMPANIES[i % len(COMPANIES)]
        query = template.format(company=company)
        # Default expectation is fundamentals (historical data), but either is acceptable
        cases.append(
            {
                "id": generate_id(query),
                "nl_query": query,
                "expected_tool_calls": [{"tool_name": "route_to_fundamentals", "arguments": {}}],
                "expected_response": None,
                "expected_nl_response": None,
                "complexity": 1,
                "category": "routing",
                "subcategory": "temporal_ambiguous",
                "tags": ["routing", "ambiguous", "temporal"],
                "metadata": {
                    "expected_domain": "fundamentals",
                    "acceptable_domains": acceptable,
                    "max_confidence": 0.6,
                    "note": "Ambiguous - no temporal context",
                },
            }
        )
    return cases


def generate_domain_boundary() -> list[dict]:
    """Generate multi-domain boundary cases."""
    templates = [
        (
            "What is {company}'s PE ratio and EPS forecast?",
            "datastream",
            ["datastream", "estimates"],
        ),
        (
            "Compare {company}'s reported revenue with analyst estimates",
            "fundamentals",
            ["fundamentals", "estimates"],
        ),
        ("Show me {company}'s stock price and CEO", "datastream", ["datastream", "officers"]),
        (
            "Top tech stocks by market cap and their CEO compensation",
            "screening",
            ["screening", "officers"],
        ),
        (
            "Find companies with high growth and show their financials",
            "screening",
            ["screening", "fundamentals"],
        ),
        (
            "{company}'s current price and historical revenue",
            "datastream",
            ["datastream", "fundamentals"],
        ),
        (
            "Show {company}'s analyst ratings and board members",
            "estimates",
            ["estimates", "officers"],
        ),
        (
            "Screen for high dividend stocks and show their payout history",
            "screening",
            ["screening", "fundamentals"],
        ),
        (
            "{company}'s forward PE and trailing PE comparison",
            "estimates",
            ["estimates", "datastream"],
        ),
        (
            "Top 10 companies by market cap with CEO compensation details",
            "screening",
            ["screening", "officers"],
        ),
        (
            "Show {company}'s price performance and earnings history",
            "datastream",
            ["datastream", "fundamentals"],
        ),
        (
            "Find growth stocks and their analyst recommendations",
            "screening",
            ["screening", "estimates"],
        ),
        (
            "{company}'s stock chart and financial ratios",
            "datastream",
            ["datastream", "fundamentals"],
        ),
        (
            "Compare {company}'s valuation with peer fundamentals",
            "datastream",
            ["datastream", "fundamentals"],
        ),
        (
            "Show {company}'s price target and reported earnings",
            "estimates",
            ["estimates", "fundamentals"],
        ),
        (
            "Find undervalued stocks with strong management teams",
            "screening",
            ["screening", "officers"],
        ),
        (
            "{company}'s dividend yield and executive stock ownership",
            "datastream",
            ["datastream", "officers"],
        ),
        (
            "Screen for profitable companies and show their cash flows",
            "screening",
            ["screening", "fundamentals"],
        ),
        (
            "Show {company}'s beta and earnings volatility",
            "datastream",
            ["datastream", "fundamentals"],
        ),
        (
            "{company}'s market cap and revenue breakdown",
            "datastream",
            ["datastream", "fundamentals"],
        ),
        ("Top banks by assets and their board composition", "screening", ["screening", "officers"]),
        (
            "{company}'s price momentum and analyst sentiment",
            "datastream",
            ["datastream", "estimates"],
        ),
        (
            "Find high-margin companies with strong governance",
            "screening",
            ["screening", "officers"],
        ),
        (
            "Show {company}'s trading volume and insider transactions",
            "datastream",
            ["datastream", "officers"],
        ),
        (
            "{company}'s current valuation and historical profitability",
            "datastream",
            ["datastream", "fundamentals"],
        ),
    ]

    cases = []
    for i, (template, primary, acceptable) in enumerate(templates[: TARGETS["domain_boundary"]]):
        company = COMPANIES[i % len(COMPANIES)]
        query = template.format(company=company)
        cases.append(
            {
                "id": generate_id(query),
                "nl_query": query,
                "expected_tool_calls": [{"tool_name": f"route_to_{primary}", "arguments": {}}],
                "expected_response": None,
                "expected_nl_response": None,
                "complexity": 2,
                "category": "routing",
                "subcategory": "domain_boundary",
                "tags": ["routing", "boundary", "multi_domain"],
                "metadata": {
                    "expected_domain": primary,
                    "acceptable_domains": acceptable,
                    "min_confidence": 0.5,
                    "note": "Multi-domain query - primary domain expected",
                },
            }
        )
    return cases


def generate_edge_cases() -> list[dict]:
    """Generate edge cases: abbreviations, typos, incomplete queries."""
    cases = []

    # Abbreviations and informal language
    abbrev_queries = [
        ("What's AAPL's mkt cap?", "datastream"),
        ("MSFT P/E?", "datastream"),
        ("TSLA rev growth?", "fundamentals"),
        ("NVDA analyst recs?", "estimates"),
        ("Who's the CEO @ Google?", "officers"),
        ("Top 10 by mcap in tech?", "screening"),
        ("AMZN EPS est?", "estimates"),
        ("JPM NII last yr?", "fundamentals"),
        ("Show me GOOGL vol", "datastream"),
        ("META forward P/E?", "estimates"),
    ]

    for query, domain in abbrev_queries:
        cases.append(
            {
                "id": generate_id(query),
                "nl_query": query,
                "expected_tool_calls": [{"tool_name": f"route_to_{domain}", "arguments": {}}],
                "expected_response": None,
                "expected_nl_response": None,
                "complexity": 2,
                "category": "routing",
                "subcategory": "edge_cases",
                "tags": ["routing", "edge_case", "abbreviation"],
                "metadata": {"expected_domain": domain, "min_confidence": 0.7},
            }
        )

    # Incomplete/casual queries
    casual_queries = [
        ("apple stock", "datastream"),
        ("tesla earnings", "fundamentals"),
        ("microsoft ceo", "officers"),
        ("best tech stocks", "screening"),
        ("nvidia forecast", "estimates"),
        ("amazon financials", "fundamentals"),
        ("google board", "officers"),
        ("bank stocks ranking", "screening"),
        ("apple price target", "estimates"),
        ("tesla balance sheet", "fundamentals"),
    ]

    for query, domain in casual_queries:
        cases.append(
            {
                "id": generate_id(query),
                "nl_query": query,
                "expected_tool_calls": [{"tool_name": f"route_to_{domain}", "arguments": {}}],
                "expected_response": None,
                "expected_nl_response": None,
                "complexity": 1,
                "category": "routing",
                "subcategory": "edge_cases",
                "tags": ["routing", "edge_case", "casual"],
                "metadata": {"expected_domain": domain, "min_confidence": 0.6},
            }
        )

    # Questions with context
    context_queries = [
        ("I'm looking at Apple's fundamentals, what's their debt level?", "fundamentals"),
        ("For my portfolio analysis, show me Tesla's beta", "datastream"),
        ("Researching tech stocks - who runs Microsoft?", "officers"),
        ("Building a screener - find high dividend stocks", "screening"),
        ("Checking analyst views on Amazon", "estimates"),
    ]

    for query, domain in context_queries:
        cases.append(
            {
                "id": generate_id(query),
                "nl_query": query,
                "expected_tool_calls": [{"tool_name": f"route_to_{domain}", "arguments": {}}],
                "expected_response": None,
                "expected_nl_response": None,
                "complexity": 2,
                "category": "routing",
                "subcategory": "edge_cases",
                "tags": ["routing", "edge_case", "context"],
                "metadata": {"expected_domain": domain, "min_confidence": 0.7},
            }
        )

    return cases[: TARGETS["edge_cases"]]


def generate_negative_cases() -> list[dict]:
    """Generate negative cases - queries that don't fit any domain well."""
    queries = [
        "What's the weather like in New York?",
        "How do I open a brokerage account?",
        "What is the best programming language?",
        "Tell me a joke about stocks",
        "What time does the market open?",
        "How does compound interest work?",
        "What is a stock split?",
        "Explain options trading",
        "What's the difference between stocks and bonds?",
        "How do I calculate my taxes on investments?",
        "What is the history of the stock market?",
        "Who invented the mutual fund?",
        "What causes inflation?",
        "How does the Federal Reserve work?",
        "What is cryptocurrency?",
        "Should I invest in gold?",
        "What is a hedge fund?",
        "How do ETFs work?",
        "What is dollar cost averaging?",
        "Explain market capitalization",
    ]

    cases = []
    for query in queries[: TARGETS["negative_cases"]]:
        # For negative cases, we expect low confidence or datastream as fallback
        cases.append(
            {
                "id": generate_id(query),
                "nl_query": query,
                "expected_tool_calls": [{"tool_name": "route_to_datastream", "arguments": {}}],
                "expected_response": None,
                "expected_nl_response": None,
                "complexity": 1,
                "category": "routing",
                "subcategory": "negative_cases",
                "tags": ["routing", "negative", "out_of_domain"],
                "metadata": {
                    "expected_domain": "datastream",
                    "acceptable_domains": [
                        "datastream",
                        "fundamentals",
                        "estimates",
                        "officers",
                        "screening",
                    ],
                    "max_confidence": 0.5,
                    "note": "Out-of-domain query - any routing acceptable but low confidence expected",
                },
            }
        )
    return cases


def generate_all() -> dict:
    """Generate all routing test cases."""
    all_cases = []

    all_cases.extend(generate_datastream_clear())
    all_cases.extend(generate_estimates_clear())
    all_cases.extend(generate_fundamentals_clear())
    all_cases.extend(generate_officers_clear())
    all_cases.extend(generate_screening_clear())
    all_cases.extend(generate_temporal_ambiguous())
    all_cases.extend(generate_domain_boundary())
    all_cases.extend(generate_edge_cases())
    all_cases.extend(generate_negative_cases())

    # Count by subcategory
    subcategory_counts = {}
    for case in all_cases:
        subcat = case["subcategory"]
        subcategory_counts[subcat] = subcategory_counts.get(subcat, 0) + 1

    return {
        "_meta": {
            "name": "query_routing",
            "capability": "query_routing",
            "description": "Query routing to domain agent tests",
            "requires_nl_response": False,
            "requires_expected_response": False,
            "schema_version": "1.0",
            "generated_at": datetime.now(UTC).isoformat(),
            "generator": "scripts/generators/routing_generator.py",
            "subcategory_counts": subcategory_counts,
        },
        "metadata": {
            "category": "routing",
            "generator": "RoutingGenerator",
            "count": len(all_cases),
        },
        "test_cases": all_cases,
    }


def main():
    """Generate routing fixtures and write to file."""
    output_path = Path("tests/fixtures/lseg/generated/routing/routing.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    data = generate_all()

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    print(f"Generated {data['metadata']['count']} routing test cases")
    print(f"Subcategory counts: {data['_meta']['subcategory_counts']}")
    print(f"Output: {output_path}")


if __name__ == "__main__":
    main()
