"""
Screening Domain Agent

Handles Screening & Ranking API queries including:
- Top N queries (by market cap, revenue, etc.)
- Multi-criteria filtering
- Index constituents
- Sector/industry screening
- Value/growth screening
"""

from __future__ import annotations

import re
from typing import Any

from CONTRACTS import ToolCall
from src.nl2api.agents.base import BaseDomainAgent
from src.nl2api.agents.protocols import AgentContext, AgentResult
from src.nl2api.llm.protocols import LLMProvider, LLMToolDefinition
from src.nl2api.rag.protocols import RAGRetriever


class ScreeningAgent(BaseDomainAgent):
    """
    Domain agent for Screening & Ranking API.

    Handles natural language queries about:
    - Top N companies by metric
    - Multi-criteria screening
    - Index constituents
    - Sector/industry filtering
    - Value and growth screening
    """

    # Ranking metrics
    RANKING_METRICS = {
        "market cap": "TR.CompanyMarketCap",
        "market capitalization": "TR.CompanyMarketCap",
        "revenue": "TR.Revenue",
        "sales": "TR.Revenue",
        "net income": "TR.NetIncome",
        "profit": "TR.NetIncome",
        "total assets": "TR.TotalAssets",
        "dividend yield": "TR.DividendYield",
        "pe ratio": "TR.PE",
        "p/e": "TR.PE",
        "roe": "TR.ROE",
        "roa": "TR.ROA",
    }

    # Filter metrics for comparison operators
    FILTER_METRICS = {
        "pe ratio": "TR.PE",
        "pe": "TR.PE",
        "p/e": "TR.PE",
        "dividend yield": "TR.DividendYield",
        "roe": "TR.ROE",
        "return on equity": "TR.ROE",
        "roa": "TR.ROA",
        "return on assets": "TR.ROA",
        "market cap": "TR.CompanyMarketCap",
        "revenue growth": "TR.RevenueGrowth",
        "free cash flow": "TR.FreeCashFlow",
        "fcf": "TR.FreeCashFlow",
        "profit margin": "TR.NetProfitMargin",
        "eps surprise": "TR.EPSSurprisePct(Period=FQ0)",
        "earnings surprise": "TR.EPSSurprisePct(Period=FQ0)",
    }

    # Sector codes (TRBC Economic Sector)
    SECTOR_CODES = {
        "tech": "57",
        "technology": "57",
        "healthcare": "55",
        "health care": "55",
        "financials": "55",
        "finance": "55",
        "consumer": "53",
        "energy": "50",
        "oil": "50",
        "industrials": "52",
        "materials": "51",
        "utilities": "59",
        "real estate": "60",
    }

    # Index codes for constituents
    INDEX_CODES = {
        "s&p 500": "0#.SPX",
        "sp500": "0#.SPX",
        "s&p500": "0#.SPX",
        "ftse 100": "0#.FTSE",
        "ftse100": "0#.FTSE",
        "nasdaq 100": "0#.NDX",
        "nasdaq100": "0#.NDX",
        "dow jones": "0#.DJI",
        "dow": "0#.DJI",
    }

    # Country codes
    COUNTRY_CODES = {
        "us": "US",
        "usa": "US",
        "united states": "US",
        "uk": "GB",
        "united kingdom": "GB",
        "germany": "DE",
        "japan": "JP",
        "china": "CN",
        "france": "FR",
    }

    # Display fields for results
    DEFAULT_DISPLAY_FIELDS = ["TR.CommonName", "TR.CompanyMarketCap"]

    # Keywords for domain classification
    DOMAIN_KEYWORDS = [
        # Ranking
        "top", "largest", "biggest", "highest", "best",
        "smallest", "lowest", "bottom",
        # Screening
        "find", "screen", "filter", "show me", "list",
        "companies with", "stocks with",
        # Criteria
        "above", "below", "over", "under", "greater than", "less than",
        "more than", "at least",
        # Index
        "in the", "constituents", "members of",
        # Sectors
        "tech", "healthcare", "financial", "energy",
        # Styles
        "value", "growth", "dividend", "undervalued",
    ]

    def __init__(
        self,
        llm: LLMProvider,
        rag: RAGRetriever | None = None,
    ):
        """Initialize the Screening agent."""
        super().__init__(llm, rag)

    @property
    def domain_name(self) -> str:
        """Return the domain name."""
        return "screening"

    @property
    def domain_description(self) -> str:
        """Return the domain description."""
        return (
            "Screening & Ranking API - find companies by criteria, "
            "top N rankings, index constituents, multi-criteria filtering"
        )

    def get_system_prompt(self) -> str:
        """Return the system prompt for the Screening domain."""
        return """You are an expert at translating natural language screening queries into LSEG Screener API calls.

Your task is to generate accurate `refinitiv.get_data` tool calls with SCREEN expressions.

## SCREEN Expression Syntax

```
SCREEN(Universe, Filter1, Filter2, ..., Options)
```

### Universe
Always use: `U(IN(Equity(active,public,primary)))`

### TOP Function (Ranking)
```
TOP(metric, count, nnumber)
```
Example: `TOP(TR.CompanyMarketCap, 10, nnumber)` - top 10 by market cap

### Filter Operators
- Greater than: `TR.PE>15` or `TR.PE>=15`
- Less than: `TR.PE<15` or `TR.PE<=15`
- Equals: `TR.HQCountryCode=US`
- In set: `IN(TR.TRBCEconSectorCode,57)` for tech sector

### Common Filters
- Country: `TR.HQCountryCode=US`
- Sector: `IN(TR.TRBCEconSectorCode,57)` (57 = Technology)
- Index: `IN(TR.IndexConstituentRIC,"0#.SPX")` for S&P 500

### Options
- Currency: `CURN=USD` (always include for consistency)

## Sector Codes (TRBC)
- 50 = Energy
- 51 = Basic Materials
- 52 = Industrials
- 53 = Consumer Cyclical
- 54 = Consumer Non-Cyclical
- 55 = Healthcare
- 56 = Financials
- 57 = Technology
- 59 = Utilities
- 60 = Real Estate

## Examples

Query: "What are the top 10 companies by market cap in the S&P 500?"
Tool call: refinitiv.get_data(
  instruments=["SCREEN(U(IN(Equity(active,public,primary))),IN(TR.IndexConstituentRIC,\"0#.SPX\"),TOP(TR.CompanyMarketCap,10,nnumber),CURN=USD)"],
  fields=["TR.CommonName", "TR.CompanyMarketCap"]
)

Query: "Find US tech stocks with dividend yield above 2%"
Tool call: refinitiv.get_data(
  instruments=["SCREEN(U(IN(Equity(active,public,primary))),TR.HQCountryCode=US,IN(TR.TRBCEconSectorCode,57),TR.DividendYield>2,CURN=USD)"],
  fields=["TR.CommonName", "TR.DividendYield", "TR.CompanyMarketCap"]
)

Query: "Find undervalued stocks with PE ratio below 15 and ROE above 15%"
Tool call: refinitiv.get_data(
  instruments=["SCREEN(U(IN(Equity(active,public,primary))),TR.PE<15,TR.ROE>15,CURN=USD)"],
  fields=["TR.CommonName", "TR.PE", "TR.ROE", "TR.CompanyMarketCap"]
)

## Rules
1. Always include the base universe: `U(IN(Equity(active,public,primary)))`
2. Always include `CURN=USD` for currency normalization
3. Use TOP() for ranking queries
4. Use comparison operators for filtering
5. Include relevant fields for display
6. If the query is ambiguous, ask for clarification

Generate the most appropriate refinitiv.get_data tool call for the user's query."""

    def get_tools(self) -> list[LLMToolDefinition]:
        """Return the tools available for this domain."""
        return [
            LLMToolDefinition(
                name="refinitiv.get_data",
                description="Execute a screening query against the Refinitiv database",
                parameters={
                    "type": "object",
                    "properties": {
                        "instruments": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List containing the SCREEN expression",
                        },
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of TR field codes to return",
                        },
                    },
                    "required": ["instruments", "fields"],
                },
            ),
        ]

    async def can_handle(self, query: str) -> float:
        """
        Check if this agent can handle the given query.

        Args:
            query: Natural language query

        Returns:
            Confidence score (0.0 to 1.0)
        """
        query_lower = query.lower()

        # Count matching keywords
        matches = sum(1 for kw in self.DOMAIN_KEYWORDS if kw in query_lower)

        # Boost for clear screening patterns
        if any(p in query_lower for p in ["top ", "largest ", "find ", "screen "]):
            matches += 2

        if matches == 0:
            return 0.0

        if matches >= 4:
            return 0.9
        elif matches >= 3:
            return 0.8
        elif matches >= 2:
            return 0.7
        elif matches >= 1:
            return 0.5

        return 0.0

    async def process(
        self,
        context: AgentContext,
    ) -> AgentResult:
        """
        Process a query and generate API calls.

        Args:
            context: AgentContext with query and context

        Returns:
            AgentResult with tool calls or clarification
        """
        # Try rule-based extraction first
        rule_result = self._try_rule_based_extraction(context)
        if rule_result and rule_result.confidence >= 0.8:
            return rule_result

        # Fall back to LLM
        return await super().process(context)

    def _try_rule_based_extraction(
        self,
        context: AgentContext,
    ) -> AgentResult | None:
        """
        Try to extract API call using rules.

        Args:
            context: AgentContext with query

        Returns:
            AgentResult if successful, None otherwise
        """
        query = context.query.lower()

        # Build SCREEN expression
        screen_parts = ["U(IN(Equity(active,public,primary)))"]
        display_fields = list(self.DEFAULT_DISPLAY_FIELDS)

        # Detect index filter
        index_filter = self._detect_index_filter(query)
        if index_filter:
            screen_parts.append(index_filter)

        # Detect country filter
        country_filter = self._detect_country_filter(query)
        if country_filter:
            screen_parts.append(country_filter)

        # Detect sector filter
        sector_filter = self._detect_sector_filter(query)
        if sector_filter:
            screen_parts.append(sector_filter)

        # Detect metric filters (e.g., PE < 15, ROE > 15)
        metric_filters = self._detect_metric_filters(query)
        for mf in metric_filters:
            screen_parts.append(mf["filter"])
            if mf["field"] not in display_fields:
                display_fields.append(mf["field"])

        # Detect TOP ranking
        top_clause = self._detect_top_clause(query)
        if top_clause:
            screen_parts.append(top_clause["clause"])
            if top_clause["field"] not in display_fields:
                display_fields.append(top_clause["field"])

        # Always add currency normalization
        screen_parts.append("CURN=USD")

        # Need at least some filters or ranking to be valid
        if len(screen_parts) <= 2:  # Just universe and CURN
            return None

        # Build SCREEN expression
        screen_expr = "SCREEN(" + ",".join(screen_parts) + ")"

        # Build tool call
        tool_call = ToolCall(
            tool_name="refinitiv.get_data",
            arguments={
                "instruments": [screen_expr],
                "fields": display_fields,
            },
        )

        return AgentResult(
            tool_calls=(tool_call,),
            confidence=0.85,
            reasoning=f"Rule-based screening: {screen_expr}",
            domain=self.domain_name,
        )

    def _detect_index_filter(self, query: str) -> str | None:
        """Detect index filter from query."""
        query_lower = query.lower()

        for index_name, index_code in self.INDEX_CODES.items():
            if index_name in query_lower:
                return f'IN(TR.IndexConstituentRIC,"{index_code}")'

        return None

    def _detect_country_filter(self, query: str) -> str | None:
        """Detect country filter from query."""
        query_lower = query.lower()

        for country_name, country_code in self.COUNTRY_CODES.items():
            if country_name in query_lower:
                return f"TR.HQCountryCode={country_code}"

        return None

    def _detect_sector_filter(self, query: str) -> str | None:
        """Detect sector filter from query."""
        query_lower = query.lower()

        for sector_name, sector_code in self.SECTOR_CODES.items():
            if sector_name in query_lower:
                return f"IN(TR.TRBCEconSectorCode,{sector_code})"

        return None

    def _detect_metric_filters(self, query: str) -> list[dict[str, str]]:
        """Detect metric filter expressions from query."""
        filters = []
        query_lower = query.lower()

        # Patterns for numeric comparisons
        patterns = [
            # "PE ratio below 15", "PE under 15"
            (r'(pe ratio?|pe|p/e)\s+(?:below|under|less than)\s+(\d+(?:\.\d+)?)', 'TR.PE', '<'),
            # "PE ratio above 15", "PE over 15"
            (r'(pe ratio?|pe|p/e)\s+(?:above|over|greater than|more than)\s+(\d+(?:\.\d+)?)', 'TR.PE', '>'),
            # "dividend yield above 2%"
            (r'dividend yield\s+(?:above|over|greater than|more than)\s+(\d+(?:\.\d+)?)', 'TR.DividendYield', '>'),
            # "dividend yield below 5%"
            (r'dividend yield\s+(?:below|under|less than)\s+(\d+(?:\.\d+)?)', 'TR.DividendYield', '<'),
            # "ROE above 15%"
            (r'roe\s+(?:above|over|greater than)\s+(\d+(?:\.\d+)?)', 'TR.ROE', '>'),
            (r'return on equity\s+(?:above|over|greater than)\s+(\d+(?:\.\d+)?)', 'TR.ROE', '>'),
            # "ROE below 10%"
            (r'roe\s+(?:below|under|less than)\s+(\d+(?:\.\d+)?)', 'TR.ROE', '<'),
            # "revenue growth over 20%"
            (r'revenue growth\s+(?:above|over|greater than|more than)\s+(\d+(?:\.\d+)?)', 'TR.RevenueGrowth', '>'),
            # "free cash flow" / "fcf" positive
            (r'(?:free cash flow|fcf)\s*(?:>|positive)', 'TR.FreeCashFlow', '>0'),
            (r'positive\s+(?:free cash flow|fcf)', 'TR.FreeCashFlow', '>0'),
            # "beat earnings by more than 10%", "beat earnings estimates by more than 10%"
            (r'(?:beat|earnings surprise|eps surprise).*?(?:by\s+)?(?:more than|over|above)\s+(\d+(?:\.\d+)?)', 'TR.EPSSurprisePct(Period=FQ0)', '>'),
            # "beat estimates by 10%" - simpler pattern
            (r'beat.*?(?:more than|over|by)\s+(\d+(?:\.\d+)?)\s*%?', 'TR.EPSSurprisePct(Period=FQ0)', '>'),
        ]

        for pattern_tuple in patterns:
            if len(pattern_tuple) == 3:
                pattern, field, op = pattern_tuple
                match = re.search(pattern, query_lower)
                if match:
                    if op == '>0':
                        filters.append({
                            "filter": f"{field}>0",
                            "field": field,
                        })
                    else:
                        # Get the numeric value from the match
                        groups = match.groups()
                        # Find the numeric value (last group that looks like a number)
                        value = None
                        for g in reversed(groups):
                            if g and re.match(r'^\d+(?:\.\d+)?$', g):
                                value = g
                                break
                        if value:
                            filters.append({
                                "filter": f"{field}{op}{value}",
                                "field": field,
                            })

        return filters

    def _detect_top_clause(self, query: str) -> dict[str, str] | None:
        """Detect TOP ranking clause from query."""
        query_lower = query.lower()

        # Detect count (top N)
        count = 10  # default
        count_match = re.search(r'(?:top|largest|biggest|highest)\s+(\d+)', query_lower)
        if count_match:
            count = int(count_match.group(1))

        # Detect ranking metric
        for metric_name, metric_field in self.RANKING_METRICS.items():
            if metric_name in query_lower:
                return {
                    "clause": f"TOP({metric_field},{count},nnumber)",
                    "field": metric_field,
                }

        # Check for implicit ranking (largest, biggest without metric)
        if any(kw in query_lower for kw in ["largest", "biggest", "top "]):
            # Default to market cap
            return {
                "clause": f"TOP(TR.CompanyMarketCap,{count},nnumber)",
                "field": "TR.CompanyMarketCap",
            }

        return None
