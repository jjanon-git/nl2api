"""
Datastream Domain Agent

Handles Datastream API queries including:
- Price and trading data
- Valuation metrics
- Dividend data
- Time series data
"""

from __future__ import annotations

import re
from typing import Any

from CONTRACTS import ToolCall
from src.nl2api.agents.base import BaseDomainAgent
from src.nl2api.agents.protocols import AgentContext, AgentResult
from src.nl2api.llm.protocols import LLMProvider, LLMToolDefinition
from src.nl2api.rag.protocols import RAGRetriever


class DatastreamAgent(BaseDomainAgent):
    """
    Domain agent for LSEG Datastream Web Service (DSWS).

    Handles natural language queries about:
    - Stock prices (open, high, low, close, volume)
    - Valuation metrics (market cap, PE, EV)
    - Dividend data (yield, DPS, ex-date)
    - Time series and historical data
    - Index data
    """

    # Price field mappings
    PRICE_FIELDS = {
        "price": "P",
        "stock price": "P",
        "share price": "P",
        "closing price": "P",
        "close": "P",
        "open": "PO",
        "opening price": "PO",
        "high": "PH",
        "highest price": "PH",
        "intraday high": "PH",
        "low": "PL",
        "lowest price": "PL",
        "intraday low": "PL",
        "volume": "VO",
        "trading volume": "VO",
        "shares traded": "VO",
        "bid price": "PB",
        "bid": "PB",
        "ask price": "PA",
        "ask": "PA",
        "offer price": "PA",
        "last price": "LP",
        "latest price": "LP",
        "ohlc": ["PO", "PH", "PL", "P"],
        "ohlcv": ["PO", "PH", "PL", "P", "VO"],
    }

    # Valuation field mappings
    VALUATION_FIELDS = {
        "market cap": "MV",
        "market capitalization": "MV",
        "market value": "MV",
        "pe ratio": "PE",
        "pe": "PE",
        "price to earnings": "PE",
        "p/e": "PE",
        "pb ratio": "PTBV",
        "price to book": "PTBV",
        "p/b": "PTBV",
        "enterprise value": "EV",
        "ev": "EV",
        "ev/ebitda": "EVEBID",
        "price to sales": "PS",
        "p/s": "PS",
    }

    # Dividend field mappings
    DIVIDEND_FIELDS = {
        "dividend yield": "DY",
        "yield": "DY",
        "dividend": "DPS",
        "dividend per share": "DPS",
        "dps": "DPS",
        "ex-dividend date": "EXDT",
        "ex-date": "EXDT",
        "payout ratio": "POUT",
    }

    # Company info field mappings
    INFO_FIELDS = {
        "name": "NAME",
        "company name": "NAME",
        "sector": "SECTOR",
        "industry": "SECTOR",
        "country": "GEOG",
        "currency": "CURR",
        "exchange": "EXCH",
        "isin": "ISIN",
        "ticker": "MNEM",
        "symbol": "MNEM",
    }

    # Fundamentals via Worldscope codes
    FUNDAMENTAL_FIELDS = {
        "revenue": "WC01001",
        "sales": "WC01001",
        "net income": "WC01751",
        "earnings": "WC01751",
        "profit": "WC01751",
        "eps": "EPS",
        "earnings per share": "EPS",
        "ebitda": "WC18198",
        "operating income": "WC01250",
        "total assets": "WC02999",
        "total debt": "WC03255",
        "cash": "WC02001",
        "free cash flow": "WC04860",
        "fcf": "WC04860",
        "roe": "WC08301",
        "return on equity": "WC08301",
        "roa": "WC08326",
        "return on assets": "WC08326",
    }

    # Known US market prefixes (@ prefix for Datastream)
    US_COMPANY_PATTERNS = {
        "apple": "@AAPL",
        "microsoft": "@MSFT",
        "google": "@GOOGL",
        "alphabet": "@GOOGL",
        "amazon": "@AMZN",
        "tesla": "@TSLA",
        "nvidia": "@NVDA",
        "meta": "@META",
        "facebook": "@META",
        "netflix": "@NFLX",
        "disney": "@DIS",
        "coca-cola": "@KO",
        "pepsi": "@PEP",
        "jpmorgan": "@JPM",
        "goldman sachs": "@GS",
        "ibm": "@IBM",
        "intel": "@INTC",
        "amd": "@AMD",
        "walmart": "@WMT",
        "exxon": "@XOM",
        "chevron": "@CVX",
    }

    # Domain keywords for classification
    DOMAIN_KEYWORDS = [
        "price", "stock price", "share price",
        "open", "high", "low", "close", "volume", "ohlc",
        "market cap", "market capitalization",
        "pe ratio", "p/e", "price to earnings",
        "dividend", "yield",
        "historical", "time series",
        "trading", "quote", "quotes",
        "index", "indices",
    ]

    def __init__(
        self,
        llm: LLMProvider,
        rag: RAGRetriever | None = None,
    ):
        """Initialize the Datastream agent."""
        super().__init__(llm, rag)

    @property
    def domain_name(self) -> str:
        """Return the domain name."""
        return "datastream"

    @property
    def domain_description(self) -> str:
        """Return the domain description."""
        return (
            "Datastream API - stock prices, trading data, market cap, "
            "PE ratios, dividend yields, historical time series, index data"
        )

    def get_system_prompt(self) -> str:
        """Return the system prompt for the Datastream domain."""
        return """You are an expert at translating natural language queries into LSEG Datastream API calls.

Your task is to generate accurate `get_data` tool calls based on the user's query about financial market data.

## Key Field Codes

### Price & Trading Data
- `P` - Adjusted closing price (default price field)
- `PO` - Opening price
- `PH` - High price
- `PL` - Low price
- `VO` - Trading volume
- `LP` - Last traded price

### Valuation Metrics
- `MV` - Market capitalization
- `PE` - Price/Earnings ratio
- `PTBV` - Price to book value
- `EV` - Enterprise value
- `EVEBID` - EV/EBITDA

### Dividend Data
- `DY` - Dividend yield %
- `DPS` - Dividend per share
- `EXDT` - Ex-dividend date
- `POUT` - Payout ratio

### Company Information
- `NAME` - Security name
- `SECTOR` - Industry sector
- `GEOG` - Geographic region
- `CURR` - Trading currency

### Fundamentals (Worldscope)
- `WC01001` - Revenue/sales
- `WC01751` - Net income
- `EPS` - Earnings per share
- `WC08301` - Return on equity

## Ticker Format

US equities use the `U:` prefix:
- Apple: `U:AAPL`
- Microsoft: `U:MSFT`
- Tesla: `U:TSLA`

UK equities have no prefix:
- Barclays: `BARC`
- Vodafone: `VOD`

Indices use mnemonics:
- S&P 500: `S&PCOMP`
- FTSE 100: `FTSE100`
- NASDAQ: `NASCOMP`

## Date Format

Relative dates:
- `0D` - Today
- `-1D` - Yesterday
- `-1M` - 1 month ago
- `-1Y` - 1 year ago
- `-5Y` - 5 years ago

Frequency codes:
- `D` - Daily
- `W` - Weekly
- `M` - Monthly
- `Q` - Quarterly
- `Y` - Yearly

## Examples

Query: "What is Apple's current stock price?"
Tool call: get_data(tickers=["U:AAPL"], fields=["P"])

Query: "Get Microsoft's opening, high, low, close for the past month"
Tool call: get_data(tickers=["U:MSFT"], fields=["PO", "PH", "PL", "P"], SDate="-1M", EDate="0D", Frq="D")

Query: "What is Tesla's market cap and PE ratio?"
Tool call: get_data(tickers=["U:TSLA"], fields=["MV", "PE"])

Query: "Compare dividend yields for Apple and Microsoft"
Tool call: get_data(tickers=["U:AAPL", "U:MSFT"], fields=["DY"])

Query: "S&P 500 level yesterday"
Tool call: get_data(tickers=["S&PCOMP"], fields=["PI"], SDate="-1D", EDate="-1D")

## Rules
1. Always use the appropriate ticker prefix for the market
2. For US stocks, use U: prefix (e.g., U:AAPL)
3. Use appropriate date parameters for historical queries
4. Combine multiple fields when user asks for related data
5. If the query is ambiguous, ask for clarification

Generate the most appropriate get_data tool call for the user's query."""

    def get_tools(self) -> list[LLMToolDefinition]:
        """Return the tools available for this domain."""
        return [
            LLMToolDefinition(
                name="datastream.get_data",
                description="Retrieve market data from LSEG Datastream",
                parameters={
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "string",
                            "description": "Datastream ticker(s), use @ prefix for US stocks (e.g., '@AAPL')",
                        },
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of field codes (e.g., ['P', 'MV', 'PE'])",
                        },
                        "start": {
                            "type": "string",
                            "description": "Start date for time series (e.g., '-1Y', '-30D')",
                        },
                        "end": {
                            "type": "string",
                            "description": "End date for time series (e.g., '0D')",
                        },
                        "freq": {
                            "type": "string",
                            "description": "Frequency (D=daily, W=weekly, M=monthly, Q=quarterly, Y=yearly)",
                        },
                    },
                    "required": ["tickers", "fields"],
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

        if matches == 0:
            return 0.0

        # Higher score for more matches
        if matches >= 3:
            return 0.9
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

        First tries rule-based extraction, then falls back to LLM.

        Args:
            context: AgentContext with query and context

        Returns:
            AgentResult with tool calls or clarification
        """
        # Try rule-based extraction first for simple queries
        rule_result = self._try_rule_based_extraction(context)
        if rule_result and rule_result.confidence >= 0.8:
            return rule_result

        # Fall back to LLM-based generation
        return await super().process(context)

    def _try_rule_based_extraction(
        self,
        context: AgentContext,
    ) -> AgentResult | None:
        """
        Try to extract API call using rules.

        Handles common query patterns without LLM.

        Args:
            context: AgentContext with query

        Returns:
            AgentResult if successful, None otherwise
        """
        query = context.query.lower()
        tickers = self._extract_tickers(query, context.resolved_entities)

        if not tickers:
            return None

        # Detect fields
        fields = self._detect_fields(query)

        if not fields:
            return None

        # Detect time range
        time_params = self._detect_time_range(query)

        # Build tool call - tickers as single string for simple case
        ticker_value = tickers[0] if len(tickers) == 1 else tickers
        arguments: dict[str, Any] = {
            "tickers": ticker_value,
            "fields": fields,
        }

        if time_params:
            arguments.update(time_params)

        tool_call = ToolCall(
            tool_name="datastream.get_data",
            arguments=arguments,
        )

        return AgentResult(
            tool_calls=(tool_call,),
            confidence=0.85,
            reasoning=f"Rule-based extraction: detected fields {fields} for tickers {tickers}",
            domain=self.domain_name,
        )

    def _extract_tickers(
        self,
        query: str,
        resolved_entities: dict[str, str],
    ) -> list[str]:
        """Extract tickers from query and resolved entities."""
        tickers = []

        # Use resolved entities first (these should be in RIC format)
        if resolved_entities:
            for name, ric in resolved_entities.items():
                # Convert RIC to Datastream format if needed
                ticker = self._ric_to_datastream(ric)
                tickers.append(ticker)

        # Also check for known company patterns
        if not tickers:
            query_lower = query.lower()
            for company, ticker in self.US_COMPANY_PATTERNS.items():
                if company in query_lower:
                    tickers.append(ticker)

        return tickers

    def _ric_to_datastream(self, ric: str) -> str:
        """Convert a RIC code to Datastream ticker format."""
        # Common conversions - use @ prefix for US stocks
        if ric.endswith(".O"):  # NASDAQ
            return f"@{ric[:-2]}"
        elif ric.endswith(".N"):  # NYSE
            return f"@{ric[:-2]}"
        elif ric.endswith(".L"):  # London
            return ric[:-2]  # No prefix for UK

        return ric

    def _detect_fields(self, query: str) -> list[str]:
        """Detect the fields to request based on the query."""
        query_lower = query.lower()
        fields = []

        def add_fields(field_value: str | list[str]) -> None:
            if isinstance(field_value, list):
                for f in field_value:
                    if f not in fields:
                        fields.append(f)
            elif field_value not in fields:
                fields.append(field_value)

        # Check for OHLC patterns first (compound fields)
        if "ohlcv" in query_lower or ("ohlc" in query_lower and "volume" in query_lower):
            add_fields(self.PRICE_FIELDS["ohlcv"])
            return fields
        elif "ohlc" in query_lower:
            add_fields(self.PRICE_FIELDS["ohlc"])
            return fields

        # Check each field category - order by specificity (longer phrases first)
        all_fields = [
            (self.PRICE_FIELDS, True),
            (self.VALUATION_FIELDS, True),
            (self.DIVIDEND_FIELDS, True),
            (self.INFO_FIELDS, False),
            (self.FUNDAMENTAL_FIELDS, True),
        ]

        for field_map, single_match in all_fields:
            sorted_keys = sorted(field_map.keys(), key=len, reverse=True)
            for keyword in sorted_keys:
                if keyword in query_lower:
                    add_fields(field_map[keyword])
                    if single_match:
                        break  # Take first match per category

        # Default to price if no specific field detected
        if not fields:
            fields = ["P"]

        return fields

    def _detect_time_range(self, query: str) -> dict[str, str] | None:
        """Detect time range parameters from the query."""
        query_lower = query.lower()
        params: dict[str, str] = {}

        # Historical patterns - use start/end/freq parameter names
        historical_patterns = [
            (r"past (\d+) year", lambda m: {
                "start": f"-{m.group(1)}Y", "end": "0D", "freq": "D"
            }),
            (r"last (\d+) year", lambda m: {
                "start": f"-{m.group(1)}Y", "end": "0D", "freq": "D"
            }),
            # "for the past year", "over the past year"
            (r"(?:for |over )?the past year", lambda m: {
                "start": "-1Y", "end": "0D", "freq": "D"
            }),
            (r"past (\d+) month", lambda m: {
                "start": f"-{m.group(1)}M", "end": "0D", "freq": "D"
            }),
            (r"last (\d+) month", lambda m: {
                "start": f"-{m.group(1)}M", "end": "0D", "freq": "D"
            }),
            (r"last (\d+) day", lambda m: {
                "start": f"-{m.group(1)}D", "end": "0D", "freq": "D"
            }),
            (r"past (\d+) day", lambda m: {
                "start": f"-{m.group(1)}D", "end": "0D", "freq": "D"
            }),
            (r"past (\d+) week", lambda m: {
                "start": f"-{m.group(1)}W", "end": "0D", "freq": "D"
            }),
            (r"yesterday", lambda m: {
                "start": "-1D", "end": "-1D"
            }),
            (r"last week", lambda m: {
                "start": "-1W", "end": "0D", "freq": "D"
            }),
            (r"last month", lambda m: {
                "start": "-1M", "end": "0D", "freq": "D"
            }),
            (r"ytd|year.to.date", lambda m: {
                "start": "-0Y", "end": "0D", "freq": "D"
            }),
            (r"historical|time series|history", lambda m: {
                "start": "-1Y", "end": "0D", "freq": "D"
            }),
        ]

        for pattern, extractor in historical_patterns:
            match = re.search(pattern, query_lower)
            if match:
                params = extractor(match)
                break

        # Frequency adjustments
        if "weekly" in query_lower:
            params["freq"] = "W"
        elif "monthly" in query_lower:
            params["freq"] = "M"
        elif "quarterly" in query_lower:
            params["freq"] = "Q"
        elif "annual" in query_lower or "yearly" in query_lower:
            params["freq"] = "Y"

        return params if params else None
