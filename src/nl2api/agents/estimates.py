"""
Estimates Domain Agent

Handles I/B/E/S Estimates API queries including:
- Consensus estimates (EPS, revenue, EBITDA)
- Analyst recommendations
- Price targets
- Earnings surprises
- Estimate revisions
"""

from __future__ import annotations

from CONTRACTS import ToolCall, ToolRegistry
from src.nl2api.agents.base import BaseDomainAgent
from src.nl2api.agents.protocols import AgentContext, AgentResult
from src.nl2api.llm.protocols import LLMProvider, LLMToolDefinition
from src.nl2api.rag.protocols import RAGRetriever


class EstimatesAgent(BaseDomainAgent):
    """
    Domain agent for I/B/E/S Estimates API.

    Handles natural language queries about:
    - EPS and revenue estimates
    - Analyst recommendations (buy/hold/sell)
    - Price targets
    - Earnings surprises (beat/miss)
    - Estimate revisions
    - Forward valuations (forward PE, PEG)
    """

    # Field code mappings for common estimate types
    ESTIMATE_FIELDS = {
        # Consensus mean estimates
        "eps": "TR.EPSMean",
        "earnings per share": "TR.EPSMean",
        "earnings": "TR.EPSMean",
        "revenue": "TR.RevenueMean",
        "sales": "TR.RevenueMean",
        "ebitda": "TR.EBITDAMean",
        "ebit": "TR.EBITMean",
        "operating income": "TR.EBITMean",
        "net income": "TR.NetProfitMean",
        "net profit": "TR.NetProfitMean",
        "profit": "TR.NetProfitMean",
        "free cash flow": "TR.FCFMean",
        "fcf": "TR.FCFMean",
        "cash flow per share": "TR.CFPSMean",
        "cfps": "TR.CFPSMean",
        "dividend": "TR.DPSMean",
        "dps": "TR.DPSMean",
        "dividend per share": "TR.DPSMean",
        "book value": "TR.BVPSMean",
        "book value per share": "TR.BVPSMean",
        "bvps": "TR.BVPSMean",
        "roe": "TR.ROEMean",
        "return on equity": "TR.ROEMean",
        "roa": "TR.ROAMean",
        "return on assets": "TR.ROAMean",
        "long term growth": "TR.LTGMean",
        "ltg": "TR.LTGMean",
    }

    # Actual/reported value fields
    ACTUAL_FIELDS = {
        "actual eps": "TR.EPSActValue",
        "reported eps": "TR.EPSActValue",
        "actual revenue": "TR.RevenueActValue",
        "reported revenue": "TR.RevenueActValue",
        "actual ebitda": "TR.EBITDAActValue",
    }

    # Surprise fields
    SURPRISE_FIELDS = {
        "surprise": "TR.EPSSurprisePct",
        "eps surprise": "TR.EPSSurprisePct",
        "earnings surprise": "TR.EPSSurprisePct",
        "revenue surprise": "TR.RevenueSurprise",
        "beat": "TR.EPSSurprisePct",
        "miss": "TR.EPSSurprisePct",
    }

    # Recommendation fields
    RECOMMENDATION_FIELDS = {
        "rating": "TR.RecMean",
        "recommendation": "TR.RecMean",
        "analyst rating": "TR.RecMean",
        "buy rating": "TR.NumBuys",
        "hold rating": "TR.NumHolds",
        "sell rating": "TR.NumSells",
        "price target": "TR.PriceTargetMean",
        "target price": "TR.PriceTargetMean",
    }

    # Keywords for domain classification
    DOMAIN_KEYWORDS = [
        "estimate",
        "forecast",
        "projection",
        "consensus",
        "eps",
        "earnings",
        "revenue",
        "sales",
        "ebitda",
        "analyst",
        "rating",
        "recommendation",
        "buy",
        "sell",
        "hold",
        "price target",
        "target price",
        "surprise",
        "beat",
        "miss",
        "revision",
        "upgrade",
        "downgrade",
        "forward pe",
        "peg ratio",
        "valuation",
        "ltg",
        "long-term growth",
    ]

    def __init__(
        self,
        llm: LLMProvider,
        rag: RAGRetriever | None = None,
    ):
        """Initialize the Estimates agent."""
        super().__init__(llm, rag)

    @property
    def domain_name(self) -> str:
        """Return the domain name."""
        return "estimates"

    @property
    def domain_description(self) -> str:
        """Return the domain description."""
        return (
            "I/B/E/S Estimates API - analyst forecasts for EPS, revenue, EBITDA; "
            "recommendations (buy/hold/sell); price targets; earnings surprises; "
            "estimate revisions; forward valuations"
        )

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Return the data types this agent handles."""
        return (
            "EPS estimates",
            "revenue forecasts",
            "analyst recommendations",
            "buy/hold/sell ratings",
            "price targets",
            "earnings surprises",
            "estimate revisions",
            "forward PE",
            "consensus estimates",
        )

    @property
    def example_queries(self) -> tuple[str, ...]:
        """Return example queries this agent handles well."""
        return (
            "What are the EPS estimates for Apple?",
            "Show analyst recommendations for Tesla",
            "What is the consensus price target for Microsoft?",
            "Did Amazon beat earnings last quarter?",
            "What is Google's forward PE ratio?",
        )

    def get_system_prompt(self) -> str:
        """Return the system prompt for the Estimates domain."""
        return """You are an expert at translating natural language queries into LSEG I/B/E/S Estimates API calls.

Your task is to generate accurate `get_data` tool calls based on the user's query about financial estimates.

## Key Field Codes

### Consensus Estimates
- `TR.EPSMean(Period=X)` - Mean EPS estimate
- `TR.RevenueMean(Period=X)` - Mean revenue estimate
- `TR.EBITDAMean(Period=X)` - Mean EBITDA estimate
- `TR.NetProfitMean(Period=X)` - Mean net income
- `TR.FCFMean(Period=X)` - Mean free cash flow
- `TR.DPSMean(Period=X)` - Mean dividend per share

### Actual Values
- `TR.EPSActValue(Period=X)` - Reported EPS
- `TR.RevenueActValue(Period=X)` - Reported revenue

### Earnings Surprise
- `TR.EPSSurprisePct(Period=X)` - EPS surprise percentage
- `TR.RevenueSurprise(Period=X)` - Revenue surprise

### Analyst Recommendations
- `TR.RecMean` - Mean recommendation (1=Buy, 5=Sell)
- `TR.NumBuys` - Number of buy ratings
- `TR.NumHolds` - Number of hold ratings
- `TR.NumSells` - Number of sell ratings
- `TR.PriceTargetMean` - Mean price target
- `TR.PriceTargetHigh` - Highest price target
- `TR.PriceTargetLow` - Lowest price target

### Estimate Statistics
- `TR.EPSHigh(Period=X)` - Highest EPS estimate
- `TR.EPSLow(Period=X)` - Lowest EPS estimate
- `TR.EPSNumIncEstimates(Period=X)` - Number of estimates
- `TR.EPSStdDev(Period=X)` - Standard deviation

### Estimate Revisions
- `TR.EPSMeanChgPct(Period=X)` - % change in mean estimate
- `TR.EPSNumUp(Period=X)` - Number of upward revisions
- `TR.EPSNumDown(Period=X)` - Number of downward revisions

### Forward Valuations
- `TR.PtoEPSMeanEst(Period=X)` - Forward P/E ratio
- `TR.EVToEBITDAMean(Period=X)` - Forward EV/EBITDA
- `TR.PEGRatio` - Price/Earnings to Growth ratio
- `TR.LTGMean` - Long-term growth estimate

## Period Parameters
- `FY0` - Current fiscal year (most recent actual)
- `FY1` - Next fiscal year estimate (default for estimates)
- `FY2` - Two years ahead
- `FQ0` - Current quarter (most recent)
- `FQ1` - Next quarter estimate

## Examples

Query: "What is Apple's EPS estimate?"
Tool call: get_data(RICs=["AAPL.O"], fields=["TR.EPSMean(Period=FY1)"])

Query: "Get Microsoft's revenue forecast for next quarter"
Tool call: get_data(RICs=["MSFT.O"], fields=["TR.RevenueMean(Period=FQ1)"])

Query: "What's the analyst rating for Tesla?"
Tool call: get_data(RICs=["TSLA.O"], fields=["TR.RecMean", "TR.NumBuys", "TR.NumHolds", "TR.NumSells"])

Query: "Did Amazon beat earnings last quarter?"
Tool call: get_data(RICs=["AMZN.O"], fields=["TR.EPSSurprisePct(Period=FQ0)"])

## Rules
1. Always use RIC codes for instruments (e.g., AAPL.O for Apple)
2. Default to FY1 for annual estimates if no period specified
3. Default to FQ1 for quarterly estimates if no period specified
4. Include relevant metadata fields when useful (e.g., TR.CommonName for multi-company queries)
5. If the query is ambiguous, ask for clarification

Generate the most appropriate get_data tool call for the user's query."""

    def get_tools(self) -> list[LLMToolDefinition]:
        """Return the tools available for this domain.

        Note: Uses canonical format (get_data with tickers) for fixture compatibility.
        The execution layer would convert to API-specific format when calling real APIs.
        """
        return [
            LLMToolDefinition(
                name=ToolRegistry.GET_DATA,  # Canonical name, not domain-specific
                description="Retrieve financial estimates data from the LSEG I/B/E/S database",
                parameters={
                    "type": "object",
                    "properties": {
                        "tickers": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of Reuters Instrument Codes in RIC format (e.g., ['AAPL.O', 'MSFT.O'])",
                        },
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of TR field codes with optional parameters (e.g., ['TR.EPSMean(Period=FY1)'])",
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Optional parameters for time series data (SDate, EDate, Frq)",
                            "properties": {
                                "SDate": {
                                    "type": "string",
                                    "description": "Start date (e.g., '-1Y', '2023-01-01')",
                                },
                                "EDate": {
                                    "type": "string",
                                    "description": "End date (e.g., '0D', '2024-12-31')",
                                },
                                "Frq": {
                                    "type": "string",
                                    "description": "Frequency (D, W, M, Q, Y)",
                                },
                            },
                        },
                    },
                    "required": ["tickers", "fields"],
                },
            ),
        ]

    async def can_handle(self, query: str) -> float:
        """
        Check if this agent can handle the given query.

        Uses keyword matching to determine relevance.

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
        rics = list(context.resolved_entities.values()) if context.resolved_entities else []

        if not rics:
            # Can't proceed without entities
            return None

        # Detect period
        period = self._detect_period(query)

        # Detect metrics
        fields = self._detect_fields(query, period)

        if not fields:
            return None

        # Build tool call (canonical format)
        tool_call = ToolCall(
            tool_name=ToolRegistry.GET_DATA,
            arguments={
                "tickers": rics,
                "fields": fields,
            },
        )

        return AgentResult(
            tool_calls=(tool_call,),
            confidence=0.85,
            reasoning=f"Rule-based extraction: detected fields {fields} for period {period}",
            domain=self.domain_name,
        )

    def _detect_period(self, query: str) -> str:
        """Detect the time period from the query."""
        query_lower = query.lower()

        # Quarterly patterns
        if any(p in query_lower for p in ["next quarter", "quarterly", "q1", "q2", "q3", "q4"]):
            return "FQ1"
        if any(p in query_lower for p in ["last quarter", "previous quarter"]):
            return "FQ0"

        # Annual patterns
        if any(p in query_lower for p in ["next year", "fy2", "two years"]):
            return "FY2"
        if any(p in query_lower for p in ["three years", "fy3"]):
            return "FY3"

        # Default to FY1 for annual estimates
        return "FY1"

    def _detect_fields(self, query: str, period: str) -> list[str]:
        """Detect the fields to request based on the query."""
        query_lower = query.lower()
        fields = []

        def word_match(keyword: str, text: str) -> bool:
            """Check if keyword matches as a word (not substring)."""
            import re

            # Escape special regex chars in keyword
            pattern = r"\b" + re.escape(keyword) + r"\b"
            return bool(re.search(pattern, text))

        # Check for estimate types - order by specificity (longer keywords first)
        sorted_estimates = sorted(self.ESTIMATE_FIELDS.items(), key=lambda x: -len(x[0]))
        for keyword, field_base in sorted_estimates:
            if word_match(keyword, query_lower):
                # Add period parameter
                fields.append(f"{field_base}(Period={period})")
                break  # Take first match

        # Check for actual values
        for keyword, field_base in self.ACTUAL_FIELDS.items():
            if keyword in query_lower:
                fields.append(f"{field_base}(Period={period})")

        # Check for surprise
        for keyword, field_base in self.SURPRISE_FIELDS.items():
            if keyword in query_lower:
                # Surprise typically uses FQ0
                fields.append(f"{field_base}(Period=FQ0)")
                break

        # Check for recommendations
        for keyword, field in self.RECOMMENDATION_FIELDS.items():
            if keyword in query_lower:
                if "rating" in query_lower or "recommendation" in query_lower:
                    # Full recommendation package
                    fields.extend(["TR.RecMean", "TR.NumBuys", "TR.NumHolds", "TR.NumSells"])
                    break
                elif "price target" in query_lower or "target price" in query_lower:
                    fields.extend(["TR.PriceTargetMean", "TR.PriceTargetHigh", "TR.PriceTargetLow"])
                    break
                else:
                    fields.append(field)
                    break

        # Check for valuation
        if "forward pe" in query_lower or "forward p/e" in query_lower:
            fields.append(f"TR.PtoEPSMeanEst(Period={period})")
        if "peg" in query_lower:
            fields.append("TR.PEGRatio")
        if "long-term growth" in query_lower or "ltg" in query_lower:
            fields.append("TR.LTGMean")

        # Deduplicate while preserving order
        seen = set()
        unique_fields = []
        for f in fields:
            if f not in seen:
                seen.add(f)
                unique_fields.append(f)

        return unique_fields
