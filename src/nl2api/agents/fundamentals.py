"""
Fundamentals Domain Agent

Handles Refinitiv Fundamentals API queries including:
- Income statement (revenue, net income, EBITDA)
- Balance sheet (assets, liabilities, equity)
- Cash flow (operating, investing, financing)
- Financial ratios (ROE, ROA, margins)
- Per share data (EPS, DPS, BVPS)
"""

from __future__ import annotations

import re
from typing import Any

from CONTRACTS import ToolCall
from src.nl2api.agents.base import BaseDomainAgent
from src.nl2api.agents.protocols import AgentContext, AgentResult
from src.nl2api.llm.protocols import LLMProvider, LLMToolDefinition
from src.nl2api.rag.protocols import RAGRetriever


class FundamentalsAgent(BaseDomainAgent):
    """
    Domain agent for Refinitiv Fundamentals API.

    Handles natural language queries about:
    - Income statement items (revenue, costs, profit)
    - Balance sheet items (assets, liabilities, equity)
    - Cash flow items (operating, investing, financing)
    - Financial ratios (profitability, liquidity, leverage)
    - Per share metrics (EPS, DPS, BVPS)
    """

    # Income statement field mappings
    INCOME_STATEMENT_FIELDS = {
        # Revenue
        "revenue": "TR.Revenue",
        "sales": "TR.Revenue",
        "total revenue": "TR.Revenue",
        "net sales": "TR.Revenue",
        "turnover": "TR.Revenue",
        # Costs
        "cogs": "TR.CostofRevenueTotal",
        "cost of goods sold": "TR.CostofRevenueTotal",
        "cost of sales": "TR.CostofRevenueTotal",
        "cost of revenue": "TR.CostofRevenueTotal",
        "sga": "TR.SGandAExp",
        "sg&a": "TR.SGandAExp",
        "selling general admin": "TR.SGandAExp",
        "r&d": "TR.ResearchAndDevelopment",
        "research and development": "TR.ResearchAndDevelopment",
        "depreciation": "TR.DepreciationAmort",
        "d&a": "TR.DepreciationAmort",
        "depreciation and amortization": "TR.DepreciationAmort",
        "operating expenses": "TR.TotalOperatingExpense",
        "interest expense": "TR.InterestExpense",
        "income tax": "TR.IncomeTaxExpense",
        "tax expense": "TR.IncomeTaxExpense",
        # Profit metrics
        "gross profit": "TR.GrossProfit",
        "gross income": "TR.GrossProfit",
        "operating income": "TR.OperatingIncome",
        "operating profit": "TR.OperatingIncome",
        "ebit": "TR.EBIT",
        "ebitda": "TR.EBITDA",
        "pretax income": "TR.NetIncomeBeforeTaxes",
        "income before tax": "TR.NetIncomeBeforeTaxes",
        "ebt": "TR.NetIncomeBeforeTaxes",
        "net income": "TR.NetIncome",
        "net profit": "TR.NetIncome",
        "earnings": "TR.NetIncome",
        "profit": "TR.NetIncome",
        "net income after taxes": "TR.NetIncomeAfterTaxes",
    }

    # Balance sheet - Assets
    BALANCE_SHEET_ASSETS = {
        "cash": "TR.Cash",
        "cash position": "TR.CashAndSTInvestments",
        "cash and short-term investments": "TR.CashAndSTInvestments",
        "cash and st investments": "TR.CashAndSTInvestments",
        "receivables": "TR.TotalReceivablesNet",
        "accounts receivable": "TR.TotalReceivablesNet",
        "inventory": "TR.Inventories",
        "inventories": "TR.Inventories",
        "current assets": "TR.CurrentAssets",
        "total assets": "TR.TotalAssets",
        "assets": "TR.TotalAssets",
    }

    # Balance sheet - Liabilities
    BALANCE_SHEET_LIABILITIES = {
        "accounts payable": "TR.AccountsPayable",
        "payables": "TR.AccountsPayable",
        "current liabilities": "TR.CurrentLiabilities",
        "long-term debt": "TR.LongTermDebt",
        "lt debt": "TR.LongTermDebt",
        "total debt": "TR.TotalDebt",
        "debt": "TR.TotalDebt",
        "total liabilities": "TR.TotalLiabilities",
        "liabilities": "TR.TotalLiabilities",
    }

    # Balance sheet - Equity
    BALANCE_SHEET_EQUITY = {
        "preferred stock": "TR.PreferredStockNet",
        "common equity": "TR.CommonEquity",
        "shareholders equity": "TR.TotalEquity",
        "total equity": "TR.TotalEquity",
        "equity": "TR.TotalEquity",
        "minority interest": "TR.MinorityInterestBSStmt",
        "total liabilities and equity": "TR.TtlLiabShareholderEqty",
    }

    # Cash flow fields
    CASH_FLOW_FIELDS = {
        "operating cash flow": "TR.OperatingCashFlow",
        "cash flow from operations": "TR.OperatingCashFlow",
        "cfo": "TR.OperatingCashFlow",
        "capex": "TR.CapitalExpenditures",
        "capital expenditures": "TR.CapitalExpenditures",
        "free cash flow": "TR.FreeCashFlow",
        "fcf": "TR.FreeCashFlow",
        "dividends paid": "TR.DividendsPaid",
    }

    # Financial ratios
    RATIO_FIELDS = {
        # Profitability
        "roe": "TR.ROE",
        "return on equity": "TR.ROE",
        "roa": "TR.ROA",
        "return on assets": "TR.ROA",
        "gross margin": "TR.GrossMargin",
        "operating margin": "TR.OperatingMargin",
        "net margin": "TR.NetProfitMargin",
        "profit margin": "TR.NetProfitMargin",
        "net profit margin": "TR.NetProfitMargin",
        # Liquidity
        "current ratio": "TR.CurrentRatio",
        "quick ratio": "TR.QuickRatio",
        # Leverage
        "debt to equity": "TR.DebtToEquity",
        "d/e": "TR.DebtToEquity",
        "net debt to ebitda": "TR.NetDebtToEBITDA",
        # Valuation
        "pe ratio": "TR.PE",
        "p/e": "TR.PE",
        "price to book": "TR.PriceToBVPerShare",
        "price to sales": "TR.PriceToSalesPerShare",
        "price to cash flow": "TR.PricetoCFPerShare",
        "ev/ebitda": "TR.EVToEBITDA",
        "dividend yield": "TR.DividendYield",
        "market cap": "TR.CompanyMarketCap",
        "market capitalization": "TR.CompanyMarketCap",
        "enterprise value": "TR.EV",
        "ev": "TR.EV",
        "beta": "TR.Beta",
    }

    # Per share data
    PER_SHARE_FIELDS = {
        "eps": "TR.BasicEPS",
        "earnings per share": "TR.BasicEPS",
        "basic eps": "TR.BasicEPS",
        "diluted eps": "TR.DilutedNormalizedEps",
        "book value per share": "TR.BookValuePerShare",
        "bvps": "TR.BookValuePerShare",
        "cash flow per share": "TR.CFPSActValue",
        "cfps": "TR.CFPSActValue",
        "fcf per share": "TR.FCFPSActValue",
        "dividends per share": "TR.DpsCommonStock",
        "dps": "TR.DpsCommonStock",
        "sales per share": "TR.SalesPerShare",
        "shares outstanding": "TR.SharesOutstanding",
    }

    # Company information
    COMPANY_INFO_FIELDS = {
        "company name": "TR.CommonName",
        "ticker": "TR.TickerSymbol",
        "exchange": "TR.ExchangeName",
        "country": "TR.HeadquartersCountry",
        "sector": "TR.GICSSector",
        "industry": "TR.TRBCIndustryGroup",
        "employees": "TR.Employees",
        "number of employees": "TR.Employees",
    }

    # Keywords for domain classification
    DOMAIN_KEYWORDS = [
        # Income statement
        "revenue", "sales", "net income", "profit", "earnings",
        "gross profit", "operating income", "ebit", "ebitda",
        "cost of goods", "cogs", "expenses", "margin",
        # Balance sheet
        "assets", "liabilities", "equity", "debt", "cash",
        "balance sheet", "receivables", "inventory", "payables",
        # Cash flow
        "cash flow", "operating cash", "free cash flow", "fcf",
        "capex", "capital expenditure", "dividends paid",
        # Ratios
        "roe", "roa", "current ratio", "quick ratio",
        "debt to equity", "profit margin", "gross margin",
        "return on", "leverage",
        # Per share
        "per share", "eps", "dps", "bvps",
        # General
        "fundamentals", "financials", "financial data",
        "annual report", "quarterly report", "fiscal",
    ]

    # Known company patterns for rule-based extraction
    KNOWN_COMPANIES = {
        "apple": "AAPL.O",
        "microsoft": "MSFT.O",
        "google": "GOOGL.O",
        "alphabet": "GOOGL.O",
        "amazon": "AMZN.O",
        "tesla": "TSLA.O",
        "nvidia": "NVDA.O",
        "meta": "META.O",
        "facebook": "META.O",
        "netflix": "NFLX.O",
    }

    def __init__(
        self,
        llm: LLMProvider,
        rag: RAGRetriever | None = None,
    ):
        """Initialize the Fundamentals agent."""
        super().__init__(llm, rag)
        # Combine all field mappings for easy lookup
        self._all_fields = {
            **self.INCOME_STATEMENT_FIELDS,
            **self.BALANCE_SHEET_ASSETS,
            **self.BALANCE_SHEET_LIABILITIES,
            **self.BALANCE_SHEET_EQUITY,
            **self.CASH_FLOW_FIELDS,
            **self.RATIO_FIELDS,
            **self.PER_SHARE_FIELDS,
            **self.COMPANY_INFO_FIELDS,
        }

    @property
    def domain_name(self) -> str:
        """Return the domain name."""
        return "fundamentals"

    @property
    def domain_description(self) -> str:
        """Return the domain description."""
        return (
            "Refinitiv Fundamentals API - income statement, balance sheet, "
            "cash flow, financial ratios, and per share data for companies"
        )

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Return the data types this agent handles."""
        return (
            "income statement",
            "balance sheet",
            "cash flow statement",
            "revenue and profit",
            "assets and liabilities",
            "financial ratios",
            "ROE and ROA",
            "profit margins",
            "EPS and book value",
            "debt ratios",
        )

    @property
    def example_queries(self) -> tuple[str, ...]:
        """Return example queries this agent handles well."""
        return (
            "What is Apple's revenue for the last 3 years?",
            "Show Microsoft's balance sheet",
            "What is Tesla's debt to equity ratio?",
            "Get Amazon's free cash flow",
            "What is Google's operating margin?",
        )

    def get_system_prompt(self) -> str:
        """Return the system prompt for the Fundamentals domain."""
        return """You are an expert at translating natural language queries into LSEG Refinitiv Fundamentals API calls.

Your task is to generate accurate `refinitiv.get_data` tool calls based on the user's query about company financials.

## Key Field Codes

### Income Statement
- `TR.Revenue` - Total Revenue
- `TR.GrossProfit` - Gross Profit
- `TR.OperatingIncome` - Operating Income
- `TR.EBIT` - Earnings Before Interest & Taxes
- `TR.EBITDA` - EBITDA
- `TR.NetIncomeBeforeTaxes` - Pretax Income
- `TR.NetIncome` - Net Income

### Balance Sheet
- `TR.CashAndSTInvestments` - Cash & Short-Term Investments
- `TR.TotalReceivablesNet` - Receivables
- `TR.Inventories` - Inventory
- `TR.CurrentAssets` - Current Assets
- `TR.TotalAssets` - Total Assets
- `TR.AccountsPayable` - Accounts Payable
- `TR.CurrentLiabilities` - Current Liabilities
- `TR.LongTermDebt` - Long-Term Debt
- `TR.TotalDebt` - Total Debt (use for generic debt queries)
- `TR.TotalLiabilities` - Total Liabilities
- `TR.TotalEquity` - Total Equity

### Cash Flow
- `TR.OperatingCashFlow` - Operating Cash Flow
- `TR.CapitalExpenditures` - Capital Expenditures
- `TR.FreeCashFlow` - Free Cash Flow
- `TR.DividendsPaid` - Dividends Paid

### Financial Ratios
- `TR.ROE` - Return on Equity
- `TR.ROA` - Return on Assets
- `TR.GrossMargin` - Gross Margin
- `TR.OperatingMargin` - Operating Margin
- `TR.NetProfitMargin` - Net Profit Margin
- `TR.CurrentRatio` - Current Ratio
- `TR.QuickRatio` - Quick Ratio
- `TR.DebtToEquity` - Debt to Equity

### Valuation
- `TR.CompanyMarketCap` - Market Capitalization
- `TR.EV` - Enterprise Value
- `TR.PE` - Price/Earnings Ratio
- `TR.PriceToBVPerShare` - Price to Book Value
- `TR.EVToEBITDA` - EV to EBITDA
- `TR.DividendYield` - Dividend Yield

### Per Share Data
- `TR.BasicEPS` - Basic EPS
- `TR.DilutedNormalizedEps` - Diluted EPS
- `TR.BookValuePerShare` - Book Value Per Share
- `TR.DpsCommonStock` - Dividends Per Share
- `TR.SharesOutstanding` - Shares Outstanding

## Parameters
- `Period`: Fiscal period
  - `FY0` - Current/most recent fiscal year
  - `FY-1`, `FY-2` - Prior years
  - `FQ0` - Current quarter
  - `FQ-1` - Prior quarter
- `Frq`: Frequency
  - `FY` - Fiscal Year
  - `FQ` - Fiscal Quarter
- `SDate`: Start date for time series (e.g., "0" for current, "-2" for 2 periods ago)
- `EDate`: End date for time series

## Examples

Query: "What was Apple's revenue last year?"
Tool call: refinitiv.get_data(instruments=["AAPL.O"], fields=["TR.Revenue"], parameters={"Period": "FY0", "Frq": "FY"})

Query: "What are Google's total assets and total debt?"
Tool call: refinitiv.get_data(instruments=["GOOGL.O"], fields=["TR.TotalAssets", "TR.TotalDebt"], parameters={"Period": "FY0"})

Query: "Get Microsoft's net income and operating income for the last 3 years"
Tool call: refinitiv.get_data(instruments=["MSFT.O"], fields=["TR.NetIncome", "TR.OperatingIncome"], parameters={"SDate": "0", "EDate": "-2", "Frq": "FY"})

Query: "What is NVIDIA's ROE, ROA, and profit margin?"
Tool call: refinitiv.get_data(instruments=["NVDA.O"], fields=["TR.ROE", "TR.ROA", "TR.NetProfitMargin"], parameters={"Period": "FY0"})

## Rules
1. Always use RIC codes for instruments (e.g., AAPL.O for Apple on NASDAQ)
2. Default to FY0 (current fiscal year) if no period specified
3. For time series, use SDate/EDate with numeric offsets
4. Include all relevant fields mentioned in the query
5. If the query is ambiguous, ask for clarification

Generate the most appropriate refinitiv.get_data tool call for the user's query."""

    def get_tools(self) -> list[LLMToolDefinition]:
        """Return the tools available for this domain."""
        return [
            LLMToolDefinition(
                name="refinitiv_get_data",
                description="Retrieve fundamental financial data from Refinitiv",
                parameters={
                    "type": "object",
                    "properties": {
                        "instruments": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of Reuters Instrument Codes (e.g., ['AAPL.O', 'MSFT.O'])",
                        },
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of TR field codes (e.g., ['TR.Revenue', 'TR.NetIncome'])",
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Optional parameters for data retrieval",
                            "properties": {
                                "Period": {
                                    "type": "string",
                                    "description": "Fiscal period (e.g., 'FY0', 'FQ1')",
                                },
                                "Frq": {
                                    "type": "string",
                                    "description": "Frequency (FY for annual, FQ for quarterly)",
                                },
                                "SDate": {
                                    "type": "string",
                                    "description": "Start date/offset for time series",
                                },
                                "EDate": {
                                    "type": "string",
                                    "description": "End date/offset for time series",
                                },
                            },
                        },
                    },
                    "required": ["instruments", "fields"],
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

        # Get instruments from resolved entities or known patterns
        instruments = self._get_instruments(context)
        if not instruments:
            return None

        # Detect fields from query
        fields = self._detect_fields(query)
        if not fields:
            return None

        # Detect time parameters
        params = self._detect_parameters(query)

        # Build tool call
        arguments: dict[str, Any] = {
            "instruments": instruments,
            "fields": fields,
        }
        if params:
            arguments["parameters"] = params

        tool_call = ToolCall(
            tool_name="refinitiv_get_data",
            arguments=arguments,
        )

        return AgentResult(
            tool_calls=(tool_call,),
            confidence=0.85,
            reasoning=f"Rule-based extraction: detected fields {fields}",
            domain=self.domain_name,
        )

    def _get_instruments(self, context: AgentContext) -> list[str]:
        """Get instrument RICs from context or query patterns."""
        # First try resolved entities
        if context.resolved_entities:
            return list(context.resolved_entities.values())

        # Fall back to known company patterns
        query_lower = context.query.lower()
        for company, ric in self.KNOWN_COMPANIES.items():
            if company in query_lower:
                return [ric]

        return []

    def _detect_fields(self, query: str) -> list[str]:
        """Detect field codes from query keywords."""
        query_lower = query.lower()
        fields = []

        # Sort by keyword length (longest first) to match more specific terms
        sorted_fields = sorted(self._all_fields.items(), key=lambda x: -len(x[0]))

        matched_keywords = set()
        for keyword, field_code in sorted_fields:
            # Skip if we already matched a longer keyword that contains this one
            if any(keyword in mk for mk in matched_keywords):
                continue

            # Use word boundary matching
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, query_lower):
                if field_code not in fields:
                    fields.append(field_code)
                    matched_keywords.add(keyword)

        return fields

    def _detect_parameters(self, query: str) -> dict[str, str] | None:
        """Detect time parameters from query."""
        query_lower = query.lower()
        params: dict[str, str] = {}

        # Time series patterns (last N years)
        time_series_match = re.search(r'(?:last|past)\s+(\d+)\s+years?', query_lower)
        if time_series_match:
            years = int(time_series_match.group(1))
            params["SDate"] = "0"
            params["EDate"] = f"-{years - 1}"
            params["Frq"] = "FY"
            return params

        # Quarterly patterns
        if any(p in query_lower for p in ["quarterly", "quarter", "q1", "q2", "q3", "q4"]):
            if "last quarter" in query_lower or "previous quarter" in query_lower:
                params["Period"] = "FQ-1"
            else:
                params["Period"] = "FQ0"
            params["Frq"] = "FQ"
            return params if params else None

        # Annual patterns
        if "last year" in query_lower or "fiscal year" in query_lower:
            params["Period"] = "FY0"
            params["Frq"] = "FY"
            return params

        # Default: current fiscal year
        params["Period"] = "FY0"
        return params
