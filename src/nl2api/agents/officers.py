"""
Officers Domain Agent

Handles Officers & Directors API queries including:
- Executive information (CEO, CFO, etc.)
- Board of directors
- Executive compensation
- Employment history
- Education background
"""

from __future__ import annotations

import re
from typing import Any

from CONTRACTS import ToolCall
from src.nl2api.agents.base import BaseDomainAgent
from src.nl2api.agents.protocols import AgentContext, AgentResult
from src.nl2api.llm.protocols import LLMProvider, LLMToolDefinition
from src.nl2api.rag.protocols import RAGRetriever


class OfficersAgent(BaseDomainAgent):
    """
    Domain agent for Officers & Directors API.

    Handles natural language queries about:
    - Executive identification (CEO, CFO, etc.)
    - Board of directors composition
    - Executive compensation
    - Officer tenure and history
    - Education and background
    """

    # C-suite executive fields
    EXECUTIVE_FIELDS = {
        "ceo": "TR.CEOName",
        "chief executive": "TR.CEOName",
        "chief executive officer": "TR.CEOName",
        "cfo": "TR.CFOName",
        "chief financial officer": "TR.CFOName",
        "chairman": "TR.ChairmanName",
        "board chairman": "TR.ChairmanName",
        "chairperson": "TR.ChairmanName",
    }

    # Officer information fields
    OFFICER_INFO_FIELDS = {
        "officer name": "TR.OfficerName",
        "executive name": "TR.OfficerName",
        "officer title": "TR.OfficerTitle",
        "position": "TR.OfficerTitle",
        "role": "TR.OfficerTitle",
        "officer age": "TR.OfficerAge",
        "age": "TR.OfficerAge",
        "title since": "TR.OfficerTitleSince",
        "tenure": "TR.OfficerTitleSince",
        "start date": "TR.OfficerTitleSince",
    }

    # Board and governance fields
    BOARD_FIELDS = {
        "board members": "TR.ODDirectorName",
        "directors": "TR.ODDirectorName",
        "board of directors": "TR.ODDirectorName",
        "board size": "TR.BoardSize",
        "independent directors": "TR.IndependentBoardMembers",
        "independent board members": "TR.IndependentBoardMembers",
        "director tenure": "TR.ODDirectorTenure",
        "board tenure": "TR.ODDirectorTenure",
        "independent director": "TR.ODIndependentDirector",
    }

    # Compensation fields
    COMPENSATION_FIELDS = {
        "salary": "TR.ODOfficerSalary",
        "base salary": "TR.ODOfficerSalary",
        "bonus": "TR.ODOfficerBonus",
        "stock awards": "TR.ODOfficerStockAwards",
        "option awards": "TR.ODOfficerOptionAwards",
        "total compensation": "TR.ODOfficerTotalComp",
        "compensation": "TR.ODOfficerTotalComp",
        "total comp": "TR.ODOfficerTotalComp",
        "pay": "TR.ODOfficerTotalComp",
    }

    # Education and biography fields
    EDUCATION_FIELDS = {
        "education": "TR.ODOfficerUniversityName",
        "university": "TR.ODOfficerUniversityName",
        "school": "TR.ODOfficerUniversityName",
        "degree": "TR.ODOfficerGraduationDegree",
        "graduation": "TR.ODOfficerGraduationYear",
        "biography": "TR.ODOfficerBiography",
        "bio": "TR.ODOfficerBiography",
    }

    # Keywords for domain classification
    DOMAIN_KEYWORDS = [
        # Executive roles
        "ceo", "cfo", "coo", "cto", "cio",
        "chief executive", "chief financial", "chief operating",
        "president", "chairman", "chairperson",
        # General terms
        "executive", "officer", "management", "leadership",
        "who runs", "who leads", "who is the",
        # Board terms
        "board", "director", "board member", "governance",
        "independent director", "board size",
        # Compensation
        "salary", "compensation", "bonus", "pay",
        "stock awards", "total comp",
        # Background
        "tenure", "how long", "education", "university",
        "background", "biography", "bio",
    ]

    # Known company patterns
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
        "jp morgan": "JPM.N",
        "jpmorgan": "JPM.N",
        "goldman sachs": "GS.N",
    }

    def __init__(
        self,
        llm: LLMProvider,
        rag: RAGRetriever | None = None,
    ):
        """Initialize the Officers agent."""
        super().__init__(llm, rag)
        # Combine all field mappings
        self._all_fields = {
            **self.EXECUTIVE_FIELDS,
            **self.OFFICER_INFO_FIELDS,
            **self.BOARD_FIELDS,
            **self.COMPENSATION_FIELDS,
            **self.EDUCATION_FIELDS,
        }

    @property
    def domain_name(self) -> str:
        """Return the domain name."""
        return "officers"

    @property
    def domain_description(self) -> str:
        """Return the domain description."""
        return (
            "Officers & Directors API - executive information, board composition, "
            "compensation data, and officer backgrounds"
        )

    @property
    def capabilities(self) -> tuple[str, ...]:
        """Return the data types this agent handles."""
        return (
            "CEO and CFO information",
            "board of directors",
            "executive compensation",
            "officer tenure",
            "management team",
            "governance data",
            "officer biography",
            "education background",
        )

    @property
    def example_queries(self) -> tuple[str, ...]:
        """Return example queries this agent handles well."""
        return (
            "Who is the CEO of Apple?",
            "Show Microsoft's board of directors",
            "What is Tim Cook's compensation?",
            "List Tesla's executive team",
            "How long has Satya Nadella been CEO?",
        )

    def get_system_prompt(self) -> str:
        """Return the system prompt for the Officers domain."""
        return """You are an expert at translating natural language queries into LSEG Officers & Directors API calls.

Your task is to generate accurate `refinitiv.get_data` tool calls based on the user's query about company executives and directors.

## Key Field Codes

### C-Suite Executives
- `TR.CEOName` - Chief Executive Officer name
- `TR.CFOName` - Chief Financial Officer name
- `TR.ChairmanName` - Board Chairman name

### Officer Information
- `TR.OfficerName` - Officer full name
- `TR.OfficerTitle` - Officer title/position
- `TR.OfficerAge` - Officer age
- `TR.OfficerTitleSince` - When they started current role

### Board & Governance
- `TR.ODDirectorName` - Board director name
- `TR.BoardSize` - Number of board members
- `TR.IndependentBoardMembers` - Number of independent directors
- `TR.ODDirectorTenure` - Years on board
- `TR.ODIndependentDirector` - Independence status

### Compensation
- `TR.ODOfficerSalary` - Base salary
- `TR.ODOfficerBonus` - Annual bonus
- `TR.ODOfficerStockAwards` - Stock-based compensation
- `TR.ODOfficerOptionAwards` - Option grants value
- `TR.ODOfficerTotalComp` - Total annual compensation

### Education & Background
- `TR.ODOfficerUniversityName` - University attended
- `TR.ODOfficerGraduationDegree` - Academic degree
- `TR.ODOfficerGraduationYear` - Year of graduation
- `TR.ODOfficerBiography` - Full biographical text

## Parameters
- `OfficerType`: Filter by officer type
  - `Executive` - Executive officers
  - `Director` - Board directors
  - `CEO` - CEO only

## Examples

Query: "Who is the CEO of Apple?"
Tool call: refinitiv.get_data(instruments=["AAPL.O"], fields=["TR.CEOName"])

Query: "What is the total compensation of Tesla's CEO?"
Tool call: refinitiv.get_data(instruments=["TSLA.O"], fields=["TR.CEOName", "TR.ODOfficerTotalComp"], parameters={"OfficerType": "CEO"})

Query: "How many board members does Microsoft have and how many are independent?"
Tool call: refinitiv.get_data(instruments=["MSFT.O"], fields=["TR.BoardSize", "TR.IndependentBoardMembers"])

Query: "Who are the top executives at NVIDIA?"
Tool call: refinitiv.get_data(instruments=["NVDA.O"], fields=["TR.OfficerName", "TR.OfficerTitle"], parameters={"OfficerType": "Executive"})

## Rules
1. Always use RIC codes for instruments (e.g., AAPL.O for Apple)
2. For CEO/CFO queries, use the specific field (TR.CEOName, TR.CFOName)
3. For compensation queries, include the officer name field
4. For executive lists, add OfficerType parameter
5. If the query is ambiguous, ask for clarification

Generate the most appropriate refinitiv.get_data tool call for the user's query."""

    def get_tools(self) -> list[LLMToolDefinition]:
        """Return the tools available for this domain."""
        return [
            LLMToolDefinition(
                name="refinitiv_get_data",
                description="Retrieve officer and director data from Refinitiv",
                parameters={
                    "type": "object",
                    "properties": {
                        "instruments": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of Reuters Instrument Codes (e.g., ['AAPL.O'])",
                        },
                        "fields": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "List of TR field codes (e.g., ['TR.CEOName'])",
                        },
                        "parameters": {
                            "type": "object",
                            "description": "Optional parameters for filtering",
                            "properties": {
                                "OfficerType": {
                                    "type": "string",
                                    "description": "Filter by officer type (Executive, Director, CEO)",
                                },
                                "RNK": {
                                    "type": "string",
                                    "description": "Ranking range (e.g., 'R1:R10' for top 10)",
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

        # Get instruments
        instruments = self._get_instruments(context)
        if not instruments:
            return None

        # Detect fields
        fields = self._detect_fields(query)
        if not fields:
            return None

        # Detect parameters
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
        """Get instrument RICs from context or patterns."""
        if context.resolved_entities:
            return list(context.resolved_entities.values())

        query_lower = context.query.lower()
        for company, ric in self.KNOWN_COMPANIES.items():
            if company in query_lower:
                return [ric]

        return []

    def _detect_fields(self, query: str) -> list[str]:
        """Detect field codes from query keywords."""
        query_lower = query.lower()
        fields = []

        # Check for CEO/CFO/Chairman specifically first
        if any(kw in query_lower for kw in ["ceo", "chief executive"]):
            fields.append("TR.CEOName")
        if any(kw in query_lower for kw in ["cfo", "chief financial"]):
            fields.append("TR.CFOName")
        if any(kw in query_lower for kw in ["chairman", "chairperson", "board chair"]):
            fields.append("TR.ChairmanName")

        # Check for board size queries
        if "board size" in query_lower or "how many board" in query_lower:
            fields.append("TR.BoardSize")
        if "independent" in query_lower:
            fields.append("TR.IndependentBoardMembers")

        # Check for compensation
        if any(kw in query_lower for kw in ["compensation", "salary", "pay", "total comp"]):
            # Add officer name if not already present
            if "TR.CEOName" not in fields:
                fields.append("TR.CEOName")
            fields.append("TR.ODOfficerTotalComp")

        # Check for executives list
        if any(kw in query_lower for kw in ["executives", "top executives", "management team", "leadership"]):
            if "TR.OfficerName" not in fields:
                fields.append("TR.OfficerName")
            if "TR.OfficerTitle" not in fields:
                fields.append("TR.OfficerTitle")

        # Check for board members
        if any(kw in query_lower for kw in ["board members", "directors", "board of directors"]):
            fields.append("TR.ODDirectorName")

        # Check for tenure
        if any(kw in query_lower for kw in ["tenure", "how long", "since when", "start date"]):
            fields.append("TR.OfficerTitleSince")

        # Check for age
        if "age" in query_lower:
            fields.append("TR.OfficerAge")

        # Check for education
        if any(kw in query_lower for kw in ["education", "university", "school", "degree"]):
            fields.append("TR.ODOfficerUniversityName")
            fields.append("TR.ODOfficerGraduationDegree")

        # Check for biography
        if any(kw in query_lower for kw in ["biography", "bio", "background"]):
            fields.append("TR.ODOfficerBiography")

        # Deduplicate while preserving order
        seen = set()
        unique_fields = []
        for f in fields:
            if f not in seen:
                seen.add(f)
                unique_fields.append(f)

        return unique_fields

    def _detect_parameters(self, query: str) -> dict[str, str] | None:
        """Detect parameters from query."""
        query_lower = query.lower()
        params: dict[str, str] = {}

        # Detect officer type
        if any(kw in query_lower for kw in ["executives", "top executives", "management", "leadership"]):
            params["OfficerType"] = "Executive"
        elif any(kw in query_lower for kw in ["directors", "board members", "board of directors"]):
            params["OfficerType"] = "Director"
        elif "ceo" in query_lower and "compensation" in query_lower:
            # For CEO compensation queries
            params["OfficerType"] = "CEO"

        return params if params else None
