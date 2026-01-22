"""
NL2API MCP Tools

Exposes the NL2API orchestrator and domain agents via MCP protocol.
"""

from __future__ import annotations

import json
import logging
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.nl2api.llm.protocols import LLMProvider
    from src.nl2api.orchestrator import NL2APIOrchestrator
    from src.nl2api.agents.protocols import DomainAgent
    from src.nl2api.resolution.protocols import EntityResolver

logger = logging.getLogger(__name__)

# =============================================================================
# Tool Definitions
# =============================================================================

NL2API_TOOL_DEFINITIONS = [
    {
        "name": "nl2api_query",
        "description": (
            "Process a natural language financial query through the full NL2API pipeline. "
            "Returns step-by-step results: entity resolution (company→RIC), routing (which domain), "
            "tool generation (API calls), execution data (illustrative), and natural language response. "
            "Use this for general financial queries when you're not sure which specific API to use."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language financial query. Examples: 'What is Apple's P/E ratio?', 'Compare Tesla and Ford revenue'"
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_datastream",
        "description": (
            "Query the Datastream API for price data, time series, and technical indicators. "
            "Use for: current prices, historical prices, market values, volume, calculated fields. "
            "Examples: 'Get Apple's closing price', 'MSFT price for last 30 days', 'Tesla's 52-week high'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query for price/time series data"
                },
                "rics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional pre-resolved RICs. If not provided, entities will be resolved from the query."
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_estimates",
        "description": (
            "Query the Estimates API for analyst forecasts and recommendations (I/B/E/S data). "
            "Use for: EPS estimates, revenue forecasts, analyst recommendations, target prices. "
            "Examples: 'Tesla's FY1 EPS estimate', 'Apple's revenue forecast', 'MSFT analyst recommendations'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query for estimates/forecasts"
                },
                "rics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional pre-resolved RICs"
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_fundamentals",
        "description": (
            "Query the Fundamentals API for financial ratios and statement data. "
            "Use for: P/E ratio, debt-to-equity, ROE, revenue, net income, balance sheet items. "
            "Examples: 'Apple's P/E ratio', 'Compare bank debt ratios', 'Microsoft's net income'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query for fundamental data"
                },
                "rics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional pre-resolved RICs"
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_officers",
        "description": (
            "Query the Officers API for executive and board information. "
            "Use for: CEO names, executive compensation, board members, governance data. "
            "Examples: 'Who is Amazon's CEO?', 'Apple executive compensation', 'Tesla board members'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query for officer/executive data"
                },
                "rics": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Optional pre-resolved RICs"
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "query_screening",
        "description": (
            "Query the Screening API to filter and rank stocks based on criteria. "
            "Use for: stock screeners, rankings, universe filtering. "
            "Examples: 'Top 10 S&P 500 by market cap', 'Tech stocks with P/E under 20', 'Highest dividend yield in DOW'"
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language screening query"
                },
            },
            "required": ["query"],
        },
    },
]


# =============================================================================
# Tool Handlers
# =============================================================================

class NL2APIToolHandlers:
    """Handlers for NL2API MCP tools."""

    def __init__(
        self,
        orchestrator: NL2APIOrchestrator,
        agents: dict[str, DomainAgent],
        llm: LLMProvider,
        resolver: EntityResolver,
    ):
        self.orchestrator = orchestrator
        self.agents = agents
        self.llm = llm
        self.resolver = resolver

    async def handle_tool_call(self, name: str, arguments: dict) -> dict[str, Any]:
        """Route tool call to appropriate handler."""
        handlers = {
            "nl2api_query": self._handle_nl2api_query,
            "query_datastream": lambda args: self._handle_agent_query("datastream", args),
            "query_estimates": lambda args: self._handle_agent_query("estimates", args),
            "query_fundamentals": lambda args: self._handle_agent_query("fundamentals", args),
            "query_officers": lambda args: self._handle_agent_query("officers", args),
            "query_screening": lambda args: self._handle_agent_query("screening", args),
        }

        handler = handlers.get(name)
        if not handler:
            raise ValueError(f"Unknown tool: {name}")

        return await handler(arguments)

    async def _handle_nl2api_query(self, arguments: dict) -> dict[str, Any]:
        """Handle full orchestrator pipeline query."""
        query = arguments.get("query", "")
        if not query:
            return {"success": False, "error": "Query is required"}

        start_time = time.time()

        try:
            # Process through orchestrator
            result = await self.orchestrator.process(query)

            # Check for clarification needed
            if result.needs_clarification:
                return {
                    "success": True,
                    "needs_clarification": True,
                    "clarification_questions": [
                        q.question for q in result.clarification_questions
                    ],
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                }

            # Build step-by-step response
            steps = {
                "entity_resolution": {
                    "resolved": dict(result.resolved_entities),
                },
                "routing": {
                    "domain": result.domain,
                    "confidence": result.confidence,
                    "reasoning": result.reasoning,
                },
                "tool_generation": {
                    "tool_calls": [
                        {
                            "tool_name": tc.tool_name,
                            "arguments": dict(tc.arguments),
                        }
                        for tc in result.tool_calls
                    ],
                },
            }

            # Generate placeholder execution and NL response
            if result.tool_calls:
                execution_data, nl_response = await self._generate_placeholder_response(
                    query, result.tool_calls, result.domain
                )
                steps["execution"] = {
                    "status": "placeholder",
                    "data": execution_data,
                    "note": "Illustrative data - API execution not yet implemented",
                }
                steps["nl_response"] = {
                    "text": nl_response,
                    "generated_by": "llm",
                    "note": "Response based on illustrative data",
                }
            else:
                steps["execution"] = {"status": "no_tool_calls"}
                steps["nl_response"] = {"text": "No tool calls generated for this query."}

            return {
                "success": True,
                "steps": steps,
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

        except Exception as e:
            logger.exception(f"Error processing nl2api_query: {e}")
            return {
                "success": False,
                "error": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

    async def _handle_agent_query(self, domain: str, arguments: dict) -> dict[str, Any]:
        """Handle direct agent query."""
        query = arguments.get("query", "")
        rics = arguments.get("rics", [])

        if not query:
            return {"success": False, "error": "Query is required"}

        agent = self.agents.get(domain)
        if not agent:
            return {"success": False, "error": f"Unknown domain: {domain}"}

        start_time = time.time()

        try:
            # Resolve entities if not provided
            resolved_entities: dict[str, str] = {}
            if not rics:
                # Extract and resolve entities from query
                from src.nl2api.resolution.extractor import extract_entities
                extracted = extract_entities(query)
                for entity in extracted:
                    result = await self.resolver.resolve(entity)
                    if result and result.identifier:
                        resolved_entities[entity] = result.identifier
            else:
                # Use provided RICs
                resolved_entities = {ric: ric for ric in rics}

            # Build agent context
            from src.nl2api.agents.protocols import AgentContext
            context = AgentContext(
                query=query,
                resolved_entities=resolved_entities,
                field_codes={},  # TODO: Add RAG context
                query_examples={},
                conversation_history=[],
            )

            # Process with agent
            result = await agent.process(context)

            # Check for clarification
            if result.needs_clarification:
                return {
                    "success": True,
                    "domain": domain,
                    "needs_clarification": True,
                    "clarification_questions": list(result.clarification_questions),
                    "processing_time_ms": int((time.time() - start_time) * 1000),
                }

            # Build response
            response = {
                "success": True,
                "domain": domain,
                "resolved_entities": resolved_entities,
                "tool_calls": [
                    {
                        "tool_name": tc.tool_name,
                        "arguments": dict(tc.arguments),
                    }
                    for tc in result.tool_calls
                ],
                "confidence": result.confidence,
                "reasoning": result.reasoning,
            }

            # Generate placeholder execution and NL response
            if result.tool_calls:
                execution_data, nl_response = await self._generate_placeholder_response(
                    query, result.tool_calls, domain
                )
                response["execution"] = {
                    "status": "placeholder",
                    "data": execution_data,
                    "note": "Illustrative data - API execution not yet implemented",
                }
                response["nl_response"] = {
                    "text": nl_response,
                    "generated_by": "llm",
                }
            else:
                response["execution"] = {"status": "no_tool_calls"}
                response["nl_response"] = {"text": "No tool calls generated."}

            response["processing_time_ms"] = int((time.time() - start_time) * 1000)
            return response

        except Exception as e:
            logger.exception(f"Error processing {domain} query: {e}")
            return {
                "success": False,
                "domain": domain,
                "error": str(e),
                "processing_time_ms": int((time.time() - start_time) * 1000),
            }

    async def _generate_placeholder_response(
        self,
        query: str,
        tool_calls: tuple,
        domain: str | None,
    ) -> tuple[dict, str]:
        """Generate illustrative execution data and NL response using Haiku."""
        from src.nl2api.llm.protocols import LLMMessage, MessageRole

        tool_calls_json = json.dumps([
            {"tool_name": tc.tool_name, "arguments": dict(tc.arguments)}
            for tc in tool_calls
        ], indent=2)

        prompt = f"""You are generating illustrative sample data for a financial data API demo.

Given this query and the API calls that would be made, generate realistic sample data.

Query: {query}
Domain: {domain or "unknown"}
Tool calls that would be made:
{tool_calls_json}

Generate a JSON response with:
1. "execution_data": A dictionary with realistic sample values the API would return. Use the RICs as keys.
2. "nl_response": A natural language summary of the data (1-2 sentences).

Important:
- Use realistic but clearly illustrative values (round numbers are fine)
- Include units where appropriate (e.g., "$178.50", "28.5x")
- The nl_response should answer the original query using the sample data

Respond with valid JSON only, no markdown formatting."""

        try:
            response = await self.llm.complete(
                messages=[LLMMessage(role=MessageRole.USER, content=prompt)],
                temperature=0.3,
                max_tokens=1000,
            )

            # Parse JSON response
            content = response.content.strip()
            # Handle potential markdown code blocks
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
                content = content.strip()

            parsed = json.loads(content)
            return parsed.get("execution_data", {}), parsed.get("nl_response", "")

        except Exception as e:
            logger.warning(f"Failed to generate placeholder response: {e}")
            return (
                {"error": "Failed to generate illustrative data"},
                f"Unable to generate sample response for: {query}"
            )
