"""
Response Generators for Batch Evaluation

Provides different response generators for batch evaluation:
- simulate_correct_response: Always returns expected answers (pipeline testing)
- create_entity_resolver_generator: Uses real EntityResolver (accuracy measurement)

IMPORTANT: For meaningful accuracy tracking, use real response generators.
Simulated responses should only be used for pipeline/infrastructure testing.
"""

from __future__ import annotations

import asyncio
import json
import random

from CONTRACTS import SystemResponse, TestCase
from src.nl2api.resolution.protocols import EntityResolver


async def simulate_correct_response(test_case: TestCase) -> SystemResponse:
    """
    Simulate a correct response matching expected tool calls.

    WARNING: This generator always produces correct responses.
    Use ONLY for pipeline testing, NOT for accuracy measurement.
    Results from this generator should NOT be persisted for tracking.

    For real accuracy measurement, use create_entity_resolver_generator().
    """
    # Simulate some latency
    latency_ms = random.randint(50, 200)
    await asyncio.sleep(latency_ms / 1000)

    # Build raw output from expected tool calls
    raw_output = json.dumps([
        {"tool_name": tc.tool_name, "arguments": dict(tc.arguments)}
        for tc in test_case.expected_tool_calls
    ])

    return SystemResponse(
        raw_output=raw_output,
        nl_response=test_case.expected_nl_response,
        latency_ms=latency_ms,
    )


def create_entity_resolver_generator(resolver: EntityResolver):
    """
    Create a response generator that uses the real EntityResolver.

    This generator produces actual system responses by running the resolver,
    enabling meaningful accuracy measurement and tracking over time.

    Args:
        resolver: EntityResolver instance to use for resolution

    Returns:
        Async function that generates SystemResponse from TestCase
    """

    async def generate_resolver_response(test_case: TestCase) -> SystemResponse:
        """
        Generate response by running the real EntityResolver.

        For entity_resolution tests, uses input_entity from metadata (stored
        in expected_response) to call resolve_single directly. This avoids
        regex extraction bugs when parsing full NL queries.
        """
        import time

        start_time = time.perf_counter()

        try:
            # Get input_entity from metadata stored in expected_response
            metadata = test_case.expected_response or {}
            input_entity = metadata.get("input_entity") if metadata else None

            tool_calls = []
            if input_entity:
                # Use resolve_single for direct entity → RIC mapping
                # This is the correct approach for entity_resolution tests
                result = await resolver.resolve_single(input_entity)
                if result:
                    tool_calls.append({
                        "tool_name": "get_data",
                        "arguments": {
                            "tickers": [result.identifier],
                            "fields": ["TR.Revenue"]  # Default field
                        }
                    })
            else:
                # Fallback for non-entity_resolution tests: parse full query
                resolved = await resolver.resolve(test_case.nl_query)
                if resolved:
                    rics = list(resolved.values())
                    if rics:
                        tool_calls.append({
                            "tool_name": "get_data",
                            "arguments": {
                                "tickers": rics,
                                "fields": ["TR.Revenue"]  # Default field
                            }
                        })

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            return SystemResponse(
                raw_output=json.dumps(tool_calls),
                nl_response=None,  # Resolver doesn't generate NL
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            # Return empty response on error
            return SystemResponse(
                raw_output=json.dumps([]),
                nl_response=None,
                latency_ms=latency_ms,
                error=str(e),
            )

    return generate_resolver_response


def create_routing_generator(router):
    """
    Create a response generator that uses the LLM router for routing evaluation.

    This generator tests only the routing component by calling router.route()
    and comparing the selected domain against expected domain.

    Args:
        router: Router instance (LLMToolRouter or similar)

    Returns:
        Async function that generates SystemResponse from TestCase
    """

    async def generate_routing_response(test_case: TestCase) -> SystemResponse:
        """
        Generate response by running the real router.

        Returns a tool call with the routed domain as the tool_name.
        """
        import time

        start_time = time.perf_counter()

        try:
            # Call the router to get routing decision
            result = await router.route(test_case.nl_query)

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Convert router result to tool call format for comparison
            # Tool name is "route_to_{domain}" to match expected format
            # Use empty arguments - we only compare the domain (tool_name)
            # Actual confidence/reasoning is stored in metadata for analysis
            tool_calls = [{
                "tool_name": f"route_to_{result.domain}",
                "arguments": {}
            }]

            # Extract token usage from router result
            input_tokens = result.input_tokens if result.input_tokens > 0 else None
            output_tokens = result.output_tokens if result.output_tokens > 0 else None

            return SystemResponse(
                raw_output=json.dumps(tool_calls),
                nl_response=None,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return SystemResponse(
                raw_output=json.dumps([]),
                nl_response=None,
                latency_ms=latency_ms,
                error=str(e),
            )

    return generate_routing_response


def create_nl2api_generator(orchestrator):
    """
    Create a response generator that uses the full NL2API orchestrator.

    This generator runs complete queries through the orchestrator,
    including entity resolution, agent routing, and response generation.

    Args:
        orchestrator: NL2APIOrchestrator instance

    Returns:
        Async function that generates SystemResponse from TestCase
    """

    async def generate_orchestrator_response(test_case: TestCase) -> SystemResponse:
        """
        Generate response by running the full NL2API orchestrator.
        """
        import time

        start_time = time.perf_counter()

        try:
            result = await orchestrator.process(test_case.nl_query)

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Convert orchestrator result to SystemResponse
            tool_calls = []
            if result.tool_calls:
                tool_calls = [
                    {"tool_name": tc.tool_name, "arguments": dict(tc.arguments)}
                    for tc in result.tool_calls
                ]

            # Extract token usage - NL2APIResponse now has input_tokens and output_tokens
            input_tokens = result.input_tokens if result.input_tokens > 0 else None
            output_tokens = result.output_tokens if result.output_tokens > 0 else None

            return SystemResponse(
                raw_output=json.dumps(tool_calls),
                nl_response=None,  # NL2APIResponse doesn't have nl_response field
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return SystemResponse(
                raw_output=json.dumps([]),
                nl_response=None,
                latency_ms=latency_ms,
                error=str(e),
            )

    return generate_orchestrator_response


def create_tool_only_generator(agent, resolved_entities: dict[str, str] | None = None):
    """
    Create a response generator that tests a single agent directly.

    This generator bypasses routing and entity resolution to test
    a specific agent's tool generation capabilities in isolation.

    Args:
        agent: Agent instance to test (e.g., DatastreamAgent, EstimatesAgent)
        resolved_entities: Pre-resolved entity mapping {company_name: RIC}
                          If None, uses entities from test case metadata

    Returns:
        Async function that generates SystemResponse from TestCase
    """

    async def generate_tool_only_response(test_case: TestCase) -> SystemResponse:
        """
        Generate response by running a single agent directly.

        Uses pre-resolved entities from test case metadata (stored in expected_response)
        or the provided resolved_entities mapping.
        """
        import time

        start_time = time.perf_counter()

        try:
            # Get resolved entities from test case metadata or provided mapping
            entities = resolved_entities
            if entities is None and test_case.expected_response:
                # Look for resolved_entities in the expected_response metadata
                entities = test_case.expected_response.get("resolved_entities", {})

            # Build context for the agent
            context = {
                "query": test_case.nl_query,
                "resolved_entities": entities or {},
            }

            # Call the agent's process method
            result = await agent.process(test_case.nl_query, context)

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Convert agent result to tool calls
            tool_calls = []
            if hasattr(result, 'tool_calls') and result.tool_calls:
                tool_calls = [
                    {"tool_name": tc.tool_name, "arguments": dict(tc.arguments)}
                    for tc in result.tool_calls
                ]

            # Extract token usage if available
            input_tokens = getattr(result, 'input_tokens', None)
            output_tokens = getattr(result, 'output_tokens', None)

            return SystemResponse(
                raw_output=json.dumps(tool_calls),
                nl_response=getattr(result, 'nl_response', None),
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return SystemResponse(
                raw_output=json.dumps([]),
                nl_response=None,
                latency_ms=latency_ms,
                error=str(e),
            )

    return generate_tool_only_response
