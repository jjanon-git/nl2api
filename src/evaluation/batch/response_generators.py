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

            return SystemResponse(
                raw_output=json.dumps(tool_calls),
                nl_response=None,
                latency_ms=latency_ms,
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

            return SystemResponse(
                raw_output=json.dumps(tool_calls),
                nl_response=result.nl_response,
                latency_ms=latency_ms,
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
