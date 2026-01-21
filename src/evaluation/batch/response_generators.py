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

        Extracts input_entity from test case metadata and resolves it.
        """
        import time

        start_time = time.perf_counter()

        # The nl_query contains the entity reference
        # e.g., "What is Apple's revenue?" -> resolver extracts "Apple"
        query = test_case.nl_query

        try:
            # Use the resolver to process the query
            resolved = await resolver.resolve(query)

            # Build tool calls from resolution results
            tool_calls = []
            if resolved:
                # Get the first resolved RIC (for single entity queries)
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
