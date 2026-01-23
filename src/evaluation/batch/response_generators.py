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
    raw_output = json.dumps(
        [
            {"tool_name": tc.tool_name, "arguments": dict(tc.arguments)}
            for tc in test_case.expected_tool_calls
        ]
    )

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
                    tool_calls.append(
                        {
                            "tool_name": "get_data",
                            "arguments": {
                                "tickers": [result.identifier],
                                "fields": ["TR.Revenue"],  # Default field
                            },
                        }
                    )
            else:
                # Fallback for non-entity_resolution tests: parse full query
                resolved = await resolver.resolve(test_case.nl_query)
                if resolved:
                    rics = list(resolved.values())
                    if rics:
                        tool_calls.append(
                            {
                                "tool_name": "get_data",
                                "arguments": {
                                    "tickers": rics,
                                    "fields": ["TR.Revenue"],  # Default field
                                },
                            }
                        )

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
            tool_calls = [{"tool_name": f"route_to_{result.domain}", "arguments": {}}]

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


def create_tool_only_generator(
    agent,
    resolved_entities: dict[str, str] | None = None,
    entity_resolver=None,
):
    """
    Create a response generator that tests a single agent directly.

    This generator bypasses routing to test a specific agent's tool generation
    capabilities. Entity resolution can be handled in three ways:
    1. Pre-resolved entities passed directly (resolved_entities parameter)
    2. Live resolution using entity_resolver (recommended for accuracy testing)
    3. Entities from test case metadata (fallback)

    Args:
        agent: Agent instance to test (e.g., DatastreamAgent, EstimatesAgent)
        resolved_entities: Pre-resolved entity mapping {company_name: RIC}
        entity_resolver: Optional resolver to resolve entities from queries live

    Returns:
        Async function that generates SystemResponse from TestCase
    """
    from src.nl2api.agents.protocols import AgentContext

    async def generate_tool_only_response(test_case: TestCase) -> SystemResponse:
        """
        Generate response by running a single agent directly.

        Resolution priority:
        1. Use provided resolved_entities if given
        2. Use entity_resolver to resolve from query if provided
        3. Look for resolved_entities in test case metadata
        """
        import time

        start_time = time.perf_counter()

        try:
            # Priority 1: Use provided resolved_entities
            entities = resolved_entities

            # Priority 2: Resolve entities live using resolver
            if entities is None and entity_resolver is not None:
                entities = await entity_resolver.resolve(test_case.nl_query)

            # Priority 3: Look in test case metadata
            if entities is None and test_case.expected_response:
                entities = test_case.expected_response.get("resolved_entities", {})

            # Build AgentContext for the agent (matches process() signature)
            context = AgentContext(
                query=test_case.nl_query,
                resolved_entities=entities or {},
            )

            # Call the agent's process method with AgentContext
            result = await agent.process(context)

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Convert agent result to tool calls
            tool_calls = []
            if hasattr(result, "tool_calls") and result.tool_calls:
                tool_calls = [
                    {"tool_name": tc.tool_name, "arguments": dict(tc.arguments)}
                    for tc in result.tool_calls
                ]

            # Extract token usage from AgentResult
            input_tokens = getattr(result, "tokens_prompt", None)
            output_tokens = getattr(result, "tokens_completion", None)

            return SystemResponse(
                raw_output=json.dumps(tool_calls),
                nl_response=None,  # AgentResult doesn't have nl_response
                latency_ms=latency_ms,
                input_tokens=input_tokens if input_tokens and input_tokens > 0 else None,
                output_tokens=output_tokens if output_tokens and output_tokens > 0 else None,
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


def create_rag_simulated_generator(pass_rate: float = 0.7):
    """
    Create a response generator for RAG evaluation with simulated responses.

    This generator produces simulated RAG responses for testing the RAG evaluation
    pipeline. The pass_rate parameter controls what percentage of responses will
    be "correct" (good retrieval, faithful response, proper citations).

    WARNING: This generator is for PIPELINE TESTING ONLY.
    For real RAG accuracy measurement, integrate with an actual RAG system.

    Args:
        pass_rate: Probability (0.0-1.0) that a response will be "correct"

    Returns:
        Async function that generates SystemResponse from TestCase
    """

    async def generate_rag_response(test_case: TestCase) -> SystemResponse:
        """
        Generate simulated RAG response for testing.

        For passing responses:
        - Uses expected relevant_docs as retrieved_doc_ids
        - Generates plausible response text
        - Includes proper citations

        For failing responses:
        - Returns wrong/empty documents
        - Generates hallucinated content
        - Omits citations
        """
        import time

        start_time = time.perf_counter()
        latency_ms = random.randint(100, 500)
        await asyncio.sleep(latency_ms / 1000)

        # Determine if this should be a passing or failing response
        should_pass = random.random() < pass_rate

        # Get expected values from test case
        expected = test_case.expected or {}
        query = test_case.input.get("query", test_case.nl_query or "")
        relevant_docs = expected.get("relevant_docs", [])
        expected_behavior = expected.get("behavior", "answer")

        if should_pass:
            # Generate a passing response
            retrieved_doc_ids = relevant_docs.copy() if relevant_docs else ["doc-1"]
            retrieved_chunks = [
                {"id": doc_id, "text": f"Content from {doc_id} about {query[:30]}..."}
                for doc_id in retrieved_doc_ids
            ]
            context = " ".join([chunk["text"] for chunk in retrieved_chunks])
            sources = [
                {"id": str(i + 1), "text": chunk["text"], "doc_id": chunk["id"]}
                for i, chunk in enumerate(retrieved_chunks)
            ]

            if expected_behavior == "reject":
                # Should reject - generate proper rejection
                response = "I cannot answer this question as it falls outside my knowledge base."
            else:
                # Generate response with citations
                response = f"Based on the retrieved documents [1], {query[:50]}... [2]"

        else:
            # Generate a failing response (varies type of failure)
            failure_type = random.choice(
                [
                    "wrong_docs",
                    "no_citation",
                    "hallucination",
                    "empty_retrieval",
                ]
            )

            if failure_type == "wrong_docs":
                # Retrieved wrong documents
                retrieved_doc_ids = ["wrong-doc-1", "wrong-doc-2"]
                retrieved_chunks = [
                    {"id": doc_id, "text": f"Irrelevant content from {doc_id}"}
                    for doc_id in retrieved_doc_ids
                ]
                context = "Irrelevant information about something else entirely."
                sources = []
                response = f"The answer is {query[:30]}... based on my knowledge."

            elif failure_type == "no_citation":
                # Correct retrieval but no citations
                retrieved_doc_ids = relevant_docs.copy() if relevant_docs else ["doc-1"]
                retrieved_chunks = [
                    {"id": doc_id, "text": f"Content from {doc_id}"} for doc_id in retrieved_doc_ids
                ]
                context = " ".join([chunk["text"] for chunk in retrieved_chunks])
                sources = []  # No sources/citations
                response = f"The answer is definitely {query[:30]}..."

            elif failure_type == "hallucination":
                # Makes up information not in context
                retrieved_doc_ids = relevant_docs.copy() if relevant_docs else ["doc-1"]
                retrieved_chunks = [
                    {"id": doc_id, "text": "Simple factual information."}
                    for doc_id in retrieved_doc_ids
                ]
                context = "Simple factual information."
                sources = [{"id": "1", "text": "Simple factual information."}]
                # Response contains information not in context
                response = "Based on the retrieved documents [1], the answer involves complex calculations showing 47.3% growth over the last fiscal year."

            else:  # empty_retrieval
                retrieved_doc_ids = []
                retrieved_chunks = []
                context = ""
                sources = []
                response = "I don't have any relevant information to answer this question."

        actual_latency = int((time.perf_counter() - start_time) * 1000)

        return SystemResponse(
            raw_output=json.dumps(
                {
                    "response": response,
                    "retrieved_doc_ids": retrieved_doc_ids,
                    "retrieved_chunks": retrieved_chunks,
                    "context": context,
                    "sources": sources,
                }
            ),
            nl_response=response,
            latency_ms=actual_latency,
            # Store extra RAG-specific data that BatchRunner._response_to_output can use
            # These are accessed via getattr in the runner
        )

    return generate_rag_response
