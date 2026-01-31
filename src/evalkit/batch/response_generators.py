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
import logging
import random

from CONTRACTS import EntityResolver, SystemResponse, TestCase

logger = logging.getLogger(__name__)


async def simulate_correct_response(test_case: TestCase) -> SystemResponse:
    """
    Simulate a correct response matching expected tool calls.

    WARNING: This generator always produces correct responses.
    Use ONLY for pipeline testing, NOT for accuracy measurement.
    Results from this generator should NOT be persisted for tracking.

    For real accuracy measurement, use create_entity_resolver_generator().

    Supports both NL2API-specific and generic test case formats.
    """
    # Simulate some latency
    latency_ms = random.randint(50, 200)
    await asyncio.sleep(latency_ms / 1000)

    # Get expected tool calls from either NL2API or generic format
    if test_case.expected_tool_calls:
        tool_calls = [
            {"tool_name": tc.tool_name, "arguments": dict(tc.arguments)}
            for tc in test_case.expected_tool_calls
        ]
    elif test_case.expected.get("tool_calls"):
        tool_calls = test_case.expected["tool_calls"]
    else:
        tool_calls = []

    raw_output = json.dumps(tool_calls)

    # Get expected NL response from either format
    nl_response = (
        test_case.expected_nl_response
        or test_case.expected.get("nl_response")
        or test_case.expected.get("response")
    )

    return SystemResponse(
        raw_output=raw_output,
        nl_response=nl_response,
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
            # Get input_entity from metadata stored in expected_response or expected
            metadata = test_case.expected_response or test_case.expected or {}
            input_entity = metadata.get("input_entity") if metadata else None

            # Get query from generic or NL2API-specific field
            query = (
                test_case.input.get("nl_query")
                or test_case.input.get("query")
                or test_case.nl_query
                or ""
            )

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
                resolved = await resolver.resolve(query)
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
            # Log error for observability, then return error response
            test_case_id = getattr(test_case, "id", "unknown")
            logger.warning(
                f"Entity resolver failed for test case {test_case_id}: {e}",
                exc_info=True,
                extra={"test_case_id": test_case_id},
            )
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
        Supports both NL2API-specific and generic test case formats.
        """
        import time

        start_time = time.perf_counter()

        try:
            # Get query from generic or NL2API-specific field
            query = (
                test_case.input.get("nl_query")
                or test_case.input.get("query")
                or test_case.nl_query
                or ""
            )

            # Call the router to get routing decision
            result = await router.route(query)

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
            test_case_id = getattr(test_case, "id", "unknown")
            logger.warning(
                f"Routing generator failed for test case {test_case_id}: {e}",
                exc_info=True,
                extra={"test_case_id": test_case_id},
            )
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

        Supports both NL2API-specific and generic test case formats.
        """
        import time

        start_time = time.perf_counter()

        try:
            # Get query from generic or NL2API-specific field
            query = (
                test_case.input.get("nl_query")
                or test_case.input.get("query")
                or test_case.nl_query
                or ""
            )

            result = await orchestrator.process(query)

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
            test_case_id = getattr(test_case, "id", "unknown")
            logger.warning(
                f"Orchestrator generator failed for test case {test_case_id}: {e}",
                exc_info=True,
                extra={"test_case_id": test_case_id},
            )
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

        Supports both NL2API-specific and generic test case formats.
        """
        import time

        start_time = time.perf_counter()

        try:
            # Get query from generic or NL2API-specific field
            query = (
                test_case.input.get("nl_query")
                or test_case.input.get("query")
                or test_case.nl_query
                or ""
            )

            # Priority 1: Use provided resolved_entities
            entities = resolved_entities

            # Priority 2: Resolve entities live using resolver
            if entities is None and entity_resolver is not None:
                entities = await entity_resolver.resolve(query)

            # Priority 3: Look in test case metadata (generic or NL2API-specific)
            if entities is None:
                metadata = test_case.expected or test_case.expected_response or {}
                entities = metadata.get("resolved_entities", {})

            # Build AgentContext for the agent (matches process() signature)
            context = AgentContext(
                query=query,
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
            test_case_id = getattr(test_case, "id", "unknown")
            logger.warning(
                f"Tool-only generator failed for test case {test_case_id}: {e}",
                exc_info=True,
                extra={"test_case_id": test_case_id},
            )
            return SystemResponse(
                raw_output=json.dumps([]),
                nl_response=None,
                latency_ms=latency_ms,
                error=str(e),
            )

    return generate_tool_only_response


def create_rag_retrieval_generator(retriever, embedder=None):
    """
    Create a response generator that uses real RAG retrieval.

    This generator runs actual retrieval queries through the HybridRAGRetriever,
    returning the retrieved chunks for comparison against ground truth.

    Args:
        retriever: HybridRAGRetriever instance
        embedder: Optional embedder to set on retriever

    Returns:
        Async function that generates SystemResponse from TestCase
    """

    async def generate_rag_retrieval_response(test_case: TestCase) -> SystemResponse:
        """
        Generate response by running real RAG retrieval.

        Uses test_case.input['query'] to run retrieval and returns
        the retrieved document IDs for comparison against expected.
        """
        import time

        start_time = time.perf_counter()

        try:
            # Get query from test case
            # Note: nl_query may not exist as attribute if not provided, use getattr
            nl_query = getattr(test_case, "nl_query", None)
            query = test_case.input.get("query", nl_query or "")

            if not query:
                return SystemResponse(
                    raw_output=json.dumps({"error": "No query provided"}),
                    nl_response=None,
                    latency_ms=0,
                    error="No query in test case",
                )

            # Add company context if available (improves retrieval precision)
            # Company info is stored in input_json from load_rag_fixtures.py
            company_name = test_case.input.get("company_name")

            if company_name:
                # Prepend company context to query for better retrieval
                query = f"{company_name}: {query}"

            # Run retrieval
            results = await retriever.retrieve(
                query=query,
                document_types=None,  # All types
                limit=10,
                threshold=0.0,  # No threshold for evaluation
                use_cache=False,
            )

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Extract retrieved doc IDs
            retrieved_doc_ids = [str(r.id) for r in results]
            retrieved_chunks = [
                {
                    "id": str(r.id),
                    "text": r.content[:500],  # Truncate for storage
                    "score": r.score,
                }
                for r in results
            ]

            return SystemResponse(
                raw_output=json.dumps(
                    {
                        "response": f"Retrieved {len(results)} documents for: {query[:50]}...",
                        "retrieved_doc_ids": retrieved_doc_ids,
                        "retrieved_chunks": retrieved_chunks,
                        "context": "\n\n".join([r.content[:200] for r in results[:3]]),
                        "sources": [],
                    }
                ),
                nl_response=f"Retrieved {len(results)} documents.",
                latency_ms=latency_ms,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            test_case_id = getattr(test_case, "id", "unknown")
            logger.warning(
                f"RAG retrieval generator failed for test case {test_case_id}: {e}",
                exc_info=True,
                extra={"test_case_id": test_case_id},
            )
            return SystemResponse(
                raw_output=json.dumps({"error": str(e)}),
                nl_response=None,
                latency_ms=latency_ms,
                error=str(e),
            )

    return generate_rag_retrieval_response


def create_rag_generation_generator(retriever, llm_client, embedder=None):
    """
    Create a response generator that does full RAG: retrieval + LLM generation.

    This generator:
    1. Retrieves relevant documents using the retriever
    2. Builds a prompt with the retrieved context
    3. Calls an LLM to generate an answer with citations
    4. Returns the full response for evaluation

    This enables evaluation of:
    - faithfulness: Is the response grounded in retrieved context?
    - answer_relevance: Does the response answer the question?
    - citation: Are citations present and accurate?
    - context_relevance: Is the retrieved context relevant?

    Args:
        retriever: HybridRAGRetriever instance
        llm_client: Anthropic client for generation
        embedder: Optional embedder to set on retriever

    Returns:
        Async function that generates SystemResponse from TestCase
    """

    RAG_SYSTEM_PROMPT = """You are a helpful financial analyst assistant that answers questions based on SEC filings.

IMPORTANT RULES:
1. ONLY use information from the provided context to answer
2. If the context doesn't contain enough information, say so
3. Cite your sources using [Source N] format where N is the chunk number
4. Be specific and factual - avoid speculation
5. If asked for financial advice, politely decline and explain you can only provide factual information from filings

For questions that violate policy (financial advice, PII requests, etc.), respond with:
"I cannot provide [type of request]. I can only answer factual questions based on SEC filing information."
"""

    RAG_USER_PROMPT = """Based on the following SEC filing excerpts, answer this question:

Question: {query}

Context:
{context}

Provide a clear, factual answer with citations. Use [Source N] to cite specific excerpts."""

    async def generate_rag_full_response(test_case: TestCase) -> SystemResponse:
        """
        Generate response using full RAG pipeline: retrieval + LLM generation.
        """
        import time

        start_time = time.perf_counter()
        input_tokens = 0
        output_tokens = 0

        try:
            # Get query from test case
            nl_query = getattr(test_case, "nl_query", None)
            query = test_case.input.get("query", nl_query or "")

            if not query:
                return SystemResponse(
                    raw_output=json.dumps({"error": "No query provided"}),
                    nl_response=None,
                    latency_ms=0,
                    error="No query in test case",
                )

            # Add company context if available
            company_name = test_case.input.get("company_name")
            retrieval_query = f"{company_name}: {query}" if company_name else query

            # Step 1: Retrieve documents
            results = await retriever.retrieve(
                query=retrieval_query,
                document_types=None,
                limit=5,  # Top 5 for generation context
                threshold=0.0,
                use_cache=False,
            )

            # Extract retrieved doc IDs and build context
            retrieved_doc_ids = [str(r.id) for r in results]
            retrieved_chunks = [
                {
                    "id": str(r.id),
                    "text": r.content[:1000],  # More content for generation
                    "score": r.score,
                }
                for r in results
            ]

            # Build context string with numbered sources
            context_parts = []
            for i, r in enumerate(results, 1):
                context_parts.append(f"[Source {i}]\n{r.content[:1500]}")
            context = "\n\n---\n\n".join(context_parts)

            # Step 2: Generate answer using LLM
            user_prompt = RAG_USER_PROMPT.format(query=query, context=context)

            response = llm_client.messages.create(
                model="claude-3-5-haiku-20241022",
                max_tokens=1024,
                system=RAG_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )

            generated_answer = response.content[0].text
            input_tokens = response.usage.input_tokens
            output_tokens = response.usage.output_tokens

            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Build sources list with citations
            sources = []
            for i, r in enumerate(results, 1):
                # Check if this source was cited
                cited = f"[Source {i}]" in generated_answer
                sources.append(
                    {
                        "id": str(r.id),
                        "citation_marker": f"[Source {i}]",
                        "cited": cited,
                        "text_preview": r.content[:200],
                    }
                )

            return SystemResponse(
                raw_output=json.dumps(
                    {
                        "response": generated_answer,
                        "retrieved_doc_ids": retrieved_doc_ids,
                        "retrieved_chunks": retrieved_chunks,
                        "context": context[:2000],  # Truncate for storage
                        "sources": sources,
                    }
                ),
                nl_response=generated_answer,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            test_case_id = getattr(test_case, "id", "unknown")
            logger.warning(
                f"RAG generation failed for test case {test_case_id}: {e}",
                exc_info=True,
                extra={"test_case_id": test_case_id},
            )
            return SystemResponse(
                raw_output=json.dumps({"error": str(e)}),
                nl_response=None,
                latency_ms=latency_ms,
                error=str(e),
            )

    return generate_rag_full_response


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

        # Simulate token usage for testing cost tracking
        # Real evaluations should use create_rag_generation_generator for actual tokens
        simulated_input_tokens = random.randint(500, 2000)
        simulated_output_tokens = random.randint(100, 500)

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
            input_tokens=simulated_input_tokens,
            output_tokens=simulated_output_tokens,
        )

    return generate_rag_response
