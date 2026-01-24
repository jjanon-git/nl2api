"""
NL2API Orchestrator

Main entry point for the NL2API system.
Coordinates query classification, entity resolution, context retrieval,
and domain agent routing. Supports multi-turn conversations.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any
from uuid import UUID

from src.evalkit.common.telemetry import record_exception, trace_span
from src.nl2api.agents.protocols import AgentContext, DomainAgent
from src.nl2api.clarification.detector import AmbiguityDetector
from src.nl2api.conversation.manager import ConversationManager, ConversationStorage
from src.nl2api.conversation.models import ConversationTurn
from src.nl2api.llm.protocols import LLMMessage, LLMProvider, MessageRole
from src.nl2api.mcp.context import ContextProvider
from src.nl2api.models import ClarificationQuestion, NL2APIResponse
from src.nl2api.observability import RequestMetrics, emit_metrics
from src.nl2api.resolution.protocols import EntityResolver
from src.nl2api.routing.protocols import QueryRouter
from src.rag.retriever.protocols import RAGRetriever

if TYPE_CHECKING:
    from src.nl2api.routing.cache import RoutingCache

logger = logging.getLogger(__name__)


class NL2APIOrchestrator:
    """
    Main orchestrator for the NL2API system.

    Coordinates:
    - Query classification (routing to domain agent)
    - Entity resolution (company names to RICs)
    - RAG retrieval (field codes and examples)
    - Domain agent invocation
    - Clarification flow
    """

    def __init__(
        self,
        llm: LLMProvider,
        agents: dict[str, DomainAgent],
        rag: RAGRetriever | None = None,
        entity_resolver: EntityResolver | None = None,
        conversation_storage: ConversationStorage | None = None,
        ambiguity_detector: AmbiguityDetector | None = None,
        router: QueryRouter | None = None,
        routing_cache: RoutingCache | None = None,
        routing_confidence_threshold: float = 0.5,
        history_limit: int = 5,
        session_ttl_minutes: int = 30,
        context_retriever: ContextProvider | None = None,
    ):
        """
        Initialize the orchestrator.

        Args:
            llm: LLM provider for classification and fallback
            agents: Dictionary of domain name to agent
            rag: Optional RAG retriever for field codes and examples
            entity_resolver: Optional entity resolver
            conversation_storage: Optional storage for multi-turn conversations
            ambiguity_detector: Optional detector for ambiguous queries
            router: Optional query router (defaults to LLMToolRouter)
            routing_cache: Optional cache for routing decisions
            routing_confidence_threshold: Threshold below which clarification is needed
            history_limit: Maximum conversation turns to keep in context
            session_ttl_minutes: Session timeout in minutes
            context_retriever: Optional context provider (overrides rag)
        """
        # Validate protocol conformance for required parameters
        if not isinstance(llm, LLMProvider):
            raise TypeError(f"llm must implement LLMProvider protocol, got {type(llm).__name__}")

        # Validate agents dict
        for domain, agent in agents.items():
            if not isinstance(agent, DomainAgent):
                raise TypeError(
                    f"Agent for domain '{domain}' must implement DomainAgent protocol, "
                    f"got {type(agent).__name__}"
                )

        # Validate optional protocol parameters
        if rag is not None and not isinstance(rag, RAGRetriever):
            raise TypeError(f"rag must implement RAGRetriever protocol, got {type(rag).__name__}")
        if entity_resolver is not None and not isinstance(entity_resolver, EntityResolver):
            raise TypeError(
                f"entity_resolver must implement EntityResolver protocol, "
                f"got {type(entity_resolver).__name__}"
            )
        if conversation_storage is not None and not isinstance(
            conversation_storage, ConversationStorage
        ):
            raise TypeError(
                f"conversation_storage must implement ConversationStorage protocol, "
                f"got {type(conversation_storage).__name__}"
            )
        if router is not None and not isinstance(router, QueryRouter):
            raise TypeError(
                f"router must implement QueryRouter protocol, got {type(router).__name__}"
            )
        if context_retriever is not None and not isinstance(context_retriever, ContextProvider):
            raise TypeError(
                f"context_retriever must implement ContextProvider protocol, "
                f"got {type(context_retriever).__name__}"
            )

        self._llm = llm
        self._agents = agents
        self._rag = rag
        self._entity_resolver = entity_resolver
        self._ambiguity_detector = ambiguity_detector or AmbiguityDetector()
        self._routing_confidence_threshold = routing_confidence_threshold
        self._conversation_manager = ConversationManager(
            storage=conversation_storage,
            history_limit=history_limit,
            session_ttl_minutes=session_ttl_minutes,
        )

        # Context retriever: use provided or fall back to RAG wrapper
        self._context_retriever = context_retriever

        # Initialize router (create default if not provided)
        self._router = router
        self._routing_cache = routing_cache
        if self._router is None:
            self._router = self._create_default_router()

    async def process(
        self,
        query: str,
        session_id: str | None = None,
        clarification_response: str | None = None,
    ) -> NL2APIResponse:
        """
        Process a natural language query and generate API calls.

        Args:
            query: User's natural language query
            session_id: Optional session ID for multi-turn conversations
            clarification_response: Response to a previous clarification question

        Returns:
            NL2APIResponse with tool calls or clarification request
        """
        start_time = time.perf_counter()

        # Initialize metrics collection
        metrics = RequestMetrics(query=query, session_id=session_id)

        # Root span for entire request
        with trace_span("nl2api.process", {"query_length": len(query)}) as root_span:
            try:
                # Step 0: Get or create conversation session
                with trace_span("session.get_or_create") as session_span:
                    session_uuid = UUID(session_id) if session_id else None
                    session = await self._conversation_manager.get_or_create_session(
                        session_id=session_uuid,
                    )
                    metrics.session_id = str(session.id)
                    session_span.set_attribute("session.id", str(session.id))
                    session_span.set_attribute("session.turns", session.total_turns)
                    logger.info(f"Using session: {session.id}")

                # Step 1: Expand query if this is a follow-up
                with trace_span("query.expansion") as expansion_span:
                    effective_query = query
                    expansion_result = None
                    if session.total_turns > 0:
                        expansion_result = self._conversation_manager.expand_query(query, session)
                        if expansion_result.was_expanded:
                            effective_query = expansion_result.expanded_query
                            metrics.query_expanded = True
                            metrics.query_expansion_reason = "follow_up"
                            expansion_span.set_attribute("query.expanded", True)
                            expansion_span.set_attribute(
                                "query.expanded_length", len(effective_query)
                            )
                            logger.info(f"Expanded query: '{query}' -> '{effective_query}'")
                        else:
                            expansion_span.set_attribute("query.expanded", False)
                    else:
                        expansion_span.set_attribute("query.expanded", False)

                # Step 2: Classify query to determine domain
                with trace_span("routing") as routing_span:
                    routing_start = time.perf_counter()
                    domain, confidence = await self._classify_query(effective_query)
                    routing_latency = int((time.perf_counter() - routing_start) * 1000)

                    # Get routing details from last router result
                    routing_cached = getattr(self, "_last_router_result_cached", False)
                    routing_model = getattr(self, "_last_router_result_model", None)

                    routing_span.set_attribute("routing.domain", domain)
                    routing_span.set_attribute("routing.confidence", confidence)
                    routing_span.set_attribute("routing.cached", routing_cached)
                    routing_span.set_attribute("routing.latency_ms", routing_latency)
                    if routing_model:
                        routing_span.set_attribute("routing.model", routing_model)

                    metrics.set_routing_result(
                        domain=domain,
                        confidence=confidence,
                        cached=routing_cached,
                        model=routing_model,
                        latency_ms=routing_latency,
                    )
                    logger.info(
                        f"Query classified to domain: {domain} (confidence={confidence:.2f})"
                    )

                # Check if confidence is too low
                if confidence < self._routing_confidence_threshold or domain not in self._agents:
                    processing_time_ms = int((time.perf_counter() - start_time) * 1000)
                    turn_number = session.total_turns + 1
                    metrics.set_clarification("domain")
                    metrics.finalize(processing_time_ms)
                    root_span.set_attribute("clarification.type", "domain")
                    root_span.set_attribute("total_latency_ms", processing_time_ms)
                    await self._emit_metrics_safe(metrics)

                    # Save turn even when clarification is needed (preserves context for next turn)
                    turn = ConversationTurn(
                        turn_number=turn_number,
                        user_query=query,
                        expanded_query=effective_query if effective_query != query else None,
                        needs_clarification=True,
                        clarification_questions=("domain",),
                        processing_time_ms=processing_time_ms,
                    )
                    await self._conversation_manager.add_turn(session, turn)

                    return NL2APIResponse(
                        needs_clarification=True,
                        clarification_questions=(
                            ClarificationQuestion(
                                question="I'm not sure which API domain to use for this query. Could you specify what type of data you're looking for?",
                                options=tuple(self._agents.keys()),
                                category="domain",
                            ),
                        ),
                        session_id=str(session.id),
                        turn_number=turn_number,
                        processing_time_ms=processing_time_ms,
                    )

                # Step 3: Resolve entities (company names to RICs)
                with trace_span("entity.resolution") as entity_span:
                    entity_start = time.perf_counter()
                    resolved_entities = {}
                    if self._entity_resolver:
                        resolved_entities = await self._entity_resolver.resolve(effective_query)
                        logger.info(f"Resolved entities: {resolved_entities}")

                    entity_latency = int((time.perf_counter() - entity_start) * 1000)

                    # Also include entities from conversation context
                    context_entities = session.get_all_entities()
                    merged_entities = {**context_entities, **resolved_entities}

                    entity_span.set_attribute("entities.count", len(resolved_entities))
                    entity_span.set_attribute("entities.names", list(resolved_entities.keys()))
                    entity_span.set_attribute("entities.latency_ms", entity_latency)

                    metrics.set_entity_resolution(
                        extracted=list(resolved_entities.keys()),
                        resolved=resolved_entities,
                        method="static" if resolved_entities else "none",
                        latency_ms=entity_latency,
                    )

                # Step 4: Check for ambiguity before proceeding
                with trace_span("ambiguity.detection") as ambiguity_span:
                    ambiguity_analysis = await self._ambiguity_detector.analyze(
                        effective_query,
                        resolved_entities=merged_entities,
                    )
                    ambiguity_span.set_attribute(
                        "ambiguity.is_ambiguous", ambiguity_analysis.is_ambiguous
                    )
                    if ambiguity_analysis.is_ambiguous:
                        processing_time_ms = int((time.perf_counter() - start_time) * 1000)
                        turn_number = session.total_turns + 1
                        ambiguity_span.set_attribute(
                            "ambiguity.types", list(ambiguity_analysis.ambiguity_types)
                        )
                        logger.info(f"Query is ambiguous: {ambiguity_analysis.ambiguity_types}")
                        metrics.set_clarification("ambiguity")
                        metrics.finalize(processing_time_ms)
                        root_span.set_attribute("clarification.type", "ambiguity")
                        root_span.set_attribute("total_latency_ms", processing_time_ms)
                        await self._emit_metrics_safe(metrics)

                        # Save turn even when clarification is needed (preserves context for next turn)
                        turn = ConversationTurn(
                            turn_number=turn_number,
                            user_query=query,
                            expanded_query=effective_query if effective_query != query else None,
                            needs_clarification=True,
                            clarification_questions=tuple(
                                q.question for q in ambiguity_analysis.clarification_questions
                            ),
                            domain=domain,
                            resolved_entities=merged_entities,
                            processing_time_ms=processing_time_ms,
                        )
                        await self._conversation_manager.add_turn(session, turn)

                        return NL2APIResponse(
                            needs_clarification=True,
                            clarification_questions=ambiguity_analysis.clarification_questions,
                            domain=domain,
                            resolved_entities=merged_entities,
                            session_id=str(session.id),
                            turn_number=turn_number,
                            processing_time_ms=processing_time_ms,
                        )

                # Step 5: Retrieve context (dual-mode: RAG or MCP)
                with trace_span("context.retrieval") as context_span:
                    context_start = time.perf_counter()
                    field_codes, query_examples = await self._retrieve_context(
                        query=effective_query,
                        domain=domain,
                    )
                    context_latency = int((time.perf_counter() - context_start) * 1000)

                    context_span.set_attribute("context.field_codes", len(field_codes))
                    context_span.set_attribute("context.examples", len(query_examples))
                    context_span.set_attribute(
                        "context.mode", getattr(self, "_context_mode", "local")
                    )
                    context_span.set_attribute("context.latency_ms", context_latency)

                    metrics.set_context_retrieval(
                        mode=getattr(self, "_context_mode", "local"),
                        field_codes_count=len(field_codes),
                        examples_count=len(query_examples),
                        latency_ms=context_latency,
                    )

                # Step 6: Build context for agent including conversation history
                history_prompt = self._conversation_manager.build_history_prompt(session)
                context = AgentContext(
                    query=effective_query,
                    resolved_entities=merged_entities,
                    field_codes=field_codes,
                    query_examples=query_examples,
                    conversation_history=history_prompt,
                    session_id=str(session.id),
                )

                # Step 7: Invoke domain agent
                with trace_span("agent.process", {"agent": domain}) as agent_span:
                    agent_start = time.perf_counter()
                    agent = self._agents[domain]
                    result = await agent.process(context)
                    agent_latency = int((time.perf_counter() - agent_start) * 1000)

                    agent_span.set_attribute("agent.used_llm", result.used_llm)
                    agent_span.set_attribute("agent.tool_calls", len(result.tool_calls))
                    agent_span.set_attribute("agent.latency_ms", agent_latency)
                    if result.rule_matched:
                        agent_span.set_attribute("agent.rule_matched", result.rule_matched)
                    if result.tokens_used:
                        agent_span.set_attribute("agent.tokens", result.tokens_used)

                    metrics.set_agent_result(
                        domain=domain,
                        used_llm=result.used_llm,
                        rule_matched=result.rule_matched,
                        llm_model=result.llm_model,
                        tokens_prompt=result.tokens_prompt,
                        tokens_completion=result.tokens_completion,
                        latency_ms=agent_latency,
                    )

                # Step 8: Build response and save conversation turn
                processing_time_ms = int((time.perf_counter() - start_time) * 1000)
                turn_number = session.total_turns + 1

                if result.needs_clarification:
                    metrics.set_clarification("agent")
                    metrics.finalize(processing_time_ms)
                    root_span.set_attribute("clarification.type", "agent")
                    root_span.set_attribute("total_latency_ms", processing_time_ms)
                    await self._emit_metrics_safe(metrics)

                    # Save turn with clarification request
                    turn = ConversationTurn(
                        turn_number=turn_number,
                        user_query=query,
                        expanded_query=effective_query if effective_query != query else None,
                        needs_clarification=True,
                        clarification_questions=tuple(result.clarification_questions),
                        domain=domain,
                        resolved_entities=merged_entities,
                        processing_time_ms=processing_time_ms,
                    )
                    await self._conversation_manager.add_turn(session, turn)

                    return NL2APIResponse(
                        needs_clarification=True,
                        clarification_questions=tuple(
                            ClarificationQuestion(question=q)
                            for q in result.clarification_questions
                        ),
                        domain=domain,
                        resolved_entities=merged_entities,
                        session_id=str(session.id),
                        turn_number=turn_number,
                        raw_llm_response=result.raw_llm_response,
                        tokens_used=result.tokens_used,
                        processing_time_ms=processing_time_ms,
                        input_tokens=result.tokens_prompt,
                        output_tokens=result.tokens_completion,
                    )

                # Set successful output metrics
                metrics.set_tool_calls(result.tool_calls)
                metrics.finalize(processing_time_ms)
                root_span.set_attribute("success", True)
                root_span.set_attribute("tool_calls.count", len(result.tool_calls))
                root_span.set_attribute("total_latency_ms", processing_time_ms)
                await self._emit_metrics_safe(metrics)

                # Save successful turn
                turn = ConversationTurn(
                    turn_number=turn_number,
                    user_query=query,
                    expanded_query=effective_query if effective_query != query else None,
                    tool_calls=result.tool_calls,
                    domain=domain,
                    confidence=result.confidence,
                    resolved_entities=merged_entities,
                    processing_time_ms=processing_time_ms,
                )
                await self._conversation_manager.add_turn(session, turn)

                return NL2APIResponse(
                    tool_calls=result.tool_calls,
                    confidence=result.confidence,
                    reasoning=result.reasoning,
                    domain=domain,
                    resolved_entities=merged_entities,
                    session_id=str(session.id),
                    turn_number=turn_number,
                    raw_llm_response=result.raw_llm_response,
                    tokens_used=result.tokens_used,
                    processing_time_ms=processing_time_ms,
                    input_tokens=result.tokens_prompt,
                    output_tokens=result.tokens_completion,
                )

            except Exception as e:
                logger.error(f"Error processing query: {e}", exc_info=True)
                processing_time_ms = int((time.perf_counter() - start_time) * 1000)

                # Record error in metrics and span
                record_exception(e)
                root_span.set_attribute("success", False)
                root_span.set_attribute("error.type", type(e).__name__)
                root_span.set_attribute("total_latency_ms", processing_time_ms)
                metrics.set_error(e)
                metrics.finalize(processing_time_ms)
                await self._emit_metrics_safe(metrics)

                return NL2APIResponse(
                    needs_clarification=True,
                    clarification_questions=(
                        ClarificationQuestion(
                            question="Sorry, I encountered an error processing your request. Please try again.",
                        ),
                    ),
                    processing_time_ms=processing_time_ms,
                )

    async def _emit_metrics_safe(self, metrics: RequestMetrics) -> None:
        """Emit metrics without raising exceptions."""
        try:
            await emit_metrics(metrics)
        except Exception as e:
            logger.warning(f"Failed to emit metrics: {e}")

    def _create_default_router(self) -> QueryRouter:
        """
        Create the default LLMToolRouter.

        IMPORTANT: Uses the injected LLM provider directly. If you need a different
        model for routing (e.g., Haiku for cost savings), create the router externally
        and pass it to __init__. This avoids hidden dependencies on environment
        variables like NL2API_ANTHROPIC_API_KEY.
        """
        from src.nl2api.routing.llm_router import LLMToolRouter
        from src.nl2api.routing.providers import AgentToolProvider

        # Wrap agents as tool providers
        providers = [AgentToolProvider(agent) for agent in self._agents.values()]

        return LLMToolRouter(
            llm=self._llm,
            tool_providers=providers,
            cache=self._routing_cache,
        )

    async def _classify_query(self, query: str) -> tuple[str, float]:
        """
        Classify the query to determine the appropriate domain.

        Uses the FM-first router for classification.

        Args:
            query: Natural language query

        Returns:
            Tuple of (domain name, confidence score)
        """
        # Use the router for classification
        if self._router:
            try:
                result = await self._router.route(query)
                logger.info(
                    f"Router result: domain={result.domain}, "
                    f"confidence={result.confidence:.2f}, "
                    f"cached={result.cached}, "
                    f"latency={result.latency_ms}ms"
                )
                return result.domain, result.confidence
            except Exception as e:
                logger.warning(f"Router failed, falling back to legacy: {e}")
                # Fall through to legacy classification

        # Legacy fallback: keyword-based classification
        return await self._classify_query_legacy(query)

    async def _classify_query_legacy(self, query: str) -> tuple[str, float]:
        """
        Legacy query classification using keyword matching and LLM fallback.

        Deprecated: Will be removed when FM-first routing is fully validated.

        Args:
            query: Natural language query

        Returns:
            Tuple of (domain name, confidence score)
        """
        # First, try quick keyword-based classification
        domain_scores: dict[str, float] = {}
        for domain_name, agent in self._agents.items():
            score = await agent.can_handle(query)
            if score > 0:
                domain_scores[domain_name] = score

        # If we have a clear winner, use it
        if domain_scores:
            best_domain = max(domain_scores, key=domain_scores.get)
            if domain_scores[best_domain] >= 0.7:
                return best_domain, domain_scores[best_domain]

        # Otherwise, use LLM for classification
        domain_descriptions = "\n".join(
            f"- {name}: {agent.domain_description}" for name, agent in self._agents.items()
        )

        classification_prompt = f"""Given the following query, determine which API domain is most appropriate.

Available domains:
{domain_descriptions}

Query: {query}

Respond with ONLY the domain name (one of: {", ".join(self._agents.keys())}).
If you cannot determine the domain, respond with "unknown".
"""

        response = await self._llm.complete(
            messages=[
                LLMMessage(
                    role=MessageRole.USER,
                    content=classification_prompt,
                ),
            ],
            temperature=0.0,
            max_tokens=50,
        )

        # Parse response
        domain = response.content.strip().lower()

        # Validate domain
        if domain in self._agents:
            return domain, 0.8  # LLM classification confidence

        # Try to fuzzy match
        for domain_name in self._agents.keys():
            if domain_name.lower() in domain:
                return domain_name, 0.6  # Lower confidence for fuzzy match

        # Return first domain as fallback
        if self._agents:
            return list(self._agents.keys())[0], 0.3
        return "unknown", 0.0

    def register_agent(self, agent: DomainAgent) -> None:
        """
        Register a domain agent.

        Args:
            agent: Domain agent to register

        Raises:
            TypeError: If agent doesn't implement DomainAgent protocol
        """
        if not isinstance(agent, DomainAgent):
            raise TypeError(
                f"agent must implement DomainAgent protocol, got {type(agent).__name__}"
            )
        self._agents[agent.domain_name] = agent
        logger.info(f"Registered agent for domain: {agent.domain_name}")

    def get_domains(self) -> list[str]:
        """Return list of available domain names."""
        return list(self._agents.keys())

    async def _retrieve_context(
        self,
        query: str,
        domain: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Retrieve field codes and query examples for context.

        Uses the configured context retriever if available, otherwise
        falls back to direct RAG retrieval.

        Args:
            query: User's natural language query
            domain: Target domain for context retrieval

        Returns:
            Tuple of (field_codes, query_examples) lists
        """
        field_codes: list[dict[str, Any]] = []
        query_examples: list[dict[str, Any]] = []

        if self._context_retriever:
            try:
                # Use unified context retriever interface
                field_codes = await self._context_retriever.get_field_codes(
                    query=query,
                    domain=domain,
                    limit=5,
                )
                query_examples = await self._context_retriever.get_query_examples(
                    query=query,
                    domain=domain,
                    limit=3,
                )
                logger.debug(
                    f"Retrieved {len(field_codes)} field codes, "
                    f"{len(query_examples)} examples for domain={domain}"
                )
            except Exception as e:
                logger.warning(f"Context retrieval failed: {e}")
                # Continue with empty context rather than failing

        elif self._rag:
            # Legacy fallback: direct RAG retrieval
            try:
                field_results = await self._rag.retrieve_field_codes(
                    query=query,
                    domain=domain,
                    limit=5,
                )
                field_codes = [
                    {
                        "code": r.field_code,
                        "description": r.content,
                    }
                    for r in field_results
                ]

                example_results = await self._rag.retrieve_examples(
                    query=query,
                    domain=domain,
                    limit=3,
                )
                query_examples = [
                    {
                        "query": r.example_query,
                        "api_call": r.example_api_call,
                    }
                    for r in example_results
                ]
            except Exception as e:
                logger.warning(f"RAG retrieval failed: {e}")

        return field_codes, query_examples
