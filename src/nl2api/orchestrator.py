"""
NL2API Orchestrator

Main entry point for the NL2API system.
Coordinates query classification, entity resolution, context retrieval,
and domain agent routing. Supports multi-turn conversations.

Dual-Mode Context Retrieval:
- local: Use RAG retriever only
- mcp: Use MCP servers only
- hybrid: Try MCP first, fall back to RAG
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID, uuid4

from CONTRACTS import ToolCall
from src.nl2api.agents.protocols import AgentContext, DomainAgent
from src.nl2api.clarification.detector import AmbiguityDetector
from src.nl2api.conversation.manager import ConversationManager, ConversationStorage
from src.nl2api.conversation.models import ConversationTurn
from src.nl2api.models import ClarificationQuestion, NL2APIResponse
from src.nl2api.llm.protocols import LLMMessage, LLMProvider, MessageRole
from src.nl2api.mcp.context import ContextProvider, DualModeContextRetriever
from src.nl2api.rag.protocols import RAGRetriever
from src.nl2api.resolution.protocols import EntityResolver
from src.nl2api.routing.protocols import QueryRouter, RouterResult

if TYPE_CHECKING:
    from src.nl2api.mcp.client import MCPClient
    from src.nl2api.mcp.context import MCPContextRetriever
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
        context_mode: str = "local",
        mcp_retriever: MCPContextRetriever | None = None,
        mcp_fallback_enabled: bool = True,
    ):
        """
        Initialize the orchestrator.

        Args:
            llm: LLM provider for classification and fallback
            agents: Dictionary of domain name to agent
            rag: Optional RAG retriever (legacy, prefer context_retriever)
            entity_resolver: Optional entity resolver
            conversation_storage: Optional storage for multi-turn conversations
            ambiguity_detector: Optional detector for ambiguous queries
            router: Optional query router (defaults to LLMToolRouter)
            routing_cache: Optional cache for routing decisions
            routing_confidence_threshold: Threshold below which clarification is needed
            history_limit: Maximum conversation turns to keep in context
            session_ttl_minutes: Session timeout in minutes
            context_retriever: Optional context provider (overrides rag/mcp setup)
            context_mode: Mode for context retrieval ("local", "mcp", "hybrid")
            mcp_retriever: Optional MCP context retriever for dual-mode
            mcp_fallback_enabled: In hybrid mode, fall back to RAG if MCP fails
        """
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

        # Initialize context retriever (dual-mode support)
        self._context_retriever = context_retriever
        if self._context_retriever is None and (rag or mcp_retriever):
            # Create DualModeContextRetriever from individual retrievers
            self._context_retriever = self._create_context_retriever(
                rag=rag,
                mcp_retriever=mcp_retriever,
                mode=context_mode,
                fallback_enabled=mcp_fallback_enabled,
            )

        # Initialize router
        self._router = router
        self._routing_cache = routing_cache
        if self._router is None:
            # Create default LLMToolRouter
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

        try:
            # Step 0: Get or create conversation session
            session_uuid = UUID(session_id) if session_id else None
            session = await self._conversation_manager.get_or_create_session(
                session_id=session_uuid,
            )
            logger.info(f"Using session: {session.id}")

            # Step 1: Expand query if this is a follow-up
            effective_query = query
            expansion_result = None
            if session.total_turns > 0:
                expansion_result = self._conversation_manager.expand_query(
                    query, session
                )
                if expansion_result.was_expanded:
                    effective_query = expansion_result.expanded_query
                    logger.info(
                        f"Expanded query: '{query}' -> '{effective_query}'"
                    )

            # Step 2: Classify query to determine domain
            domain, confidence = await self._classify_query(effective_query)
            logger.info(f"Query classified to domain: {domain} (confidence={confidence:.2f})")

            # Check if confidence is too low
            if confidence < self._routing_confidence_threshold or domain not in self._agents:
                # Fallback to generic response
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
                    turn_number=session.total_turns + 1,
                    processing_time_ms=int((time.perf_counter() - start_time) * 1000),
                )

            # Step 3: Resolve entities (company names to RICs)
            resolved_entities = {}
            if self._entity_resolver:
                resolved_entities = await self._entity_resolver.resolve(effective_query)
                logger.info(f"Resolved entities: {resolved_entities}")

            # Also include entities from conversation context
            context_entities = session.get_all_entities()
            # Context entities provide defaults, new resolutions override
            merged_entities = {**context_entities, **resolved_entities}

            # Step 4: Check for ambiguity before proceeding
            ambiguity_analysis = await self._ambiguity_detector.analyze(
                effective_query,
                resolved_entities=merged_entities,
            )
            if ambiguity_analysis.is_ambiguous:
                logger.info(f"Query is ambiguous: {ambiguity_analysis.ambiguity_types}")
                return NL2APIResponse(
                    needs_clarification=True,
                    clarification_questions=ambiguity_analysis.clarification_questions,
                    domain=domain,
                    resolved_entities=merged_entities,
                    session_id=str(session.id),
                    turn_number=session.total_turns + 1,
                    processing_time_ms=int((time.perf_counter() - start_time) * 1000),
                )

            # Step 5: Retrieve context (dual-mode: RAG or MCP)
            field_codes, query_examples = await self._retrieve_context(
                query=effective_query,
                domain=domain,
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
            agent = self._agents[domain]
            result = await agent.process(context)

            # Step 8: Build response and save conversation turn
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            turn_number = session.total_turns + 1

            if result.needs_clarification:
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
                )

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
            )

        except Exception as e:
            logger.error(f"Error processing query: {e}", exc_info=True)
            processing_time_ms = int((time.perf_counter() - start_time) * 1000)
            return NL2APIResponse(
                needs_clarification=True,
                clarification_questions=(
                    ClarificationQuestion(
                        question="Sorry, I encountered an error processing your request. Please try again.",
                    ),
                ),
                processing_time_ms=processing_time_ms,
            )

    def _create_default_router(self) -> QueryRouter:
        """
        Create the default LLMToolRouter.

        Called when no router is provided to __init__.
        """
        from src.nl2api.routing.llm_router import LLMToolRouter
        from src.nl2api.routing.providers import AgentToolProvider

        # Wrap agents as tool providers
        providers = [
            AgentToolProvider(agent)
            for agent in self._agents.values()
        ]

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
            f"- {name}: {agent.domain_description}"
            for name, agent in self._agents.items()
        )

        classification_prompt = f"""Given the following query, determine which API domain is most appropriate.

Available domains:
{domain_descriptions}

Query: {query}

Respond with ONLY the domain name (one of: {', '.join(self._agents.keys())}).
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
        """
        self._agents[agent.domain_name] = agent
        logger.info(f"Registered agent for domain: {agent.domain_name}")

    def get_domains(self) -> list[str]:
        """Return list of available domain names."""
        return list(self._agents.keys())

    def _create_context_retriever(
        self,
        rag: RAGRetriever | None,
        mcp_retriever: MCPContextRetriever | None,
        mode: str,
        fallback_enabled: bool,
    ) -> DualModeContextRetriever:
        """
        Create a DualModeContextRetriever from individual retrievers.

        Args:
            rag: Optional RAG retriever
            mcp_retriever: Optional MCP context retriever
            mode: Context mode ("local", "mcp", "hybrid")
            fallback_enabled: Whether to fall back to RAG in hybrid mode

        Returns:
            Configured DualModeContextRetriever
        """
        # Wrap RAG retriever to match ContextProvider protocol
        rag_wrapper = None
        if rag:
            rag_wrapper = _RAGContextAdapter(rag)

        return DualModeContextRetriever(
            rag_retriever=rag_wrapper,
            mcp_retriever=mcp_retriever,
            mode=mode,
            fallback_enabled=fallback_enabled,
        )

    async def _retrieve_context(
        self,
        query: str,
        domain: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """
        Retrieve field codes and query examples for context.

        Uses the configured context retriever (DualModeContextRetriever)
        which can source from RAG, MCP, or both depending on mode.

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


class _RAGContextAdapter:
    """
    Adapter to make RAGRetriever compatible with ContextProvider protocol.

    This allows the DualModeContextRetriever to use the existing RAG
    infrastructure without changes.
    """

    def __init__(self, rag: RAGRetriever):
        self._rag = rag

    async def get_field_codes(
        self,
        query: str,
        domain: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """Retrieve field codes via RAG and convert to dict format."""
        results = await self._rag.retrieve_field_codes(
            query=query,
            domain=domain,
            limit=limit,
        )
        return [
            {
                "code": r.field_code,
                "description": r.content,
                "source": "rag",
            }
            for r in results
        ]

    async def get_query_examples(
        self,
        query: str,
        domain: str,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve query examples via RAG and convert to dict format."""
        results = await self._rag.retrieve_examples(
            query=query,
            domain=domain,
            limit=limit,
        )
        return [
            {
                "query": r.example_query,
                "api_call": r.example_api_call,
                "source": "rag",
            }
            for r in results
        ]
