"""
NL2API Orchestrator

Main entry point for the NL2API system.
Coordinates query classification, entity resolution, RAG retrieval,
and domain agent routing. Supports multi-turn conversations.
"""

from __future__ import annotations

import logging
import time
from datetime import timedelta
from typing import Any
from uuid import UUID, uuid4

from CONTRACTS import ToolCall
from src.nl2api.agents.protocols import AgentContext, DomainAgent
from src.nl2api.clarification.detector import AmbiguityDetector
from src.nl2api.conversation.manager import ConversationManager, ConversationStorage
from src.nl2api.conversation.models import ConversationTurn
from src.nl2api.models import ClarificationQuestion, NL2APIResponse
from src.nl2api.llm.protocols import LLMMessage, LLMProvider, MessageRole
from src.nl2api.rag.protocols import RAGRetriever
from src.nl2api.resolution.protocols import EntityResolver

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
        history_limit: int = 5,
        session_ttl_minutes: int = 30,
    ):
        """
        Initialize the orchestrator.

        Args:
            llm: LLM provider for classification and fallback
            agents: Dictionary of domain name to agent
            rag: Optional RAG retriever
            entity_resolver: Optional entity resolver
            conversation_storage: Optional storage for multi-turn conversations
            ambiguity_detector: Optional detector for ambiguous queries
            history_limit: Maximum conversation turns to keep in context
            session_ttl_minutes: Session timeout in minutes
        """
        self._llm = llm
        self._agents = agents
        self._rag = rag
        self._entity_resolver = entity_resolver
        self._ambiguity_detector = ambiguity_detector or AmbiguityDetector()
        self._conversation_manager = ConversationManager(
            storage=conversation_storage,
            history_limit=history_limit,
            session_ttl_minutes=session_ttl_minutes,
        )

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
            domain = await self._classify_query(effective_query)
            logger.info(f"Query classified to domain: {domain}")

            if domain not in self._agents:
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

            # Step 5: Retrieve context from RAG
            field_codes = []
            query_examples = []
            if self._rag:
                # Get relevant field codes
                field_results = await self._rag.retrieve_field_codes(
                    query=effective_query,
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

                # Get similar query examples
                example_results = await self._rag.retrieve_examples(
                    query=effective_query,
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

    async def _classify_query(self, query: str) -> str:
        """
        Classify the query to determine the appropriate domain.

        Uses LLM to classify the query into one of the available domains.

        Args:
            query: Natural language query

        Returns:
            Domain name string
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
                return best_domain

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
            return domain

        # Try to fuzzy match
        for domain_name in self._agents.keys():
            if domain_name.lower() in domain:
                return domain_name

        # Return first domain as fallback (should be configurable)
        return list(self._agents.keys())[0] if self._agents else "unknown"

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
