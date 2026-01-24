"""
Base Domain Agent

Abstract base class providing common functionality for domain agents.
"""

from __future__ import annotations

import logging
import warnings
from abc import ABC, abstractmethod

from CONTRACTS import ToolCall
from src.evalkit.common.telemetry import get_tracer
from src.nl2api.agents.protocols import AgentContext, AgentResult
from src.nl2api.llm.protocols import (
    LLMMessage,
    LLMProvider,
    LLMToolDefinition,
    MessageRole,
)
from src.rag.retriever.protocols import RAGRetriever

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class BaseDomainAgent(ABC):
    """
    Abstract base class for domain agents.

    Provides common functionality for LLM-based API call generation.
    """

    def __init__(
        self,
        llm: LLMProvider,
        rag: RAGRetriever | None = None,
    ):
        """
        Initialize the domain agent.

        Args:
            llm: LLM provider for generating API calls
            rag: Optional RAG retriever for context
        """
        self._llm = llm
        self._rag = rag

    @property
    @abstractmethod
    def domain_name(self) -> str:
        """Return the domain name."""
        ...

    @property
    @abstractmethod
    def domain_description(self) -> str:
        """Return the domain description."""
        ...

    @property
    def capabilities(self) -> tuple[str, ...]:
        """
        Return the data types this agent handles.

        Override in subclasses for better routing context.
        """
        return ()

    @property
    def example_queries(self) -> tuple[str, ...]:
        """
        Return example queries this agent handles well.

        Override in subclasses for better routing context.
        """
        return ()

    @abstractmethod
    def get_system_prompt(self) -> str:
        """Return the system prompt for this domain."""
        ...

    @abstractmethod
    def get_tools(self) -> list[LLMToolDefinition]:
        """Return the tools available for this domain."""
        ...

    async def process(
        self,
        context: AgentContext,
    ) -> AgentResult:
        """
        Process a query and generate API calls.

        Uses the LLM with domain-specific system prompt and tools
        to generate appropriate API calls.

        Args:
            context: AgentContext with query and context

        Returns:
            AgentResult with tool calls or clarification
        """
        with tracer.start_as_current_span("agent.process") as span:
            span.set_attribute("agent.name", self.domain_name)
            span.set_attribute("agent.query_length", len(context.query))
            span.set_attribute("agent.has_conversation_history", bool(context.conversation_history))
            span.set_attribute(
                "agent.resolved_entities_count",
                len(context.resolved_entities) if context.resolved_entities else 0,
            )

            # Build the prompt
            messages = self._build_messages(context)
            tools = self.get_tools()
            span.set_attribute("agent.tools_count", len(tools))

            # Call LLM
            try:
                response = await self._llm.complete_with_retry(
                    messages=messages,
                    tools=tools,
                    temperature=0.0,
                )
            except Exception as e:
                logger.error(f"LLM call failed: {e}")
                span.set_attribute("agent.error", str(e))
                span.set_attribute("agent.result", "error")
                return AgentResult(
                    needs_clarification=True,
                    clarification_questions=(
                        "Sorry, I encountered an error processing your request. Please try again.",
                    ),
                    domain=self.domain_name,
                )

            # Parse response
            if response.has_tool_calls:
                tool_calls = tuple(
                    ToolCall(
                        tool_name=tc.name,
                        arguments=tc.arguments,
                    )
                    for tc in response.tool_calls
                )
                span.set_attribute("agent.tool_calls_count", len(tool_calls))
                span.set_attribute("agent.confidence", 0.9)
                span.set_attribute("agent.result", "tool_calls")
                span.set_attribute("agent.tokens_used", response.usage.get("total_tokens", 0))
                return AgentResult(
                    tool_calls=tool_calls,
                    confidence=0.9,  # High confidence when tool calls generated
                    reasoning=response.content,
                    domain=self.domain_name,
                    raw_llm_response=response.content,
                    tokens_used=response.usage.get("total_tokens", 0),
                    tokens_prompt=response.usage.get("prompt_tokens", 0),
                    tokens_completion=response.usage.get("completion_tokens", 0),
                )
            else:
                # No tool calls - might need clarification
                result = self._parse_text_response(response.content)
                span.set_attribute("agent.confidence", result.confidence)
                span.set_attribute("agent.needs_clarification", result.needs_clarification)
                span.set_attribute(
                    "agent.result", "clarification" if result.needs_clarification else "text"
                )
                return result

    def _build_messages(self, context: AgentContext) -> list[LLMMessage]:
        """Build the message list for the LLM."""
        messages = []

        # System prompt
        system_prompt = self.get_system_prompt()

        # Add retrieved context if available
        if context.field_codes:
            field_codes_text = "\n".join(
                f"- {fc.get('code', '')}: {fc.get('description', '')}" for fc in context.field_codes
            )
            system_prompt += f"\n\nAvailable field codes:\n{field_codes_text}"

        if context.query_examples:
            examples_text = "\n".join(
                f"Q: {ex.get('query', '')}\nA: {ex.get('api_call', '')}"
                for ex in context.query_examples
            )
            system_prompt += f"\n\nExample queries:\n{examples_text}"

        # Add string-format conversation history to system prompt
        if context.conversation_history and isinstance(context.conversation_history, str):
            if context.conversation_history.strip():
                system_prompt += f"\n\n{context.conversation_history}"

        messages.append(
            LLMMessage(
                role=MessageRole.SYSTEM,
                content=system_prompt,
            )
        )

        # Add list-format conversation history as separate messages
        if context.conversation_history and isinstance(context.conversation_history, list):
            for turn in context.conversation_history:
                if turn.get("role") == "user":
                    messages.append(
                        LLMMessage(
                            role=MessageRole.USER,
                            content=turn.get("content", ""),
                        )
                    )
                elif turn.get("role") == "assistant":
                    messages.append(
                        LLMMessage(
                            role=MessageRole.ASSISTANT,
                            content=turn.get("content", ""),
                        )
                    )

        # Add resolved entities if any
        query_with_context = context.expanded_query or context.query
        if context.resolved_entities:
            entities_text = ", ".join(
                f"{name} -> {ric}" for name, ric in context.resolved_entities.items()
            )
            query_with_context = f"{query_with_context}\n\n[Resolved entities: {entities_text}]"

        messages.append(
            LLMMessage(
                role=MessageRole.USER,
                content=query_with_context,
            )
        )

        return messages

    def _parse_text_response(self, content: str) -> AgentResult:
        """Parse a text response (no tool calls) into an AgentResult."""
        # Check for clarification patterns
        clarification_indicators = [
            "could you clarify",
            "could you specify",
            "did you mean",
            "which one",
            "please specify",
            "i need more information",
            "can you be more specific",
        ]

        content_lower = content.lower()
        if any(indicator in content_lower for indicator in clarification_indicators):
            return AgentResult(
                needs_clarification=True,
                clarification_questions=(content,),
                domain=self.domain_name,
                raw_llm_response=content,
            )

        # Otherwise, treat as low-confidence response
        return AgentResult(
            confidence=0.3,
            reasoning=content,
            domain=self.domain_name,
            raw_llm_response=content,
        )

    async def can_handle(self, query: str) -> float:
        """
        Check if this agent can handle the given query.

        .. deprecated:: 0.1.0
            Will be removed in v2.0. Use LLMToolRouter for query routing instead.

        Uses keyword matching and optional LLM classification.

        Args:
            query: Natural language query

        Returns:
            Confidence score (0.0 to 1.0)
        """
        warnings.warn(
            "can_handle() is deprecated and will be removed in v2.0. "
            "Use LLMToolRouter for query routing instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Subclasses should override with domain-specific logic
        return 0.0
