"""
NL2API Evaluation Adapter

Adapts NL2API to work as a target system in the evaluation framework.
Enables using test cases to validate NL2API accuracy.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Protocol, runtime_checkable

from CONTRACTS import SystemResponse
from src.nl2api.models import NL2APIResponse
from src.nl2api.orchestrator import NL2APIOrchestrator

logger = logging.getLogger(__name__)


@runtime_checkable
class TargetSystem(Protocol):
    """Protocol for target systems that can be evaluated."""

    async def invoke(self, nl_query: str) -> SystemResponse:
        """
        Invoke the target system with a natural language query.

        Args:
            nl_query: Natural language query

        Returns:
            SystemResponse with the system's output
        """
        ...


class NL2APITargetAdapter:
    """
    Makes NL2API behave like a target system for WaterfallEvaluator.

    Wraps the NL2APIOrchestrator and converts its responses to the
    SystemResponse format expected by the evaluation pipeline.

    Example usage:
        ```python
        # Create orchestrator with agents
        orchestrator = NL2APIOrchestrator(llm=llm, agents={"estimates": agent})

        # Wrap in adapter
        adapter = NL2APITargetAdapter(orchestrator)

        # Use in evaluation
        response = await adapter.invoke("What is Apple's EPS estimate?")
        scorecard = await evaluator.evaluate(test_case, response, "worker-1")
        ```
    """

    def __init__(
        self,
        orchestrator: NL2APIOrchestrator,
        session_id: str | None = None,
    ):
        """
        Initialize the adapter.

        Args:
            orchestrator: The NL2API orchestrator to wrap
            session_id: Optional session ID for multi-turn conversations
        """
        self._orchestrator = orchestrator
        self._session_id = session_id
        self._turn_count = 0

    async def invoke(self, nl_query: str) -> SystemResponse:
        """
        Invoke NL2API and convert the response to SystemResponse format.

        Args:
            nl_query: Natural language query

        Returns:
            SystemResponse with tool calls as raw_output
        """
        start_time = time.perf_counter()
        error_msg: str | None = None

        try:
            # Call the orchestrator
            # Session management is now handled internally by the orchestrator
            nl2api_response = await self._orchestrator.process(
                query=nl_query,
                session_id=self._session_id,
            )

            # Track turns for multi-turn conversations
            if self._session_id:
                self._turn_count += 1
                # Update session_id from response if not set
                if nl2api_response.session_id:
                    self._session_id = nl2api_response.session_id

            # Convert to SystemResponse
            return self._convert_response(nl2api_response, start_time)

        except Exception as e:
            logger.error(f"NL2API invocation failed: {e}", exc_info=True)
            error_msg = str(e)
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            return SystemResponse(
                raw_output="[]",  # Empty tool calls
                parsed_tool_calls=None,
                nl_response=None,
                latency_ms=latency_ms,
                error=error_msg,
            )

    def _convert_response(
        self,
        nl2api_response: NL2APIResponse,
        start_time: float,
    ) -> SystemResponse:
        """
        Convert NL2APIResponse to SystemResponse format.

        Args:
            nl2api_response: Response from NL2API orchestrator
            start_time: Start time for latency calculation

        Returns:
            SystemResponse suitable for evaluation
        """
        latency_ms = int((time.perf_counter() - start_time) * 1000)

        # Handle clarification responses as errors
        if nl2api_response.needs_clarification:
            questions = [q.question for q in nl2api_response.clarification_questions]
            error_msg = f"Needs clarification: {'; '.join(questions)}"

            return SystemResponse(
                raw_output="[]",
                parsed_tool_calls=None,
                nl_response=error_msg,
                latency_ms=latency_ms,
                error=error_msg,
            )

        # Convert tool calls to raw JSON format
        tool_calls_json = [
            {
                "tool_name": tc.tool_name,
                "arguments": dict(tc.arguments),
            }
            for tc in nl2api_response.tool_calls
        ]

        raw_output = json.dumps(tool_calls_json)

        # Include reasoning as nl_response if available
        nl_response = nl2api_response.reasoning if nl2api_response.reasoning else None

        return SystemResponse(
            raw_output=raw_output,
            parsed_tool_calls=nl2api_response.tool_calls if nl2api_response.tool_calls else None,
            nl_response=nl_response,
            execution_data={
                "domain": nl2api_response.domain,
                "confidence": nl2api_response.confidence,
                "resolved_entities": nl2api_response.resolved_entities,
                "tokens_used": nl2api_response.tokens_used,
            },
            latency_ms=latency_ms,
            # Pass through token counts for cost calculation
            input_tokens=nl2api_response.input_tokens if nl2api_response.input_tokens > 0 else None,
            output_tokens=nl2api_response.output_tokens
            if nl2api_response.output_tokens > 0
            else None,
        )

    def reset_conversation(self) -> None:
        """Reset the conversation for a new session."""
        self._session_id = None
        self._turn_count = 0

    def set_session_id(self, session_id: str | None) -> None:
        """
        Set the session ID for multi-turn conversations.

        Args:
            session_id: New session ID or None to disable session tracking
        """
        self._session_id = session_id
        if session_id is None:
            self._turn_count = 0

    @property
    def turn_count(self) -> int:
        """Return the number of turns in the current session."""
        return self._turn_count


class NL2APIBatchAdapter:
    """
    Adapter for batch evaluation of NL2API.

    Creates fresh orchestrator instances for each batch to ensure
    isolation between test cases.

    Example usage:
        ```python
        # Define a factory function to create orchestrators
        async def create_orchestrator() -> NL2APIOrchestrator:
            llm = await create_llm_provider(config)
            agent = EstimatesAgent(llm=llm)
            return NL2APIOrchestrator(llm=llm, agents={"estimates": agent})

        # Create batch adapter
        batch_adapter = NL2APIBatchAdapter(factory=create_orchestrator)

        # Use in batch runner
        async def evaluate_case(test_case: TestCase) -> SystemResponse:
            return await batch_adapter.invoke(test_case.nl_query)
        ```
    """

    def __init__(
        self,
        orchestrator_factory: Any,  # Callable[[], Awaitable[NL2APIOrchestrator]]
    ):
        """
        Initialize the batch adapter.

        Args:
            orchestrator_factory: Async factory function that creates orchestrators
        """
        self._factory = orchestrator_factory
        self._orchestrator: NL2APIOrchestrator | None = None

    async def invoke(self, nl_query: str) -> SystemResponse:
        """
        Invoke NL2API for a test case.

        Lazily creates the orchestrator on first call.

        Args:
            nl_query: Natural language query

        Returns:
            SystemResponse with tool calls
        """
        if self._orchestrator is None:
            self._orchestrator = await self._factory()

        adapter = NL2APITargetAdapter(self._orchestrator)
        return await adapter.invoke(nl_query)

    async def reset(self) -> None:
        """Reset the orchestrator for a fresh batch."""
        self._orchestrator = None
