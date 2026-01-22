"""
Integration tests for NL2API MCP server initialization.

These tests verify that the MCP server correctly initializes NL2API components
with various environment variable configurations.

Regression test for: Orchestrator creating new NL2APIConfig internally,
which looked for NL2API_ANTHROPIC_API_KEY instead of using the injected LLM.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestNL2APIInitialization:
    """Test NL2API component initialization in MCP server context."""

    @pytest.mark.asyncio
    async def test_orchestrator_uses_injected_llm_for_router(self):
        """
        Verify orchestrator uses injected LLM, not environment variables.

        This is a regression test for the bug where the orchestrator's
        _create_default_router() method created a new NL2APIConfig()
        which looked for NL2API_ANTHROPIC_API_KEY instead of using
        the LLM provider that was passed to __init__.
        """
        from src.nl2api.orchestrator import NL2APIOrchestrator
        from src.nl2api.agents.protocols import AgentContext, AgentResult, DomainAgent

        # Create a mock LLM provider
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock(return_value=MagicMock(
            content='{"domain": "fundamentals", "confidence": 0.9}',
            usage=MagicMock(total_tokens=100),
        ))

        # Create a mock agent
        mock_agent = MagicMock(spec=DomainAgent)
        mock_agent.domain_name = "fundamentals"
        mock_agent.domain_description = "Financial ratios and statements"
        mock_agent.can_handle = AsyncMock(return_value=0.5)
        mock_agent.process = AsyncMock(return_value=AgentResult(
            tool_calls=(),
            confidence=0.9,
            reasoning="Test",
        ))

        # Ensure NL2API_ANTHROPIC_API_KEY is NOT set
        env_backup = os.environ.get("NL2API_ANTHROPIC_API_KEY")
        if "NL2API_ANTHROPIC_API_KEY" in os.environ:
            del os.environ["NL2API_ANTHROPIC_API_KEY"]

        try:
            # This should NOT raise "ANTHROPIC_API_KEY not set"
            # because the orchestrator should use the injected LLM
            orchestrator = NL2APIOrchestrator(
                llm=mock_llm,
                agents={"fundamentals": mock_agent},
            )

            # Verify the router was created using the injected LLM
            assert orchestrator._router is not None
            assert orchestrator._router._llm is mock_llm

        finally:
            # Restore env var if it was set
            if env_backup is not None:
                os.environ["NL2API_ANTHROPIC_API_KEY"] = env_backup

    @pytest.mark.asyncio
    async def test_mcp_server_lifespan_with_anthropic_api_key_only(self):
        """
        Verify MCP server initializes NL2API tools with just ANTHROPIC_API_KEY.

        The server should work with ANTHROPIC_API_KEY (no prefix) because
        the stdio transport passes the API key directly to ClaudeProvider,
        then passes that provider to the orchestrator.
        """
        from src.nl2api.orchestrator import NL2APIOrchestrator
        from src.nl2api.routing.llm_router import LLMToolRouter
        from src.nl2api.routing.providers import AgentToolProvider
        from src.nl2api.agents.protocols import DomainAgent

        # Simulate the stdio.py initialization flow
        # Step 1: Create mock LLM provider (simulating ClaudeProvider with API key)
        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock()

        # Step 2: Create mock agent (simulating DatastreamAgent)
        mock_agent = MagicMock(spec=DomainAgent)
        mock_agent.domain_name = "datastream"
        mock_agent.domain_description = "Price and time series data"

        agents = {"datastream": mock_agent}

        # Step 3: Create router with the same LLM (this is the fix in stdio.py)
        router = LLMToolRouter(
            llm=mock_llm,
            tool_providers=[AgentToolProvider(agent) for agent in agents.values()],
        )

        # Step 4: Create orchestrator with pre-configured router
        # This should NOT try to load NL2API_ANTHROPIC_API_KEY from env
        orchestrator = NL2APIOrchestrator(
            llm=mock_llm,
            agents=agents,
            router=router,  # Pass router to avoid _create_default_router
        )

        # Verify initialization succeeded
        assert orchestrator._llm is mock_llm
        assert orchestrator._router is router
        assert "datastream" in orchestrator._agents

    @pytest.mark.asyncio
    async def test_orchestrator_default_router_uses_same_llm(self):
        """
        Verify that when no router is passed, orchestrator creates one
        using the injected LLM (not a new one from NL2APIConfig).
        """
        from src.nl2api.orchestrator import NL2APIOrchestrator
        from src.nl2api.agents.protocols import DomainAgent

        mock_llm = MagicMock()
        mock_llm.complete = AsyncMock()

        mock_agent = MagicMock(spec=DomainAgent)
        mock_agent.domain_name = "test"
        mock_agent.domain_description = "Test agent"

        # Create orchestrator without passing a router
        orchestrator = NL2APIOrchestrator(
            llm=mock_llm,
            agents={"test": mock_agent},
        )

        # The default router should use the SAME LLM instance
        assert orchestrator._router._llm is mock_llm
