"""
E2E Test Fixtures

Provides real LLM providers and full system setup for end-to-end testing.
These tests use real API calls and cost money - run sparingly.

Requires:
    - NL2API_ANTHROPIC_API_KEY in environment or .env file
    - Optionally: DATABASE_URL for tests that need persistence
"""

import os
from pathlib import Path

import pytest


# Load .env file before any imports that might need env vars
def _load_env():
    """Load environment variables from .env file."""
    env_file = Path(__file__).parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value


_load_env()


def has_api_key() -> bool:
    """Check if Anthropic API key is available."""
    return bool(os.environ.get("NL2API_ANTHROPIC_API_KEY"))


# Skip all e2e tests if no API key
pytestmark = pytest.mark.skipif(
    not has_api_key(), reason="NL2API_ANTHROPIC_API_KEY not set - skipping e2e tests"
)


@pytest.fixture(scope="session")
def api_key() -> str:
    """Get Anthropic API key."""
    key = os.environ.get("NL2API_ANTHROPIC_API_KEY")
    if not key:
        pytest.skip("NL2API_ANTHROPIC_API_KEY not set")
    return key


@pytest.fixture(scope="session")
def llm_provider(api_key):
    """Create a real Claude LLM provider (session-scoped for cost efficiency)."""
    from src.nl2api.llm.claude import ClaudeProvider

    return ClaudeProvider(
        api_key=api_key,
        model="claude-3-5-haiku-latest",  # Use Haiku for cost efficiency
    )


@pytest.fixture(scope="session")
def routing_llm(api_key):
    """Separate LLM for routing (can use smaller/faster model)."""
    from src.nl2api.llm.claude import ClaudeProvider

    return ClaudeProvider(
        api_key=api_key,
        model="claude-3-5-haiku-latest",
    )


@pytest.fixture(scope="session")
def entity_resolver():
    """Create real entity resolver with static mappings."""
    from src.nl2api.resolution.resolver import ExternalEntityResolver

    return ExternalEntityResolver()


@pytest.fixture(scope="session")
def all_agents(llm_provider):
    """Create all domain agents."""
    from src.nl2api.agents.datastream import DatastreamAgent
    from src.nl2api.agents.estimates import EstimatesAgent
    from src.nl2api.agents.fundamentals import FundamentalsAgent
    from src.nl2api.agents.officers import OfficersAgent
    from src.nl2api.agents.screening import ScreeningAgent

    return {
        "datastream": DatastreamAgent(llm=llm_provider),
        "estimates": EstimatesAgent(llm=llm_provider),
        "fundamentals": FundamentalsAgent(llm=llm_provider),
        "officers": OfficersAgent(llm=llm_provider),
        "screening": ScreeningAgent(llm=llm_provider),
    }


@pytest.fixture(scope="session")
def orchestrator(llm_provider, routing_llm, all_agents, entity_resolver):
    """Create full orchestrator with all components."""
    from src.nl2api.orchestrator import NL2APIOrchestrator
    from src.nl2api.routing.llm_router import LLMToolRouter
    from src.nl2api.routing.providers import AgentToolProvider

    # Create router with agent tool providers
    tool_providers = [AgentToolProvider(agent) for agent in all_agents.values()]
    router = LLMToolRouter(llm=routing_llm, tool_providers=tool_providers)

    return NL2APIOrchestrator(
        llm=llm_provider,
        agents=all_agents,
        entity_resolver=entity_resolver,
        router=router,
    )


@pytest.fixture
def fresh_orchestrator(llm_provider, routing_llm, all_agents, entity_resolver):
    """Create a fresh orchestrator per test (for conversation isolation)."""
    from src.nl2api.orchestrator import NL2APIOrchestrator
    from src.nl2api.routing.llm_router import LLMToolRouter
    from src.nl2api.routing.providers import AgentToolProvider

    tool_providers = [AgentToolProvider(agent) for agent in all_agents.values()]
    router = LLMToolRouter(llm=routing_llm, tool_providers=tool_providers)

    return NL2APIOrchestrator(
        llm=llm_provider,
        agents=all_agents,
        entity_resolver=entity_resolver,
        router=router,
    )
