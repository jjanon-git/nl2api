"""
Domain Agents

Specialized agents for each API domain (Datastream, Fundamentals,
Officers, Estimates, Screening).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.nl2api.agents.base import BaseDomainAgent
from src.nl2api.agents.datastream import DatastreamAgent
from src.nl2api.agents.estimates import EstimatesAgent
from src.nl2api.agents.fundamentals import FundamentalsAgent
from src.nl2api.agents.officers import OfficersAgent
from src.nl2api.agents.protocols import (
    AgentContext,
    AgentResult,
    DomainAgent,
)
from src.nl2api.agents.screening import ScreeningAgent

if TYPE_CHECKING:
    from src.nl2api.llm.protocols import LLMProvider
    from src.nl2api.rag.protocols import RAGRetriever


# Agent registry mapping domain names to agent classes
AGENT_REGISTRY: dict[str, type[BaseDomainAgent]] = {
    "datastream": DatastreamAgent,
    "estimates": EstimatesAgent,
    "fundamentals": FundamentalsAgent,
    "officers": OfficersAgent,
    "screening": ScreeningAgent,
}


def get_agent_by_name(
    name: str,
    llm: LLMProvider,
    rag: RAGRetriever | None = None,
) -> BaseDomainAgent:
    """
    Create an agent instance by domain name.

    This factory function allows creating agents dynamically for component-level
    evaluation (tool_only mode in eval matrix).

    Args:
        name: Domain name (datastream, estimates, fundamentals, officers, screening)
        llm: LLM provider for the agent
        rag: Optional RAG retriever for context

    Returns:
        Instantiated domain agent

    Raises:
        ValueError: If the domain name is not recognized

    Example:
        >>> from src.nl2api.llm.factory import create_llm_provider
        >>> llm = create_llm_provider("anthropic", api_key, "claude-3-5-sonnet-20241022")
        >>> agent = get_agent_by_name("datastream", llm)
        >>> result = await agent.process(context)
    """
    name_lower = name.lower()
    if name_lower not in AGENT_REGISTRY:
        available = ", ".join(AGENT_REGISTRY.keys())
        raise ValueError(f"Unknown agent: '{name}'. Available agents: {available}")

    agent_class = AGENT_REGISTRY[name_lower]
    return agent_class(llm=llm, rag=rag)


def list_available_agents() -> list[str]:
    """Return list of available agent domain names."""
    return list(AGENT_REGISTRY.keys())


__all__ = [
    # Protocols
    "AgentContext",
    "AgentResult",
    "DomainAgent",
    # Base class
    "BaseDomainAgent",
    # Domain agents
    "DatastreamAgent",
    "EstimatesAgent",
    "FundamentalsAgent",
    "OfficersAgent",
    "ScreeningAgent",
    # Factory functions
    "get_agent_by_name",
    "list_available_agents",
    "AGENT_REGISTRY",
]
