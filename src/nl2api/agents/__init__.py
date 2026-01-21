"""
Domain Agents

Specialized agents for each API domain (Datastream, Fundamentals,
Officers, Estimates, Screening).
"""

from src.nl2api.agents.protocols import (
    AgentContext,
    AgentResult,
    DomainAgent,
)
from src.nl2api.agents.base import BaseDomainAgent
from src.nl2api.agents.datastream import DatastreamAgent
from src.nl2api.agents.estimates import EstimatesAgent
from src.nl2api.agents.fundamentals import FundamentalsAgent
from src.nl2api.agents.officers import OfficersAgent
from src.nl2api.agents.screening import ScreeningAgent

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
]
