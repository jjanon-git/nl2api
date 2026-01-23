"""NL2API - Natural Language to API Translation System.

A system for translating natural language queries into precise API calls
for LSEG financial data services.

Key components:
- Orchestrator: Main entry point coordinating all components
- LLM Providers: Pluggable LLM backends (Claude, OpenAI)
- RAG: Hybrid retrieval for field codes and examples
- Agents: Domain-specific agents for each API domain
- Entity Resolution: Company name to RIC resolution
- Clarification: Ambiguity detection and question generation
- Routing: FM-first query routing with caching
"""

from src.nl2api.config import NL2APIConfig, load_config
from src.nl2api.evaluation.adapter import NL2APIBatchAdapter, NL2APITargetAdapter
from src.nl2api.models import ClarificationQuestion, ConversationTurn, NL2APIResponse
from src.nl2api.orchestrator import NL2APIOrchestrator
from src.nl2api.routing import (
    AgentToolProvider,
    EscalatingLLMRouter,
    InMemoryRoutingCache,
    LLMToolRouter,
    QueryRouter,
    RouterResult,
    RoutingCache,
)

__all__ = [
    # Core
    "NL2APIOrchestrator",
    "NL2APIResponse",
    "ClarificationQuestion",
    "ConversationTurn",
    # Config
    "NL2APIConfig",
    "load_config",
    # Evaluation
    "NL2APITargetAdapter",
    "NL2APIBatchAdapter",
    # Routing
    "QueryRouter",
    "RouterResult",
    "LLMToolRouter",
    "EscalatingLLMRouter",
    "AgentToolProvider",
    "RoutingCache",
    "InMemoryRoutingCache",
]
