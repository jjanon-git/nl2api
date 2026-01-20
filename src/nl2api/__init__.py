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
"""

from src.nl2api.orchestrator import NL2APIOrchestrator
from src.nl2api.models import NL2APIResponse, ClarificationQuestion, ConversationTurn
from src.nl2api.config import NL2APIConfig, load_config

__all__ = [
    # Core
    "NL2APIOrchestrator",
    "NL2APIResponse",
    "ClarificationQuestion",
    "ConversationTurn",
    # Config
    "NL2APIConfig",
    "load_config",
]
