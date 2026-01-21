"""
NL2API Query Routing Module

Provides FM-first routing for the NL2API orchestrator.

Components:
- QueryRouter: Protocol for routing strategies
- LLMToolRouter: Routes queries using LLM tool selection
- EscalatingLLMRouter: Routes with automatic model escalation
- RoutingCache: Tiered cache (Redis + pgvector semantic)
- AgentToolProvider: Wraps DomainAgent as ToolProvider

Example usage:

    from src.nl2api.routing import (
        LLMToolRouter,
        AgentToolProvider,
        InMemoryRoutingCache,
    )

    # Wrap agents as tool providers
    providers = [
        AgentToolProvider(datastream_agent),
        AgentToolProvider(estimates_agent),
    ]

    # Create router
    router = LLMToolRouter(
        llm=llm_provider,
        tool_providers=providers,
        cache=InMemoryRoutingCache(),
    )

    # Route a query
    result = await router.route("What is Apple's stock price?")
    print(f"Domain: {result.domain}, Confidence: {result.confidence}")
"""

from src.nl2api.routing.cache import InMemoryRoutingCache, RoutingCache
from src.nl2api.routing.escalating_router import (
    EscalatingLLMRouter,
    ModelTier,
    create_escalating_router,
)
from src.nl2api.routing.llm_router import LLMToolRouter
from src.nl2api.routing.protocols import (
    QueryRouter,
    RouterResult,
    RoutingToolDefinition,
    ToolExecutor,
    ToolProvider,
    deprecated_can_handle,
)
from src.nl2api.routing.providers import (
    AgentToolExecutor,
    AgentToolProvider,
    MCPToolExecutor,
    MCPToolProvider,
    create_providers_from_agents,
    create_dual_mode_providers,
)

__all__ = [
    # Protocols
    "QueryRouter",
    "RouterResult",
    "RoutingToolDefinition",
    "ToolProvider",
    "ToolExecutor",
    # Routers
    "LLMToolRouter",
    "EscalatingLLMRouter",
    "ModelTier",
    "create_escalating_router",
    # Cache
    "RoutingCache",
    "InMemoryRoutingCache",
    # Providers
    "AgentToolProvider",
    "AgentToolExecutor",
    "MCPToolProvider",
    "MCPToolExecutor",
    "create_providers_from_agents",
    "create_dual_mode_providers",
    # Utilities
    "deprecated_can_handle",
]
