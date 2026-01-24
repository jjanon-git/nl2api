# NL2API MCP Migration Plan

**Status:** Ready for Review - Implementation Pending

---

## Executive Summary

This document outlines the **dual-mode architecture** for supporting both existing local functionality AND MCP (Model Context Protocol) server integration in the NL2API system. The key principle is **additive, not replacement** - we keep all existing code working while adding MCP as an alternative source.

### Core Principle: Dual-Mode Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DUAL-MODE ARCHITECTURE                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  Config: tool_source = "local" | "mcp" | "auto"                            │
│          context_source = "local" | "mcp" | "auto"                         │
│                                                                             │
│  ┌─────────────────────┐         ┌─────────────────────┐                   │
│  │   LOCAL MODE        │         │   MCP MODE          │                   │
│  │   (Existing)        │         │   (New)             │                   │
│  ├─────────────────────┤         ├─────────────────────┤                   │
│  │ • Hardcoded tools   │         │ • tools/list        │                   │
│  │ • Local RAG         │         │ • resources/read    │                   │
│  │ • JSON generation   │         │ • tools/call (opt)  │                   │
│  └─────────────────────┘         └─────────────────────┘                   │
│            │                               │                                │
│            └───────────┬───────────────────┘                                │
│                        ▼                                                    │
│              ┌─────────────────────────────┐                                │
│              │  Unified Interface          │                                │
│              │  - get_tools()              │                                │
│              │  - get_field_codes()        │                                │
│              │  - process()                │                                │
│              └─────────────────────────────┘                                │
│                        │                                                    │
│                        ▼                                                    │
│              ┌─────────────────────────────┐                                │
│              │  LLM + Router (Unchanged)   │                                │
│              └─────────────────────────────┘                                │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Benefits of Dual-Mode

| Benefit | Description |
|---------|-------------|
| **Zero Breaking Changes** | All existing tests and functionality preserved |
| **Gradual Migration** | Enable MCP per-domain, not all-at-once |
| **Fallback Resilience** | If MCP unavailable, system continues with local |
| **A/B Testing** | Compare MCP vs local accuracy on same queries |
| **Environment Flexibility** | Local for dev, MCP for staging/prod |

---

## Architecture Overview

### Current vs MCP Sources

| Component | Local Mode (Current) | MCP Mode (New) |
|-----------|---------------------|----------------|
| **Tool Definitions** | `agent.get_tools()` hardcoded | `mcp.tools/list` dynamic |
| **Field Codes** | RAG pgvector search | `mcp.resources/read` |
| **Query Examples** | RAG pgvector search | `mcp.resources/read` |
| **Tool Execution** | Not implemented | `mcp.tools/call` optional |
| **Entity Resolution** | External API | Keep as-is (or MCP tool) |

### Mode Selection

```python
class NL2APIConfig:
    # Tool definition source
    tool_source: Literal["local", "mcp", "auto"] = "auto"

    # Context (field codes, examples) source
    context_source: Literal["local", "mcp", "auto"] = "auto"

    # Fallback behavior
    mcp_fallback_to_local: bool = True

    # Execution (only applies when using MCP)
    mcp_execution_enabled: bool = False
```

**Mode Behavior:**

| Mode | Behavior |
|------|----------|
| `"local"` | Use hardcoded tools / local RAG only |
| `"mcp"` | Use MCP servers only (fail if unavailable, unless fallback enabled) |
| `"auto"` | Try MCP first, fall back to local if unavailable |

---

## Component Design

### 1. MCP Client (`src/nl2api/mcp/client.py`)

```python
@dataclass(frozen=True)
class MCPToolDefinition:
    """Tool definition from MCP server."""
    name: str
    description: str
    input_schema: dict[str, Any]

@dataclass(frozen=True)
class MCPResource:
    """Resource content from MCP server."""
    uri: str
    mime_type: str
    content: str | bytes

@dataclass(frozen=True)
class MCPToolResult:
    """Result from MCP tool execution."""
    content: Any
    is_error: bool = False
    error_message: str | None = None


class MCPClient:
    """
    Client for MCP server communication.

    Supports tools/list, resources/read, and tools/call operations.
    """

    def __init__(
        self,
        server_uris: dict[str, str],  # domain -> URI mapping
        cache: MCPCache | None = None,
        timeout_seconds: float = 10.0,
    ):
        self._server_uris = server_uris
        self._cache = cache
        self._timeout = timeout_seconds

    async def list_tools(self, domain: str) -> list[MCPToolDefinition]:
        """Fetch available tools from MCP server."""
        ...

    async def read_resource(self, domain: str, resource_uri: str) -> MCPResource:
        """Read a resource from MCP server."""
        ...

    async def call_tool(
        self,
        domain: str,
        tool_name: str,
        arguments: dict[str, Any]
    ) -> MCPToolResult:
        """Execute a tool on MCP server."""
        ...
```

### 2. Source-Agnostic Base Agent (`src/nl2api/agents/base.py`)

```python
class BaseDomainAgent(ABC):
    def __init__(
        self,
        llm: LLMProvider,
        rag: RAGRetriever | None = None,
        mcp_client: MCPClient | None = None,
        tool_source: Literal["local", "mcp", "auto"] = "auto",
    ):
        self._llm = llm
        self._rag = rag
        self._mcp_client = mcp_client
        self._tool_source = tool_source
        self._tools_cache: list[LLMToolDefinition] | None = None

    async def get_tools(self) -> list[LLMToolDefinition]:
        """Get tools from configured source (cached)."""
        if self._tools_cache is not None:
            return self._tools_cache

        source = self._resolve_source(self._tool_source)

        if source == "mcp":
            try:
                tools = await self._get_tools_from_mcp()
            except MCPError as e:
                if self._should_fallback():
                    logger.warning(f"MCP failed, using local tools: {e}")
                    tools = self._get_tools_local()
                else:
                    raise
        else:
            tools = self._get_tools_local()

        self._tools_cache = tools
        return tools

    @abstractmethod
    def _get_tools_local(self) -> list[LLMToolDefinition]:
        """Return hardcoded tools (existing implementation)."""
        ...

    async def _get_tools_from_mcp(self) -> list[LLMToolDefinition]:
        """Fetch tools from MCP server."""
        mcp_tools = await self._mcp_client.list_tools(self.domain_name)
        return [self._convert_mcp_tool(t) for t in mcp_tools]
```

### 3. Dual-Mode Context Retriever

```python
class ContextRetriever:
    """
    Retrieves field codes and examples from configured source.

    Supports local RAG, MCP resources, or auto (MCP with fallback).
    """

    def __init__(
        self,
        rag: RAGRetriever | None = None,
        mcp_client: MCPClient | None = None,
        source: Literal["local", "mcp", "auto"] = "auto",
        fallback_enabled: bool = True,
    ):
        self._rag = rag
        self._mcp_client = mcp_client
        self._source = source
        self._fallback_enabled = fallback_enabled

    async def get_field_codes(
        self,
        query: str,
        domain: str,
        limit: int = 5
    ) -> list[dict]:
        """Get relevant field codes for the query."""
        source = self._resolve_source()

        if source == "mcp":
            try:
                return await self._get_field_codes_mcp(domain, query, limit)
            except MCPError:
                if self._fallback_enabled and self._rag:
                    return await self._get_field_codes_rag(query, domain, limit)
                raise

        return await self._get_field_codes_rag(query, domain, limit)

    async def _get_field_codes_mcp(
        self, domain: str, query: str, limit: int
    ) -> list[dict]:
        """Get field codes from MCP resources."""
        resource = await self._mcp_client.read_resource(
            domain=domain,
            resource_uri=f"fieldcodes://{domain}"
        )
        field_codes = self._parse_field_codes(resource.content)

        # Optional: rank by relevance to query
        if self._rag:
            return self._rank_by_similarity(field_codes, query, limit)
        return field_codes[:limit]

    async def _get_field_codes_rag(
        self, query: str, domain: str, limit: int
    ) -> list[dict]:
        """Get field codes from local RAG (existing implementation)."""
        results = await self._rag.retrieve_field_codes(
            query=query, domain=domain, limit=limit
        )
        return [{"code": r.field_code, "description": r.content} for r in results]
```

### 4. Updated Orchestrator

```python
class NL2APIOrchestrator:
    def __init__(
        self,
        llm: LLMProvider,
        agents: dict[str, DomainAgent],
        rag: RAGRetriever | None = None,
        mcp_client: MCPClient | None = None,
        context_retriever: ContextRetriever | None = None,
        # ... existing params ...
    ):
        # ... existing init ...

        # Create context retriever if not provided
        self._context_retriever = context_retriever or ContextRetriever(
            rag=rag,
            mcp_client=mcp_client,
            source=config.context_source,
            fallback_enabled=config.mcp_fallback_to_local,
        )

    async def process(self, query: str, ...) -> NL2APIResponse:
        # ... existing steps 1-4 ...

        # Step 5: Get context (now source-agnostic)
        field_codes = await self._context_retriever.get_field_codes(
            query=effective_query,
            domain=domain,
            limit=5,
        )
        examples = await self._context_retriever.get_examples(
            query=effective_query,
            domain=domain,
            limit=3,
        )

        # ... rest unchanged ...
```

---

## Configuration Schema

```python
class NL2APIConfig(BaseSettings):
    # ... existing config ...

    # ==========================================================================
    # DUAL-MODE CONFIGURATION
    # ==========================================================================

    # Tool definition source
    tool_source: Literal["local", "mcp", "auto"] = Field(
        default="auto",
        description=(
            "Source for tool definitions. "
            "'local': hardcoded in agents, "
            "'mcp': dynamic from MCP servers, "
            "'auto': try MCP first, fall back to local"
        ),
    )

    # Context (field codes, examples) source
    context_source: Literal["local", "mcp", "auto"] = Field(
        default="auto",
        description=(
            "Source for field codes and examples. "
            "'local': RAG (pgvector), "
            "'mcp': MCP resources, "
            "'auto': try MCP first, fall back to local"
        ),
    )

    # Fallback behavior
    mcp_fallback_to_local: bool = Field(
        default=True,
        description="Fall back to local sources if MCP unavailable",
    )

    # Tool execution (MCP only)
    mcp_execution_enabled: bool = Field(
        default=False,
        description="Execute tool calls via MCP tools/call",
    )

    # MCP server configuration
    mcp_server_uris: dict[str, str] = Field(
        default={},
        description="Domain to MCP server URI mapping",
        examples=[{
            "datastream": "mcp://datastream.lseg.com",
            "estimates": "mcp://estimates.lseg.com",
        }],
    )

    # MCP client settings
    mcp_timeout_seconds: float = Field(
        default=10.0,
        description="Timeout for MCP operations",
    )
    mcp_cache_tools_ttl_seconds: int = Field(
        default=3600,
        description="TTL for cached tool definitions (1 hour)",
    )
    mcp_cache_resources_ttl_seconds: int = Field(
        default=86400,
        description="TTL for cached resources (24 hours)",
    )
```

---

## Environment Configurations

### Development (Local Only)
```bash
NL2API_TOOL_SOURCE=local
NL2API_CONTEXT_SOURCE=local
# No MCP servers needed
```

### Staging (MCP with Fallback)
```bash
NL2API_TOOL_SOURCE=auto
NL2API_CONTEXT_SOURCE=auto
NL2API_MCP_FALLBACK_TO_LOCAL=true
NL2API_MCP_SERVER_URIS='{"datastream":"mcp://datastream.staging.lseg.com","estimates":"mcp://estimates.staging.lseg.com"}'
```

### Production (MCP Primary)
```bash
NL2API_TOOL_SOURCE=mcp
NL2API_CONTEXT_SOURCE=mcp
NL2API_MCP_FALLBACK_TO_LOCAL=true
NL2API_MCP_EXECUTION_ENABLED=true
NL2API_MCP_SERVER_URIS='{"datastream":"mcp://datastream.lseg.com","estimates":"mcp://estimates.lseg.com"}'
```

### A/B Testing (Compare Modes)
```bash
# Instance A: Local only
NL2API_TOOL_SOURCE=local

# Instance B: MCP only
NL2API_TOOL_SOURCE=mcp
NL2API_MCP_FALLBACK_TO_LOCAL=false
```

---

## Implementation Plan

### Phase 1: MCP Foundation (This PR)

**Create:**
| File | Purpose |
|------|---------|
| `src/nl2api/mcp/__init__.py` | Module exports |
| `src/nl2api/mcp/protocols.py` | MCP types (MCPToolDefinition, MCPResource, etc.) |
| `src/nl2api/mcp/client.py` | MCP client with caching |
| `src/nl2api/mcp/cache.py` | MCP-specific cache layer |
| `src/nl2api/mcp/context_retriever.py` | Dual-mode context retrieval |
| `tests/unit/nl2api/mcp/` | Unit tests |

**Modify:**
| File | Changes |
|------|---------|
| `src/nl2api/config.py` | Add dual-mode configuration |
| `src/nl2api/agents/base.py` | Source-agnostic `get_tools()` |
| `src/nl2api/orchestrator.py` | Use `ContextRetriever` |
| `src/nl2api/routing/providers.py` | Implement `MCPToolProvider` |

### Phase 2: Agent Integration (Future)

- Update all 5 agents to use source-agnostic base class
- Add MCP server URIs for each domain
- Integration testing with real/mock MCP servers

### Phase 3: Execution Support (Future)

- Add `tools/call` execution to orchestrator
- Update evaluation framework for execution testing
- Add execution results to `NL2APIResponse`

---

## Files to Create

| File | Purpose |
|------|---------|
| `src/nl2api/mcp/__init__.py` | Module exports |
| `src/nl2api/mcp/protocols.py` | Data types for MCP operations |
| `src/nl2api/mcp/client.py` | MCP client implementation |
| `src/nl2api/mcp/cache.py` | Tool/resource caching |
| `src/nl2api/mcp/context_retriever.py` | Dual-mode field code/example retrieval |
| `tests/unit/nl2api/mcp/test_client.py` | Client unit tests |
| `tests/unit/nl2api/mcp/test_context_retriever.py` | Retriever tests |

## Files to Modify

| File | Changes |
|------|---------|
| `src/nl2api/config.py` | Add dual-mode settings |
| `src/nl2api/agents/base.py` | Source-agnostic `get_tools()` |
| `src/nl2api/orchestrator.py` | Integrate `ContextRetriever` |
| `src/nl2api/routing/providers.py` | Implement `MCPToolProvider.list_tools()` |
| `src/nl2api/__init__.py` | Export MCP components |

---

## Testing Strategy

### Unit Tests
- Mock MCP client for all tests
- Test each mode: local, mcp, auto
- Test fallback behavior
- Test caching

### Integration Tests
- Test with mock MCP server
- Test mode switching
- Test fallback scenarios

### Backward Compatibility
- All existing 638 tests must pass
- Run with `tool_source=local` to verify no regressions

---

## Success Criteria

1. **All existing tests pass** with `tool_source=local`, `context_source=local`
2. **New MCP tests pass** with mock MCP client
3. **Fallback works** - MCP failure gracefully falls back to local
4. **Configuration works** - Can switch modes via environment variables
5. **No breaking changes** - Existing code paths unchanged

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| MCP server unavailable | Fallback to local (configurable) |
| MCP latency spikes | Aggressive caching, circuit breaker |
| Schema mismatches | Validation layer, version pinning |
| Regression in local mode | All tests run with local mode |

---

## Appendix: MCP Protocol Reference

### tools/list Response
```json
{
  "tools": [
    {
      "name": "get_data",
      "description": "Retrieve financial data",
      "inputSchema": {
        "type": "object",
        "properties": {
          "RICs": {"type": "array", "items": {"type": "string"}},
          "fields": {"type": "array", "items": {"type": "string"}}
        },
        "required": ["RICs", "fields"]
      }
    }
  ]
}
```

### resources/read Response
```json
{
  "contents": [
    {
      "uri": "fieldcodes://estimates",
      "mimeType": "application/json",
      "text": "[{\"code\": \"TR.EPSMean\", \"description\": \"Mean EPS estimate\"}]"
    }
  ]
}
```

### tools/call Request/Response
```json
// Request
{
  "name": "get_data",
  "arguments": {
    "RICs": ["AAPL.O"],
    "fields": ["TR.EPSMean"]
  }
}

// Response
{
  "content": [
    {
      "type": "text",
      "text": "{\"AAPL.O\": {\"TR.EPSMean\": 6.52}}"
    }
  ],
  "isError": false
}
```
