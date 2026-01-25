# Entity Resolution Service Split: Implementation Plan

**Date:** 2026-01-24
**Status:** Approved - Ready for Implementation

---

## Summary

Extract the entity resolution component from nl2api into a standalone service at `src/services/entity_resolution/`. The service will be independently deployable while remaining in the same monorepo.

---

## Phase 1: Restructure and Extract (3-4 days)

### 1.1 Create Service Directory Structure

```
src/services/entity_resolution/
├── __init__.py
├── __main__.py                 # CLI entry point
├── core/
│   ├── __init__.py
│   ├── models.py               # ResolvedEntity, EntityType
│   ├── protocols.py            # EntityResolver protocol
│   ├── resolver.py             # ExternalEntityResolver
│   ├── extractor.py            # Extract entities from NL text
│   └── openfigi.py             # OpenFIGI API client
├── adapters/
│   ├── __init__.py
│   ├── database.py             # PostgreSQL lookups
│   └── cache.py                # Redis caching
├── transports/
│   ├── __init__.py
│   ├── http/
│   │   ├── __init__.py
│   │   ├── app.py              # FastAPI application
│   │   ├── routes.py           # REST endpoints
│   │   └── middleware.py       # Rate limiting, validation
│   ├── mcp/
│   │   ├── __init__.py
│   │   ├── server.py           # MCP server
│   │   ├── tools.py            # MCP tools
│   │   └── resources.py        # MCP resources
│   └── stdio.py                # Claude Desktop transport
├── config.py                   # Pydantic settings
├── observability.py            # Inline OTEL setup
├── resilience.py               # Inline CircuitBreaker, Retry
├── Dockerfile
├── pyproject.toml
└── README.md
```

### 1.2 Files to Move/Copy

| Source | Destination | Action |
|--------|-------------|--------|
| `src/nl2api/resolution/protocols.py` | `src/services/entity_resolution/core/protocols.py` | Copy |
| `src/nl2api/resolution/resolver.py` | `src/services/entity_resolution/core/resolver.py` | Move + refactor |
| `src/nl2api/resolution/openfigi.py` | `src/services/entity_resolution/core/openfigi.py` | Move |
| `src/mcp_servers/entity_resolution/server.py` | `src/services/entity_resolution/transports/mcp/server.py` | Move + refactor |
| `src/mcp_servers/entity_resolution/tools.py` | `src/services/entity_resolution/transports/mcp/tools.py` | Move |
| `src/mcp_servers/entity_resolution/resources.py` | `src/services/entity_resolution/transports/mcp/resources.py` | Move |
| `src/mcp_servers/entity_resolution/transports/sse.py` | `src/services/entity_resolution/transports/http/app.py` | Move + refactor |
| `src/mcp_servers/entity_resolution/config.py` | `src/services/entity_resolution/config.py` | Move |

### 1.3 Inline Shared Utilities

Copy minimal versions of these utilities into the service (no external deps):

| Utility | Source | Destination |
|---------|--------|-------------|
| CircuitBreaker | `src/evalkit/common/resilience/circuit_breaker.py` | `resilience.py` |
| RetryConfig | `src/evalkit/common/resilience/retry.py` | `resilience.py` |
| RedisCache (optional) | `src/evalkit/common/cache/redis_cache.py` | `adapters/cache.py` |
| OTEL setup | `src/evalkit/common/telemetry/` | `observability.py` |

### 1.4 Create Independent pyproject.toml

```toml
[project]
name = "entity-resolution-service"
version = "0.1.0"
description = "Entity resolution service for financial identifiers"
requires-python = ">=3.11"
dependencies = [
    "fastapi>=0.109.0",
    "uvicorn>=0.27.0",
    "httpx>=0.26.0",
    "asyncpg>=0.29.0",
    "pydantic>=2.5.0",
    "pydantic-settings>=2.1.0",
    "mcp>=1.0.0",
    "opentelemetry-api>=1.22.0",
    "opentelemetry-sdk>=1.22.0",
    "redis>=5.0.0",
]

[project.scripts]
entity-resolution = "entity_resolution.__main__:main"
```

### 1.5 Create Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

COPY src/services/entity_resolution/ ./entity_resolution/
COPY src/services/entity_resolution/pyproject.toml ./

RUN pip install --no-cache-dir .

EXPOSE 8080

CMD ["entity-resolution", "--host", "0.0.0.0", "--port", "8080"]
```

### 1.6 Update docker-compose.yml

Add entity resolution service:

```yaml
services:
  entity-resolution:
    build:
      context: .
      dockerfile: src/services/entity_resolution/Dockerfile
    ports:
      - "8085:8080"
    environment:
      - ENTITY_POSTGRES_URL=postgresql://nl2api:nl2api@postgres:5432/nl2api
      - ENTITY_REDIS_URL=redis://redis:6379/1
    depends_on:
      - postgres
      - redis
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3
```

---

## Phase 2: Create Client Adapter in nl2api (2-3 days)

### 2.1 Create HTTP Client

```python
# src/nl2api/resolution/client.py

class EntityResolutionClient:
    """HTTP client for Entity Resolution Service."""

    def __init__(
        self,
        base_url: str = "http://localhost:8085",
        timeout: float = 5.0,
    ):
        self._base_url = base_url
        self._timeout = timeout
        self._client: httpx.AsyncClient | None = None

    async def resolve(
        self,
        query: str,
        entity_types: list[str] | None = None,
    ) -> dict[str, ResolvedEntity]:
        """Extract and resolve entities from a query."""
        response = await self._client.post(
            f"{self._base_url}/api/extract",
            json={"text": query, "entity_types": entity_types},
            timeout=self._timeout,
        )
        response.raise_for_status()
        return self._parse_response(response.json())
```

### 2.2 Add Configuration

```python
# src/nl2api/config.py (additions)

class NL2APIConfig(BaseSettings):
    # Entity Resolution Settings
    entity_resolution_mode: Literal["local", "remote"] = Field(
        default="local",
        description="'local' uses in-process resolver, 'remote' calls entity service",
    )
    entity_resolution_service_url: str = Field(
        default="http://localhost:8085",
        description="URL of entity resolution service (when mode='remote')",
    )
```

### 2.3 Create Factory

```python
# src/nl2api/resolution/factory.py

def create_resolver(config: NL2APIConfig) -> EntityResolver:
    """Create entity resolver based on configuration."""
    if config.entity_resolution_mode == "remote":
        from .client import EntityResolutionClient
        return EntityResolutionClient(
            base_url=config.entity_resolution_service_url,
            timeout=config.entity_resolution_timeout_seconds,
        )

    # Local mode - existing implementation
    from .resolver import ExternalEntityResolver
    return ExternalEntityResolver(...)
```

### 2.4 Update Orchestrator

```python
# src/nl2api/orchestrator.py (change)

# Before:
self._entity_resolver = ExternalEntityResolver(...)

# After:
from .resolution.factory import create_resolver
self._entity_resolver = create_resolver(config)
```

---

## Phase 3: Deprecate Local Mode (1-2 days)

### 3.1 Add Deprecation Warning

```python
# src/nl2api/resolution/resolver.py

class ExternalEntityResolver:
    def __init__(self, ...):
        warnings.warn(
            "Local ExternalEntityResolver is deprecated. "
            "Set NL2API_ENTITY_RESOLUTION_MODE=remote to use the service.",
            DeprecationWarning,
            stacklevel=2,
        )
```

### 3.2 Change Default

```python
# src/nl2api/config.py
entity_resolution_mode: Literal["local", "remote"] = Field(
    default="remote",  # Changed from "local"
    ...
)
```

### 3.3 Update Documentation

- Add deployment guide for entity service
- Update README with new architecture
- Add migration guide for users

---

## Phase 4: Cleanup (Optional, 1 day)

### 4.1 Remove Deprecated Code

Delete from `src/nl2api/resolution/`:
- `resolver.py`
- `mock_resolver.py`
- `openfigi.py`

Keep:
- `protocols.py` (shared interface)
- `client.py` (service client)
- `factory.py` (simplified)

### 4.2 Remove Old MCP Server

Delete `src/mcp_servers/entity_resolution/` (moved to services)

---

## Testing Strategy

### Service Tests

| Test Type | Location | What |
|-----------|----------|------|
| Unit | `tests/unit/services/entity_resolution/` | Core logic |
| Integration | `tests/integration/services/entity_resolution/` | DB + cache |
| API | `tests/api/entity_resolution/` | HTTP contract |

### Client Tests in nl2api

| Test Type | Location | What |
|-----------|----------|------|
| Unit | `tests/unit/nl2api/resolution/` | Client with mocked HTTP |
| Integration | `tests/integration/nl2api/` | Full flow with service |

---

## Rollback Plan

If issues arise:
1. Set `NL2API_ENTITY_RESOLUTION_MODE=local` to revert to in-process
2. Local code remains functional through Phase 3
3. Phase 4 (removal) is optional and can be deferred indefinitely

---

## Success Criteria

- [ ] Entity service starts independently with `docker compose up entity-resolution`
- [ ] Health endpoint returns 200
- [ ] `/api/resolve` returns correct results
- [ ] nl2api works with `mode=remote`
- [ ] nl2api works with `mode=local` (backwards compatibility)
- [ ] All existing tests pass
- [ ] New service has >80% test coverage
