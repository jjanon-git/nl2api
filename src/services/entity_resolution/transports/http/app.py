"""
FastAPI HTTP Transport for Entity Resolution Service

Provides REST endpoints for entity resolution:
- /health - Liveness probe
- /ready - Readiness probe (checks DB, cache)
- /api/resolve - Resolve single entity
- /api/resolve/batch - Resolve multiple entities
- /api/extract - Extract and resolve from query

NOTE: Do NOT add `from __future__ import annotations` to this file.
PEP 563 breaks FastAPI's runtime introspection for parameter sources.
"""

import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from pydantic import BaseModel, Field

from ...adapters.cache import RedisCache, check_redis_health, create_redis_client
from ...adapters.database import check_db_health, create_db_pool
from ...config import EntityServiceConfig
from ...core.resolver import EntityResolver

logger = logging.getLogger(__name__)


# Request/Response models
class ResolveRequest(BaseModel):
    """Request body for /api/resolve endpoint."""

    entity: str = Field(..., min_length=1, max_length=500)
    entity_type: str | None = Field(None, max_length=50)


class BatchResolveRequest(BaseModel):
    """Request body for /api/resolve/batch endpoint."""

    entities: list[str] = Field(..., min_length=1, max_length=100)


class ExtractRequest(BaseModel):
    """Request body for /api/extract endpoint."""

    query: str = Field(..., min_length=1, max_length=2000)


class ResolveResponse(BaseModel):
    """Response for entity resolution."""

    found: bool
    original: str
    identifier: str | None = None
    entity_type: str | None = None
    confidence: float | None = None
    alternatives: list[str] = []
    metadata: dict[str, str] = {}


# Global state
_resolver: EntityResolver | None = None
_db_pool: Any = None
_redis_client: Any = None
_config: EntityServiceConfig | None = None


def get_resolver() -> EntityResolver:
    """Get the global resolver instance."""
    if _resolver is None:
        raise RuntimeError("Resolver not initialized")
    return _resolver


def create_app(config: EntityServiceConfig | None = None) -> Any:
    """
    Create a FastAPI application for the entity resolution service.

    Args:
        config: Service configuration

    Returns:
        FastAPI application instance
    """
    try:
        from fastapi import FastAPI, HTTPException
        from fastapi.responses import JSONResponse
    except ImportError as e:
        raise ImportError(
            "FastAPI is required. Install with: pip install fastapi uvicorn"
        ) from e

    global _resolver, _db_pool, _redis_client, _config
    _config = config or EntityServiceConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Application lifespan handler."""
        global _resolver, _db_pool, _redis_client

        logger.info(f"Starting entity resolution service: {_config.server_name}")

        # Initialize database pool
        _db_pool = await create_db_pool(
            _config.postgres_url,
            min_size=_config.postgres_pool_min,
            max_size=_config.postgres_pool_max,
        )

        # Initialize Redis (optional)
        redis_cache = None
        if _config.redis_enabled:
            try:
                _redis_client = await create_redis_client(_config.redis_url)
                redis_cache = RedisCache(_redis_client)
            except Exception as e:
                logger.warning(f"Redis not available: {e}")

        # Initialize resolver
        _resolver = EntityResolver(
            db_pool=_db_pool,
            redis_client=redis_cache,
            api_key=_config.openfigi_api_key if _config.openfigi_enabled else None,
            use_cache=True,
            timeout_seconds=_config.timeout_seconds,
            circuit_failure_threshold=_config.circuit_failure_threshold,
            circuit_recovery_seconds=_config.circuit_recovery_seconds,
            retry_max_attempts=_config.retry_max_attempts,
            redis_cache_ttl_seconds=_config.redis_cache_ttl_seconds,
        )

        logger.info("Entity resolution service initialized")
        yield

        # Shutdown
        logger.info("Shutting down entity resolution service")
        if _db_pool:
            await _db_pool.close()
        if _redis_client:
            await _redis_client.close()
        logger.info("Entity resolution service shut down")

    app = FastAPI(
        title="Entity Resolution Service",
        description="Resolves entity names to financial identifiers (RICs, tickers)",
        version=_config.server_version,
        lifespan=lifespan,
    )

    @app.get("/health")
    async def liveness_check() -> JSONResponse:
        """Liveness probe - just checks if process is alive."""
        return JSONResponse(
            content={"status": "alive"},
            status_code=200,
        )

    @app.get("/ready")
    async def readiness_check() -> JSONResponse:
        """Readiness probe - checks dependencies."""
        checks: dict[str, Any] = {}
        all_ready = True

        # Check database
        if _db_pool:
            db_health = await check_db_health(_db_pool)
            checks["database"] = db_health
            if not db_health.get("connected"):
                all_ready = False
        else:
            checks["database"] = {"connected": False, "error": "pool not initialized"}
            all_ready = False

        # Check Redis (optional)
        if _redis_client:
            redis_health = await check_redis_health(_redis_client)
            checks["redis"] = redis_health
            # Redis failure is not critical
        else:
            checks["redis"] = {"enabled": False}

        status = "ready" if all_ready else "not_ready"
        return JSONResponse(
            content={"status": status, "checks": checks},
            status_code=200 if all_ready else 503,
        )

    @app.get("/")
    async def root() -> dict[str, Any]:
        """Root endpoint with service info."""
        return {
            "name": _config.server_name,
            "version": _config.server_version,
            "endpoints": {
                "health": "/health",
                "ready": "/ready",
                "resolve": "/api/resolve",
                "batch": "/api/resolve/batch",
                "extract": "/api/extract",
            },
        }

    @app.post("/api/resolve", response_model=ResolveResponse)
    async def resolve_entity_endpoint(body: ResolveRequest) -> ResolveResponse:
        """Resolve a single entity to its identifier."""
        resolver = get_resolver()

        try:
            result = await resolver.resolve_single(body.entity, body.entity_type)

            if result:
                return ResolveResponse(
                    found=True,
                    original=result.original,
                    identifier=result.identifier,
                    entity_type=result.entity_type,
                    confidence=result.confidence,
                    alternatives=list(result.alternatives),
                    metadata=dict(result.metadata),
                )
            else:
                return ResolveResponse(found=False, original=body.entity)

        except Exception as e:
            logger.exception(f"Error resolving entity: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/resolve/batch")
    async def resolve_batch_endpoint(body: BatchResolveRequest) -> JSONResponse:
        """Resolve multiple entities."""
        resolver = get_resolver()

        try:
            results = await resolver.resolve_batch(body.entities)
            return JSONResponse(
                content={
                    "resolved": [r.to_dict() for r in results],
                    "count": len(results),
                }
            )
        except Exception as e:
            logger.exception(f"Error resolving entities: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/api/extract")
    async def extract_and_resolve_endpoint(body: ExtractRequest) -> JSONResponse:
        """Extract and resolve entities from a natural language query."""
        resolver = get_resolver()

        try:
            resolved = await resolver.resolve(body.query)
            return JSONResponse(
                content={
                    "query": body.query,
                    "entities": resolved,
                    "count": len(resolved),
                }
            )
        except Exception as e:
            logger.exception(f"Error extracting entities: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/api/stats")
    async def get_stats() -> JSONResponse:
        """Get service statistics."""
        resolver = get_resolver()
        return JSONResponse(
            content={
                "circuit_breaker": resolver.circuit_breaker_stats,
                "cache": {
                    "l1_size": len(resolver._cache),
                    "redis_enabled": _redis_client is not None,
                },
            }
        )

    return app


async def run_http_server(config: EntityServiceConfig | None = None) -> None:
    """
    Run the HTTP server.

    Args:
        config: Service configuration
    """
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError(
            "Uvicorn is required. Install with: pip install uvicorn"
        ) from e

    _config = config or EntityServiceConfig()
    app = create_app(_config)

    logger.info(f"Starting HTTP server on {_config.host}:{_config.port}")

    server_config = uvicorn.Config(
        app,
        host=_config.host,
        port=_config.port,
        log_level=_config.log_level.lower(),
    )
    server = uvicorn.Server(server_config)
    await server.serve()
