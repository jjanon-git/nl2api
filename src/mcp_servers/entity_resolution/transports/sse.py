"""
HTTP/SSE Transport for Entity Resolution MCP Server

Implements the HTTP transport with Server-Sent Events (SSE) for production
deployments. Provides a FastAPI application with:
- /health endpoint for load balancer health checks
- /mcp endpoint for JSON-RPC requests
- /sse endpoint for Server-Sent Events streaming

Security features:
- Rate limiting (in-memory or Redis-backed)
- Input validation with size limits
- Request body size limits

NOTE: Do NOT add `from __future__ import annotations` to this file.
PEP 563 stringifies type annotations, which breaks FastAPI's runtime
introspection for determining parameter sources (body vs query vs path).
Symptoms: 422 errors for valid JSON, "missing query parameter" errors.
Fix: Use Pydantic models for request bodies (see ResolveRequest, etc.).
"""

import asyncio
import json
import logging
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

from pydantic import BaseModel, Field

from src.mcp_servers.entity_resolution.config import EntityServerConfig
from src.mcp_servers.entity_resolution.context import (
    create_sse_context,
    set_client_context,
)
from src.mcp_servers.entity_resolution.middleware import (
    RateLimitConfig,
    create_rate_limiter,
    create_security_middleware,
)
from src.mcp_servers.entity_resolution.server import EntityResolutionMCPServer

logger = logging.getLogger(__name__)

# Default security configuration
DEFAULT_RATE_LIMIT_CONFIG = RateLimitConfig(
    requests_per_window=100,  # 100 requests
    window_seconds=60,  # per minute
    max_body_size=1_048_576,  # 1MB
    max_entity_length=500,
    max_batch_size=100,
    max_query_length=2000,
)


# Pydantic models for REST API requests with validation
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


# Global server instance for the FastAPI app
_server: EntityResolutionMCPServer | None = None


def get_server() -> EntityResolutionMCPServer:
    """Get the global server instance."""
    if _server is None:
        raise RuntimeError("Server not initialized")
    return _server


def create_app(
    config: EntityServerConfig | None = None,
) -> Any:
    """
    Create a FastAPI application for the MCP server.

    Args:
        config: Server configuration

    Returns:
        FastAPI application instance
    """
    # Import FastAPI here to make it an optional dependency
    try:
        from fastapi import FastAPI, HTTPException, Request
        from fastapi.responses import JSONResponse, Response, StreamingResponse
    except ImportError as e:
        raise ImportError(
            "FastAPI is required for SSE transport. Install with: pip install 'nl2api[mcp-server]'"
        ) from e

    global _server
    _config = config or EntityServerConfig()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        """Application lifespan handler."""
        global _server

        # Startup
        logger.info(f"Starting MCP server: {_config.server_name}")
        _server = EntityResolutionMCPServer(_config)
        await _server.initialize()
        logger.info("MCP server initialized")

        yield

        # Shutdown
        logger.info("Shutting down MCP server")
        if _server:
            await _server.shutdown()
        logger.info("MCP server shut down")

    app = FastAPI(
        title="NL2API MCP Server",
        description="MCP server for natural language queries to financial APIs",
        version=_config.server_version,
        lifespan=lifespan,
    )

    # Paths exempt from client_id requirement and rate limiting
    EXEMPT_PATHS = {"/health", "/ready", "/", "/docs", "/openapi.json"}

    # Initialize rate limiter (Redis-backed if available, otherwise in-memory)
    rate_limit_config = DEFAULT_RATE_LIMIT_CONFIG
    rate_limiter = create_rate_limiter(rate_limit_config, redis_client=None)

    # Add security middleware (rate limiting, request size limits)
    security_middleware = create_security_middleware(rate_limit_config, rate_limiter, EXEMPT_PATHS)
    app.middleware("http")(security_middleware)

    @app.middleware("http")
    async def client_context_middleware(request: Request, call_next: Any) -> Any:
        """
        Extract client identification from headers and set context.

        Supported headers:
        - X-Client-ID: Unique client identifier (REQUIRED unless path is exempt)
        - X-Client-Name: Human-readable client name
        - User-Agent: Fallback for client name detection

        Returns 401 if X-Client-ID is missing and require_client_id is True.
        """
        client_id = request.headers.get("X-Client-ID")
        client_name = request.headers.get("X-Client-Name")
        user_agent = request.headers.get("User-Agent")

        # Enforce client_id for non-exempt paths
        if _config.require_client_id and request.url.path not in EXEMPT_PATHS:
            if not client_id or not client_id.strip():
                return JSONResponse(
                    status_code=401,
                    content={
                        "error": "missing_client_id",
                        "message": "X-Client-ID header is required",
                    },
                )

        ctx = create_sse_context(
            client_id=client_id,
            client_name=client_name,
            user_agent=user_agent,
        )
        set_client_context(ctx)

        response = await call_next(request)
        return response

    @app.get("/health")
    async def liveness_check() -> JSONResponse:
        """
        Liveness probe endpoint.

        Kubernetes uses this to determine if the container should be restarted.
        Returns 200 if the process is alive and responsive.

        This is intentionally simple - just checks if the process can respond.
        Use /ready for dependency checks.
        """
        return JSONResponse(
            content={
                "status": "alive",
                "checks": {
                    "process": "ok",
                },
            },
            status_code=200,
        )

    @app.get("/ready")
    async def readiness_check() -> JSONResponse:
        """
        Readiness probe endpoint.

        Kubernetes uses this to determine if the pod should receive traffic.
        Returns 200 only when all dependencies are ready.

        Checks:
        - Server initialized
        - Database connection (if configured)
        - Redis connection (if configured)
        """
        checks: dict[str, str] = {}
        all_ready = True

        # Check 1: Server initialized
        try:
            server = get_server()
            checks["server"] = "ok"
        except RuntimeError:
            checks["server"] = "not_initialized"
            all_ready = False
            return JSONResponse(
                content={
                    "status": "not_ready",
                    "checks": checks,
                },
                status_code=503,
            )

        # Check 2: Database connection
        try:
            health = await server.read_resource("entity://health")
            db_status = health.get("database", {})
            if isinstance(db_status, dict) and db_status.get("connected"):
                checks["database"] = "ok"
            elif health.get("status") == "healthy":
                # Older health format - assume DB is OK if overall healthy
                checks["database"] = "ok"
            else:
                checks["database"] = "not_connected"
                all_ready = False
        except Exception as e:
            checks["database"] = f"error: {str(e)[:50]}"
            all_ready = False

        # Check 3: Redis connection (if available)
        try:
            stats = await server.read_resource("entity://stats")
            cache_info = stats.get("cache", {})
            if cache_info.get("enabled"):
                if cache_info.get("connected", True):  # Default to True if not specified
                    checks["redis"] = "ok"
                else:
                    checks["redis"] = "not_connected"
                    all_ready = False
            else:
                checks["redis"] = "disabled"
        except Exception:
            checks["redis"] = "unknown"

        status = "ready" if all_ready else "not_ready"
        status_code = 200 if all_ready else 503

        return JSONResponse(
            content={
                "status": status,
                "checks": checks,
            },
            status_code=status_code,
        )

    @app.get("/")
    async def root() -> dict[str, Any]:
        """Root endpoint with server info."""
        server = get_server()
        return {
            "name": server.server_info["name"],
            "version": server.server_info["version"],
            "protocol": "MCP",
            "protocolVersion": server.server_info["protocolVersion"],
            "endpoints": {
                "health": "/health",
                "ready": "/ready",
                "mcp": "/mcp",
                "sse": "/sse",
            },
        }

    @app.post("/mcp")
    async def mcp_endpoint(request: Request) -> JSONResponse:
        """
        MCP JSON-RPC endpoint.

        Accepts JSON-RPC requests and returns responses.
        """
        server = get_server()

        try:
            body = await request.json()
        except json.JSONDecodeError as e:
            return JSONResponse(
                content={
                    "jsonrpc": "2.0",
                    "id": None,
                    "error": {
                        "code": -32700,
                        "message": f"Parse error: {e}",
                    },
                },
                status_code=400,
            )

        # Handle batch requests
        if isinstance(body, list):
            responses = []
            for message in body:
                response = await server.handle_message(message)
                # Filter out None responses (notifications don't expect responses)
                if response is not None:
                    responses.append(response)
            return JSONResponse(content=responses)

        # Handle single request
        response = await server.handle_message(body)
        # Notifications return None - respond with 204 No Content
        if response is None:
            return Response(status_code=204)
        return JSONResponse(content=response)

    @app.get("/sse")
    async def sse_endpoint(request: Request) -> StreamingResponse:
        """
        Server-Sent Events endpoint for streaming MCP communication.

        Clients can connect here to receive server-initiated messages.
        """

        async def event_stream() -> AsyncGenerator[str, None]:
            """Generate SSE events."""
            server = get_server()

            # Send initial connection event
            yield f"event: connected\ndata: {json.dumps(server.server_info)}\n\n"

            # Keep connection alive with periodic pings
            try:
                while True:
                    await asyncio.sleep(30)
                    yield "event: ping\ndata: {}\n\n"
            except asyncio.CancelledError:
                logger.debug("SSE connection closed")
                return

        return StreamingResponse(
            event_stream(),
            media_type="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",  # Disable nginx buffering
            },
        )

    # Tool-specific REST endpoints for convenience
    @app.post("/api/resolve")
    async def resolve_entity_endpoint(body: ResolveRequest) -> JSONResponse:
        """
        REST endpoint to resolve a single entity.

        Convenience wrapper around the resolve_entity MCP tool.
        """
        server = get_server()

        try:
            result = await server.call_tool(
                "resolve_entity",
                {"entity": body.entity, "entity_type": body.entity_type},
            )
            return JSONResponse(content=result)
        except ValueError as e:
            logger.warning(f"Invalid resolve request: {e}")
            raise HTTPException(status_code=400, detail="Invalid request")
        except Exception as e:
            logger.exception(f"Error resolving entity: {e}")
            raise HTTPException(status_code=500, detail="An internal error occurred")

    @app.post("/api/resolve/batch")
    async def resolve_batch_endpoint(body: BatchResolveRequest) -> JSONResponse:
        """
        REST endpoint to resolve multiple entities.

        Convenience wrapper around the resolve_entities_batch MCP tool.
        """
        server = get_server()

        try:
            result = await server.call_tool(
                "resolve_entities_batch",
                {"entities": body.entities},
            )
            return JSONResponse(content=result)
        except ValueError as e:
            logger.warning(f"Invalid batch resolve request: {e}")
            raise HTTPException(status_code=400, detail="Invalid request")
        except Exception as e:
            logger.exception(f"Error resolving entities: {e}")
            raise HTTPException(status_code=500, detail="An internal error occurred")

    @app.post("/api/extract")
    async def extract_and_resolve_endpoint(body: ExtractRequest) -> JSONResponse:
        """
        REST endpoint to extract and resolve entities from a query.

        Convenience wrapper around the extract_and_resolve MCP tool.
        """
        server = get_server()

        try:
            result = await server.call_tool(
                "extract_and_resolve",
                {"query": body.query},
            )
            return JSONResponse(content=result)
        except ValueError as e:
            logger.warning(f"Invalid extract request: {e}")
            raise HTTPException(status_code=400, detail="Invalid request")
        except Exception as e:
            logger.exception(f"Error extracting entities: {e}")
            raise HTTPException(status_code=500, detail="An internal error occurred")

    return app


async def run_sse_server(
    config: EntityServerConfig | None = None,
) -> None:
    """
    Run the MCP server using HTTP/SSE transport.

    Args:
        config: Server configuration
    """
    try:
        import uvicorn
    except ImportError as e:
        raise ImportError(
            "Uvicorn is required for SSE transport. Install with: pip install 'nl2api[mcp-server]'"
        ) from e

    _config = config or EntityServerConfig()
    app = create_app(_config)

    logger.info(f"Starting SSE transport on {_config.host}:{_config.port}")

    server_config = uvicorn.Config(
        app,
        host=_config.host,
        port=_config.port,
        log_level=_config.log_level.lower(),
    )
    server = uvicorn.Server(server_config)
    await server.serve()
