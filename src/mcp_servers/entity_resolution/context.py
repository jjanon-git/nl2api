"""
Client Context for MCP Server Observability

Provides client identification and request correlation for tracing.
Every request should carry context identifying the client and request.
"""

import uuid
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Optional

# Context variable for current client context (async-safe)
_current_context: ContextVar[Optional["ClientContext"]] = ContextVar(
    "mcp_client_context", default=None
)


@dataclass
class ClientContext:
    """
    Client identification for observability.

    Attributes:
        session_id: Unique ID for this client session/connection
        client_id: Client-provided identifier (from X-Client-ID header or MCP client info)
        client_name: Human-readable client name (from User-Agent or X-Client-Name)
        transport: Transport type (sse, stdio)
        request_id: Current request correlation ID
    """

    session_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    client_id: Optional[str] = None
    client_name: Optional[str] = None
    transport: str = "unknown"
    request_id: Optional[str] = None

    def with_request(self, request_id: str) -> "ClientContext":
        """Create a new context with a specific request ID."""
        return ClientContext(
            session_id=self.session_id,
            client_id=self.client_id,
            client_name=self.client_name,
            transport=self.transport,
            request_id=request_id,
        )

    def to_span_attributes(self) -> dict[str, str]:
        """Convert to OTEL span attributes."""
        attrs = {
            "client.session_id": self.session_id,
            "client.transport": self.transport,
        }
        if self.client_id:
            attrs["client.id"] = self.client_id
        if self.client_name:
            attrs["client.name"] = self.client_name
        if self.request_id:
            attrs["request.correlation_id"] = self.request_id
        return attrs


def get_client_context() -> Optional[ClientContext]:
    """Get the current client context."""
    return _current_context.get()


def set_client_context(ctx: ClientContext) -> None:
    """Set the current client context."""
    _current_context.set(ctx)


def clear_client_context() -> None:
    """Clear the current client context."""
    _current_context.set(None)


def create_sse_context(
    client_id: Optional[str] = None,
    client_name: Optional[str] = None,
    user_agent: Optional[str] = None,
) -> ClientContext:
    """
    Create client context for SSE/HTTP transport.

    Args:
        client_id: From X-Client-ID header
        client_name: From X-Client-Name header
        user_agent: From User-Agent header (fallback for client_name)
    """
    name = client_name or _parse_user_agent(user_agent)
    return ClientContext(
        client_id=client_id,
        client_name=name,
        transport="sse",
    )


def create_stdio_context() -> ClientContext:
    """
    Create client context for stdio transport.

    stdio is single-client, so we generate one session ID for the process.
    """
    return ClientContext(
        client_name="stdio-client",
        transport="stdio",
    )


def _parse_user_agent(user_agent: Optional[str]) -> Optional[str]:
    """Extract meaningful client name from User-Agent."""
    if not user_agent:
        return None

    # Common patterns
    ua_lower = user_agent.lower()
    if "claude" in ua_lower:
        return "claude-desktop"
    if "curl" in ua_lower:
        return "curl"
    if "python" in ua_lower:
        return "python-client"
    if "node" in ua_lower or "axios" in ua_lower:
        return "node-client"

    # Return first part of UA (typically the client name)
    return user_agent.split("/")[0][:50] if user_agent else None
