"""
Entity Resolution MCP Server Configuration

Configuration settings using pydantic-settings for environment variable support.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EntityServerConfig(BaseSettings):
    """
    Configuration for the Entity Resolution MCP Server.

    Reads from environment variables with ENTITY_MCP_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="ENTITY_MCP_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",  # Ignore extra fields from .env file
    )

    # Server Identity
    server_name: str = Field(
        default="entity-resolution",
        description="Server name for MCP protocol identification",
    )
    server_version: str = Field(
        default="1.0.0",
        description="Server version for MCP protocol",
    )

    # Transport Configuration
    transport: Literal["stdio", "sse"] = Field(
        default="sse",
        description="Transport mode: stdio for Claude Desktop, sse for HTTP",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind for SSE transport",
    )
    port: int = Field(
        default=8080,
        description="Port for SSE transport",
    )
    require_client_id: bool = Field(
        default=True,
        description="Require X-Client-ID header for SSE transport (for throttling/brownout)",
    )

    # Database Configuration
    postgres_url: str = Field(
        default="postgresql://nl2api:nl2api@localhost:5432/nl2api",
        description="PostgreSQL connection URL for entity database",
    )
    postgres_pool_min: int = Field(
        default=2,
        description="Minimum connections in pool",
    )
    postgres_pool_max: int = Field(
        default=10,
        description="Maximum connections in pool",
    )

    # Redis Cache Configuration
    redis_enabled: bool = Field(
        default=True,
        description="Enable Redis L2 cache",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )
    redis_cache_ttl_seconds: int = Field(
        default=86400,
        description="TTL for entity cache entries (24 hours)",
    )

    # Resilience Configuration
    circuit_failure_threshold: int = Field(
        default=5,
        description="Failures before opening circuit breaker",
    )
    circuit_recovery_seconds: float = Field(
        default=30.0,
        description="Seconds before trying to recover from open circuit",
    )
    timeout_seconds: float = Field(
        default=5.0,
        description="Timeout for external API calls",
    )
    retry_max_attempts: int = Field(
        default=3,
        description="Maximum retry attempts for transient failures",
    )

    # OpenFIGI Fallback
    openfigi_enabled: bool = Field(
        default=True,
        description="Enable OpenFIGI API as fallback for unknown entities",
    )
    openfigi_api_key: str | None = Field(
        default=None,
        description="Optional API key for higher OpenFIGI rate limits",
    )

    # Telemetry
    telemetry_enabled: bool = Field(
        default=False,
        description="Enable OpenTelemetry instrumentation",
    )
    telemetry_service_name: str = Field(
        default="entity-resolution-mcp",
        description="Service name for telemetry",
    )
    telemetry_otlp_endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP collector endpoint",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )


def load_config() -> EntityServerConfig:
    """Load configuration from environment."""
    return EntityServerConfig()
