"""
Entity Resolution Service Configuration

Configuration settings using pydantic-settings for environment variable support.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class EntityServiceConfig(BaseSettings):
    """
    Configuration for the Entity Resolution Service.

    Reads from environment variables with ENTITY_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="ENTITY_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Server Identity
    server_name: str = Field(
        default="entity-resolution",
        description="Server name for identification",
    )
    server_version: str = Field(
        default="0.1.0",
        description="Server version",
    )

    # Transport Configuration
    transport: Literal["http", "mcp", "stdio"] = Field(
        default="http",
        description="Transport mode: http, mcp, or stdio",
    )
    host: str = Field(
        default="0.0.0.0",
        description="Host to bind for HTTP transport",
    )
    port: int = Field(
        default=8080,
        description="Port for HTTP transport",
    )
    require_client_id: bool = Field(
        default=False,
        description="Require X-Client-ID header",
    )

    # Database Configuration
    postgres_url: str = Field(
        default="postgresql://nl2api:nl2api@localhost:5432/nl2api",
        description="PostgreSQL connection URL",
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
        default=False,
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
        description="Seconds before trying to recover",
    )
    timeout_seconds: float = Field(
        default=5.0,
        description="Timeout for external API calls",
    )
    retry_max_attempts: int = Field(
        default=3,
        description="Maximum retry attempts",
    )

    # OpenFIGI Fallback
    openfigi_enabled: bool = Field(
        default=True,
        description="Enable OpenFIGI API as fallback",
    )
    openfigi_api_key: str | None = Field(
        default=None,
        description="Optional API key for higher rate limits",
    )

    # External API (optional)
    external_api_endpoint: str | None = Field(
        default=None,
        description="External API endpoint for resolution",
    )
    external_api_key: str | None = Field(
        default=None,
        description="API key for external service",
    )

    # Logging
    log_level: str = Field(
        default="INFO",
        description="Logging level",
    )


def load_config() -> EntityServiceConfig:
    """Load configuration from environment."""
    return EntityServiceConfig()
