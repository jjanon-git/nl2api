"""
NL2API Configuration

Configuration settings using pydantic-settings for environment variable support.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class NL2APIConfig(BaseSettings):
    """
    Configuration for the NL2API system.

    Reads from environment variables with NL2API_ prefix.
    """

    model_config = SettingsConfigDict(
        env_prefix="NL2API_",
        env_file=".env",
        env_file_encoding="utf-8",
    )

    # LLM Provider Settings
    llm_provider: Literal["claude", "openai", "azure_openai"] = Field(
        default="claude",
        description="LLM provider to use",
    )
    llm_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Model name for the LLM provider",
    )
    llm_temperature: float = Field(
        default=0.0,
        description="Temperature for LLM completions",
    )
    llm_max_tokens: int = Field(
        default=4096,
        description="Maximum tokens for LLM completions",
    )

    # API Keys
    anthropic_api_key: str | None = Field(
        default=None,
        description="Anthropic API key for Claude",
    )
    openai_api_key: str | None = Field(
        default=None,
        description="OpenAI API key",
    )
    azure_openai_endpoint: str | None = Field(
        default=None,
        description="Azure OpenAI endpoint URL",
    )
    azure_openai_api_version: str = Field(
        default="2024-02-15-preview",
        description="Azure OpenAI API version",
    )

    # RAG Settings
    rag_enabled: bool = Field(
        default=True,
        description="Whether to use RAG retrieval",
    )
    rag_vector_weight: float = Field(
        default=0.7,
        description="Weight for vector similarity in hybrid search",
    )
    rag_keyword_weight: float = Field(
        default=0.3,
        description="Weight for keyword matching in hybrid search",
    )
    rag_field_code_limit: int = Field(
        default=5,
        description="Maximum field codes to retrieve",
    )
    rag_example_limit: int = Field(
        default=3,
        description="Maximum examples to retrieve",
    )

    # Entity Resolution Settings
    entity_resolution_enabled: bool = Field(
        default=True,
        description="Whether to use entity resolution",
    )
    entity_resolution_api_endpoint: str | None = Field(
        default=None,
        description="External API endpoint for entity resolution",
    )
    entity_resolution_api_key: str | None = Field(
        default=None,
        description="API key for entity resolution service",
    )
    entity_resolution_cache_enabled: bool = Field(
        default=True,
        description="Whether to cache resolved entities",
    )
    entity_resolution_timeout_seconds: float = Field(
        default=5.0,
        description="Timeout for entity resolution API calls",
    )
    entity_resolution_circuit_failure_threshold: int = Field(
        default=5,
        description="Failures before opening circuit breaker",
    )
    entity_resolution_circuit_recovery_seconds: float = Field(
        default=30.0,
        description="Seconds before trying to recover from open circuit",
    )
    entity_resolution_retry_max_attempts: int = Field(
        default=3,
        description="Maximum retry attempts for entity resolution",
    )

    # Clarification Settings
    clarification_enabled: bool = Field(
        default=True,
        description="Whether to detect and request clarifications",
    )
    clarification_use_llm: bool = Field(
        default=False,
        description="Whether to use LLM for ambiguity detection",
    )

    # Multi-turn Settings
    multi_turn_enabled: bool = Field(
        default=False,
        description="Whether to enable multi-turn conversations",
    )
    multi_turn_history_limit: int = Field(
        default=5,
        description="Maximum conversation turns to keep in context",
    )
    multi_turn_session_ttl_minutes: int = Field(
        default=30,
        description="Session timeout in minutes",
    )

    # Database Settings (shared with storage)
    postgres_url: str = Field(
        default="postgresql://nl2api:nl2api@localhost:5432/nl2api",
        description="PostgreSQL connection URL",
    )

    # Embedding Settings
    embedding_provider: Literal["openai"] = Field(
        default="openai",
        description="Embedding provider for RAG",
    )
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Embedding model name",
    )
    embedding_dimension: int = Field(
        default=1536,
        description="Embedding dimension",
    )
    embedding_max_concurrent: int = Field(
        default=5,
        description="Maximum concurrent embedding requests",
    )
    embedding_requests_per_minute: int = Field(
        default=3000,
        description="Rate limit for embedding requests per minute",
    )

    # RAG Indexing Settings
    rag_indexing_batch_size: int = Field(
        default=100,
        description="Batch size for RAG indexing operations",
    )
    rag_indexing_use_bulk_insert: bool = Field(
        default=True,
        description="Use COPY protocol for bulk inserts",
    )
    rag_indexing_checkpoint_enabled: bool = Field(
        default=True,
        description="Enable checkpointing for large indexing jobs",
    )

    # Redis Cache Settings
    redis_enabled: bool = Field(
        default=False,
        description="Enable Redis caching (requires redis package)",
    )
    redis_url: str = Field(
        default="redis://localhost:6379/0",
        description="Redis connection URL",
    )
    redis_default_ttl_seconds: int = Field(
        default=3600,
        description="Default TTL for cached values (1 hour)",
    )
    redis_key_prefix: str = Field(
        default="nl2api:",
        description="Prefix for all Redis keys",
    )
    redis_max_connections: int = Field(
        default=10,
        description="Maximum Redis connections",
    )
    redis_entity_cache_ttl_seconds: int = Field(
        default=86400,
        description="TTL for entity resolution cache (24 hours)",
    )
    redis_rag_cache_ttl_seconds: int = Field(
        default=3600,
        description="TTL for RAG query cache (1 hour)",
    )

    # Query Routing Settings
    routing_enabled: bool = Field(
        default=True,
        description="Enable FM-first routing (uses LLM for domain classification)",
    )
    routing_model: str = Field(
        default="claude-3-5-haiku-20241022",
        description="Model for routing. Haiku recommended (94.1% accuracy at 1/10th cost vs Sonnet)",
    )
    routing_cache_enabled: bool = Field(
        default=True,
        description="Enable routing decision caching",
    )
    routing_cache_ttl_seconds: int = Field(
        default=3600,
        description="TTL for cached routing decisions (1 hour)",
    )
    routing_semantic_cache_enabled: bool = Field(
        default=True,
        description="Enable semantic similarity cache for routing (requires pgvector)",
    )
    routing_semantic_threshold: float = Field(
        default=0.92,
        description="Minimum similarity for semantic cache hits",
    )
    routing_confidence_threshold: float = Field(
        default=0.5,
        description="Confidence threshold below which clarification is requested",
    )
    routing_escalation_enabled: bool = Field(
        default=False,
        description="Enable model escalation for complex queries",
    )
    routing_escalation_threshold: float = Field(
        default=0.7,
        description="Confidence threshold for escalating to a more capable model",
    )
    routing_tier1_model: str = Field(
        default="claude-3-haiku-20240307",
        description="Tier 1 model for routing (fast/cheap)",
    )
    routing_tier2_model: str = Field(
        default="claude-sonnet-4-20250514",
        description="Tier 2 model for routing (balanced)",
    )
    routing_tier3_model: str | None = Field(
        default=None,
        description="Tier 3 model for routing (most capable). None = skip tier 3",
    )

    # MCP (Model Context Protocol) Settings - Dual Mode Support
    mcp_enabled: bool = Field(
        default=False,
        description="Enable MCP server integration for tools and context",
    )
    mcp_mode: Literal["local", "mcp", "hybrid"] = Field(
        default="local",
        description=(
            "Context retrieval mode: "
            "'local' = existing RAG/agents only, "
            "'mcp' = MCP servers only, "
            "'hybrid' = prefer MCP with local fallback"
        ),
    )
    mcp_servers: str = Field(
        default="",
        description=(
            "Comma-separated list of MCP server URIs "
            "(e.g., 'mcp://datastream.lseg.com,mcp://estimates.lseg.com')"
        ),
    )
    mcp_cache_enabled: bool = Field(
        default=True,
        description="Enable caching for MCP tool/resource lookups",
    )
    mcp_cache_ttl_seconds: int = Field(
        default=300,
        description="TTL for cached MCP data (5 minutes)",
    )
    mcp_default_timeout_seconds: int = Field(
        default=30,
        description="Default timeout for MCP server requests",
    )
    mcp_max_retries: int = Field(
        default=3,
        description="Maximum retry attempts for MCP requests",
    )
    mcp_fallback_to_local: bool = Field(
        default=True,
        description="Fall back to local agents if MCP servers fail (hybrid mode)",
    )
    mcp_datastream_uri: str | None = Field(
        default=None,
        description="MCP server URI for Datastream API",
    )
    mcp_estimates_uri: str | None = Field(
        default=None,
        description="MCP server URI for Estimates API",
    )
    mcp_fundamentals_uri: str | None = Field(
        default=None,
        description="MCP server URI for Fundamentals API",
    )
    mcp_officers_uri: str | None = Field(
        default=None,
        description="MCP server URI for Officers API",
    )
    mcp_screening_uri: str | None = Field(
        default=None,
        description="MCP server URI for Screening API",
    )
    mcp_entity_resolution_uri: str | None = Field(
        default=None,
        description="MCP server URI for Entity Resolution (e.g., 'http://localhost:8080')",
    )
    mcp_entity_resolution_enabled: bool = Field(
        default=False,
        description=(
            "Use MCP server for entity resolution instead of local resolver. "
            "When enabled, calls mcp_entity_resolution_uri for entity lookups."
        ),
    )

    # Telemetry Settings (OpenTelemetry)
    telemetry_enabled: bool = Field(
        default=False,
        description="Enable OpenTelemetry instrumentation",
    )
    telemetry_service_name: str = Field(
        default="nl2api",
        description="Service name for telemetry identification",
    )
    telemetry_otlp_endpoint: str = Field(
        default="http://localhost:4317",
        description="OTLP collector endpoint (gRPC)",
    )
    telemetry_tracing_enabled: bool = Field(
        default=True,
        description="Enable distributed tracing",
    )
    telemetry_metrics_enabled: bool = Field(
        default=True,
        description="Enable metrics export",
    )
    telemetry_export_interval_ms: int = Field(
        default=10000,
        description="Metrics export interval in milliseconds",
    )

    # Metrics Emission Settings
    metrics_log_enabled: bool = Field(
        default=True,
        description="Enable logging emitter for metrics",
    )
    metrics_file_path: str | None = Field(
        default=None,
        description="Path for JSONL metrics file (None = disabled)",
    )
    metrics_otel_enabled: bool = Field(
        default=False,
        description="Enable OpenTelemetry emitter for metrics",
    )

    def get_mcp_server_uris(self) -> list[str]:
        """Get list of configured MCP server URIs."""
        uris = []

        # From comma-separated list
        if self.mcp_servers:
            uris.extend(
                uri.strip()
                for uri in self.mcp_servers.split(",")
                if uri.strip()
            )

        # From individual domain configs
        for uri in [
            self.mcp_datastream_uri,
            self.mcp_estimates_uri,
            self.mcp_fundamentals_uri,
            self.mcp_officers_uri,
            self.mcp_screening_uri,
        ]:
            if uri and uri not in uris:
                uris.append(uri)

        return uris

    def get_llm_api_key(self) -> str:
        """Get the appropriate API key for the configured LLM provider."""
        if self.llm_provider == "claude":
            if not self.anthropic_api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")
            return self.anthropic_api_key
        elif self.llm_provider in ("openai", "azure_openai"):
            if not self.openai_api_key:
                raise ValueError("OPENAI_API_KEY not set")
            return self.openai_api_key
        else:
            raise ValueError(f"Unknown LLM provider: {self.llm_provider}")


def load_config() -> NL2APIConfig:
    """Load configuration from environment."""
    return NL2APIConfig()
