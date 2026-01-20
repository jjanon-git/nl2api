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
        default="postgresql://postgres:postgres@localhost:5432/evalplatform",
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
