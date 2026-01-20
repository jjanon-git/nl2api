"""
Storage Configuration

Configuration-driven backend selection with environment variable support.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class StorageConfig(BaseSettings):
    """
    Configuration for storage backends.

    Supports environment variables with EVAL_ prefix:
    - EVAL_BACKEND: "postgres", "azure", or "memory"
    - EVAL_POSTGRES_URL: PostgreSQL connection string
    - EVAL_AZURE_SEARCH_ENDPOINT: Azure AI Search endpoint
    - EVAL_AZURE_SEARCH_KEY: Azure AI Search API key
    - EVAL_AZURE_TABLE_CONNECTION_STRING: Azure Table Storage connection string
    """

    model_config = SettingsConfigDict(
        env_prefix="EVAL_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Backend selection
    backend: Literal["postgres", "azure", "memory"] = Field(
        default="postgres",
        description="Storage backend: postgres (local), azure (prod), or memory (tests)",
    )

    # PostgreSQL settings
    postgres_url: str = Field(
        default="postgresql://eval:eval@localhost:5432/evalplatform",
        description="PostgreSQL connection URL (asyncpg format)",
    )
    postgres_pool_min: int = Field(
        default=2,
        ge=1,
        description="Minimum connection pool size",
    )
    postgres_pool_max: int = Field(
        default=10,
        ge=1,
        description="Maximum connection pool size",
    )

    # Azure AI Search settings (for future use)
    azure_search_endpoint: str | None = Field(
        default=None,
        description="Azure AI Search endpoint URL",
    )
    azure_search_key: str | None = Field(
        default=None,
        description="Azure AI Search API key",
    )
    azure_search_index: str = Field(
        default="test-cases",
        description="Azure AI Search index name for test cases",
    )

    # Azure Table Storage settings (for future use)
    azure_table_connection_string: str | None = Field(
        default=None,
        description="Azure Table Storage connection string",
    )
    azure_table_scorecards: str = Field(
        default="scorecards",
        description="Azure Table Storage table name for scorecards",
    )
