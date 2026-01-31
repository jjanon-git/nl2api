"""
RAG UI Configuration

Settings for the RAG question interface.
"""

from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class RAGUIConfig(BaseSettings):
    """Configuration for RAG UI."""

    # Database (used for PostgreSQL backend and evaluation storage)
    database_url: str = "postgresql://nl2api:nl2api@localhost:5432/nl2api"

    # Backend selection: "postgres" (local pgvector) or "azure" (Azure AI Search)
    rag_backend: Literal["postgres", "azure"] = "postgres"

    # Azure AI Search settings (when rag_backend="azure")
    azure_search_endpoint: str | None = None  # e.g., https://your-search.search.windows.net
    azure_search_api_key: str | None = None
    azure_search_index: str = "rag-documents"
    # Field mappings for Azure index schema
    azure_search_vector_field: str = "embedding"
    azure_search_content_field: str = "content"
    azure_search_id_field: str = "id"
    azure_search_document_type_field: str = "document_type"
    azure_search_domain_field: str = "domain"
    azure_search_metadata_field: str = "metadata"
    # Optional Azure search features
    azure_search_hybrid: bool = True  # Use hybrid search (vector + full-text)
    azure_search_semantic_config: str | None = None  # Optional semantic configuration

    # Retriever settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_provider: str = "local"  # "local" or "openai"
    default_top_k: int = 5
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    retrieval_threshold: float = 0.1  # Lower threshold needed for SEC filing retrieval

    # Small-to-big retrieval settings (PostgreSQL backend only)
    use_small_to_big: bool = False  # Enable small-to-big (child search, parent return)
    small_to_big_child_limit: int = 30  # Number of child chunks to search

    # LLM settings
    llm_model: str = "claude-3-5-haiku-latest"
    max_context_chunks: int = 8
    max_answer_tokens: int = 1000

    # API keys (loaded from environment)
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Entity resolution settings
    entity_resolution_endpoint: str | None = None  # None = use local resolver
    entity_resolution_timeout: float = 5.0

    model_config = SettingsConfigDict(
        env_prefix="RAG_UI_",
        env_file=".env",
        extra="ignore",
    )

    @property
    def embedding_dimension(self) -> int:
        """Return embedding dimension based on provider."""
        if self.embedding_provider == "openai":
            return 1536
        # Local embedder (all-MiniLM-L6-v2) uses 384 dimensions
        return 384
