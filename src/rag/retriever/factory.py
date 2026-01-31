"""
Retriever Factory

Factory function to create the appropriate RAG retriever based on configuration.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

from src.rag.retriever.embedders import LocalEmbedder, OpenAIEmbedder
from src.rag.retriever.protocols import RAGRetriever

if TYPE_CHECKING:
    import asyncpg

    from src.rag.ui.config import RAGUIConfig

logger = logging.getLogger(__name__)


def create_retriever(
    config: RAGUIConfig,
    pool: asyncpg.Pool | None = None,
) -> RAGRetriever:
    """
    Create appropriate retriever based on configuration.

    Args:
        config: RAG UI configuration
        pool: PostgreSQL connection pool (required for postgres backend)

    Returns:
        RAGRetriever instance (either HybridRAGRetriever or AzureAISearchRetriever)

    Raises:
        ValueError: If required configuration is missing
    """
    # Create embedder first (needed for both backends)
    if config.embedding_provider == "openai":
        if not config.openai_api_key:
            raise ValueError("OpenAI API key required for openai embedding provider")
        embedder = OpenAIEmbedder(api_key=config.openai_api_key)
    else:
        embedder = LocalEmbedder(model_name=config.embedding_model)

    if config.rag_backend == "azure":
        # Azure AI Search backend
        if not config.azure_search_endpoint:
            raise ValueError(
                "Azure AI Search endpoint required when rag_backend='azure'. "
                "Set RAG_UI_AZURE_SEARCH_ENDPOINT environment variable."
            )
        if not config.azure_search_api_key:
            raise ValueError(
                "Azure AI Search API key required when rag_backend='azure'. "
                "Set RAG_UI_AZURE_SEARCH_API_KEY environment variable."
            )

        from src.rag.retriever.azure_search import AzureAISearchRetriever

        logger.info(
            f"Creating AzureAISearchRetriever: endpoint={config.azure_search_endpoint}, "
            f"index={config.azure_search_index}"
        )

        return AzureAISearchRetriever(
            endpoint=config.azure_search_endpoint,
            api_key=config.azure_search_api_key,
            index_name=config.azure_search_index,
            embedder=embedder,
            vector_field=config.azure_search_vector_field,
            content_field=config.azure_search_content_field,
            id_field=config.azure_search_id_field,
            document_type_field=config.azure_search_document_type_field,
            domain_field=config.azure_search_domain_field,
            metadata_field=config.azure_search_metadata_field,
            hybrid_search=config.azure_search_hybrid,
            semantic_configuration=config.azure_search_semantic_config,
        )

    else:  # postgres backend (default)
        if pool is None:
            raise ValueError(
                "PostgreSQL connection pool required when rag_backend='postgres'. "
                "Pass a pool argument to create_retriever()."
            )

        from src.rag.retriever.retriever import HybridRAGRetriever

        logger.info(
            f"Creating HybridRAGRetriever: dimension={config.embedding_dimension}, "
            f"vector_weight={config.vector_weight}, keyword_weight={config.keyword_weight}"
        )

        retriever = HybridRAGRetriever(
            pool=pool,
            embedding_dimension=config.embedding_dimension,
            vector_weight=config.vector_weight,
            keyword_weight=config.keyword_weight,
        )
        retriever.set_embedder(embedder)
        return retriever
