"""
Azure AI Search Retriever

RAG retriever implementation using Azure AI Search as the backend.
Supports hybrid search (vector + full-text) with configurable field mappings.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from src.rag.retriever.protocols import DocumentType, RetrievalResult

if TYPE_CHECKING:
    from azure.search.documents import SearchClient

    from src.rag.retriever.embedders import Embedder

logger = logging.getLogger(__name__)


def _get_vectorized_query_class():
    """Lazy import of VectorizedQuery to allow mocking in tests."""
    from azure.search.documents.models import VectorizedQuery

    return VectorizedQuery


class AzureAISearchRetriever:
    """
    RAG retriever using Azure AI Search as backend.

    Implements the RAGRetriever protocol for use with the evaluation framework.
    Supports hybrid search combining vector similarity and full-text search.

    Field mappings are configurable to match your Azure index schema.
    """

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        index_name: str,
        embedder: Embedder,
        *,
        # Field mappings (configure to match your index schema)
        vector_field: str = "embedding",
        content_field: str = "content",
        id_field: str = "id",
        document_type_field: str = "document_type",
        domain_field: str = "domain",
        metadata_field: str = "metadata",
        # Optional field code fields
        field_code_field: str | None = "field_code",
        # Optional example fields
        example_query_field: str | None = "example_query",
        example_api_call_field: str | None = "example_api_call",
        # Search parameters
        hybrid_search: bool = True,
        semantic_configuration: str | None = None,
    ):
        """
        Initialize the Azure AI Search retriever.

        Args:
            endpoint: Azure AI Search service endpoint (e.g., https://your-search.search.windows.net)
            api_key: Azure AI Search API key
            index_name: Name of the search index
            embedder: Embedder for generating query vectors
            vector_field: Name of the vector field in the index
            content_field: Name of the content/text field
            id_field: Name of the document ID field
            document_type_field: Name of the document type field
            domain_field: Name of the domain field (for filtering)
            metadata_field: Name of the metadata JSON field
            field_code_field: Name of the field code field (for field_code documents)
            example_query_field: Name of the example query field (for examples)
            example_api_call_field: Name of the example API call field (for examples)
            hybrid_search: Whether to use hybrid search (vector + full-text)
            semantic_configuration: Optional semantic configuration name for reranking
        """
        try:
            from azure.core.credentials import AzureKeyCredential
            from azure.search.documents import SearchClient
        except ImportError:
            raise ImportError(
                "azure-search-documents package required. "
                "Install with: pip install azure-search-documents"
            )

        self._endpoint = endpoint
        self._index_name = index_name
        self._embedder = embedder

        # Field mappings
        self._vector_field = vector_field
        self._content_field = content_field
        self._id_field = id_field
        self._document_type_field = document_type_field
        self._domain_field = domain_field
        self._metadata_field = metadata_field
        self._field_code_field = field_code_field
        self._example_query_field = example_query_field
        self._example_api_call_field = example_api_call_field

        # Search options
        self._hybrid_search = hybrid_search
        self._semantic_configuration = semantic_configuration

        # Initialize client
        self._client: SearchClient = SearchClient(
            endpoint=endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(api_key),
        )

        logger.info(
            f"Initialized AzureAISearchRetriever: endpoint={endpoint}, "
            f"index={index_name}, hybrid={hybrid_search}"
        )

    def _build_filter(
        self,
        domain: str | None = None,
        document_types: list[DocumentType] | None = None,
    ) -> str | None:
        """Build OData filter expression for Azure AI Search."""
        filters = []

        if domain:
            # OData string comparison
            filters.append(f"{self._domain_field} eq '{domain}'")

        if document_types:
            if len(document_types) == 1:
                # Single type - simple equality
                filters.append(f"{self._document_type_field} eq '{document_types[0].value}'")
            else:
                # Multiple types - use search.in function
                type_values = ",".join(dt.value for dt in document_types)
                filters.append(f"search.in({self._document_type_field}, '{type_values}')")

        if not filters:
            return None

        return " and ".join(filters)

    def _to_retrieval_result(self, result: dict[str, Any]) -> RetrievalResult:
        """Convert Azure search result to RetrievalResult."""
        # Extract score - Azure uses @search.score for hybrid/semantic
        score = result.get("@search.score", 0.0)
        # Normalize score to 0-1 range if needed (Azure scores can be > 1)
        if score > 1.0:
            score = min(1.0, score / 10.0)  # Simple normalization

        # Get document type
        doc_type_str = result.get(self._document_type_field, "sec_filing")
        try:
            doc_type = DocumentType(doc_type_str)
        except ValueError:
            doc_type = DocumentType.SEC_FILING

        # Get metadata
        metadata = result.get(self._metadata_field, {})
        if isinstance(metadata, str):
            import json

            try:
                metadata = json.loads(metadata)
            except json.JSONDecodeError:
                metadata = {}

        return RetrievalResult(
            id=str(result.get(self._id_field, "")),
            content=result.get(self._content_field, ""),
            document_type=doc_type,
            score=score,
            domain=result.get(self._domain_field),
            field_code=result.get(self._field_code_field) if self._field_code_field else None,
            example_query=result.get(self._example_query_field)
            if self._example_query_field
            else None,
            example_api_call=result.get(self._example_api_call_field)
            if self._example_api_call_field
            else None,
            metadata=metadata,
        )

    async def retrieve(
        self,
        query: str,
        domain: str | None = None,
        document_types: list[DocumentType] | None = None,
        limit: int = 10,
        threshold: float = 0.5,
        use_cache: bool = True,  # Ignored - Azure handles caching
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant documents for a query.

        Uses hybrid search combining vector similarity and full-text search.

        Args:
            query: Natural language query
            domain: Optional domain filter (e.g., "estimates")
            document_types: Optional filter by document type
            limit: Maximum results to return
            threshold: Minimum relevance score (0.0 to 1.0)
            use_cache: Ignored (Azure handles caching internally)

        Returns:
            List of RetrievalResult ordered by relevance descending
        """
        # Generate query embedding
        embedding = await self._embedder.embed(query)

        # Build vector query (lazy import for testability)
        VectorizedQuery = _get_vectorized_query_class()
        vector_query = VectorizedQuery(
            vector=embedding,
            k_nearest_neighbors=limit * 2,  # Oversample for filtering
            fields=self._vector_field,
        )

        # Build filter
        filter_expr = self._build_filter(domain, document_types)

        # Execute search
        search_kwargs: dict[str, Any] = {
            "vector_queries": [vector_query],
            "top": limit * 2,  # Oversample to allow threshold filtering
        }

        if self._hybrid_search:
            search_kwargs["search_text"] = query

        if filter_expr:
            search_kwargs["filter"] = filter_expr

        if self._semantic_configuration:
            search_kwargs["query_type"] = "semantic"
            search_kwargs["semantic_configuration_name"] = self._semantic_configuration

        # Select fields to return
        select_fields = [
            self._id_field,
            self._content_field,
            self._document_type_field,
            self._domain_field,
            self._metadata_field,
        ]
        if self._field_code_field:
            select_fields.append(self._field_code_field)
        if self._example_query_field:
            select_fields.append(self._example_query_field)
        if self._example_api_call_field:
            select_fields.append(self._example_api_call_field)

        search_kwargs["select"] = select_fields

        try:
            results = list(self._client.search(**search_kwargs))
        except Exception as e:
            logger.error(f"Azure AI Search query failed: {e}")
            raise

        # Convert to RetrievalResult and filter by threshold
        retrieval_results = []
        for result in results:
            rr = self._to_retrieval_result(result)
            if rr.score >= threshold:
                retrieval_results.append(rr)

        # Sort by score descending and limit
        retrieval_results.sort(key=lambda x: x.score, reverse=True)
        return retrieval_results[:limit]

    async def retrieve_field_codes(
        self,
        query: str,
        domain: str,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        """
        Retrieve relevant field codes for a domain query.

        Args:
            query: Natural language query describing the data needed
            domain: API domain (e.g., "estimates", "fundamentals")
            limit: Maximum field codes to return

        Returns:
            List of RetrievalResult for field codes
        """
        return await self.retrieve(
            query=query,
            domain=domain,
            document_types=[DocumentType.FIELD_CODE],
            limit=limit,
            threshold=0.3,  # Lower threshold for field codes
        )

    async def retrieve_examples(
        self,
        query: str,
        domain: str | None = None,
        limit: int = 3,
    ) -> list[RetrievalResult]:
        """
        Retrieve similar query examples with their API calls.

        Args:
            query: Natural language query
            domain: Optional domain filter
            limit: Maximum examples to return

        Returns:
            List of RetrievalResult for query examples
        """
        return await self.retrieve(
            query=query,
            domain=domain,
            document_types=[DocumentType.QUERY_EXAMPLE],
            limit=limit,
            threshold=0.4,  # Moderate threshold for examples
        )

    async def close(self) -> None:
        """Close the search client."""
        if hasattr(self._client, "close"):
            self._client.close()
