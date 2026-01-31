# RAG-07: Azure AI Search Integration for RAG Evaluation

**Status:** Proposed
**Created:** 2026-01-25

## Goal

Connect evalkit RAG evaluation to production Azure AI Search backend while keeping evaluation infrastructure (questions, scorecards) in local PostgreSQL.

## Context

- Production RAG system runs on Azure AI Search
- Want to evaluate production system locally
- Keep test cases and scorecards in local PostgreSQL
- Azure index uses: standard field names, Azure OpenAI embeddings, hybrid search

## Current State

- RAG retriever uses `RAGRetriever` protocol (`src/rag/retriever/protocols.py`)
- Current implementation: `HybridRAGRetriever` using PostgreSQL + pgvector
- Evaluation pack (`src/rag/evaluation/pack.py`) is retriever-agnostic
- Results stored in PostgreSQL via `ScorecardRepository`

## Approach: New Azure Retriever Implementation

The codebase has protocol-based design - we implement `AzureAISearchRetriever` that satisfies the existing `RAGRetriever` protocol:

```python
@runtime_checkable
class RAGRetriever(Protocol):
    async def retrieve(query, domain, document_types, limit, threshold) -> list[RetrievalResult]
```

## Files to Create/Modify

### New Files

| File | Purpose |
|------|---------|
| `src/rag/retriever/azure_search.py` | AzureAISearchRetriever implementation |
| `src/rag/retriever/factory.py` | Factory to create retriever by backend type |
| `tests/unit/rag/test_azure_search_retriever.py` | Unit tests with mocked Azure SDK |

### Modified Files

| File | Change |
|------|--------|
| `src/rag/retriever/__init__.py` | Export new classes |
| `src/rag/ui/config.py` | Add Azure config fields |
| `src/rag/ui/query_handler.py` | Use factory to create retriever |
| `pyproject.toml` | Add `azure-search-documents` dependency |

## Implementation Details

### 1. AzureAISearchRetriever Class (~150 lines)

```python
# src/rag/retriever/azure_search.py
from azure.search.documents import SearchClient
from azure.search.documents.models import VectorizedQuery
from azure.core.credentials import AzureKeyCredential

class AzureAISearchRetriever:
    """RAG retriever using Azure AI Search as backend."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        index_name: str,
        embedder: Embedder,
        vector_field: str = "embedding",
        content_field: str = "content",
    ):
        self._client = SearchClient(endpoint, index_name, AzureKeyCredential(api_key))
        self._embedder = embedder
        self._vector_field = vector_field
        self._content_field = content_field

    async def retrieve(
        self,
        query: str,
        domain: str | None = None,
        document_types: list[DocumentType] | None = None,
        limit: int = 10,
        threshold: float = 0.5,
    ) -> list[RetrievalResult]:
        # Generate embedding using Azure OpenAI
        embedding = await self._embedder.embed(query)

        # Hybrid search: vector + full-text
        vector_query = VectorizedQuery(
            vector=embedding,
            k_nearest_neighbors=limit * 2,
            fields=self._vector_field,
        )

        # Build OData filter if domain/document_types specified
        filter_expr = self._build_filter(domain, document_types)

        # Execute hybrid search
        results = self._client.search(
            search_text=query,
            vector_queries=[vector_query],
            filter=filter_expr,
            top=limit,
        )

        # Convert to RetrievalResult
        return [
            self._to_retrieval_result(r)
            for r in results
            if r["@search.score"] >= threshold
        ]

    def _build_filter(self, domain, document_types) -> str | None:
        filters = []
        if domain:
            filters.append(f"domain eq '{domain}'")
        if document_types:
            types = ",".join(f"'{t.value}'" for t in document_types)
            filters.append(f"document_type in ({types})")
        return " and ".join(filters) if filters else None

    def _to_retrieval_result(self, hit: dict) -> RetrievalResult:
        return RetrievalResult(
            id=hit["id"],
            content=hit.get("content", ""),
            document_type=DocumentType(hit.get("document_type", "unknown")),
            score=hit["@search.score"],
            metadata=hit.get("metadata", {}),
            domain=hit.get("domain"),
            field_code=hit.get("field_code"),
            example_query=hit.get("example_query"),
            example_api_call=hit.get("example_api_call"),
        )
```

### 2. Azure OpenAI Embedder Addition (~50 lines)

```python
# src/rag/retriever/embedders.py (add to existing file)
class AzureOpenAIEmbedder(Embedder):
    """Embedder using Azure OpenAI Service."""

    def __init__(
        self,
        endpoint: str,
        api_key: str,
        deployment: str,
        api_version: str = "2024-02-01",
    ):
        from openai import AsyncAzureOpenAI
        self._client = AsyncAzureOpenAI(
            azure_endpoint=endpoint,
            api_key=api_key,
            api_version=api_version,
        )
        self._deployment = deployment
        self._dimension = 1536

    @property
    def dimension(self) -> int:
        return self._dimension

    async def embed(self, text: str) -> list[float]:
        response = await self._client.embeddings.create(
            model=self._deployment,
            input=text,
        )
        return response.data[0].embedding
```

### 3. Configuration Changes (~30 lines)

```python
# src/rag/ui/config.py
class RAGUIConfig(BaseSettings):
    # Existing fields...

    # Backend selection
    rag_backend: Literal["postgres", "azure"] = "postgres"

    # Azure AI Search
    azure_search_endpoint: str | None = None
    azure_search_api_key: str | None = None
    azure_search_index: str = "rag-documents"

    # Azure OpenAI (for embeddings)
    azure_openai_endpoint: str | None = None
    azure_openai_api_key: str | None = None
    azure_openai_embedding_deployment: str = "text-embedding-ada-002"
```

### 4. Retriever Factory (~40 lines)

```python
# src/rag/retriever/factory.py
def create_retriever(config: RAGUIConfig, pool: asyncpg.Pool | None = None) -> RAGRetriever:
    """Create retriever based on configuration."""
    if config.rag_backend == "azure":
        # Azure OpenAI embedder
        embedder = AzureOpenAIEmbedder(
            endpoint=config.azure_openai_endpoint,
            api_key=config.azure_openai_api_key,
            deployment=config.azure_openai_embedding_deployment,
        )
        return AzureAISearchRetriever(
            endpoint=config.azure_search_endpoint,
            api_key=config.azure_search_api_key,
            index_name=config.azure_search_index,
            embedder=embedder,
        )
    else:
        # Existing pgvector retriever
        embedder = create_embedder(config)
        retriever = HybridRAGRetriever(pool, config.embedding_dimension)
        retriever.set_embedder(embedder)
        return retriever
```

## Environment Variables

```bash
# Backend selection
RAG_BACKEND=azure

# Azure AI Search
AZURE_SEARCH_ENDPOINT=https://your-search.search.windows.net
AZURE_SEARCH_API_KEY=your-search-api-key
AZURE_SEARCH_INDEX=rag-documents

# Azure OpenAI (for query embeddings)
AZURE_OPENAI_ENDPOINT=https://your-openai.openai.azure.com
AZURE_OPENAI_API_KEY=your-openai-api-key
AZURE_OPENAI_EMBEDDING_DEPLOYMENT=text-embedding-ada-002
```

## Data Flow

```
                    ┌─────────────────┐
                    │   Test Cases    │
                    │  (PostgreSQL)   │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │ RAGQueryHandler │
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │     create_retriever()       │
              └──────────────┬──────────────┘
                             │
         ┌───────────────────┴───────────────────┐
         │                                       │
         ▼                                       ▼
┌─────────────────┐                    ┌─────────────────┐
│ HybridRAGRetriever │                │AzureAISearchRetriever│
│   (pgvector)    │                    │   (Azure)       │
└────────┬────────┘                    └────────┬────────┘
         │                                       │
         ▼                                       ▼
┌─────────────────┐                    ┌─────────────────┐
│   PostgreSQL    │                    │ Azure AI Search │
│   + pgvector    │                    │   (Production)  │
└─────────────────┘                    └─────────────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │  RAG Eval Pack  │
                    │   (8 stages)    │
                    └────────┬────────┘
                             │
                             ▼
                    ┌─────────────────┐
                    │   Scorecards    │
                    │  (PostgreSQL)   │
                    └─────────────────┘
```

## Verification

```bash
# 1. Set Azure credentials
export RAG_BACKEND=azure
export AZURE_SEARCH_ENDPOINT=https://...
export AZURE_SEARCH_API_KEY=...
export AZURE_OPENAI_ENDPOINT=https://...
export AZURE_OPENAI_API_KEY=...

# 2. Run unit tests
pytest tests/unit/rag/test_azure_search_retriever.py -v

# 3. Test retrieval manually
python -c "
import asyncio
from src.rag.retriever.azure_search import AzureAISearchRetriever
from src.rag.retriever.embedders import AzureOpenAIEmbedder
import os

async def test():
    embedder = AzureOpenAIEmbedder(
        endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
        api_key=os.environ['AZURE_OPENAI_API_KEY'],
        deployment='text-embedding-ada-002',
    )
    retriever = AzureAISearchRetriever(
        endpoint=os.environ['AZURE_SEARCH_ENDPOINT'],
        api_key=os.environ['AZURE_SEARCH_API_KEY'],
        index_name=os.environ.get('AZURE_SEARCH_INDEX', 'rag-documents'),
        embedder=embedder,
    )
    results = await retriever.retrieve('What was Apple revenue?', limit=5)
    for r in results:
        print(f'{r.score:.3f}: {r.content[:100]}...')

asyncio.run(test())
"

# 4. Run RAG evaluation against Azure backend
python -m src.evalkit.cli.main batch run --pack rag --tag rag --limit 10 --label azure-test
```

## Dependencies

```toml
# pyproject.toml additions
azure-search-documents = ">=11.4.0"
azure-identity = ">=1.15.0"  # Optional: for managed identity auth
```

## Scope

| In Scope | Out of Scope |
|----------|--------------|
| New `AzureAISearchRetriever` class | Index creation/management |
| `AzureOpenAIEmbedder` for query embedding | Data migration from pgvector |
| Configuration for Azure backend | Azure index schema definition |
| Factory pattern for retriever creation | Authentication via managed identity (can add later) |
| Unit tests with mocked SDK | Integration tests (manual verification) |

## Estimate

- AzureAISearchRetriever: ~150 lines
- AzureOpenAIEmbedder: ~50 lines
- Factory + config: ~70 lines
- Tests: ~150 lines
- **Total: ~420 lines**

## Risks

1. **Field name mismatch** - Azure index may have different field names than expected. Mitigation: Make field names configurable.
2. **Embedding dimension mismatch** - Production may use different embedding model. Mitigation: Verify deployment name matches.
3. **Score normalization** - Azure AI Search scores differ from pgvector cosine similarity. Mitigation: May need to adjust threshold defaults.
