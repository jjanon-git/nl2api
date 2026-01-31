# RAG Retriever Architecture

This document describes the RAG retriever implementation in the nl2api codebase.

## RAGRetriever Protocol

The protocol defines three main retrieval methods (`src/rag/retriever/protocols.py`):

```python
@runtime_checkable
class RAGRetriever(Protocol):
    async def retrieve(
        self,
        query: str,
        domain: str | None = None,
        document_types: list[DocumentType] | None = None,
        limit: int = 10,
        threshold: float = 0.5,
    ) -> list[RetrievalResult]:
        """Retrieve relevant documents for a query."""
        ...

    async def retrieve_field_codes(
        self,
        query: str,
        domain: str,
        limit: int = 5,
    ) -> list[RetrievalResult]:
        """Retrieve relevant field codes for a domain query."""
        ...

    async def retrieve_examples(
        self,
        query: str,
        domain: str | None = None,
        limit: int = 3,
    ) -> list[RetrievalResult]:
        """Retrieve similar query examples with their API calls."""
        ...
```

## RetrievalResult Model

```python
@dataclass(frozen=True)
class RetrievalResult:
    id: str
    content: str
    document_type: DocumentType
    score: float  # Relevance score (0.0 to 1.0)
    metadata: dict[str, Any] = field(default_factory=dict)

    # Domain-specific fields
    domain: str | None = None
    field_code: str | None = None
    example_query: str | None = None
    example_api_call: str | None = None
```

The model is **frozen** (immutable), which is critical for the reranker's score updates.

## HybridRAGRetriever Implementation

Located in `src/rag/retriever/retriever.py`.

### Constructor

```python
def __init__(
    self,
    pool: asyncpg.Pool,
    embedding_dimension: int = 1536,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    redis_cache: RedisCache | None = None,
    cache_ttl_seconds: int = 3600,
    reranker: Reranker | None = None,
    first_stage_limit: int = 50,
):
```

Key configuration options:
- **Weights**: Hybrid search combines vector (0.7) + keyword (0.3) scores
- **Caching**: Optional Redis caching with 1-hour default TTL
- **Reranking**: Optional two-stage retrieval with cross-encoder
- **First Stage Limit**: Candidates before reranking (default 50)

### Core Methods

| Method | Purpose |
|--------|---------|
| `retrieve()` | Main hybrid search (vector + keyword) |
| `retrieve_field_codes()` | Specialized field code search |
| `retrieve_examples()` | Query example retrieval |
| `retrieve_by_keyword()` | Keyword-only search (fallback) |
| `retrieve_with_parents()` | Small-to-big retrieval strategy |

#### `retrieve()` - Main Hybrid Search
- Uses PostgreSQL with pgvector for vector similarity
- Combines vector and keyword search with configurable weights
- Supports optional Redis caching
- Two-stage retrieval: first stage (50 candidates) → rerank → top K results
- SQL uses full outer join of vector and keyword search results

#### `retrieve_field_codes()` - Specialized Field Code Search
- Falls back to keyword-only if embedder not set
- Uses lower threshold (0.3) for field codes
- Delegates to `retrieve()` with `DocumentType.FIELD_CODE` filter

#### `retrieve_examples()` - Query Example Retrieval
- Falls back to keyword-only if embedder not set
- Uses moderate threshold (0.4)
- Filters to `DocumentType.QUERY_EXAMPLE`

#### `retrieve_by_keyword()` - Keyword-Only Search
- No embedder required
- Uses PostgreSQL full-text search (`ts_rank`)
- Fallback when embeddings not available

#### `retrieve_with_parents()` - Small-to-Big Retrieval
- Two-stage: search small child chunks (512 chars), return parent chunks (4000 chars)
- Aggregates child scores by parent
- Better context for RAG with precision of small-to-big strategy
- Normalizes scores by matching child count

## Embedder Interface

Located in `src/rag/retriever/embedders.py`.

### Protocol

```python
@runtime_checkable
class Embedder(Protocol):
    @property
    def dimension(self) -> int:
        """Return the embedding dimension."""
        ...

    async def embed(self, text: str) -> list[float]:
        """Generate embedding for a single text."""
        ...

    async def embed_batch(self, texts: list[str]) -> list[list[float]]:
        """Generate embeddings for multiple texts."""
        ...

    @property
    def stats(self) -> dict[str, int]:
        """Get embedder statistics (optional)."""
        ...
```

### Implementations

| Implementation | Model | Dimensions | Notes |
|----------------|-------|------------|-------|
| `LocalEmbedder` | all-MiniLM-L6-v2 | 384 | No API key required, runs locally |
| `OpenAIEmbedder` | text-embedding-3-small | 1536 | Concurrency control, auto-retry |

### Factory Function

```python
def create_embedder(
    provider: str = "local",  # "local" or "openai"
    **kwargs,
) -> Embedder:
```

## Reranker

Located in `src/rag/retriever/reranker.py`.

```python
@runtime_checkable
class Reranker(Protocol):
    async def rerank(
        self,
        query: str,
        results: list[RetrievalResult],
        top_k: int = 10,
    ) -> list[RetrievalResult]:
        """Rerank retrieval results."""
        ...
```

**CrossEncoderReranker** Implementation:
- Uses sentence-transformers cross-encoder models
- Default: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Creates query-document pairs and scores them jointly
- Async execution via thread pool
- Uses `dataclasses.replace()` to create new frozen RetrievalResult instances with updated scores
- Returns top K reranked results sorted by cross-encoder score

## RAG UI Configuration

Located in `src/rag/ui/config.py`.

```python
class RAGUIConfig(BaseSettings):
    # Database
    database_url: str = "postgresql://nl2api:nl2api@localhost:5432/nl2api"

    # Retriever settings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_provider: str = "local"  # "local" or "openai"
    default_top_k: int = 5
    vector_weight: float = 0.7
    keyword_weight: float = 0.3
    retrieval_threshold: float = 0.1

    # Small-to-big retrieval
    use_small_to_big: bool = False
    small_to_big_child_limit: int = 30

    # LLM settings
    llm_model: str = "claude-3-5-haiku-latest"
    max_context_chunks: int = 8
    max_answer_tokens: int = 1000

    # API keys
    anthropic_api_key: str = ""
    openai_api_key: str = ""

    # Entity resolution
    entity_resolution_endpoint: str | None = None
    entity_resolution_timeout: float = 5.0

    @property
    def embedding_dimension(self) -> int:
        """384 for local, 1536 for OpenAI"""
```

**Environment Variable Prefix**: `RAG_UI_` (e.g., `RAG_UI_ANTHROPIC_API_KEY`)

## Document Types

```python
class DocumentType(str, Enum):
    FIELD_CODE = "field_code"  # API field codes
    QUERY_EXAMPLE = "query_example"  # Example NL queries with API calls
    ECONOMIC_INDICATOR = "economic_indicator"  # Economic indicator codes
    SEC_FILING = "sec_filing"  # SEC 10-K/10-Q filing chunks
```

## RAG Query Handler

Located in `src/rag/ui/query_handler.py`.

### Query Flow

```python
async def query(self, question: str, top_k: int | None = None) -> QueryResult:
    # 1. Extract company ticker using entity resolver
    ticker = await self._extract_ticker(question)

    # 2. Detect if user wants latest data
    wants_latest = self._detect_temporal_intent(question)

    # 3. Retrieve relevant chunks
    # - If ticker detected: use specialized retrieval with recency boost
    # - If small-to-big enabled: use small-to-big strategy
    # - Otherwise: standard hybrid retrieval

    # 4. Build context from chunks
    context = self._build_context(chunks)

    # 5. Generate answer using Claude
    answer = await self._generate_answer(question, context)

    return QueryResult(answer, sources, query, metadata)
```

### Retrieval Paths

1. **Ticker-Filtered**: Custom SQL with vector search + recency boost (5% penalty per year older than max)
2. **Small-to-Big**: Search 30 child chunks, return parent chunks
3. **Standard Hybrid**: Full vector + keyword hybrid search on all SEC filings

## Instantiation Pattern

The UI creates a fresh connection pool and handler for each query (`src/rag/ui/app.py`):

```python
async def query_async(config: RAGUIConfig, question: str, top_k: int) -> dict:
    pool = await asyncpg.create_pool(config.database_url, min_size=1, max_size=3)
    try:
        handler = RAGQueryHandler(pool=pool, config=config)
        await handler.initialize()  # Loads embedder, retriever, resolver
        result = await handler.query(question, top_k=top_k)
        return {...}
    finally:
        await pool.close()
```

## Key Architectural Decisions

| Aspect | Decision | Rationale |
|--------|----------|-----------|
| **Immutability** | RetrievalResult is frozen | Prevents accidental mutations, works with dataclass.replace() |
| **Async Embeddings** | Thread pool for local models | Prevents blocking event loop during encoding |
| **Hybrid Search** | Vector (0.7) + Keyword (0.3) | Better recall than vector alone, precision of keyword alone |
| **Caching** | Optional Redis with TTL | Reduces database load for repeated queries |
| **Two-Stage Retrieval** | First stage (50) → Rerank → Top K | Balances recall and latency |
| **Small-to-Big** | Search children, return parents | Precise matching + full context for RAG |
| **Entity Resolution** | HTTP service OR local | Supports both centralized and standalone deployment |
| **Temporal Boost** | Penalty older docs in same company | Recent filings more relevant for current queries |

## Method Signature Summary

| Method | Signature | Returns |
|--------|-----------|---------|
| `retrieve()` | `(query, domain=None, document_types=None, limit=10, threshold=0.5, use_cache=True)` | `list[RetrievalResult]` |
| `retrieve_field_codes()` | `(query, domain, limit=5)` | `list[RetrievalResult]` |
| `retrieve_examples()` | `(query, domain=None, limit=3)` | `list[RetrievalResult]` |
| `retrieve_by_keyword()` | `(query, domain=None, document_types=None, limit=10)` | `list[RetrievalResult]` |
| `retrieve_with_parents()` | `(query, limit=10, child_limit=30, threshold=0.0, domain=None, use_cache=True)` | `list[RetrievalResult]` |
| `embed()` | `(text: str)` | `list[float]` |
| `embed_batch()` | `(texts: list[str])` | `list[list[float]]` |
| `rerank()` | `(query, results, top_k=10)` | `list[RetrievalResult]` |
