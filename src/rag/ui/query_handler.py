"""
RAG Query Handler

Orchestrates RAG queries against SEC filings:
1. Extract company/ticker from query using entity resolution
2. Retrieve relevant chunks (filtered by company if detected)
3. Build context from chunks
4. Generate answer using Claude
"""

import logging
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import anthropic
import asyncpg

from src.evalkit.common.llm import create_anthropic_client

if TYPE_CHECKING:
    pass
from src.nl2api.resolution import HttpEntityResolver
from src.nl2api.resolution.resolver import ExternalEntityResolver
from src.rag.retriever.embedders import LocalEmbedder, OpenAIEmbedder
from src.rag.retriever.protocols import DocumentType, RetrievalResult
from src.rag.retriever.retriever import HybridRAGRetriever
from src.rag.ui.config import RAGUIConfig

logger = logging.getLogger(__name__)

# Temporal keywords that indicate user wants the most recent data
TEMPORAL_KEYWORDS = {
    "latest",
    "recent",
    "last",
    "most recent",
    "current",
    "newest",
    "this year",
    "fiscal 2024",
    "fy2024",
    "fy 2024",
    "2024",
}

# Recency boost factor per year difference (e.g., 0.05 = 5% boost per year newer)
RECENCY_BOOST_PER_YEAR = 0.05

ANSWER_SYSTEM_PROMPT = """You are a financial analyst answering questions based on SEC filing excerpts.

Given a question and retrieved context from 10-K/10-Q filings, provide a clear, factual answer.

Rules:
- Answer ONLY based on the provided context - do not use external knowledge
- Include specific numbers, dates, and facts from the filings when available
- If the context doesn't contain enough information to answer, say so clearly
- Keep answers concise but complete (typically 2-5 sentences)
- Use precise financial terminology
- Do NOT speculate beyond what the filings state
- Do NOT provide investment advice

When citing sources, reference the company ticker and filing type (e.g., "According to Apple's 10-K...")."""


@dataclass
class QueryResult:
    """Result of a RAG query."""

    answer: str
    sources: list[RetrievalResult]
    query: str
    metadata: dict[str, Any] = field(default_factory=dict)


class RAGQueryHandler:
    """Handles RAG queries against SEC filings."""

    def __init__(
        self,
        pool: asyncpg.Pool,
        config: RAGUIConfig,
    ):
        """
        Initialize the query handler.

        Args:
            pool: asyncpg connection pool
            config: RAG UI configuration
        """
        self._pool = pool
        self._config = config
        self._retriever: HybridRAGRetriever | None = None
        self._embedder: LocalEmbedder | OpenAIEmbedder | None = None
        self._client: anthropic.AsyncAnthropic | None = None
        self._entity_resolver: ExternalEntityResolver | None = None
        self._known_tickers: set[str] = set()

    async def initialize(self) -> None:
        """Initialize embedder, retriever, entity resolver, and LLM client."""
        # Load known tickers from database
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT DISTINCT metadata->>'ticker' as ticker
                FROM rag_documents
                WHERE document_type = 'sec_filing'
                AND metadata->>'ticker' IS NOT NULL
                """
            )
            self._known_tickers = {r["ticker"] for r in rows if r["ticker"]}
            logger.info(f"Loaded {len(self._known_tickers)} known tickers")

        # Initialize embedder
        if self._config.embedding_provider == "openai":
            if not self._config.openai_api_key:
                raise ValueError("OpenAI API key required for openai embedder")
            self._embedder = OpenAIEmbedder(api_key=self._config.openai_api_key)
        else:
            self._embedder = LocalEmbedder(model_name=self._config.embedding_model)

        # Initialize retriever
        self._retriever = HybridRAGRetriever(
            pool=self._pool,
            embedding_dimension=self._config.embedding_dimension,
            vector_weight=self._config.vector_weight,
            keyword_weight=self._config.keyword_weight,
        )
        self._retriever.set_embedder(self._embedder)

        # Initialize entity resolver
        if self._config.entity_resolution_endpoint:
            # Use HTTP service
            logger.info(f"Using HTTP entity resolver: {self._config.entity_resolution_endpoint}")
            self._entity_resolver = HttpEntityResolver(
                base_url=self._config.entity_resolution_endpoint,
                timeout_seconds=self._config.entity_resolution_timeout,
            )
        else:
            # Use local resolver (DB lookups + OpenFIGI fallback)
            logger.info("Using local entity resolver")
            self._entity_resolver = ExternalEntityResolver(
                db_pool=self._pool,
                use_cache=True,
            )

        # Initialize Anthropic client using shared factory
        self._client = create_anthropic_client(
            async_client=True,
            api_key=self._config.anthropic_api_key,
        )

        logger.info(
            f"RAGQueryHandler initialized: embedder={self._config.embedding_provider}, "
            f"model={self._config.llm_model}"
        )

    async def _extract_ticker(self, query: str) -> str | None:
        """
        Extract ticker symbol from query using entity resolution.

        Uses ExternalEntityResolver to extract and resolve company mentions
        from the query, then maps RICs to tickers we have filings for.

        Args:
            query: Natural language query

        Returns:
            Ticker symbol if a known company is mentioned, None otherwise
        """
        if self._entity_resolver is None:
            logger.warning("Entity resolver not initialized")
            return None

        try:
            # Use entity resolver to extract and resolve company mentions
            resolved = await self._entity_resolver.resolve(query)

            if not resolved:
                return None

            # Get the first resolved entity and extract ticker
            for entity_name, ric in resolved.items():
                # Use resolve_single to get full entity info with metadata
                entity_result = await self._entity_resolver.resolve_single(entity_name)
                if entity_result:
                    # Prefer ticker from metadata, fallback to extracting from RIC
                    ticker = entity_result.metadata.get("ticker")
                    if not ticker:
                        # Fallback: extract from RIC (e.g., "AAPL.O" -> "AAPL")
                        ticker = ric.split(".")[0] if "." in ric else ric

                    # Check if we have filings for this ticker
                    if ticker in self._known_tickers:
                        logger.info(
                            f"Resolved '{entity_name}' -> {ric} -> ticker {ticker} "
                            f"(from {'metadata' if entity_result.metadata.get('ticker') else 'RIC'})"
                        )
                        return ticker

            # If no match in known tickers, log what we found
            logger.debug(f"Resolved entities {resolved} but no matching tickers in filings")
            return None

        except Exception as e:
            logger.warning(f"Entity resolution failed: {e}")
            return None

    def _detect_temporal_intent(self, query: str) -> bool:
        """Detect if query indicates user wants the most recent data."""
        query_lower = query.lower()
        return any(keyword in query_lower for keyword in TEMPORAL_KEYWORDS)

    async def _retrieve_for_company(
        self,
        query: str,
        ticker: str,
        top_k: int,
        latest_only: bool = False,
    ) -> list[RetrievalResult]:
        """
        Retrieve chunks filtered to a specific company.

        Args:
            query: The search query
            ticker: Company ticker to filter by
            top_k: Number of results to return
            latest_only: If True, filter to most recent fiscal year only
        """
        if self._embedder is None:
            raise RuntimeError("Embedder not initialized")

        # Generate query embedding
        embedding = await self._embedder.embed(query)
        embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                await conn.execute("SET LOCAL enable_indexscan = off")

                # Get max fiscal year for recency calculations (and filtering if latest_only)
                max_fiscal_year = await conn.fetchval(
                    """
                    SELECT MAX((metadata->>'fiscal_year')::int)
                    FROM rag_documents
                    WHERE document_type = 'sec_filing'
                    AND metadata->>'ticker' = $1
                    AND metadata->>'fiscal_year' IS NOT NULL
                    """,
                    ticker,
                )

                if latest_only and max_fiscal_year:
                    # Filter to only the most recent fiscal year
                    logger.info(f"Filtering to latest fiscal year: {max_fiscal_year}")
                    rows = await conn.fetch(
                        """
                        SELECT
                            id, content, document_type, domain, field_code,
                            example_query, example_api_call, metadata,
                            1 - (embedding <=> $1::vector) as score
                        FROM rag_documents
                        WHERE document_type = 'sec_filing'
                        AND metadata->>'ticker' = $2
                        AND (metadata->>'fiscal_year')::int = $3
                        ORDER BY embedding <=> $1::vector
                        LIMIT $4
                        """,
                        embedding_str,
                        ticker,
                        max_fiscal_year,
                        top_k,
                    )
                else:
                    # Return all years, but we'll apply recency boost
                    rows = await conn.fetch(
                        """
                        SELECT
                            id, content, document_type, domain, field_code,
                            example_query, example_api_call, metadata,
                            1 - (embedding <=> $1::vector) as score
                        FROM rag_documents
                        WHERE document_type = 'sec_filing'
                        AND metadata->>'ticker' = $2
                        ORDER BY embedding <=> $1::vector
                        LIMIT $3
                        """,
                        embedding_str,
                        ticker,
                        top_k * 2,  # Fetch more to allow re-ranking
                    )

        results = []
        for row in rows:
            metadata = row["metadata"]
            if isinstance(metadata, str):
                import json

                metadata = json.loads(metadata)

            base_score = float(row["score"])

            # Apply recency boost if we have fiscal_year data
            fiscal_year = metadata.get("fiscal_year") if metadata else None
            if fiscal_year and max_fiscal_year and not latest_only:
                # Penalize older documents (5% per year older than max)
                year_diff = max_fiscal_year - int(fiscal_year)
                recency_factor = 1.0 - (year_diff * RECENCY_BOOST_PER_YEAR)
                recency_factor = max(0.5, recency_factor)  # Don't penalize more than 50%
                adjusted_score = base_score * recency_factor
            else:
                adjusted_score = base_score

            results.append(
                RetrievalResult(
                    id=str(row["id"]),
                    content=row["content"],
                    document_type=DocumentType(row["document_type"]),
                    score=adjusted_score,
                    domain=row["domain"],
                    field_code=row["field_code"],
                    example_query=row["example_query"],
                    example_api_call=row["example_api_call"],
                    metadata=metadata or {},
                )
            )

        # Re-sort by adjusted score and limit to top_k
        results.sort(key=lambda x: x.score, reverse=True)
        return results[:top_k]

    async def query(
        self,
        question: str,
        top_k: int | None = None,
    ) -> QueryResult:
        """
        Process a RAG query.

        Args:
            question: Natural language question about SEC filings
            top_k: Number of chunks to retrieve (default from config)

        Returns:
            QueryResult with answer and source chunks
        """
        if self._retriever is None or self._client is None:
            raise RuntimeError("Handler not initialized. Call initialize() first.")

        top_k = top_k or self._config.default_top_k

        # Step 1: Check if a specific company is mentioned (uses entity resolver)
        ticker = await self._extract_ticker(question)

        # Step 2: Check for temporal intent (wants latest data)
        wants_latest = self._detect_temporal_intent(question)
        if wants_latest:
            logger.info("Detected temporal intent - filtering to latest fiscal year")

        # Step 3: Retrieve relevant chunks
        # Use small-to-big retrieval if enabled and we're not filtering by ticker
        # (ticker filtering has its own optimized path)
        use_small_to_big = self._config.use_small_to_big and not ticker

        if ticker:
            logger.info(f"Detected company ticker: {ticker}")
            chunks = await self._retrieve_for_company(
                question, ticker, top_k, latest_only=wants_latest
            )
        elif use_small_to_big:
            logger.info("Using small-to-big retrieval (search children, return parents)")
            chunks = await self._retriever.retrieve_with_parents(
                query=question,
                limit=top_k,
                child_limit=self._config.small_to_big_child_limit,
                threshold=self._config.retrieval_threshold,
                use_cache=True,
            )
        else:
            chunks = await self._retriever.retrieve(
                query=question,
                document_types=[DocumentType.SEC_FILING],
                limit=top_k,
                threshold=self._config.retrieval_threshold,
                use_cache=True,
            )

        # Step 4: Build context from chunks
        context = self._build_context(chunks)

        # Step 5: Generate answer using LLM
        answer = await self._generate_answer(question, context)

        return QueryResult(
            answer=answer,
            sources=chunks,
            query=question,
            metadata={
                "chunks_retrieved": len(chunks),
                "model": self._config.llm_model,
                "ticker_detected": ticker,
                "temporal_filter_applied": wants_latest,
                "retrieval_type": "small_to_big" if use_small_to_big else "standard",
            },
        )

    def _build_context(self, chunks: list[RetrievalResult]) -> str:
        """Build context string from retrieved chunks."""
        if not chunks:
            return "No relevant context found in SEC filings."

        context_parts = []
        for i, chunk in enumerate(chunks, 1):
            metadata = chunk.metadata or {}
            ticker = metadata.get("ticker", "Unknown")
            filing_type = metadata.get("filing_type", "10-K")
            fiscal_year = metadata.get("fiscal_year", "")
            section = metadata.get("section", "")

            header = f"--- Source {i}: {ticker} {filing_type}"
            if fiscal_year:
                header += f" FY{fiscal_year}"
            if section:
                header += f" ({section})"
            header += f" [Score: {chunk.score:.2f}] ---"

            context_parts.append(f"{header}\n{chunk.content}")

        return "\n\n".join(context_parts)

    async def _generate_answer(self, question: str, context: str) -> str:
        """Generate an answer using Claude."""
        if self._client is None:
            raise RuntimeError("LLM client not initialized")

        user_prompt = f"""Question: {question}

Context from SEC filings:
{context}

Please provide a clear, factual answer based on the context above."""

        try:
            response = await self._client.messages.create(
                model=self._config.llm_model,
                max_tokens=self._config.max_answer_tokens,
                system=ANSWER_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": user_prompt}],
            )
            return response.content[0].text

        except anthropic.APIError as e:
            logger.error(f"LLM API error: {e}")
            return f"Sorry, I encountered an error generating the answer: {e}"

    async def get_stats(self) -> dict[str, Any]:
        """Get statistics about the RAG system."""
        stats: dict[str, Any] = {
            "config": {
                "embedding_provider": self._config.embedding_provider,
                "llm_model": self._config.llm_model,
                "default_top_k": self._config.default_top_k,
            }
        }

        # Get document count
        async with self._pool.acquire() as conn:
            sec_count = await conn.fetchval(
                "SELECT COUNT(*) FROM rag_documents WHERE document_type = 'sec_filing'"
            )
            stats["sec_filing_chunks"] = sec_count

            # Get unique companies
            companies = await conn.fetch(
                """
                SELECT DISTINCT metadata->>'ticker' as ticker
                FROM rag_documents
                WHERE document_type = 'sec_filing'
                AND metadata->>'ticker' IS NOT NULL
                """
            )
            stats["unique_companies"] = len(companies)
            stats["companies"] = sorted([r["ticker"] for r in companies if r["ticker"]])

        return stats


async def create_handler(config: RAGUIConfig | None = None) -> RAGQueryHandler:
    """
    Factory function to create and initialize a RAGQueryHandler.

    Args:
        config: Configuration (uses defaults if not provided)

    Returns:
        Initialized RAGQueryHandler
    """
    if config is None:
        config = RAGUIConfig()

    # Create connection pool
    pool = await asyncpg.create_pool(config.database_url)
    if pool is None:
        raise RuntimeError("Failed to create database connection pool")

    # Create and initialize handler
    handler = RAGQueryHandler(pool=pool, config=config)
    await handler.initialize()

    return handler
