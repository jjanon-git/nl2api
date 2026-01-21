"""
RAG Indexer

Indexes field codes, query examples, and other documents into the RAG system.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.nl2api.rag.protocols import DocumentType

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)


@dataclass
class FieldCodeDocument:
    """A field code document to be indexed."""

    field_code: str
    description: str
    domain: str
    natural_language_hints: list[str]
    metadata: dict[str, Any] | None = None


@dataclass
class QueryExampleDocument:
    """A query example document to be indexed."""

    query: str
    api_call: str
    domain: str
    complexity_level: int = 1
    metadata: dict[str, Any] | None = None


class RAGIndexer:
    """
    Indexes documents into the RAG system.

    Handles:
    - Field code indexing with embeddings
    - Query example indexing
    - Batch operations for efficiency
    """

    def __init__(
        self,
        pool: "asyncpg.Pool",
        embedder: Any | None = None,
    ):
        """
        Initialize the indexer.

        Args:
            pool: asyncpg connection pool
            embedder: Optional embedder for generating embeddings
        """
        self._pool = pool
        self._embedder = embedder

    def set_embedder(self, embedder: Any) -> None:
        """Set the embedder for generating embeddings."""
        self._embedder = embedder

    async def index_field_code(
        self,
        doc: FieldCodeDocument,
        generate_embedding: bool = True,
    ) -> str:
        """
        Index a single field code document.

        Args:
            doc: Field code document to index
            generate_embedding: Whether to generate embedding

        Returns:
            Document ID
        """
        # Build content from description and hints
        content = doc.description
        if doc.natural_language_hints:
            content += " | Keywords: " + ", ".join(doc.natural_language_hints)

        # Generate embedding if requested and embedder is available
        embedding = None
        if generate_embedding and self._embedder:
            embedding = await self._embedder.embed(content)

        doc_id = str(uuid.uuid4())

        async with self._pool.acquire() as conn:
            if embedding:
                await conn.execute(
                    """
                    INSERT INTO rag_documents (id, content, document_type, domain, field_code, metadata, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6, $7::vector)
                    ON CONFLICT (domain, field_code) WHERE document_type = 'field_code' AND field_code IS NOT NULL
                    DO UPDATE SET content = $2, metadata = $6, embedding = $7::vector, updated_at = NOW()
                    """,
                    doc_id,
                    content,
                    DocumentType.FIELD_CODE.value,
                    doc.domain,
                    doc.field_code,
                    doc.metadata or {},
                    embedding,
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO rag_documents (id, content, document_type, domain, field_code, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (domain, field_code) WHERE document_type = 'field_code' AND field_code IS NOT NULL
                    DO UPDATE SET content = $2, metadata = $6, updated_at = NOW()
                    """,
                    doc_id,
                    content,
                    DocumentType.FIELD_CODE.value,
                    doc.domain,
                    doc.field_code,
                    doc.metadata or {},
                )

        logger.debug(f"Indexed field code: {doc.field_code}")
        return doc_id

    async def index_field_codes_batch(
        self,
        docs: list[FieldCodeDocument],
        generate_embeddings: bool = True,
        batch_size: int = 50,
    ) -> list[str]:
        """
        Index multiple field code documents in batch.

        Args:
            docs: List of field code documents
            generate_embeddings: Whether to generate embeddings
            batch_size: Batch size for embedding generation

        Returns:
            List of document IDs
        """
        doc_ids = []

        # Process in batches for embedding generation
        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]

            # Generate embeddings in batch
            embeddings = None
            if generate_embeddings and self._embedder:
                contents = []
                for doc in batch:
                    content = doc.description
                    if doc.natural_language_hints:
                        content += " | Keywords: " + ", ".join(doc.natural_language_hints)
                    contents.append(content)
                embeddings = await self._embedder.embed_batch(contents)

            # Insert documents
            for j, doc in enumerate(batch):
                content = doc.description
                if doc.natural_language_hints:
                    content += " | Keywords: " + ", ".join(doc.natural_language_hints)

                doc_id = str(uuid.uuid4())
                embedding = embeddings[j] if embeddings else None

                async with self._pool.acquire() as conn:
                    if embedding:
                        await conn.execute(
                            """
                            INSERT INTO rag_documents (id, content, document_type, domain, field_code, metadata, embedding)
                            VALUES ($1, $2, $3, $4, $5, $6, $7::vector)
                            ON CONFLICT (domain, field_code) WHERE document_type = 'field_code' AND field_code IS NOT NULL
                            DO UPDATE SET content = $2, metadata = $6, embedding = $7::vector, updated_at = NOW()
                            """,
                            doc_id,
                            content,
                            DocumentType.FIELD_CODE.value,
                            doc.domain,
                            doc.field_code,
                            doc.metadata or {},
                            embedding,
                        )
                    else:
                        await conn.execute(
                            """
                            INSERT INTO rag_documents (id, content, document_type, domain, field_code, metadata)
                            VALUES ($1, $2, $3, $4, $5, $6)
                            ON CONFLICT (domain, field_code) WHERE document_type = 'field_code' AND field_code IS NOT NULL
                            DO UPDATE SET content = $2, metadata = $6, updated_at = NOW()
                            """,
                            doc_id,
                            content,
                            DocumentType.FIELD_CODE.value,
                            doc.domain,
                            doc.field_code,
                            doc.metadata or {},
                        )

                doc_ids.append(doc_id)

            logger.info(f"Indexed batch of {len(batch)} field codes")

        return doc_ids

    async def index_query_example(
        self,
        doc: QueryExampleDocument,
        generate_embedding: bool = True,
    ) -> str:
        """
        Index a query example document.

        Args:
            doc: Query example document
            generate_embedding: Whether to generate embedding

        Returns:
            Document ID
        """
        content = f"Q: {doc.query}\nA: {doc.api_call}"

        embedding = None
        if generate_embedding and self._embedder:
            embedding = await self._embedder.embed(content)

        doc_id = str(uuid.uuid4())

        async with self._pool.acquire() as conn:
            if embedding:
                await conn.execute(
                    """
                    INSERT INTO rag_documents (id, content, document_type, domain, example_query, example_api_call, metadata, embedding)
                    VALUES ($1, $2, $3, $4, $5, $6, $7, $8::vector)
                    """,
                    doc_id,
                    content,
                    DocumentType.QUERY_EXAMPLE.value,
                    doc.domain,
                    doc.query,
                    doc.api_call,
                    {"complexity_level": doc.complexity_level, **(doc.metadata or {})},
                    embedding,
                )
            else:
                await conn.execute(
                    """
                    INSERT INTO rag_documents (id, content, document_type, domain, example_query, example_api_call, metadata)
                    VALUES ($1, $2, $3, $4, $5, $6, $7)
                    """,
                    doc_id,
                    content,
                    DocumentType.QUERY_EXAMPLE.value,
                    doc.domain,
                    doc.query,
                    doc.api_call,
                    {"complexity_level": doc.complexity_level, **(doc.metadata or {})},
                )

        logger.debug(f"Indexed query example: {doc.query[:50]}...")
        return doc_id

    async def clear_domain(self, domain: str) -> int:
        """
        Clear all documents for a domain.

        Args:
            domain: Domain to clear

        Returns:
            Number of documents deleted
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                "DELETE FROM rag_documents WHERE domain = $1",
                domain,
            )
            count = int(result.split()[-1])
            logger.info(f"Cleared {count} documents for domain: {domain}")
            return count

    async def get_stats(self) -> dict[str, Any]:
        """Get indexing statistics."""
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT domain, document_type, COUNT(*) as count,
                       COUNT(embedding) as with_embedding
                FROM rag_documents
                GROUP BY domain, document_type
                ORDER BY domain, document_type
                """
            )

            stats = {}
            for row in rows:
                domain = row["domain"] or "unknown"
                if domain not in stats:
                    stats[domain] = {}
                stats[domain][row["document_type"]] = {
                    "count": row["count"],
                    "with_embedding": row["with_embedding"],
                }

            return stats


def parse_estimates_reference(content: str) -> list[FieldCodeDocument]:
    """
    Parse ESTIMATES_REFERENCE.md to extract field codes.

    Args:
        content: Markdown content of the reference file

    Returns:
        List of FieldCodeDocument objects
    """
    docs = []

    # Pattern to match table rows with field codes
    # Format: | Natural Language | TR Code | Description |
    table_pattern = r'\|\s*([^|]+)\s*\|\s*`([^`]+)`\s*\|\s*([^|]+)\s*\|'

    for match in re.finditer(table_pattern, content):
        natural_lang = match.group(1).strip()
        field_code = match.group(2).strip()
        description = match.group(3).strip()

        # Skip header rows
        if natural_lang.lower() in ("natural language", "---", "metric", "expression"):
            continue

        # Extract keywords from natural language hints
        keywords = [kw.strip() for kw in natural_lang.split(",")]

        docs.append(FieldCodeDocument(
            field_code=field_code,
            description=description,
            domain="estimates",
            natural_language_hints=keywords,
            metadata={"source": "ESTIMATES_REFERENCE.md"},
        ))

    return docs


def parse_query_examples(content: str) -> list[QueryExampleDocument]:
    """
    Parse ESTIMATES_REFERENCE.md to extract query examples.

    Args:
        content: Markdown content of the reference file

    Returns:
        List of QueryExampleDocument objects
    """
    docs = []

    # Pattern to match Q&A pairs
    # Format: **Q1:** "query text"
    # ```python
    # get_data(...)
    # ```
    qa_pattern = r'\*\*Q\d+:\*\*\s*"([^"]+)"\s*```python\s*(get_data\([^`]+)\s*```'

    for match in re.finditer(qa_pattern, content, re.DOTALL):
        query = match.group(1).strip()
        api_call = match.group(2).strip()

        # Determine complexity from section (rough heuristic)
        complexity = 1
        if "Level 5" in content[:match.start()]:
            complexity = 5
        elif "Level 4" in content[:match.start()]:
            complexity = 4
        elif "Level 3" in content[:match.start()]:
            complexity = 3
        elif "Level 2" in content[:match.start()]:
            complexity = 2

        docs.append(QueryExampleDocument(
            query=query,
            api_call=api_call,
            domain="estimates",
            complexity_level=complexity,
            metadata={"source": "ESTIMATES_REFERENCE.md"},
        ))

    return docs
