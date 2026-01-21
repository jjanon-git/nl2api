"""
RAG Indexer

Indexes field codes, query examples, and other documents into the RAG system.

Features:
- Bulk insert using PostgreSQL COPY protocol (10-50x faster)
- Checkpoint/resume for large indexing jobs
- Progress tracking with optional Rich progress bar
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.common.telemetry import get_tracer
from src.nl2api.rag.checkpoint import CheckpointManager, IndexingCheckpoint
from src.nl2api.rag.protocols import DocumentType

if TYPE_CHECKING:
    import asyncpg

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

# Type alias for progress callback
ProgressCallback = Callable[[int, int], None]  # (processed, total) -> None


@dataclass
class FieldCodeDocument:
    """A field code document to be indexed."""

    field_code: str
    description: str
    domain: str
    natural_language_hints: list[str]
    metadata: dict[str, Any] | None = None


@dataclass
class EconomicIndicatorDocument:
    """An economic indicator document to be indexed."""

    mnemonic: str
    description: str
    country: str
    indicator_type: str
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
    - Bulk insert using COPY protocol
    - Checkpoint/resume for large jobs
    """

    def __init__(
        self,
        pool: "asyncpg.Pool",
        embedder: Any | None = None,
        use_bulk_insert: bool = True,
    ):
        """
        Initialize the indexer.

        Args:
            pool: asyncpg connection pool
            embedder: Optional embedder for generating embeddings
            use_bulk_insert: Use COPY protocol for bulk inserts (faster)
        """
        self._pool = pool
        self._embedder = embedder
        self._use_bulk_insert = use_bulk_insert
        self._checkpoint_manager = CheckpointManager(pool)

    def set_embedder(self, embedder: Any) -> None:
        """Set the embedder for generating embeddings."""
        self._embedder = embedder

    @property
    def checkpoint_manager(self) -> CheckpointManager:
        """Get the checkpoint manager for resume operations."""
        return self._checkpoint_manager

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
                    json.dumps(doc.metadata or {}),
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
                    json.dumps(doc.metadata or {}),
                )

        logger.debug(f"Indexed field code: {doc.field_code}")
        return doc_id

    async def index_field_codes_batch(
        self,
        docs: list[FieldCodeDocument],
        generate_embeddings: bool = True,
        batch_size: int = 50,
        checkpoint_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[str]:
        """
        Index multiple field code documents in batch.

        Uses bulk insert with COPY protocol for better performance.
        Supports checkpoint/resume for large jobs.

        Args:
            docs: List of field code documents
            generate_embeddings: Whether to generate embeddings
            batch_size: Batch size for embedding generation
            checkpoint_id: Optional checkpoint ID for resume support
            progress_callback: Optional callback for progress updates

        Returns:
            List of document IDs
        """
        with tracer.start_as_current_span("rag.index_field_codes_batch") as span:
            span.set_attribute("rag.total_docs", len(docs))
            span.set_attribute("rag.batch_size", batch_size)
            span.set_attribute("rag.generate_embeddings", generate_embeddings)
            span.set_attribute("rag.use_bulk_insert", self._use_bulk_insert)
            if docs:
                span.set_attribute("rag.domain", docs[0].domain)

            return await self._index_field_codes_batch_impl(
                docs, generate_embeddings, batch_size, checkpoint_id, progress_callback, span
            )

    async def _index_field_codes_batch_impl(
        self,
        docs: list[FieldCodeDocument],
        generate_embeddings: bool,
        batch_size: int,
        checkpoint_id: str | None,
        progress_callback: ProgressCallback | None,
        span: Any,
    ) -> list[str]:
        """Internal implementation of index_field_codes_batch."""
        if not docs:
            return []

        # Create or resume checkpoint
        checkpoint: IndexingCheckpoint | None = None
        start_offset = 0

        if checkpoint_id:
            checkpoint = await self._checkpoint_manager.get_checkpoint(checkpoint_id)
            if checkpoint and checkpoint.is_resumable:
                start_offset = checkpoint.last_offset
                logger.info(
                    f"Resuming indexing from offset {start_offset} "
                    f"({checkpoint.processed_items}/{checkpoint.total_items} done)"
                )
        elif len(docs) > batch_size * 2:
            # Auto-create checkpoint for large jobs
            domain = docs[0].domain if docs else None
            checkpoint = await self._checkpoint_manager.create_checkpoint(
                total_items=len(docs),
                domain=domain,
                batch_size=batch_size,
            )
            checkpoint_id = checkpoint.job_id
            logger.info(f"Created checkpoint {checkpoint_id} for {len(docs)} documents")

        doc_ids = []
        processed = start_offset

        try:
            # Process in batches for embedding generation
            for i in range(start_offset, len(docs), batch_size):
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

                # Use bulk insert or individual inserts
                if self._use_bulk_insert and embeddings:
                    batch_ids = await self._bulk_insert_field_codes(batch, embeddings)
                    doc_ids.extend(batch_ids)
                else:
                    # Fall back to individual inserts
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
                                    json.dumps(doc.metadata or {}),
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
                                    json.dumps(doc.metadata or {}),
                                )

                        doc_ids.append(doc_id)

                processed = i + len(batch)
                logger.info(f"Indexed batch of {len(batch)} field codes ({processed}/{len(docs)})")

                # Update checkpoint
                if checkpoint_id:
                    await self._checkpoint_manager.update_progress(
                        checkpoint_id, processed, processed
                    )

                # Call progress callback
                if progress_callback:
                    progress_callback(processed, len(docs))

            # Mark checkpoint complete
            if checkpoint_id:
                await self._checkpoint_manager.mark_completed(checkpoint_id)

        except Exception as e:
            # Mark checkpoint as failed
            if checkpoint_id:
                await self._checkpoint_manager.mark_failed(checkpoint_id, str(e))
            span.set_attribute("rag.error", str(e))
            raise

        span.set_attribute("rag.indexed_count", len(doc_ids))
        return doc_ids

    async def _bulk_insert_field_codes(
        self,
        docs: list[FieldCodeDocument],
        embeddings: list[list[float]],
    ) -> list[str]:
        """
        Bulk insert field codes using PostgreSQL COPY protocol.

        Uses a staging table for UPSERT behavior with COPY.

        Args:
            docs: Documents to insert
            embeddings: Corresponding embeddings

        Returns:
            List of document IDs
        """
        doc_ids = [str(uuid.uuid4()) for _ in docs]

        # Prepare records for COPY
        records = []
        for i, doc in enumerate(docs):
            content = doc.description
            if doc.natural_language_hints:
                content += " | Keywords: " + ", ".join(doc.natural_language_hints)

            # Convert embedding to PostgreSQL array format
            embedding_str = "[" + ",".join(str(x) for x in embeddings[i]) + "]"

            records.append((
                doc_ids[i],
                content,
                DocumentType.FIELD_CODE.value,
                doc.domain,
                doc.field_code,
                json.dumps(doc.metadata or {}),
                embedding_str,
            ))

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Create temp staging table
                await conn.execute("""
                    CREATE TEMP TABLE staging_rag_documents (
                        id TEXT,
                        content TEXT,
                        document_type TEXT,
                        domain TEXT,
                        field_code TEXT,
                        metadata JSONB,
                        embedding TEXT
                    ) ON COMMIT DROP
                """)

                # COPY records to staging table
                await conn.copy_records_to_table(
                    "staging_rag_documents",
                    records=records,
                    columns=["id", "content", "document_type", "domain", "field_code", "metadata", "embedding"],
                )

                # UPSERT from staging to main table
                await conn.execute("""
                    INSERT INTO rag_documents (id, content, document_type, domain, field_code, metadata, embedding)
                    SELECT
                        s.id::uuid,
                        s.content,
                        s.document_type,
                        s.domain,
                        s.field_code,
                        s.metadata,
                        s.embedding::vector
                    FROM staging_rag_documents s
                    ON CONFLICT (domain, field_code) WHERE document_type = 'field_code' AND field_code IS NOT NULL
                    DO UPDATE SET
                        content = EXCLUDED.content,
                        metadata = EXCLUDED.metadata,
                        embedding = EXCLUDED.embedding,
                        updated_at = NOW()
                """)

        logger.debug(f"Bulk inserted {len(docs)} field codes")
        return doc_ids

    async def index_economic_indicators_batch(
        self,
        docs: list[EconomicIndicatorDocument],
        generate_embeddings: bool = True,
        batch_size: int = 50,
        checkpoint_id: str | None = None,
        progress_callback: ProgressCallback | None = None,
    ) -> list[str]:
        """
        Index multiple economic indicator documents in batch.

        Uses bulk insert with COPY protocol.
        Supports checkpoint/resume.

        Args:
            docs: List of economic indicator documents
            generate_embeddings: Whether to generate embeddings
            batch_size: Batch size for embedding generation
            checkpoint_id: Optional checkpoint ID
            progress_callback: Optional progress callback

        Returns:
            List of document IDs
        """
        if not docs:
            return []

        # Create or resume checkpoint
        checkpoint: IndexingCheckpoint | None = None
        start_offset = 0

        if checkpoint_id:
            checkpoint = await self._checkpoint_manager.get_checkpoint(checkpoint_id)
            if checkpoint and checkpoint.is_resumable:
                start_offset = checkpoint.last_offset
                logger.info(
                    f"Resuming indexing from offset {start_offset} "
                    f"({checkpoint.processed_items}/{checkpoint.total_items} done)"
                )
        elif len(docs) > batch_size * 2:
            checkpoint = await self._checkpoint_manager.create_checkpoint(
                total_items=len(docs),
                domain="economic_indicators",
                batch_size=batch_size,
            )
            checkpoint_id = checkpoint.job_id
            logger.info(f"Created checkpoint {checkpoint_id} for {len(docs)} documents")

        doc_ids = []
        processed = start_offset

        try:
            for i in range(start_offset, len(docs), batch_size):
                batch = docs[i : i + batch_size]

                # Generate embeddings in batch
                embeddings = None
                if generate_embeddings and self._embedder:
                    contents = []
                    for doc in batch:
                        content = f"{doc.description} ({doc.country}, {doc.indicator_type})"
                        if doc.natural_language_hints:
                            content += " | Keywords: " + ", ".join(doc.natural_language_hints)
                        contents.append(content)
                    embeddings = await self._embedder.embed_batch(contents)

                # Bulk insert
                batch_doc_ids = [str(uuid.uuid4()) for _ in batch]
                records = []
                for j, doc in enumerate(batch):
                    content = f"{doc.description} ({doc.country}, {doc.indicator_type})"
                    if doc.natural_language_hints:
                        content += " | Keywords: " + ", ".join(doc.natural_language_hints)

                    embedding_str = None
                    if embeddings:
                        embedding_str = "[" + ",".join(str(x) for x in embeddings[j]) + "]"

                    records.append((
                        batch_doc_ids[j],
                        content,
                        DocumentType.ECONOMIC_INDICATOR.value,
                        "datastream",  # Economic indicators are primarily Datastream
                        doc.mnemonic,
                        json.dumps({
                            "country": doc.country,
                            "indicator_type": doc.indicator_type,
                            **(doc.metadata or {})
                        }),
                        embedding_str,
                    ))

                async with self._pool.acquire() as conn:
                    async with conn.transaction():
                        # Create temp staging table
                        await conn.execute("""
                            CREATE TEMP TABLE staging_econ_indicators (
                                id TEXT,
                                content TEXT,
                                document_type TEXT,
                                domain TEXT,
                                field_code TEXT,
                                metadata JSONB,
                                embedding TEXT
                            ) ON COMMIT DROP
                        """)

                        # COPY records
                        await conn.copy_records_to_table(
                            "staging_econ_indicators",
                            records=records,
                            columns=["id", "content", "document_type", "domain", "field_code", "metadata", "embedding"],
                        )

                        # UPSERT
                        await conn.execute("""
                            INSERT INTO rag_documents (id, content, document_type, domain, field_code, metadata, embedding)
                            SELECT
                                s.id::uuid,
                                s.content,
                                s.document_type,
                                s.domain,
                                s.field_code,
                                s.metadata,
                                s.embedding::vector
                            FROM staging_econ_indicators s
                            ON CONFLICT (domain, field_code) WHERE document_type = 'economic_indicator' AND field_code IS NOT NULL
                            DO UPDATE SET
                                content = EXCLUDED.content,
                                metadata = EXCLUDED.metadata,
                                embedding = EXCLUDED.embedding,
                                updated_at = NOW()
                        """)

                doc_ids.extend(batch_doc_ids)
                processed = i + len(batch)
                logger.info(f"Indexed batch of {len(batch)} economic indicators ({processed}/{len(docs)})")

                if checkpoint_id:
                    await self._checkpoint_manager.update_progress(
                        checkpoint_id, processed, processed
                    )

                if progress_callback:
                    progress_callback(processed, len(docs))

            if checkpoint_id:
                await self._checkpoint_manager.mark_completed(checkpoint_id)

        except Exception as e:
            if checkpoint_id:
                await self._checkpoint_manager.mark_failed(checkpoint_id, str(e))
            raise

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
                    json.dumps({"complexity_level": doc.complexity_level, **(doc.metadata or {})}),
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
                    json.dumps({"complexity_level": doc.complexity_level, **(doc.metadata or {})}),
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
    Parse estimates.md to extract field codes.

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
            metadata={"source": "estimates.md"},
        ))

    return docs


def parse_datastream_reference(content: str) -> list[FieldCodeDocument]:
    """
    Parse datastream.md to extract field codes.

    Args:
        content: Markdown content of the reference file

    Returns:
        List of FieldCodeDocument objects
    """
    docs = []

    # Pattern to match table rows with field codes
    # Format: | Natural Language | Field Code | Description |
    table_pattern = r'\|\s*([^|]+)\s*\|\s*`([^`]+)`\s*\|\s*([^|]+)\s*\|'

    for match in re.finditer(table_pattern, content):
        natural_lang = match.group(1).strip()
        field_code = match.group(2).strip()
        description = match.group(3).strip()

        # Skip header rows and non-data rows
        if natural_lang.lower() in ("natural language", "---", "metric", "expression", "interface", "universe", "operator", "option", "parameter"):
            continue
        if field_code.lower() in ("field code", "wc code", "tr code", "---"):
            continue
        if "varies by" in field_code.lower():
            continue

        # Extract keywords from natural language hints
        keywords = [kw.strip() for kw in natural_lang.split(",")]

        docs.append(FieldCodeDocument(
            field_code=field_code,
            description=description,
            domain="datastream",
            natural_language_hints=keywords,
            metadata={"source": "datastream.md"},
        ))

    return docs


def parse_fundamentals_reference(content: str) -> list[FieldCodeDocument]:
    """
    Parse fundamentals.md to extract field codes.

    Args:
        content: Markdown content of the reference file

    Returns:
        List of FieldCodeDocument objects
    """
    docs = []

    # Pattern to match table rows with WC or TR codes
    table_pattern = r'\|\s*([^|]+)\s*\|\s*`([^`]+)`\s*\|\s*([^|]+)\s*\|'

    for match in re.finditer(table_pattern, content):
        natural_lang = match.group(1).strip()
        field_code = match.group(2).strip()
        description = match.group(3).strip()

        # Skip header rows
        if natural_lang.lower() in ("natural language", "---", "metric", "expression"):
            continue
        if field_code.lower() in ("wc code", "tr code", "---"):
            continue

        keywords = [kw.strip() for kw in natural_lang.split(",")]

        docs.append(FieldCodeDocument(
            field_code=field_code,
            description=description,
            domain="fundamentals",
            natural_language_hints=keywords,
            metadata={"source": "fundamentals.md"},
        ))

    return docs


def parse_officers_reference(content: str) -> list[FieldCodeDocument]:
    """
    Parse officers-directors.md to extract field codes.

    Args:
        content: Markdown content of the reference file

    Returns:
        List of FieldCodeDocument objects
    """
    docs = []

    # Pattern to match table rows with TR codes
    table_pattern = r'\|\s*([^|]+)\s*\|\s*`([^`]+)`\s*\|\s*([^|]+)\s*\|'

    for match in re.finditer(table_pattern, content):
        natural_lang = match.group(1).strip()
        field_code = match.group(2).strip()
        description = match.group(3).strip()

        # Skip header rows
        if natural_lang.lower() in ("natural language", "---", "metric"):
            continue
        if field_code.lower() in ("tr code", "---"):
            continue
        # Skip calculated fields
        if field_code.lower() == "calculated":
            continue

        keywords = [kw.strip() for kw in natural_lang.split(",")]

        docs.append(FieldCodeDocument(
            field_code=field_code,
            description=description,
            domain="officers",
            natural_language_hints=keywords,
            metadata={"source": "officers-directors.md"},
        ))

    return docs


def parse_screening_reference(content: str) -> list[FieldCodeDocument]:
    """
    Parse screening.md to extract field codes and SCREEN syntax.

    Args:
        content: Markdown content of the reference file

    Returns:
        List of FieldCodeDocument objects
    """
    docs = []

    # Pattern to match table rows with expressions/codes
    table_pattern = r'\|\s*([^|]+)\s*\|\s*`([^`]+)`\s*\|\s*([^|]+)\s*\|'

    for match in re.finditer(table_pattern, content):
        natural_lang = match.group(1).strip()
        field_code = match.group(2).strip()
        description = match.group(3).strip()

        # Skip header rows
        if natural_lang.lower() in ("natural language", "---", "universe", "operator", "option", "parameter", "query type"):
            continue
        if field_code.lower() in ("expression", "syntax", "example", "---"):
            continue

        keywords = [kw.strip() for kw in natural_lang.split(",")]

        docs.append(FieldCodeDocument(
            field_code=field_code,
            description=description,
            domain="screening",
            natural_language_hints=keywords,
            metadata={"source": "screening.md"},
        ))

    return docs


def parse_query_examples(content: str) -> list[QueryExampleDocument]:
    """
    Parse estimates.md to extract query examples.

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
            metadata={"source": "estimates.md"},
        ))

    return docs


def create_progress_callback(
    progress: Any,
    task_id: Any,
) -> ProgressCallback:
    """
    Create a progress callback for Rich progress bar.

    Args:
        progress: Rich Progress instance
        task_id: Task ID from progress.add_task()

    Returns:
        Callback function compatible with index_field_codes_batch

    Example:
        from rich.progress import Progress

        with Progress() as progress:
            task = progress.add_task("Indexing...", total=len(docs))
            callback = create_progress_callback(progress, task)
            await indexer.index_field_codes_batch(docs, progress_callback=callback)
    """
    def callback(processed: int, total: int) -> None:
        progress.update(task_id, completed=processed, total=total)
    return callback


async def index_with_rich_progress(
    indexer: RAGIndexer,
    docs: list[FieldCodeDocument],
    title: str = "Indexing documents",
    batch_size: int = 100,
    checkpoint_id: str | None = None,
) -> list[str]:
    """
    Index documents with Rich progress bar display.

    Convenience function that wraps index_field_codes_batch with Rich progress.

    Args:
        indexer: RAGIndexer instance
        docs: Documents to index
        title: Progress bar title
        batch_size: Batch size for indexing
        checkpoint_id: Optional checkpoint ID for resume

    Returns:
        List of document IDs

    Example:
        async with asyncpg.create_pool(url) as pool:
            indexer = RAGIndexer(pool)
            indexer.set_embedder(OpenAIEmbedder(api_key))
            doc_ids = await index_with_rich_progress(indexer, docs)
    """
    from rich.progress import (
        BarColumn,
        MofNCompleteColumn,
        Progress,
        SpinnerColumn,
        TaskProgressColumn,
        TextColumn,
        TimeElapsedColumn,
        TimeRemainingColumn,
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
    ) as progress:
        task = progress.add_task(f"[cyan]{title}", total=len(docs))
        callback = create_progress_callback(progress, task)

        return await indexer.index_field_codes_batch(
            docs,
            batch_size=batch_size,
            checkpoint_id=checkpoint_id,
            progress_callback=callback,
        )
