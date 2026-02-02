"""
SEC Filing RAG Indexer

Indexes filing chunks into the RAG system using existing infrastructure.
"""

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

from src.evalkit.common.telemetry import get_tracer
from src.rag.ingestion.sec_filings.models import FilingChunk
from src.rag.retriever.protocols import DocumentType

if TYPE_CHECKING:
    import asyncpg

    from src.rag.retriever.embedders import Embedder

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class FilingRAGIndexer:
    """
    Indexes SEC filing chunks into the RAG system.

    Uses the existing RAG infrastructure with bulk insert optimization.
    Supports parallel embedding workers for faster throughput.
    """

    def __init__(
        self,
        pool: "asyncpg.Pool",
        embedder: "Embedder | None" = None,
        batch_size: int = 50,
        embedding_workers: int = 1,
    ):
        """
        Initialize filing indexer.

        Args:
            pool: asyncpg connection pool
            embedder: Embedder for generating vectors (uses local if not provided)
            batch_size: Batch size for embedding generation (chunks per API call)
            embedding_workers: Number of parallel embedding API calls (default 1)
        """
        self._pool = pool
        self._embedder = embedder
        self._batch_size = batch_size
        self._embedding_workers = max(1, embedding_workers)

    def set_embedder(self, embedder: "Embedder") -> None:
        """Set the embedder for generating embeddings."""
        self._embedder = embedder

    async def _ensure_embedder(self) -> "Embedder":
        """Ensure embedder is available, creating default if needed."""
        if self._embedder is None:
            # Import here to avoid circular dependency
            from src.rag.retriever.embedders import create_embedder

            self._embedder = create_embedder("local")
            logger.info("Created default local embedder (384 dimensions)")
        return self._embedder

    async def index_chunks(
        self,
        chunks: list[FilingChunk],
        filing_accession: str | None = None,
    ) -> list[str]:
        """
        Index filing chunks into RAG system.

        Args:
            chunks: List of FilingChunk objects to index
            filing_accession: Optional accession number for logging

        Returns:
            List of document IDs
        """
        if not chunks:
            return []

        with tracer.start_as_current_span("sec_filing.index_chunks") as span:
            span.set_attribute("sec.chunk_count", len(chunks))
            span.set_attribute("sec.embedding_workers", self._embedding_workers)
            if filing_accession:
                span.set_attribute("sec.accession_number", filing_accession)

            embedder = await self._ensure_embedder()
            doc_ids = []

            # Split chunks into batches
            batches = []
            for i in range(0, len(chunks), self._batch_size):
                batch = chunks[i : i + self._batch_size]
                contents = [self._build_content(chunk) for chunk in batch]
                batches.append((batch, contents))

            # Process batches with parallel workers
            if self._embedding_workers > 1:
                import asyncio

                semaphore = asyncio.Semaphore(self._embedding_workers)

                async def process_batch(batch_data: tuple) -> list[str]:
                    batch, contents = batch_data
                    async with semaphore:
                        embeddings = await embedder.embed_batch(contents)
                        return await self._bulk_insert_chunks(batch, contents, embeddings)

                results = await asyncio.gather(*[process_batch(b) for b in batches])
                for batch_ids in results:
                    doc_ids.extend(batch_ids)
            else:
                # Sequential processing (original behavior)
                for batch, contents in batches:
                    embeddings = await embedder.embed_batch(contents)
                    batch_ids = await self._bulk_insert_chunks(batch, contents, embeddings)
                    doc_ids.extend(batch_ids)

                    logger.debug(
                        f"Indexed batch of {len(batch)} chunks ({len(doc_ids)}/{len(chunks)})"
                    )

            span.set_attribute("sec.indexed_count", len(doc_ids))
            logger.info(f"Indexed {len(doc_ids)} chunks for filing {filing_accession or 'unknown'}")

            return doc_ids

    def _build_content(self, chunk: FilingChunk) -> str:
        """
        Build content string for embedding.

        Args:
            chunk: Filing chunk

        Returns:
            Content string for embedding
        """
        # Include metadata context in the content for better retrieval
        metadata = chunk.metadata
        company = metadata.get("company_name", "")
        ticker = metadata.get("ticker", "")
        section = metadata.get("section", chunk.section)
        filing_type = metadata.get("filing_type", "")
        filing_date = metadata.get("filing_date", "")

        # Build context prefix
        context_parts = []
        if company:
            context_parts.append(company)
        if ticker:
            context_parts.append(f"({ticker})")
        if filing_type:
            context_parts.append(filing_type)
        if filing_date:
            context_parts.append(filing_date[:10])  # Just the date part
        if section:
            context_parts.append(f"Section: {section}")

        context = " | ".join(context_parts)

        # Combine context with content
        if context:
            return f"{context}\n\n{chunk.content}"
        return chunk.content

    async def _bulk_insert_chunks(
        self,
        chunks: list[FilingChunk],
        contents: list[str],
        embeddings: list[list[float]],
    ) -> list[str]:
        """
        Bulk insert chunks using PostgreSQL COPY protocol.

        Args:
            chunks: Filing chunks
            contents: Content strings (with context)
            embeddings: Embedding vectors

        Returns:
            List of document IDs
        """
        doc_ids = [str(uuid.uuid4()) for _ in chunks]

        # Prepare records for COPY
        records = []
        for i, chunk in enumerate(chunks):
            # Convert embedding to PostgreSQL array format
            embedding_str = "[" + ",".join(str(x) for x in embeddings[i]) + "]"

            # Build metadata JSONB
            metadata = {
                **chunk.metadata,
                "chunk_id": chunk.chunk_id,
                "section": chunk.section if isinstance(chunk.section, str) else chunk.section,
                "chunk_index": chunk.chunk_index,
                "char_start": chunk.char_start,
                "char_end": chunk.char_end,
            }

            records.append(
                (
                    doc_ids[i],
                    contents[i],
                    DocumentType.SEC_FILING.value,
                    "sec_filings",  # domain
                    chunk.filing_accession,  # Use accession_number as field_code for uniqueness
                    json.dumps(metadata),
                    embedding_str,
                )
            )

        async with self._pool.acquire() as conn:
            async with conn.transaction():
                # Create temp staging table
                await conn.execute("""
                    CREATE TEMP TABLE staging_sec_chunks (
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
                    "staging_sec_chunks",
                    records=records,
                    columns=[
                        "id",
                        "content",
                        "document_type",
                        "domain",
                        "field_code",
                        "metadata",
                        "embedding",
                    ],
                )

                # INSERT from staging to main table
                # Don't use UPSERT since each chunk is unique by chunk_id in metadata
                await conn.execute("""
                    INSERT INTO rag_documents (id, content, document_type, domain, field_code, metadata, embedding)
                    SELECT
                        id::uuid,
                        content,
                        document_type,
                        domain,
                        field_code,
                        metadata,
                        embedding::vector
                    FROM staging_sec_chunks
                """)

        return doc_ids

    async def delete_filing_chunks(self, accession_number: str) -> int:
        """
        Delete all chunks for a specific filing.

        Args:
            accession_number: Filing accession number

        Returns:
            Number of chunks deleted
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM rag_documents
                WHERE document_type = $1
                  AND metadata->>'accession_number' = $2
                """,
                DocumentType.SEC_FILING.value,
                accession_number,
            )
            count = int(result.split()[-1])
            logger.info(f"Deleted {count} chunks for filing {accession_number}")
            return count

    async def delete_company_chunks(self, cik: str) -> int:
        """
        Delete all chunks for a specific company.

        Args:
            cik: Company CIK

        Returns:
            Number of chunks deleted
        """
        async with self._pool.acquire() as conn:
            result = await conn.execute(
                """
                DELETE FROM rag_documents
                WHERE document_type = $1
                  AND metadata->>'cik' = $2
                """,
                DocumentType.SEC_FILING.value,
                cik,
            )
            count = int(result.split()[-1])
            logger.info(f"Deleted {count} chunks for company CIK {cik}")
            return count

    async def get_stats(self) -> dict[str, Any]:
        """Get SEC filing indexing statistics."""
        async with self._pool.acquire() as conn:
            # Overall stats
            overall = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_chunks,
                    COUNT(DISTINCT metadata->>'accession_number') as total_filings,
                    COUNT(DISTINCT metadata->>'cik') as total_companies
                FROM rag_documents
                WHERE document_type = $1
                """,
                DocumentType.SEC_FILING.value,
            )

            # Stats by section
            sections = await conn.fetch(
                """
                SELECT
                    metadata->>'section' as section,
                    COUNT(*) as chunk_count
                FROM rag_documents
                WHERE document_type = $1
                GROUP BY metadata->>'section'
                ORDER BY chunk_count DESC
                """,
                DocumentType.SEC_FILING.value,
            )

            return {
                "total_chunks": overall["total_chunks"],
                "total_filings": overall["total_filings"],
                "total_companies": overall["total_companies"],
                "by_section": {row["section"]: row["chunk_count"] for row in sections},
            }


class FilingMetadataRepo:
    """
    Repository for SEC filing metadata.

    Tracks filing ingestion status in the sec_filings table.
    """

    def __init__(self, pool: "asyncpg.Pool"):
        """Initialize repository."""
        self._pool = pool

    async def upsert_filing(
        self,
        accession_number: str,
        cik: str,
        ticker: str | None,
        company_name: str,
        filing_type: str,
        filing_date: Any,
        period_of_report: Any,
        status: str = "pending",
    ) -> None:
        """
        Insert or update filing metadata.

        Args:
            accession_number: Filing accession number
            cik: Company CIK
            ticker: Stock ticker
            company_name: Company name
            filing_type: Filing type (10-K, 10-Q)
            filing_date: Filing date
            period_of_report: Period of report
            status: Ingestion status
        """
        async with self._pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO sec_filings (
                    accession_number, cik, ticker, company_name,
                    filing_type, filing_date, period_of_report, status
                )
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                ON CONFLICT (accession_number) DO UPDATE SET
                    ticker = COALESCE(EXCLUDED.ticker, sec_filings.ticker),
                    status = EXCLUDED.status,
                    updated_at = NOW()
                """,
                accession_number,
                cik,
                ticker,
                company_name,
                filing_type,
                filing_date,
                period_of_report,
                status,
            )

    async def update_status(
        self,
        accession_number: str,
        status: str,
        error_message: str | None = None,
        download_path: str | None = None,
        sections_extracted: int | None = None,
        chunks_count: int | None = None,
    ) -> None:
        """
        Update filing status.

        Args:
            accession_number: Filing accession number
            status: New status
            error_message: Error message if failed
            download_path: Path to downloaded file
            sections_extracted: Number of sections extracted
            chunks_count: Number of chunks created
        """
        async with self._pool.acquire() as conn:
            # Build dynamic update
            updates = ["status = $2"]
            params: list[Any] = [accession_number, status]
            param_idx = 3

            if error_message is not None:
                updates.append(f"error_message = ${param_idx}")
                params.append(error_message)
                param_idx += 1

            if download_path is not None:
                updates.append(f"download_path = ${param_idx}")
                updates.append("downloaded_at = NOW()")
                params.append(download_path)
                param_idx += 1

            if sections_extracted is not None:
                updates.append(f"sections_extracted = ${param_idx}")
                updates.append("parsed_at = NOW()")
                params.append(sections_extracted)
                param_idx += 1

            if chunks_count is not None:
                updates.append(f"chunks_count = ${param_idx}")
                updates.append("indexed_at = NOW()")
                params.append(chunks_count)
                param_idx += 1

            query = f"""
                UPDATE sec_filings
                SET {", ".join(updates)}, updated_at = NOW()
                WHERE accession_number = $1
            """
            await conn.execute(query, *params)

    async def get_pending_filings(self, limit: int = 100) -> list[dict[str, Any]]:
        """
        Get filings pending processing.

        Args:
            limit: Maximum number to return

        Returns:
            List of filing records
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT *
                FROM sec_filings
                WHERE status = 'pending'
                ORDER BY filing_date DESC
                LIMIT $1
                """,
                limit,
            )
            return [dict(row) for row in rows]

    async def get_filings_for_company(self, cik: str) -> list[dict[str, Any]]:
        """
        Get all filings for a company.

        Args:
            cik: Company CIK

        Returns:
            List of filing records
        """
        async with self._pool.acquire() as conn:
            rows = await conn.fetch(
                """
                SELECT *
                FROM sec_filings
                WHERE cik = $1
                ORDER BY filing_date DESC
                """,
                cik,
            )
            return [dict(row) for row in rows]

    async def get_ingestion_summary(self) -> dict[str, Any]:
        """Get summary of filing ingestion status."""
        async with self._pool.acquire() as conn:
            row = await conn.fetchrow(
                """
                SELECT
                    COUNT(*) as total_filings,
                    COUNT(*) FILTER (WHERE status = 'complete') as completed,
                    COUNT(*) FILTER (WHERE status = 'pending') as pending,
                    COUNT(*) FILTER (WHERE status = 'failed') as failed,
                    COUNT(DISTINCT cik) as companies,
                    SUM(chunks_count) as total_chunks
                FROM sec_filings
                """
            )
            return dict(row) if row else {}
