#!/usr/bin/env python3
"""
Re-embed existing RAG documents using OpenAI embeddings.

This script regenerates embeddings for all documents in rag_documents table
using OpenAI's text-embedding-3-small model (1536 dimensions).

Usage:
    python scripts/reembed_with_openai.py                    # All documents
    python scripts/reembed_with_openai.py --limit 100        # First 100 only
    python scripts/reembed_with_openai.py --batch-size 50    # Custom batch size
    python scripts/reembed_with_openai.py --dry-run          # Count only
"""

import argparse
import asyncio
import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from dotenv import load_dotenv

from src.rag.retriever.embedders import OpenAIEmbedder

# Load environment
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def get_documents_without_embeddings(
    pool: asyncpg.Pool,
    limit: int | None = None,
) -> list[dict]:
    """Get documents that need embedding."""
    query = """
        SELECT id, content
        FROM rag_documents
        WHERE embedding IS NULL
        ORDER BY created_at
    """
    if limit:
        query += f" LIMIT {limit}"

    async with pool.acquire() as conn:
        rows = await conn.fetch(query)
        return [{"id": str(row["id"]), "content": row["content"]} for row in rows]


async def update_embeddings_batch(
    pool: asyncpg.Pool,
    ids: list[str],
    embeddings: list[list[float]],
) -> int:
    """Update embeddings for a batch of documents."""
    async with pool.acquire() as conn:
        # Use a prepared statement for efficiency
        updated = 0
        for doc_id, embedding in zip(ids, embeddings):
            embedding_str = "[" + ",".join(str(x) for x in embedding) + "]"
            result = await conn.execute(
                """
                UPDATE rag_documents
                SET embedding = $1::vector
                WHERE id = $2::uuid
                """,
                embedding_str,
                doc_id,
            )
            if "UPDATE 1" in result:
                updated += 1
        return updated


async def count_documents(pool: asyncpg.Pool) -> dict:
    """Get document counts."""
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            """
            SELECT
                COUNT(*) as total,
                COUNT(*) FILTER (WHERE embedding IS NULL) as needs_embedding,
                COUNT(*) FILTER (WHERE embedding IS NOT NULL) as has_embedding
            FROM rag_documents
            """
        )
        return dict(row)


async def main(args: argparse.Namespace) -> None:
    """Main entry point."""
    # Validate API key
    api_key = os.getenv("NL2API_OPENAI_API_KEY")
    if not api_key:
        logger.error("NL2API_OPENAI_API_KEY not set in environment")
        sys.exit(1)

    # Connect to database
    postgres_url = os.getenv(
        "NL2API_POSTGRES_URL",
        "postgresql://nl2api:nl2api@localhost:5432/nl2api",
    )
    pool = await asyncpg.create_pool(postgres_url, min_size=2, max_size=10)

    try:
        # Get counts
        counts = await count_documents(pool)
        logger.info(
            f"Documents: {counts['total']} total, "
            f"{counts['needs_embedding']} need embedding, "
            f"{counts['has_embedding']} have embedding"
        )

        if args.dry_run:
            logger.info("Dry run - no changes made")
            return

        if counts["needs_embedding"] == 0:
            logger.info("All documents already have embeddings")
            return

        # Create OpenAI embedder
        embedder = OpenAIEmbedder(
            api_key=api_key,
            model="text-embedding-3-small",
            max_concurrent=5,
        )
        logger.info(f"Using OpenAI embedder: text-embedding-3-small ({embedder.dimension} dims)")

        # Get documents to embed
        limit = args.limit or counts["needs_embedding"]
        docs = await get_documents_without_embeddings(pool, limit=limit)
        logger.info(f"Processing {len(docs)} documents")

        # Process in batches
        batch_size = args.batch_size
        total_processed = 0
        start_time = time.time()

        for i in range(0, len(docs), batch_size):
            batch = docs[i : i + batch_size]
            batch_ids = [d["id"] for d in batch]
            batch_contents = [d["content"] for d in batch]

            try:
                # Generate embeddings
                embeddings = await embedder.embed_batch(batch_contents)

                # Update database
                updated = await update_embeddings_batch(pool, batch_ids, embeddings)
                total_processed += updated

                # Log progress
                elapsed = time.time() - start_time
                rate = total_processed / elapsed if elapsed > 0 else 0
                eta = (len(docs) - total_processed) / rate if rate > 0 else 0

                logger.info(
                    f"Progress: {total_processed}/{len(docs)} "
                    f"({100 * total_processed / len(docs):.1f}%) | "
                    f"Rate: {rate:.1f}/s | "
                    f"ETA: {eta / 60:.1f} min"
                )

            except Exception as e:
                logger.error(f"Error processing batch {i}-{i + batch_size}: {e}")
                # Continue with next batch
                await asyncio.sleep(5)  # Back off on error

        # Final stats
        elapsed = time.time() - start_time
        stats = embedder.stats
        logger.info(f"\nCompleted: {total_processed} documents in {elapsed / 60:.1f} minutes")
        logger.info(f"Embedder stats: {stats}")

        # Verify
        final_counts = await count_documents(pool)
        logger.info(
            f"Final: {final_counts['has_embedding']} with embeddings, "
            f"{final_counts['needs_embedding']} remaining"
        )

    finally:
        await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Re-embed RAG documents with OpenAI")
    parser.add_argument(
        "--limit",
        type=int,
        help="Limit number of documents to process",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100,
        help="Batch size for embedding (default: 100)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Just count documents, don't embed",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
