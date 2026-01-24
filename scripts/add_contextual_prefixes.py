#!/usr/bin/env python3
"""
Add contextual prefixes to existing RAG documents and re-embed.

This script:
1. Reads existing chunks from rag_documents table
2. Prepends context (company, filing type, section) to content
3. Updates the content and clears the embedding
4. Re-embeds with OpenAI embeddings

Run with: python scripts/add_contextual_prefixes.py
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


# Section name mappings for human-readable context
SECTION_LABELS = {
    "business": "Business Description",
    "mda": "Management's Discussion and Analysis",
    "risk_factors": "Risk Factors",
    "properties": "Properties",
    "legal_proceedings": "Legal Proceedings",
    "controls_procedures": "Controls and Procedures",
    "financial_statements": "Financial Statements",
    "exhibits": "Exhibits and Financial Statement Schedules",
}


def generate_context_prefix(metadata: dict) -> str:
    """Generate context prefix from chunk metadata."""
    company_name = metadata.get("company_name", "Unknown Company")
    ticker = metadata.get("ticker", "UNKN")
    filing_type = metadata.get("filing_type", "SEC Filing")
    section = metadata.get("section", "unknown")
    period_of_report = metadata.get("period_of_report", "")
    fiscal_quarter = metadata.get("fiscal_quarter")
    fiscal_year = metadata.get("fiscal_year")

    section_label = SECTION_LABELS.get(section, section.title() if section else "Unknown")

    # Format period
    if "10-K" in filing_type:
        period_str = f"Fiscal Year {fiscal_year}" if fiscal_year else ""
    elif fiscal_quarter and fiscal_year:
        period_str = f"Q{fiscal_quarter} {fiscal_year}"
    elif period_of_report:
        try:
            period_date = datetime.fromisoformat(period_of_report)
            period_str = period_date.strftime("%B %Y")
        except ValueError:
            period_str = period_of_report
    else:
        period_str = ""

    # Build context prefix
    context_lines = [
        f"Company: {company_name} ({ticker})",
        f"Filing: {filing_type}, {period_str}" if period_str else f"Filing: {filing_type}",
        f"Section: {section_label}",
        "",  # Blank line before content
    ]

    return "\n".join(context_lines)


async def count_documents(pool: asyncpg.Pool, doc_type: str = "sec_filing") -> dict:
    """Count documents with and without contextual prefixes."""
    async with pool.acquire() as conn:
        # Count total
        total = await conn.fetchval(
            """
            SELECT COUNT(*) FROM rag_documents WHERE document_type = $1
        """,
            doc_type,
        )

        # Count documents that already have context prefix
        with_prefix = await conn.fetchval(
            """
            SELECT COUNT(*) FROM rag_documents
            WHERE document_type = $1
              AND content LIKE 'Company:%'
        """,
            doc_type,
        )

        # Count documents with embeddings
        with_embedding = await conn.fetchval(
            """
            SELECT COUNT(*) FROM rag_documents
            WHERE document_type = $1
              AND embedding IS NOT NULL
        """,
            doc_type,
        )

        return {
            "total": total,
            "with_prefix": with_prefix,
            "without_prefix": total - with_prefix,
            "with_embedding": with_embedding,
        }


async def add_contextual_prefixes(
    pool: asyncpg.Pool,
    batch_size: int = 1000,
    dry_run: bool = False,
) -> int:
    """Add contextual prefixes to documents that don't have them."""
    updated = 0

    async with pool.acquire() as conn:
        # Get documents without context prefix
        offset = 0
        while True:
            rows = await conn.fetch(
                """
                SELECT id, content, metadata
                FROM rag_documents
                WHERE document_type = 'sec_filing'
                  AND content NOT LIKE 'Company:%'
                ORDER BY id
                LIMIT $1 OFFSET $2
            """,
                batch_size,
                offset,
            )

            if not rows:
                break

            if dry_run:
                logger.info(f"Would update {len(rows)} documents (dry run)")
                offset += batch_size
                continue

            # Update each document
            for row in rows:
                doc_id = row["id"]
                content = row["content"]
                metadata = (
                    row["metadata"]
                    if isinstance(row["metadata"], dict)
                    else json.loads(row["metadata"])
                )

                # Generate and prepend context prefix
                prefix = generate_context_prefix(metadata)
                new_content = prefix + content

                # Update document and clear embedding (needs re-embedding)
                await conn.execute(
                    """
                    UPDATE rag_documents
                    SET content = $2,
                        embedding = NULL,
                        metadata = metadata || '{"contextual_chunking": true}'::jsonb
                    WHERE id = $1
                """,
                    doc_id,
                    new_content,
                )

                updated += 1

            logger.info(f"Updated {updated} documents...")
            offset += batch_size

    return updated


async def main(args: argparse.Namespace):
    """Main entry point."""
    postgres_url = os.getenv(
        "NL2API_POSTGRES_URL",
        "postgresql://nl2api:nl2api@localhost:5432/nl2api",
    )
    pool = await asyncpg.create_pool(postgres_url, min_size=2, max_size=10)

    try:
        # Check current state
        counts = await count_documents(pool)
        logger.info("Current state:")
        logger.info(f"  Total SEC filing documents: {counts['total']}")
        logger.info(f"  Already have context prefix: {counts['with_prefix']}")
        logger.info(f"  Need context prefix: {counts['without_prefix']}")
        logger.info(f"  Have embeddings: {counts['with_embedding']}")

        if counts["without_prefix"] == 0:
            logger.info("All documents already have context prefixes!")
            return

        if args.dry_run:
            logger.info(
                "\n[DRY RUN] Would add context prefixes to {} documents".format(
                    counts["without_prefix"]
                )
            )
            # Show example
            async with pool.acquire() as conn:
                row = await conn.fetchrow("""
                    SELECT content, metadata FROM rag_documents
                    WHERE document_type = 'sec_filing'
                      AND content NOT LIKE 'Company:%'
                    LIMIT 1
                """)
                if row:
                    metadata = (
                        row["metadata"]
                        if isinstance(row["metadata"], dict)
                        else json.loads(row["metadata"])
                    )
                    prefix = generate_context_prefix(metadata)
                    logger.info("\nExample prefix that would be added:")
                    logger.info("-" * 40)
                    logger.info(prefix)
                    logger.info("-" * 40)
            return

        # Confirm before proceeding
        if not args.yes:
            response = input(
                f"\nAdd context prefixes to {counts['without_prefix']} documents? (y/N): "
            )
            if response.lower() != "y":
                logger.info("Aborted.")
                return

        # Add prefixes
        logger.info(f"\nAdding context prefixes to {counts['without_prefix']} documents...")
        updated = await add_contextual_prefixes(
            pool,
            batch_size=args.batch_size,
            dry_run=args.dry_run,
        )

        logger.info(f"\nCompleted! Updated {updated} documents.")
        logger.info(
            "Note: Embeddings have been cleared. Run scripts/reembed_with_openai.py to regenerate."
        )

        # Show final state
        final_counts = await count_documents(pool)
        logger.info("\nFinal state:")
        logger.info(f"  Total SEC filing documents: {final_counts['total']}")
        logger.info(f"  Have context prefix: {final_counts['with_prefix']}")
        logger.info(f"  Have embeddings: {final_counts['with_embedding']}")

    finally:
        await pool.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Add contextual prefixes to RAG documents")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1000,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without making changes",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompt",
    )

    args = parser.parse_args()
    asyncio.run(main(args))
