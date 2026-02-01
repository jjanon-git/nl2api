#!/usr/bin/env python3
"""
Index Field Codes into RAG Database

Parses all reference documents and indexes field codes into the
rag_documents table for retrieval during query processing.

Usage:
    # Index all domains
    .venv/bin/python scripts/index_field_codes.py

    # Index specific domain
    .venv/bin/python scripts/index_field_codes.py --domain datastream

    # Dry run - show what would be indexed
    .venv/bin/python scripts/index_field_codes.py --dry-run

    # Clear and re-index
    .venv/bin/python scripts/index_field_codes.py --clear
"""

import argparse
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_env():
    """Load environment variables from .env file."""
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value


_load_env()


# Reference document paths and their parsers
# Files are in docs/api-reference/
REFERENCE_DOCS = {
    "datastream": ("docs/api-reference/datastream.md", "parse_datastream_reference"),
    "fundamentals": ("docs/api-reference/fundamentals.md", "parse_fundamentals_reference"),
    "estimates": ("docs/api-reference/estimates.md", "parse_estimates_reference"),
    "officers": ("docs/api-reference/officers-directors.md", "parse_officers_reference"),
    "screening": ("docs/api-reference/screening.md", "parse_screening_reference"),
}


async def main():
    parser = argparse.ArgumentParser(description="Index field codes into RAG database")
    parser.add_argument(
        "--domain", type=str, choices=list(REFERENCE_DOCS.keys()), help="Index specific domain only"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be indexed without actually indexing",
    )
    parser.add_argument(
        "--clear", action="store_true", help="Clear existing documents before indexing"
    )
    parser.add_argument(
        "--batch-size", type=int, default=50, help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--no-embeddings",
        action="store_true",
        help="Skip embedding generation (faster, but no vector search)",
    )
    args = parser.parse_args()

    # Import here to avoid circular imports
    from src.rag.retriever import indexer as indexer_module
    from src.rag.retriever.indexer import (
        FieldCodeDocument,
        RAGIndexer,
        index_with_rich_progress,
    )

    print("=" * 60)
    print("FIELD CODE INDEXER")
    print("=" * 60)

    # Determine which domains to index
    domains = [args.domain] if args.domain else list(REFERENCE_DOCS.keys())
    print(f"Domains: {', '.join(domains)}")
    print()

    # Parse all reference documents
    all_docs: list[FieldCodeDocument] = []

    for domain in domains:
        filename, parser_name = REFERENCE_DOCS[domain]
        filepath = PROJECT_ROOT / filename

        if not filepath.exists():
            print(f"WARNING: {filename} not found, skipping {domain}")
            continue

        # Get the parser function
        parser_func = getattr(indexer_module, parser_name)

        # Parse the document
        content = filepath.read_text()
        docs = parser_func(content)

        print(f"{domain:15} {len(docs):4} field codes from {filename}")
        all_docs.extend(docs)

    print(f"\nTotal: {len(all_docs)} field codes")

    if args.dry_run:
        print("\n[DRY RUN] Sample documents:")
        for doc in all_docs[:10]:
            print(f"  - {doc.field_code:20} | {doc.description[:40]}...")
            print(f"    Keywords: {', '.join(doc.natural_language_hints[:3])}")
        return

    if not all_docs:
        print("No documents to index!")
        return

    # Check embedding provider
    embedding_provider = os.environ.get("EMBEDDING_PROVIDER", "local")
    if not args.no_embeddings and embedding_provider == "openai":
        openai_key = os.environ.get("OPENAI_API_KEY")
        if not openai_key:
            print("ERROR: OPENAI_API_KEY not set!")
            print("  Set EMBEDDING_PROVIDER=local to use local embeddings, or")
            print("  Set OPENAI_API_KEY to use OpenAI embeddings")
            sys.exit(1)

    # Connect to database
    import asyncpg

    db_url = os.environ.get("DATABASE_URL", "postgresql://nl2api:nl2api@localhost:5432/nl2api")

    print("\nConnecting to database...")
    pool = await asyncpg.create_pool(db_url)

    try:
        # Create indexer
        indexer = RAGIndexer(pool)

        # Set up embedder if generating embeddings
        if not args.no_embeddings:
            from src.rag.retriever.embedders import create_embedder

            if embedding_provider == "openai":
                embedder = create_embedder(
                    "openai",
                    api_key=os.environ["OPENAI_API_KEY"],
                )
                print("Embedder initialized (OpenAI text-embedding-3-small, 1536 dims)")
            else:
                model_name = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
                embedder = create_embedder(
                    "local",
                    model_name=model_name,
                )
                print(f"Embedder initialized (local: {model_name}, {embedder.dimension} dims)")

            indexer.set_embedder(embedder)

        # Clear if requested
        if args.clear:
            print("\nClearing existing documents...")
            for domain in domains:
                count = await indexer.clear_domain(domain)
                print(f"  Cleared {count} documents from {domain}")

        # Index documents with progress bar
        print(f"\nIndexing {len(all_docs)} documents...")
        doc_ids = await index_with_rich_progress(
            indexer,
            all_docs,
            title="Indexing field codes",
            batch_size=args.batch_size,
        )

        print(f"\nIndexed {len(doc_ids)} documents")

        # Show stats
        print("\nDatabase stats:")
        stats = await indexer.get_stats()
        for domain, type_stats in sorted(stats.items()):
            for doc_type, counts in type_stats.items():
                print(
                    f"  {domain:15} {doc_type:15} {counts['count']:4} docs ({counts['with_embedding']} with embeddings)"
                )

    finally:
        await pool.close()

    print("\nDone!")


if __name__ == "__main__":
    asyncio.run(main())
