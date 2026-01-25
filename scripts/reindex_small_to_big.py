#!/usr/bin/env python3
"""
Reindex RAG documents with small-to-big chunking.

Creates child chunks (512 chars) for parent chunks (4000 chars) of specified companies.
Generates embeddings for children using OpenAI text-embedding-3-small.

Usage:
    # Reindex all test companies (246 companies, ~7.5 GB)
    python scripts/reindex_small_to_big.py --test-companies

    # Reindex specific companies
    python scripts/reindex_small_to_big.py --tickers AAPL,MSFT,GOOGL

    # Dry run (no changes)
    python scripts/reindex_small_to_big.py --test-companies --dry-run

    # Resume from checkpoint
    python scripts/reindex_small_to_big.py --test-companies --resume
"""

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import asyncpg
from openai import AsyncOpenAI

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Child chunk parameters (from successful A/B test)
CHILD_CHUNK_SIZE = 512
CHILD_OVERLAP = 64

# Batch sizes
EMBEDDING_BATCH_SIZE = 100
DB_INSERT_BATCH_SIZE = 500

# Checkpoint file
CHECKPOINT_FILE = Path("/tmp/reindex_checkpoint.json")

# All 246 test company tickers
TEST_COMPANY_TICKERS = [
    "AAL",
    "AAPL",
    "ABT",
    "ACN",
    "ADBE",
    "ADI",
    "ADP",
    "ADSK",
    "AFL",
    "AFRM",
    "AGCO",
    "ALGM",
    "ALGN",
    "ALK",
    "ALL",
    "ALSN",
    "AMAT",
    "AMBA",
    "AMD",
    "AMGN",
    "AON",
    "AOSL",
    "APH",
    "ATOS",
    "ATNM",
    "AWI",
    "BA",
    "BCRX",
    "BDX",
    "BHVN",
    "BIIB",
    "BILL",
    "BLD",
    "BLDR",
    "BMI",
    "BMY",
    "BXP",
    "CAH",
    "CARR",
    "CB",
    "CC",
    "CCL",
    "CDNS",
    "CF",
    "CHTR",
    "CL",
    "CME",
    "CMI",
    "CMCSA",
    "CNMD",
    "COHR",
    "COIN",
    "COP",
    "COST",
    "CRM",
    "CRUS",
    "CSCO",
    "CVS",
    "CVX",
    "CW",
    "CZR",
    "DD",
    "DDOG",
    "DELL",
    "DIOD",
    "DIS",
    "DLR",
    "DOCU",
    "DOW",
    "DUK",
    "DXCM",
    "EA",
    "EMN",
    "EMR",
    "EOG",
    "EQR",
    "ESS",
    "ESTC",
    "EW",
    "EXP",
    "EXPE",
    "EXR",
    "FAST",
    "FBIN",
    "FDX",
    "FISV",
    "GMED",
    "GM",
    "GNRC",
    "GOOGL",
    "GS",
    "HD",
    "HLT",
    "HON",
    "HOLX",
    "HOOD",
    "HSIC",
    "HUM",
    "HUN",
    "IBRX",
    "ICE",
    "ICUI",
    "IFF",
    "INO",
    "INTU",
    "ITT",
    "JAZZ",
    "JBLU",
    "JCI",
    "JELD",
    "JNJ",
    "KEYS",
    "KMI",
    "KMT",
    "KO",
    "LC",
    "LECO",
    "LITE",
    "LIVN",
    "LLY",
    "LMAT",
    "LSCC",
    "LUV",
    "LVS",
    "LYFT",
    "MAA",
    "MAR",
    "MAS",
    "MASI",
    "MCD",
    "MDU",
    "MET",
    "META",
    "MHK",
    "MLM",
    "MMM",
    "MMSI",
    "MO",
    "MOS",
    "MPWR",
    "MSI",
    "MSFT",
    "MTCH",
    "MTD",
    "MWA",
    "NBIX",
    "NDSN",
    "NEE",
    "NET",
    "NFLX",
    "NSC",
    "NTAP",
    "NVAX",
    "NVDA",
    "NVCR",
    "NVT",
    "NWSA",
    "NXPI",
    "O",
    "OC",
    "OCGN",
    "OFIX",
    "OKE",
    "OLED",
    "ON",
    "OSK",
    "OSUR",
    "PANW",
    "PATH",
    "PAYX",
    "PEN",
    "PEP",
    "PFE",
    "PGR",
    "PII",
    "PLD",
    "PNC",
    "PODD",
    "POOL",
    "PPG",
    "PRLB",
    "PSA",
    "PTC",
    "QCOM",
    "RARE",
    "RCL",
    "REG",
    "REGN",
    "RRX",
    "RTX",
    "S",
    "SBAC",
    "SBUX",
    "SENS",
    "SHW",
    "SLAB",
    "SMTC",
    "SNPS",
    "SPXC",
    "STAA",
    "STE",
    "STX",
    "SWKS",
    "SYK",
    "SYNA",
    "T",
    "TEAM",
    "TEX",
    "TFX",
    "TGT",
    "THO",
    "TJX",
    "TOST",
    "TROX",
    "TRV",
    "TTWO",
    "TXN",
    "UAL",
    "UDR",
    "UNH",
    "UNP",
    "UPS",
    "V",
    "VICI",
    "VIAV",
    "VRTX",
    "VTRS",
    "VTR",
    "VZ",
    "WAB",
    "WBD",
    "WDC",
    "WEX",
    "WFC",
    "WGO",
    "WLK",
    "WSC",
    "WYNN",
    "XRAY",
    "XYL",
    "ZG",
    "ZM",
    "ZS",
    "ZTS",
]


@dataclass
class ChildChunk:
    """A child chunk derived from a parent."""

    id: str
    parent_id: str
    content: str
    metadata: dict
    document_type: str
    domain: str | None = None
    embedding: list[float] | None = None


def create_child_chunks(
    parent_id: str,
    parent_content: str,
    parent_metadata: dict,
    document_type: str,
    domain: str | None,
) -> list[ChildChunk]:
    """Split parent content into overlapping child chunks."""
    children = []
    text = parent_content

    start = 0
    child_index = 0

    while start < len(text):
        end = min(start + CHILD_CHUNK_SIZE, len(text))
        chunk_text = text[start:end]

        # Skip very short final chunks
        if len(chunk_text) < 50 and children:
            break

        child_meta = {**parent_metadata}
        child_meta["parent_id"] = parent_id
        child_meta["child_index"] = child_index
        child_meta["chunk_level"] = 1

        child = ChildChunk(
            id=str(uuid.uuid4()),
            parent_id=parent_id,
            content=chunk_text,
            metadata=child_meta,
            document_type=document_type,
            domain=domain,
        )
        children.append(child)

        start = end - CHILD_OVERLAP if end < len(text) else end
        child_index += 1

    return children


async def generate_embeddings(client: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    """Generate embeddings for a batch of texts."""
    response = await client.embeddings.create(
        model="text-embedding-3-small",
        input=texts,
    )
    return [item.embedding for item in response.data]


async def get_parents_needing_children(
    conn: asyncpg.Connection,
    tickers: list[str],
    processed_ids: set[str],
) -> list[dict]:
    """Get parent chunks that don't have children yet."""
    # Build ticker filter
    ticker_filter = ", ".join(f"'{t}'" for t in tickers)

    query = f"""
        SELECT id, content, metadata, document_type, domain
        FROM rag_documents
        WHERE chunk_level = 0
        AND metadata->>'ticker' IN ({ticker_filter})
        AND id NOT IN (
            SELECT DISTINCT parent_id
            FROM rag_documents
            WHERE parent_id IS NOT NULL
        )
        ORDER BY metadata->>'ticker', id
    """

    rows = await conn.fetch(query)

    # Filter out already processed (from checkpoint)
    results = []
    for row in rows:
        if str(row["id"]) in processed_ids:
            continue
        meta = row["metadata"]
        if isinstance(meta, str):
            meta = json.loads(meta)
        results.append(
            {
                "id": str(row["id"]),
                "content": row["content"],
                "metadata": meta,
                "document_type": row["document_type"],
                "domain": row["domain"],
            }
        )
    return results


async def insert_children(conn: asyncpg.Connection, children: list[ChildChunk]) -> int:
    """Insert child chunks into database."""
    if not children:
        return 0

    # Build values for bulk insert
    values = []
    for child in children:
        embedding_str = "[" + ",".join(str(x) for x in child.embedding) + "]"
        values.append(
            (
                uuid.UUID(child.id),
                child.content,
                child.document_type,
                child.domain,
                json.dumps(child.metadata),
                embedding_str,
                1,  # chunk_level
                uuid.UUID(child.parent_id),
            )
        )

    await conn.executemany(
        """
        INSERT INTO rag_documents (id, content, document_type, domain, metadata, embedding, chunk_level, parent_id)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6::vector, $7, $8)
    """,
        values,
    )

    return len(values)


def load_checkpoint() -> set[str]:
    """Load processed parent IDs from checkpoint."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            data = json.load(f)
            return set(data.get("processed_ids", []))
    return set()


def save_checkpoint(processed_ids: set[str]):
    """Save processed parent IDs to checkpoint."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump({"processed_ids": list(processed_ids)}, f)


async def main():
    parser = argparse.ArgumentParser(description="Reindex RAG documents with small-to-big chunking")
    parser.add_argument(
        "--test-companies", action="store_true", help="Reindex all 246 test companies"
    )
    parser.add_argument("--tickers", type=str, help="Comma-separated list of tickers to reindex")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done without making changes"
    )
    parser.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    args = parser.parse_args()

    # Determine tickers to process
    if args.test_companies:
        tickers = TEST_COMPANY_TICKERS
    elif args.tickers:
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
    else:
        parser.error("Must specify --test-companies or --tickers")

    logger.info(f"Processing {len(tickers)} companies")

    # Load checkpoint if resuming
    processed_ids = load_checkpoint() if args.resume else set()
    if processed_ids:
        logger.info(f"Resuming from checkpoint: {len(processed_ids)} parents already processed")

    # Connect to database
    db_url = os.environ.get("DATABASE_URL", "postgresql://nl2api:nl2api@localhost:5432/nl2api")
    conn = await asyncpg.connect(db_url)

    try:
        # Get parents needing children
        logger.info("Finding parents that need children...")
        parents = await get_parents_needing_children(conn, tickers, processed_ids)
        logger.info(f"Found {len(parents)} parents needing children")

        if args.dry_run:
            # Estimate children and space
            sample_size = min(100, len(parents))
            sample_parents = parents[:sample_size]
            total_children = 0
            for p in sample_parents:
                children = create_child_chunks(
                    p["id"], p["content"], p["metadata"], p["document_type"], p["domain"]
                )
                total_children += len(children)

            avg_children = total_children / sample_size if sample_size > 0 else 0
            estimated_total = int(len(parents) * avg_children)
            estimated_gb = estimated_total * 6.3 / 1024 / 1024  # KB to GB

            logger.info(
                f"DRY RUN - Would create ~{estimated_total:,} children (~{estimated_gb:.1f} GB)"
            )
            return

        # Initialize OpenAI client (only needed for actual run)
        openai_key = (
            os.environ.get("OPENAI_API_KEY")
            or os.environ.get("NL2API_OPENAI_API_KEY")
            or os.environ.get("RAG_UI_OPENAI_API_KEY")
        )
        if not openai_key:
            logger.error("OPENAI_API_KEY (or NL2API_OPENAI_API_KEY) not set")
            sys.exit(1)
        client = AsyncOpenAI(api_key=openai_key)

        # Process in batches
        total_children_created = 0
        total_parents_processed = 0
        start_time = time.time()

        for i in range(0, len(parents), EMBEDDING_BATCH_SIZE):
            batch_parents = parents[i : i + EMBEDDING_BATCH_SIZE]

            # Create children for this batch
            all_children = []
            for parent in batch_parents:
                children = create_child_chunks(
                    parent["id"],
                    parent["content"],
                    parent["metadata"],
                    parent["document_type"],
                    parent["domain"],
                )
                all_children.extend(children)

            if not all_children:
                continue

            # Generate embeddings
            texts = [c.content for c in all_children]
            logger.info(f"Generating embeddings for {len(texts)} children...")
            embeddings = await generate_embeddings(client, texts)

            for child, embedding in zip(all_children, embeddings):
                child.embedding = embedding

            # Insert into database
            async with conn.transaction():
                inserted = await insert_children(conn, all_children)
                total_children_created += inserted

            # Update checkpoint
            for parent in batch_parents:
                processed_ids.add(parent["id"])
            save_checkpoint(processed_ids)

            total_parents_processed += len(batch_parents)
            elapsed = time.time() - start_time
            rate = total_parents_processed / elapsed if elapsed > 0 else 0
            eta = (len(parents) - total_parents_processed) / rate if rate > 0 else 0

            logger.info(
                f"Progress: {total_parents_processed}/{len(parents)} parents "
                f"({total_children_created:,} children) - "
                f"{rate:.1f} parents/sec - ETA: {eta / 60:.1f} min"
            )

        elapsed = time.time() - start_time
        logger.info(
            f"Done! Created {total_children_created:,} children in {elapsed / 60:.1f} minutes"
        )

        # Clean up checkpoint
        if CHECKPOINT_FILE.exists():
            CHECKPOINT_FILE.unlink()

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
