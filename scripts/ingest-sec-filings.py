#!/usr/bin/env python
"""
SEC EDGAR Filing Ingestion Script

Downloads, parses, chunks, and indexes SEC 10-K and 10-Q filings for RAG evaluation.

Usage:
    # Full S&P 500 ingestion (2 years of filings)
    python scripts/ingest_sec_filings.py

    # Specific tickers
    python scripts/ingest_sec_filings.py --tickers AAPL,MSFT,GOOGL

    # Resume from checkpoint
    python scripts/ingest_sec_filings.py --resume

    # Dry run (count filings only)
    python scripts/ingest_sec_filings.py --dry-run

    # Limit number of companies
    python scripts/ingest_sec_filings.py --limit 10
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment variables from .env file
from dotenv import load_dotenv

load_dotenv(PROJECT_ROOT / ".env")

import asyncpg

from src.nl2api.ingestion.errors import DownloadError, IngestionAbortError
from src.rag.ingestion.sec_filings.chunker import DocumentChunker
from src.rag.ingestion.sec_filings.client import (
    SECEdgarClient,
    load_sp500_companies,
)
from src.rag.ingestion.sec_filings.config import SECFilingConfig
from src.rag.ingestion.sec_filings.indexer import FilingMetadataRepo, FilingRAGIndexer
from src.rag.ingestion.sec_filings.models import Filing, FilingCheckpoint, FilingType
from src.rag.ingestion.sec_filings.parser import FilingParser
from src.rag.retriever.embedders import create_embedder

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class CheckpointManager:
    """Manages filing ingestion checkpoints."""

    def __init__(self, checkpoint_dir: Path):
        self._checkpoint_dir = checkpoint_dir
        self._checkpoint_path = checkpoint_dir / "sec_filing_checkpoint.json"

    def save(self, checkpoint: FilingCheckpoint) -> None:
        """Save checkpoint atomically."""
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)
        temp_path = self._checkpoint_path.with_suffix(".tmp")

        with open(temp_path, "w") as f:
            json.dump(checkpoint.to_dict(), f, indent=2)

        os.replace(temp_path, self._checkpoint_path)
        logger.debug(
            f"Saved checkpoint: {checkpoint.companies_processed}/{checkpoint.total_companies} companies"
        )

    def load(self) -> FilingCheckpoint | None:
        """Load existing checkpoint."""
        if not self._checkpoint_path.exists():
            return None

        try:
            with open(self._checkpoint_path) as f:
                data = json.load(f)
            return FilingCheckpoint.from_dict(data)
        except (json.JSONDecodeError, KeyError, ValueError) as e:
            logger.warning(f"Could not load checkpoint: {e}")
            return None

    def delete(self) -> None:
        """Delete checkpoint file."""
        if self._checkpoint_path.exists():
            self._checkpoint_path.unlink()
            logger.info("Deleted checkpoint file")


async def process_filing(
    client: SECEdgarClient,
    filing: Filing,
    parser: FilingParser,
    chunker: DocumentChunker,
    indexer: FilingRAGIndexer,
    metadata_repo: FilingMetadataRepo | None,
    config: SECFilingConfig,
) -> int:
    """
    Process a single filing: download, parse, chunk, and index.

    Args:
        client: SEC EDGAR client
        filing: Filing to process
        parser: Filing parser
        chunker: Document chunker
        indexer: RAG indexer
        metadata_repo: Optional metadata repository
        config: Configuration

    Returns:
        Number of chunks indexed
    """
    try:
        # Download filing
        logger.info(
            f"Processing: {filing.company_name} ({filing.ticker}) - {filing.filing_type.value} {filing.filing_date.strftime('%Y-%m-%d')}"
        )

        if metadata_repo:
            await metadata_repo.upsert_filing(
                accession_number=filing.accession_number,
                cik=filing.cik,
                ticker=filing.ticker,
                company_name=filing.company_name,
                filing_type=filing.filing_type.value,
                filing_date=filing.filing_date,
                period_of_report=filing.period_of_report,
                status="downloading",
            )

        html_path = await client.download_filing(filing)

        if metadata_repo:
            await metadata_repo.update_status(
                filing.accession_number,
                status="parsing",
                download_path=str(html_path),
            )

        # Parse sections
        sections = parser.parse(html_path, filing)

        if not sections:
            logger.warning(f"No sections extracted from filing {filing.accession_number}")
            if metadata_repo:
                await metadata_repo.update_status(
                    filing.accession_number,
                    status="complete",
                    sections_extracted=0,
                    chunks_count=0,
                )
            return 0

        if metadata_repo:
            await metadata_repo.update_status(
                filing.accession_number,
                status="indexing",
                sections_extracted=len(sections),
            )

        # Chunk sections (hierarchical for small-to-big retrieval)
        if config.hierarchical_chunking:
            chunks = chunker.chunk_filing_hierarchical(sections, filing)
        else:
            chunks = chunker.chunk_filing(sections, filing)

        if not chunks:
            logger.warning(f"No chunks generated from filing {filing.accession_number}")
            if metadata_repo:
                await metadata_repo.update_status(
                    filing.accession_number,
                    status="complete",
                    chunks_count=0,
                )
            return 0

        # Index chunks
        doc_ids = await indexer.index_chunks(chunks, filing.accession_number)

        if metadata_repo:
            await metadata_repo.update_status(
                filing.accession_number,
                status="complete",
                chunks_count=len(doc_ids),
            )

        logger.info(f"Indexed {len(doc_ids)} chunks from {filing.accession_number}")
        return len(doc_ids)

    except Exception as e:
        logger.error(f"Failed to process filing {filing.accession_number}: {e}")
        if metadata_repo:
            await metadata_repo.update_status(
                filing.accession_number,
                status="failed",
                error_message=str(e),
            )
        raise


async def process_company(
    company: dict,
    client: SECEdgarClient,
    parser: FilingParser,
    chunker: DocumentChunker,
    indexer: FilingRAGIndexer,
    metadata_repo: FilingMetadataRepo | None,
    config: SECFilingConfig,
    last_n_quarters: int | None = None,
) -> tuple[int, int, int]:
    """
    Process all filings for a company.

    Args:
        company: Company dict with ticker, cik, name
        client: SEC EDGAR client
        parser: Filing parser
        chunker: Document chunker
        indexer: RAG indexer
        metadata_repo: Optional metadata repository
        config: Configuration

    Returns:
        Tuple of (filings_downloaded, filings_parsed, chunks_indexed)
    """
    cik = company["cik"]
    ticker = company.get("ticker")
    name = company.get("name", "Unknown")

    logger.info(f"Processing company: {name} ({ticker or cik})")

    # Get filings for company
    after_date = datetime.now() - timedelta(days=365 * config.years_back)
    try:
        filings = await client.get_company_filings(
            cik=cik,
            filing_types=config.get_filing_types(),
            after_date=after_date,
        )
    except DownloadError as e:
        logger.error(f"Failed to get filings for {name}: {e}")
        return (0, 0, 0)

    if not filings:
        logger.info(f"No filings found for {name}")
        return (0, 0, 0)

    # Filter 10-Q filings to last N quarters if specified
    if last_n_quarters is not None:
        # Separate 10-K and 10-Q filings
        ten_k_filings = [f for f in filings if f.filing_type == FilingType.FORM_10K]
        ten_q_filings = [f for f in filings if f.filing_type == FilingType.FORM_10Q]

        # Sort 10-Q by date descending and take last N
        ten_q_filings.sort(key=lambda f: f.filing_date, reverse=True)
        ten_q_filings = ten_q_filings[:last_n_quarters]

        filings = ten_k_filings + ten_q_filings
        logger.info(f"Filtered to {len(ten_q_filings)} most recent 10-Q filings")

    logger.info(f"Found {len(filings)} filings for {name}")

    filings_downloaded = 0
    filings_parsed = 0
    chunks_indexed = 0

    for filing in filings:
        try:
            num_chunks = await process_filing(
                client=client,
                filing=filing,
                parser=parser,
                chunker=chunker,
                indexer=indexer,
                metadata_repo=metadata_repo,
                config=config,
            )
            filings_downloaded += 1
            if num_chunks > 0:
                filings_parsed += 1
                chunks_indexed += num_chunks
        except Exception as e:
            logger.error(f"Error processing filing {filing.accession_number}: {e}")
            continue

    return (filings_downloaded, filings_parsed, chunks_indexed)


async def run_ingestion(
    config: SECFilingConfig,
    companies: list[dict],
    resume_checkpoint: FilingCheckpoint | None = None,
    dry_run: bool = False,
    last_n_quarters: int | None = None,
) -> FilingCheckpoint:
    """
    Run the full ingestion pipeline.

    Args:
        config: Configuration
        companies: List of companies to process
        resume_checkpoint: Optional checkpoint to resume from
        dry_run: If True, only count filings without downloading

    Returns:
        Final checkpoint
    """
    # Initialize checkpoint
    if resume_checkpoint and resume_checkpoint.is_resumable:
        checkpoint = resume_checkpoint
        start_idx = checkpoint.current_company_index
        logger.info(f"Resuming from company {start_idx + 1}/{len(companies)}")
    else:
        checkpoint = FilingCheckpoint(
            started_at=datetime.now(UTC),
            total_companies=len(companies),
        )
        start_idx = 0

    checkpoint_manager = CheckpointManager(config.checkpoint_dir)

    if dry_run:
        logger.info("DRY RUN - counting filings only")
        async with SECEdgarClient(config) as client:
            total_filings = 0
            after_date = datetime.now() - timedelta(days=365 * config.years_back)

            for company in companies[start_idx:]:
                try:
                    filings = await client.get_company_filings(
                        cik=company["cik"],
                        filing_types=config.get_filing_types(),
                        after_date=after_date,
                    )
                    total_filings += len(filings)
                    logger.info(f"{company.get('ticker', company['cik'])}: {len(filings)} filings")
                except Exception as e:
                    logger.warning(f"Error for {company.get('ticker', company['cik'])}: {e}")

            logger.info("\nDRY RUN SUMMARY:")
            logger.info(f"  Companies: {len(companies)}")
            logger.info(f"  Total filings: {total_filings}")
            logger.info(f"  Estimated chunks: ~{total_filings * 75} (avg 75/filing)")

        checkpoint.mark_complete()
        return checkpoint

    # Connect to database
    database_url = os.environ.get(
        "DATABASE_URL", "postgresql://nl2api:nl2api@localhost:5432/nl2api"
    )

    pool = await asyncpg.create_pool(database_url)
    if pool is None:
        raise RuntimeError("Failed to create database connection pool")

    try:
        # Initialize components
        parser = FilingParser(extract_all_sections=False)  # Only key sections
        chunker = DocumentChunker.from_config(config)

        # Create embedder based on config (must match existing index dimensions)
        if config.embedder_type == "openai":
            # Check multiple possible env var names
            openai_api_key = (
                os.environ.get("OPENAI_API_KEY")
                or os.environ.get("NL2API_OPENAI_API_KEY")
                or os.environ.get("RAG_UI_OPENAI_API_KEY")
            )
            if not openai_api_key:
                raise RuntimeError(
                    "OpenAI API key required. Set OPENAI_API_KEY or NL2API_OPENAI_API_KEY"
                )
            embedder = create_embedder("openai", api_key=openai_api_key)
        else:
            embedder = create_embedder(config.embedder_type)
        logger.info(f"Using {config.embedder_type} embedder ({embedder.dimension} dimensions)")

        indexer = FilingRAGIndexer(pool, embedder=embedder, batch_size=config.embedding_batch_size)
        metadata_repo = FilingMetadataRepo(pool)

        async with SECEdgarClient(config) as client:
            for idx, company in enumerate(companies[start_idx:], start=start_idx):
                checkpoint.mark_downloading(company["cik"], idx)

                try:
                    filings_dl, filings_parsed, chunks = await process_company(
                        company=company,
                        client=client,
                        parser=parser,
                        chunker=chunker,
                        indexer=indexer,
                        metadata_repo=metadata_repo,
                        config=config,
                        last_n_quarters=last_n_quarters,
                    )

                    checkpoint.update_filing_progress(
                        filings_downloaded=filings_dl,
                        filings_parsed=filings_parsed,
                        chunks_indexed=chunks,
                    )
                    checkpoint.complete_company()

                except Exception as e:
                    logger.error(
                        f"Error processing company {company.get('name', company['cik'])}: {e}"
                    )
                    checkpoint.error_count += 1
                    checkpoint.last_error = str(e)

                    if checkpoint.error_count >= config.max_errors:
                        checkpoint.mark_failed(f"Too many errors: {checkpoint.error_count}")
                        checkpoint_manager.save(checkpoint)
                        raise IngestionAbortError(
                            f"Aborting: {checkpoint.error_count} errors exceeded max {config.max_errors}",
                            source="sec_edgar",
                            error_count=checkpoint.error_count,
                            max_errors=config.max_errors,
                        )

                # Save checkpoint periodically
                if (idx + 1) % config.checkpoint_interval == 0:
                    checkpoint_manager.save(checkpoint)
                    logger.info(
                        f"Checkpoint saved: {checkpoint.companies_processed}/{checkpoint.total_companies} companies"
                    )

                # Delay between companies to be respectful of SEC servers
                if config.inter_company_delay_seconds > 0 and idx < len(companies) - 1:
                    await asyncio.sleep(config.inter_company_delay_seconds)

        checkpoint.mark_complete()
        checkpoint_manager.save(checkpoint)

        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("INGESTION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Companies processed: {checkpoint.companies_processed}")
        logger.info(f"Filings downloaded:  {checkpoint.filings_downloaded}")
        logger.info(f"Filings parsed:      {checkpoint.filings_parsed}")
        logger.info(f"Chunks indexed:      {checkpoint.chunks_indexed}")
        logger.info(f"Errors:              {checkpoint.error_count}")
        logger.info("=" * 60)

    finally:
        await pool.close()

    return checkpoint


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Ingest SEC EDGAR filings for RAG evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Ingest all S&P 500 companies (2 years)
    python scripts/ingest_sec_filings.py

    # Specific tickers only
    python scripts/ingest_sec_filings.py --tickers AAPL,MSFT,GOOGL

    # Resume interrupted ingestion
    python scripts/ingest_sec_filings.py --resume

    # Dry run to see filing counts
    python scripts/ingest_sec_filings.py --dry-run
        """,
    )

    parser.add_argument(
        "--tickers",
        type=str,
        help="Comma-separated list of tickers to process (default: all S&P 500)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum number of companies to process",
    )
    parser.add_argument(
        "--years",
        type=int,
        default=2,
        help="Number of years of filings to download (default: 2)",
    )
    parser.add_argument(
        "--filing-types",
        type=str,
        default="10-K,10-Q",
        help="Comma-separated filing types to download (default: 10-K,10-Q)",
    )
    parser.add_argument(
        "--last-n-quarters",
        type=int,
        help="Limit 10-Q filings to last N quarters per company (e.g., 4 for one year)",
    )
    parser.add_argument(
        "--embedder",
        type=str,
        default="openai",
        choices=["local", "openai"],
        help="Embedder type: 'local' (384 dims) or 'openai' (1536 dims, default)",
    )
    parser.add_argument(
        "--embedding-batch-size",
        type=int,
        default=32,
        help="Batch size for embedding API calls (default: 32, reduce if hitting rate limits)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint if available",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Count filings without downloading",
    )
    parser.add_argument(
        "--data-file",
        type=str,
        default="data/tickers/sp500.json",
        help="Path to S&P 500 data file (default: data/tickers/sp500.json)",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--hierarchical",
        action="store_true",
        default=True,
        help="Use small-to-big hierarchical chunking (default: True)",
    )
    parser.add_argument(
        "--no-hierarchical",
        dest="hierarchical",
        action="store_false",
        help="Use flat chunking instead of hierarchical",
    )

    return parser.parse_args()


async def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    filing_types = [ft.strip() for ft in args.filing_types.split(",")]
    config = SECFilingConfig(
        years_back=args.years,
        filing_types=filing_types,
        embedder_type=args.embedder,
        embedding_batch_size=args.embedding_batch_size,
        hierarchical_chunking=args.hierarchical,
    )
    config.ensure_data_dir()

    # Store last_n_quarters for filtering later
    last_n_quarters = args.last_n_quarters

    # Load companies
    if args.tickers:
        # Specific tickers provided
        tickers = [t.strip().upper() for t in args.tickers.split(",")]
        try:
            all_companies = load_sp500_companies(Path(args.data_file))
            companies = [c for c in all_companies if c.get("ticker", "").upper() in tickers]

            # Add any tickers not in S&P 500 with just the ticker
            found_tickers = {c.get("ticker", "").upper() for c in companies}
            for ticker in tickers:
                if ticker not in found_tickers:
                    logger.warning(
                        f"Ticker {ticker} not in S&P 500 data, will attempt to look up CIK"
                    )
                    # We'd need to look up CIK from SEC - for now skip
        except FileNotFoundError:
            logger.error(f"S&P 500 data file not found: {args.data_file}")
            logger.error("Please run: python scripts/create_sp500_data.py")
            return 1
    else:
        # Load all S&P 500 companies
        try:
            companies = load_sp500_companies(Path(args.data_file))
        except FileNotFoundError:
            logger.error(f"S&P 500 data file not found: {args.data_file}")
            logger.error("Please create the file or specify --tickers")
            return 1

    if args.limit:
        companies = companies[: args.limit]

    logger.info(f"Processing {len(companies)} companies")

    # Check for checkpoint
    checkpoint_manager = CheckpointManager(config.checkpoint_dir)
    resume_checkpoint = None

    if args.resume:
        resume_checkpoint = checkpoint_manager.load()
        if resume_checkpoint and resume_checkpoint.is_resumable:
            logger.info(
                f"Found resumable checkpoint: {resume_checkpoint.companies_processed}/{resume_checkpoint.total_companies} companies"
            )
        else:
            logger.info("No resumable checkpoint found, starting fresh")
            resume_checkpoint = None

    try:
        await run_ingestion(
            config=config,
            companies=companies,
            resume_checkpoint=resume_checkpoint,
            dry_run=args.dry_run,
            last_n_quarters=last_n_quarters,
        )
        return 0
    except IngestionAbortError as e:
        logger.error(f"Ingestion aborted: {e}")
        return 1
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        return 130


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
