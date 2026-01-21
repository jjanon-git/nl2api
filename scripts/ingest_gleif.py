#!/usr/bin/env python3
"""
GLEIF LEI Data Ingestion Pipeline

Downloads and imports 2M+ legal entities from GLEIF golden copy.
Supports both full ingestion and incremental delta updates.

Data source: https://www.gleif.org/en/lei-data/gleif-golden-copy

Incremental Update Strategy:
    - First run: Downloads full golden copy (~2M entities)
    - Subsequent runs with --delta: Downloads only daily delta files
    - Delta files contain new, modified, and retired LEIs
    - Uses ON CONFLICT upserts for idempotent updates
    - Tracks last successful ingestion date in checkpoint

Usage:
    # Full ingestion (first time or reset)
    .venv/bin/python scripts/ingest_gleif.py --mode full

    # Incremental update (daily deltas since last run)
    .venv/bin/python scripts/ingest_gleif.py --mode delta

    # Auto mode: delta if previous ingestion exists, else full
    .venv/bin/python scripts/ingest_gleif.py --mode auto

    # Dry run (validate only)
    .venv/bin/python scripts/ingest_gleif.py --dry-run

    # Resume from checkpoint
    .venv/bin/python scripts/ingest_gleif.py --resume
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import gzip
import hashlib
import json
import logging
import os
import re
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterator

import httpx

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import ingestion telemetry (after path setup)
from src.nl2api.ingestion.telemetry import (
    trace_ingestion_operation,
    record_ingestion_metric,
    SpanAttributes,
)
from src.nl2api.ingestion.errors import DownloadError, LoadError


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

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


# GLEIF API endpoints
GLEIF_API_BASE = "https://leidata.gleif.org/api/v2"
GLEIF_GOLDEN_COPY_API = f"{GLEIF_API_BASE}/golden-copies/publishes/latest"
GLEIF_DELTA_API = f"{GLEIF_API_BASE}/delta-files"


# Company suffixes to strip for alias generation
COMPANY_SUFFIXES = [
    "Inc", "Inc.", "Incorporated",
    "Corp", "Corp.", "Corporation",
    "Ltd", "Ltd.", "Limited",
    "LLC", "L.L.C.", "L.L.C",
    "PLC", "P.L.C.", "P.L.C",
    "Co", "Co.", "Company",
    "Holdings", "Group", "International", "Intl",
    "SA", "S.A.", "AG", "A.G.", "GmbH", "NV", "N.V.", "BV", "B.V.",
    "SE", "S.E.", "LP", "L.P.", "LLP", "L.L.P.",
]


def normalize_alias(s: str) -> str:
    """Normalize string for matching."""
    return s.lower().strip()


def generate_aliases(name: str) -> list[str]:
    """Generate aliases from entity name."""
    aliases = set()

    # Original name (normalized)
    aliases.add(normalize_alias(name))

    # Without company suffixes
    stripped = name
    for suffix in COMPANY_SUFFIXES:
        pattern = rf"\s*[,&]?\s*{re.escape(suffix)}\.?$"
        stripped = re.sub(pattern, "", stripped, flags=re.I).strip()

    if stripped and stripped != name:
        aliases.add(normalize_alias(stripped))

    # Without punctuation
    clean = re.sub(r"[^\w\s]", "", name)
    if clean:
        aliases.add(normalize_alias(clean))

    # Acronym for multi-word names
    words = name.split()
    if len(words) >= 3:
        initials = "".join(w[0] for w in words if w and w[0].isupper())
        if len(initials) >= 2:
            aliases.add(normalize_alias(initials))

    # Filter out empty or too short aliases
    return [a for a in aliases if len(a) >= 2]


async def get_gleif_golden_copy_info(timeout: int = 30) -> dict:
    """
    Get metadata about the latest GLEIF golden copy.

    Returns:
        Dict with download_url, publish_date, record_count
    """
    async with httpx.AsyncClient(timeout=timeout) as client:
        response = await client.get(GLEIF_GOLDEN_COPY_API)
        response.raise_for_status()
        data = response.json()

        # Extract info from response
        result = {
            "publish_date": data.get("data", {}).get("publishDate"),
            "download_url": None,
            "record_count": None,
        }

        # Find the CSV download URL
        for item in data.get("data", {}).get("concatenatedFiles", []):
            if item.get("fileExtension") == "csv.gz":
                result["download_url"] = item.get("url")
                result["record_count"] = item.get("recordCount")
                break

        if not result["download_url"]:
            # Fallback URL
            result["download_url"] = f"{GLEIF_API_BASE}/golden-copies/publishes/latest/download"

        return result


async def get_gleif_delta_files(since_date: datetime, timeout: int = 30) -> list[dict]:
    """
    Get list of delta files since a specific date.

    GLEIF provides daily delta files with:
    - New LEIs
    - Modified LEIs
    - Retired/Lapsed LEIs

    Args:
        since_date: Get deltas since this date
        timeout: Request timeout

    Returns:
        List of delta file info dicts sorted by date
    """
    delta_files = []

    async with httpx.AsyncClient(timeout=timeout) as client:
        # GLEIF delta files are organized by date
        current_date = since_date.date()
        today = datetime.now(timezone.utc).date()

        while current_date <= today:
            date_str = current_date.strftime("%Y/%m/%d")
            url = f"{GLEIF_DELTA_API}/{date_str}"

            try:
                response = await client.get(url)
                if response.status_code == 200:
                    data = response.json()
                    for item in data.get("data", {}).get("deltaFiles", []):
                        if item.get("fileExtension") == "csv.gz":
                            delta_files.append({
                                "date": current_date.isoformat(),
                                "url": item.get("url"),
                                "record_count": item.get("recordCount", 0),
                                "type": item.get("deltaType", "unknown"),
                            })
            except httpx.HTTPError:
                # Skip dates with no delta files
                pass

            current_date += timedelta(days=1)

    return sorted(delta_files, key=lambda x: x["date"])


async def download_file_streaming(
    url: str,
    output_path: Path,
    chunk_size: int = 65536,
    timeout: int = 600,
    description: str = "Downloading",
) -> int:
    """
    Download file with streaming (memory efficient).

    Args:
        url: Download URL
        output_path: Path to save downloaded file
        chunk_size: Chunk size for streaming
        timeout: Download timeout in seconds
        description: Description for logging

    Returns:
        Number of bytes downloaded

    Raises:
        DownloadError: If download fails
    """
    logger.info("%s from %s", description, url)

    with trace_ingestion_operation(
        "download_file",
        {SpanAttributes.SOURCE: "gleif", SpanAttributes.SOURCE_URL: url},
    ) as span:
        try:
            async with httpx.AsyncClient(timeout=httpx.Timeout(timeout)) as client:
                async with client.stream("GET", url, follow_redirects=True) as response:
                    response.raise_for_status()
                    total_size = int(response.headers.get("content-length", 0))
                    downloaded = 0

                    with open(output_path, "wb") as f:
                        async for chunk in response.aiter_bytes(chunk_size=chunk_size):
                            f.write(chunk)
                            downloaded += len(chunk)

                            # Log progress every 50MB
                            if total_size and downloaded % (50 * 1024 * 1024) < chunk_size:
                                percent = downloaded / total_size * 100
                                logger.info(
                                    "Download progress: %.1f%% (%d MB / %d MB)",
                                    percent,
                                    downloaded // (1024 * 1024),
                                    total_size // (1024 * 1024),
                                )

            if span:
                span.set_attribute(SpanAttributes.FILE_SIZE_BYTES, downloaded)
                span.set_attribute(SpanAttributes.FILE_PATH, str(output_path))

            logger.info("Download complete: %d bytes", downloaded)
            record_ingestion_metric("download_bytes_total", downloaded, {"source": "gleif"})
            return downloaded

        except httpx.HTTPStatusError as e:
            raise DownloadError(
                f"HTTP error downloading {url}: {e.response.status_code}",
                source="gleif",
                url=url,
                status_code=e.response.status_code,
            ) from e
        except httpx.RequestError as e:
            raise DownloadError(
                f"Request error downloading {url}: {e}",
                source="gleif",
                url=url,
            ) from e


def parse_gleif_csv_streaming(
    csv_path: Path,
    skip_rows: int = 0,
    include_inactive: bool = False,
) -> Iterator[dict]:
    """
    Parse GLEIF CSV as generator - never loads full file into memory.

    Args:
        csv_path: Path to gzipped CSV file
        skip_rows: Number of rows to skip (for resume)
        include_inactive: Include inactive/retired entities (for delta processing)

    Yields:
        Entity dictionaries
    """
    with gzip.open(csv_path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        for i, row in enumerate(reader):
            if i < skip_rows:
                continue

            status = row.get("Entity.EntityStatus", "").upper()

            # For full load, skip inactive; for delta, include all
            if not include_inactive and status != "ACTIVE":
                continue

            # Skip entities without legal name
            legal_name = row.get("Entity.LegalName", "").strip()
            if not legal_name:
                continue

            yield {
                "row_number": i,
                "lei": row.get("LEI", "").strip().upper(),
                "primary_name": legal_name,
                "country_code": row.get("Entity.LegalAddress.Country", "").strip().upper(),
                "region": row.get("Entity.LegalAddress.Region", "").strip(),
                "city": row.get("Entity.LegalAddress.City", "").strip(),
                "entity_category": row.get("Entity.EntityCategory", "GENERAL").strip().lower(),
                "entity_status": "active" if status == "ACTIVE" else "inactive",
                "data_source": "gleif",
            }


def transform_to_db_records(
    entities: Iterator[dict],
) -> Iterator[tuple[Any, ...]]:
    """
    Transform entity dicts to database records for COPY protocol.

    Args:
        entities: Iterator of entity dictionaries

    Yields:
        Tuples matching entities table columns
    """
    for entity in entities:
        entity_id = uuid.uuid4()

        # Map entity category to type
        category = entity.get("entity_category", "general")
        entity_type = "fund" if category == "fund" else "company"

        yield (
            entity_id,                          # id
            entity.get("lei"),                  # lei
            None,                               # cik
            None,                               # permid
            None,                               # figi
            entity.get("primary_name"),         # primary_name
            None,                               # display_name
            None,                               # ticker
            None,                               # ric
            None,                               # exchange
            entity_type,                        # entity_type
            entity.get("entity_status", "active"),  # entity_status
            False,                              # is_public
            entity.get("country_code") or None, # country_code
            entity.get("region") or None,       # region
            entity.get("city") or None,         # city
            None,                               # sic_code
            None,                               # naics_code
            None,                               # gics_sector
            None,                               # parent_entity_id
            None,                               # ultimate_parent_id
            "gleif",                            # data_source
            1.0,                                # confidence_score
            False,                              # ric_validated
            None,                               # last_verified_at
        )


async def count_csv_rows(csv_path: Path, include_inactive: bool = False) -> int:
    """Count rows in gzipped CSV for progress tracking."""
    count = 0
    with gzip.open(csv_path, "rt", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            status = row.get("Entity.EntityStatus", "").upper()
            if include_inactive or status == "ACTIVE":
                count += 1
    return count


async def bulk_load_entities(
    pool,
    records: Iterator[tuple[Any, ...]],
    batch_size: int = 50000,
    checkpoint_manager=None,
    checkpoint=None,
    progress=None,
) -> int:
    """
    Bulk load using PostgreSQL COPY protocol (10-100x faster than INSERT).

    Raises:
        LoadError: If bulk load fails
    """
    columns = [
        "id", "lei", "cik", "permid", "figi",
        "primary_name", "display_name",
        "ticker", "ric", "exchange",
        "entity_type", "entity_status", "is_public",
        "country_code", "region", "city",
        "sic_code", "naics_code", "gics_sector",
        "parent_entity_id", "ultimate_parent_id",
        "data_source", "confidence_score", "ric_validated", "last_verified_at",
    ]

    total_loaded = 0
    batch = []
    batch_number = 0

    with trace_ingestion_operation(
        "bulk_load",
        {
            SpanAttributes.SOURCE: "gleif",
            SpanAttributes.DB_OPERATION: "copy",
            SpanAttributes.DB_TABLE: "entities",
            SpanAttributes.DB_BATCH_SIZE: batch_size,
        },
    ) as span:
        async with pool.acquire() as conn:
            for record in records:
                batch.append(record)

                if len(batch) >= batch_size:
                    batch_number += 1
                    try:
                        result = await conn.copy_records_to_table(
                            "entities",
                            records=batch,
                            columns=columns,
                        )
                        loaded = int(result.split()[-1])
                        total_loaded += loaded

                        if progress:
                            progress.update(loaded)

                        if checkpoint and checkpoint_manager:
                            checkpoint.update_progress(
                                offset=total_loaded,
                                imported=loaded,
                                last_entity_id=str(batch[-1][0]),
                            )
                            checkpoint_manager.save(checkpoint)

                        logger.info("Batch loaded: %d (total: %d)", loaded, total_loaded)
                    except Exception as e:
                        logger.error("Batch load failed: %s", e)
                        raise LoadError(
                            f"Bulk load failed at batch {batch_number}: {e}",
                            source="gleif",
                            batch_number=batch_number,
                            rows_affected=len(batch),
                        ) from e

                    batch = []

            # Load remaining records
            if batch:
                result = await conn.copy_records_to_table(
                    "entities",
                    records=batch,
                    columns=columns,
                )
                loaded = int(result.split()[-1])
                total_loaded += loaded
                if progress:
                    progress.update(loaded)

        if span:
            span.set_attribute(SpanAttributes.RECORDS_INSERTED, total_loaded)

        record_ingestion_metric("entities_loaded_total", total_loaded, {"source": "gleif"})

    return total_loaded


async def upsert_entities_from_delta(
    pool,
    entities: Iterator[dict],
    batch_size: int = 1000,
) -> dict:
    """
    Upsert entities from delta file (handles updates and status changes).

    Uses INSERT ... ON CONFLICT for idempotent updates.

    Args:
        pool: Database connection pool
        entities: Iterator of entity dicts
        batch_size: Records per batch

    Returns:
        Stats dict with inserted, updated, retired counts
    """
    stats = {"inserted": 0, "updated": 0, "retired": 0}
    batch = []

    async with pool.acquire() as conn:
        for entity in entities:
            batch.append(entity)

            if len(batch) >= batch_size:
                result = await _upsert_batch(conn, batch)
                stats["inserted"] += result["inserted"]
                stats["updated"] += result["updated"]
                stats["retired"] += result["retired"]
                batch = []

        if batch:
            result = await _upsert_batch(conn, batch)
            stats["inserted"] += result["inserted"]
            stats["updated"] += result["updated"]
            stats["retired"] += result["retired"]

    return stats


async def _upsert_batch(conn, entities: list[dict]) -> dict:
    """Upsert a batch of entities."""
    stats = {"inserted": 0, "updated": 0, "retired": 0}

    for entity in entities:
        lei = entity.get("lei")
        if not lei:
            continue

        status = entity.get("entity_status", "active")

        # Check if entity exists
        existing = await conn.fetchval(
            "SELECT id FROM entities WHERE lei = $1",
            lei,
        )

        if status != "active":
            # Mark as inactive/retired
            if existing:
                await conn.execute(
                    "UPDATE entities SET entity_status = $1, updated_at = NOW() WHERE lei = $2",
                    status,
                    lei,
                )
                stats["retired"] += 1
        elif existing:
            # Update existing entity
            await conn.execute(
                """
                UPDATE entities SET
                    primary_name = $1,
                    country_code = $2,
                    region = $3,
                    city = $4,
                    entity_status = 'active',
                    updated_at = NOW()
                WHERE lei = $5
                """,
                entity.get("primary_name"),
                entity.get("country_code") or None,
                entity.get("region") or None,
                entity.get("city") or None,
                lei,
            )
            stats["updated"] += 1
        else:
            # Insert new entity
            await conn.execute(
                """
                INSERT INTO entities (
                    id, lei, primary_name, country_code, region, city,
                    entity_type, entity_status, is_public, data_source,
                    confidence_score
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11
                )
                """,
                uuid.uuid4(),
                lei,
                entity.get("primary_name"),
                entity.get("country_code") or None,
                entity.get("region") or None,
                entity.get("city") or None,
                "fund" if entity.get("entity_category") == "fund" else "company",
                "active",
                False,
                "gleif",
                1.0,
            )
            stats["inserted"] += 1

    return stats


async def generate_aliases_for_entities(
    pool,
    batch_size: int = 10000,
    only_missing: bool = True,
) -> int:
    """Generate aliases for GLEIF entities."""
    logger.info("Generating aliases for GLEIF entities...")

    total_aliases = 0
    offset = 0

    async with pool.acquire() as conn:
        while True:
            if only_missing:
                # Only entities without aliases
                rows = await conn.fetch(
                    """
                    SELECT e.id, e.primary_name
                    FROM entities e
                    WHERE e.data_source = 'gleif'
                    AND NOT EXISTS (
                        SELECT 1 FROM entity_aliases a WHERE a.entity_id = e.id
                    )
                    ORDER BY e.id
                    LIMIT $1 OFFSET $2
                    """,
                    batch_size,
                    offset,
                )
            else:
                rows = await conn.fetch(
                    """
                    SELECT e.id, e.primary_name
                    FROM entities e
                    WHERE e.data_source = 'gleif'
                    ORDER BY e.id
                    LIMIT $1 OFFSET $2
                    """,
                    batch_size,
                    offset,
                )

            if not rows:
                break

            alias_records = []
            for row in rows:
                entity_id = row["id"]
                name = row["primary_name"]

                aliases = generate_aliases(name)
                for i, alias in enumerate(aliases):
                    alias_records.append((
                        uuid.uuid4(),
                        entity_id,
                        alias,
                        "generated" if i > 0 else "legal_name",
                        i == 0,
                    ))

            if alias_records:
                await conn.executemany(
                    """
                    INSERT INTO entity_aliases (id, entity_id, alias, alias_type, is_primary)
                    VALUES ($1, $2, $3, $4, $5)
                    ON CONFLICT (entity_id, alias) DO NOTHING
                    """,
                    alias_records,
                )
                total_aliases += len(alias_records)

            logger.info("Generated aliases: batch=%d, total=%d", len(alias_records), total_aliases)
            offset += batch_size

    return total_aliases


class IngestionState:
    """Tracks ingestion state for incremental updates."""

    def __init__(self, state_file: Path):
        self.state_file = state_file
        self._state = self._load()

    def _load(self) -> dict:
        if self.state_file.exists():
            with open(self.state_file) as f:
                return json.load(f)
        return {}

    def save(self) -> None:
        self.state_file.parent.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w") as f:
            json.dump(self._state, f, indent=2, default=str)

    @property
    def last_full_ingestion(self) -> datetime | None:
        """Date of last full golden copy ingestion."""
        date_str = self._state.get("last_full_ingestion")
        if date_str:
            return datetime.fromisoformat(date_str)
        return None

    @last_full_ingestion.setter
    def last_full_ingestion(self, value: datetime) -> None:
        self._state["last_full_ingestion"] = value.isoformat()

    @property
    def last_delta_date(self) -> datetime | None:
        """Date of last processed delta file."""
        date_str = self._state.get("last_delta_date")
        if date_str:
            return datetime.fromisoformat(date_str)
        return None

    @last_delta_date.setter
    def last_delta_date(self, value: datetime) -> None:
        self._state["last_delta_date"] = value.isoformat()

    @property
    def golden_copy_hash(self) -> str | None:
        """Hash of last processed golden copy."""
        return self._state.get("golden_copy_hash")

    @golden_copy_hash.setter
    def golden_copy_hash(self, value: str) -> None:
        self._state["golden_copy_hash"] = value

    @property
    def has_full_ingestion(self) -> bool:
        """Whether a full ingestion has been completed."""
        return self.last_full_ingestion is not None


async def run_full_ingestion(
    pool,
    config,
    checkpoint_mgr,
    args,
    state: IngestionState,
) -> dict:
    """Run full golden copy ingestion."""
    logger.info("Starting FULL ingestion (golden copy)")

    # Get golden copy info
    gc_info = await get_gleif_golden_copy_info()
    logger.info(
        "Golden copy: %s, ~%s records",
        gc_info["publish_date"],
        f"{gc_info['record_count']:,}" if gc_info["record_count"] else "unknown",
    )

    # Download
    download_path = config.download_dir / "gleif_golden_copy.csv.gz"

    if not args.skip_download:
        checkpoint = checkpoint_mgr.create_new()
        checkpoint.mark_downloading()
        checkpoint_mgr.save(checkpoint)

        await download_file_streaming(
            url=gc_info["download_url"],
            output_path=download_path,
            timeout=config.download_timeout_seconds,
            description="Downloading golden copy",
        )

    # Count records
    logger.info("Counting records...")
    total_records = await count_csv_rows(download_path)
    logger.info("Total active records: %d", total_records)

    if args.dry_run:
        logger.info("[DRY RUN] Would load %d records", total_records)
        return {"mode": "full", "dry_run": True, "total": total_records}

    # Load
    from src.nl2api.ingestion import CheckpointManager, ProgressTracker

    checkpoint = checkpoint_mgr.load() or checkpoint_mgr.create_new()
    checkpoint.mark_loading(total_records)
    checkpoint_mgr.save(checkpoint)

    skip_rows = checkpoint.last_offset if args.resume else 0

    progress = ProgressTracker(
        total=total_records - skip_rows,
        description="Loading entities",
    )

    entities = parse_gleif_csv_streaming(download_path, skip_rows=skip_rows)
    records = transform_to_db_records(entities)

    loaded = await bulk_load_entities(
        pool=pool,
        records=records,
        batch_size=args.batch_size,
        checkpoint_manager=checkpoint_mgr,
        checkpoint=checkpoint,
        progress=progress,
    )

    progress.finish()

    # Generate aliases
    if not args.skip_aliases:
        aliases = await generate_aliases_for_entities(pool, only_missing=True)
    else:
        aliases = 0

    # Update state
    state.last_full_ingestion = datetime.now(timezone.utc)
    state.last_delta_date = datetime.now(timezone.utc)
    state.save()

    checkpoint.mark_complete()
    checkpoint_mgr.save(checkpoint)

    return {
        "mode": "full",
        "loaded": loaded,
        "aliases": aliases,
        "duration_seconds": progress.elapsed_seconds,
    }


async def run_delta_ingestion(
    pool,
    config,
    args,
    state: IngestionState,
) -> dict:
    """Run incremental delta ingestion."""
    if not state.has_full_ingestion:
        logger.error("No full ingestion found. Run with --mode full first.")
        return {"mode": "delta", "error": "no_full_ingestion"}

    since_date = state.last_delta_date or state.last_full_ingestion
    logger.info("Starting DELTA ingestion (since %s)", since_date.date())

    # Get delta files
    delta_files = await get_gleif_delta_files(since_date)

    if not delta_files:
        logger.info("No delta files found since %s", since_date.date())
        return {"mode": "delta", "files_processed": 0}

    logger.info("Found %d delta files to process", len(delta_files))

    if args.dry_run:
        for df in delta_files:
            logger.info("  [DRY RUN] %s: %d records", df["date"], df["record_count"])
        return {"mode": "delta", "dry_run": True, "files": len(delta_files)}

    # Process each delta file
    total_stats = {"inserted": 0, "updated": 0, "retired": 0, "files": 0}

    for delta_file in delta_files:
        logger.info("Processing delta: %s (%d records)", delta_file["date"], delta_file["record_count"])

        # Download delta file
        delta_path = config.download_dir / f"gleif_delta_{delta_file['date'].replace('-', '')}.csv.gz"

        await download_file_streaming(
            url=delta_file["url"],
            output_path=delta_path,
            description=f"Downloading delta {delta_file['date']}",
        )

        # Process delta
        entities = parse_gleif_csv_streaming(delta_path, include_inactive=True)
        stats = await upsert_entities_from_delta(pool, entities)

        total_stats["inserted"] += stats["inserted"]
        total_stats["updated"] += stats["updated"]
        total_stats["retired"] += stats["retired"]
        total_stats["files"] += 1

        logger.info(
            "Delta %s: +%d new, ~%d updated, -%d retired",
            delta_file["date"],
            stats["inserted"],
            stats["updated"],
            stats["retired"],
        )

        # Update state after each file
        state.last_delta_date = datetime.fromisoformat(delta_file["date"])
        state.save()

        # Clean up delta file
        delta_path.unlink(missing_ok=True)

    # Generate aliases for new entities
    if not args.skip_aliases and total_stats["inserted"] > 0:
        aliases = await generate_aliases_for_entities(pool, only_missing=True)
        total_stats["aliases"] = aliases

    return {"mode": "delta", **total_stats}


async def main():
    parser = argparse.ArgumentParser(
        description="Ingest GLEIF LEI data into entity database"
    )
    parser.add_argument(
        "--mode",
        choices=["auto", "full", "delta"],
        default="auto",
        help="Ingestion mode: full (golden copy), delta (incremental), auto (delta if possible)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and count records without loading",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from checkpoint (full mode only)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50000,
        help="Batch size for COPY protocol (default: 50000)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip download if file already exists",
    )
    parser.add_argument(
        "--skip-aliases",
        action="store_true",
        help="Skip alias generation",
    )
    args = parser.parse_args()

    # Import dependencies
    import asyncpg
    from src.nl2api.ingestion import CheckpointManager, EntityIngestionConfig

    print("=" * 60)
    print("GLEIF LEI DATA INGESTION")
    print("=" * 60)

    # Load configuration
    config = EntityIngestionConfig()
    config.ensure_data_dir()

    # Initialize state tracker
    state = IngestionState(config.data_dir / "gleif_state.json")

    # Initialize checkpoint manager
    checkpoint_mgr = CheckpointManager(config.checkpoint_dir, "gleif")

    # Determine mode
    mode = args.mode
    if mode == "auto":
        if state.has_full_ingestion:
            mode = "delta"
            logger.info("Auto mode: Using delta ingestion (previous full ingestion found)")
        else:
            mode = "full"
            logger.info("Auto mode: Using full ingestion (no previous ingestion)")

    # Connect to database
    db_url = os.environ.get("DATABASE_URL", "postgresql://nl2api:nl2api@localhost:5432/nl2api")
    logger.info("Connecting to database...")
    pool = await asyncpg.create_pool(db_url)

    try:
        if mode == "full":
            result = await run_full_ingestion(pool, config, checkpoint_mgr, args, state)
        else:
            result = await run_delta_ingestion(pool, config, args, state)

        # Show final stats
        async with pool.acquire() as conn:
            entity_count = await conn.fetchval(
                "SELECT COUNT(*) FROM entities WHERE data_source = 'gleif'"
            )
            alias_count = await conn.fetchval(
                """
                SELECT COUNT(*) FROM entity_aliases a
                JOIN entities e ON a.entity_id = e.id
                WHERE e.data_source = 'gleif'
                """
            )

        print("\n" + "=" * 60)
        print("INGESTION COMPLETE")
        print("=" * 60)
        print(f"Mode: {result['mode'].upper()}")
        if result.get("dry_run"):
            print("[DRY RUN - No data modified]")
        else:
            print(f"Total GLEIF entities: {entity_count:,}")
            print(f"Total GLEIF aliases: {alias_count:,}")
            if result.get("loaded"):
                print(f"Records loaded: {result['loaded']:,}")
            if result.get("inserted"):
                print(f"New entities: {result['inserted']:,}")
            if result.get("updated"):
                print(f"Updated entities: {result['updated']:,}")
            if result.get("retired"):
                print(f"Retired entities: {result['retired']:,}")

    except Exception as e:
        logger.exception("Ingestion failed: %s", e)
        sys.exit(1)
    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
