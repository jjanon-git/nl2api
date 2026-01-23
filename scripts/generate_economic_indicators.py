#!/usr/bin/env python3
"""
Generate and Index Economic Indicators into RAG Database

Generates a large set of economic indicator mnemonics and descriptions
and indexes them into the rag_documents table.

Usage:
    .venv/bin/python scripts/generate_economic_indicators.py --limit 1000
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

# Sample data for generation
COUNTRIES = {
    "US": "United States",
    "UK": "United Kingdom",
    "JP": "Japan",
    "DE": "Germany",
    "FR": "France",
    "CN": "China",
    "IN": "India",
    "BR": "Brazil",
    "CA": "Canada",
    "AU": "Australia",
    "EU": "Eurozone",
    "CH": "Switzerland",
    "IT": "Italy",
    "ES": "Spain",
    "MX": "Mexico",
    "KR": "South Korea",
    "RU": "Russia",
    "SA": "Saudi Arabia",
    "ZA": "South Africa",
    "TR": "Turkey",
    "ID": "Indonesia",
    "NL": "Netherlands",
    "SE": "Sweden",
    "PL": "Poland",
    "SG": "Singapore",
    "HK": "Hong Kong",
    "TW": "Taiwan",
    "TH": "Thailand",
    "MY": "Malaysia",
    "VN": "Vietnam",
    "PH": "Philippines",
    "EG": "Egypt",
}

INDICATORS = [
    ("GDP", "Gross Domestic Product", ["growth", "economy", "output"]),
    ("CPI", "Consumer Price Index", ["inflation", "consumer prices", "cost of living"]),
    ("UNR", "Unemployment Rate", ["jobs", "labor market", "employment"]),
    ("INT", "Central Bank Interest Rate", ["policy rate", "monetary policy"]),
    ("PMI", "Purchasing Managers Index", ["manufacturing", "business activity"]),
    ("RTL", "Retail Sales", ["consumer spending", "retail"]),
    ("IP", "Industrial Production", ["factory output", "manufacturing"]),
    ("HOU", "Housing Starts", ["construction", "real estate", "housing"]),
    ("CONF", "Consumer Confidence", ["sentiment", "consumer survey"]),
    ("TRADE", "Trade Balance", ["exports", "imports", "current account"]),
    ("PPI", "Producer Price Index", ["wholesale inflation", "input prices"]),
    ("M2", "Money Supply", ["monetary aggregate", "liquidity"]),
    ("EMP", "Employment Change", ["payrolls", "new jobs"]),
    ("WAGE", "Wage Growth", ["earnings", "labor cost"]),
    ("GFCF", "Gross Fixed Capital Formation", ["investment", "capital spending"]),
    ("BUD", "Budget Balance", ["fiscal deficit", "government spending"]),
]

SUB_TYPES = [
    ("", "", []),
    ("CORE", "Core", ["excluding food and energy"]),
    ("YOY", "Year-on-Year", ["annual growth"]),
    ("MOM", "Month-on-Month", ["monthly change"]),
    ("QOQ", "Quarter-on-Quarter", ["quarterly change"]),
    ("SA", "Seasonally Adjusted", ["adjusted"]),
    ("NSA", "Non-Seasonally Adjusted", ["raw data"]),
    ("REV", "Revised", ["updated"]),
    ("EST", "Estimate", ["forecast"]),
    ("PROJ", "Projection", ["target"]),
]


def generate_indicators(limit: int):
    """Generate a list of economic indicator documents."""
    from src.nl2api.rag.indexer import EconomicIndicatorDocument

    docs = []
    count = 0

    # Nested loops to generate combinations
    for country_code, country_name in COUNTRIES.items():
        for ind_code, ind_name, ind_hints in INDICATORS:
            for sub_code, sub_name, sub_hints in SUB_TYPES:
                if count >= limit:
                    break

                mnemonic = f"{country_code}{ind_code}{sub_code}"
                description = f"{country_name} {sub_name} {ind_name}".replace("  ", " ").strip()

                hints = list(
                    set(ind_hints + sub_hints + [country_name.lower(), country_code.lower()])
                )

                docs.append(
                    EconomicIndicatorDocument(
                        mnemonic=mnemonic,
                        description=description,
                        country=country_name,
                        indicator_type=f"{sub_name} {ind_name}".strip(),
                        natural_language_hints=hints,
                        metadata={"source": "generator"},
                    )
                )
                count += 1
            if count >= limit:
                break
        if count >= limit:
            break

    # 3. If still need more, add a secondary suffix (e.g. Regions/Cities)
    REGIONS = ["North", "South", "East", "West", "Central", "Urban", "Rural"]
    if count < limit:
        for country_code, country_name in COUNTRIES.items():
            for region in REGIONS:
                for ind_code, ind_name, ind_hints in INDICATORS:
                    if count >= limit:
                        break

                    mnemonic = f"{country_code}{region[:1].upper()}{ind_code}"
                    description = f"{country_name} {region} {ind_name}"
                    hints = ind_hints + [country_name.lower(), region.lower()]

                    docs.append(
                        EconomicIndicatorDocument(
                            mnemonic=mnemonic,
                            description=description,
                            country=country_name,
                            indicator_type=f"{region} {ind_name}",
                            natural_language_hints=hints,
                            metadata={"source": "generator"},
                        )
                    )
                    count += 1
                if count >= limit:
                    break
            if count >= limit:
                break

    return docs


async def main():
    parser = argparse.ArgumentParser(description="Generate and index economic indicators")
    parser.add_argument("--limit", type=int, default=100, help="Number of indicators to generate")
    parser.add_argument("--batch-size", type=int, default=50, help="Batch size for indexing")
    parser.add_argument("--no-embeddings", action="store_true", help="Skip embedding generation")
    args = parser.parse_args()

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

    from src.nl2api.rag.indexer import RAGIndexer

    print(f"Generating {args.limit} economic indicators...")
    docs = generate_indicators(args.limit)
    print(f"Generated {len(docs)} indicators.")

    # Connect to database
    import asyncpg

    db_url = os.environ.get("DATABASE_URL", "postgresql://nl2api:nl2api@localhost:5432/nl2api")

    print("\nConnecting to database...")
    try:
        pool = await asyncpg.create_pool(db_url)
    except Exception as e:
        print(f"ERROR: Could not connect to database: {e}")
        return

    try:
        indexer = RAGIndexer(pool)

        if not args.no_embeddings:
            openai_key = os.environ.get("OPENAI_API_KEY")
            if not openai_key:
                print("ERROR: OPENAI_API_KEY not set! Use --no-embeddings to skip.")
                return

            from src.nl2api.rag.retriever import OpenAIEmbedder

            embedder = OpenAIEmbedder(api_key=openai_key)
            indexer.set_embedder(embedder)
            print("Embedder initialized.")

        print(f"\nIndexing {len(docs)} indicators...")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
        ) as progress:
            task = progress.add_task("[cyan]Indexing economic indicators", total=len(docs))

            def callback(processed, total):
                progress.update(task, completed=processed, total=total)

            doc_ids = await indexer.index_economic_indicators_batch(
                docs,
                generate_embeddings=not args.no_embeddings,
                batch_size=args.batch_size,
                progress_callback=callback,
            )

        print(f"\nSuccessfully indexed {len(doc_ids)} economic indicators.")

    finally:
        await pool.close()


if __name__ == "__main__":
    asyncio.run(main())
