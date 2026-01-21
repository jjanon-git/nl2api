"""
Entity Resolution Generator - Generates test cases for entity resolution evaluation.

Tests the entity resolver's ability to map company names, tickers, and identifiers
to canonical RICs (Reuters Instrument Codes).

Target: ~5,000 test cases
- exact_match: 800 cases (exact name/ticker/RIC from DB)
- ticker_lookup: 500 cases (pure ticker resolution, including short tickers)
- alias_match: 600 cases (trade names, former names)
- suffix_variations: 400 cases (Inc, Corp, Ltd, SE, AG, GmbH, N.V.)
- fuzzy_misspellings: 500 cases (typos)
- abbreviations: 300 cases (IBM, J&J, P&G)
- international: 600 cases (non-US companies)
- ambiguous: 400 cases (same name, different companies)
- ticker_collisions: 200 cases (same ticker, multiple exchanges)
- edge_case_names: 300 cases (3M, Coca-Cola, McDonald's, AT&T)
- negative_cases: 400 cases (should NOT resolve)

Usage:
    python -m scripts.generators.entity_resolution_generator \
        --output tests/fixtures/lseg/generated/entity_resolution/entity_resolution.json

Requires:
    - PostgreSQL database with entities table (from 007_entities.sql migration)
    - DATABASE_URL environment variable or --db-url argument
"""

import argparse
import asyncio
import hashlib
import json
import os
import random
import string
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


# Lazy import asyncpg to allow import without postgres deps
def _get_asyncpg():
    try:
        import asyncpg
        return asyncpg
    except ImportError:
        raise ImportError(
            "asyncpg is required for entity resolution generation. "
            "Install with: pip install asyncpg"
        )


@dataclass
class EntityTestCase:
    """A single entity resolution test case."""
    id: str
    nl_query: str
    expected_tool_calls: list[dict[str, Any]]
    expected_response: None  # Always null for entity resolution
    expected_nl_response: None  # Always null for entity resolution
    complexity: int
    category: str
    subcategory: str
    tags: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EntityResolutionGenerator:
    """
    Generator for entity resolution test cases using database.

    Unlike other generators, this one queries the PostgreSQL database
    directly to get real entity data from GLEIF and SEC EDGAR.
    """

    # Query templates for natural language queries that contain entity references
    NL_QUERY_TEMPLATES = [
        "What is {entity}'s stock price?",
        "Get {entity}'s revenue for last quarter",
        "Show me {entity}'s P/E ratio",
        "What is {entity}'s market cap?",
        "Get financial data for {entity}",
        "Show {entity} historical prices",
        "{entity} stock information",
        "How is {entity} performing?",
        "What's {entity}'s dividend yield?",
        "Get {entity}'s EPS",
    ]

    # Typo patterns for fuzzy matching tests
    TYPO_PATTERNS = [
        "swap_adjacent",     # Microsoft -> Mircosoft
        "delete_char",       # Apple -> Aple
        "double_char",       # Google -> Googgle
        "wrong_char",        # Tesla -> Telsa
        "missing_space",     # Home Depot -> HomeDepot
        "extra_space",       # IBM -> I B M
    ]

    # Common words that should NOT be resolved
    NEGATIVE_COMMON_WORDS = [
        "THE", "FOR", "AND", "EPS", "ROE", "GDP", "IPO", "CEO", "CFO",
        "NYSE", "NASDAQ", "ETF", "USD", "EUR", "GBP", "JPY",
        "BUY", "SELL", "HOLD", "LONG", "SHORT",
        "Q1", "Q2", "Q3", "Q4", "FY", "YTD", "MOM", "YOY",
        "DIV", "VOL", "AVG", "MAX", "MIN", "SUM",
    ]

    # Fictional company names for negative tests
    NEGATIVE_FICTIONAL = [
        "XYZ Corp International",
        "Acme Holdings LLC",
        "Globex Corporation",
        "Initech Systems",
        "Hooli Inc",
        "Pied Piper",
        "Massive Dynamic",
        "Umbrella Corporation",
        "Soylent Corp",
        "Cyberdyne Systems",
        "Tyrell Corporation",
        "Weyland-Yutani",
        "Oscorp Industries",
        "LexCorp",
        "Wayne Enterprises",
    ]

    def __init__(self, db_url: str):
        """
        Initialize the generator.

        Args:
            db_url: PostgreSQL connection URL
        """
        self.db_url = db_url
        self.generated_ids: set[str] = set()
        self.category = "entity_resolution"

    def _generate_id(self, nl_query: str, subcategory: str) -> str:
        """Generate a unique test case ID."""
        content = f"{self.category}:{subcategory}:{nl_query}"
        hash_str = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"entity_resolution_{hash_str}"

    def _create_test_case(
        self,
        nl_query: str,
        subcategory: str,
        input_entity: str,
        expected_ric: str | None,
        tags: list[str],
        metadata: dict[str, Any],
    ) -> EntityTestCase | None:
        """Create a test case if it doesn't already exist."""
        test_id = self._generate_id(nl_query, subcategory)

        if test_id in self.generated_ids:
            return None

        self.generated_ids.add(test_id)

        # Build expected tool call (using tool_name per CONTRACTS.py)
        if expected_ric:
            expected_tool_calls = [{
                "tool_name": "get_data",
                "arguments": {
                    "tickers": [expected_ric],
                    "fields": ["TR.Revenue"]  # Placeholder field
                }
            }]
        else:
            # Negative case - should not resolve
            expected_tool_calls = []

        return EntityTestCase(
            id=test_id,
            nl_query=nl_query,
            expected_tool_calls=expected_tool_calls,
            expected_response=None,
            expected_nl_response=None,
            complexity=1,
            category=self.category,
            subcategory=subcategory,
            tags=tags,
            metadata={
                "input_entity": input_entity,
                "expected_ric": expected_ric,
                **metadata
            }
        )

    def _select_random_template(self) -> str:
        """Select a random NL query template."""
        return random.choice(self.NL_QUERY_TEMPLATES)

    def _introduce_typo(self, name: str, pattern: str) -> str:
        """Introduce a typo into the name based on pattern."""
        if len(name) < 3:
            return name

        if pattern == "swap_adjacent":
            # Swap two adjacent characters
            idx = random.randint(1, len(name) - 2)
            chars = list(name)
            chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            return "".join(chars)

        elif pattern == "delete_char":
            # Delete a character
            idx = random.randint(1, len(name) - 2)
            return name[:idx] + name[idx + 1:]

        elif pattern == "double_char":
            # Double a character
            idx = random.randint(0, len(name) - 1)
            return name[:idx] + name[idx] + name[idx:]

        elif pattern == "wrong_char":
            # Replace with a nearby keyboard character
            keyboard_nearby = {
                'a': 'sq', 'b': 'vn', 'c': 'xv', 'd': 'sf', 'e': 'wr',
                'f': 'dg', 'g': 'fh', 'h': 'gj', 'i': 'uo', 'j': 'hk',
                'k': 'jl', 'l': 'k;', 'm': 'n,', 'n': 'bm', 'o': 'ip',
                'p': 'o[', 'q': 'wa', 'r': 'et', 's': 'ad', 't': 'ry',
                'u': 'yi', 'v': 'cb', 'w': 'qe', 'x': 'zc', 'y': 'tu',
                'z': 'xa'
            }
            # Find a letter to replace
            letters = [(i, c) for i, c in enumerate(name.lower()) if c.isalpha()]
            if letters:
                idx, char = random.choice(letters)
                if char in keyboard_nearby:
                    replacement = random.choice(keyboard_nearby[char])
                    # Preserve case
                    if name[idx].isupper():
                        replacement = replacement.upper()
                    return name[:idx] + replacement + name[idx + 1:]

        elif pattern == "missing_space":
            # Remove a space
            if " " in name:
                idx = name.index(" ")
                return name[:idx] + name[idx + 1:]

        elif pattern == "extra_space":
            # Add a space
            idx = random.randint(1, len(name) - 1)
            return name[:idx] + " " + name[idx:]

        return name

    async def _generate_exact_matches(
        self, pool, count: int = 800
    ) -> list[EntityTestCase]:
        """Generate exact match test cases from database."""
        test_cases = []

        # Query entities with RICs
        query = """
            SELECT
                primary_name, display_name, ric, ticker, lei, country_code,
                exchange, data_source
            FROM entities
            WHERE ric IS NOT NULL
              AND entity_status = 'active'
              AND is_public = true
            ORDER BY RANDOM()
            LIMIT $1
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, count)

        for row in rows:
            # Use display_name if available, otherwise primary_name
            entity_name = row["display_name"] or row["primary_name"]

            template = self._select_random_template()
            nl_query = template.format(entity=entity_name)

            tags = ["exact_match", row["data_source"]]
            if row["country_code"] == "US":
                tags.append("us_company")
            else:
                tags.append("international")

            metadata = {
                "expected_lei": row["lei"],
                "expected_ticker": row["ticker"],
                "match_type": "exact",
                "confidence_threshold": 0.95,
                "country_code": row["country_code"],
                "exchange": row["exchange"],
                "data_source": row["data_source"],
            }

            tc = self._create_test_case(
                nl_query=nl_query,
                subcategory="exact_match",
                input_entity=entity_name,
                expected_ric=row["ric"],
                tags=tags,
                metadata=metadata,
            )
            if tc:
                test_cases.append(tc)

        return test_cases

    async def _generate_ticker_lookups(
        self, pool, count: int = 500
    ) -> list[EntityTestCase]:
        """Generate ticker lookup test cases, emphasizing short tickers."""
        test_cases = []

        # Prioritize short tickers (1-2 chars) which are a known weakness
        query = """
            SELECT ticker, ric, primary_name, exchange, country_code, data_source
            FROM entities
            WHERE ticker IS NOT NULL AND ric IS NOT NULL
              AND entity_status = 'active'
            ORDER BY
                CASE WHEN LENGTH(ticker) <= 2 THEN 0 ELSE 1 END,
                RANDOM()
            LIMIT $1
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, count)

        for row in rows:
            ticker = row["ticker"]

            # Use ticker directly as the entity reference
            template = self._select_random_template()
            nl_query = template.format(entity=ticker)

            is_short = len(ticker) <= 2
            tags = ["ticker_lookup", row["data_source"]]
            if is_short:
                tags.append("short_ticker")

            metadata = {
                "ticker_length": len(ticker),
                "expected_ticker": ticker,
                "match_type": "ticker",
                "confidence_threshold": 1.0,
                "country_code": row["country_code"],
                "exchange": row["exchange"],
                "data_source": row["data_source"],
            }

            tc = self._create_test_case(
                nl_query=nl_query,
                subcategory="ticker_lookup",
                input_entity=ticker,
                expected_ric=row["ric"],
                tags=tags,
                metadata=metadata,
            )
            if tc:
                test_cases.append(tc)

        return test_cases

    async def _generate_alias_matches(
        self, pool, count: int = 600
    ) -> list[EntityTestCase]:
        """Generate alias match test cases (legal names, ticker aliases)."""
        test_cases = []

        # Use actual alias types in database: generated, legal_name, ticker
        query = """
            SELECT a.alias, a.alias_type, e.ric, e.primary_name, e.country_code,
                   e.exchange, e.data_source
            FROM entity_aliases a
            JOIN entities e ON a.entity_id = e.id
            WHERE e.ric IS NOT NULL
              AND e.entity_status = 'active'
              AND a.alias_type IN ('legal_name', 'ticker', 'generated')
              AND lower(a.alias) != lower(e.primary_name)
              AND length(a.alias) > 3
            ORDER BY RANDOM()
            LIMIT $1
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, count)

        for row in rows:
            alias = row["alias"]

            template = self._select_random_template()
            nl_query = template.format(entity=alias)

            tags = ["alias_match", row["alias_type"], row["data_source"]]

            metadata = {
                "alias_type": row["alias_type"],
                "canonical_name": row["primary_name"],
                "match_type": "alias",
                "confidence_threshold": 0.90,
                "country_code": row["country_code"],
                "exchange": row["exchange"],
                "data_source": row["data_source"],
            }

            tc = self._create_test_case(
                nl_query=nl_query,
                subcategory="alias_match",
                input_entity=alias,
                expected_ric=row["ric"],
                tags=tags,
                metadata=metadata,
            )
            if tc:
                test_cases.append(tc)

        return test_cases

    async def _generate_suffix_variations(
        self, pool, count: int = 400
    ) -> list[EntityTestCase]:
        """Generate suffix variation test cases (Inc, Corp, Ltd, SE, AG, GmbH, N.V.)."""
        test_cases = []

        # Find entities with legal suffixes
        query = """
            SELECT primary_name, display_name, ric, ticker, country_code,
                   exchange, data_source
            FROM entities
            WHERE ric IS NOT NULL
              AND entity_status = 'active'
              AND is_public = true
              AND (
                primary_name ILIKE '% Inc' OR primary_name ILIKE '% Inc.' OR
                primary_name ILIKE '% Corp' OR primary_name ILIKE '% Corp.' OR
                primary_name ILIKE '% Corporation' OR
                primary_name ILIKE '% Ltd' OR primary_name ILIKE '% Ltd.' OR
                primary_name ILIKE '% Limited' OR
                primary_name ILIKE '% Co' OR primary_name ILIKE '% Co.' OR
                primary_name ILIKE '% PLC' OR
                primary_name ILIKE '% SE' OR
                primary_name ILIKE '% AG' OR
                primary_name ILIKE '% GmbH' OR
                primary_name ILIKE '% N.V.' OR primary_name ILIKE '% NV' OR
                primary_name ILIKE '% SARL' OR
                primary_name ILIKE '% S.A.' OR primary_name ILIKE '% SA'
              )
            ORDER BY RANDOM()
            LIMIT $1
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, count)

        # Suffixes to strip for test variations
        suffixes = [
            " Inc", " Inc.", " Corp", " Corp.", " Corporation",
            " Ltd", " Ltd.", " Limited", " Co", " Co.",
            " PLC", " SE", " AG", " GmbH", " N.V.", " NV",
            " SARL", " S.A.", " SA", " LLC"
        ]

        for row in rows:
            full_name = row["primary_name"]

            # Create stripped version
            stripped_name = full_name
            for suffix in suffixes:
                if stripped_name.lower().endswith(suffix.lower()):
                    stripped_name = stripped_name[:-len(suffix)].strip()
                    break

            # Only create test if we actually stripped something
            if stripped_name == full_name:
                continue

            template = self._select_random_template()
            nl_query = template.format(entity=stripped_name)

            tags = ["suffix_variation", row["data_source"]]

            metadata = {
                "full_name": full_name,
                "stripped_name": stripped_name,
                "match_type": "suffix_stripped",
                "confidence_threshold": 0.80,
                "country_code": row["country_code"],
                "exchange": row["exchange"],
                "data_source": row["data_source"],
            }

            tc = self._create_test_case(
                nl_query=nl_query,
                subcategory="suffix_variations",
                input_entity=stripped_name,
                expected_ric=row["ric"],
                tags=tags,
                metadata=metadata,
            )
            if tc:
                test_cases.append(tc)

        return test_cases

    async def _generate_fuzzy_misspellings(
        self, pool, count: int = 500
    ) -> list[EntityTestCase]:
        """Generate fuzzy misspelling test cases with programmatic typos."""
        test_cases = []

        # Get well-known companies with longer names (easier to typo)
        query = """
            SELECT primary_name, display_name, ric, ticker, country_code,
                   exchange, data_source
            FROM entities
            WHERE ric IS NOT NULL
              AND entity_status = 'active'
              AND is_public = true
              AND LENGTH(COALESCE(display_name, primary_name)) >= 5
            ORDER BY RANDOM()
            LIMIT $1
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, count)

        for row in rows:
            name = row["display_name"] or row["primary_name"]

            # Choose a random typo pattern
            pattern = random.choice(self.TYPO_PATTERNS)
            misspelled = self._introduce_typo(name, pattern)

            # Skip if typo didn't change anything
            if misspelled == name:
                continue

            template = self._select_random_template()
            nl_query = template.format(entity=misspelled)

            tags = ["fuzzy_misspelling", pattern, row["data_source"]]

            metadata = {
                "original_name": name,
                "misspelled_name": misspelled,
                "typo_pattern": pattern,
                "match_type": "fuzzy",
                "confidence_threshold": 0.70,
                "country_code": row["country_code"],
                "exchange": row["exchange"],
                "data_source": row["data_source"],
            }

            tc = self._create_test_case(
                nl_query=nl_query,
                subcategory="fuzzy_misspellings",
                input_entity=misspelled,
                expected_ric=row["ric"],
                tags=tags,
                metadata=metadata,
            )
            if tc:
                test_cases.append(tc)

        return test_cases

    async def _generate_abbreviations(
        self, pool, count: int = 300
    ) -> list[EntityTestCase]:
        """Generate abbreviation test cases (IBM, J&J, P&G, etc.)."""
        test_cases = []

        # Look for aliases that are abbreviations
        query = """
            SELECT a.alias, e.ric, e.primary_name, e.country_code,
                   e.exchange, e.data_source
            FROM entity_aliases a
            JOIN entities e ON a.entity_id = e.id
            WHERE e.ric IS NOT NULL
              AND e.entity_status = 'active'
              AND a.alias_type = 'abbreviation'
            ORDER BY RANDOM()
            LIMIT $1
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, count)

        # If not enough aliases, generate some known abbreviations
        if len(rows) < count // 2:
            # Known abbreviations (hardcoded fallback)
            known_abbrevs = [
                ("IBM", "IBM.N"),
                ("GE", "GE.N"),
                ("HP", "HPQ.N"),
                ("3M", "MMM.N"),
            ]
            for abbrev, ric in known_abbrevs[:count - len(rows)]:
                template = self._select_random_template()
                nl_query = template.format(entity=abbrev)

                tc = self._create_test_case(
                    nl_query=nl_query,
                    subcategory="abbreviations",
                    input_entity=abbrev,
                    expected_ric=ric,
                    tags=["abbreviation", "hardcoded"],
                    metadata={
                        "match_type": "abbreviation",
                        "confidence_threshold": 0.85,
                    },
                )
                if tc:
                    test_cases.append(tc)

        for row in rows:
            abbrev = row["alias"]

            template = self._select_random_template()
            nl_query = template.format(entity=abbrev)

            tags = ["abbreviation", row["data_source"]]

            metadata = {
                "full_name": row["primary_name"],
                "match_type": "abbreviation",
                "confidence_threshold": 0.85,
                "country_code": row["country_code"],
                "exchange": row["exchange"],
                "data_source": row["data_source"],
            }

            tc = self._create_test_case(
                nl_query=nl_query,
                subcategory="abbreviations",
                input_entity=abbrev,
                expected_ric=row["ric"],
                tags=tags,
                metadata=metadata,
            )
            if tc:
                test_cases.append(tc)

        return test_cases

    async def _generate_international(
        self, pool, count: int = 600
    ) -> list[EntityTestCase]:
        """Generate international company test cases (non-US)."""
        test_cases = []

        # Get entities from different regions
        # Note: Most international entities from GLEIF don't have RICs
        # So we generate test cases without expected RICs (for future resolver improvement)
        query = """
            SELECT primary_name, display_name, ric, ticker, country_code,
                   lei, data_source
            FROM entities
            WHERE entity_status = 'active'
              AND country_code IS NOT NULL
              AND country_code != 'US'
              AND length(primary_name) > 3
            ORDER BY
              CASE WHEN ric IS NOT NULL THEN 0 ELSE 1 END,
              country_code,
              RANDOM()
            LIMIT $1
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, count)

        # Map country codes to regions
        region_map = {
            "GB": "europe", "DE": "europe", "FR": "europe", "CH": "europe",
            "NL": "europe", "ES": "europe", "IT": "europe", "SE": "europe",
            "JP": "asia_pacific", "CN": "asia_pacific", "HK": "asia_pacific",
            "KR": "asia_pacific", "TW": "asia_pacific", "SG": "asia_pacific",
            "AU": "asia_pacific", "IN": "asia_pacific",
            "CA": "americas", "BR": "americas", "MX": "americas",
        }

        for row in rows:
            name = row["display_name"] or row["primary_name"]

            template = self._select_random_template()
            nl_query = template.format(entity=name)

            country = row["country_code"]
            region = region_map.get(country, "other")

            tags = ["international", country, region, row["data_source"]]

            metadata = {
                "match_type": "international",
                "confidence_threshold": 0.60,
                "country_code": country,
                "region": region,
                "lei": row["lei"],
                "data_source": row["data_source"],
            }

            tc = self._create_test_case(
                nl_query=nl_query,
                subcategory="international",
                input_entity=name,
                expected_ric=row["ric"],
                tags=tags,
                metadata=metadata,
            )
            if tc:
                test_cases.append(tc)

        return test_cases

    async def _generate_ambiguous(
        self, pool, count: int = 400
    ) -> list[EntityTestCase]:
        """Generate ambiguous entity test cases (same name, different companies)."""
        test_cases = []

        # Find aliases that map to multiple entities
        # Use min(a.alias) to get a representative alias for each group
        query = """
            SELECT min(a.alias) as alias, array_agg(DISTINCT e.ric) as rics,
                   array_agg(DISTINCT e.primary_name) as names
            FROM entity_aliases a
            JOIN entities e ON a.entity_id = e.id
            WHERE e.ric IS NOT NULL
              AND e.entity_status = 'active'
            GROUP BY lower(a.alias)
            HAVING COUNT(DISTINCT e.id) > 1
            LIMIT $1
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, count)

        for row in rows:
            alias = row["alias"]
            rics = row["rics"]
            names = row["names"]

            template = self._select_random_template()
            nl_query = template.format(entity=alias)

            tags = ["ambiguous", f"candidates_{len(rics)}"]

            # For ambiguous, we expect one of the possible RICs
            # The resolver should pick the most likely one
            primary_ric = rics[0] if rics else None

            metadata = {
                "match_type": "ambiguous",
                "candidate_rics": rics,
                "candidate_names": names,
                "num_candidates": len(rics),
                "confidence_threshold": 0.50,
            }

            tc = self._create_test_case(
                nl_query=nl_query,
                subcategory="ambiguous",
                input_entity=alias,
                expected_ric=primary_ric,
                tags=tags,
                metadata=metadata,
            )
            if tc:
                test_cases.append(tc)

        return test_cases

    async def _generate_ticker_collisions(
        self, pool, count: int = 200
    ) -> list[EntityTestCase]:
        """Generate ticker collision test cases (same ticker, multiple exchanges)."""
        test_cases = []

        # Find tickers with multiple exchanges
        query = """
            SELECT ticker, array_agg(DISTINCT ric) as rics,
                   array_agg(DISTINCT exchange) as exchanges,
                   array_agg(DISTINCT primary_name) as names
            FROM entities
            WHERE ticker IS NOT NULL
              AND ric IS NOT NULL
              AND entity_status = 'active'
              AND exchange IS NOT NULL
            GROUP BY ticker
            HAVING COUNT(DISTINCT exchange) > 1
            LIMIT $1
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, count)

        for row in rows:
            ticker = row["ticker"]
            rics = row["rics"]
            exchanges = row["exchanges"]
            names = row["names"]

            template = self._select_random_template()
            nl_query = template.format(entity=ticker)

            tags = ["ticker_collision", f"exchanges_{len(exchanges)}"]

            # For collisions, the resolver should prefer US exchange if available
            # or return the most liquid one
            primary_ric = rics[0] if rics else None

            metadata = {
                "match_type": "ticker_collision",
                "candidate_rics": rics,
                "candidate_exchanges": exchanges,
                "candidate_names": names,
                "num_exchanges": len(exchanges),
                "confidence_threshold": 0.50,
            }

            tc = self._create_test_case(
                nl_query=nl_query,
                subcategory="ticker_collisions",
                input_entity=ticker,
                expected_ric=primary_ric,
                tags=tags,
                metadata=metadata,
            )
            if tc:
                test_cases.append(tc)

        return test_cases

    async def _generate_edge_case_names(
        self, pool, count: int = 300
    ) -> list[EntityTestCase]:
        """Generate edge case name tests (3M, Coca-Cola, McDonald's, AT&T)."""
        test_cases = []

        # Find entities with special characters in names
        query = """
            SELECT primary_name, display_name, ric, ticker, country_code,
                   exchange, data_source
            FROM entities
            WHERE ric IS NOT NULL
              AND entity_status = 'active'
              AND is_public = true
              AND (
                primary_name ~ '^[0-9]' OR        -- Starts with number (3M)
                primary_name LIKE '%-%' OR        -- Has hyphen (Coca-Cola)
                primary_name LIKE '%''%' OR       -- Has apostrophe (McDonald's)
                primary_name LIKE '%&%' OR        -- Has ampersand (AT&T)
                primary_name ~ '[^a-zA-Z0-9\\s\\-\\.,''&]'  -- Other special chars
              )
            ORDER BY RANDOM()
            LIMIT $1
        """

        async with pool.acquire() as conn:
            rows = await conn.fetch(query, count)

        for row in rows:
            name = row["display_name"] or row["primary_name"]

            template = self._select_random_template()
            nl_query = template.format(entity=name)

            # Determine the edge case type
            edge_type = []
            if name[0].isdigit():
                edge_type.append("starts_with_number")
            if "-" in name:
                edge_type.append("has_hyphen")
            if "'" in name:
                edge_type.append("has_apostrophe")
            if "&" in name:
                edge_type.append("has_ampersand")

            tags = ["edge_case_name", row["data_source"]] + edge_type

            metadata = {
                "match_type": "edge_case",
                "edge_case_types": edge_type,
                "confidence_threshold": 0.60,
                "country_code": row["country_code"],
                "exchange": row["exchange"],
                "data_source": row["data_source"],
            }

            tc = self._create_test_case(
                nl_query=nl_query,
                subcategory="edge_case_names",
                input_entity=name,
                expected_ric=row["ric"],
                tags=tags,
                metadata=metadata,
            )
            if tc:
                test_cases.append(tc)

        return test_cases

    def _generate_negative_cases(self, count: int = 400) -> list[EntityTestCase]:
        """Generate negative test cases (should NOT resolve)."""
        test_cases = []

        # 1. Common words that look like tickers
        for word in self.NEGATIVE_COMMON_WORDS:
            if len(test_cases) >= count // 3:
                break

            template = self._select_random_template()
            nl_query = template.format(entity=word)

            tc = self._create_test_case(
                nl_query=nl_query,
                subcategory="negative_cases",
                input_entity=word,
                expected_ric=None,  # Should NOT resolve
                tags=["negative", "common_word"],
                metadata={
                    "match_type": "negative",
                    "negative_type": "common_word",
                    "should_resolve": False,
                },
            )
            if tc:
                test_cases.append(tc)

        # 2. Fictional companies
        for company in self.NEGATIVE_FICTIONAL:
            if len(test_cases) >= 2 * count // 3:
                break

            template = self._select_random_template()
            nl_query = template.format(entity=company)

            tc = self._create_test_case(
                nl_query=nl_query,
                subcategory="negative_cases",
                input_entity=company,
                expected_ric=None,
                tags=["negative", "fictional"],
                metadata={
                    "match_type": "negative",
                    "negative_type": "fictional",
                    "should_resolve": False,
                },
            )
            if tc:
                test_cases.append(tc)

        # 3. Random gibberish that looks like company names
        while len(test_cases) < count:
            # Generate random "company" name
            prefix = random.choice(["Global", "United", "American", "First", "Premier"])
            suffix = random.choice(["Holdings", "Industries", "Group", "Partners", "Capital"])
            random_part = "".join(random.choices(string.ascii_uppercase, k=3))
            fake_company = f"{prefix} {random_part} {suffix}"

            template = self._select_random_template()
            nl_query = template.format(entity=fake_company)

            tc = self._create_test_case(
                nl_query=nl_query,
                subcategory="negative_cases",
                input_entity=fake_company,
                expected_ric=None,
                tags=["negative", "generated_gibberish"],
                metadata={
                    "match_type": "negative",
                    "negative_type": "generated_gibberish",
                    "should_resolve": False,
                },
            )
            if tc:
                test_cases.append(tc)

        return test_cases

    async def generate(self) -> list[EntityTestCase]:
        """Generate all entity resolution test cases."""
        asyncpg = _get_asyncpg()

        print("Connecting to database...")
        pool = await asyncpg.create_pool(self.db_url, min_size=2, max_size=5)

        try:
            # Check entity counts
            async with pool.acquire() as conn:
                counts = await conn.fetchrow("""
                    SELECT
                        COUNT(*) as total,
                        COUNT(*) FILTER (WHERE ric IS NOT NULL) as with_ric,
                        COUNT(*) FILTER (WHERE is_public) as public,
                        COUNT(DISTINCT country_code) as countries
                    FROM entities WHERE entity_status = 'active'
                """)
                print(f"Database stats: {counts['total']:,} entities, "
                      f"{counts['with_ric']:,} with RIC, "
                      f"{counts['public']:,} public, "
                      f"{counts['countries']} countries")

            test_cases = []

            # Generate each subcategory
            print("\nGenerating exact_match (800)...")
            test_cases.extend(await self._generate_exact_matches(pool, 800))
            print(f"  Generated {len(test_cases)} cases")

            print("Generating ticker_lookup (500)...")
            prev_count = len(test_cases)
            test_cases.extend(await self._generate_ticker_lookups(pool, 500))
            print(f"  Generated {len(test_cases) - prev_count} cases")

            print("Generating alias_match (600)...")
            prev_count = len(test_cases)
            test_cases.extend(await self._generate_alias_matches(pool, 600))
            print(f"  Generated {len(test_cases) - prev_count} cases")

            print("Generating suffix_variations (400)...")
            prev_count = len(test_cases)
            test_cases.extend(await self._generate_suffix_variations(pool, 400))
            print(f"  Generated {len(test_cases) - prev_count} cases")

            print("Generating fuzzy_misspellings (500)...")
            prev_count = len(test_cases)
            test_cases.extend(await self._generate_fuzzy_misspellings(pool, 500))
            print(f"  Generated {len(test_cases) - prev_count} cases")

            print("Generating abbreviations (300)...")
            prev_count = len(test_cases)
            test_cases.extend(await self._generate_abbreviations(pool, 300))
            print(f"  Generated {len(test_cases) - prev_count} cases")

            print("Generating international (600)...")
            prev_count = len(test_cases)
            test_cases.extend(await self._generate_international(pool, 600))
            print(f"  Generated {len(test_cases) - prev_count} cases")

            print("Generating ambiguous (400)...")
            prev_count = len(test_cases)
            test_cases.extend(await self._generate_ambiguous(pool, 400))
            print(f"  Generated {len(test_cases) - prev_count} cases")

            print("Generating ticker_collisions (200)...")
            prev_count = len(test_cases)
            test_cases.extend(await self._generate_ticker_collisions(pool, 200))
            print(f"  Generated {len(test_cases) - prev_count} cases")

            print("Generating edge_case_names (300)...")
            prev_count = len(test_cases)
            test_cases.extend(await self._generate_edge_case_names(pool, 300))
            print(f"  Generated {len(test_cases) - prev_count} cases")

            print("Generating negative_cases (400)...")
            prev_count = len(test_cases)
            test_cases.extend(self._generate_negative_cases(400))
            print(f"  Generated {len(test_cases) - prev_count} cases")

            print(f"\nTotal: {len(test_cases)} test cases generated")
            return test_cases

        finally:
            await pool.close()

    def save_test_cases(
        self,
        test_cases: list[EntityTestCase],
        output_path: Path,
        source_counts: dict[str, int] | None = None
    ):
        """Save test cases to a JSON file with _meta block."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Count by subcategory
        subcategory_counts = {}
        for tc in test_cases:
            subcat = tc.subcategory
            subcategory_counts[subcat] = subcategory_counts.get(subcat, 0) + 1

        data = {
            "_meta": {
                "name": "entity_resolution",
                "capability": "entity_resolution",
                "description": "Entity name/ticker to RIC resolution tests",
                "requires_nl_response": False,
                "requires_expected_response": False,
                "schema_version": "1.0",
                "generated_at": datetime.now(UTC).isoformat(),
                "generator": "scripts/generators/entity_resolution_generator.py",
                "source_counts": source_counts or {},
                "subcategory_counts": subcategory_counts,
            },
            "metadata": {
                "category": self.category,
                "generator": self.__class__.__name__,
                "count": len(test_cases),
            },
            "test_cases": [tc.to_dict() for tc in test_cases]
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"\nSaved {len(test_cases)} test cases to {output_path}")


async def main():
    parser = argparse.ArgumentParser(
        description="Generate entity resolution test cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--db-url",
        type=str,
        default=os.environ.get(
            "DATABASE_URL",
            "postgresql://nl2api:nl2api@localhost:5432/nl2api"
        ),
        help="PostgreSQL connection URL"
    )

    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/lseg/generated/entity_resolution/entity_resolution.json"),
        help="Output JSON file path"
    )

    args = parser.parse_args()

    generator = EntityResolutionGenerator(args.db_url)

    # Generate test cases
    test_cases = await generator.generate()

    # Get source counts for metadata
    asyncpg = _get_asyncpg()
    pool = await asyncpg.create_pool(args.db_url)
    try:
        async with pool.acquire() as conn:
            source_counts = await conn.fetch("""
                SELECT data_source, COUNT(*) as count
                FROM entities
                WHERE entity_status = 'active'
                GROUP BY data_source
            """)
            source_counts = {row["data_source"]: row["count"] for row in source_counts}
    finally:
        await pool.close()

    # Save to file
    generator.save_test_cases(test_cases, args.output, source_counts)


if __name__ == "__main__":
    asyncio.run(main())
