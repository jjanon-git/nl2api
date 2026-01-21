"""
Entity Extraction Generator - Generates test cases for entity extraction evaluation.

Tests the full extraction → resolution flow where:
1. NL query is input (no pre-extracted entity)
2. System must extract company names from natural language
3. Then resolve each to a RIC

This generator samples from the existing gold eval fixtures (lookups, temporal,
comparisons, complex) to create extraction-focused test cases.

Key difference from entity_resolution:
- entity_resolution: has `input_entity` in metadata → tests resolution only
- entity_extraction: NO `input_entity` → tests extraction + resolution

Target: ~2,000 test cases
- possessive: 400 cases ("Apple's", "Microsoft's")
- of_pattern: 300 cases ("CEO of Apple")
- for_pattern: 300 cases ("data for Microsoft")
- at_pattern: 200 cases ("executives at Tesla")
- multi_entity: 300 cases ("Apple and Microsoft")
- ticker_inline: 200 cases ("AAPL stock")
- embedded: 200 cases ("I want Apple's price")
- complex_sentence: 100 cases

Usage:
    python -m scripts.generators.entity_extraction_generator \
        --output tests/fixtures/lseg/generated/entity_extraction/entity_extraction.json

Requires:
    - Existing fixture files in tests/fixtures/lseg/generated/
"""

import argparse
import hashlib
import json
import random
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class ExtractionTestCase:
    """A single entity extraction test case."""
    id: str
    nl_query: str
    expected_tool_calls: list[dict[str, Any]]
    expected_response: None  # Always null for entity extraction
    expected_nl_response: None  # Always null for entity extraction
    complexity: int
    category: str
    subcategory: str
    tags: list[str]
    metadata: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class EntityExtractionGenerator:
    """
    Generator for entity extraction test cases from existing gold fixtures.

    Samples from lookups, temporal, comparisons, and complex fixtures
    to create test cases that exercise the full extraction → resolution flow.
    """

    # Patterns to detect extraction type from NL query
    EXTRACTION_PATTERNS = {
        "possessive": re.compile(r"(\w+(?:\s+\w+)?(?:\s+(?:Inc|Corp|Ltd|SA|AG|SE|NV|PLC)\.?)?)'s\b", re.IGNORECASE),
        "of_pattern": re.compile(r"\b(?:of|for)\s+(\w+(?:\s+\w+)?(?:\s+(?:Inc|Corp|Ltd|SA|AG|SE|NV|PLC)\.?)?)\b", re.IGNORECASE),
        "ticker_inline": re.compile(r"\b([A-Z]{2,5})\s+(?:stock|price|data|earnings|revenue)", re.IGNORECASE),
        "embedded": re.compile(r"\b(?:how|what|show|get)\s+.*?\b(\w+(?:\s+\w+)?)\s+(?:stock|price|data)", re.IGNORECASE),
    }

    # Target counts per subcategory
    SUBCATEGORY_TARGETS = {
        "possessive": 400,
        "of_pattern": 300,
        "for_pattern": 300,
        "at_pattern": 200,
        "multi_entity": 300,
        "ticker_inline": 200,
        "embedded": 200,
        "complex_sentence": 100,
    }

    def __init__(self, fixtures_dir: Path):
        self.fixtures_dir = fixtures_dir
        self.source_fixtures: list[dict] = []
        self.generated_ids: set[str] = set()

    def load_source_fixtures(self) -> None:
        """Load all source fixtures from the gold eval set."""
        source_files = [
            "lookups/lookups.json",
            "temporal/temporal.json",
            "comparisons/comparisons.json",
            "complex/complex.json",
        ]

        for source_file in source_files:
            file_path = self.fixtures_dir / source_file
            if file_path.exists():
                with open(file_path) as f:
                    data = json.load(f)
                    test_cases = data.get("test_cases", [])
                    # Tag with source
                    for tc in test_cases:
                        tc["_source"] = source_file.split("/")[0]
                    self.source_fixtures.extend(test_cases)

        print(f"Loaded {len(self.source_fixtures)} source fixtures")

    def _generate_id(self, nl_query: str) -> str:
        """Generate a unique test case ID."""
        content = f"entity_extraction:{nl_query}"
        hash_str = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"extraction_{hash_str}"

    def _detect_extraction_pattern(self, nl_query: str) -> str | None:
        """Detect the extraction pattern type from the query."""
        query_lower = nl_query.lower()

        # Check for possessive pattern
        if "'s " in query_lower or "'s " in nl_query:
            return "possessive"

        # Check for "of" pattern
        if " of " in query_lower:
            return "of_pattern"

        # Check for "for" pattern
        if " for " in query_lower:
            return "for_pattern"

        # Check for "at" pattern
        if " at " in query_lower:
            return "at_pattern"

        # Check for multi-entity patterns
        if " and " in query_lower or " vs " in query_lower or ", " in query_lower:
            return "multi_entity"

        # Check for ticker inline
        if self.EXTRACTION_PATTERNS["ticker_inline"].search(nl_query):
            return "ticker_inline"

        # Default to embedded
        return "embedded"

    def _extract_company_from_query(self, nl_query: str, expected_ticker: str) -> str | None:
        """
        Extract the company name from the NL query.

        Uses the expected ticker as a hint to find the correct entity.
        """
        # Try to find company name near possessive
        possessive_match = re.search(r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?(?:\s+(?:Inc|Corp|Ltd|SA|AG|SE|NV|PLC)\.?)?)'s", nl_query)
        if possessive_match:
            return possessive_match.group(1)

        # Try "of Company" pattern
        of_match = re.search(r"\bof\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", nl_query)
        if of_match:
            return of_match.group(1)

        # Try "for Company" pattern
        for_match = re.search(r"\bfor\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", nl_query)
        if for_match:
            return for_match.group(1)

        # Try "at Company" pattern
        at_match = re.search(r"\bat\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)", nl_query)
        if at_match:
            return at_match.group(1)

        # Try to find capitalized words that could be company names
        cap_words = re.findall(r"\b([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)\b", nl_query)
        # Filter out common words
        stop_words = {"What", "Show", "Get", "Find", "How", "The", "Is", "Are", "Tell", "Can", "Please", "Give"}
        for word in cap_words:
            if word not in stop_words:
                return word

        return None

    def _get_tickers_from_tool_calls(self, tool_calls: list[dict]) -> list[str]:
        """Extract tickers from expected tool calls."""
        tickers = []
        for tc in tool_calls:
            args = tc.get("arguments", {})
            # Handle both single ticker and list of tickers
            ticker = args.get("tickers") or args.get("instruments")
            if isinstance(ticker, str):
                tickers.append(ticker)
            elif isinstance(ticker, list):
                tickers.extend(ticker)
        return tickers

    def _is_multi_entity(self, tool_calls: list[dict]) -> bool:
        """Check if the fixture involves multiple entities."""
        tickers = self._get_tickers_from_tool_calls(tool_calls)
        # Count unique base tickers (without exchange suffixes)
        unique = set()
        for t in tickers:
            # Extract base ticker (e.g., "U:AAPL" -> "AAPL", "AAPL.O" -> "AAPL")
            base = re.sub(r"^[A-Z]:|\..*$", "", t)
            unique.add(base)
        return len(unique) > 1

    def _convert_to_extraction_case(
        self,
        source: dict,
        subcategory: str,
    ) -> ExtractionTestCase | None:
        """Convert a source fixture to an extraction test case."""
        nl_query = source.get("nl_query", "")
        tool_calls = source.get("expected_tool_calls", [])

        if not nl_query or not tool_calls:
            return None

        tickers = self._get_tickers_from_tool_calls(tool_calls)
        if not tickers:
            return None

        # Try to extract company name from query
        company = self._extract_company_from_query(nl_query, tickers[0])
        if not company:
            return None

        test_id = self._generate_id(nl_query)
        if test_id in self.generated_ids:
            return None
        self.generated_ids.add(test_id)

        # Determine complexity
        complexity = 1
        if len(tickers) > 1:
            complexity = len(tickers)
        if subcategory in ["multi_entity", "complex_sentence"]:
            complexity = max(complexity, 2)

        # Build metadata - NO input_entity to force extraction
        metadata = {
            "expected_extractions": [company] if not self._is_multi_entity(tool_calls) else self._extract_all_companies(nl_query),
            "expected_tickers": tickers,
            "extraction_pattern": subcategory,
            "source_fixture": source.get("id", "unknown"),
            "source_category": source.get("_source", "unknown"),
        }

        tags = [
            subcategory,
            f"source:{source.get('_source', 'unknown')}",
        ]
        if len(tickers) > 1:
            tags.append("multi_entity")

        return ExtractionTestCase(
            id=test_id,
            nl_query=nl_query,
            expected_tool_calls=tool_calls,
            expected_response=None,
            expected_nl_response=None,
            complexity=complexity,
            category="entity_extraction",
            subcategory=subcategory,
            tags=tags,
            metadata=metadata,
        )

    def _extract_all_companies(self, nl_query: str) -> list[str]:
        """Extract all company names from a multi-entity query."""
        companies = []

        # Find possessives
        for match in re.finditer(r"([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)?)'s", nl_query):
            companies.append(match.group(1))

        # Find "and" separated companies
        and_match = re.search(r"([A-Z][a-zA-Z]+)\s+(?:and|vs)\s+([A-Z][a-zA-Z]+)", nl_query)
        if and_match:
            companies.extend([and_match.group(1), and_match.group(2)])

        # Deduplicate while preserving order
        seen = set()
        unique = []
        for c in companies:
            if c not in seen:
                seen.add(c)
                unique.append(c)

        return unique if unique else ["unknown"]

    def generate(self) -> list[ExtractionTestCase]:
        """Generate entity extraction test cases."""
        if not self.source_fixtures:
            self.load_source_fixtures()

        test_cases: list[ExtractionTestCase] = []
        subcategory_counts: dict[str, int] = {k: 0 for k in self.SUBCATEGORY_TARGETS}

        # Shuffle source fixtures for random sampling
        random.shuffle(self.source_fixtures)

        for source in self.source_fixtures:
            nl_query = source.get("nl_query", "")
            tool_calls = source.get("expected_tool_calls", [])

            if not nl_query or not tool_calls:
                continue

            # Detect extraction pattern
            subcategory = self._detect_extraction_pattern(nl_query)
            if subcategory is None:
                subcategory = "embedded"

            # Check if we need more of this subcategory
            target = self.SUBCATEGORY_TARGETS.get(subcategory, 100)
            if subcategory_counts.get(subcategory, 0) >= target:
                continue

            # Convert to extraction case
            test_case = self._convert_to_extraction_case(source, subcategory)
            if test_case:
                test_cases.append(test_case)
                subcategory_counts[subcategory] = subcategory_counts.get(subcategory, 0) + 1

            # Check if we've hit all targets
            all_done = all(
                subcategory_counts.get(k, 0) >= v
                for k, v in self.SUBCATEGORY_TARGETS.items()
            )
            if all_done:
                break

        print(f"Generated {len(test_cases)} entity extraction test cases")
        print(f"Subcategory distribution: {subcategory_counts}")

        return test_cases

    def save(self, test_cases: list[ExtractionTestCase], output_path: Path) -> None:
        """Save test cases to JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Calculate subcategory counts
        subcategory_counts = {}
        for tc in test_cases:
            subcat = tc.subcategory
            subcategory_counts[subcat] = subcategory_counts.get(subcat, 0) + 1

        data = {
            "_meta": {
                "name": "entity_extraction",
                "capability": "entity_extraction",
                "description": "Entity extraction from NL queries - tests full extraction -> resolution flow",
                "requires_nl_response": False,
                "requires_expected_response": False,
                "schema_version": "1.0",
                "generated_at": datetime.now(UTC).isoformat(),
                "generator": "scripts/generators/entity_extraction_generator.py",
                "notes": "These fixtures do NOT have input_entity in metadata, forcing the full extraction flow",
                "subcategory_counts": subcategory_counts,
            },
            "metadata": {
                "category": "entity_extraction",
                "generator": "EntityExtractionGenerator",
                "count": len(test_cases),
            },
            "test_cases": [tc.to_dict() for tc in test_cases],
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"Saved {len(test_cases)} test cases to {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate entity extraction test cases from existing gold fixtures"
    )
    parser.add_argument(
        "--fixtures-dir",
        type=Path,
        default=Path("tests/fixtures/lseg/generated"),
        help="Directory containing source fixture files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("tests/fixtures/lseg/generated/entity_extraction/entity_extraction.json"),
        help="Output path for generated fixtures",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    generator = EntityExtractionGenerator(args.fixtures_dir)
    generator.load_source_fixtures()
    test_cases = generator.generate()
    generator.save(test_cases, args.output)


if __name__ == "__main__":
    main()
