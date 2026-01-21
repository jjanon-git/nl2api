"""
Base Generator class with common functionality for all test case generators.

All generators should produce output aligned with CONTRACTS.py TestCase and
TestCaseSetConfig models. Key requirements:
- Use "tool_name" (not "function") in tool calls
- Include _meta block with TestCaseSetConfig fields
- Include expected_response and expected_nl_response fields (can be null)
"""

import json
import random
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict, field


@dataclass
class TestCase:
    """
    Represents a single test case.

    Aligned with CONTRACTS.py TestCase model.
    """
    id: str
    nl_query: str
    expected_tool_calls: List[Dict[str, Any]]  # Must use "tool_name", not "function"
    complexity: int
    category: str
    subcategory: str
    tags: List[str]
    metadata: Dict[str, Any]
    expected_response: Optional[Dict[str, Any]] = None  # Structured API response data
    expected_nl_response: Optional[str] = None  # Human-readable response sentence

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class BaseGenerator:
    """Base class for all test case generators."""

    def __init__(self, data_dir: Path):
        self.data_dir = data_dir
        self.field_codes_dir = data_dir / "field_codes"
        self.tickers_dir = data_dir / "tickers"

        # Load common data files
        self.datastream_fields = self._load_json("field_codes/datastream_fields.json")
        self.fundamentals_wc = self._load_json("field_codes/fundamentals_wc.json")
        self.fundamentals_tr = self._load_json("field_codes/fundamentals_tr.json")
        self.estimates_tr = self._load_json("field_codes/estimates_tr.json")
        self.officers_tr = self._load_json("field_codes/officers_tr.json")

        self.us_mega_caps = self._load_json("tickers/us_mega_caps.json")
        self.us_by_sector = self._load_json("tickers/us_by_sector.json")
        self.international = self._load_json("tickers/international.json")
        self.indices = self._load_json("tickers/indices.json")
        self.edge_cases = self._load_json("tickers/edge_cases.json")

        self.temporal_patterns = self._load_json("temporal_patterns.json")
        self.nl_templates = self._load_json("nl_templates.json")
        self.comparison_pairs = self._load_json("comparison_pairs.json")
        self.screening_criteria = self._load_json("screening_criteria.json")

        # Track generated test IDs for deduplication
        self.generated_ids = set()

    def _load_json(self, relative_path: str) -> Dict:
        """Load a JSON file from the data directory."""
        file_path = self.data_dir / relative_path
        if file_path.exists():
            with open(file_path, 'r') as f:
                return json.load(f)
        return {}

    def _generate_id(self, nl_query: str, category: str) -> str:
        """Generate a unique test case ID."""
        content = f"{category}:{nl_query}"
        hash_str = hashlib.md5(content.encode()).hexdigest()[:8]
        return f"{category}_{hash_str}"

    def _select_random_template(self, template_list: List[str]) -> str:
        """Select a random template from a list."""
        return random.choice(template_list)

    def _get_all_tickers(self, count: Optional[int] = None, include_international: bool = True) -> List[Dict]:
        """Get a list of tickers for test generation."""
        tickers = self.us_mega_caps.get("tickers", [])

        if include_international:
            for region_tickers in self.international.get("regions", {}).values():
                tickers.extend(region_tickers)

        if count:
            return random.sample(tickers, min(count, len(tickers)))
        return tickers

    def _get_tickers_by_sector(self, sector: str) -> List[Dict]:
        """Get tickers for a specific sector."""
        return self.us_by_sector.get("sectors", {}).get(sector, [])

    def _flatten_fields(self, field_dict: Dict) -> List[Dict]:
        """Flatten nested field code dictionary into a list of fields."""
        fields = []
        for category, category_fields in field_dict.items():
            if isinstance(category_fields, dict):
                for code, details in category_fields.items():
                    if isinstance(details, dict):
                        fields.append({
                            "code": code,
                            "category": category,
                            **details
                        })
        return fields

    def _calculate_complexity(self,
                             num_tickers: int = 1,
                             num_fields: int = 1,
                             is_time_series: bool = False,
                             has_calculations: bool = False,
                             is_multi_step: bool = False) -> int:
        """Calculate complexity score based on query characteristics."""
        score = 0

        # Ticker complexity
        if num_tickers <= 1:
            score += 0
        elif num_tickers <= 5:
            score += 1
        else:
            score += 2

        # Field complexity
        if num_fields <= 1:
            score += 0
        elif num_fields <= 3:
            score += 1
        else:
            score += 2

        # Time series adds complexity
        if is_time_series:
            score += 1

        # Calculations (moving averages, % change) add complexity
        if has_calculations:
            score += 2

        # Multi-step queries are most complex
        if is_multi_step:
            score += 3

        # Map to complexity levels (1-15)
        if score <= 1:
            return 1
        elif score <= 2:
            return 2
        elif score <= 3:
            return 3
        elif score <= 4:
            return 4
        elif score <= 5:
            return 5
        elif score <= 6:
            return 6
        elif score <= 7:
            return 7
        elif score <= 9:
            return 8
        elif score <= 11:
            return 10
        else:
            return 12

    def _create_test_case(self,
                         nl_query: str,
                         tool_calls: List[Dict[str, Any]],
                         category: str,
                         subcategory: str,
                         complexity: int,
                         tags: List[str],
                         metadata: Optional[Dict[str, Any]] = None,
                         expected_response: Optional[Dict[str, Any]] = None,
                         expected_nl_response: Optional[str] = None) -> Optional[TestCase]:
        """
        Create a test case if it doesn't already exist.

        Args:
            nl_query: Natural language query
            tool_calls: Expected tool calls (must use "tool_name" key, not "function")
            category: Test category (e.g., "lookups", "temporal")
            subcategory: Test subcategory
            complexity: Complexity level (1-5)
            tags: List of tags for filtering
            metadata: Additional metadata
            expected_response: Expected structured data response (can be None)
            expected_nl_response: Expected natural language response (can be None)

        Returns:
            TestCase if created, None if duplicate
        """
        test_id = self._generate_id(nl_query, category)

        if test_id in self.generated_ids:
            return None

        self.generated_ids.add(test_id)

        return TestCase(
            id=test_id,
            nl_query=nl_query,
            expected_tool_calls=tool_calls,
            complexity=complexity,
            category=category,
            subcategory=subcategory,
            tags=tags,
            metadata=metadata or {},
            expected_response=expected_response,
            expected_nl_response=expected_nl_response,
        )

    def generate(self) -> List[TestCase]:
        """Generate test cases. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement generate()")

    def _get_capability(self) -> str:
        """
        Get the capability name for this generator.
        Override in subclasses if different from 'nl2api'.
        """
        return "nl2api"

    def _get_category_name(self) -> str:
        """
        Get the category name for this generator.
        Override in subclasses.
        """
        return self.__class__.__name__.replace("Generator", "").lower()

    def _requires_nl_response(self) -> bool:
        """
        Whether this generator's test cases require expected_nl_response.
        Override in subclasses if False (e.g., entity_resolution).
        """
        return True

    def _requires_expected_response(self) -> bool:
        """
        Whether this generator's test cases require expected_response.
        Override in subclasses if True.
        """
        return False

    def _create_meta_block(self, test_cases: List[TestCase]) -> Dict[str, Any]:
        """
        Create the _meta block for TestCaseSetConfig.

        This block defines per-dataset field requirements and generation metadata.
        """
        return {
            "name": self._get_category_name(),
            "capability": self._get_capability(),
            "description": f"Generated test cases for {self._get_category_name()} category",
            "requires_nl_response": self._requires_nl_response(),
            "requires_expected_response": self._requires_expected_response(),
            "schema_version": "1.0",
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "generator": f"scripts/generators/{self.__class__.__module__.split('.')[-1]}.py",
        }

    def save_test_cases(self, test_cases: List[TestCase], output_path: Path):
        """
        Save test cases to a JSON file with _meta block.

        Output format aligned with CONTRACTS.py TestCaseSetConfig:
        {
            "_meta": { ... TestCaseSetConfig fields ... },
            "metadata": { ... legacy metadata ... },
            "test_cases": [ ... ]
        }
        """
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "_meta": self._create_meta_block(test_cases),
            "metadata": {
                "generator": self.__class__.__name__,
                "count": len(test_cases)
            },
            "test_cases": [tc.to_dict() for tc in test_cases]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
