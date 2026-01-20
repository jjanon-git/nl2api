"""
Base Generator class with common functionality for all test case generators.
"""

import json
import random
import hashlib
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict


@dataclass
class TestCase:
    """Represents a single test case."""
    id: str
    nl_query: str
    expected_tool_calls: List[Dict[str, Any]]
    complexity: int
    category: str
    subcategory: str
    tags: List[str]
    metadata: Dict[str, Any]

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
                         metadata: Optional[Dict[str, Any]] = None) -> Optional[TestCase]:
        """Create a test case if it doesn't already exist."""
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
            metadata=metadata or {}
        )

    def generate(self) -> List[TestCase]:
        """Generate test cases. Override in subclasses."""
        raise NotImplementedError("Subclasses must implement generate()")

    def save_test_cases(self, test_cases: List[TestCase], output_path: Path):
        """Save test cases to a JSON file."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "generator": self.__class__.__name__,
                "count": len(test_cases)
            },
            "test_cases": [tc.to_dict() for tc in test_cases]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)
