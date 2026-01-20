#!/usr/bin/env python3
"""
LSEG Test Case Generator - Main Orchestrator

This script generates comprehensive test cases for the LSEG financial data
evaluation platform, targeting 10,000+ test cases across all categories.

Usage:
    python generate_test_cases.py [--category CATEGORY] [--output-dir DIR]

Categories:
    - lookups: Single/multi-field lookup queries (~4,000)
    - temporal: Time series and temporal variants (~2,500)
    - comparisons: Multi-ticker comparison queries (~3,000)
    - screening: Stock screening queries (~500)
    - errors: Error scenario test cases (~500)
    - complex: Multi-step workflows (~500)
    - all: Generate all categories (default)
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.generators import (
    LookupGenerator,
    TemporalGenerator,
    ComparisonGenerator,
    ScreeningGenerator,
    ErrorGenerator,
    ComplexGenerator
)


class TestCaseOrchestrator:
    """Main orchestrator for test case generation."""

    def __init__(self, data_dir: Path, output_dir: Path):
        self.data_dir = data_dir
        self.output_dir = output_dir

        # Initialize all generators
        self.generators = {
            "lookups": LookupGenerator(data_dir),
            "temporal": TemporalGenerator(data_dir),
            "comparisons": ComparisonGenerator(data_dir),
            "screening": ScreeningGenerator(data_dir),
            "errors": ErrorGenerator(data_dir),
            "complex": ComplexGenerator(data_dir),
        }

    def generate_category(self, category: str) -> Dict:
        """Generate test cases for a specific category."""
        if category not in self.generators:
            raise ValueError(f"Unknown category: {category}")

        generator = self.generators[category]
        print(f"\nGenerating {category} test cases...")

        test_cases = generator.generate()

        # Save to output file
        output_path = self.output_dir / category / f"{category}.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "metadata": {
                "category": category,
                "generator": generator.__class__.__name__,
                "count": len(test_cases),
                "generated_at": datetime.now().isoformat()
            },
            "test_cases": [tc.to_dict() for tc in test_cases]
        }

        with open(output_path, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"  Generated {len(test_cases)} test cases -> {output_path}")

        return {
            "category": category,
            "count": len(test_cases),
            "output_path": str(output_path)
        }

    def generate_all(self) -> Dict:
        """Generate test cases for all categories."""
        results = {}
        total_count = 0

        for category in self.generators.keys():
            result = self.generate_category(category)
            results[category] = result
            total_count += result["count"]

        # Generate summary report
        summary = {
            "generated_at": datetime.now().isoformat(),
            "total_test_cases": total_count,
            "categories": results,
            "target": 11000,
            "coverage_pct": round(total_count / 11000 * 100, 1)
        }

        summary_path = self.output_dir / "generation_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n{'='*60}")
        print("Generation Summary")
        print(f"{'='*60}")
        print(f"Total test cases generated: {total_count:,}")
        print(f"Target: 11,000")
        print(f"Coverage: {summary['coverage_pct']}%")
        print(f"\nBreakdown by category:")
        for cat, result in results.items():
            print(f"  - {cat}: {result['count']:,} test cases")
        print(f"\nSummary saved to: {summary_path}")

        return summary

    def validate_output(self) -> Dict:
        """Validate generated test cases."""
        print("\nValidating generated test cases...")

        validation_results = {
            "valid": True,
            "issues": [],
            "stats": {}
        }

        for category_dir in self.output_dir.iterdir():
            if not category_dir.is_dir():
                continue

            for json_file in category_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    test_cases = data.get("test_cases", [])

                    # Check for required fields
                    for tc in test_cases:
                        if not tc.get("id"):
                            validation_results["issues"].append(
                                f"Missing ID in {json_file}"
                            )
                            validation_results["valid"] = False

                        if not tc.get("nl_query"):
                            validation_results["issues"].append(
                                f"Missing nl_query in {json_file}: {tc.get('id')}"
                            )
                            validation_results["valid"] = False

                        if not tc.get("expected_tool_calls"):
                            validation_results["issues"].append(
                                f"Missing expected_tool_calls in {json_file}: {tc.get('id')}"
                            )
                            validation_results["valid"] = False

                    validation_results["stats"][category_dir.name] = {
                        "file": str(json_file),
                        "count": len(test_cases)
                    }

                except json.JSONDecodeError as e:
                    validation_results["issues"].append(
                        f"Invalid JSON in {json_file}: {e}"
                    )
                    validation_results["valid"] = False

        if validation_results["valid"]:
            print("  All test cases validated successfully!")
        else:
            print(f"  Found {len(validation_results['issues'])} issues:")
            for issue in validation_results["issues"][:10]:
                print(f"    - {issue}")
            if len(validation_results["issues"]) > 10:
                print(f"    ... and {len(validation_results['issues']) - 10} more")

        return validation_results

    def generate_coverage_report(self) -> Dict:
        """Generate a coverage report showing test case distribution."""
        print("\nGenerating coverage report...")

        coverage = {
            "by_category": {},
            "by_complexity": {},
            "by_tags": {},
            "field_coverage": set(),
            "ticker_coverage": set()
        }

        for category_dir in self.output_dir.iterdir():
            if not category_dir.is_dir():
                continue

            for json_file in category_dir.glob("*.json"):
                try:
                    with open(json_file, 'r') as f:
                        data = json.load(f)

                    test_cases = data.get("test_cases", [])
                    category = category_dir.name

                    coverage["by_category"][category] = len(test_cases)

                    for tc in test_cases:
                        # Complexity distribution
                        complexity = tc.get("complexity", 1)
                        coverage["by_complexity"][complexity] = \
                            coverage["by_complexity"].get(complexity, 0) + 1

                        # Tag distribution
                        for tag in tc.get("tags", []):
                            coverage["by_tags"][tag] = \
                                coverage["by_tags"].get(tag, 0) + 1

                        # Field coverage
                        metadata = tc.get("metadata", {})
                        if "field_code" in metadata:
                            coverage["field_coverage"].add(metadata["field_code"])
                        if "fields" in metadata:
                            coverage["field_coverage"].update(metadata["fields"])

                        # Ticker coverage
                        if "ticker" in metadata:
                            coverage["ticker_coverage"].add(metadata["ticker"])
                        if "tickers" in metadata:
                            coverage["ticker_coverage"].update(metadata["tickers"])

                except Exception as e:
                    print(f"  Warning: Could not process {json_file}: {e}")

        # Convert sets to lists for JSON serialization
        coverage["field_coverage"] = list(coverage["field_coverage"])
        coverage["ticker_coverage"] = list(coverage["ticker_coverage"])

        # Save coverage report
        report_path = self.output_dir / "coverage_report.json"
        with open(report_path, 'w') as f:
            json.dump(coverage, f, indent=2)

        print(f"\nCoverage Report:")
        print(f"  - Fields covered: {len(coverage['field_coverage'])}")
        print(f"  - Tickers covered: {len(coverage['ticker_coverage'])}")
        print(f"  - Complexity levels: {len(coverage['by_complexity'])}")
        print(f"  - Unique tags: {len(coverage['by_tags'])}")
        print(f"\nReport saved to: {report_path}")

        return coverage


def main():
    parser = argparse.ArgumentParser(
        description="Generate LSEG test cases",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    parser.add_argument(
        "--category",
        choices=["lookups", "temporal", "comparisons", "screening",
                 "errors", "complex", "all"],
        default="all",
        help="Category to generate (default: all)"
    )

    parser.add_argument(
        "--data-dir",
        type=Path,
        default=PROJECT_ROOT / "data",
        help="Path to data directory"
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "tests" / "fixtures" / "lseg" / "generated",
        help="Path to output directory"
    )

    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate generated test cases after generation"
    )

    parser.add_argument(
        "--coverage",
        action="store_true",
        help="Generate coverage report after generation"
    )

    args = parser.parse_args()

    # Ensure directories exist
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"LSEG Test Case Generator")
    print(f"{'='*60}")
    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")

    # Initialize orchestrator
    orchestrator = TestCaseOrchestrator(args.data_dir, args.output_dir)

    # Generate test cases
    if args.category == "all":
        summary = orchestrator.generate_all()
    else:
        result = orchestrator.generate_category(args.category)
        summary = {"categories": {args.category: result}}

    # Optional validation
    if args.validate:
        orchestrator.validate_output()

    # Optional coverage report
    if args.coverage:
        orchestrator.generate_coverage_report()

    print(f"\n{'='*60}")
    print("Generation complete!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
