#!/usr/bin/env python3
"""
Generate Evaluation Data (expected_response and expected_nl_response)

Populates test case fixtures with synthetic API response data and NL summaries
using an LLM. This data is used by the SemanticsEvaluator (Stage 4).

Usage:
    # Dry run - show prompts without making LLM calls
    python scripts/generate_eval_data.py --dry-run

    # Generate for a specific category
    python scripts/generate_eval_data.py --category lookups --limit 10

    # Generate for all categories (REVIEW PROMPT FIRST per CLAUDE.md)
    python scripts/generate_eval_data.py --all

Estimated cost: ~$10-15 for all ~12,651 test cases (Claude 3.5 Haiku)
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.core.semantics.prompts import (  # noqa: E402
    GENERATION_SYSTEM_PROMPT,
    GENERATION_USER_PROMPT_TEMPLATE,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

FIXTURE_DIR = PROJECT_ROOT / "tests" / "fixtures" / "lseg" / "generated"


def estimate_cost(test_case_count: int) -> dict:
    """Estimate the cost of generating data for test cases."""
    # Average tokens per test case (rough estimates)
    avg_input_tokens = 200  # System prompt + user prompt
    avg_output_tokens = 150  # expected_response + expected_nl_response

    total_input = test_case_count * avg_input_tokens
    total_output = test_case_count * avg_output_tokens

    # Claude 3.5 Haiku pricing: $0.25/1M input, $1.25/1M output
    input_cost = (total_input / 1_000_000) * 0.25
    output_cost = (total_output / 1_000_000) * 1.25
    total_cost = input_cost + output_cost

    return {
        "test_cases": test_case_count,
        "input_tokens": total_input,
        "output_tokens": total_output,
        "input_cost_usd": input_cost,
        "output_cost_usd": output_cost,
        "total_cost_usd": total_cost,
    }


def show_prompt_preview(test_case: dict) -> None:
    """Show a preview of the prompt for a test case."""
    user_prompt = GENERATION_USER_PROMPT_TEMPLATE.format(
        nl_query=test_case.get("nl_query", ""),
        tool_calls=json.dumps(test_case.get("expected_tool_calls", []), indent=2),
    )

    print("\n" + "=" * 80)
    print("SYSTEM PROMPT:")
    print("=" * 80)
    print(GENERATION_SYSTEM_PROMPT)
    print("\n" + "=" * 80)
    print("USER PROMPT (template):")
    print("=" * 80)
    print(user_prompt)
    print("\n" + "=" * 80)
    print("EXPECTED OUTPUT FORMAT:")
    print("=" * 80)
    print("""{
  "expected_response": {"AAPL.O": {"P": 185.42, "MV": 2850000000000}},
  "expected_nl_response": "Apple's stock price is $185.42 with a market cap of $2.85 trillion."
}""")


async def generate_for_test_case(
    llm,
    test_case: dict,
    semaphore: asyncio.Semaphore,
) -> dict | None:
    """Generate expected_response and expected_nl_response for a single test case."""
    async with semaphore:
        from src.nl2api.llm.protocols import LLMMessage, MessageRole

        user_prompt = GENERATION_USER_PROMPT_TEMPLATE.format(
            nl_query=test_case.get("nl_query", ""),
            tool_calls=json.dumps(test_case.get("expected_tool_calls", []), indent=2),
        )

        messages = [
            LLMMessage(role=MessageRole.SYSTEM, content=GENERATION_SYSTEM_PROMPT),
            LLMMessage(role=MessageRole.USER, content=user_prompt),
        ]

        try:
            response = await llm.complete_with_retry(
                messages=messages,
                temperature=0.3,  # Some creativity for realistic values
                max_tokens=512,
                max_retries=3,
            )

            # Parse the response
            result = json.loads(response.content)
            return {
                "expected_response": result.get("expected_response"),
                "expected_nl_response": result.get("expected_nl_response"),
            }
        except json.JSONDecodeError as e:
            logger.warning(f"Failed to parse response for {test_case.get('id')}: {e}")
            return None
        except Exception as e:
            logger.error(f"Error generating for {test_case.get('id')}: {e}")
            return None


async def process_category(
    category: str,
    limit: int | None = None,
    dry_run: bool = False,
    concurrency: int = 5,
) -> tuple[int, int]:
    """Process a single category of fixtures.

    Returns:
        Tuple of (processed_count, failed_count)
    """
    fixture_path = FIXTURE_DIR / category / f"{category}.json"

    if not fixture_path.exists():
        logger.warning(f"Fixture file not found: {fixture_path}")
        return 0, 0

    with open(fixture_path) as f:
        data = json.load(f)

    test_cases = data.get("test_cases", [])
    if limit:
        test_cases = test_cases[:limit]

    # Filter to only test cases that need generation
    needs_generation = [
        tc for tc in test_cases
        if tc.get("expected_response") is None or tc.get("expected_nl_response") is None
    ]

    if not needs_generation:
        logger.info(f"{category}: All {len(test_cases)} test cases already have generated data")
        return 0, 0

    logger.info(f"{category}: {len(needs_generation)} of {len(test_cases)} need generation")

    if dry_run:
        # Show preview for first test case
        show_prompt_preview(needs_generation[0])
        cost = estimate_cost(len(needs_generation))
        print(f"\nScope: {cost['test_cases']} test cases")
        print(f"Estimated cost: ${cost['total_cost_usd']:.2f}")
        print(f"  Input tokens: {cost['input_tokens']:,} (${cost['input_cost_usd']:.2f})")
        print(f"  Output tokens: {cost['output_tokens']:,} (${cost['output_cost_usd']:.2f})")
        print("Model: claude-3-5-haiku-20241022")
        return 0, 0

    # Initialize LLM
    from src.nl2api.config import NL2APIConfig
    from src.nl2api.llm.claude import ClaudeProvider

    cfg = NL2APIConfig()
    llm = ClaudeProvider(
        api_key=cfg.get_llm_api_key(),
        model="claude-3-5-haiku-20241022",
    )

    semaphore = asyncio.Semaphore(concurrency)
    processed = 0
    failed = 0

    # Create a mapping from test case id to index for updating
    tc_id_to_idx = {tc.get("id"): i for i, tc in enumerate(data["test_cases"])}

    # Process in batches for progress reporting
    batch_size = 50
    for batch_start in range(0, len(needs_generation), batch_size):
        batch = needs_generation[batch_start:batch_start + batch_size]

        # Generate for batch
        tasks = [
            generate_for_test_case(llm, tc, semaphore)
            for tc in batch
        ]
        results = await asyncio.gather(*tasks)

        # Update test cases with results
        for tc, result in zip(batch, results):
            tc_id = tc.get("id")
            idx = tc_id_to_idx.get(tc_id)
            if idx is None:
                continue

            if result:
                data["test_cases"][idx]["expected_response"] = result["expected_response"]
                data["test_cases"][idx]["expected_nl_response"] = result["expected_nl_response"]
                processed += 1
            else:
                failed += 1

        # Progress
        total_done = batch_start + len(batch)
        logger.info(f"{category}: Processed {total_done}/{len(needs_generation)} ({processed} success, {failed} failed)")

    # Save updated fixture
    with open(fixture_path, "w") as f:
        json.dump(data, f, indent=2)

    logger.info(f"{category}: Saved {processed} updated test cases to {fixture_path}")
    return processed, failed


async def main():
    parser = argparse.ArgumentParser(description="Generate evaluation data for test cases")
    parser.add_argument(
        "--category",
        type=str,
        help="Category to process (e.g., lookups, temporal)"
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Process all categories"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum test cases per category"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show prompts and cost estimate without making LLM calls"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=5,
        help="Maximum concurrent LLM calls"
    )

    args = parser.parse_args()

    if not args.category and not args.all:
        parser.error("Must specify --category or --all")

    # Check for API key if not dry run
    if not args.dry_run:
        api_key = os.environ.get("NL2API_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            print("Error: ANTHROPIC_API_KEY or NL2API_ANTHROPIC_API_KEY must be set")
            sys.exit(1)

    if args.all:
        categories = [d.name for d in FIXTURE_DIR.iterdir() if d.is_dir()]
    else:
        categories = [args.category]

    # Count total test cases needing generation
    if args.dry_run:
        total_need_gen = 0
        for category in categories:
            fixture_path = FIXTURE_DIR / category / f"{category}.json"
            if fixture_path.exists():
                with open(fixture_path) as f:
                    data = json.load(f)
                    test_cases = data.get("test_cases", [])
                    if args.limit:
                        test_cases = test_cases[:args.limit]
                    needs = sum(
                        1 for tc in test_cases
                        if tc.get("expected_response") is None or tc.get("expected_nl_response") is None
                    )
                    total_need_gen += needs
                    print(f"{category}: {needs} test cases need generation")

        cost = estimate_cost(total_need_gen)
        print(f"\n{'=' * 50}")
        print(f"TOTAL: {cost['test_cases']} test cases")
        print(f"Estimated cost: ${cost['total_cost_usd']:.2f}")
        print(f"Model: claude-3-5-haiku-20241022")
        print(f"\nTo proceed, run without --dry-run")

        # Show sample prompt from first category with data
        for category in categories:
            fixture_path = FIXTURE_DIR / category / f"{category}.json"
            if fixture_path.exists():
                with open(fixture_path) as f:
                    data = json.load(f)
                    test_cases = data.get("test_cases", [])
                    for tc in test_cases:
                        if tc.get("expected_response") is None:
                            print(f"\n{'=' * 50}")
                            print("SAMPLE PROMPT PREVIEW:")
                            show_prompt_preview(tc)
                            break
                    break
        return

    # Process categories
    total_processed = 0
    total_failed = 0

    for category in categories:
        print(f"\nProcessing category: {category}")
        processed, failed = await process_category(
            category,
            limit=args.limit,
            dry_run=args.dry_run,
            concurrency=args.concurrency,
        )
        total_processed += processed
        total_failed += failed

    print(f"\n{'=' * 50}")
    print(f"Total processed: {total_processed}")
    print(f"Total failed: {total_failed}")


if __name__ == "__main__":
    asyncio.run(main())
