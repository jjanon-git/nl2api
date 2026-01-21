#!/usr/bin/env python3
"""
Generate expected_nl_response values using Claude 3.5 Haiku.

This script:
1. Loads existing test cases from fixtures
2. Generates natural language responses using Claude 3.5 Haiku
3. Updates fixtures with the generated responses
4. Tracks generation metadata (model, timestamp, cost)

Usage:
    python scripts/generate_nl_responses.py --category lookups
    python scripts/generate_nl_responses.py --all
    python scripts/generate_nl_responses.py --all --dry-run  # Preview without calling API

Cost estimate: ~$5 for 12,651 test cases using Claude 3.5 Haiku
"""

import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.nl2api.llm.claude import ClaudeProvider
from src.nl2api.llm.protocols import LLMMessage, MessageRole

# Constants
HAIKU_MODEL = "claude-3-5-haiku-20241022"
FIXTURES_DIR = PROJECT_ROOT / "tests" / "fixtures" / "lseg" / "generated"
CHECKPOINT_FILE = PROJECT_ROOT / "scripts" / ".nl_response_checkpoint.json"

# Categories that need NL responses (based on _meta.requires_nl_response)
NL_RESPONSE_CATEGORIES = [
    "lookups",
    "temporal",
    "comparisons",
    "screening",
    "errors",
    "complex",
]

SYSTEM_PROMPT = """You are generating expected natural language responses for an NL2API evaluation system.

Given:
- A natural language query from a user
- The expected API tool calls that would be made
- The tool call arguments (fields, tickers, dates, etc.)

Generate a concise, natural language response that a financial data assistant would give.

Rules:
1. Be factual and specific about what data would be retrieved
2. Reference the specific fields and tickers from the tool calls
3. Keep responses under 50 words
4. Use natural, conversational phrasing
5. Do NOT invent specific numeric values - describe what would be returned
6. For time series queries, mention the date range
7. For comparisons, mention all companies being compared

Examples:
- Query: "What is Apple's stock price?"
  Response: "Apple's current stock price is retrieved from the market data feed."

- Query: "Compare the PE ratios of Microsoft and Google"
  Response: "Here are the PE ratios for Microsoft and Google, allowing you to compare their valuations."

- Query: "Show me Tesla's revenue for the last 5 years"
  Response: "Here is Tesla's annual revenue data for the past 5 years from their financial statements."

Generate ONLY the response text, nothing else."""


def format_tool_calls_for_prompt(tool_calls: list[dict]) -> str:
    """Format tool calls for inclusion in the prompt."""
    lines = []
    for i, tc in enumerate(tool_calls, 1):
        tool_name = tc.get("tool_name", tc.get("function", "unknown"))
        args = tc.get("arguments", {})

        # Extract key info
        tickers = args.get("tickers", "")
        fields = args.get("fields", [])
        start = args.get("start", "")
        end = args.get("end", "")

        parts = [f"Tool: {tool_name}"]
        if tickers:
            parts.append(f"Tickers: {tickers}")
        if fields:
            parts.append(f"Fields: {', '.join(fields) if isinstance(fields, list) else fields}")
        if start and end:
            parts.append(f"Period: {start} to {end}")

        lines.append(f"{i}. " + " | ".join(parts))

    return "\n".join(lines)


async def generate_nl_response(
    provider: ClaudeProvider,
    nl_query: str,
    tool_calls: list[dict],
) -> tuple[str, dict]:
    """
    Generate NL response for a single test case.

    Returns:
        Tuple of (response_text, usage_dict)
    """
    tool_calls_str = format_tool_calls_for_prompt(tool_calls)

    user_prompt = f"""Query: {nl_query}

Expected tool calls:
{tool_calls_str}

Generate a natural language response for this query:"""

    messages = [
        LLMMessage(role=MessageRole.SYSTEM, content=SYSTEM_PROMPT),
        LLMMessage(role=MessageRole.USER, content=user_prompt),
    ]

    response = await provider.complete_with_retry(
        messages=messages,
        temperature=0.3,  # Slight variation for natural responses
        max_tokens=150,
        max_retries=3,
    )

    return response.content.strip(), response.usage


def load_checkpoint() -> dict:
    """Load checkpoint file if exists."""
    if CHECKPOINT_FILE.exists():
        with open(CHECKPOINT_FILE) as f:
            return json.load(f)
    return {"completed": {}, "total_cost": 0.0}


def save_checkpoint(checkpoint: dict) -> None:
    """Save checkpoint to file."""
    with open(CHECKPOINT_FILE, "w") as f:
        json.dump(checkpoint, f, indent=2)


def calculate_cost(usage: dict) -> float:
    """Calculate cost for a single API call (Claude 3.5 Haiku rates)."""
    input_cost = usage.get("prompt_tokens", 0) * 0.80 / 1_000_000
    output_cost = usage.get("completion_tokens", 0) * 4.00 / 1_000_000
    return input_cost + output_cost


async def process_category(
    category: str,
    provider: ClaudeProvider,
    checkpoint: dict,
    dry_run: bool = False,
    batch_size: int = 10,
) -> dict:
    """
    Process a single category, generating NL responses for all test cases.

    Returns:
        Stats dict with counts and cost
    """
    fixture_path = FIXTURES_DIR / category / f"{category}.json"

    if not fixture_path.exists():
        print(f"  Skipping {category}: fixture file not found")
        return {"skipped": True}

    with open(fixture_path) as f:
        data = json.load(f)

    # Check if this category requires NL responses
    meta = data.get("_meta", {})
    if not meta.get("requires_nl_response", True):
        print(f"  Skipping {category}: requires_nl_response=false")
        return {"skipped": True, "reason": "not_required"}

    test_cases = data.get("test_cases", [])
    total = len(test_cases)
    completed_ids = set(checkpoint.get("completed", {}).get(category, []))

    print(f"  Processing {category}: {total} test cases ({len(completed_ids)} already done)")

    stats = {
        "total": total,
        "generated": 0,
        "skipped": 0,
        "errors": 0,
        "cost": 0.0,
    }

    # Process in batches
    for i, tc in enumerate(test_cases):
        tc_id = tc.get("id", f"idx_{i}")

        # Skip if already completed
        if tc_id in completed_ids:
            stats["skipped"] += 1
            continue

        # Skip if already has NL response
        if tc.get("expected_nl_response"):
            stats["skipped"] += 1
            completed_ids.add(tc_id)
            continue

        nl_query = tc.get("nl_query", "")
        tool_calls = tc.get("expected_tool_calls", [])

        if not nl_query or not tool_calls:
            stats["skipped"] += 1
            continue

        if dry_run:
            print(f"    [DRY RUN] Would generate for: {tc_id[:30]}...")
            stats["generated"] += 1
            continue

        try:
            response_text, usage = await generate_nl_response(
                provider, nl_query, tool_calls
            )

            # Update test case
            tc["expected_nl_response"] = response_text

            # Track cost
            cost = calculate_cost(usage)
            stats["cost"] += cost
            stats["generated"] += 1

            # Update checkpoint
            completed_ids.add(tc_id)

            # Progress update every batch_size
            if stats["generated"] % batch_size == 0:
                print(f"    Progress: {stats['generated']}/{total - len(completed_ids) + stats['generated']} (${stats['cost']:.4f})")

                # Save checkpoint periodically
                checkpoint["completed"][category] = list(completed_ids)
                checkpoint["total_cost"] = checkpoint.get("total_cost", 0) + cost
                save_checkpoint(checkpoint)

            # Small delay to avoid rate limits
            await asyncio.sleep(0.1)

        except Exception as e:
            print(f"    Error for {tc_id}: {e}")
            stats["errors"] += 1

    # Save updated fixture
    if not dry_run and stats["generated"] > 0:
        # Update _meta with generation info
        data["_meta"]["nl_response_generated_at"] = datetime.now(timezone.utc).isoformat()
        data["_meta"]["nl_response_model"] = HAIKU_MODEL

        with open(fixture_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"    Saved {fixture_path}")

    # Final checkpoint save
    checkpoint["completed"][category] = list(completed_ids)
    save_checkpoint(checkpoint)

    return stats


async def main_async(args: argparse.Namespace) -> int:
    """Async main function."""
    # Get API key
    api_key = os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("NL2API_ANTHROPIC_API_KEY")

    if not api_key and not args.dry_run:
        print("Error: ANTHROPIC_API_KEY or NL2API_ANTHROPIC_API_KEY environment variable required")
        return 1

    # Initialize provider (only if not dry run)
    provider = None
    if not args.dry_run:
        provider = ClaudeProvider(api_key=api_key, model=HAIKU_MODEL)
        print(f"Using model: {HAIKU_MODEL}")

    # Load checkpoint
    checkpoint = load_checkpoint()
    print(f"Loaded checkpoint: {sum(len(v) for v in checkpoint.get('completed', {}).values())} test cases already processed")

    # Determine categories to process
    if args.category == "all":
        categories = NL_RESPONSE_CATEGORIES
    else:
        categories = [args.category]

    print(f"\nProcessing categories: {', '.join(categories)}")
    if args.dry_run:
        print("[DRY RUN MODE - No API calls will be made]\n")

    # Process each category
    total_stats = {
        "total": 0,
        "generated": 0,
        "skipped": 0,
        "errors": 0,
        "cost": 0.0,
    }

    for category in categories:
        print(f"\n{'='*60}")
        print(f"Category: {category}")
        print(f"{'='*60}")

        stats = await process_category(
            category=category,
            provider=provider,
            checkpoint=checkpoint,
            dry_run=args.dry_run,
            batch_size=args.batch_size,
        )

        if not stats.get("skipped") == True:
            for key in ["total", "generated", "skipped", "errors", "cost"]:
                total_stats[key] += stats.get(key, 0)

    # Print summary
    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"Total test cases: {total_stats['total']}")
    print(f"Generated: {total_stats['generated']}")
    print(f"Skipped: {total_stats['skipped']}")
    print(f"Errors: {total_stats['errors']}")
    print(f"Total cost: ${total_stats['cost']:.4f}")
    print(f"Cumulative cost (all runs): ${checkpoint.get('total_cost', 0):.4f}")

    if args.dry_run:
        print("\n[DRY RUN - No changes made]")

    return 0


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate expected_nl_response values using Claude 3.5 Haiku",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--category",
        choices=NL_RESPONSE_CATEGORIES + ["all"],
        default="all",
        help="Category to process (default: all)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Preview without making API calls",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Checkpoint save frequency (default: 10)",
    )

    parser.add_argument(
        "--reset-checkpoint",
        action="store_true",
        help="Reset checkpoint and start fresh",
    )

    args = parser.parse_args()

    # Reset checkpoint if requested
    if args.reset_checkpoint and CHECKPOINT_FILE.exists():
        CHECKPOINT_FILE.unlink()
        print("Checkpoint reset.")

    return asyncio.run(main_async(args))


if __name__ == "__main__":
    sys.exit(main())
