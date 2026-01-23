#!/usr/bin/env python3
"""
Run NL2API evaluation using Anthropic's Message Batches API with Haiku.

Benefits over synchronous calls:
- 50% cost reduction
- No rate limits (processed asynchronously)
- Better for large evaluation runs

Usage:
    # Create and submit batch for 100 test cases
    python scripts/run_batch_haiku_eval.py --limit 100

    # Check status of existing batch
    python scripts/run_batch_haiku_eval.py --status <batch_id>

    # Process results from completed batch
    python scripts/run_batch_haiku_eval.py --results <batch_id>

Or use the integrated CLI:
    python -m src.evaluation.cli.main batch api submit --limit 100
    python -m src.evaluation.cli.main batch api status <batch_id>
    python -m src.evaluation.cli.main batch api results <batch_id>
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# Load .env file
def _load_env():
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


@dataclass
class BatchRequest:
    """A single request in the batch."""

    custom_id: str
    params: dict


def get_agent_config(agent_type: str = "datastream") -> tuple[str, list[dict]]:
    """
    Get system prompt and tools from the actual domain agent.

    This ensures batch evaluation uses the exact same prompts as the agents.
    """
    # Import agents dynamically to get their actual prompts
    if agent_type == "datastream":
        from src.nl2api.agents.datastream import DatastreamAgent

        agent = DatastreamAgent(llm=None)  # type: ignore
    elif agent_type == "estimates":
        from src.nl2api.agents.estimates import EstimatesAgent

        agent = EstimatesAgent(llm=None)  # type: ignore
    elif agent_type == "fundamentals":
        from src.nl2api.agents.fundamentals import FundamentalsAgent

        agent = FundamentalsAgent(llm=None)  # type: ignore
    elif agent_type == "screening":
        from src.nl2api.agents.screening import ScreeningAgent

        agent = ScreeningAgent(llm=None)  # type: ignore
    elif agent_type == "officers":
        from src.nl2api.agents.officers import OfficersAgent

        agent = OfficersAgent(llm=None)  # type: ignore
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    # Get the system prompt from the agent
    system_prompt = agent.get_system_prompt()

    # Get tools and convert to API format
    tools = []
    for tool_def in agent.get_tools():
        tools.append(
            {
                "name": tool_def.name,
                "description": tool_def.description,
                "input_schema": tool_def.parameters,
            }
        )

    return system_prompt, tools


def create_batch_file(
    limit: int = 100,
    category: str | None = None,
    agent_type: str = "datastream",
    model: str = "claude-3-haiku-20240307",
) -> Path:
    """
    Create a JSONL batch file from test fixtures.

    Uses the actual agent's system prompt and tools for accurate evaluation.

    Returns path to the created file.
    """
    from tests.unit.nl2api.fixture_loader import FixtureLoader

    print(f"Loading {agent_type} agent configuration...")
    system_prompt, tools = get_agent_config(agent_type)
    print(f"  Loaded prompt ({len(system_prompt)} chars) and {len(tools)} tools")

    print("Loading test fixtures...")
    loader = FixtureLoader()

    if category:
        test_cases = loader.load_category(category)
    else:
        test_cases = list(loader.iterate_all())

    if limit:
        test_cases = test_cases[:limit]

    print(f"  Loaded {len(test_cases)} test cases")

    # Create batch requests
    batch_file = PROJECT_ROOT / "batch_requests.jsonl"

    with open(batch_file, "w") as f:
        for tc in test_cases:
            request = {
                "custom_id": tc.id,
                "params": {
                    "model": model,
                    "max_tokens": 1024,
                    "system": system_prompt,
                    "messages": [{"role": "user", "content": tc.nl_query}],
                    "tools": tools,
                    "tool_choice": {"type": "auto"},
                },
            }
            f.write(json.dumps(request) + "\n")

    print(f"  Created batch file: {batch_file}")
    print(f"  File size: {batch_file.stat().st_size / 1024:.1f} KB")

    return batch_file


def submit_batch(batch_file: Path) -> str:
    """
    Submit batch file to Anthropic's Message Batches API.

    Returns batch_id.
    """
    import anthropic

    client = anthropic.Anthropic()

    print("\nSubmitting batch to Anthropic...")

    # Upload the batch file
    with open(batch_file, "rb") as f:
        batch = client.beta.messages.batches.create(requests=[json.loads(line) for line in f])

    print(f"  Batch ID: {batch.id}")
    print(f"  Status: {batch.processing_status}")
    print(f"  Request counts: {batch.request_counts}")

    return batch.id


def check_batch_status(batch_id: str) -> dict:
    """Check the status of a batch."""
    import anthropic

    client = anthropic.Anthropic()
    batch = client.beta.messages.batches.retrieve(batch_id)

    return {
        "id": batch.id,
        "status": batch.processing_status,
        "created_at": batch.created_at,
        "ended_at": batch.ended_at,
        "request_counts": {
            "processing": batch.request_counts.processing,
            "succeeded": batch.request_counts.succeeded,
            "errored": batch.request_counts.errored,
            "canceled": batch.request_counts.canceled,
            "expired": batch.request_counts.expired,
        },
    }


def poll_until_complete(batch_id: str, poll_interval: int = 30) -> dict:
    """Poll batch status until complete."""
    print(f"\nPolling batch {batch_id} until complete...")
    print(f"  Poll interval: {poll_interval}s")

    while True:
        status = check_batch_status(batch_id)

        succeeded = status["request_counts"]["succeeded"]
        errored = status["request_counts"]["errored"]
        processing = status["request_counts"]["processing"]
        total = succeeded + errored + processing

        print(
            f"  [{datetime.now().strftime('%H:%M:%S')}] "
            f"Status: {status['status']} | "
            f"Succeeded: {succeeded}/{total} | "
            f"Errors: {errored}"
        )

        if status["status"] == "ended":
            print("\n  Batch complete!")
            return status

        time.sleep(poll_interval)


def download_results(batch_id: str) -> Path:
    """Download batch results to a JSONL file."""
    import anthropic

    client = anthropic.Anthropic()

    print(f"\nDownloading results for batch {batch_id}...")

    results_file = PROJECT_ROOT / f"batch_results_{batch_id}.jsonl"

    with open(results_file, "w") as f:
        for result in client.beta.messages.batches.results(batch_id):
            f.write(
                json.dumps(
                    {
                        "custom_id": result.custom_id,
                        "result": result.result.model_dump()
                        if hasattr(result.result, "model_dump")
                        else result.result,
                    }
                )
                + "\n"
            )

    print(f"  Saved to: {results_file}")
    return results_file


def evaluate_results(results_file: Path, limit: int = 100, category: str | None = None) -> dict:
    """
    Evaluate batch results against expected outcomes.

    Returns summary statistics.
    """
    from CONTRACTS import ToolRegistry
    from tests.unit.nl2api.fixture_loader import FixtureLoader

    print("\nEvaluating results...")

    # Load test cases to get expected values
    loader = FixtureLoader()
    if category:
        test_cases = loader.load_category(category)
    else:
        test_cases = list(loader.iterate_all())

    if limit:
        test_cases = test_cases[:limit]

    # Create lookup by ID
    expected_by_id = {tc.id: tc for tc in test_cases}

    # Process results
    results = []
    with open(results_file) as f:
        for line in f:
            results.append(json.loads(line))

    # Evaluate
    passed = 0
    failed = 0
    errors = 0
    failures = []

    for result in results:
        custom_id = result["custom_id"]
        expected = expected_by_id.get(custom_id)

        if not expected:
            errors += 1
            continue

        result_data = result["result"]

        # Check for API errors
        if result_data.get("type") == "error":
            errors += 1
            failures.append(
                {
                    "id": custom_id,
                    "query": expected.nl_query[:50],
                    "error": result_data.get("error", {}).get("message", "Unknown error"),
                }
            )
            continue

        # Extract tool calls from response
        message = result_data.get("message", {})
        content = message.get("content", [])

        tool_calls = [c for c in content if c.get("type") == "tool_use"]

        if not tool_calls:
            failed += 1
            failures.append(
                {
                    "id": custom_id,
                    "query": expected.nl_query[:50],
                    "error": "No tool calls returned",
                }
            )
            continue

        # Compare tool calls
        actual_func = tool_calls[0].get("name", "")
        expected_func = (
            expected.expected_tool_calls[0].get("function", "")
            if expected.expected_tool_calls
            else ""
        )

        actual_normalized = ToolRegistry.normalize(actual_func)
        expected_normalized = ToolRegistry.normalize(expected_func)

        if actual_normalized == expected_normalized:
            # Check fields if function matches
            actual_args = tool_calls[0].get("input", {})
            expected_args = expected.expected_tool_calls[0].get("arguments", {})

            actual_fields = set(actual_args.get("fields", []))
            expected_fields = set(expected_args.get("fields", []))

            if not expected_fields or actual_fields == expected_fields:
                passed += 1
            else:
                failed += 1
                failures.append(
                    {
                        "id": custom_id,
                        "query": expected.nl_query[:50],
                        "error": f"Fields mismatch: expected {expected_fields}, got {actual_fields}",
                    }
                )
        else:
            failed += 1
            failures.append(
                {
                    "id": custom_id,
                    "query": expected.nl_query[:50],
                    "error": f"Function mismatch: expected {expected_func}, got {actual_func}",
                }
            )

    total = passed + failed + errors
    pass_rate = (passed / total * 100) if total > 0 else 0

    summary = {
        "total": total,
        "passed": passed,
        "failed": failed,
        "errors": errors,
        "pass_rate": pass_rate,
        "failures": failures[:20],  # Keep first 20
    }

    return summary


def print_summary(summary: dict):
    """Print evaluation summary."""
    print("\n" + "=" * 60)
    print("BATCH EVALUATION COMPLETE")
    print("=" * 60)

    print("\nOverall Results:")
    print(f"  Total:      {summary['total']}")
    print(f"  Passed:     {summary['passed']} ({summary['pass_rate']:.1f}%)")
    print(f"  Failed:     {summary['failed']}")
    print(f"  Errors:     {summary['errors']}")

    if summary["failures"]:
        print(f"\nSample Failures (first {len(summary['failures'])}):")
        for f in summary["failures"][:10]:
            print(f"  - {f['query']}...")
            print(f"    {f['error']}")

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Run NL2API batch evaluation")
    parser.add_argument("--limit", type=int, default=100, help="Number of test cases")
    parser.add_argument("--category", type=str, help="Specific category to test")
    parser.add_argument(
        "--agent",
        type=str,
        default="datastream",
        choices=["datastream", "estimates", "fundamentals", "screening", "officers"],
        help="Agent to evaluate (default: datastream)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="haiku",
        choices=["haiku", "sonnet"],
        help="Model to use (default: haiku)",
    )
    parser.add_argument("--status", type=str, help="Check status of batch ID")
    parser.add_argument("--results", type=str, help="Process results from batch ID")
    parser.add_argument("--poll", type=str, help="Poll batch ID until complete")
    parser.add_argument(
        "--no-submit", action="store_true", help="Create batch file but don't submit"
    )
    args = parser.parse_args()

    # Model mapping
    model_map = {
        "haiku": "claude-3-haiku-20240307",
        "sonnet": "claude-sonnet-4-20250514",
    }
    model_id = model_map.get(args.model, args.model)

    # Check for API key
    api_key = os.environ.get("NL2API_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("ERROR: ANTHROPIC_API_KEY or NL2API_ANTHROPIC_API_KEY not set!")
        sys.exit(1)

    # Set for anthropic client
    os.environ["ANTHROPIC_API_KEY"] = api_key

    try:
        # Check status of existing batch
        if args.status:
            status = check_batch_status(args.status)
            print(json.dumps(status, indent=2, default=str))
            return

        # Poll until complete
        if args.poll:
            status = poll_until_complete(args.poll)
            print(json.dumps(status, indent=2, default=str))
            return

        # Process results from completed batch
        if args.results:
            results_file = download_results(args.results)
            summary = evaluate_results(results_file, args.limit, args.category)
            print_summary(summary)

            # Save summary
            summary_file = PROJECT_ROOT / f"batch_summary_{args.results}.json"
            with open(summary_file, "w") as f:
                json.dump(summary, f, indent=2)
            print(f"\nSummary saved to: {summary_file}")
            return

        # Create and submit new batch
        print("=" * 60)
        print("NL2API Batch Evaluation")
        print("=" * 60)
        print(f"  Agent: {args.agent}")
        print(f"  Model: {model_id}")
        print(f"  Limit: {args.limit}")
        if args.category:
            print(f"  Category: {args.category}")
        print()

        # Create batch file
        batch_file = create_batch_file(args.limit, args.category, args.agent, model_id)

        if args.no_submit:
            print(f"\nBatch file created (not submitted): {batch_file}")
            return

        # Submit batch
        batch_id = submit_batch(batch_file)

        print("\nBatch submitted successfully!")
        print("\nNext steps:")
        print(f"  1. Check status:  python scripts/run_batch_haiku_eval.py --status {batch_id}")
        print(f"  2. Poll progress: python scripts/run_batch_haiku_eval.py --poll {batch_id}")
        print(f"  3. Get results:   python scripts/run_batch_haiku_eval.py --results {batch_id}")

        # Save batch ID for reference
        batch_info = {
            "batch_id": batch_id,
            "created_at": datetime.now().isoformat(),
            "limit": args.limit,
            "category": args.category,
            "agent": args.agent,
            "model": model_id,
        }
        batch_info_file = PROJECT_ROOT / "batch_info.json"
        with open(batch_info_file, "w") as f:
            json.dump(batch_info, f, indent=2)
        print(f"\nBatch info saved to: {batch_info_file}")

    except Exception as e:
        print(f"\nERROR: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
