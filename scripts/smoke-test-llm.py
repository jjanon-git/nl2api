#!/usr/bin/env python3
"""
LLM API Smoke Tests

Verifies that LLM client code works correctly against real APIs.
Run this after modifying LLM clients or adding new model support.

Usage:
    python scripts/smoke-test-llm.py           # Run all tests
    python scripts/smoke-test-llm.py --openai  # OpenAI only
    python scripts/smoke-test-llm.py --anthropic  # Anthropic only
"""

import argparse
import asyncio
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv

load_dotenv()


async def test_openai_gpt5() -> bool:
    """Test OpenAI GPT-5-nano with reasoning_effort parameter."""
    from src.evalkit.common.llm import create_openai_client, openai_chat_completion

    print("Testing OpenAI gpt-5-nano with reasoning_effort='minimal'...")
    try:
        client = create_openai_client(async_client=True)
        response = await openai_chat_completion(
            client,
            model="gpt-5-nano",
            messages=[{"role": "user", "content": "Say 'ok' and nothing else"}],
            max_tokens=10,
            reasoning_effort="minimal",
        )
        result = response.choices[0].message.content
        print(f"  ✓ Response: {result}")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {type(e).__name__}: {e}")
        return False


async def test_openai_gpt4o() -> bool:
    """Test OpenAI gpt-4o-mini with temperature parameter."""
    from src.evalkit.common.llm import create_openai_client, openai_chat_completion

    print("Testing OpenAI gpt-4o-mini with temperature=0...")
    try:
        client = create_openai_client(async_client=True)
        response = await openai_chat_completion(
            client,
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'ok' and nothing else"}],
            max_tokens=10,
            temperature=0,
        )
        result = response.choices[0].message.content
        print(f"  ✓ Response: {result}")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {type(e).__name__}: {e}")
        return False


async def test_anthropic() -> bool:
    """Test Anthropic claude-3-5-haiku with standard parameters."""
    from src.evalkit.common.llm import anthropic_message_create, create_anthropic_client

    print("Testing Anthropic claude-3-5-haiku-20241022 with temperature=0...")
    try:
        client = create_anthropic_client(async_client=True)
        message = await anthropic_message_create(
            client,
            model="claude-3-5-haiku-20241022",
            messages=[{"role": "user", "content": "Say 'ok' and nothing else"}],
            max_tokens=10,
            temperature=0,
        )
        result = message.content[0].text
        print(f"  ✓ Response: {result}")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {type(e).__name__}: {e}")
        return False


async def main(args: argparse.Namespace) -> int:
    """Run smoke tests."""
    results = []

    if args.openai or (not args.openai and not args.anthropic):
        results.append(("OpenAI gpt-5-nano", await test_openai_gpt5()))
        results.append(("OpenAI gpt-4o-mini", await test_openai_gpt4o()))

    if args.anthropic or (not args.openai and not args.anthropic):
        results.append(("Anthropic claude-3-5-haiku", await test_anthropic()))

    print()
    print("=" * 50)
    passed = sum(1 for _, ok in results if ok)
    failed = len(results) - passed
    print(f"Results: {passed} passed, {failed} failed")

    if failed > 0:
        print("\nFailed tests:")
        for name, ok in results:
            if not ok:
                print(f"  - {name}")
        return 1

    print("\n✓ All smoke tests passed!")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM API smoke tests")
    parser.add_argument("--openai", action="store_true", help="Test OpenAI only")
    parser.add_argument("--anthropic", action="store_true", help="Test Anthropic only")
    args = parser.parse_args()

    sys.exit(asyncio.run(main(args)))
