"""
Integration tests for batch CLI commands.

Tests the compare and trend commands against a real database.
Requires: docker compose up -d
"""

import uuid
from datetime import UTC, datetime, timedelta

import pytest

from CONTRACTS import (
    EvaluationStage,
    Scorecard,
    StageResult,
    TestCase,
    TestCaseMetadata,
    ToolCall,
)
import pytest_asyncio

from src.common.storage.postgres import (
    PostgresScorecardRepository,
    PostgresTestCaseRepository,
)


@pytest_asyncio.fixture(loop_scope="session")
async def scorecard_repo(db_pool):
    """Create a scorecard repository."""
    return PostgresScorecardRepository(db_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def test_case_repo(db_pool):
    """Create a test case repository."""
    return PostgresTestCaseRepository(db_pool)


@pytest_asyncio.fixture(loop_scope="session")
async def cleanup_test_data(db_pool):
    """Clean up test data after each test."""
    yield
    async with db_pool.acquire() as conn:
        # Clean up test scorecards with our test client types
        await conn.execute(
            "DELETE FROM scorecards WHERE client_type IN ('test_internal', 'test_mcp_claude')"
        )


def create_test_scorecard(
    test_case_id: str,
    batch_id: str,
    client_type: str = "test_internal",
    client_version: str | None = None,
    passed: bool = True,
    score: float = 1.0,
    latency_ms: int = 100,
    input_tokens: int | None = None,
    output_tokens: int | None = None,
    cost_usd: float | None = None,
) -> Scorecard:
    """Create a test scorecard with the given parameters."""
    return Scorecard(
        test_case_id=test_case_id,
        batch_id=batch_id,
        scorecard_id=str(uuid.uuid4()),
        client_type=client_type,
        client_version=client_version,
        eval_mode="orchestrator",
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=cost_usd,
        syntax_result=StageResult(
            stage=EvaluationStage.SYNTAX,
            passed=passed,
            score=score,
            reason="Test result",
        ),
        overall_passed=passed,
        overall_score=score,
        worker_id="test-worker",
        total_latency_ms=latency_ms,
    )


@pytest.mark.asyncio(loop_scope="session")
class TestScorecardRepositoryClientMethods:
    """Tests for the new client-based query methods."""

    async def test_get_by_client(self, scorecard_repo, cleanup_test_data):
        """Test fetching scorecards by client type."""
        batch_id = str(uuid.uuid4())
        test_case_id = str(uuid.uuid4())

        # Create scorecards for different clients
        sc1 = create_test_scorecard(
            test_case_id, batch_id,
            client_type="test_internal",
            passed=True,
        )
        sc2 = create_test_scorecard(
            test_case_id, batch_id,
            client_type="test_mcp_claude",
            client_version="claude-3-5-sonnet",
            passed=False,
        )

        await scorecard_repo.save(sc1)
        await scorecard_repo.save(sc2)

        # Fetch by client type
        internal_cards = await scorecard_repo.get_by_client("test_internal")
        claude_cards = await scorecard_repo.get_by_client("test_mcp_claude")

        assert len(internal_cards) >= 1
        assert len(claude_cards) >= 1
        assert all(sc.client_type == "test_internal" for sc in internal_cards)
        assert all(sc.client_type == "test_mcp_claude" for sc in claude_cards)

    async def test_get_by_client_with_version(self, scorecard_repo, cleanup_test_data):
        """Test fetching scorecards by client type and version."""
        batch_id = str(uuid.uuid4())
        test_case_id = str(uuid.uuid4())

        # Create scorecards with different versions
        sc1 = create_test_scorecard(
            test_case_id, batch_id,
            client_type="test_mcp_claude",
            client_version="claude-3-5-sonnet",
        )
        sc2 = create_test_scorecard(
            test_case_id, batch_id,
            client_type="test_mcp_claude",
            client_version="claude-3-5-haiku",
        )

        await scorecard_repo.save(sc1)
        await scorecard_repo.save(sc2)

        # Fetch by client type and version
        sonnet_cards = await scorecard_repo.get_by_client(
            "test_mcp_claude", "claude-3-5-sonnet"
        )
        haiku_cards = await scorecard_repo.get_by_client(
            "test_mcp_claude", "claude-3-5-haiku"
        )

        assert len(sonnet_cards) >= 1
        assert len(haiku_cards) >= 1
        assert all(sc.client_version == "claude-3-5-sonnet" for sc in sonnet_cards)
        assert all(sc.client_version == "claude-3-5-haiku" for sc in haiku_cards)

    async def test_get_comparison_summary(self, scorecard_repo, cleanup_test_data):
        """Test getting comparison summary across clients."""
        batch_id = str(uuid.uuid4())
        test_case_id = str(uuid.uuid4())

        # Create scorecards for different clients with different results
        for i in range(5):
            sc = create_test_scorecard(
                test_case_id, batch_id,
                client_type="test_internal",
                passed=True,
                score=0.9,
                input_tokens=100,
                output_tokens=50,
                cost_usd=0.001,
            )
            await scorecard_repo.save(sc)

        for i in range(5):
            sc = create_test_scorecard(
                test_case_id, batch_id,
                client_type="test_mcp_claude",
                passed=i < 3,  # 3 pass, 2 fail
                score=0.8 if i < 3 else 0.2,
                input_tokens=200,
                output_tokens=100,
                cost_usd=0.002,
            )
            await scorecard_repo.save(sc)

        # Get comparison summary
        summaries = await scorecard_repo.get_comparison_summary(
            ["test_internal", "test_mcp_claude"]
        )

        assert len(summaries) >= 2

        internal_summary = next(
            (s for s in summaries if s["client_type"] == "test_internal"), None
        )
        claude_summary = next(
            (s for s in summaries if s["client_type"] == "test_mcp_claude"), None
        )

        assert internal_summary is not None
        assert claude_summary is not None
        assert internal_summary["total_tests"] >= 5
        assert internal_summary["pass_rate"] == 1.0  # All passed
        assert claude_summary["total_tests"] >= 5

    async def test_get_client_trend(self, scorecard_repo, cleanup_test_data):
        """Test getting trend data for a client."""
        test_case_id = str(uuid.uuid4())

        # Create scorecards
        for i in range(3):
            batch_id = str(uuid.uuid4())
            sc = create_test_scorecard(
                test_case_id, batch_id,
                client_type="test_internal",
                passed=True,
                score=0.9,
            )
            await scorecard_repo.save(sc)

        # Get trend data
        trend = await scorecard_repo.get_client_trend(
            "test_internal", "pass_rate", days=7
        )

        assert len(trend) >= 1
        assert all("date" in point and "value" in point for point in trend)

    async def test_scorecard_preserves_token_and_cost_data(
        self, scorecard_repo, cleanup_test_data
    ):
        """Test that token and cost data is properly saved and retrieved."""
        batch_id = str(uuid.uuid4())
        test_case_id = str(uuid.uuid4())

        sc = create_test_scorecard(
            test_case_id, batch_id,
            client_type="test_internal",
            input_tokens=1500,
            output_tokens=500,
            cost_usd=0.0075,
        )

        await scorecard_repo.save(sc)

        # Retrieve and verify
        retrieved = await scorecard_repo.get_by_client("test_internal", limit=1)
        assert len(retrieved) >= 1

        latest = retrieved[0]
        assert latest.input_tokens == 1500
        assert latest.output_tokens == 500
        assert latest.estimated_cost_usd == pytest.approx(0.0075, abs=0.0001)
