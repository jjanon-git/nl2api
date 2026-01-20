"""
Unit tests for in-memory repository implementations.

These tests verify the InMemoryTestCaseRepository and InMemoryScorecardRepository
work correctly without requiring any external dependencies.
"""

from __future__ import annotations

import pytest

from CONTRACTS import (
    EvaluationStage,
    Scorecard,
    StageResult,
    TestCase,
    TestCaseMetadata,
    ToolCall,
)
from src.storage.memory import InMemoryTestCaseRepository, InMemoryScorecardRepository


# =============================================================================
# TestCaseRepository Tests
# =============================================================================


@pytest.fixture
def test_case_repo() -> InMemoryTestCaseRepository:
    """Create a fresh in-memory test case repository."""
    return InMemoryTestCaseRepository()


@pytest.fixture
def sample_test_case() -> TestCase:
    """Create a sample test case for testing."""
    return TestCase(
        id="test-123",
        nl_query="Find all products under $50",
        expected_tool_calls=(
            ToolCall(tool_name="search_products", arguments={"max_price": 50}),
        ),
        expected_nl_response="Here are the products under $50",
        metadata=TestCaseMetadata(
            api_version="v1.0.0",
            complexity_level=2,
            tags=("search", "products"),
        ),
    )


@pytest.mark.asyncio
async def test_save_and_get(test_case_repo: InMemoryTestCaseRepository, sample_test_case: TestCase):
    """Test saving and retrieving a test case."""
    await test_case_repo.save(sample_test_case)

    retrieved = await test_case_repo.get(sample_test_case.id)
    assert retrieved is not None
    assert retrieved.id == sample_test_case.id
    assert retrieved.nl_query == sample_test_case.nl_query


@pytest.mark.asyncio
async def test_get_nonexistent(test_case_repo: InMemoryTestCaseRepository):
    """Test retrieving a non-existent test case returns None."""
    result = await test_case_repo.get("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_get_many(test_case_repo: InMemoryTestCaseRepository, sample_test_case: TestCase):
    """Test retrieving multiple test cases."""
    # Create a second test case
    test_case_2 = TestCase(
        id="test-456",
        nl_query="Get user profile",
        expected_tool_calls=(
            ToolCall(tool_name="get_user", arguments={"user_id": 123}),
        ),
        expected_nl_response="User profile loaded",
        metadata=TestCaseMetadata(api_version="v1.0.0", complexity_level=1),
    )

    await test_case_repo.save(sample_test_case)
    await test_case_repo.save(test_case_2)

    # Get both
    results = await test_case_repo.get_many(["test-123", "test-456", "nonexistent"])
    assert len(results) == 2
    assert {tc.id for tc in results} == {"test-123", "test-456"}


@pytest.mark.asyncio
async def test_delete(test_case_repo: InMemoryTestCaseRepository, sample_test_case: TestCase):
    """Test deleting a test case."""
    await test_case_repo.save(sample_test_case)

    # Delete should return True
    deleted = await test_case_repo.delete(sample_test_case.id)
    assert deleted is True

    # Should no longer exist
    result = await test_case_repo.get(sample_test_case.id)
    assert result is None

    # Delete again should return False
    deleted_again = await test_case_repo.delete(sample_test_case.id)
    assert deleted_again is False


@pytest.mark.asyncio
async def test_list_with_filters(test_case_repo: InMemoryTestCaseRepository):
    """Test listing with filters."""
    # Create test cases with different attributes
    tc1 = TestCase(
        id="tc-1",
        nl_query="Query 1",
        expected_tool_calls=(ToolCall(tool_name="tool1", arguments={}),),
        expected_nl_response="Response 1",
        metadata=TestCaseMetadata(
            api_version="v1.0.0",
            complexity_level=1,
            tags=("search", "simple"),
        ),
    )
    tc2 = TestCase(
        id="tc-2",
        nl_query="Query 2",
        expected_tool_calls=(ToolCall(tool_name="tool2", arguments={}),),
        expected_nl_response="Response 2",
        metadata=TestCaseMetadata(
            api_version="v1.0.0",
            complexity_level=3,
            tags=("search", "complex"),
        ),
    )
    tc3 = TestCase(
        id="tc-3",
        nl_query="Query 3",
        expected_tool_calls=(ToolCall(tool_name="tool3", arguments={}),),
        expected_nl_response="Response 3",
        metadata=TestCaseMetadata(
            api_version="v1.0.0",
            complexity_level=5,
            tags=("checkout",),
        ),
    )

    await test_case_repo.save(tc1)
    await test_case_repo.save(tc2)
    await test_case_repo.save(tc3)

    # Filter by tags
    results = await test_case_repo.list(tags=["search"])
    assert len(results) == 2

    # Filter by complexity
    results = await test_case_repo.list(complexity_min=3)
    assert len(results) == 2

    results = await test_case_repo.list(complexity_max=3)
    assert len(results) == 2

    # Combined filters
    results = await test_case_repo.list(tags=["search"], complexity_min=2)
    assert len(results) == 1
    assert results[0].id == "tc-2"


@pytest.mark.asyncio
async def test_search_text(test_case_repo: InMemoryTestCaseRepository, sample_test_case: TestCase):
    """Test text search."""
    await test_case_repo.save(sample_test_case)

    results = await test_case_repo.search_text("products")
    assert len(results) == 1
    assert results[0].id == sample_test_case.id

    results = await test_case_repo.search_text("nonexistent")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_count(test_case_repo: InMemoryTestCaseRepository, sample_test_case: TestCase):
    """Test counting test cases."""
    await test_case_repo.save(sample_test_case)

    count = await test_case_repo.count()
    assert count == 1

    count = await test_case_repo.count(tags=["search"])
    assert count == 1

    count = await test_case_repo.count(tags=["nonexistent"])
    assert count == 0


# =============================================================================
# ScorecardRepository Tests
# =============================================================================


@pytest.fixture
def scorecard_repo() -> InMemoryScorecardRepository:
    """Create a fresh in-memory scorecard repository."""
    return InMemoryScorecardRepository()


@pytest.fixture
def sample_scorecard() -> Scorecard:
    """Create a sample scorecard for testing."""
    return Scorecard(
        test_case_id="test-123",
        batch_id="batch-001",
        syntax_result=StageResult(
            stage=EvaluationStage.SYNTAX,
            passed=True,
            score=1.0,
            reason="Valid JSON",
        ),
        logic_result=StageResult(
            stage=EvaluationStage.LOGIC,
            passed=True,
            score=1.0,
            reason="All tool calls match",
        ),
        worker_id="test-worker",
    )


@pytest.mark.asyncio
async def test_scorecard_save_and_get(scorecard_repo: InMemoryScorecardRepository, sample_scorecard: Scorecard):
    """Test saving and retrieving a scorecard."""
    await scorecard_repo.save(sample_scorecard)

    retrieved = await scorecard_repo.get(sample_scorecard.scorecard_id)
    assert retrieved is not None
    assert retrieved.scorecard_id == sample_scorecard.scorecard_id
    assert retrieved.test_case_id == sample_scorecard.test_case_id


@pytest.mark.asyncio
async def test_get_by_test_case(scorecard_repo: InMemoryScorecardRepository, sample_scorecard: Scorecard):
    """Test getting scorecards by test case ID."""
    await scorecard_repo.save(sample_scorecard)

    results = await scorecard_repo.get_by_test_case("test-123")
    assert len(results) == 1
    assert results[0].test_case_id == "test-123"

    results = await scorecard_repo.get_by_test_case("nonexistent")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_get_by_batch(scorecard_repo: InMemoryScorecardRepository, sample_scorecard: Scorecard):
    """Test getting scorecards by batch ID."""
    await scorecard_repo.save(sample_scorecard)

    results = await scorecard_repo.get_by_batch("batch-001")
    assert len(results) == 1

    results = await scorecard_repo.get_by_batch("nonexistent")
    assert len(results) == 0


@pytest.mark.asyncio
async def test_get_latest(scorecard_repo: InMemoryScorecardRepository):
    """Test getting the latest scorecard for a test case."""
    # Create two scorecards for the same test case
    sc1 = Scorecard(
        test_case_id="test-123",
        syntax_result=StageResult(stage=EvaluationStage.SYNTAX, passed=True, score=1.0),
        worker_id="worker-1",
    )
    sc2 = Scorecard(
        test_case_id="test-123",
        syntax_result=StageResult(stage=EvaluationStage.SYNTAX, passed=False, score=0.0),
        worker_id="worker-2",
    )

    await scorecard_repo.save(sc1)
    await scorecard_repo.save(sc2)

    latest = await scorecard_repo.get_latest("test-123")
    assert latest is not None
    # The latest should be sc2 (second saved)
    assert latest.worker_id == "worker-2"


@pytest.mark.asyncio
async def test_batch_summary(scorecard_repo: InMemoryScorecardRepository):
    """Test getting batch summary statistics."""
    # Create scorecards with different pass/fail states
    for i in range(5):
        sc = Scorecard(
            test_case_id=f"test-{i}",
            batch_id="batch-001",
            syntax_result=StageResult(
                stage=EvaluationStage.SYNTAX,
                passed=i < 3,  # First 3 pass
                score=1.0 if i < 3 else 0.0,
            ),
            logic_result=StageResult(
                stage=EvaluationStage.LOGIC,
                passed=i < 3,
                score=1.0 if i < 3 else 0.0,
            ),
            worker_id="worker",
        )
        await scorecard_repo.save(sc)

    summary = await scorecard_repo.get_batch_summary("batch-001")
    assert summary["total"] == 5
    assert summary["passed"] == 3
    assert summary["failed"] == 2


@pytest.mark.asyncio
async def test_scorecard_delete(scorecard_repo: InMemoryScorecardRepository, sample_scorecard: Scorecard):
    """Test deleting a scorecard."""
    await scorecard_repo.save(sample_scorecard)

    deleted = await scorecard_repo.delete(sample_scorecard.scorecard_id)
    assert deleted is True

    result = await scorecard_repo.get(sample_scorecard.scorecard_id)
    assert result is None


@pytest.mark.asyncio
async def test_clear(test_case_repo: InMemoryTestCaseRepository, sample_test_case: TestCase):
    """Test clearing all data."""
    await test_case_repo.save(sample_test_case)
    assert await test_case_repo.count() == 1

    test_case_repo.clear()
    assert await test_case_repo.count() == 0
