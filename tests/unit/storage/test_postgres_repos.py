"""Unit tests for PostgreSQL repository implementations."""

import os
import uuid

import pytest

from CONTRACTS import (
    BatchJob as BatchJobModel,
)
from CONTRACTS import (
    EvaluationStage as EvaluationStageEnum,
)
from CONTRACTS import (
    Scorecard as ScorecardModel,
)
from CONTRACTS import (
    StageResult as StageResultModel,
)
from CONTRACTS import (
    TaskPriority as TaskPriorityEnum,
)
from CONTRACTS import (
    TaskStatus as TaskStatusEnum,
)
from CONTRACTS import (
    TestCase as TestCaseModel,
)
from CONTRACTS import (
    TestCaseMetadata as TestCaseMetadataModel,
)
from CONTRACTS import (
    ToolCall as ToolCallModel,
)
from src.common.storage.postgres import (
    PostgresBatchJobRepository,
    PostgresScorecardRepository,
    PostgresTestCaseRepository,
    close_pool,
    create_pool,
)

# Use the default database URL if not provided in environment
DATABASE_URL = os.environ.get("DATABASE_URL", "postgresql://nl2api:nl2api@localhost:5432/nl2api")


@pytest.fixture
async def db_pool():
    """Create a database connection pool for tests."""
    await close_pool()
    pool = await create_pool(DATABASE_URL)
    yield pool
    await close_pool()


@pytest.fixture(autouse=True)
async def cleanup_db(db_pool):
    """Clean up the database before each test."""
    async with db_pool.acquire() as conn:
        await conn.execute("DELETE FROM scorecards")
        await conn.execute("DELETE FROM batch_jobs")
        await conn.execute("DELETE FROM test_cases")


def create_test_metadata():
    return TestCaseMetadataModel(
        api_version="1.0.0", complexity_level=1, tags=("price", "tech"), author="test-bot"
    )


class TestPostgresTestCaseRepository:
    """Test suite for PostgresTestCaseRepository."""

    @pytest.mark.asyncio
    async def test_save_and_get_test_case(self, db_pool):
        repo = PostgresTestCaseRepository(db_pool)
        tc_id = str(uuid.uuid4())

        test_case = TestCaseModel(
            id=tc_id,
            nl_query="What is Apple's price?",
            expected_tool_calls=(
                ToolCallModel(tool_name="get_price", arguments={"ticker": "AAPL"}),
            ),
            expected_nl_response="The price of Apple is $150.",
            metadata=create_test_metadata(),
        )

        await repo.save(test_case)

        retrieved = await repo.get(tc_id)
        assert retrieved is not None
        assert retrieved.id == tc_id
        assert retrieved.nl_query == test_case.nl_query
        assert retrieved.metadata.tags == ("price", "tech")
        assert len(retrieved.expected_tool_calls) == 1
        assert retrieved.expected_tool_calls[0].tool_name == "get_price"

    @pytest.mark.asyncio
    async def test_search_text(self, db_pool):
        repo = PostgresTestCaseRepository(db_pool)

        for i in range(3):
            tc = TestCaseModel(
                id=str(uuid.uuid4()),
                nl_query=f"Query number {i} about growth",
                expected_tool_calls=(ToolCallModel(tool_name="get_growth", arguments={}),),
                expected_nl_response="Response",
                metadata=create_test_metadata(),
            )
            await repo.save(tc)

        results = await repo.search_text("growth")
        assert len(results) >= 3


class TestPostgresBatchJobRepository:
    """Test suite for PostgresBatchJobRepository."""

    @pytest.mark.asyncio
    async def test_create_and_get_batch(self, db_pool):
        repo = PostgresBatchJobRepository(db_pool)
        batch_id = str(uuid.uuid4())

        batch = BatchJobModel(
            batch_id=batch_id,
            total_tests=10,
            completed_count=0,
            failed_count=0,
            status=TaskStatusEnum.PENDING,
            submitted_by="test-user",
            priority=TaskPriorityEnum.NORMAL,
            tags=("test",),
        )

        await repo.create(batch)

        retrieved = await repo.get(batch_id)
        assert retrieved is not None
        assert retrieved.batch_id == batch_id
        assert retrieved.total_tests == 10
        assert retrieved.status == TaskStatusEnum.PENDING

    @pytest.mark.asyncio
    async def test_update_batch(self, db_pool):
        repo = PostgresBatchJobRepository(db_pool)
        batch_id = str(uuid.uuid4())

        batch = BatchJobModel(batch_id=batch_id, total_tests=10, status=TaskStatusEnum.PENDING)
        await repo.create(batch)

        updated_batch = batch.model_copy(
            update={"completed_count": 5, "status": TaskStatusEnum.IN_PROGRESS}
        )
        await repo.update(updated_batch)

        retrieved = await repo.get(batch_id)
        assert retrieved.completed_count == 5
        assert retrieved.status == TaskStatusEnum.IN_PROGRESS


class TestPostgresScorecardRepository:
    """Test suite for PostgresScorecardRepository."""

    @pytest.mark.asyncio
    async def test_save_and_get_scorecard(self, db_pool):
        # We need a test case first
        tc_repo = PostgresTestCaseRepository(db_pool)
        tc_id = str(uuid.uuid4())
        await tc_repo.save(
            TestCaseModel(
                id=tc_id,
                nl_query="Q",
                expected_tool_calls=(ToolCallModel(tool_name="t", arguments={}),),
                expected_nl_response="R",
                metadata=create_test_metadata(),
            )
        )

        repo = PostgresScorecardRepository(db_pool)
        sc_id = str(uuid.uuid4())

        scorecard = ScorecardModel(
            scorecard_id=sc_id,
            test_case_id=tc_id,
            batch_id="test-batch",
            syntax_result=StageResultModel(
                stage=EvaluationStageEnum.SYNTAX, passed=True, score=1.0, duration_ms=100
            ),
            worker_id="worker-1",
        )

        await repo.save(scorecard)

        retrieved = await repo.get(sc_id)
        assert retrieved is not None
        assert retrieved.scorecard_id == sc_id
        assert retrieved.test_case_id == tc_id
        assert retrieved.syntax_result.passed is True
        assert retrieved.overall_score == 1.0

    @pytest.mark.asyncio
    async def test_save_batch(self, db_pool):
        repo = PostgresScorecardRepository(db_pool)
        tc_id = str(uuid.uuid4())

        scorecards = [
            ScorecardModel(
                scorecard_id=str(uuid.uuid4()),
                test_case_id=tc_id,
                batch_id="batch-1",
                syntax_result=StageResultModel(
                    stage=EvaluationStageEnum.SYNTAX, passed=True, score=1.0
                ),
                worker_id="w1",
            )
            for _ in range(5)
        ]

        saved_count = await repo.save_batch(scorecards)
        assert saved_count == 5

        retrieved_list = await repo.get_by_batch("batch-1")
        assert len(retrieved_list) == 5
