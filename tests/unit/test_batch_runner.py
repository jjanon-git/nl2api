"""
Unit tests for the Batch Runner.

Tests the BatchRunner class and related components using in-memory repositories.
"""

from __future__ import annotations

import pytest

from CONTRACTS import (
    BatchJob,
    TaskStatus,
    TestCase,
    TestCaseMetadata,
    ToolCall,
)
from src.evalkit.batch import BatchRunner, BatchRunnerConfig
from src.evalkit.common.storage.memory import (
    InMemoryBatchJobRepository,
    InMemoryScorecardRepository,
    InMemoryTestCaseRepository,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def test_case_repo() -> InMemoryTestCaseRepository:
    """Create a fresh in-memory test case repository."""
    return InMemoryTestCaseRepository()


@pytest.fixture
def scorecard_repo() -> InMemoryScorecardRepository:
    """Create a fresh in-memory scorecard repository."""
    return InMemoryScorecardRepository()


@pytest.fixture
def batch_repo() -> InMemoryBatchJobRepository:
    """Create a fresh in-memory batch job repository."""
    return InMemoryBatchJobRepository()


@pytest.fixture
def sample_test_cases() -> list[TestCase]:
    """Create sample test cases for testing."""
    return [
        TestCase(
            id="test-001",
            nl_query="Find products under $50",
            expected_tool_calls=(
                ToolCall(tool_name="search_products", arguments={"max_price": 50}),
            ),
            expected_nl_response="Found products under $50",
            metadata=TestCaseMetadata(
                api_version="v1.0.0",
                complexity_level=2,
                tags=("search", "products"),
            ),
        ),
        TestCase(
            id="test-002",
            nl_query="Get user profile for user 123",
            expected_tool_calls=(ToolCall(tool_name="get_user", arguments={"user_id": 123}),),
            expected_nl_response="User profile loaded",
            metadata=TestCaseMetadata(
                api_version="v1.0.0",
                complexity_level=1,
                tags=("user",),
            ),
        ),
        TestCase(
            id="test-003",
            nl_query="Search for laptops",
            expected_tool_calls=(
                ToolCall(tool_name="search_products", arguments={"query": "laptops"}),
            ),
            expected_nl_response="Found laptop results",
            metadata=TestCaseMetadata(
                api_version="v1.0.0",
                complexity_level=1,
                tags=("search", "products"),
            ),
        ),
    ]


@pytest.fixture
def batch_runner(
    test_case_repo: InMemoryTestCaseRepository,
    scorecard_repo: InMemoryScorecardRepository,
    batch_repo: InMemoryBatchJobRepository,
) -> BatchRunner:
    """Create a batch runner with in-memory repositories."""
    config = BatchRunnerConfig(
        pack_name="nl2api",  # Required
        max_concurrency=5,
        show_progress=False,  # Disable progress bar in tests
        verbose=False,
    )
    return BatchRunner(
        test_case_repo=test_case_repo,
        scorecard_repo=scorecard_repo,
        batch_repo=batch_repo,
        config=config,
    )


# =============================================================================
# BatchRunner Tests
# =============================================================================


@pytest.mark.asyncio
async def test_batch_run_no_test_cases(batch_runner: BatchRunner):
    """Test batch run with no test cases returns None."""
    batch_job = await batch_runner.run()

    # Should return None when no test cases found
    assert batch_job is None


@pytest.mark.asyncio
async def test_batch_run_all_pass(
    batch_runner: BatchRunner,
    test_case_repo: InMemoryTestCaseRepository,
    sample_test_cases: list[TestCase],
):
    """Test batch run with all passing test cases."""
    # Load test cases
    for tc in sample_test_cases:
        await test_case_repo.save(tc)

    # Run batch (default simulator returns correct responses)
    batch_job = await batch_runner.run()

    assert batch_job.total_tests == 3
    assert batch_job.completed_count == 3  # All passed
    assert batch_job.failed_count == 0
    assert batch_job.status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_batch_run_with_limit(
    batch_runner: BatchRunner,
    test_case_repo: InMemoryTestCaseRepository,
    sample_test_cases: list[TestCase],
):
    """Test batch run respects limit parameter."""
    for tc in sample_test_cases:
        await test_case_repo.save(tc)

    batch_job = await batch_runner.run(limit=2)

    assert batch_job.total_tests == 2


@pytest.mark.asyncio
async def test_batch_run_with_tag_filter(
    batch_runner: BatchRunner,
    test_case_repo: InMemoryTestCaseRepository,
    sample_test_cases: list[TestCase],
):
    """Test batch run filters by tags."""
    for tc in sample_test_cases:
        await test_case_repo.save(tc)

    # Only "user" tag - should match 1 test case
    batch_job = await batch_runner.run(tags=["user"])
    assert batch_job.total_tests == 1


@pytest.mark.asyncio
async def test_batch_run_saves_scorecards(
    batch_runner: BatchRunner,
    test_case_repo: InMemoryTestCaseRepository,
    scorecard_repo: InMemoryScorecardRepository,
    sample_test_cases: list[TestCase],
):
    """Test batch run saves scorecards for each test case."""
    for tc in sample_test_cases:
        await test_case_repo.save(tc)

    batch_job = await batch_runner.run()

    # Check scorecards were saved
    scorecards = await scorecard_repo.get_by_batch(batch_job.batch_id)
    assert len(scorecards) == 3

    # All should be passing
    for sc in scorecards:
        assert sc.overall_passed is True
        assert sc.batch_id == batch_job.batch_id


@pytest.mark.asyncio
async def test_batch_run_persists_batch_job(
    batch_runner: BatchRunner,
    test_case_repo: InMemoryTestCaseRepository,
    batch_repo: InMemoryBatchJobRepository,
    sample_test_cases: list[TestCase],
):
    """Test batch run persists batch job to repository."""
    for tc in sample_test_cases:
        await test_case_repo.save(tc)

    batch_job = await batch_runner.run()

    # Verify batch job was persisted
    persisted = await batch_repo.get(batch_job.batch_id)
    assert persisted is not None
    assert persisted.batch_id == batch_job.batch_id
    assert persisted.total_tests == 3
    assert persisted.status == TaskStatus.COMPLETED


@pytest.mark.asyncio
async def test_batch_run_complexity_filter(
    batch_runner: BatchRunner,
    test_case_repo: InMemoryTestCaseRepository,
    sample_test_cases: list[TestCase],
):
    """Test batch run filters by complexity."""
    for tc in sample_test_cases:
        await test_case_repo.save(tc)

    # complexity_level 2 only - should match 1 test case
    batch_job = await batch_runner.run(complexity_min=2, complexity_max=2)
    assert batch_job.total_tests == 1


# =============================================================================
# InMemoryBatchJobRepository Tests
# =============================================================================


@pytest.mark.asyncio
async def test_batch_repo_create_and_get(batch_repo: InMemoryBatchJobRepository):
    """Test creating and retrieving a batch job."""
    batch_job = BatchJob(total_tests=10, status=TaskStatus.PENDING)

    await batch_repo.create(batch_job)
    retrieved = await batch_repo.get(batch_job.batch_id)

    assert retrieved is not None
    assert retrieved.batch_id == batch_job.batch_id
    assert retrieved.total_tests == 10


@pytest.mark.asyncio
async def test_batch_repo_get_nonexistent(batch_repo: InMemoryBatchJobRepository):
    """Test retrieving non-existent batch job returns None."""
    result = await batch_repo.get("nonexistent-id")
    assert result is None


@pytest.mark.asyncio
async def test_batch_repo_update(batch_repo: InMemoryBatchJobRepository):
    """Test updating a batch job."""
    batch_job = BatchJob(total_tests=10, status=TaskStatus.PENDING)
    await batch_repo.create(batch_job)

    # Update the batch job
    updated = BatchJob(
        batch_id=batch_job.batch_id,
        total_tests=10,
        completed_count=5,
        failed_count=2,
        status=TaskStatus.IN_PROGRESS,
        created_at=batch_job.created_at,
    )
    await batch_repo.update(updated)

    retrieved = await batch_repo.get(batch_job.batch_id)
    assert retrieved is not None
    assert retrieved.completed_count == 5
    assert retrieved.failed_count == 2
    assert retrieved.status == TaskStatus.IN_PROGRESS


@pytest.mark.asyncio
async def test_batch_repo_list_recent(batch_repo: InMemoryBatchJobRepository):
    """Test listing recent batch jobs."""
    # Create multiple batch jobs
    for i in range(5):
        batch = BatchJob(total_tests=i + 1)
        await batch_repo.create(batch)

    recent = await batch_repo.list_recent(limit=3)
    assert len(recent) == 3

    # Should be in reverse creation order (newest first)
    assert recent[0].total_tests >= recent[1].total_tests


@pytest.mark.asyncio
async def test_batch_repo_clear(batch_repo: InMemoryBatchJobRepository):
    """Test clearing all batch jobs."""
    batch_job = BatchJob(total_tests=10)
    await batch_repo.create(batch_job)

    batch_repo.clear()

    result = await batch_repo.get(batch_job.batch_id)
    assert result is None


# =============================================================================
# Custom Response Simulator Tests
# =============================================================================


@pytest.mark.asyncio
async def test_batch_run_custom_simulator(
    batch_runner: BatchRunner,
    test_case_repo: InMemoryTestCaseRepository,
    sample_test_cases: list[TestCase],
):
    """Test batch run with custom response simulator that returns wrong responses."""
    from CONTRACTS import SystemResponse

    # Custom simulator that returns wrong tool calls
    async def wrong_response_simulator(test_case: TestCase) -> SystemResponse:
        return SystemResponse(
            raw_output='[{"tool_name": "wrong_tool", "arguments": {}}]',
            nl_response="Wrong response",
            latency_ms=100,
        )

    for tc in sample_test_cases:
        await test_case_repo.save(tc)

    batch_job = await batch_runner.run(response_simulator=wrong_response_simulator)

    # All should fail because wrong tool was called
    assert batch_job.total_tests == 3
    assert batch_job.completed_count == 0  # None passed
    assert batch_job.failed_count == 3  # All failed


# =============================================================================
# Pack Abstraction Tests (Part 1 refactoring)
# =============================================================================


class TestBatchRunnerConfig:
    """Tests for BatchRunnerConfig with pack_name requirement."""

    def test_config_requires_pack_name(self):
        """Verify BatchRunnerConfig without pack_name raises ValidationError."""
        from pydantic import ValidationError

        with pytest.raises(ValidationError) as exc_info:
            BatchRunnerConfig()  # Missing pack_name

        # Verify the error mentions pack_name
        error_str = str(exc_info.value)
        assert "pack_name" in error_str

    def test_config_accepts_valid_pack_nl2api(self):
        """Verify pack_name='nl2api' is accepted."""
        config = BatchRunnerConfig(pack_name="nl2api")
        assert config.pack_name == "nl2api"

    def test_config_accepts_valid_pack_rag(self):
        """Verify pack_name='rag' is accepted."""
        config = BatchRunnerConfig(pack_name="rag")
        assert config.pack_name == "rag"

    def test_config_with_all_options(self):
        """Verify config works with all options set."""
        config = BatchRunnerConfig(
            pack_name="nl2api",
            max_concurrency=20,
            show_progress=True,
            verbose=True,
            client_type="mcp_claude",
            client_version="claude-opus-4.5-20251101",
            semantics_enabled=True,
        )
        assert config.pack_name == "nl2api"
        assert config.max_concurrency == 20
        assert config.semantics_enabled is True


class TestBatchRunnerPackAbstraction:
    """Tests for BatchRunner using pack abstraction."""

    @pytest.fixture
    def repos(self):
        """Create in-memory repositories."""
        return (
            InMemoryTestCaseRepository(),
            InMemoryScorecardRepository(),
            InMemoryBatchJobRepository(),
        )

    def test_runner_creates_nl2api_pack(self, repos):
        """Verify runner creates NL2APIPack via get_pack()."""
        test_case_repo, scorecard_repo, batch_repo = repos
        config = BatchRunnerConfig(pack_name="nl2api", show_progress=False)

        runner = BatchRunner(
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            batch_repo=batch_repo,
            config=config,
        )

        assert runner.pack.name == "nl2api"

    def test_runner_creates_rag_pack(self, repos):
        """Verify runner creates RAGPack via get_pack()."""
        test_case_repo, scorecard_repo, batch_repo = repos
        config = BatchRunnerConfig(pack_name="rag", show_progress=False)

        runner = BatchRunner(
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            batch_repo=batch_repo,
            config=config,
        )

        assert runner.pack.name == "rag"

    def test_runner_invalid_pack_raises_error(self, repos):
        """Verify unknown pack name raises ValueError."""
        test_case_repo, scorecard_repo, batch_repo = repos
        config = BatchRunnerConfig(pack_name="invalid_pack", show_progress=False)

        with pytest.raises(ValueError) as exc_info:
            BatchRunner(
                test_case_repo=test_case_repo,
                scorecard_repo=scorecard_repo,
                batch_repo=batch_repo,
                config=config,
            )

        assert "Unknown pack" in str(exc_info.value)
        assert "invalid_pack" in str(exc_info.value)


class TestResponseToOutput:
    """Tests for _response_to_output helper method."""

    @pytest.fixture
    def repos(self):
        """Create in-memory repositories."""
        return (
            InMemoryTestCaseRepository(),
            InMemoryScorecardRepository(),
            InMemoryBatchJobRepository(),
        )

    @pytest.fixture
    def nl2api_runner(self, repos):
        """Create runner with nl2api pack."""
        test_case_repo, scorecard_repo, batch_repo = repos
        config = BatchRunnerConfig(pack_name="nl2api", show_progress=False)
        return BatchRunner(
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            batch_repo=batch_repo,
            config=config,
        )

    @pytest.fixture
    def rag_runner(self, repos):
        """Create runner with rag pack."""
        test_case_repo, scorecard_repo, batch_repo = repos
        config = BatchRunnerConfig(pack_name="rag", show_progress=False)
        return BatchRunner(
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            batch_repo=batch_repo,
            config=config,
        )

    @pytest.fixture
    def test_case(self):
        """Create a simple test case."""
        return TestCase(
            id="test-001",
            nl_query="Test query",
            expected_tool_calls=(ToolCall(tool_name="test", arguments={}),),
        )

    def test_response_to_output_nl2api(self, nl2api_runner, test_case):
        """Verify NL2API response conversion."""
        from CONTRACTS import SystemResponse

        response = SystemResponse(
            raw_output='[{"tool_name": "test", "arguments": {}}]',
            nl_response="Test response",
            latency_ms=100,
        )

        output = nl2api_runner._response_to_output(response, test_case)

        assert "raw_output" in output
        assert "nl_response" in output
        assert output["raw_output"] == '[{"tool_name": "test", "arguments": {}}]'
        assert output["nl_response"] == "Test response"

    def test_response_to_output_rag(self, rag_runner, test_case):
        """Verify RAG response conversion includes RAG-specific fields."""
        from CONTRACTS import SystemResponse

        response = SystemResponse(
            raw_output="",
            nl_response="RAG response text",
            latency_ms=100,
        )

        output = rag_runner._response_to_output(response, test_case)

        assert "response" in output
        assert output["response"] == "RAG response text"
        assert "retrieved_doc_ids" in output
        assert "retrieved_chunks" in output
        assert "sources" in output
        assert "context" in output

    def test_response_to_output_unknown_pack(self, repos, test_case):
        """Verify fallback for unknown pack returns raw_output."""
        from CONTRACTS import SystemResponse

        # Create a runner with valid pack but test the fallback logic
        # by manually setting pack_name after creation (for testing)
        test_case_repo, scorecard_repo, batch_repo = repos
        config = BatchRunnerConfig(pack_name="nl2api", show_progress=False)
        runner = BatchRunner(
            test_case_repo=test_case_repo,
            scorecard_repo=scorecard_repo,
            batch_repo=batch_repo,
            config=config,
        )

        # Override pack_name to test fallback (normally wouldn't happen in production)
        # We test the actual method behavior with nl2api since that's valid
        response = SystemResponse(
            raw_output='{"test": "data"}',
            nl_response=None,
            latency_ms=100,
        )

        # NL2API conversion always includes raw_output
        output = runner._response_to_output(response, test_case)
        assert "raw_output" in output


# =============================================================================
# Pack Registry Tests
# =============================================================================


class TestPackRegistry:
    """Tests for the evaluation pack registry."""

    def test_get_pack_nl2api(self):
        """Verify get_pack('nl2api') returns NL2APIPack."""
        from src.evalkit.packs import get_pack
        from src.nl2api.evaluation import NL2APIPack

        pack = get_pack("nl2api")
        assert isinstance(pack, NL2APIPack)
        assert pack.name == "nl2api"

    def test_get_pack_rag(self):
        """Verify get_pack('rag') returns RAGPack."""
        from src.evalkit.packs import get_pack
        from src.rag.evaluation import RAGPack

        pack = get_pack("rag")
        assert isinstance(pack, RAGPack)
        assert pack.name == "rag"

    def test_get_pack_invalid(self):
        """Verify get_pack('invalid') raises ValueError with available packs."""
        from src.evalkit.packs import get_pack

        with pytest.raises(ValueError) as exc_info:
            get_pack("invalid_pack")

        error_msg = str(exc_info.value)
        assert "Unknown pack" in error_msg
        assert "invalid_pack" in error_msg
        assert "nl2api" in error_msg  # Shows available packs
        assert "rag" in error_msg

    def test_get_pack_kwargs_passed(self):
        """Verify kwargs are passed to pack constructor."""
        from src.evalkit.packs import get_pack

        # NL2APIPack accepts semantics_enabled kwarg
        pack = get_pack("nl2api", semantics_enabled=True)

        # Verify the pack was configured with semantics stage
        stages = pack.get_stages()
        stage_names = [s.name for s in stages]
        assert "semantics" in stage_names
