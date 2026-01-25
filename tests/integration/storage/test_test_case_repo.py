"""
Integration tests for TestCaseRepository.

Tests CRUD operations, list, search, and count against real PostgreSQL.
"""

import uuid

import pytest

from CONTRACTS import (
    TestCase,
    TestCaseMetadata,
    TestCaseStatus,
    ToolCall,
)
from src.evalkit.common.storage import close_repositories, create_repositories
from src.evalkit.common.storage.config import StorageConfig


class TestTestCaseRepository:
    """Integration tests for PostgresTestCaseRepository."""

    @pytest.mark.asyncio
    async def test_test_case_repository_full_lifecycle(self):
        """
        Comprehensive test of test case repository operations.

        Tests:
        1. Save and retrieve a test case
        2. Get non-existent returns None
        3. Get with invalid UUID returns None
        4. Get many test cases
        5. Get many with invalid IDs
        6. Get many with empty list
        7. Update existing test case
        8. Delete a test case
        9. Delete non-existent returns False
        10. List with filters
        11. List with pagination
        12. Count with filters
        13. Full-text search
        14. Test case with expected_response
        15. Status lifecycle transitions
        """
        # Get repository
        config = StorageConfig(backend="postgres")
        repos = await create_repositories(config)
        test_case_repo, _, _ = repos

        # Track test cases for cleanup
        test_cases_to_cleanup = []

        # =================================================================
        # Test 1: Save and retrieve a test case
        # =================================================================
        test_case_id_1 = str(uuid.uuid4())
        metadata_1 = TestCaseMetadata(
            api_version="1.0.0",
            complexity_level=3,
            tags=("integration", "test"),
            author="test-suite",
            source="integration-test",
        )
        test_case_1 = TestCase(
            id=test_case_id_1,
            nl_query="What is Apple's stock price?",
            expected_tool_calls=(
                ToolCall(tool_name="get_data", arguments={"ticker": "AAPL"}),
            ),
            expected_nl_response="Apple's stock price is $150.",
            metadata=metadata_1,
        )

        await test_case_repo.save(test_case_1)
        test_cases_to_cleanup.append(test_case_id_1)

        retrieved_1 = await test_case_repo.get(test_case_id_1)
        assert retrieved_1 is not None
        assert retrieved_1.id == test_case_id_1
        assert retrieved_1.nl_query == "What is Apple's stock price?"
        assert len(retrieved_1.expected_tool_calls) == 1
        assert retrieved_1.expected_tool_calls[0].tool_name == "get_data"
        assert retrieved_1.expected_tool_calls[0].arguments["ticker"] == "AAPL"
        assert retrieved_1.expected_nl_response == "Apple's stock price is $150."
        assert retrieved_1.metadata.complexity_level == 3
        assert "integration" in retrieved_1.metadata.tags

        # =================================================================
        # Test 2: Get non-existent returns None
        # =================================================================
        nonexistent_id = str(uuid.uuid4())
        result = await test_case_repo.get(nonexistent_id)
        assert result is None

        # =================================================================
        # Test 3: Get with invalid UUID returns None
        # =================================================================
        result = await test_case_repo.get("not-a-valid-uuid")
        assert result is None

        # =================================================================
        # Test 4: Get many test cases
        # =================================================================
        batch_tcs = []
        for i in range(3):
            tc = TestCase(
                id=str(uuid.uuid4()),
                nl_query=f"Test query {i}",
                metadata=TestCaseMetadata(
                    api_version="1.0.0",
                    complexity_level=1,
                    tags=("batch-test",),
                ),
            )
            await test_case_repo.save(tc)
            batch_tcs.append(tc)
            test_cases_to_cleanup.append(tc.id)

        ids = [tc.id for tc in batch_tcs]
        retrieved = await test_case_repo.get_many(ids)
        assert len(retrieved) == 3
        retrieved_ids = {tc.id for tc in retrieved}
        for tc in batch_tcs:
            assert tc.id in retrieved_ids

        # =================================================================
        # Test 5: Get many with invalid IDs
        # =================================================================
        valid_tc = TestCase(
            id=str(uuid.uuid4()),
            nl_query="Valid test case",
            metadata=TestCaseMetadata(api_version="1.0.0", complexity_level=1),
        )
        await test_case_repo.save(valid_tc)
        test_cases_to_cleanup.append(valid_tc.id)

        ids_with_invalid = [valid_tc.id, "invalid-uuid", "also-not-valid"]
        retrieved = await test_case_repo.get_many(ids_with_invalid)
        assert len(retrieved) == 1
        assert retrieved[0].id == valid_tc.id

        # =================================================================
        # Test 6: Get many with empty list
        # =================================================================
        result = await test_case_repo.get_many([])
        assert result == []

        # =================================================================
        # Test 7: Update existing test case
        # =================================================================
        update_tc_id = str(uuid.uuid4())
        original = TestCase(
            id=update_tc_id,
            nl_query="Original query",
            expected_nl_response="Original response",
            metadata=TestCaseMetadata(
                api_version="1.0.0",
                complexity_level=1,
                tags=("original",),
            ),
        )
        await test_case_repo.save(original)
        test_cases_to_cleanup.append(update_tc_id)

        updated = TestCase(
            id=update_tc_id,
            nl_query="Updated query",
            expected_nl_response="Updated response",
            metadata=TestCaseMetadata(
                api_version="2.0.0",
                complexity_level=3,
                tags=("updated",),
            ),
        )
        await test_case_repo.save(updated)

        retrieved = await test_case_repo.get(update_tc_id)
        assert retrieved.nl_query == "Updated query"
        assert retrieved.expected_nl_response == "Updated response"
        assert retrieved.metadata.api_version == "2.0.0"
        assert "updated" in retrieved.metadata.tags

        # =================================================================
        # Test 8: Delete a test case
        # =================================================================
        delete_tc_id = str(uuid.uuid4())
        tc = TestCase(
            id=delete_tc_id,
            nl_query="To be deleted",
            metadata=TestCaseMetadata(api_version="1.0.0", complexity_level=1),
        )
        await test_case_repo.save(tc)

        assert await test_case_repo.get(delete_tc_id) is not None
        deleted = await test_case_repo.delete(delete_tc_id)
        assert deleted is True
        assert await test_case_repo.get(delete_tc_id) is None

        # =================================================================
        # Test 9: Delete non-existent returns False
        # =================================================================
        nonexistent_delete_id = str(uuid.uuid4())
        deleted = await test_case_repo.delete(nonexistent_delete_id)
        assert deleted is False

        # =================================================================
        # Test 10: List with filters
        # =================================================================
        filter_tcs = []
        for i, (complexity, tags) in enumerate([
            (1, ("simple", "list-test-unique")),
            (3, ("medium", "list-test-unique")),
            (5, ("complex", "list-test-unique")),
        ]):
            tc = TestCase(
                id=str(uuid.uuid4()),
                nl_query=f"List test query {i}",
                metadata=TestCaseMetadata(
                    api_version="1.0.0",
                    complexity_level=complexity,
                    tags=tags,
                ),
            )
            await test_case_repo.save(tc)
            filter_tcs.append(tc)
            test_cases_to_cleanup.append(tc.id)

        # Filter by tag
        results = await test_case_repo.list(tags=["list-test-unique"], limit=100)
        list_test_ids = {tc.id for tc in filter_tcs}
        found_ids = {tc.id for tc in results if tc.id in list_test_ids}
        assert len(found_ids) == 3

        # Filter by complexity range
        results = await test_case_repo.list(
            tags=["list-test-unique"],
            complexity_min=2,
            complexity_max=4,
            limit=100,
        )
        found_ids = {tc.id for tc in results if tc.id in list_test_ids}
        assert len(found_ids) == 1  # Only the medium complexity one

        # =================================================================
        # Test 11: List with pagination
        # =================================================================
        page_tcs = []
        for i in range(5):
            tc = TestCase(
                id=str(uuid.uuid4()),
                nl_query=f"Pagination test {i}",
                metadata=TestCaseMetadata(
                    api_version="1.0.0",
                    complexity_level=1,
                    tags=("pagination-test-unique",),
                ),
            )
            await test_case_repo.save(tc)
            page_tcs.append(tc)
            test_cases_to_cleanup.append(tc.id)

        page1 = await test_case_repo.list(tags=["pagination-test-unique"], limit=2, offset=0)
        page2 = await test_case_repo.list(tags=["pagination-test-unique"], limit=2, offset=2)

        page1_ids = {tc.id for tc in page1}
        page2_ids = {tc.id for tc in page2}
        assert page1_ids.isdisjoint(page2_ids)

        # =================================================================
        # Test 12: Count with filters
        # =================================================================
        count_tcs = []
        for i, complexity in enumerate([1, 2, 3, 4, 5]):
            tc = TestCase(
                id=str(uuid.uuid4()),
                nl_query=f"Count test {i}",
                metadata=TestCaseMetadata(
                    api_version="1.0.0",
                    complexity_level=complexity,
                    tags=("count-test-unique",),
                ),
            )
            await test_case_repo.save(tc)
            count_tcs.append(tc)
            test_cases_to_cleanup.append(tc.id)

        total = await test_case_repo.count(tags=["count-test-unique"])
        assert total >= 5

        medium_high = await test_case_repo.count(
            tags=["count-test-unique"],
            complexity_min=3,
            complexity_max=5,
        )
        assert medium_high >= 3

        # =================================================================
        # Test 13: Full-text search
        # =================================================================
        search_tcs = []
        queries = [
            "What is Microsoft's stock price?",
            "Get Amazon revenue for 2023",
            "Show me Google employee count",
        ]
        for i, query in enumerate(queries):
            tc = TestCase(
                id=str(uuid.uuid4()),
                nl_query=query,
                metadata=TestCaseMetadata(
                    api_version="1.0.0",
                    complexity_level=1,
                    tags=("search-test",),
                ),
            )
            await test_case_repo.save(tc)
            search_tcs.append(tc)
            test_cases_to_cleanup.append(tc.id)

        results = await test_case_repo.search_text("stock price", limit=10)
        result_queries = [tc.nl_query for tc in results]
        assert any("Microsoft" in q for q in result_queries)

        results = await test_case_repo.search_text("revenue", limit=10)
        result_queries = [tc.nl_query for tc in results]
        assert any("Amazon" in q for q in result_queries)

        # =================================================================
        # Test 14: Test case with expected_response
        # =================================================================
        response_tc_id = str(uuid.uuid4())
        expected_response = {"AAPL.O": {"P": 150.25, "MV": 2400000000000}}

        tc = TestCase(
            id=response_tc_id,
            nl_query="Get Apple price and market value",
            expected_tool_calls=(
                ToolCall(
                    tool_name="get_data",
                    arguments={"ticker": "AAPL", "fields": ["P", "MV"]},
                ),
            ),
            expected_response=expected_response,
            expected_nl_response="Apple's price is $150.25 with market cap $2.4T",
            metadata=TestCaseMetadata(api_version="1.0.0", complexity_level=2),
        )
        await test_case_repo.save(tc)
        test_cases_to_cleanup.append(response_tc_id)

        retrieved = await test_case_repo.get(response_tc_id)
        assert retrieved.expected_response is not None
        assert retrieved.expected_response["AAPL.O"]["P"] == 150.25
        assert retrieved.expected_response["AAPL.O"]["MV"] == 2400000000000

        # =================================================================
        # Test 15: Status lifecycle transitions
        # =================================================================
        status_tc_id = str(uuid.uuid4())
        tc = TestCase(
            id=status_tc_id,
            nl_query="Status test",
            metadata=TestCaseMetadata(api_version="1.0.0", complexity_level=1),
            status=TestCaseStatus.ACTIVE,
        )
        await test_case_repo.save(tc)
        test_cases_to_cleanup.append(status_tc_id)

        stale_tc = tc.model_copy(
            update={
                "status": TestCaseStatus.STALE,
                "stale_reason": "API spec changed",
            }
        )
        await test_case_repo.save(stale_tc)

        retrieved = await test_case_repo.get(status_tc_id)
        assert retrieved.status == TestCaseStatus.STALE
        assert retrieved.stale_reason == "API spec changed"

        # =================================================================
        # Cleanup
        # =================================================================
        for tc_id in test_cases_to_cleanup:
            await test_case_repo.delete(tc_id)

        # Close repositories (also resets the connection pool singleton)
        await close_repositories()
