"""
Unit tests for RAGIndexer.

Tests the indexing functionality for RAG documents.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.rag.retriever.indexer import (
    FieldCodeDocument,
    QueryExampleDocument,
    RAGIndexer,
    parse_datastream_reference,
    parse_estimates_reference,
)

# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def mock_pool():
    """Create a mock asyncpg pool."""
    mock_conn = MagicMock()
    mock_conn.execute = AsyncMock(return_value="INSERT 1")
    mock_conn.fetch = AsyncMock(return_value=[])
    mock_conn.fetchrow = AsyncMock(return_value=None)
    mock_conn.copy_records_to_table = AsyncMock()

    # Mock transaction context manager
    mock_transaction = MagicMock()
    mock_transaction.__aenter__ = AsyncMock(return_value=mock_transaction)
    mock_transaction.__aexit__ = AsyncMock(return_value=None)
    mock_conn.transaction = MagicMock(return_value=mock_transaction)

    mock_pool = MagicMock()
    mock_pool.acquire = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)

    return mock_pool, mock_conn


@pytest.fixture
def mock_embedder():
    """Create a mock embedder."""
    embedder = MagicMock()
    embedder.embed = AsyncMock(return_value=[0.1] * 384)
    embedder.embed_batch = AsyncMock(return_value=[[0.1] * 384, [0.2] * 384])
    embedder.dimension = 384
    return embedder


@pytest.fixture
def indexer(mock_pool):
    """Create a RAGIndexer with mocked pool."""
    pool, _ = mock_pool
    return RAGIndexer(pool=pool)


@pytest.fixture
def sample_field_code_doc():
    """Sample field code document."""
    return FieldCodeDocument(
        field_code="TR.PERatio",
        description="Price to earnings ratio",
        domain="fundamentals",
        natural_language_hints=["PE ratio", "price earnings"],
        metadata={"source": "test"},
    )


@pytest.fixture
def sample_query_example_doc():
    """Sample query example document."""
    return QueryExampleDocument(
        query="What is Apple's PE ratio?",
        api_call='get_data("AAPL.O", ["TR.PERatio"])',
        domain="fundamentals",
        complexity_level=1,
        metadata={"source": "test"},
    )


# =============================================================================
# Initialization Tests
# =============================================================================


class TestRAGIndexerInit:
    """Tests for RAGIndexer initialization."""

    def test_default_initialization(self, mock_pool):
        """Indexer initializes with default parameters."""
        pool, _ = mock_pool
        indexer = RAGIndexer(pool=pool)

        assert indexer._embedder is None
        assert indexer._use_bulk_insert is True

    def test_with_embedder(self, mock_pool, mock_embedder):
        """Indexer accepts embedder in constructor."""
        pool, _ = mock_pool
        indexer = RAGIndexer(pool=pool, embedder=mock_embedder)

        assert indexer._embedder == mock_embedder

    def test_set_embedder(self, indexer, mock_embedder):
        """set_embedder sets the embedder."""
        indexer.set_embedder(mock_embedder)
        assert indexer._embedder == mock_embedder

    def test_checkpoint_manager_available(self, indexer):
        """Checkpoint manager is available."""
        assert indexer.checkpoint_manager is not None


# =============================================================================
# Single Document Indexing Tests
# =============================================================================


class TestIndexFieldCode:
    """Tests for index_field_code() method."""

    @pytest.mark.asyncio
    async def test_index_without_embedding(self, indexer, mock_pool, sample_field_code_doc):
        """index_field_code without embedding."""
        _, conn = mock_pool

        doc_id = await indexer.index_field_code(sample_field_code_doc, generate_embedding=False)

        assert doc_id is not None
        conn.execute.assert_called_once()

        # Verify SQL contains expected values
        call_args = conn.execute.call_args
        sql = call_args[0][0]
        assert "INSERT INTO rag_documents" in sql
        assert "ON CONFLICT" in sql

    @pytest.mark.asyncio
    async def test_index_with_embedding(self, mock_pool, mock_embedder, sample_field_code_doc):
        """index_field_code with embedding generation."""
        pool, conn = mock_pool
        indexer = RAGIndexer(pool=pool, embedder=mock_embedder)

        doc_id = await indexer.index_field_code(sample_field_code_doc, generate_embedding=True)

        assert doc_id is not None
        # Embedder should be called
        mock_embedder.embed.assert_called_once()
        # SQL should include embedding
        call_args = conn.execute.call_args
        sql = call_args[0][0]
        assert "embedding" in sql

    @pytest.mark.asyncio
    async def test_index_builds_content_from_hints(self, indexer, mock_pool, sample_field_code_doc):
        """Content includes natural language hints."""
        _, conn = mock_pool

        await indexer.index_field_code(sample_field_code_doc, generate_embedding=False)

        # Check content parameter
        call_args = conn.execute.call_args[0]
        content = call_args[2]  # Third positional arg is content
        assert "PE ratio" in content
        assert "price earnings" in content


class TestIndexQueryExample:
    """Tests for index_query_example() method."""

    @pytest.mark.asyncio
    async def test_index_query_example(self, indexer, mock_pool, sample_query_example_doc):
        """index_query_example creates document."""
        _, conn = mock_pool

        doc_id = await indexer.index_query_example(
            sample_query_example_doc, generate_embedding=False
        )

        assert doc_id is not None
        conn.execute.assert_called_once()

        # Verify content format
        call_args = conn.execute.call_args[0]
        content = call_args[2]
        assert "Q:" in content
        assert "A:" in content
        assert sample_query_example_doc.query in content


# =============================================================================
# Batch Indexing Tests
# =============================================================================


class TestIndexFieldCodesBatch:
    """Tests for index_field_codes_batch() method."""

    @pytest.mark.asyncio
    async def test_batch_index_empty_list(self, indexer):
        """Batch index handles empty list."""
        result = await indexer.index_field_codes_batch([])
        assert result == []

    @pytest.mark.asyncio
    async def test_batch_index_without_embeddings(self, mock_pool, sample_field_code_doc):
        """Batch index without embeddings uses individual inserts."""
        pool, conn = mock_pool
        indexer = RAGIndexer(pool=pool, use_bulk_insert=False)

        docs = [sample_field_code_doc, sample_field_code_doc]
        result = await indexer.index_field_codes_batch(
            docs, generate_embeddings=False, batch_size=10
        )

        assert len(result) == 2

    @pytest.mark.asyncio
    async def test_batch_index_with_bulk_insert(
        self, mock_pool, mock_embedder, sample_field_code_doc
    ):
        """Batch index uses COPY protocol for bulk insert."""
        pool, conn = mock_pool
        indexer = RAGIndexer(pool=pool, embedder=mock_embedder, use_bulk_insert=True)

        docs = [sample_field_code_doc, sample_field_code_doc]
        result = await indexer.index_field_codes_batch(
            docs, generate_embeddings=True, batch_size=10
        )

        assert len(result) == 2
        # Should use copy_records_to_table
        conn.copy_records_to_table.assert_called()

    @pytest.mark.asyncio
    async def test_batch_index_calls_progress_callback(self, mock_pool, sample_field_code_doc):
        """Batch index calls progress callback."""
        pool, conn = mock_pool
        indexer = RAGIndexer(pool=pool, use_bulk_insert=False)

        docs = [sample_field_code_doc, sample_field_code_doc]
        progress_calls = []

        def callback(processed: int, total: int):
            progress_calls.append((processed, total))

        await indexer.index_field_codes_batch(
            docs,
            generate_embeddings=False,
            batch_size=10,
            progress_callback=callback,
        )

        assert len(progress_calls) == 1
        assert progress_calls[0] == (2, 2)


# =============================================================================
# Clear Domain Tests
# =============================================================================


class TestClearDomain:
    """Tests for clear_domain() method."""

    @pytest.mark.asyncio
    async def test_clear_domain(self, indexer, mock_pool):
        """clear_domain deletes documents for domain."""
        _, conn = mock_pool
        conn.execute = AsyncMock(return_value="DELETE 5")

        count = await indexer.clear_domain("fundamentals")

        assert count == 5
        conn.execute.assert_called_once()
        call_args = conn.execute.call_args[0]
        assert "DELETE" in call_args[0]
        assert call_args[1] == "fundamentals"


# =============================================================================
# Stats Tests
# =============================================================================


class TestGetStats:
    """Tests for get_stats() method."""

    @pytest.mark.asyncio
    async def test_get_stats(self, indexer, mock_pool):
        """get_stats returns document counts."""
        _, conn = mock_pool
        conn.fetch = AsyncMock(
            return_value=[
                {
                    "domain": "fundamentals",
                    "document_type": "field_code",
                    "count": 100,
                    "with_embedding": 80,
                },
                {
                    "domain": "estimates",
                    "document_type": "field_code",
                    "count": 50,
                    "with_embedding": 50,
                },
            ]
        )

        stats = await indexer.get_stats()

        assert "fundamentals" in stats
        assert "estimates" in stats
        assert stats["fundamentals"]["field_code"]["count"] == 100
        assert stats["estimates"]["field_code"]["with_embedding"] == 50


# =============================================================================
# Reference Parsing Tests
# =============================================================================


class TestParseEstimatesReference:
    """Tests for parse_estimates_reference() function."""

    def test_parse_table_rows(self):
        """Parses markdown table rows."""
        content = """
# Estimates Reference

| Natural Language | TR Code | Description |
|------------------|---------|-------------|
| EPS mean | `TR.EPSMean` | Consensus EPS estimate |
| revenue estimate | `TR.RevenueMean` | Revenue consensus |
        """

        docs = parse_estimates_reference(content)

        assert len(docs) == 2
        assert docs[0].field_code == "TR.EPSMean"
        assert docs[0].domain == "estimates"
        assert "EPS mean" in docs[0].natural_language_hints

    def test_skips_header_rows(self):
        """Skips header and separator rows."""
        content = """
| Natural Language | TR Code | Description |
|------------------|---------|-------------|
| --- | `---` | --- |
| EPS mean | `TR.EPSMean` | Consensus EPS |
        """

        docs = parse_estimates_reference(content)

        # Should only have one valid row
        assert len(docs) == 1
        assert docs[0].field_code == "TR.EPSMean"


class TestParseDatastreamReference:
    """Tests for parse_datastream_reference() function."""

    def test_parse_table_rows(self):
        """Parses markdown table rows."""
        content = """
# Datastream Reference

| Natural Language | Field Code | Description |
|------------------|------------|-------------|
| price | `P` | Current price |
| volume | `VO` | Trading volume |
        """

        docs = parse_datastream_reference(content)

        assert len(docs) == 2
        assert docs[0].field_code == "P"
        assert docs[0].domain == "datastream"

    def test_skips_interface_rows(self):
        """Skips interface/parameter rows."""
        content = """
| Natural Language | Field Code | Description |
|------------------|------------|-------------|
| Interface | `VARIES BY` | Interface type |
| price | `P` | Current price |
        """

        docs = parse_datastream_reference(content)

        # Should only have one valid row
        assert len(docs) == 1
        assert docs[0].field_code == "P"


# =============================================================================
# Checkpoint Integration Tests
# =============================================================================


class TestCheckpointIntegration:
    """Tests for checkpoint/resume functionality."""

    @pytest.mark.asyncio
    async def test_auto_creates_checkpoint_for_large_jobs(self, mock_pool, sample_field_code_doc):
        """Large jobs auto-create checkpoints."""
        pool, conn = mock_pool

        # Mock checkpoint manager
        with patch("src.rag.retriever.indexer.CheckpointManager") as MockCheckpointManager:
            mock_checkpoint_manager = MagicMock()
            mock_checkpoint_manager.get_checkpoint = AsyncMock(return_value=None)
            mock_checkpoint_manager.create_checkpoint = AsyncMock(
                return_value=MagicMock(job_id="test-job", is_resumable=False)
            )
            mock_checkpoint_manager.update_progress = AsyncMock()
            mock_checkpoint_manager.mark_completed = AsyncMock()
            MockCheckpointManager.return_value = mock_checkpoint_manager

            indexer = RAGIndexer(pool=pool, use_bulk_insert=False)

            # Create enough docs to trigger checkpoint (> batch_size * 2)
            docs = [sample_field_code_doc] * 50
            await indexer.index_field_codes_batch(docs, generate_embeddings=False, batch_size=10)

            # Should have created a checkpoint for large job
            # (50 docs > 10 * 2 = 20)
            mock_checkpoint_manager.create_checkpoint.assert_called_once()


# =============================================================================
# Bulk Insert Tests
# =============================================================================


class TestBulkInsert:
    """Tests for bulk insert functionality."""

    @pytest.mark.asyncio
    async def test_bulk_insert_creates_staging_table(
        self, mock_pool, mock_embedder, sample_field_code_doc
    ):
        """Bulk insert creates and uses staging table."""
        pool, conn = mock_pool
        indexer = RAGIndexer(pool=pool, embedder=mock_embedder, use_bulk_insert=True)

        docs = [sample_field_code_doc, sample_field_code_doc]
        await indexer.index_field_codes_batch(docs, generate_embeddings=True, batch_size=10)

        # Should create temp table
        execute_calls = conn.execute.call_args_list
        create_temp_sql = execute_calls[0][0][0]
        assert "CREATE TEMP TABLE" in create_temp_sql
        assert "staging_rag_documents" in create_temp_sql

    @pytest.mark.asyncio
    async def test_bulk_insert_uses_copy_protocol(
        self, mock_pool, mock_embedder, sample_field_code_doc
    ):
        """Bulk insert uses COPY protocol."""
        pool, conn = mock_pool
        indexer = RAGIndexer(pool=pool, embedder=mock_embedder, use_bulk_insert=True)

        docs = [sample_field_code_doc, sample_field_code_doc]
        await indexer.index_field_codes_batch(docs, generate_embeddings=True, batch_size=10)

        # Should use copy_records_to_table
        conn.copy_records_to_table.assert_called_once()

        # Verify table name and columns
        call_kwargs = conn.copy_records_to_table.call_args
        assert call_kwargs[0][0] == "staging_rag_documents"
        assert "records" in call_kwargs[1]
        assert len(call_kwargs[1]["records"]) == 2


# =============================================================================
# Error Handling Tests
# =============================================================================


class TestErrorHandling:
    """Tests for error handling in indexer."""

    @pytest.mark.asyncio
    async def test_batch_index_marks_checkpoint_failed_on_error(
        self, mock_pool, sample_field_code_doc
    ):
        """Batch index marks checkpoint as failed on error."""
        pool, conn = mock_pool

        # Make execute raise an error
        conn.execute = AsyncMock(side_effect=Exception("Database error"))

        with patch("src.rag.retriever.indexer.CheckpointManager") as MockCheckpointManager:
            mock_checkpoint_manager = MagicMock()
            mock_checkpoint_manager.get_checkpoint = AsyncMock(return_value=None)
            mock_checkpoint_manager.create_checkpoint = AsyncMock(
                return_value=MagicMock(job_id="test-job", is_resumable=False)
            )
            mock_checkpoint_manager.mark_failed = AsyncMock()
            MockCheckpointManager.return_value = mock_checkpoint_manager

            indexer = RAGIndexer(pool=pool, use_bulk_insert=False)

            # Create enough docs to trigger checkpoint
            docs = [sample_field_code_doc] * 50

            with pytest.raises(Exception, match="Database error"):
                await indexer.index_field_codes_batch(
                    docs, generate_embeddings=False, batch_size=10
                )

            # Should mark checkpoint as failed
            mock_checkpoint_manager.mark_failed.assert_called_once()
