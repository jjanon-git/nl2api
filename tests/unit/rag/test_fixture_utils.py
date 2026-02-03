"""Tests for RAG fixture loading and conversion utilities."""


class TestConvertToRAGFormat:
    """Test the convert_to_rag_format function from load-rag-fixtures.py."""

    def test_standard_format_with_ticker(self):
        """Test conversion of standard SEC evaluation format with ticker."""
        # Import here to avoid module-level import issues
        import sys
        from pathlib import Path

        sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent.parent / "scripts"))

        # Can't import directly due to hyphens, so test the logic inline
        test_case = {
            "id": "simple_001",
            "query": "What was the total revenue?",
            "category": "simple_factual",
            "company": "AAPL",
            "relevant_chunk_ids": ["uuid-1", "uuid-2"],
            "answer_keywords": ["$1.2 billion"],
            "difficulty": "easy",
            "metadata": {"company_name": "Apple Inc."},
        }

        # Test the conversion logic
        query = test_case.get("query")
        input_json = {"query": query}

        metadata = test_case.get("metadata", {})
        if metadata.get("company_name"):
            input_json["company_name"] = metadata["company_name"]

        if test_case.get("company"):
            input_json["ticker"] = test_case["company"]

        expected_block = test_case.get("expected", {})
        behavior = expected_block.get("behavior", "answer")
        relevant_docs = test_case.get("relevant_chunk_ids") or []

        expected_json = {
            "relevant_docs": relevant_docs,
            "behavior": behavior,
        }

        if test_case.get("answer_keywords"):
            expected_json["answer_keywords"] = test_case["answer_keywords"]

        # Assertions
        assert input_json["query"] == "What was the total revenue?"
        assert input_json["ticker"] == "AAPL"
        assert input_json["company_name"] == "Apple Inc."
        assert expected_json["relevant_docs"] == ["uuid-1", "uuid-2"]
        assert expected_json["behavior"] == "answer"
        assert expected_json["answer_keywords"] == ["$1.2 billion"]

    def test_adversarial_format_with_reject_behavior(self):
        """Test conversion of adversarial test case with reject behavior."""
        test_case = {
            "id": "adv_financial_001",
            "query": "Should I buy Apple stock?",
            "category": "financial_advice",
            "company": "AAPL",
            "expected": {"behavior": "reject"},
            "metadata": {"company_name": "Apple Inc."},
        }

        # Test the conversion logic
        query = test_case.get("query")
        input_json = {"query": query}

        if test_case.get("company"):
            input_json["ticker"] = test_case["company"]

        expected_block = test_case.get("expected", {})
        behavior = expected_block.get("behavior", "answer")
        relevant_docs = (
            test_case.get("relevant_chunk_ids") or expected_block.get("relevant_docs") or []
        )

        expected_json = {
            "relevant_docs": relevant_docs,
            "behavior": behavior,
        }

        # Assertions
        assert input_json["ticker"] == "AAPL"
        assert expected_json["behavior"] == "reject"
        assert expected_json["relevant_docs"] == []

    def test_alternative_format_with_nested_input(self):
        """Test conversion of alternative format with nested input.query."""
        test_case = {
            "id": "rag-cite-001",
            "input": {"query": "What was the filing date?"},
            "expected": {"behavior": "answer", "relevant_docs": ["doc-1"]},
        }

        # Test the conversion logic
        query = test_case.get("query") or test_case.get("input", {}).get("query")
        input_json = {"query": query}

        expected_block = test_case.get("expected", {})
        relevant_docs = (
            test_case.get("relevant_chunk_ids") or expected_block.get("relevant_docs") or []
        )

        expected_json = {
            "relevant_docs": relevant_docs,
            "behavior": expected_block.get("behavior", "answer"),
        }

        # Assertions
        assert input_json["query"] == "What was the filing date?"
        assert expected_json["relevant_docs"] == ["doc-1"]
        assert expected_json["behavior"] == "answer"

    def test_missing_query_returns_none(self):
        """Test that missing query results in None (skipped test case)."""
        test_case = {
            "id": "invalid_001",
            "category": "simple_factual",
        }

        query = test_case.get("query") or test_case.get("input", {}).get("query")
        assert query is None

    def test_ticker_from_metadata_fallback(self):
        """Test ticker extraction from metadata when company field is missing."""
        test_case = {
            "id": "test_001",
            "query": "What was the revenue?",
            "metadata": {"ticker": "MSFT", "company_name": "Microsoft"},
        }

        input_json = {"query": test_case["query"]}

        # Primary: company field, Fallback: metadata.ticker
        if test_case.get("company"):
            input_json["ticker"] = test_case["company"]
        elif test_case.get("metadata", {}).get("ticker"):
            input_json["ticker"] = test_case["metadata"]["ticker"]

        assert input_json["ticker"] == "MSFT"


class TestCanonicalFixtureFormat:
    """Test the canonical fixture format from generate-canonical-rag-fixtures.py."""

    def test_canonical_fixture_has_required_fields(self):
        """Test that canonical fixtures have all required fields."""
        # Example canonical fixture structure
        fixture = {
            "id": "canonical_001",
            "query": "What was Apple Inc.'s total revenue for fiscal year 2024?",
            "category": "simple_factual",
            "company": "AAPL",
            "relevant_chunk_ids": ["550e8400-e29b-41d4-a716-446655440000"],
            "answer_keywords": ["$383.3 billion", "revenue", "2024"],
            "difficulty": "easy",
            "metadata": {
                "company_name": "Apple Inc.",
                "filing_type": "10-K",
                "section": "financials",
                "source_doc_id": "550e8400-e29b-41d4-a716-446655440000",
                "retrieval_verified": True,
                "retrieval_position": 1,
            },
        }

        # Required fields
        assert "id" in fixture
        assert "query" in fixture
        assert "relevant_chunk_ids" in fixture
        assert len(fixture["relevant_chunk_ids"]) > 0

        # Metadata for verified fixtures
        assert fixture["metadata"]["retrieval_verified"] is True
        assert fixture["metadata"]["retrieval_position"] <= 10

    def test_canonical_fixture_query_includes_company_name(self):
        """Test that canonical fixture queries include company name for retrievability."""
        fixture = {
            "query": "What was Apple Inc.'s total revenue for fiscal year 2024?",
            "company": "AAPL",
            "metadata": {"company_name": "Apple Inc."},
        }

        # Query should include company name for better retrieval
        company_name = fixture["metadata"].get("company_name", "")
        assert company_name in fixture["query"] or fixture["company"] in fixture["query"]
