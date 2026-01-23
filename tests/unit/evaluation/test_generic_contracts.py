"""
Unit tests for generic evaluation contracts.

Tests the generic fields and conversion methods on TestCase and Scorecard.
"""

from CONTRACTS import (
    EvaluationStage,
    Scorecard,
    StageResult,
    TestCase,
    ToolCall,
)

# =============================================================================
# TestCase Generic Fields Tests
# =============================================================================


class TestTestCaseGenericFields:
    """Tests for generic fields on TestCase."""

    def test_default_input_is_empty_dict(self):
        """Default input is empty dict."""
        tc = TestCase(id="test-001")
        assert tc.input == {}

    def test_default_expected_is_empty_dict(self):
        """Default expected is empty dict."""
        tc = TestCase(id="test-001")
        assert tc.expected == {}

    def test_input_accepts_arbitrary_keys(self):
        """Input dict accepts arbitrary keys."""
        tc = TestCase(
            id="test-001",
            input={
                "query": "test query",
                "context": {"key": "value"},
                "documents": [{"id": 1}, {"id": 2}],
            },
        )
        assert tc.input["query"] == "test query"
        assert tc.input["context"]["key"] == "value"
        assert len(tc.input["documents"]) == 2

    def test_expected_accepts_arbitrary_keys(self):
        """Expected dict accepts arbitrary keys."""
        tc = TestCase(
            id="test-001",
            expected={
                "answer": "expected answer",
                "citations": [1, 2, 3],
                "confidence": 0.95,
            },
        )
        assert tc.expected["answer"] == "expected answer"
        assert tc.expected["citations"] == [1, 2, 3]
        assert tc.expected["confidence"] == 0.95

    def test_generic_and_specific_coexist(self):
        """Generic and NL2API-specific fields can coexist."""
        tc = TestCase(
            id="test-001",
            nl_query="Get Apple price",
            expected_tool_calls=(ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
            input={"nl_query": "Get Apple price"},
            expected={"tool_calls": [{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}]},
        )
        assert tc.nl_query == "Get Apple price"
        assert len(tc.expected_tool_calls) == 1
        assert tc.input["nl_query"] == "Get Apple price"
        assert len(tc.expected["tool_calls"]) == 1


class TestTestCaseToGeneric:
    """Tests for TestCase.to_generic() conversion method."""

    def test_converts_nl_query_to_input(self):
        """Converts nl_query to input dict."""
        tc = TestCase(id="test-001", nl_query="Get Apple price")

        generic = tc.to_generic()

        assert generic.input["nl_query"] == "Get Apple price"

    def test_converts_tool_calls_to_expected(self):
        """Converts expected_tool_calls to expected dict."""
        tc = TestCase(
            id="test-001",
            expected_tool_calls=(ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
        )

        generic = tc.to_generic()

        assert len(generic.expected["tool_calls"]) == 1
        assert generic.expected["tool_calls"][0]["tool_name"] == "get_price"

    def test_converts_nl_response_to_expected(self):
        """Converts expected_nl_response to expected dict."""
        tc = TestCase(
            id="test-001",
            expected_nl_response="Apple's price is $150",
        )

        generic = tc.to_generic()

        assert generic.expected["nl_response"] == "Apple's price is $150"

    def test_preserves_existing_generic_fields(self):
        """Existing generic fields are preserved."""
        tc = TestCase(
            id="test-001",
            input={"extra_key": "value"},
            expected={"extra_key": "value"},
        )

        generic = tc.to_generic()

        assert generic.input["extra_key"] == "value"
        assert generic.expected["extra_key"] == "value"

    def test_preserves_metadata(self):
        """Metadata is preserved."""
        from CONTRACTS import TestCaseMetadata

        metadata = TestCaseMetadata(
            api_version="1.0.0",
            complexity_level=1,
            tags=("tag1", "tag2"),
        )
        tc = TestCase(
            id="test-001",
            metadata=metadata,
        )

        generic = tc.to_generic()

        assert generic.metadata is not None
        assert generic.metadata.tags == ("tag1", "tag2")


class TestTestCaseFromGeneric:
    """Tests for TestCase.from_generic() factory method."""

    def test_creates_with_minimal_args(self):
        """Creates test case with minimal arguments."""
        tc = TestCase.from_generic(
            id="test-001",
            input={"query": "test"},
            expected={"answer": "result"},
        )

        assert tc.id == "test-001"
        assert tc.input["query"] == "test"
        assert tc.expected["answer"] == "result"

    def test_accepts_metadata(self):
        """Accepts TestCaseMetadata."""
        from CONTRACTS import TestCaseMetadata

        metadata = TestCaseMetadata(
            api_version="1.0.0",
            complexity_level=2,
            tags=("tag1",),
            source="manual",
        )
        tc = TestCase.from_generic(
            id="test-001",
            input={"query": "test"},
            expected={"answer": "result"},
            metadata=metadata,
        )

        assert tc.metadata is not None
        assert tc.metadata.tags == ("tag1",)
        assert tc.metadata.source == "manual"
        assert tc.metadata.complexity_level == 2

    def test_auto_populates_nl2api_fields_from_generic(self):
        """Auto-populates NL2API-specific fields from generic input/expected."""
        tc = TestCase.from_generic(
            id="test-001",
            input={"nl_query": "Get Apple price"},
            expected={"tool_calls": [{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}]},
        )

        # from_generic intentionally extracts NL2API fields for backwards compatibility
        assert tc.nl_query == "Get Apple price"
        assert len(tc.expected_tool_calls) == 1
        assert tc.expected_tool_calls[0].tool_name == "get_price"
        assert tc.expected_tool_calls[0].arguments["ticker"] == "AAPL"


# =============================================================================
# Scorecard Generic Fields Tests
# =============================================================================


class TestScorecardGenericFields:
    """Tests for generic fields on Scorecard."""

    def test_default_pack_name(self):
        """Default pack_name is 'nl2api'."""
        sc = Scorecard(test_case_id="test-001")
        assert sc.pack_name == "nl2api"

    def test_default_stage_results_is_empty(self):
        """Default stage_results is empty dict."""
        sc = Scorecard(test_case_id="test-001")
        assert sc.stage_results == {}

    def test_default_stage_weights_is_empty(self):
        """Default stage_weights is empty dict."""
        sc = Scorecard(test_case_id="test-001")
        assert sc.stage_weights == {}

    def test_default_generated_output_is_empty(self):
        """Default generated_output is empty dict."""
        sc = Scorecard(test_case_id="test-001")
        assert sc.generated_output == {}

    def test_accepts_arbitrary_stage_results(self):
        """Accepts arbitrary stage names in stage_results."""
        sc = Scorecard(
            test_case_id="test-001",
            pack_name="rag",
            stage_results={
                "retrieval": StageResult(stage_name="retrieval", passed=True, score=0.9),
                "faithfulness": StageResult(stage_name="faithfulness", passed=True, score=0.8),
                "relevance": StageResult(stage_name="relevance", passed=True, score=0.85),
            },
        )

        assert "retrieval" in sc.stage_results
        assert "faithfulness" in sc.stage_results
        assert "relevance" in sc.stage_results
        assert sc.stage_results["retrieval"].score == 0.9

    def test_accepts_arbitrary_stage_weights(self):
        """Accepts arbitrary stage names in stage_weights."""
        sc = Scorecard(
            test_case_id="test-001",
            stage_weights={
                "retrieval": 0.3,
                "faithfulness": 0.4,
                "relevance": 0.3,
            },
        )

        assert sc.stage_weights["retrieval"] == 0.3
        assert sum(sc.stage_weights.values()) == 1.0

    def test_accepts_arbitrary_generated_output(self):
        """Accepts arbitrary keys in generated_output."""
        sc = Scorecard(
            test_case_id="test-001",
            generated_output={
                "retrieved_docs": [{"id": 1}, {"id": 2}],
                "answer": "The answer is...",
                "citations": [1],
            },
        )

        assert len(sc.generated_output["retrieved_docs"]) == 2
        assert sc.generated_output["answer"] == "The answer is..."


class TestScorecardGetAllStageResults:
    """Tests for Scorecard.get_all_stage_results() method."""

    def test_returns_generic_stage_results(self):
        """Returns stage_results when no NL2API fields."""
        sc = Scorecard(
            test_case_id="test-001",
            stage_results={
                "retrieval": StageResult(stage_name="retrieval", passed=True, score=0.9),
            },
        )

        all_results = sc.get_all_stage_results()

        assert "retrieval" in all_results
        assert all_results["retrieval"].score == 0.9

    def test_includes_nl2api_specific_results(self):
        """Includes NL2API-specific result fields."""
        sc = Scorecard(
            test_case_id="test-001",
            syntax_result=StageResult(stage_name="syntax", passed=True, score=1.0),
            logic_result=StageResult(stage_name="logic", passed=True, score=0.9),
        )

        all_results = sc.get_all_stage_results()

        assert "syntax" in all_results
        assert "logic" in all_results
        assert all_results["syntax"].score == 1.0
        assert all_results["logic"].score == 0.9

    def test_merges_generic_and_specific(self):
        """Merges generic and NL2API-specific results."""
        sc = Scorecard(
            test_case_id="test-001",
            stage_results={
                "custom_stage": StageResult(stage_name="custom_stage", passed=True, score=0.8),
            },
            syntax_result=StageResult(stage_name="syntax", passed=True, score=1.0),
            logic_result=StageResult(stage_name="logic", passed=True, score=0.9),
        )

        all_results = sc.get_all_stage_results()

        assert "custom_stage" in all_results
        assert "syntax" in all_results
        assert "logic" in all_results

    def test_generic_overrides_specific(self):
        """Generic stage_results override NL2API-specific when same key."""
        sc = Scorecard(
            test_case_id="test-001",
            stage_results={
                "syntax": StageResult(stage_name="syntax", passed=True, score=0.5),  # Override
            },
            syntax_result=StageResult(stage_name="syntax", passed=True, score=1.0),  # Original
        )

        all_results = sc.get_all_stage_results()

        # Generic should take precedence
        assert all_results["syntax"].score == 0.5

    def test_excludes_none_results(self):
        """Excludes None results from NL2API fields."""
        sc = Scorecard(
            test_case_id="test-001",
            syntax_result=StageResult(stage_name="syntax", passed=True, score=1.0),
            logic_result=None,
            execution_result=None,
            semantics_result=None,
        )

        all_results = sc.get_all_stage_results()

        assert "syntax" in all_results
        assert "logic" not in all_results
        assert "execution" not in all_results
        assert "semantics" not in all_results


# =============================================================================
# StageResult Generic Fields Tests
# =============================================================================


class TestStageResultGenericFields:
    """Tests for generic fields on StageResult."""

    def test_stage_name_default_empty(self):
        """Default stage_name is empty string."""
        sr = StageResult(passed=True, score=1.0)
        assert sr.stage_name == ""

    def test_stage_name_can_be_set(self):
        """stage_name can be set to any string."""
        sr = StageResult(stage_name="retrieval", passed=True, score=1.0)
        assert sr.stage_name == "retrieval"

    def test_stage_enum_deprecated_but_works(self):
        """Deprecated stage enum still works."""
        sr = StageResult(
            stage_name="syntax",
            stage=EvaluationStage.SYNTAX,
            passed=True,
            score=1.0,
        )
        assert sr.stage == EvaluationStage.SYNTAX

    def test_metrics_default_empty(self):
        """Default metrics is empty dict."""
        sr = StageResult(passed=True, score=1.0)
        assert sr.metrics == {}

    def test_metrics_accepts_arbitrary_values(self):
        """Metrics accepts arbitrary key/value pairs."""
        sr = StageResult(
            stage_name="retrieval",
            passed=True,
            score=0.85,
            metrics={
                "recall@5": 0.8,
                "precision@5": 0.6,
                "mrr": 0.75,
                "retrieved_count": 5,
            },
        )

        assert sr.metrics["recall@5"] == 0.8
        assert sr.metrics["precision@5"] == 0.6
        assert sr.metrics["mrr"] == 0.75
        assert sr.metrics["retrieved_count"] == 5


# =============================================================================
# Integration Tests
# =============================================================================


class TestGenericContractsIntegration:
    """Integration tests for generic contracts working together."""

    def test_full_generic_evaluation_flow(self):
        """Test complete generic evaluation flow."""
        # Create generic test case (no category - that's fixture-level, not TestCase-level)
        test_case = TestCase.from_generic(
            id="rag-test-001",
            input={
                "query": "What is the capital of France?",
                "context_window": 5,
            },
            expected={
                "answer": "Paris",
                "relevant_doc_ids": ["doc1", "doc2"],
            },
        )

        # Create generic scorecard with results
        scorecard = Scorecard(
            test_case_id=test_case.id,
            pack_name="rag",
            stage_results={
                "retrieval": StageResult(
                    stage_name="retrieval",
                    passed=True,
                    score=0.8,
                    reason="4 of 5 relevant docs retrieved",
                    metrics={"recall@5": 0.8, "mrr": 0.5},
                ),
                "faithfulness": StageResult(
                    stage_name="faithfulness",
                    passed=True,
                    score=0.9,
                    reason="Answer grounded in context",
                    metrics={"claim_support_ratio": 0.9},
                ),
                "relevance": StageResult(
                    stage_name="relevance",
                    passed=True,
                    score=0.95,
                    reason="Answer directly addresses query",
                ),
            },
            stage_weights={
                "retrieval": 0.3,
                "faithfulness": 0.4,
                "relevance": 0.3,
            },
            generated_output={
                "retrieved_docs": ["doc1", "doc2", "doc3", "doc5", "doc7"],
                "answer": "The capital of France is Paris.",
            },
        )

        # Verify structure (category is fixture-level, not on TestCase)
        assert test_case.input["query"] == "What is the capital of France?"
        assert scorecard.pack_name == "rag"
        assert len(scorecard.get_all_stage_results()) == 3

        # Verify metrics are accessible
        retrieval_result = scorecard.stage_results["retrieval"]
        assert retrieval_result.metrics["recall@5"] == 0.8

    def test_nl2api_backwards_compatibility(self):
        """Test that NL2API format still works."""
        # Create NL2API-style test case
        test_case = TestCase(
            id="nl2api-test-001",
            nl_query="Get Apple price",
            expected_tool_calls=(ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
            expected_nl_response="Apple's price is $150",
            category="lookups",
        )

        # Create NL2API-style scorecard
        scorecard = Scorecard(
            test_case_id=test_case.id,
            pack_name="nl2api",
            syntax_result=StageResult(
                stage_name="syntax",
                stage=EvaluationStage.SYNTAX,
                passed=True,
                score=1.0,
            ),
            logic_result=StageResult(
                stage_name="logic",
                stage=EvaluationStage.LOGIC,
                passed=True,
                score=1.0,
            ),
            generated_tool_calls=(ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
            generated_nl_response="Apple's current price is $150.25",
        )

        # Verify NL2API fields work
        assert test_case.nl_query == "Get Apple price"
        assert scorecard.syntax_result.passed is True
        assert scorecard.logic_result.passed is True

        # Verify get_all_stage_results includes NL2API fields
        all_results = scorecard.get_all_stage_results()
        assert "syntax" in all_results
        assert "logic" in all_results

        # Verify to_generic preserves data
        generic = test_case.to_generic()
        assert generic.input["nl_query"] == "Get Apple price"
        assert len(generic.expected["tool_calls"]) == 1
