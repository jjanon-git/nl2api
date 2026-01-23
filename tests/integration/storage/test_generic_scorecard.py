"""
Integration tests for generic scorecard storage.

Tests the PostgreSQL storage layer with the new generic evaluation fields.
"""

import uuid
import pytest
from CONTRACTS import (
    EvaluationStage,
    Scorecard,
    StageResult,
    ToolCall,
)
from src.common.storage import create_repositories
from src.common.storage.config import StorageConfig


class TestGenericScorecardStorage:
    """Tests for generic scorecard storage in PostgreSQL."""

    @pytest.mark.asyncio
    async def test_generic_scorecard_storage_full_lifecycle(self):
        """
        Comprehensive test of generic scorecard storage.

        Tests:
        1. Save and retrieve with generic stage_results
        2. Arbitrary stage names (RAG-style)
        3. Backwards compatibility with NL2API fields
        4. Batch queries with generic fields
        5. Metrics in stage results
        """
        # Get repository
        config = StorageConfig(backend="postgres")
        repos = await create_repositories(config)
        _, scorecard_repo, _ = repos

        # =================================================================
        # Test 1: Generic stage_results with NL2API pack
        # =================================================================
        stage_results_1 = {
            "syntax": StageResult(
                stage_name="syntax",
                stage=EvaluationStage.SYNTAX,
                passed=True,
                score=1.0,
                reason="Valid JSON",
            ),
            "logic": StageResult(
                stage_name="logic",
                stage=EvaluationStage.LOGIC,
                passed=True,
                score=0.95,
                reason="All calls matched",
            ),
        }

        scorecard_1 = Scorecard(
            test_case_id=str(uuid.uuid4()),
            batch_id=str(uuid.uuid4()),
            pack_name="nl2api",
            stage_results=stage_results_1,
            stage_weights={"syntax": 0.1, "logic": 0.9},
            generated_output={"raw_output": '{"tool_name": "test"}'},
            syntax_result=stage_results_1["syntax"],
            logic_result=stage_results_1["logic"],
        )

        await scorecard_repo.save(scorecard_1)
        retrieved_1 = await scorecard_repo.get(scorecard_1.scorecard_id)

        assert retrieved_1.pack_name == "nl2api"
        assert "syntax" in retrieved_1.stage_results
        assert "logic" in retrieved_1.stage_results
        assert retrieved_1.stage_results["syntax"].passed is True
        assert retrieved_1.stage_results["logic"].score == 0.95
        assert retrieved_1.stage_weights == {"syntax": 0.1, "logic": 0.9}
        assert "raw_output" in retrieved_1.generated_output

        # =================================================================
        # Test 2: Arbitrary stage names (RAG-style with metrics)
        # =================================================================
        stage_results_2 = {
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
        }

        scorecard_2 = Scorecard(
            test_case_id=str(uuid.uuid4()),
            batch_id=str(uuid.uuid4()),
            pack_name="rag",
            stage_results=stage_results_2,
            stage_weights={"retrieval": 0.5, "faithfulness": 0.5},
            generated_output={
                "retrieved_docs": ["doc1", "doc2"],
                "answer": "The capital of France is Paris.",
            },
        )

        await scorecard_repo.save(scorecard_2)
        retrieved_2 = await scorecard_repo.get(scorecard_2.scorecard_id)

        assert retrieved_2.pack_name == "rag"
        assert "retrieval" in retrieved_2.stage_results
        assert "faithfulness" in retrieved_2.stage_results
        assert retrieved_2.stage_results["retrieval"].metrics["recall@5"] == 0.8
        assert retrieved_2.stage_results["faithfulness"].metrics["claim_support_ratio"] == 0.9
        assert retrieved_2.generated_output["answer"] == "The capital of France is Paris."

        # =================================================================
        # Test 3: Backwards compatibility with NL2API fields
        # =================================================================
        syntax_result = StageResult(
            stage_name="syntax",
            stage=EvaluationStage.SYNTAX,
            passed=True,
            score=1.0,
        )
        logic_result = StageResult(
            stage_name="logic",
            stage=EvaluationStage.LOGIC,
            passed=True,
            score=1.0,
        )

        scorecard_3 = Scorecard(
            test_case_id=str(uuid.uuid4()),
            batch_id=str(uuid.uuid4()),
            syntax_result=syntax_result,
            logic_result=logic_result,
            generated_tool_calls=(
                ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),
            ),
            generated_nl_response="Apple's price is $150",
            pack_name="nl2api",
            stage_results={"syntax": syntax_result, "logic": logic_result},
        )

        await scorecard_repo.save(scorecard_3)
        retrieved_3 = await scorecard_repo.get(scorecard_3.scorecard_id)

        assert retrieved_3.syntax_result is not None
        assert retrieved_3.syntax_result.passed is True
        assert retrieved_3.logic_result is not None
        assert len(retrieved_3.generated_tool_calls) == 1
        assert retrieved_3.generated_nl_response == "Apple's price is $150"
        assert retrieved_3.pack_name == "nl2api"
        assert "syntax" in retrieved_3.stage_results

        # =================================================================
        # Test 4: Batch query with generic fields
        # =================================================================
        batch_id = str(uuid.uuid4())
        batch_scorecards = []

        for i in range(3):
            sc = Scorecard(
                test_case_id=str(uuid.uuid4()),
                batch_id=batch_id,
                pack_name="nl2api",
                stage_results={
                    "syntax": StageResult(stage_name="syntax", passed=True, score=1.0),
                },
                syntax_result=StageResult(stage_name="syntax", passed=True, score=1.0),
            )
            await scorecard_repo.save(sc)
            batch_scorecards.append(sc)

        batch_retrieved = await scorecard_repo.get_by_batch(batch_id)

        assert len(batch_retrieved) == 3
        for sc in batch_retrieved:
            assert sc.pack_name == "nl2api"
            assert "syntax" in sc.stage_results

        # =================================================================
        # Cleanup
        # =================================================================
        for sc in [scorecard_1, scorecard_2, scorecard_3] + batch_scorecards:
            await scorecard_repo.delete(sc.scorecard_id)
