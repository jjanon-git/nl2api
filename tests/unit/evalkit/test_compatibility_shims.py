"""
Compatibility Shim Validation Tests

Explicitly tests that the backwards-compatibility shims work correctly
after the codebase separation refactor.

These tests verify:
1. Old import paths still work (src.contracts, src.common, src.evaluation.packs)
2. Shims re-export the same objects as direct imports
3. Code using old paths functions correctly

Shims are located in:
- src/contracts/__init__.py -> src/evalkit/contracts
- src/common/__init__.py -> src/evalkit/common
- src/evaluation/packs/__init__.py -> src/nl2api.evaluation, src/rag.evaluation
"""

import pytest

# =============================================================================
# Contracts Shim Tests
# =============================================================================


class TestContractsShim:
    """Tests for src.contracts -> src.evalkit.contracts shim."""

    def test_testcase_importable_from_old_path(self):
        """TestCase is importable from old path."""
        from src.contracts import TestCase

        assert TestCase is not None

    def test_toolcall_importable_from_old_path(self):
        """ToolCall is importable from old path."""
        from src.contracts import ToolCall

        assert ToolCall is not None

    def test_scorecard_importable_from_old_path(self):
        """Scorecard is importable from old path."""
        from src.contracts import Scorecard

        assert Scorecard is not None

    def test_stageresult_importable_from_old_path(self):
        """StageResult is importable from old path."""
        from src.contracts import StageResult

        assert StageResult is not None

    def test_evalcontext_importable_from_old_path(self):
        """EvalContext is importable from old path."""
        from src.contracts import EvalContext

        assert EvalContext is not None

    def test_errorcode_importable_from_old_path(self):
        """ErrorCode is importable from old path."""
        from src.contracts import ErrorCode

        assert ErrorCode is not None

    def test_evaluationstage_importable_from_old_path(self):
        """EvaluationStage is importable from old path."""
        from src.contracts import EvaluationStage

        assert EvaluationStage is not None

    def test_shim_matches_direct_import(self):
        """Shim imports are identical to direct imports."""
        from src.contracts import TestCase as ShimTestCase
        from src.contracts import ToolCall as ShimToolCall
        from src.evalkit.contracts import TestCase as DirectTestCase
        from src.evalkit.contracts import ToolCall as DirectToolCall

        assert ShimTestCase is DirectTestCase
        assert ShimToolCall is DirectToolCall

    def test_old_path_objects_are_usable(self):
        """Objects from old path work correctly."""
        from src.contracts import TestCase, ToolCall

        test_case = TestCase(
            id="shim-test-001",
            nl_query="Test query",
            expected_tool_calls=(ToolCall(tool_name="test", arguments={"arg": "value"}),),
        )

        assert test_case.id == "shim-test-001"
        assert test_case.nl_query == "Test query"
        assert len(test_case.expected_tool_calls) == 1


# =============================================================================
# Common Shim Tests
# =============================================================================


class TestCommonShim:
    """Tests for src.common -> src.evalkit.common shim."""

    def test_storage_importable_from_old_path(self):
        """storage module is importable from old path."""
        from src.common import storage

        assert storage is not None

    def test_telemetry_importable_from_old_path(self):
        """telemetry module is importable from old path."""
        from src.common import telemetry

        assert telemetry is not None

    def test_cache_importable_from_old_path(self):
        """cache module is importable from old path."""
        from src.common import cache

        assert cache is not None

    def test_resilience_importable_from_old_path(self):
        """resilience module is importable from old path."""
        from src.common import resilience

        assert resilience is not None

    def test_logging_importable_from_old_path(self):
        """logging module is importable from old path."""
        from src.common import logging

        assert logging is not None

    def test_git_info_importable_from_old_path(self):
        """git_info module is importable from old path."""
        from src.common import git_info

        assert git_info is not None

    def test_storage_shim_matches_direct(self):
        """Storage shim is identical to direct import."""
        from src.common import storage as shim_storage
        from src.evalkit.common import storage as direct_storage

        assert shim_storage is direct_storage

    def test_telemetry_shim_matches_direct(self):
        """Telemetry shim is identical to direct import."""
        from src.common import telemetry as shim_telemetry
        from src.evalkit.common import telemetry as direct_telemetry

        assert shim_telemetry is direct_telemetry


# =============================================================================
# Packs Shim Tests
# =============================================================================


class TestPacksShim:
    """Tests for src.evaluation.packs shim."""

    def test_nl2api_pack_importable_from_old_path(self):
        """NL2APIPack is importable from old path."""
        from src.evaluation.packs import NL2APIPack

        assert NL2APIPack is not None

    def test_rag_pack_importable_from_old_path(self):
        """RAGPack is importable from old path."""
        from src.evaluation.packs import RAGPack

        assert RAGPack is not None

    def test_get_pack_importable_from_old_path(self):
        """get_pack function is importable from old path."""
        from src.evaluation.packs import get_pack

        assert get_pack is not None

    def test_packs_dict_importable_from_old_path(self):
        """PACKS dict is importable from old path."""
        from src.evaluation.packs import PACKS

        assert PACKS is not None
        assert "nl2api" in PACKS
        assert "rag" in PACKS

    def test_nl2api_pack_shim_matches_direct(self):
        """NL2APIPack shim is identical to direct import."""
        from src.evaluation.packs import NL2APIPack as ShimPack
        from src.nl2api.evaluation import NL2APIPack as DirectPack

        assert ShimPack is DirectPack

    def test_rag_pack_shim_matches_direct(self):
        """RAGPack shim is identical to direct import."""
        from src.evaluation.packs import RAGPack as ShimPack
        from src.rag.evaluation import RAGPack as DirectPack

        assert ShimPack is DirectPack

    def test_nl2api_submodule_importable(self):
        """NL2APIPack can be imported from submodule."""
        from src.evaluation.packs.nl2api import NL2APIPack

        assert NL2APIPack is not None

    def test_nl2api_submodule_matches_direct(self):
        """NL2APIPack from submodule is identical to direct."""
        from src.evaluation.packs.nl2api import NL2APIPack as ShimPack
        from src.nl2api.evaluation.pack import NL2APIPack as DirectPack

        assert ShimPack is DirectPack

    def test_packs_registry_works(self):
        """PACKS registry returns correct pack classes."""
        from src.evaluation.packs import PACKS

        nl2api_pack = PACKS["nl2api"]()
        rag_pack = PACKS["rag"]()

        assert nl2api_pack.name == "nl2api"
        assert rag_pack.name == "rag"


# =============================================================================
# Cross-Module Compatibility Tests
# =============================================================================


class TestCrossModuleCompatibility:
    """Tests for compatibility between shim and direct imports."""

    def test_contracts_work_with_nl2api_pack(self):
        """Contracts from shim work with NL2API pack from shim."""
        from src.contracts import TestCase, ToolCall
        from src.evaluation.packs import NL2APIPack

        test_case = TestCase(
            id="cross-001",
            nl_query="Get Apple price",
            expected_tool_calls=(ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
        )

        pack = NL2APIPack()
        errors = pack.validate_test_case(test_case)

        assert errors == []

    def test_contracts_work_with_rag_pack(self):
        """Contracts from shim work with RAG pack from shim."""
        from src.contracts import TestCase
        from src.evaluation.packs import RAGPack

        test_case = TestCase(
            id="cross-002",
            input={"query": "What is X?"},
            expected={"relevant_docs": ["doc-1"]},
        )

        pack = RAGPack()
        errors = pack.validate_test_case(test_case)

        assert errors == []

    def test_old_path_contracts_with_new_path_pack(self):
        """Contracts from old path work with pack from new path."""
        from src.contracts import TestCase, ToolCall
        from src.nl2api.evaluation.pack import NL2APIPack

        test_case = TestCase(
            id="cross-003",
            nl_query="Get Apple price",
            expected_tool_calls=(ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
        )

        pack = NL2APIPack()
        errors = pack.validate_test_case(test_case)

        assert errors == []

    def test_new_path_contracts_with_old_path_pack(self):
        """Contracts from new path work with pack from old path."""
        from src.evalkit.contracts import TestCase, ToolCall
        from src.evaluation.packs import NL2APIPack

        test_case = TestCase(
            id="cross-004",
            nl_query="Get Apple price",
            expected_tool_calls=(ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
        )

        pack = NL2APIPack()
        errors = pack.validate_test_case(test_case)

        assert errors == []


# =============================================================================
# Evaluation Results Compatibility Tests
# =============================================================================


class TestEvaluationResultsCompatibility:
    """Tests that evaluation results work with shim imports."""

    @pytest.mark.asyncio
    async def test_nl2api_evaluation_with_shim_imports(self):
        """Full NL2API evaluation works with shim imports."""
        import json

        from src.contracts import TestCase, ToolCall
        from src.evaluation.packs import NL2APIPack

        test_case = TestCase(
            id="eval-shim-001",
            nl_query="Get Apple price",
            expected_tool_calls=(ToolCall(tool_name="get_price", arguments={"ticker": "AAPL"}),),
        )

        pack = NL2APIPack()
        system_output = {
            "raw_output": json.dumps([{"tool_name": "get_price", "arguments": {"ticker": "AAPL"}}])
        }

        scorecard = await pack.evaluate(test_case, system_output)

        assert scorecard.test_case_id == test_case.id
        assert scorecard.pack_name == "nl2api"
        assert scorecard.syntax_result.passed is True
        assert scorecard.logic_result.passed is True

    @pytest.mark.asyncio
    async def test_rag_evaluation_with_shim_imports(self):
        """Full RAG evaluation works with shim imports."""
        from src.contracts import TestCase
        from src.evaluation.packs import RAGPack

        test_case = TestCase(
            id="eval-shim-002",
            input={"query": "What is the capital of France?"},
            expected={"relevant_docs": ["doc-1"]},
        )

        pack = RAGPack()
        system_output = {
            "response": "The capital of France is Paris.",
            "retrieved_doc_ids": ["doc-1"],
        }

        scorecard = await pack.evaluate(test_case, system_output)

        assert scorecard.test_case_id == test_case.id
        assert scorecard.pack_name == "rag"
        assert "retrieval" in scorecard.stage_results


# =============================================================================
# Edge Cases and Error Handling
# =============================================================================


class TestShimEdgeCases:
    """Tests for edge cases in shim behavior."""

    def test_star_import_from_contracts(self):
        """Star import from contracts works."""
        # This is what the shim does internally
        from src.evalkit.contracts import __all__ as all_names

        # Verify key exports are in __all__
        assert "TestCase" in all_names
        assert "ToolCall" in all_names
        assert "Scorecard" in all_names

    def test_double_import_is_same_object(self):
        """Importing twice returns same object."""
        from src.contracts import TestCase as TestCaseShim1
        from src.contracts import TestCase as TestCaseShim2
        from src.evalkit.contracts import TestCase as TestCaseDirect

        assert TestCaseShim1 is TestCaseShim2
        assert TestCaseShim2 is TestCaseDirect

    def test_isinstance_works_across_paths(self):
        """isinstance checks work across import paths."""
        from src.contracts import TestCase as ShimTestCase
        from src.evalkit.contracts import TestCase as DirectTestCase

        test_case = ShimTestCase(id="isinstance-001")

        assert isinstance(test_case, DirectTestCase)
        assert isinstance(test_case, ShimTestCase)

    def test_subclass_works_across_paths(self):
        """issubclass checks work across import paths."""
        from src.contracts import TestCase as ShimTestCase
        from src.evalkit.contracts import TestCase as DirectTestCase

        assert issubclass(ShimTestCase, DirectTestCase)
        assert issubclass(DirectTestCase, ShimTestCase)
