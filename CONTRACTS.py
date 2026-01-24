"""
NL2API Contracts - Pydantic v2 Schemas

BACKWARD COMPATIBILITY WRAPPER

This module re-exports all contracts from src.evalkit.contracts for backward compatibility.
New code should import from src.evalkit.contracts directly:

    # Preferred (new code)
    from src.evalkit.contracts import TestCase, Scorecard
    from src.evalkit.contracts.core import ToolCall
    from src.evalkit.contracts.evaluation import EvaluationConfig

    # Still works (backward compatible)
    from CONTRACTS import TestCase, Scorecard

The contracts have been split into focused modules:
- evalkit/contracts/core.py: Fundamental types, enums, test case models
- evalkit/contracts/evaluation.py: Scorecard, stage results, evaluator config
- evalkit/contracts/worker.py: Batch jobs, worker tasks, worker config
- evalkit/contracts/tenant.py: Multi-tenant models (clients, test suites, runs)
- evalkit/contracts/storage.py: Azure Table Storage helpers
"""

# Re-export everything from the contracts package
from src.evalkit.contracts import (
    # Worker
    BatchJob,
    # Enums
    CircuitState,
    # Management
    Client,
    ClientType,
    ErrorCode,
    # Evaluation Pack Protocol (general-purpose framework)
    EvalContext,
    EvalMode,
    EvaluationConfig,
    EvaluationPack,
    EvaluationRun,
    EvaluationStage,
    # Evaluation
    Evaluator,
    # Utility
    FrozenDict,
    # Storage
    IdempotencyRecord,
    LLMJudgeConfig,
    LLMProvider,
    RunStatus,
    Scorecard,
    Stage,
    StageResult,
    # Core Models
    SystemResponse,
    TableStorageEntity,
    TargetSystemConfig,
    TaskPriority,
    TaskStatus,
    TemporalContext,
    TemporalStability,
    TemporalValidationMode,
    TestCase,
    TestCaseMetadata,
    TestCaseSetConfig,
    TestCaseStatus,
    TestSuite,
    ToolCall,
    # Registry
    ToolRegistry,
    WorkerConfig,
    WorkerTask,
    _generate_id,
    _now_utc,
)

__all__ = [
    # Utility
    "FrozenDict",
    "_now_utc",
    "_generate_id",
    # Enums
    "EvaluationStage",
    "TaskStatus",
    "TaskPriority",
    "ErrorCode",
    "CircuitState",
    "TestCaseStatus",
    "ClientType",
    "EvalMode",
    "TemporalStability",
    "TemporalValidationMode",
    "LLMProvider",
    "RunStatus",
    # Registry
    "ToolRegistry",
    # Core Models
    "ToolCall",
    "TestCaseMetadata",
    "TemporalContext",
    "TestCase",
    "TestCaseSetConfig",
    "SystemResponse",
    # Evaluation
    "StageResult",
    "Scorecard",
    "Evaluator",
    "EvaluationConfig",
    "LLMJudgeConfig",
    # Evaluation Pack Protocol (general-purpose framework)
    "EvalContext",
    "Stage",
    "EvaluationPack",
    # Worker
    "WorkerTask",
    "BatchJob",
    "WorkerConfig",
    # Management
    "TargetSystemConfig",
    "TestSuite",
    "Client",
    "EvaluationRun",
    # Storage
    "TableStorageEntity",
    "IdempotencyRecord",
]
