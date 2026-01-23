"""
NL2API Contracts - Pydantic v2 Schemas

BACKWARD COMPATIBILITY WRAPPER

This module re-exports all contracts from src/contracts/ for backward compatibility.
New code should import from src/contracts directly:

    # Preferred (new code)
    from src.contracts import TestCase, Scorecard
    from src.contracts.core import ToolCall
    from src.contracts.evaluation import EvaluationConfig

    # Still works (backward compatible)
    from CONTRACTS import TestCase, Scorecard

The contracts have been split into focused modules:
- src/contracts/core.py: Fundamental types, enums, test case models
- src/contracts/evaluation.py: Scorecard, stage results, evaluator config
- src/contracts/worker.py: Batch jobs, worker tasks, worker config
- src/contracts/tenant.py: Multi-tenant models (clients, test suites, runs)
- src/contracts/storage.py: Azure Table Storage helpers
"""

# Re-export everything from the contracts package
from src.contracts import (
    # Worker
    BatchJob,
    # Enums
    CircuitState,
    # Management
    Client,
    ClientType,
    ErrorCode,
    EvalMode,
    EvaluationConfig,
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
