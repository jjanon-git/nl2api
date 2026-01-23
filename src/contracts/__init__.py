"""
NL2API Contracts Package

This package contains all data contracts for the NL2API system.
Split into focused modules:

- core: Fundamental types, enums, and test case models
- evaluation: Scorecard, stage results, and evaluator configuration
- worker: Batch jobs, worker tasks, and worker configuration
- tenant: Multi-tenant models (clients, test suites, runs)
- storage: Azure Table Storage helpers

For backward compatibility, all models can be imported from this package:
    from src.contracts import TestCase, Scorecard, BatchJob
"""

# Core models
from src.contracts.core import (
    CircuitState,
    ClientType,
    ErrorCode,
    EvalMode,
    EvaluationStage,
    FrozenDict,
    SystemResponse,
    TaskPriority,
    TaskStatus,
    TemporalContext,
    TemporalStability,
    TemporalValidationMode,
    TestCase,
    TestCaseMetadata,
    TestCaseSetConfig,
    TestCaseStatus,
    ToolCall,
    ToolRegistry,
    _generate_id,
    _now_utc,
)

# Evaluation models
from src.contracts.evaluation import (
    EvaluationConfig,
    Evaluator,
    LLMJudgeConfig,
    Scorecard,
    StageResult,
)

# Storage models
from src.contracts.storage import (
    IdempotencyRecord,
    TableStorageEntity,
)

# Tenant models
from src.contracts.tenant import (
    Client,
    EvaluationRun,
    LLMProvider,
    RunStatus,
    TargetSystemConfig,
    TestSuite,
)

# Worker models
from src.contracts.worker import (
    BatchJob,
    WorkerConfig,
    WorkerTask,
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
