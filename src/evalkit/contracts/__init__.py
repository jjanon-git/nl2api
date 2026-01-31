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
    from src.evalkit.contracts import TestCase, Scorecard, BatchJob
"""

# Core models
from src.evalkit.contracts.core import (
    CircuitState,
    ClientType,
    DataSourceMetadata,
    DataSourceType,
    ErrorCode,
    EvalMode,
    EvaluationStage,
    FrozenDict,
    ReviewStatus,
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
from src.evalkit.contracts.evaluation import (
    EvalContext,
    EvaluationConfig,
    EvaluationPack,
    Evaluator,
    LLMJudgeConfig,
    Scorecard,
    Stage,
    StageResult,
)

# LLM protocols and types
from src.evalkit.contracts.llm import (
    EntityResolver,
    LLMMessage,
    LLMProviderProtocol,
    LLMResponse,
    MessageRole,
    ResolvedEntity,
    create_default_llm_provider,
)

# Storage models
from src.evalkit.contracts.storage import (
    IdempotencyRecord,
    TableStorageEntity,
)

# Tenant models
from src.evalkit.contracts.tenant import (
    Client,
    EvaluationRun,
    LLMProvider,
    RunStatus,
    TargetSystemConfig,
    TestSuite,
)

# Worker models
from src.evalkit.contracts.worker import (
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
    "DataSourceType",
    "ReviewStatus",
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
    "DataSourceMetadata",
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
    # LLM
    "MessageRole",
    "LLMMessage",
    "LLMResponse",
    "LLMProviderProtocol",
    "create_default_llm_provider",
    # Entity Resolution
    "ResolvedEntity",
    "EntityResolver",
]
