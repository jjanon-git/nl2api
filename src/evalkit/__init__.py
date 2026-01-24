"""
Evalkit - Evaluation Framework

A framework for evaluating NL2API systems, RAG pipelines, and other AI applications.

Modules:
- contracts: Data models (TestCase, Scorecard, ToolCall, etc.)
- common: Shared infrastructure (storage, telemetry, cache, resilience)
- core: Core evaluators (AST comparator, temporal, semantics)
- batch: Batch evaluation runner
- cli: Command-line interface
- distributed: Distributed evaluation workers
- continuous: Continuous monitoring
- packs: Evaluation pack protocols
"""

__version__ = "0.1.0"
