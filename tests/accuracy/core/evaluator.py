"""
Accuracy Evaluator

Core evaluation infrastructure for measuring NL2API accuracy
using real LLM calls.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.accuracy.core.config import AccuracyConfig, DEFAULT_CONFIG

logger = logging.getLogger(__name__)


def _load_env():
    """Load environment variables from .env file."""
    env_file = Path(__file__).parent.parent.parent.parent / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip("'\"")
                if key and key not in os.environ:
                    os.environ[key] = value


# Load env on module import
_load_env()


@dataclass(frozen=True)
class AccuracyResult:
    """Result of evaluating a single query."""

    query: str
    expected: str  # Expected domain/output
    predicted: str  # Actual domain/output
    correct: bool
    confidence: float
    latency_ms: int = 0
    reasoning: str | None = None
    error: str | None = None


@dataclass
class AccuracyReport:
    """Report from evaluating a batch of queries."""

    total_count: int = 0
    correct_count: int = 0
    failed_count: int = 0
    error_count: int = 0
    low_confidence_count: int = 0  # Would trigger clarification

    results: list[AccuracyResult] = field(default_factory=list)
    by_category: dict[str, dict[str, int]] = field(default_factory=dict)
    confusion_matrix: dict[str, dict[str, int]] = field(default_factory=dict)

    start_time: datetime | None = None
    end_time: datetime | None = None
    model: str = ""
    tier: str = ""

    @property
    def accuracy(self) -> float:
        """Calculate overall accuracy rate."""
        if self.total_count == 0:
            return 0.0
        return self.correct_count / self.total_count

    @property
    def low_confidence_rate(self) -> float:
        """Calculate rate of low-confidence predictions (would need clarification)."""
        if self.total_count == 0:
            return 0.0
        return self.low_confidence_count / self.total_count

    @property
    def duration_seconds(self) -> float:
        """Calculate total evaluation duration."""
        if self.start_time and self.end_time:
            return (self.end_time - self.start_time).total_seconds()
        return 0.0

    def to_dict(self) -> dict[str, Any]:
        """Convert report to dictionary for serialization."""
        return {
            "accuracy": self.accuracy,
            "total_count": self.total_count,
            "correct_count": self.correct_count,
            "failed_count": self.failed_count,
            "error_count": self.error_count,
            "low_confidence_count": self.low_confidence_count,
            "low_confidence_rate": self.low_confidence_rate,
            "by_category": self.by_category,
            "confusion_matrix": self.confusion_matrix,
            "duration_seconds": self.duration_seconds,
            "model": self.model,
            "tier": self.tier,
        }


@dataclass
class RoutingTestCase:
    """A test case for routing evaluation."""

    id: str
    query: str
    expected_domain: str
    category: str = ""
    tags: list[str] = field(default_factory=list)


class AccuracyEvaluator:
    """
    Evaluates NL2API accuracy using real LLM calls.

    This is the base evaluator that can be extended for different
    evaluation types (routing, agents, end-to-end).
    """

    def __init__(
        self,
        config: AccuracyConfig | None = None,
        llm: Any = None,
    ):
        """
        Initialize the evaluator.

        Args:
            config: Accuracy test configuration
            llm: LLM provider (created from env if not provided)
        """
        self.config = config or DEFAULT_CONFIG
        self._llm = llm

    async def _get_llm(self):
        """Get or create LLM provider."""
        if self._llm is None:
            from src.nl2api.llm.claude import ClaudeProvider

            api_key = os.environ.get("NL2API_ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")

            self._llm = ClaudeProvider(
                api_key=api_key,
                model=self.config.model,
            )
        return self._llm

    async def evaluate_query(
        self,
        query: str,
        expected: str,
    ) -> AccuracyResult:
        """
        Evaluate a single query.

        Must be implemented by subclasses for specific evaluation types.
        """
        raise NotImplementedError("Subclasses must implement evaluate_query")

    async def evaluate_batch(
        self,
        test_cases: list[RoutingTestCase],
        parallel: int | None = None,
        progress_callback: Any = None,
    ) -> AccuracyReport:
        """
        Evaluate a batch of test cases.

        Args:
            test_cases: List of test cases to evaluate
            parallel: Number of parallel requests (uses config default if None)
            progress_callback: Optional callback(current, total, result) for progress

        Returns:
            AccuracyReport with results and metrics
        """
        parallel = parallel or self.config.parallel_requests
        report = AccuracyReport(
            total_count=len(test_cases),
            model=self.config.model,
            start_time=datetime.now(),
        )

        # Evaluate with concurrency control
        semaphore = asyncio.Semaphore(parallel)

        async def eval_with_semaphore(tc: RoutingTestCase, idx: int) -> AccuracyResult:
            async with semaphore:
                result = await self.evaluate_query(tc.query, tc.expected_domain)

                # Update report inline
                if result.correct:
                    report.correct_count += 1
                elif result.error:
                    report.error_count += 1
                else:
                    report.failed_count += 1

                if result.confidence <= self.config.confidence_threshold:
                    report.low_confidence_count += 1

                # Update confusion matrix
                if result.expected not in report.confusion_matrix:
                    report.confusion_matrix[result.expected] = {}
                if result.predicted not in report.confusion_matrix[result.expected]:
                    report.confusion_matrix[result.expected][result.predicted] = 0
                report.confusion_matrix[result.expected][result.predicted] += 1

                # Update by_category
                if tc.category:
                    if tc.category not in report.by_category:
                        report.by_category[tc.category] = {"correct": 0, "total": 0}
                    report.by_category[tc.category]["total"] += 1
                    if result.correct:
                        report.by_category[tc.category]["correct"] += 1

                if progress_callback:
                    progress_callback(idx + 1, len(test_cases), result)

                return result

        # Run evaluations
        results = await asyncio.gather(
            *[eval_with_semaphore(tc, i) for i, tc in enumerate(test_cases)],
            return_exceptions=True,
        )

        # Collect results
        for r in results:
            if isinstance(r, Exception):
                logger.error(f"Evaluation error: {r}")
                report.error_count += 1
            else:
                report.results.append(r)

        report.end_time = datetime.now()
        return report


class RoutingAccuracyEvaluator(AccuracyEvaluator):
    """
    Evaluates routing accuracy - whether queries are routed to the correct domain.
    """

    def __init__(
        self,
        config: AccuracyConfig | None = None,
        llm: Any = None,
        router: Any = None,
    ):
        super().__init__(config, llm)
        self._router = router

    async def _get_router(self):
        """Get or create the router."""
        if self._router is None:
            from src.nl2api.routing.llm_router import LLMToolRouter
            from src.nl2api.routing.protocols import ToolProvider

            llm = await self._get_llm()

            # Create mock providers for each domain
            class MockProvider(ToolProvider):
                def __init__(self, name: str, description: str, capabilities: tuple = (), example_queries: tuple = ()):
                    self._name = name
                    self._description = description
                    self._capabilities = capabilities
                    self._example_queries = example_queries

                @property
                def provider_name(self) -> str:
                    return self._name

                @property
                def provider_description(self) -> str:
                    return self._description

                @property
                def capabilities(self) -> tuple:
                    return self._capabilities

                @property
                def example_queries(self) -> tuple:
                    return self._example_queries

                async def list_tools(self):
                    return []

                async def get_tool_description(self, tool_name: str):
                    return None

            providers = [
                MockProvider(
                    "datastream",
                    "Stock prices, market data, trading volume, historical time series, indices",
                    capabilities=("stock prices", "market cap", "PE ratio", "volume", "historical prices"),
                    example_queries=("What is the price of AAPL?", "Show me MSFT's 52-week high"),
                ),
                MockProvider(
                    "estimates",
                    "Analyst forecasts, FUTURE EPS estimates, revenue projections, recommendations, price targets",
                    capabilities=("EPS forecasts", "revenue estimates", "analyst recommendations", "price targets"),
                    example_queries=("What is the EPS forecast for Apple?", "Show analyst recommendations"),
                ),
                MockProvider(
                    "fundamentals",
                    "HISTORICAL financial statements, balance sheet, income statement, past financial ratios",
                    capabilities=("balance sheet", "income statement", "ROE", "ROA", "historical financials"),
                    example_queries=("What was Apple's revenue last year?", "Show MSFT's debt ratio"),
                ),
                MockProvider(
                    "officers",
                    "Executives, CEO, CFO, board members, compensation, governance data",
                    capabilities=("CEO info", "board members", "executive compensation"),
                    example_queries=("Who is the CEO of Apple?", "Show Microsoft's board"),
                ),
                MockProvider(
                    "screening",
                    "Stock screening, ranking queries, top N by metric, filtering criteria",
                    capabilities=("stock screening", "top N rankings", "filter by criteria"),
                    example_queries=("Top 10 stocks by market cap", "Find stocks with PE below 15"),
                ),
            ]

            self._router = LLMToolRouter(
                llm=llm,
                tool_providers=providers,
                default_confidence=0.85,
            )

        return self._router

    async def evaluate_query(
        self,
        query: str,
        expected: str,
    ) -> AccuracyResult:
        """Evaluate routing for a single query."""
        import time

        router = await self._get_router()
        start_time = time.perf_counter()

        try:
            result = await router.route(query)
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            return AccuracyResult(
                query=query,
                expected=expected,
                predicted=result.domain,
                correct=(result.domain == expected),
                confidence=result.confidence,
                latency_ms=latency_ms,
                reasoning=result.reasoning,
            )

        except Exception as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            return AccuracyResult(
                query=query,
                expected=expected,
                predicted="error",
                correct=False,
                confidence=0.0,
                latency_ms=latency_ms,
                error=str(e),
            )
