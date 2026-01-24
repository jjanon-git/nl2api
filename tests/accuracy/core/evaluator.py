"""
Accuracy Evaluator

Core evaluation infrastructure for measuring NL2API accuracy
using real LLM calls.

Supports two modes:
- Batch API (default): 50% cheaper, higher rate limits, async processing
- Real-time API: Immediate results, uses retry with exponential backoff

Emits metrics to OTEL for tracking accuracy trends over time.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from tests.accuracy.core.config import DEFAULT_CONFIG, AccuracyConfig

logger = logging.getLogger(__name__)

# Lazy import for metrics to avoid import errors if OTEL not installed
_accuracy_metrics = None


def _get_metrics():
    """Get accuracy metrics instance (lazy load)."""
    global _accuracy_metrics
    if _accuracy_metrics is None:
        try:
            from src.evalkit.common.telemetry import get_accuracy_metrics

            _accuracy_metrics = get_accuracy_metrics()
        except ImportError:
            logger.debug("Telemetry not available, metrics disabled")
            _accuracy_metrics = None
    return _accuracy_metrics


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

    Supports two execution modes:
    - Batch API (default): Submit all requests at once, poll for results
    - Real-time API: Execute requests with concurrency control and retry
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
        self._anthropic_client = None

    async def _get_llm(self):
        """Get or create LLM provider."""
        if self._llm is None:
            from src.nl2api.llm.claude import ClaudeProvider

            api_key = os.environ.get("NL2API_ANTHROPIC_API_KEY") or os.environ.get(
                "ANTHROPIC_API_KEY"
            )
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")

            self._llm = ClaudeProvider(
                api_key=api_key,
                model=self.config.model,
            )
        return self._llm

    def _get_anthropic_client(self):
        """Get or create raw Anthropic client for batch API."""
        if self._anthropic_client is None:
            import anthropic

            # Load .env if needed
            _load_env()

            api_key = os.environ.get("NL2API_ANTHROPIC_API_KEY") or os.environ.get(
                "ANTHROPIC_API_KEY"
            )
            if not api_key:
                raise ValueError("ANTHROPIC_API_KEY not set")

            self._anthropic_client = anthropic.Anthropic(api_key=api_key)
        return self._anthropic_client

    def _calculate_retry_delay(self, attempt: int) -> float:
        """Calculate delay with exponential backoff and jitter."""
        base_delay = self.config.retry_base_delay * (2**attempt)
        capped_delay = min(base_delay, self.config.retry_max_delay)
        jitter = random.uniform(0, self.config.retry_jitter * capped_delay)
        return capped_delay + jitter

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

    async def evaluate_query_with_retry(
        self,
        query: str,
        expected: str,
    ) -> AccuracyResult:
        """Evaluate a single query with retry logic."""
        import anthropic

        last_error: Exception | None = None

        for attempt in range(self.config.max_retries):
            try:
                return await self.evaluate_query(query, expected)
            except anthropic.RateLimitError as e:
                last_error = e
                delay = self._calculate_retry_delay(attempt)
                logger.warning(
                    f"Rate limited, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.config.max_retries})"
                )
                await asyncio.sleep(delay)
            except anthropic.APIConnectionError as e:
                last_error = e
                delay = self._calculate_retry_delay(attempt)
                logger.warning(
                    f"Connection error, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.config.max_retries})"
                )
                await asyncio.sleep(delay)
            except anthropic.InternalServerError as e:
                last_error = e
                delay = self._calculate_retry_delay(attempt)
                logger.warning(
                    f"Server error, retrying in {delay:.1f}s (attempt {attempt + 1}/{self.config.max_retries})"
                )
                await asyncio.sleep(delay)
            except Exception as e:
                # Non-retryable error
                return AccuracyResult(
                    query=query,
                    expected=expected,
                    predicted="error",
                    correct=False,
                    confidence=0.0,
                    error=str(e),
                )

        # All retries exhausted
        return AccuracyResult(
            query=query,
            expected=expected,
            predicted="unknown",
            correct=False,
            confidence=0.0,
            reasoning=f"Retries exhausted: {last_error}",
            error=str(last_error) if last_error else "Unknown error",
        )

    def build_batch_request(
        self,
        test_case: RoutingTestCase,
        system_prompt: str,
        user_prompt: str,
    ) -> dict[str, Any]:
        """Build a single batch request for the Anthropic Batch API."""
        return {
            "custom_id": test_case.id,
            "params": {
                "model": self.config.model,
                "max_tokens": 1024,
                "temperature": 0.0,
                "messages": [{"role": "user", "content": user_prompt}],
                "system": system_prompt,
            },
        }

    async def evaluate_batch(
        self,
        test_cases: list[RoutingTestCase],
        parallel: int | None = None,
        progress_callback: Any = None,
        tier: str = "",
        use_batch_api: bool | None = None,
    ) -> AccuracyReport:
        """
        Evaluate a batch of test cases.

        Args:
            test_cases: List of test cases to evaluate
            parallel: Number of parallel requests (real-time mode only)
            progress_callback: Optional callback(current, total, result) for progress
            tier: Test tier for metrics (tier1, tier2, tier3)
            use_batch_api: Use Batch API (default from config, 50% cheaper)

        Returns:
            AccuracyReport with results and metrics
        """
        use_batch = use_batch_api if use_batch_api is not None else self.config.use_batch_api

        if use_batch:
            return await self._evaluate_batch_api(test_cases, progress_callback, tier)
        else:
            return await self._evaluate_realtime(test_cases, parallel, progress_callback, tier)

    async def _evaluate_batch_api(
        self,
        test_cases: list[RoutingTestCase],
        progress_callback: Any = None,
        tier: str = "",
    ) -> AccuracyReport:
        """Evaluate using Anthropic Batch API (50% cheaper, higher rate limits)."""
        report = AccuracyReport(
            total_count=len(test_cases),
            model=self.config.model,
            tier=tier,
            start_time=datetime.now(),
        )

        # Build batch requests - subclasses must implement this
        requests = self._build_batch_requests(test_cases)

        if not requests:
            logger.warning("No batch requests generated")
            report.end_time = datetime.now()
            return report

        client = self._get_anthropic_client()

        # Create batch
        logger.info(f"Creating batch with {len(requests)} requests...")
        batch = client.messages.batches.create(requests=requests)
        batch_id = batch.id
        logger.info(f"Batch created: {batch_id}")

        # Poll for completion
        start_poll = time.time()
        while True:
            batch = client.messages.batches.retrieve(batch_id)

            if batch.processing_status == "ended":
                logger.info(f"Batch {batch_id} completed")
                break

            elapsed = time.time() - start_poll
            if elapsed > self.config.batch_timeout:
                logger.error(f"Batch {batch_id} timed out after {elapsed:.0f}s")
                report.error_count = len(test_cases)
                report.end_time = datetime.now()
                return report

            # Log progress
            counts = batch.request_counts
            processed = counts.succeeded + counts.errored + counts.canceled + counts.expired
            logger.info(
                f"Batch {batch_id}: {processed}/{len(requests)} processed "
                f"(succeeded={counts.succeeded}, errored={counts.errored})"
            )

            await asyncio.sleep(self.config.batch_poll_interval)

        # Retrieve results
        results_map = {}
        for result in client.messages.batches.results(batch_id):
            results_map[result.custom_id] = result

        # Process results
        metrics = _get_metrics()
        for i, tc in enumerate(test_cases):
            batch_result = results_map.get(tc.id)
            result = self._parse_batch_result(tc, batch_result)

            # Update report
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

            # Emit per-query metrics to OTEL
            if metrics:
                metrics.record_query_result(
                    correct=result.correct,
                    confidence=result.confidence,
                    latency_ms=result.latency_ms,
                    expected_domain=result.expected,
                    predicted_domain=result.predicted,
                    tier=tier,
                    category=tc.category,
                    error=bool(result.error),
                )

            report.results.append(result)

            if progress_callback:
                progress_callback(i + 1, len(test_cases), result)

        report.end_time = datetime.now()

        # Emit batch completion metrics to OTEL
        if metrics:
            metrics.record_batch_complete(
                total_count=report.total_count,
                correct_count=report.correct_count,
                duration_seconds=report.duration_seconds,
                tier=tier,
                model=report.model,
            )

        return report

    def _build_batch_requests(self, test_cases: list[RoutingTestCase]) -> list[dict[str, Any]]:
        """Build batch requests for test cases. Subclasses must implement."""
        raise NotImplementedError("Subclasses must implement _build_batch_requests")

    def _parse_batch_result(self, test_case: RoutingTestCase, batch_result: Any) -> AccuracyResult:
        """Parse a batch result into AccuracyResult. Subclasses must implement."""
        raise NotImplementedError("Subclasses must implement _parse_batch_result")

    async def _evaluate_realtime(
        self,
        test_cases: list[RoutingTestCase],
        parallel: int | None = None,
        progress_callback: Any = None,
        tier: str = "",
    ) -> AccuracyReport:
        """Evaluate using real-time API with retry and delays."""
        parallel = parallel or self.config.parallel_requests
        report = AccuracyReport(
            total_count=len(test_cases),
            model=self.config.model,
            tier=tier,
            start_time=datetime.now(),
        )

        # Get metrics instance
        metrics = _get_metrics()

        # Evaluate with concurrency control and delays
        semaphore = asyncio.Semaphore(parallel)
        request_count = 0
        request_lock = asyncio.Lock()

        async def eval_with_semaphore(tc: RoutingTestCase, idx: int) -> AccuracyResult:
            nonlocal request_count

            async with semaphore:
                # Add delay between requests to avoid rate limits
                async with request_lock:
                    if request_count > 0 and self.config.request_delay > 0:
                        # Add jitter to delay
                        delay = self.config.request_delay * (1 + random.uniform(-0.2, 0.2))
                        await asyncio.sleep(delay)
                    request_count += 1

                # Use retry-enabled evaluation
                result = await self.evaluate_query_with_retry(tc.query, tc.expected_domain)

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

                # Emit per-query metrics to OTEL
                if metrics:
                    metrics.record_query_result(
                        correct=result.correct,
                        confidence=result.confidence,
                        latency_ms=result.latency_ms,
                        expected_domain=result.expected,
                        predicted_domain=result.predicted,
                        tier=tier,
                        category=tc.category,
                        error=bool(result.error),
                    )

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

        # Emit batch completion metrics to OTEL
        if metrics:
            metrics.record_batch_complete(
                total_count=report.total_count,
                correct_count=report.correct_count,
                duration_seconds=report.duration_seconds,
                tier=tier,
                model=report.model,
            )

        return report


BATCH_ROUTING_SYSTEM_PROMPT = """You are a query router for LSEG financial data APIs.

Analyze the user's query and select the most appropriate domain API.

## Domains

- "datastream": Current stock prices, market data, trading volume, historical price time series, indices
- "estimates": FUTURE/FORECAST data - analyst EPS forecasts, revenue projections, recommendations, price targets
- "fundamentals": HISTORICAL/REPORTED data - past financial statements, reported earnings, balance sheet, ratios
- "officers": Executives, board members, compensation, governance data
- "screening": Stock screening, ranking, filtering criteria, TOP/BOTTOM queries

## Temporal Context (Critical)

- "EPS forecast", "expected earnings", "next quarter" → estimates (FUTURE)
- "last year's EPS", "reported earnings", "2023 revenue" → fundamentals (PAST)
- "EPS" alone WITHOUT temporal context → AMBIGUOUS (confidence <= 0.5)

## Output Format

Respond with ONLY a JSON object (no markdown, no explanation):
{"domain": "<domain>", "confidence": <0.0-1.0>, "reasoning": "<brief reason>"}

Confidence:
- 0.85-1.0: Clear, unambiguous query
- 0.6-0.8: Reasonable inference but some ambiguity
- 0.3-0.5: AMBIGUOUS - needs clarification"""


class RoutingAccuracyEvaluator(AccuracyEvaluator):
    """
    Evaluates routing accuracy - whether queries are routed to the correct domain.

    Supports both batch API (default) and real-time API modes.
    """

    VALID_DOMAINS = {"datastream", "estimates", "fundamentals", "officers", "screening"}

    def __init__(
        self,
        config: AccuracyConfig | None = None,
        llm: Any = None,
        router: Any = None,
    ):
        super().__init__(config, llm)
        self._router = router

    def _build_batch_requests(self, test_cases: list[RoutingTestCase]) -> list[dict[str, Any]]:
        """Build batch requests for routing evaluation."""
        requests = []

        for tc in test_cases:
            request = {
                "custom_id": tc.id,
                "params": {
                    "model": self.config.model,
                    "max_tokens": 256,
                    "temperature": 0.0,
                    "system": BATCH_ROUTING_SYSTEM_PROMPT,
                    "messages": [{"role": "user", "content": tc.query}],
                },
            }
            requests.append(request)

        return requests

    def _parse_batch_result(self, test_case: RoutingTestCase, batch_result: Any) -> AccuracyResult:
        """Parse a batch result into AccuracyResult."""
        if batch_result is None:
            return AccuracyResult(
                query=test_case.query,
                expected=test_case.expected_domain,
                predicted="error",
                correct=False,
                confidence=0.0,
                error="No result returned from batch",
            )

        # Check for errors
        if batch_result.result.type == "errored":
            return AccuracyResult(
                query=test_case.query,
                expected=test_case.expected_domain,
                predicted="error",
                correct=False,
                confidence=0.0,
                error=str(batch_result.result.error),
            )

        # Parse successful result
        try:
            message = batch_result.result.message
            content = message.content[0].text if message.content else ""

            # Parse JSON response
            parsed = self._parse_routing_json(content)
            domain = parsed.get("domain", "unknown")
            confidence = float(parsed.get("confidence", 0.0))
            reasoning = parsed.get("reasoning", "")

            # Validate domain
            if domain not in self.VALID_DOMAINS:
                domain = "unknown"
                confidence = 0.0

            return AccuracyResult(
                query=test_case.query,
                expected=test_case.expected_domain,
                predicted=domain,
                correct=(domain == test_case.expected_domain),
                confidence=confidence,
                reasoning=reasoning,
            )

        except Exception as e:
            return AccuracyResult(
                query=test_case.query,
                expected=test_case.expected_domain,
                predicted="error",
                correct=False,
                confidence=0.0,
                error=f"Failed to parse result: {e}",
            )

    def _parse_routing_json(self, content: str) -> dict[str, Any]:
        """Parse JSON from LLM response, handling common formatting issues."""
        content = content.strip()

        # Remove markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (```json and ```)
            content = "\n".join(lines[1:-1] if lines[-1] == "```" else lines[1:])
            content = content.strip()

        # Try to parse as JSON
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            # Try to extract JSON object from content
            import re

            match = re.search(r"\{[^}]+\}", content)
            if match:
                try:
                    return json.loads(match.group())
                except json.JSONDecodeError:
                    pass

            # Return unknown if parsing fails
            return {
                "domain": "unknown",
                "confidence": 0.0,
                "reasoning": f"Failed to parse: {content[:100]}",
            }

    async def _get_router(self):
        """Get or create the router."""
        if self._router is None:
            from src.nl2api.routing.llm_router import LLMToolRouter
            from src.nl2api.routing.protocols import ToolProvider

            llm = await self._get_llm()

            # Create mock providers for each domain
            class MockProvider(ToolProvider):
                def __init__(
                    self,
                    name: str,
                    description: str,
                    capabilities: tuple = (),
                    example_queries: tuple = (),
                ):
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
                    capabilities=(
                        "stock prices",
                        "market cap",
                        "PE ratio",
                        "volume",
                        "historical prices",
                    ),
                    example_queries=("What is the price of AAPL?", "Show me MSFT's 52-week high"),
                ),
                MockProvider(
                    "estimates",
                    "Analyst forecasts, FUTURE EPS estimates, revenue projections, recommendations, price targets",
                    capabilities=(
                        "EPS forecasts",
                        "revenue estimates",
                        "analyst recommendations",
                        "price targets",
                    ),
                    example_queries=(
                        "What is the EPS forecast for Apple?",
                        "Show analyst recommendations",
                    ),
                ),
                MockProvider(
                    "fundamentals",
                    "HISTORICAL financial statements, balance sheet, income statement, past financial ratios",
                    capabilities=(
                        "balance sheet",
                        "income statement",
                        "ROE",
                        "ROA",
                        "historical financials",
                    ),
                    example_queries=(
                        "What was Apple's revenue last year?",
                        "Show MSFT's debt ratio",
                    ),
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
