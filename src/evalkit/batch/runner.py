"""
Batch Runner

Runs batch evaluations with concurrency control, progress tracking,
and result aggregation.
"""

from __future__ import annotations

import asyncio
import json
import logging
import random
import time
from collections.abc import Callable
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from CONTRACTS import (
    BatchJob,
    EvalContext,
    Scorecard,
    SystemResponse,
    TaskStatus,
    TestCase,
)
from src.evalkit.batch.config import BatchRunnerConfig
from src.evalkit.batch.metrics import get_metrics
from src.evalkit.batch.pricing import calculate_cost
from src.evaluation.packs import get_pack

if TYPE_CHECKING:
    from src.evalkit.common.storage.protocols import (
        BatchJobRepository,
        ScorecardRepository,
        TestCaseRepository,
    )


console = Console()


async def simulate_correct_response(test_case: TestCase) -> SystemResponse:
    """
    Simulate a correct response matching expected tool calls.

    This is the default simulator that always produces correct responses.
    Used for validating the pipeline at scale before integrating real targets.
    """
    # Simulate some latency
    latency_ms = random.randint(50, 200)
    await asyncio.sleep(latency_ms / 1000)

    # Build raw output from expected tool calls
    raw_output = json.dumps(
        [
            {"tool_name": tc.tool_name, "arguments": dict(tc.arguments)}
            for tc in test_case.expected_tool_calls
        ]
    )

    return SystemResponse(
        raw_output=raw_output,
        nl_response=test_case.expected_nl_response,
        latency_ms=latency_ms,
    )


class BatchRunner:
    """
    Runs batch evaluations with concurrency control.

    Features:
    - Concurrent execution via asyncio.Semaphore
    - Real-time progress tracking with Rich
    - Batch job persistence and status tracking
    - Configurable response simulation
    """

    def __init__(
        self,
        test_case_repo: TestCaseRepository,
        scorecard_repo: ScorecardRepository,
        batch_repo: BatchJobRepository,
        config: BatchRunnerConfig,
    ):
        """
        Initialize batch runner.

        Args:
            test_case_repo: Repository for fetching test cases
            scorecard_repo: Repository for saving scorecards
            batch_repo: Repository for batch job tracking
            config: Configuration options (pack_name is required)
        """
        self.test_case_repo = test_case_repo
        self.scorecard_repo = scorecard_repo
        self.batch_repo = batch_repo
        self.config = config

        # Build pack-specific configuration
        pack_kwargs: dict[str, Any] = {}
        if self.config.pack_name == "nl2api":
            pack_kwargs = {
                "semantics_enabled": self.config.semantics_enabled,
            }
        # RAG pack uses its own config structure via RAGPackConfig
        # Additional pack configs can be added here as needed

        # Get the evaluation pack
        self.pack = get_pack(self.config.pack_name, **pack_kwargs)

        self.semaphore = asyncio.Semaphore(self.config.max_concurrency)
        self.metrics = get_metrics()

    async def run(
        self,
        tags: list[str] | None = None,
        complexity_min: int | None = None,
        complexity_max: int | None = None,
        limit: int | None = None,
        response_simulator: Callable[[TestCase], Any] | None = None,
        resume_batch_id: str | None = None,
    ) -> BatchJob | None:
        """
        Run batch evaluation on filtered test cases.

        Args:
            tags: Filter test cases by tags (OR logic)
            complexity_min: Minimum complexity level (1-5)
            complexity_max: Maximum complexity level (1-5)
            limit: Maximum number of test cases to run
            response_simulator: Async function to generate responses
                              (defaults to simulate_correct_response)
            resume_batch_id: ID of batch to resume (not yet implemented)

        Returns:
            Completed BatchJob with results summary, or None if no test cases found
        """
        # TODO: Implement resume logic using resume_batch_id
        _ = resume_batch_id  # Suppress unused warning
        start_time = time.perf_counter()
        simulator = response_simulator or simulate_correct_response

        # Fetch test cases
        test_cases = await self.test_case_repo.list(
            tags=tags,
            complexity_min=complexity_min,
            complexity_max=complexity_max,
            limit=limit or 10000,  # Large default
            offset=0,
        )

        if not test_cases:
            if self.config.show_progress:
                console.print("[yellow]No test cases found matching filters[/yellow]")
            # Return None to indicate no tests were run
            # BatchJob model requires total_tests >= 1, so we can't create an empty one
            return None

        # Create batch job with run tracking
        batch_job = BatchJob(
            total_tests=len(test_cases),
            status=TaskStatus.IN_PROGRESS,
            started_at=datetime.now(UTC),
            tags=tuple(tags) if tags else (),
            run_label=self.config.run_label,
            run_description=self.config.run_description,
            git_commit=self.config.git_commit,
            git_branch=self.config.git_branch,
        )
        await self.batch_repo.create(batch_job)

        if self.config.show_progress:
            console.print("\n[bold]Running batch evaluation...[/bold]")
            console.print(f"  Batch ID: [cyan]{batch_job.batch_id}[/cyan]")
            console.print(f"  Run Label: [magenta]{self.config.run_label}[/magenta]")
            if self.config.run_description:
                console.print(f"  Description: {self.config.run_description}")
            if self.config.git_commit:
                git_info = f"{self.config.git_commit}"
                if self.config.git_branch:
                    git_info += f" ({self.config.git_branch})"
                console.print(f"  Git: [dim]{git_info}[/dim]")
            console.print(f"  Test cases: {len(test_cases)}")
            console.print(f"  Concurrency: {self.config.max_concurrency}")
            console.print(f"  Client: [cyan]{self.config.client_type}[/cyan]", end="")
            if self.config.client_version:
                console.print(f" / [cyan]{self.config.client_version}[/cyan]")
            else:
                console.print()
            console.print(f"  Eval mode: [cyan]{self.config.eval_mode}[/cyan]")
            console.print(f"  Pack: [cyan]{self.config.pack_name}[/cyan]\n")

        # Track results with lock for thread-safe counter updates
        # asyncio.gather runs tasks concurrently, so counter increments need synchronization
        counter_lock = asyncio.Lock()
        passed_count = 0
        failed_count = 0
        failed_tests: list[tuple[str, str, float]] = []  # (id, query, score)
        scorecards: list[Scorecard] = []

        # Run evaluations with progress bar
        if self.config.show_progress:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                MofNCompleteColumn(),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=console,
                refresh_per_second=10,
            ) as progress:
                task = progress.add_task(
                    "[cyan]Evaluating...",
                    total=len(test_cases),
                )

                # Create evaluation tasks
                async def evaluate_with_progress(tc: TestCase) -> Scorecard:
                    nonlocal passed_count, failed_count
                    scorecard = await self._evaluate_one(tc, batch_job.batch_id, simulator)
                    # Use lock to ensure atomic counter updates under concurrency
                    async with counter_lock:
                        if scorecard.overall_passed:
                            passed_count += 1
                        else:
                            failed_count += 1
                            # Use nl_query if available, else input.query, else id
                            query_text = getattr(tc, "nl_query", None) or tc.input.get(
                                "query", tc.id
                            )
                            display_query = (
                                query_text[:60] + "..." if len(query_text) > 60 else query_text
                            )
                            failed_tests.append(
                                (
                                    tc.id,
                                    display_query,
                                    scorecard.overall_score,
                                )
                            )
                        progress.update(
                            task,
                            advance=1,
                            description=f"[cyan]Evaluating... [green]{passed_count} passed[/green] [red]{failed_count} failed[/red]",
                        )
                    return scorecard

                # Run all evaluations concurrently with exception handling
                # return_exceptions=True ensures one failure doesn't lose all results
                results = await asyncio.gather(
                    *[evaluate_with_progress(tc) for tc in test_cases],
                    return_exceptions=True,
                )

                # Filter out exceptions and log them
                scorecards = []
                for i, result in enumerate(results):
                    if isinstance(result, Exception):
                        # Log the error but continue processing
                        logging.getLogger(__name__).error(
                            f"Evaluation failed for test case {test_cases[i].id}: {result}"
                        )
                        async with counter_lock:
                            failed_count += 1
                            failed_tests.append(
                                (
                                    test_cases[i].id,
                                    str(result)[:60],
                                    0.0,
                                )
                            )
                            progress.update(
                                task,
                                advance=1,
                                description=f"[cyan]Evaluating... [green]{passed_count} passed[/green] [red]{failed_count} failed[/red]",
                            )
                    else:
                        scorecards.append(result)
        else:
            # Run without progress bar
            async def evaluate_silent(tc: TestCase) -> Scorecard:
                nonlocal passed_count, failed_count
                scorecard = await self._evaluate_one(tc, batch_job.batch_id, simulator)
                # Use lock to ensure atomic counter updates under concurrency
                async with counter_lock:
                    if scorecard.overall_passed:
                        passed_count += 1
                    else:
                        failed_count += 1
                        # Use nl_query if available, else input.query, else id
                        query_text = getattr(tc, "nl_query", None) or tc.input.get("query", tc.id)
                        display_query = (
                            query_text[:60] + "..." if len(query_text) > 60 else query_text
                        )
                        failed_tests.append(
                            (
                                tc.id,
                                display_query,
                                scorecard.overall_score,
                            )
                        )
                return scorecard

            # Run all evaluations concurrently with exception handling
            results = await asyncio.gather(
                *[evaluate_silent(tc) for tc in test_cases],
                return_exceptions=True,
            )

            # Filter out exceptions and log them
            scorecards = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    logging.getLogger(__name__).error(
                        f"Evaluation failed for test case {test_cases[i].id}: {result}"
                    )
                    async with counter_lock:
                        failed_count += 1
                        failed_tests.append(
                            (
                                test_cases[i].id,
                                str(result)[:60],
                                0.0,
                            )
                        )
                else:
                    scorecards.append(result)

        # Calculate duration
        duration_seconds = time.perf_counter() - start_time

        # Update batch job (preserve run tracking fields)
        completed_batch = BatchJob(
            batch_id=batch_job.batch_id,
            total_tests=len(test_cases),
            completed_count=passed_count,
            failed_count=failed_count,
            status=TaskStatus.COMPLETED,
            created_at=batch_job.created_at,
            started_at=batch_job.started_at,
            completed_at=datetime.now(UTC),
            tags=batch_job.tags,
            run_label=batch_job.run_label,
            run_description=batch_job.run_description,
            git_commit=batch_job.git_commit,
            git_branch=batch_job.git_branch,
        )
        await self.batch_repo.update(completed_batch)

        # Record batch completion metrics
        self.metrics.record_batch_complete(
            completed_batch,
            duration_seconds,
            client_type=self.config.client_type,
            client_version=self.config.client_version,
            eval_mode=self.config.eval_mode,
        )

        # Display results
        if self.config.show_progress:
            self._display_summary(
                completed_batch,
                scorecards,
                failed_tests,
                duration_seconds,
            )

        return completed_batch

    def _response_to_output(self, response: SystemResponse, test_case: TestCase) -> dict[str, Any]:
        """
        Convert SystemResponse to pack-specific system_output dict.

        Args:
            response: SystemResponse from the target system
            test_case: The test case being evaluated

        Returns:
            Dict with pack-specific fields for evaluation
        """
        import json

        if self.config.pack_name == "nl2api":
            return {
                "raw_output": response.raw_output,
                "nl_response": response.nl_response,
            }
        elif self.config.pack_name == "rag":
            # RAG needs different fields - try to extract from raw_output JSON first
            # (used by simulated generator), then fall back to attributes
            rag_data: dict[str, Any] = {}
            try:
                if response.raw_output:
                    rag_data = json.loads(response.raw_output)
            except (json.JSONDecodeError, TypeError):
                pass

            return {
                "response": rag_data.get("response", response.nl_response or response.raw_output),
                "retrieved_doc_ids": rag_data.get(
                    "retrieved_doc_ids", getattr(response, "retrieved_doc_ids", [])
                ),
                "retrieved_chunks": rag_data.get(
                    "retrieved_chunks", getattr(response, "retrieved_chunks", [])
                ),
                "sources": rag_data.get("sources", getattr(response, "sources", [])),
                "context": rag_data.get("context", getattr(response, "context", "")),
            }
        else:
            # Generic fallback
            return {"raw_output": response.raw_output}

    async def _evaluate_one(
        self,
        test_case: TestCase,
        batch_id: str,
        response_simulator: Callable[[TestCase], Any],
    ) -> Scorecard:
        """
        Evaluate a single test case with semaphore control.

        Args:
            test_case: Test case to evaluate
            batch_id: Batch ID for scorecard association
            response_simulator: Function to generate system response

        Returns:
            Scorecard with evaluation results
        """
        async with self.semaphore:
            # Generate response
            response = await response_simulator(test_case)

            # Convert SystemResponse to pack-specific system_output dict
            system_output = self._response_to_output(response, test_case)

            # Create evaluation context with LLM judge for RAG pack
            llm_judge = None
            if self.config.pack_name == "rag":
                from src.rag.evaluation.llm_judge import LLMJudge

                llm_judge = LLMJudge()

            context = EvalContext(
                batch_id=batch_id,
                worker_id=f"batch-{batch_id[:8]}",
                config={},
                llm_judge=llm_judge,
            )

            # Run evaluation through pack
            scorecard = await self.pack.evaluate(
                test_case=test_case,
                system_output=system_output,
                context=context,
            )

            # Extract token usage from response
            input_tokens = response.input_tokens
            output_tokens = response.output_tokens

            # Calculate estimated cost using model-aware pricing
            # Uses client_version as the model identifier for pricing lookup
            estimated_cost = calculate_cost(
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                model=self.config.client_version,
            )

            # Update scorecard with batch_id and client tracking info
            scorecard = scorecard.model_copy(
                update={
                    "batch_id": batch_id,
                    "client_type": self.config.client_type,
                    "client_version": self.config.client_version,
                    "eval_mode": self.config.eval_mode,
                    "input_tokens": input_tokens,
                    "output_tokens": output_tokens,
                    "estimated_cost_usd": estimated_cost,
                }
            )

            # Record metrics with client dimensions
            self.metrics.record_test_result(
                scorecard,
                batch_id,
                list(test_case.metadata.tags) if test_case.metadata.tags else None,
                client_type=self.config.client_type,
                client_version=self.config.client_version,
                eval_mode=self.config.eval_mode,
            )

            # Save scorecard
            await self.scorecard_repo.save(scorecard)

            return scorecard

    def _display_summary(
        self,
        batch_job: BatchJob,
        scorecards: list[Scorecard],
        failed_tests: list[tuple[str, str, float]],
        duration_seconds: float,
    ) -> None:
        """Display batch completion summary."""
        console.print()

        # Show failed tests if any
        if failed_tests:
            console.print(f"[bold red]Failed Tests ({len(failed_tests)}):[/bold red]")
            for test_id, query, score in failed_tests[:10]:  # Show max 10
                console.print(
                    f'  [red]\u2022[/red] {test_id[:8]}... - "{query}" (score: {score:.2f})'
                )
            if len(failed_tests) > 10:
                console.print(f"  ... and {len(failed_tests) - 10} more")
            console.print()

        # Summary table
        table = Table(title=f"Batch Complete: [cyan]{batch_job.batch_id}[/cyan]")
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")

        total = batch_job.total_tests
        passed = batch_job.completed_count
        failed = batch_job.failed_count
        pass_rate = (passed / total * 100) if total > 0 else 0
        avg_score = (
            sum(sc.overall_score for sc in scorecards) / len(scorecards) if scorecards else 0.0
        )

        # Calculate token and cost totals
        total_input_tokens = sum(sc.input_tokens or 0 for sc in scorecards)
        total_output_tokens = sum(sc.output_tokens or 0 for sc in scorecards)
        total_cost = sum(sc.estimated_cost_usd or 0.0 for sc in scorecards)

        table.add_row("Total", str(total))
        table.add_row("Passed", f"[green]{passed}[/green]")
        table.add_row("Failed", f"[red]{failed}[/red]" if failed > 0 else "0")
        table.add_row("Pass Rate", f"{pass_rate:.1f}%")
        table.add_row("Avg Score", f"{avg_score:.2f}")
        table.add_row("Duration", f"{duration_seconds:.1f}s")

        # Only show token/cost info if tokens were tracked
        if total_input_tokens > 0 or total_output_tokens > 0:
            table.add_row("Input Tokens", f"{total_input_tokens:,}")
            table.add_row("Output Tokens", f"{total_output_tokens:,}")
            table.add_row("Est. Cost", f"${total_cost:.4f}")

        console.print(table)
        console.print()
        console.print(
            f"Use '[cyan]eval batch results {batch_job.batch_id}[/cyan]' for detailed results"
        )
        console.print()
