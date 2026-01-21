"""
Progress Tracking for Entity Ingestion

Provides progress reporting and ETA estimation for long-running ingestion jobs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class ProgressTracker:
    """Tracks and reports ingestion progress."""

    total: int
    report_interval: int = 10000
    description: str = "Processing"

    # Internal state
    processed: int = 0
    started_at: float = field(default_factory=time.time)
    last_report_at: float = field(default_factory=time.time)
    last_report_count: int = 0

    def update(self, count: int = 1) -> None:
        """
        Update progress counter.

        Args:
            count: Number of items processed since last update
        """
        self.processed += count

        if self.processed - self.last_report_count >= self.report_interval:
            self._report()

    def _report(self) -> None:
        """Log progress report."""
        now = time.time()
        elapsed = now - self.started_at
        interval_elapsed = now - self.last_report_at

        # Calculate rates
        overall_rate = self.processed / elapsed if elapsed > 0 else 0
        interval_rate = (
            (self.processed - self.last_report_count) / interval_elapsed
            if interval_elapsed > 0
            else 0
        )

        # Calculate ETA
        remaining = self.total - self.processed
        eta_seconds = remaining / overall_rate if overall_rate > 0 else 0

        # Calculate percentage
        percent = (self.processed / self.total * 100) if self.total > 0 else 0

        logger.info(
            "%s: %d/%d (%.1f%%) | Rate: %.0f/sec | ETA: %s",
            self.description,
            self.processed,
            self.total,
            percent,
            interval_rate,
            self._format_duration(eta_seconds),
        )

        self.last_report_at = now
        self.last_report_count = self.processed

    def _format_duration(self, seconds: float) -> str:
        """Format duration in human-readable format."""
        if seconds < 60:
            return f"{seconds:.0f}s"
        elif seconds < 3600:
            minutes = seconds / 60
            return f"{minutes:.1f}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def finish(self) -> dict:
        """
        Mark processing as complete and return summary.

        Returns:
            Summary statistics
        """
        elapsed = time.time() - self.started_at
        rate = self.processed / elapsed if elapsed > 0 else 0

        summary = {
            "processed": self.processed,
            "total": self.total,
            "elapsed_seconds": elapsed,
            "rate_per_second": rate,
        }

        logger.info(
            "%s complete: %d records in %s (%.0f/sec)",
            self.description,
            self.processed,
            self._format_duration(elapsed),
            rate,
        )

        return summary

    @property
    def percent_complete(self) -> float:
        """Get completion percentage."""
        return (self.processed / self.total * 100) if self.total > 0 else 0

    @property
    def elapsed_seconds(self) -> float:
        """Get elapsed time in seconds."""
        return time.time() - self.started_at

    @property
    def rate_per_second(self) -> float:
        """Get overall processing rate."""
        elapsed = self.elapsed_seconds
        return self.processed / elapsed if elapsed > 0 else 0

    @property
    def eta_seconds(self) -> float:
        """Get estimated time to completion in seconds."""
        rate = self.rate_per_second
        if rate <= 0:
            return float("inf")
        remaining = self.total - self.processed
        return remaining / rate


class RichProgressTracker:
    """
    Progress tracker with rich console output.

    Uses rich library for prettier progress bars if available.
    Falls back to basic ProgressTracker otherwise.
    """

    def __init__(
        self,
        total: int,
        description: str = "Processing",
        show_speed: bool = True,
        show_eta: bool = True,
    ):
        """
        Initialize rich progress tracker.

        Args:
            total: Total number of items
            description: Description shown in progress bar
            show_speed: Show processing speed
            show_eta: Show estimated time remaining
        """
        self.total = total
        self.description = description
        self.show_speed = show_speed
        self.show_eta = show_eta

        self._progress = None
        self._task = None
        self._fallback = None

        try:
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

            columns = [
                SpinnerColumn(),
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=40),
                TaskProgressColumn(),
                MofNCompleteColumn(),
            ]

            if show_speed:
                columns.append(TextColumn("[green]{task.speed:,.0f}/s"))

            if show_eta:
                columns.append(TimeRemainingColumn())

            columns.append(TimeElapsedColumn())

            self._progress = Progress(*columns)

        except ImportError:
            # Fall back to basic tracker
            self._fallback = ProgressTracker(
                total=total,
                description=description,
            )

    def __enter__(self):
        if self._progress:
            self._progress.__enter__()
            self._task = self._progress.add_task(
                self.description,
                total=self.total,
            )
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self._progress:
            self._progress.__exit__(exc_type, exc_val, exc_tb)
        return False

    def update(self, count: int = 1) -> None:
        """Update progress by count."""
        if self._progress and self._task is not None:
            self._progress.update(self._task, advance=count)
        elif self._fallback:
            self._fallback.update(count)

    def finish(self) -> dict:
        """Mark complete and return summary."""
        if self._fallback:
            return self._fallback.finish()

        return {
            "processed": self.total,
            "total": self.total,
        }
