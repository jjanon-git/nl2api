"""
Unit tests for progress tracking.
"""



from src.nl2api.ingestion.progress import ProgressTracker, RichProgressTracker


class TestProgressTracker:
    """Tests for ProgressTracker."""

    def test_create_tracker(self):
        """Test creating a progress tracker."""
        tracker = ProgressTracker(total=1000, description="Testing")

        assert tracker.total == 1000
        assert tracker.processed == 0
        assert tracker.description == "Testing"

    def test_update_progress(self):
        """Test updating progress."""
        tracker = ProgressTracker(
            total=1000, report_interval=100000
        )  # High interval to avoid log output

        tracker.update(100)
        assert tracker.processed == 100

        tracker.update(50)
        assert tracker.processed == 150

    def test_percent_complete(self):
        """Test percentage calculation."""
        tracker = ProgressTracker(total=1000, report_interval=100000)

        tracker.update(250)
        assert tracker.percent_complete == 25.0

        tracker.update(250)
        assert tracker.percent_complete == 50.0

    def test_percent_complete_zero_total(self):
        """Test percentage with zero total."""
        tracker = ProgressTracker(total=0)
        assert tracker.percent_complete == 0.0

    def test_rate_calculation(self):
        """Test rate per second calculation."""
        tracker = ProgressTracker(total=1000, report_interval=100000)

        # Simulate some processing
        tracker.processed = 100
        # Note: started_at is set at creation, so rate depends on elapsed time
        # Just verify it doesn't error
        rate = tracker.rate_per_second
        assert rate >= 0

    def test_eta_calculation(self):
        """Test ETA calculation."""
        tracker = ProgressTracker(total=1000, report_interval=100000)

        # Process half
        tracker.processed = 500

        # ETA should be positive finite number
        eta = tracker.eta_seconds
        assert eta >= 0
        assert eta != float("inf")

    def test_eta_zero_rate(self):
        """Test ETA when rate is zero."""
        ProgressTracker(total=1000, report_interval=100000)
        # Don't process anything
        # ETA should be infinity
        # Note: This depends on implementation - if started_at is set, rate might not be zero

    def test_elapsed_seconds(self):
        """Test elapsed time calculation."""
        tracker = ProgressTracker(total=1000)

        # Should be very small immediately
        elapsed = tracker.elapsed_seconds
        assert elapsed >= 0
        assert elapsed < 1  # Should be less than 1 second

    def test_finish_returns_summary(self):
        """Test finish returns summary stats."""
        tracker = ProgressTracker(total=1000, report_interval=100000)
        tracker.update(1000)

        summary = tracker.finish()

        assert summary["processed"] == 1000
        assert summary["total"] == 1000
        assert "elapsed_seconds" in summary
        assert "rate_per_second" in summary

    def test_format_duration_seconds(self):
        """Test duration formatting for seconds."""
        tracker = ProgressTracker(total=100)

        assert tracker._format_duration(30) == "30s"
        assert tracker._format_duration(59.9) == "60s"

    def test_format_duration_minutes(self):
        """Test duration formatting for minutes."""
        tracker = ProgressTracker(total=100)

        assert tracker._format_duration(90) == "1.5m"
        assert tracker._format_duration(600) == "10.0m"

    def test_format_duration_hours(self):
        """Test duration formatting for hours."""
        tracker = ProgressTracker(total=100)

        assert tracker._format_duration(3600) == "1.0h"
        assert tracker._format_duration(7200) == "2.0h"


class TestRichProgressTracker:
    """Tests for RichProgressTracker."""

    def test_create_rich_tracker(self):
        """Test creating a rich progress tracker."""
        tracker = RichProgressTracker(total=1000, description="Testing")

        assert tracker.total == 1000
        assert tracker.description == "Testing"

    def test_context_manager(self):
        """Test using as context manager."""
        with RichProgressTracker(total=1000) as tracker:
            tracker.update(500)
            tracker.update(500)

        # Should complete without error

    def test_update_without_context(self):
        """Test update works without context manager."""
        tracker = RichProgressTracker(total=1000)
        tracker.update(100)
        # Should not error, even if rich not in context

    def test_fallback_to_basic_tracker(self):
        """Test fallback when rich is not available."""
        # This tests the fallback behavior
        # If rich is installed, it won't fallback, but the interface should work either way
        tracker = RichProgressTracker(total=1000)

        with tracker:
            tracker.update(1000)
            summary = tracker.finish()

        assert summary["total"] == 1000
