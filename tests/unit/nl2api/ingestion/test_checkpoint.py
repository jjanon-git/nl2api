"""
Unit tests for checkpoint management.
"""

import json
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.nl2api.ingestion.checkpoint import CheckpointManager, IngestionCheckpoint


class TestIngestionCheckpoint:
    """Tests for IngestionCheckpoint dataclass."""

    def test_create_checkpoint(self):
        """Test creating a new checkpoint."""
        checkpoint = IngestionCheckpoint(
            source="gleif",
            started_at=datetime.now(timezone.utc),
        )

        assert checkpoint.source == "gleif"
        assert checkpoint.state == "initialized"
        assert checkpoint.last_offset == 0
        assert checkpoint.is_complete is False
        assert checkpoint.is_failed is False
        assert checkpoint.is_resumable is False

    def test_checkpoint_state_transitions(self):
        """Test checkpoint state transitions."""
        checkpoint = IngestionCheckpoint(
            source="gleif",
            started_at=datetime.now(timezone.utc),
        )

        # Initial state
        assert checkpoint.state == "initialized"
        assert not checkpoint.is_resumable

        # Mark downloading
        checkpoint.mark_downloading()
        assert checkpoint.state == "downloading"
        assert checkpoint.is_resumable

        # Mark loading
        checkpoint.mark_loading(100000)
        assert checkpoint.state == "loading"
        assert checkpoint.total_records == 100000
        assert checkpoint.is_resumable

        # Update progress
        checkpoint.update_progress(50000, imported=50000)
        assert checkpoint.last_offset == 50000
        assert checkpoint.imported_count == 50000
        assert checkpoint.progress_percent == 50.0

        # Mark complete
        checkpoint.mark_complete()
        assert checkpoint.state == "complete"
        assert checkpoint.is_complete
        assert not checkpoint.is_resumable

    def test_checkpoint_failed_state(self):
        """Test checkpoint failure handling."""
        checkpoint = IngestionCheckpoint(
            source="gleif",
            started_at=datetime.now(timezone.utc),
        )
        checkpoint.mark_loading(100000)

        checkpoint.mark_failed("Connection timeout")

        assert checkpoint.state == "failed"
        assert checkpoint.is_failed
        assert checkpoint.error_message == "Connection timeout"
        assert not checkpoint.is_resumable

    def test_checkpoint_serialization(self):
        """Test checkpoint to/from dict."""
        original = IngestionCheckpoint(
            source="sec_edgar",
            started_at=datetime(2026, 1, 21, 10, 0, 0, tzinfo=timezone.utc),
            last_offset=50000,
            state="loading",
            total_records=100000,
            imported_count=50000,
        )

        # Serialize
        d = original.to_dict()
        assert d["source"] == "sec_edgar"
        assert d["last_offset"] == 50000
        assert "2026-01-21" in d["started_at"]

        # Deserialize
        restored = IngestionCheckpoint.from_dict(d)
        assert restored.source == original.source
        assert restored.last_offset == original.last_offset
        assert restored.state == original.state


class TestCheckpointManager:
    """Tests for CheckpointManager."""

    def test_create_and_save_checkpoint(self):
        """Test creating and saving a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            manager = CheckpointManager(checkpoint_dir, "gleif")

            checkpoint = manager.create_new()
            assert checkpoint.source == "gleif"

            manager.save(checkpoint)
            assert manager.path.exists()

    def test_load_checkpoint(self):
        """Test loading an existing checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            manager = CheckpointManager(checkpoint_dir, "gleif")

            # Create and save
            original = manager.create_new()
            original.mark_loading(100000)
            original.update_progress(50000)
            manager.save(original)

            # Load
            loaded = manager.load()
            assert loaded is not None
            assert loaded.source == "gleif"
            assert loaded.last_offset == 50000
            assert loaded.state == "loading"

    def test_load_nonexistent_checkpoint(self):
        """Test loading when no checkpoint exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            manager = CheckpointManager(checkpoint_dir, "gleif")

            loaded = manager.load()
            assert loaded is None

    def test_should_resume(self):
        """Test should_resume logic."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            manager = CheckpointManager(checkpoint_dir, "gleif")

            # No checkpoint
            assert manager.should_resume() is False

            # Initialized checkpoint (not resumable)
            checkpoint = manager.create_new()
            manager.save(checkpoint)
            assert manager.should_resume() is False

            # Loading checkpoint (resumable)
            checkpoint.mark_loading(100000)
            manager.save(checkpoint)
            assert manager.should_resume() is True

            # Complete checkpoint (not resumable)
            checkpoint.mark_complete()
            manager.save(checkpoint)
            assert manager.should_resume() is False

    def test_delete_checkpoint(self):
        """Test deleting a checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            manager = CheckpointManager(checkpoint_dir, "gleif")

            checkpoint = manager.create_new()
            manager.save(checkpoint)
            assert manager.path.exists()

            result = manager.delete()
            assert result is True
            assert not manager.path.exists()

            # Delete again
            result = manager.delete()
            assert result is False

    def test_get_resume_offset(self):
        """Test getting resume offset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            manager = CheckpointManager(checkpoint_dir, "gleif")

            # No checkpoint
            assert manager.get_resume_offset() == 0

            # Resumable checkpoint
            checkpoint = manager.create_new()
            checkpoint.mark_loading(100000)
            checkpoint.update_progress(75000)
            manager.save(checkpoint)

            assert manager.get_resume_offset() == 75000

    def test_atomic_save(self):
        """Test that saves are atomic (temp file then rename)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            manager = CheckpointManager(checkpoint_dir, "gleif")

            checkpoint = manager.create_new()
            checkpoint.mark_loading(100000)
            manager.save(checkpoint)

            # Temp file should not exist
            temp_path = manager.path.with_suffix(".tmp")
            assert not temp_path.exists()

            # Main file should exist and be valid
            assert manager.path.exists()
            with open(manager.path) as f:
                data = json.load(f)
            assert data["source"] == "gleif"

    def test_corrupted_checkpoint_handled(self):
        """Test that corrupted checkpoints are handled gracefully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_dir = Path(tmpdir)
            manager = CheckpointManager(checkpoint_dir, "gleif")

            # Write invalid JSON
            manager.path.parent.mkdir(parents=True, exist_ok=True)
            manager.path.write_text("not valid json {{{")

            # Should return None and clean up
            loaded = manager.load()
            assert loaded is None
            assert not manager.path.exists()
