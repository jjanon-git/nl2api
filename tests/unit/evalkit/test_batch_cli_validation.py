"""
Unit tests for batch CLI validation logic.

Tests the safeguards that ensure evaluations are run properly:
1. Tag is required for all packs
2. Simulated mode is disabled
3. RAG pack requires --tag rag
"""

from typer.testing import CliRunner

from src.evalkit.cli.main import app

runner = CliRunner()


class TestBatchCLIValidation:
    """Tests for batch CLI validation safeguards."""

    def test_tag_required_for_nl2api_pack(self):
        """Test that --tag is required for nl2api pack."""
        result = runner.invoke(
            app,
            ["batch", "run", "--pack", "nl2api", "--label", "test"],
            catch_exceptions=False,
        )

        assert result.exit_code == 1
        assert "--tag is required for all batch evaluations" in result.stdout

    def test_tag_required_for_rag_pack(self):
        """Test that --tag is required for rag pack."""
        result = runner.invoke(
            app,
            ["batch", "run", "--pack", "rag", "--label", "test"],
            catch_exceptions=False,
        )

        assert result.exit_code == 1
        assert "--tag is required for all batch evaluations" in result.stdout

    def test_rag_pack_requires_rag_tag(self):
        """Test that rag pack specifically requires --tag rag."""
        result = runner.invoke(
            app,
            ["batch", "run", "--pack", "rag", "--tag", "lookups", "--label", "test"],
            catch_exceptions=False,
        )

        assert result.exit_code == 1
        assert "rag pack requires --tag rag" in result.stdout

    def test_simulated_mode_disabled(self):
        """Test that simulated mode is disabled."""
        result = runner.invoke(
            app,
            [
                "batch",
                "run",
                "--pack",
                "nl2api",
                "--tag",
                "lookups",
                "--label",
                "test",
                "--mode",
                "simulated",
            ],
            catch_exceptions=False,
        )

        assert result.exit_code == 1
        assert "Simulated mode is disabled" in result.stdout
        assert "100% pass rates" in result.stdout

    def test_multiple_tags_allowed(self):
        """Test that multiple tags can be provided."""
        # With multiple tags including required one, validation should pass
        # (will fail later due to DB connection, but validation passes)
        result = runner.invoke(
            app,
            [
                "batch",
                "run",
                "--pack",
                "nl2api",
                "--tag",
                "lookups",
                "--tag",
                "easy",
                "--label",
                "test",
            ],
        )

        # Should NOT fail on tag validation
        assert "--tag is required" not in result.stdout
        # May fail later for other reasons (DB connection, etc.) but that's OK

    def test_rag_with_additional_tags_allowed(self):
        """Test that rag pack allows additional tags as long as 'rag' is included."""
        result = runner.invoke(
            app,
            ["batch", "run", "--pack", "rag", "--tag", "rag", "--tag", "easy", "--label", "test"],
        )

        # Should NOT fail on tag validation
        assert "--tag is required" not in result.stdout
        assert "rag pack requires --tag rag" not in result.stdout


class TestBatchCLIErrorMessages:
    """Tests for helpful error messages."""

    def test_nl2api_missing_tag_shows_common_tags(self):
        """Test that nl2api pack shows common tags when tag is missing."""
        result = runner.invoke(
            app,
            ["batch", "run", "--pack", "nl2api", "--label", "test"],
            catch_exceptions=False,
        )

        assert "entity_resolution" in result.stdout
        assert "lookups" in result.stdout
        assert "temporal" in result.stdout
        assert "screening" in result.stdout

    def test_rag_missing_tag_shows_rag_example(self):
        """Test that rag pack shows rag-specific example when tag is missing."""
        result = runner.invoke(
            app,
            ["batch", "run", "--pack", "rag", "--label", "test"],
            catch_exceptions=False,
        )

        assert "--pack rag --tag rag" in result.stdout

    def test_simulated_mode_shows_alternatives(self):
        """Test that simulated mode error shows alternatives."""
        result = runner.invoke(
            app,
            [
                "batch",
                "run",
                "--pack",
                "nl2api",
                "--tag",
                "lookups",
                "--label",
                "test",
                "--mode",
                "simulated",
            ],
            catch_exceptions=False,
        )

        assert "pytest tests/unit/evalkit/" in result.stdout
        assert "--mode resolver" in result.stdout
        assert "--mode orchestrator" in result.stdout
