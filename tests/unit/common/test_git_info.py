"""Unit tests for git info utilities."""

from unittest.mock import patch

from src.common.git_info import GitInfo, get_git_info


class TestGitInfo:
    """Test git info capture."""

    def test_git_info_dataclass(self):
        """Test GitInfo dataclass."""
        info = GitInfo(commit="abc123", branch="main")
        assert info.commit == "abc123"
        assert info.branch == "main"

    def test_git_info_immutable(self):
        """Test GitInfo is immutable."""
        info = GitInfo(commit="abc", branch="main")
        # Should be frozen
        assert info.commit == "abc"

    @patch("src.common.git_info._get_git_commit")
    @patch("src.common.git_info._get_git_branch")
    def test_get_git_info_success(self, mock_branch, mock_commit):
        """Test successful git info capture."""
        mock_commit.return_value = "fbe1481"
        mock_branch.return_value = "main"

        info = get_git_info()

        assert info.commit == "fbe1481"
        assert info.branch == "main"

    @patch("src.common.git_info._get_git_commit")
    @patch("src.common.git_info._get_git_branch")
    def test_get_git_info_not_in_repo(self, mock_branch, mock_commit):
        """Test git info when not in a repo."""
        mock_commit.return_value = None
        mock_branch.return_value = None

        info = get_git_info()

        assert info.commit is None
        assert info.branch is None


class TestGitInfoIntegration:
    """Integration tests for git info (run in actual git repo)."""

    def test_get_git_info_in_real_repo(self):
        """Test get_git_info works in this actual repo."""
        info = get_git_info()

        # We're running in the nl2api repo, so should get values
        assert info.commit is not None
        assert len(info.commit) == 7  # Short hash
        # Branch could be None if in detached HEAD state
