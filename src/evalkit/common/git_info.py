"""
Git information capture utilities.

Provides functions to capture git commit hash and branch from CWD.
Gracefully handles non-git directories.
"""

import logging
import subprocess
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GitInfo:
    """Git repository information."""

    commit: str | None
    branch: str | None


def get_git_info() -> GitInfo:
    """
    Capture git commit hash and branch from current working directory.

    Returns:
        GitInfo with commit and branch (both None if not in a git repo)
    """
    commit = _get_git_commit()
    branch = _get_git_branch()
    return GitInfo(commit=commit, branch=branch)


def _get_git_commit() -> str | None:
    """Get the current git commit hash (short form)."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug(f"Could not get git commit: {e}")
        return None


def _get_git_branch() -> str | None:
    """Get the current git branch name."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode == 0:
            branch = result.stdout.strip()
            # HEAD means detached state
            return branch if branch != "HEAD" else None
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        logger.debug(f"Could not get git branch: {e}")
        return None
