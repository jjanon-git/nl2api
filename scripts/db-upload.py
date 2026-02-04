#!/usr/bin/env python3
"""Upload database dump to GitHub Release.

Usage:
    python scripts/db-upload.py exports/evalkit_all_20260203.dump.gz
    python scripts/db-upload.py exports/evalkit_rag_20260203.dump.gz --tag data-rag-v1

Prerequisites:
    - GitHub CLI installed and authenticated: gh auth login
"""

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


def check_gh_cli() -> bool:
    """Check if GitHub CLI is installed and authenticated."""
    try:
        result = subprocess.run(
            ["gh", "auth", "status"],
            capture_output=True,
            text=True,
        )
        return result.returncode == 0
    except FileNotFoundError:
        return False


def get_repo_info() -> tuple[str, str] | None:
    """Get owner/repo from git remote."""
    try:
        result = subprocess.run(
            ["gh", "repo", "view", "--json", "owner,name", "-q", '.owner.login + "/" + .name'],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError:
        return None


def release_exists(tag: str) -> bool:
    """Check if a release with this tag already exists."""
    result = subprocess.run(
        ["gh", "release", "view", tag],
        capture_output=True,
    )
    return result.returncode == 0


def create_release(dump_path: Path, tag: str, title: str, notes: str) -> bool:
    """Create GitHub release and upload the dump file."""
    cmd = [
        "gh",
        "release",
        "create",
        tag,
        "--title",
        title,
        "--notes",
        notes,
        str(dump_path),
    ]

    print(f"Creating release {tag}...")
    print(f"Uploading {dump_path.name} ({dump_path.stat().st_size / (1024 * 1024):.1f} MB)...")

    result = subprocess.run(cmd)
    return result.returncode == 0


def upload_to_existing(dump_path: Path, tag: str) -> bool:
    """Upload file to an existing release."""
    cmd = [
        "gh",
        "release",
        "upload",
        tag,
        str(dump_path),
        "--clobber",  # Overwrite if exists
    ]

    print(f"Uploading to existing release {tag}...")
    result = subprocess.run(cmd)
    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(description="Upload database dump to GitHub Release")
    parser.add_argument(
        "dump_file",
        type=Path,
        help="Path to the .dump.gz file to upload",
    )
    parser.add_argument(
        "--tag",
        help="Release tag (default: auto-generated from filename)",
    )
    parser.add_argument(
        "--title",
        help="Release title (default: auto-generated)",
    )
    args = parser.parse_args()

    # Validate dump file
    if not args.dump_file.exists():
        print(f"Error: File not found: {args.dump_file}", file=sys.stderr)
        sys.exit(1)

    if not args.dump_file.suffix == ".gz":
        print("Warning: File doesn't appear to be gzipped", file=sys.stderr)

    # Check GitHub CLI
    if not check_gh_cli():
        print("Error: GitHub CLI not installed or not authenticated.", file=sys.stderr)
        print("  Install: https://cli.github.com/", file=sys.stderr)
        print("  Authenticate: gh auth login", file=sys.stderr)
        sys.exit(1)

    # Get repo info
    repo = get_repo_info()
    if not repo:
        print("Error: Could not determine repository. Run from repo root.", file=sys.stderr)
        sys.exit(1)

    print(f"Repository: {repo}")

    # Generate tag from filename if not provided
    # evalkit_all_20260203_120000.dump.gz -> data-all-20260203
    if not args.tag:
        stem = args.dump_file.stem.replace(".dump", "")  # evalkit_all_20260203_120000
        parts = stem.split("_")
        if len(parts) >= 3:
            table_group = parts[1]  # all, rag, entities, etc.
            date = parts[2][:8]  # 20260203
            args.tag = f"data-{table_group}-{date}"
        else:
            args.tag = f"data-{datetime.now().strftime('%Y%m%d')}"

    # Generate title if not provided
    if not args.title:
        size_mb = args.dump_file.stat().st_size / (1024 * 1024)
        args.title = f"Database snapshot: {args.dump_file.name} ({size_mb:.0f} MB)"

    # Generate notes
    notes = f"""Database dump for evalkit.

**File:** `{args.dump_file.name}`
**Size:** {args.dump_file.stat().st_size / (1024 * 1024):.1f} MB
**Created:** {datetime.now().isoformat()}

## Restore Instructions

```bash
# Download
gh release download {args.tag} -p '*.dump.gz'

# Restore
python scripts/db-restore.py {args.dump_file.name}
```
"""

    # Create or update release
    if release_exists(args.tag):
        print(f"Release {args.tag} already exists.")
        response = input("Upload to existing release? [y/N] ")
        if response.lower() != "y":
            print("Aborted.")
            sys.exit(0)
        success = upload_to_existing(args.dump_file, args.tag)
    else:
        success = create_release(args.dump_file, args.tag, args.title, notes)

    if success:
        print()
        print("Upload complete!")
        print(f"View release: gh release view {args.tag}")
        print(f"Download on another machine: gh release download {args.tag}")
    else:
        print("Upload failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
