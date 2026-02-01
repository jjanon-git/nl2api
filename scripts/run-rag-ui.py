#!/usr/bin/env python3
"""
Launch the RAG Question UI.

Usage:
    python scripts/run_rag_ui.py
    python scripts/run_rag_ui.py --port 8502
    python scripts/run_rag_ui.py --debug
"""

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Launch the RAG Question UI")
    parser.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Port to run the Streamlit server on (default: 8501)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Run in debug mode with auto-reload",
    )
    args = parser.parse_args()

    app_path = PROJECT_ROOT / "src" / "rag_ui" / "app.py"

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--server.headless",
        "true",
    ]

    if args.debug:
        cmd.extend(["--server.runOnSave", "true"])

    print(f"Starting RAG UI on http://localhost:{args.port}")
    print("Press Ctrl+C to stop")
    print()

    try:
        subprocess.run(cmd, cwd=PROJECT_ROOT)
    except KeyboardInterrupt:
        print("\nShutting down...")


if __name__ == "__main__":
    main()
