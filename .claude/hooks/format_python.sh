#!/bin/bash
# Post-tool hook: Auto-format Python files with ruff after Edit/Write
# This prevents agents from getting confused when ruff reformats their changes

# Read the file path from JSON input (piped from Claude Code)
file_path=$(jq -r '.tool_input.file_path')

# Only format Python files
if [[ "$file_path" == *.py ]]; then
    # Use project-local ruff
    if [[ -x "$CLAUDE_PROJECT_DIR/.venv/bin/ruff" ]]; then
        "$CLAUDE_PROJECT_DIR/.venv/bin/ruff" format --quiet "$file_path"
    elif command -v ruff &> /dev/null; then
        ruff format --quiet "$file_path"
    fi
fi

exit 0
