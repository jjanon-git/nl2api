#!/bin/bash
# Run tests for changed files only
# Usage: ./scripts/test-changed.sh [base_branch]
#
# Examples:
#   ./scripts/test-changed.sh          # Compare to main
#   ./scripts/test-changed.sh develop  # Compare to develop

set -e

BASE_BRANCH="${1:-main}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_ROOT"

# Get changed Python files in src/
CHANGED_FILES=$(git diff --name-only "$BASE_BRANCH"...HEAD -- 'src/*.py' 2>/dev/null || \
                git diff --name-only HEAD -- 'src/*.py')

if [ -z "$CHANGED_FILES" ]; then
    # Fallback: check uncommitted changes
    CHANGED_FILES=$(git diff --name-only -- 'src/*.py')
fi

if [ -z "$CHANGED_FILES" ]; then
    echo "No changed Python files in src/"
    exit 0
fi

echo "Changed files:"
echo "$CHANGED_FILES" | sed 's/^/  /'
echo ""

# Map src/ paths to tests/unit/ paths
TEST_PATHS=""
for file in $CHANGED_FILES; do
    # Extract module: src/nl2api/foo.py -> nl2api
    module=$(echo "$file" | sed 's|^src/||' | cut -d'/' -f1)

    # Skip non-module files
    if [ "$module" = "__init__.py" ] || [ "$module" = "$file" ]; then
        continue
    fi

    test_dir="tests/unit/$module"
    if [ -d "$test_dir" ] && [[ ! "$TEST_PATHS" =~ "$test_dir" ]]; then
        TEST_PATHS="$TEST_PATHS $test_dir"
    fi
done

# Deduplicate and trim
TEST_PATHS=$(echo "$TEST_PATHS" | xargs -n1 | sort -u | xargs)

if [ -z "$TEST_PATHS" ]; then
    echo "No matching test directories found"
    exit 0
fi

echo "Running tests in:"
echo "$TEST_PATHS" | tr ' ' '\n' | sed 's/^/  /'
echo ""

# Run pytest
.venv/bin/pytest $TEST_PATHS -v --tb=short -x
