#!/bin/bash
# Run full CI pipeline locally
# Usage: ./scripts/ci-local.sh

set -e  # Exit on first failure

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo_step() {
    echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${YELLOW}▶ $1${NC}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

echo_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

echo_fail() {
    echo -e "${RED}✗ $1${NC}"
}

# Track results
LINT_OK=0
FORMAT_OK=0
MYPY_OK=0
SECURITY_OK=0
TEST_OK=0
BUILD_OK=0

cd "$(dirname "$0")/.."

# 1. Lint
echo_step "Lint (ruff check)"
if .venv/bin/ruff check .; then
    LINT_OK=1
    echo_success "Lint passed"
else
    echo_fail "Lint failed"
fi

# 2. Format check
echo_step "Format Check (ruff format --check)"
if .venv/bin/ruff format --check .; then
    FORMAT_OK=1
    echo_success "Format check passed"
else
    echo_fail "Format check failed (run: ruff format .)"
fi

# 3. Type check (non-blocking in CI)
echo_step "Type Check (mypy)"
if .venv/bin/mypy src/ --ignore-missing-imports 2>/dev/null; then
    MYPY_OK=1
    echo_success "Type check passed"
else
    echo -e "${YELLOW}⚠ Type check has errors (non-blocking)${NC}"
fi

# 4. Security scan (non-blocking in CI)
echo_step "Security Scan (pip-audit)"
if .venv/bin/pip-audit --strict --progress-spinner off 2>/dev/null; then
    SECURITY_OK=1
    echo_success "Security scan passed"
else
    echo -e "${YELLOW}⚠ Security scan has warnings (non-blocking)${NC}"
fi

# 5. Unit tests
echo_step "Unit Tests (pytest)"
# Check if docker services are running
if ! docker compose ps --status running 2>/dev/null | grep -q postgres; then
    echo -e "${YELLOW}Starting docker services...${NC}"
    docker compose up -d
    sleep 3
fi

if .venv/bin/pytest tests/unit/ --tb=short -x -q; then
    TEST_OK=1
    echo_success "Unit tests passed"
else
    echo_fail "Unit tests failed"
fi

# 6. Build
echo_step "Build Package"
if .venv/bin/python -m build --quiet 2>/dev/null; then
    BUILD_OK=1
    echo_success "Build passed"
else
    echo_fail "Build failed"
fi

# Summary
echo_step "Summary"
echo "Lint:     $([ $LINT_OK -eq 1 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
echo "Format:   $([ $FORMAT_OK -eq 1 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
echo "Type:     $([ $MYPY_OK -eq 1 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${YELLOW}WARN${NC}")"
echo "Security: $([ $SECURITY_OK -eq 1 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${YELLOW}WARN${NC}")"
echo "Tests:    $([ $TEST_OK -eq 1 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"
echo "Build:    $([ $BUILD_OK -eq 1 ] && echo -e "${GREEN}PASS${NC}" || echo -e "${RED}FAIL${NC}")"

# Exit with failure if required checks failed
if [ $LINT_OK -eq 0 ] || [ $FORMAT_OK -eq 0 ] || [ $TEST_OK -eq 0 ] || [ $BUILD_OK -eq 0 ]; then
    echo -e "\n${RED}CI failed${NC}"
    exit 1
fi

echo -e "\n${GREEN}CI passed${NC}"
