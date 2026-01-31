#!/bin/bash
# scripts/verify_podman.sh - Verify Podman stack works correctly
#
# Usage:
#   ./scripts/verify_podman.sh          # Full verification
#   ./scripts/verify_podman.sh --quick  # Skip service startup (assumes already running)
#
# Prerequisites:
#   - Podman and podman-compose installed
#   - Podman machine running (macOS):
#       podman machine init --cpus 4 --memory 8192 --disk-size 30
#       podman machine start

set -e

QUICK_MODE=false
if [ "$1" = "--quick" ]; then
    QUICK_MODE=true
fi

echo "=== Podman Stack Verification ==="
echo ""

# 1. Check podman-compose is available
echo "1. Checking podman-compose..."
if ! command -v podman-compose &> /dev/null; then
    echo "   FAIL: podman-compose not found"
    echo "   Install with: brew install podman-compose (macOS)"
    echo "             or: pip install podman-compose"
    exit 1
fi
echo "   OK: $(podman-compose version 2>&1 | head -1)"

# 2. Check podman is available and running
echo "2. Checking podman..."
if ! command -v podman &> /dev/null; then
    echo "   FAIL: podman not found"
    echo "   Install with: brew install podman (macOS)"
    exit 1
fi

# Check if podman machine is running (macOS)
if [[ "$OSTYPE" == "darwin"* ]]; then
    if ! podman machine list 2>/dev/null | grep -q "Currently running"; then
        echo "   WARN: No podman machine running"
        echo "   Start with: podman machine start"
        echo "   Or init first: podman machine init --cpus 4 --memory 4096"
        exit 1
    fi
    echo "   OK: Podman machine running"
else
    echo "   OK: $(podman version --format '{{.Client.Version}}' 2>/dev/null || echo 'version check skipped')"
fi

# 3. Start services (unless quick mode)
if [ "$QUICK_MODE" = false ]; then
    echo "3. Starting services..."
    podman-compose up -d
    echo "   OK: Services started"
else
    echo "3. Skipping service startup (--quick mode)"
fi

# 4. Wait for health checks (60s timeout)
echo "4. Waiting for health checks (up to 60s)..."
MAX_WAIT=12  # 12 * 5s = 60s
for i in $(seq 1 $MAX_WAIT); do
    # Count healthy containers (podman ps format may differ slightly)
    HEALTHY=$(podman ps --filter "health=healthy" --format "{{.Names}}" 2>/dev/null | wc -l | tr -d ' ')

    # We have 9 services total, but not all have health checks
    # Services with health checks: postgres, redis, entity-mcp-server, entity-resolution
    # Services without health checks: otel-collector, jaeger, prometheus, grafana
    echo "   Healthy containers: $HEALTHY (checking...)"

    # Check if core services are healthy
    POSTGRES_HEALTHY=$(podman ps --filter "name=nl2api-db" --filter "health=healthy" --format "{{.Names}}" 2>/dev/null | wc -l | tr -d ' ')
    REDIS_HEALTHY=$(podman ps --filter "name=nl2api-redis" --filter "health=healthy" --format "{{.Names}}" 2>/dev/null | wc -l | tr -d ' ')

    if [ "$POSTGRES_HEALTHY" -ge 1 ] && [ "$REDIS_HEALTHY" -ge 1 ]; then
        echo "   OK: Core services healthy"
        break
    fi

    if [ "$i" -eq "$MAX_WAIT" ]; then
        echo "   WARN: Timeout waiting for health checks"
        echo "   Continuing with verification anyway..."
    fi
    sleep 5
done

# 5. Verify each service
echo "5. Verifying services..."

# PostgreSQL
echo "   - PostgreSQL..."
if podman exec nl2api-db psql -U nl2api -c "SELECT 1" > /dev/null 2>&1; then
    echo "     OK: Connection successful"
else
    echo "     FAIL: Cannot connect to PostgreSQL"
    exit 1
fi

# pgvector extension
echo "   - pgvector extension..."
if podman exec nl2api-db psql -U nl2api -c "CREATE EXTENSION IF NOT EXISTS vector" > /dev/null 2>&1; then
    echo "     OK: pgvector available"
else
    echo "     FAIL: pgvector extension not available"
    exit 1
fi

# Redis
echo "   - Redis..."
if podman exec nl2api-redis redis-cli ping 2>/dev/null | grep -q PONG; then
    echo "     OK: Redis responding"
else
    echo "     FAIL: Redis not responding"
    exit 1
fi

# OTEL Collector
echo "   - OTEL Collector..."
if curl -sf http://localhost:8888/metrics > /dev/null 2>&1; then
    echo "     OK: OTEL Collector metrics endpoint"
else
    echo "     WARN: OTEL Collector metrics endpoint not responding (may still be starting)"
fi

# Prometheus
echo "   - Prometheus..."
if curl -sf http://localhost:9090/-/healthy > /dev/null 2>&1; then
    echo "     OK: Prometheus healthy"
else
    echo "     WARN: Prometheus not responding (may still be starting)"
fi

# Grafana
echo "   - Grafana..."
if curl -sf http://localhost:3000/api/health > /dev/null 2>&1; then
    echo "     OK: Grafana healthy"
else
    echo "     WARN: Grafana not responding (may still be starting)"
fi

# Jaeger
echo "   - Jaeger..."
if curl -sf http://localhost:16686/ > /dev/null 2>&1; then
    echo "     OK: Jaeger UI accessible"
else
    echo "     WARN: Jaeger not responding (may still be starting)"
fi

# Entity Resolution HTTP Service (if running)
echo "   - Entity Resolution HTTP..."
if curl -sf http://localhost:8085/health > /dev/null 2>&1; then
    echo "     OK: Entity Resolution service healthy"
else
    echo "     INFO: Entity Resolution service not running (optional)"
fi

# Entity MCP Server (if running)
echo "   - Entity MCP Server..."
if curl -sf http://localhost:8080/health > /dev/null 2>&1; then
    echo "     OK: Entity MCP server healthy"
else
    echo "     INFO: Entity MCP server not running (optional)"
fi

# 6. Summary
echo ""
echo "=== Verification Complete ==="
echo ""
echo "Running containers:"
podman ps --format "table {{.Names}}\t{{.Status}}" 2>/dev/null || podman ps

echo ""
echo "Next steps:"
echo "  - Run integration tests: pytest tests/integration/ -v"
echo "  - View Grafana: http://localhost:3000 (admin/admin)"
echo "  - View Jaeger: http://localhost:16686"
echo "  - View Prometheus: http://localhost:9090"
