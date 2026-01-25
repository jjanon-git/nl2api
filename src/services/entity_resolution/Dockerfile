# Entity Resolution Service
# Build: docker build -t entity-resolution-service .
# Run: docker run -p 8085:8085 -e ENTITY_POSTGRES_URL=... entity-resolution-service

FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy only what's needed for install
COPY src/services/entity_resolution/pyproject.toml .
COPY src/services/entity_resolution/README.md .
COPY src/services/entity_resolution src/services/entity_resolution

# Install the package with HTTP and Redis extras
RUN pip install --no-cache-dir ".[http,redis]"

# Production image
FROM python:3.11-slim

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application code
COPY src/services/entity_resolution src/services/entity_resolution

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Default environment
ENV ENTITY_HOST=0.0.0.0
ENV ENTITY_PORT=8085
ENV ENTITY_LOG_LEVEL=INFO

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8085/health')" || exit 1

EXPOSE 8085

# Run the service
CMD ["python", "-m", "src.services.entity_resolution", "--transport", "http"]
