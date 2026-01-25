# Entity Resolution Service

Standalone service for resolving financial entity names to identifiers (RICs, tickers, CUSIPs).

## Features

- **Entity Extraction**: Extract company names and tickers from natural language queries
- **Entity Resolution**: Resolve names to standardized identifiers via database lookup
- **OpenFIGI Fallback**: External API fallback for unresolved entities
- **Multi-level Caching**: L1 in-memory + L2 Redis caching
- **Circuit Breaker**: Resilient external API handling
- **Multiple Transports**: HTTP/REST, MCP (planned), stdio (planned)

## Quick Start

### Using Docker

```bash
# Build
docker build -t entity-resolution-service .

# Run (requires PostgreSQL)
docker run -p 8085:8085 \
  -e ENTITY_POSTGRES_URL=postgresql://user:pass@host:5432/db \
  entity-resolution-service
```

### Using Python

```bash
# Install with HTTP transport
pip install ".[http,redis]"

# Run
python -m src.services.entity_resolution --port 8085
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness probe |
| `/ready` | GET | Readiness probe (checks DB, cache) |
| `/api/resolve` | POST | Resolve single entity |
| `/api/resolve/batch` | POST | Resolve multiple entities |
| `/api/extract` | POST | Extract and resolve from query |
| `/api/stats` | GET | Service statistics |

### Resolve Single Entity

```bash
curl -X POST http://localhost:8085/api/resolve \
  -H "Content-Type: application/json" \
  -d '{"entity": "Apple", "entity_type": "company"}'
```

Response:
```json
{
  "found": true,
  "original": "Apple",
  "identifier": "AAPL.O",
  "entity_type": "company",
  "confidence": 0.95,
  "alternatives": ["AAPL.OQ"],
  "metadata": {"source": "database"}
}
```

### Extract from Query

```bash
curl -X POST http://localhost:8085/api/extract \
  -H "Content-Type: application/json" \
  -d '{"query": "What is Apple PE ratio compared to Microsoft?"}'
```

Response:
```json
{
  "query": "What is Apple PE ratio compared to Microsoft?",
  "entities": {
    "AAPL.O": "Apple",
    "MSFT.O": "Microsoft"
  },
  "count": 2
}
```

## Configuration

All configuration via environment variables with `ENTITY_` prefix:

| Variable | Default | Description |
|----------|---------|-------------|
| `ENTITY_HOST` | `0.0.0.0` | Server bind host |
| `ENTITY_PORT` | `8085` | Server bind port |
| `ENTITY_POSTGRES_URL` | required | PostgreSQL connection URL |
| `ENTITY_REDIS_ENABLED` | `true` | Enable Redis caching |
| `ENTITY_REDIS_URL` | `redis://localhost:6379` | Redis connection URL |
| `ENTITY_OPENFIGI_ENABLED` | `true` | Enable OpenFIGI fallback |
| `ENTITY_OPENFIGI_API_KEY` | `""` | OpenFIGI API key (optional) |
| `ENTITY_TIMEOUT_SECONDS` | `5.0` | Request timeout |
| `ENTITY_LOG_LEVEL` | `INFO` | Log level |

## Database Schema

Requires `entity_aliases` table:

```sql
CREATE TABLE entity_aliases (
    id SERIAL PRIMARY KEY,
    alias VARCHAR(255) NOT NULL,
    canonical_name VARCHAR(255) NOT NULL,
    ric VARCHAR(50),
    ticker VARCHAR(20),
    entity_type VARCHAR(50) DEFAULT 'company',
    confidence FLOAT DEFAULT 1.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_entity_aliases_alias ON entity_aliases(LOWER(alias));
CREATE INDEX idx_entity_aliases_ric ON entity_aliases(ric);
```

## Architecture

```
src/services/entity_resolution/
├── __init__.py          # Package exports
├── __main__.py          # CLI entry point
├── config.py            # Configuration (pydantic-settings)
├── resilience.py        # Circuit breaker, retry logic
├── core/
│   ├── models.py        # Data models (ResolvedEntity)
│   ├── protocols.py     # Protocol definitions
│   ├── extractor.py     # Entity extraction from text
│   ├── resolver.py      # Main resolution logic
│   └── openfigi.py      # OpenFIGI API client
├── adapters/
│   ├── database.py      # PostgreSQL adapter
│   └── cache.py         # Redis cache adapter
└── transports/
    └── http/
        └── app.py       # FastAPI application
```

## Development

```bash
# Install dev dependencies
pip install ".[dev]"

# Run tests
pytest tests/ -v

# Lint
ruff check .
```
