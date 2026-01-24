# Planning Documents

## Naming Convention

```
{component}-{seq}-{short-name}.md
```

| Prefix | Scope |
|--------|-------|
| `nl2api-` | Core query translation, entity resolution, agents, MCP |
| `rag-` | Retrieval, ingestion, SEC filings, embeddings |
| `eval-` | Evaluation framework, packs, metrics, distributed workers |
| `infra-` | Observability, CI/CD, deployment |
| `meta-` | Release audits, sprints, roadmaps |

## Active Plans

### NL2API (Core Query Translation)
- [nl2api-01-entity-resolution-bootstrap](nl2api-01-entity-resolution-bootstrap.md) - Initial entity resolution implementation
- [nl2api-02-entity-resolution-database](nl2api-02-entity-resolution-database.md) - Database-backed entity expansion
- [nl2api-03-entity-resolution-edge-cases](nl2api-03-entity-resolution-edge-cases.md) - Edge case handling
- [nl2api-04-mcp-dual-mode](nl2api-04-mcp-dual-mode.md) - MCP server dual-mode migration
- [nl2api-05-orchestrator-refactor](nl2api-05-orchestrator-refactor.md) - Orchestrator god class refactoring

### RAG (Retrieval & Ingestion)
- [rag-01-infrastructure](rag-01-infrastructure.md) - RAG infrastructure setup
- [rag-02-sec-edgar-ingestion](rag-02-sec-edgar-ingestion.md) - SEC EDGAR filing ingestion
- [rag-03-evaluation-design](rag-03-evaluation-design.md) - RAG evaluation design
- [rag-04-evaluation-execution](rag-04-evaluation-execution.md) - RAG evaluation execution
- [rag-05-sec-evaluation](rag-05-sec-evaluation.md) - SEC-specific evaluation
- [rag-06-ingestion-improvements](rag-06-ingestion-improvements.md) - Ingestion improvements

### Evaluation Framework
- [eval-01-data-contract](eval-01-data-contract.md) - Evaluation data contracts
- [eval-02-capability-matrix](eval-02-capability-matrix.md) - Capability evaluation matrix
- [eval-03-multi-client-platform](eval-03-multi-client-platform.md) - Multi-client evaluation platform
- [eval-04-platform-review](eval-04-platform-review.md) - Platform review and audit
- [eval-05-distributed-workers](eval-05-distributed-workers.md) - Distributed evaluation workers
- [eval-06-evalkit-extraction](eval-06-evalkit-extraction.md) - Evalkit codebase extraction
- [eval-07-standards-audit](eval-07-standards-audit.md) - Codebase standards audit
- [eval-08-temporal-data](eval-08-temporal-data.md) - Temporal evaluation data
- [eval-09-test-quality](eval-09-test-quality.md) - Test quality improvements

### Infrastructure
- [infra-01-observability](infra-01-observability.md) - Observability stack setup
- [infra-02-deployment](infra-02-deployment.md) - Kubernetes/Terraform deployment

### Meta
- [meta-01-release-audit](meta-01-release-audit.md) - Public release audit

## Completed Plans

See [completed/](completed/) - plans moved here when work is done.

Files in `completed/` use date prefix: `{YYYY-MM-DD}-{name}.md`

## Workflow

1. **New plan**: Create `{component}-{next-seq}-{name}.md`
2. **Complete plan**: `git mv` to `completed/{date}-{name}.md`
