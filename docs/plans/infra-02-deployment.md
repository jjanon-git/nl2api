# Deployment Configuration Plan

**Status:** Not Started
**Priority:** High (Production Blocker)
**Author:** Mostly Claude, with some minor assistance from Sid
**Created:** 2026-01-24

---

## Problem Statement

The project has zero deployment infrastructure. While `docker-compose.yml` works for local development, there's nothing for production deployment.

### Current State

| Component | Status |
|-----------|--------|
| Kubernetes manifests | None |
| Helm charts | None |
| Terraform/CloudFormation | None |
| Resource limits | Not defined |
| HPA policies | None |
| Deployment runbook | None |
| Secrets management | Local .env only |

### What Exists

- `docker-compose.yml` - Local development stack
- `.env.example` - Environment variable documentation
- Dockerfiles implied (not explicit in repo)

---

## Goals

1. **Kubernetes-ready** - Deploy to any K8s cluster
2. **Scalable** - HPA for workers, configurable replicas
3. **Secure** - Secrets via K8s secrets or external vault
4. **Observable** - Prometheus scraping, log aggregation
5. **Reproducible** - IaC for cloud resources

---

## Target Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Kubernetes Cluster                       │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐          │
│  │ nl2api-api  │  │ nl2api-api  │  │ nl2api-api  │          │
│  │  (replica)  │  │  (replica)  │  │  (replica)  │          │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘          │
│         │                │                │                  │
│         └────────────────┼────────────────┘                  │
│                          │                                   │
│                    ┌─────▼─────┐                             │
│                    │  Service  │                             │
│                    │ (ClusterIP)│                            │
│                    └─────┬─────┘                             │
│                          │                                   │
│  ┌───────────────────────┼───────────────────────┐          │
│  │                       │                       │          │
│  ▼                       ▼                       ▼          │
│ ┌─────────┐        ┌─────────┐           ┌─────────┐        │
│ │PostgreSQL│        │  Redis  │           │  OTEL   │        │
│ │ (StatefulSet)│    │(Deployment)│        │Collector│        │
│ └─────────┘        └─────────┘           └─────────┘        │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐                           │
│  │eval-worker  │  │eval-worker  │  ← HPA (CPU/queue depth)  │
│  │  (replica)  │  │  (replica)  │                           │
│  └─────────────┘  └─────────────┘                           │
└─────────────────────────────────────────────────────────────┘
```

---

## Phases

### Phase 1: Dockerfiles (1 day)

Create production-ready Dockerfiles.

**Tasks:**
- [ ] Create `Dockerfile` for nl2api-api
- [ ] Create `Dockerfile.worker` for evaluation workers
- [ ] Multi-stage builds (builder + runtime)
- [ ] Non-root user
- [ ] Health check instructions

**Dockerfile Structure:**
```dockerfile
# syntax=docker/dockerfile:1
FROM python:3.11-slim as builder
WORKDIR /app
COPY pyproject.toml .
RUN pip install --no-cache-dir build && python -m build --wheel

FROM python:3.11-slim as runtime
RUN useradd -m -u 1000 appuser
WORKDIR /app
COPY --from=builder /app/dist/*.whl .
RUN pip install --no-cache-dir *.whl && rm *.whl
USER appuser
EXPOSE 8080
HEALTHCHECK CMD curl -f http://localhost:8080/health || exit 1
CMD ["python", "-m", "src.mcp_servers.entity_resolution"]
```

### Phase 2: Kubernetes Manifests (3 days)

Create base K8s manifests in `deploy/k8s/`.

**Directory Structure:**
```
deploy/
├── k8s/
│   ├── base/
│   │   ├── namespace.yaml
│   │   ├── api-deployment.yaml
│   │   ├── api-service.yaml
│   │   ├── worker-deployment.yaml
│   │   ├── configmap.yaml
│   │   ├── secrets.yaml (template)
│   │   └── kustomization.yaml
│   └── overlays/
│       ├── dev/
│       │   └── kustomization.yaml
│       ├── staging/
│       │   └── kustomization.yaml
│       └── prod/
│           ├── kustomization.yaml
│           ├── hpa.yaml
│           └── pdb.yaml
```

**Tasks:**
- [ ] Create namespace manifest
- [ ] Create API deployment with:
  - Resource requests/limits
  - Liveness/readiness probes
  - Environment from ConfigMap/Secret
- [ ] Create worker deployment
- [ ] Create services (ClusterIP for internal, LoadBalancer/Ingress for external)
- [ ] Create ConfigMap for non-secret config
- [ ] Create Secret template (actual values from vault/external)
- [ ] Set up Kustomize overlays for environments

**Resource Recommendations:**
```yaml
resources:
  requests:
    memory: "256Mi"
    cpu: "100m"
  limits:
    memory: "1Gi"
    cpu: "1000m"
```

### Phase 3: Helm Chart (2 days)

Package as Helm chart for easier deployment.

**Chart Structure:**
```
deploy/helm/nl2api/
├── Chart.yaml
├── values.yaml
├── values-prod.yaml
├── templates/
│   ├── _helpers.tpl
│   ├── deployment.yaml
│   ├── service.yaml
│   ├── hpa.yaml
│   ├── configmap.yaml
│   ├── secret.yaml
│   └── ingress.yaml
```

**Tasks:**
- [ ] Create Chart.yaml with metadata
- [ ] Create values.yaml with defaults
- [ ] Template all K8s resources
- [ ] Add conditional HPA
- [ ] Add conditional Ingress
- [ ] Add NOTES.txt for post-install instructions

**values.yaml Example:**
```yaml
replicaCount: 2

image:
  repository: your-registry/nl2api
  tag: latest
  pullPolicy: IfNotPresent

resources:
  requests:
    memory: 256Mi
    cpu: 100m
  limits:
    memory: 1Gi
    cpu: 1000m

autoscaling:
  enabled: true
  minReplicas: 2
  maxReplicas: 10
  targetCPUUtilization: 70

postgresql:
  enabled: true  # or external
  host: ""
  port: 5432

redis:
  enabled: true  # or external
  host: ""
  port: 6379
```

### Phase 4: HPA and PDB (1 day)

Configure autoscaling and disruption budgets.

**Tasks:**
- [ ] Create HPA for API pods (CPU-based)
- [ ] Create HPA for workers (queue depth via KEDA or custom metrics)
- [ ] Create PodDisruptionBudget (minAvailable: 1)
- [ ] Document scaling behavior

**HPA Configuration:**
```yaml
apiVersion: autoscaling/v2
kind: HorizontalPodAutoscaler
metadata:
  name: nl2api-api
spec:
  scaleTargetRef:
    apiVersion: apps/v1
    kind: Deployment
    name: nl2api-api
  minReplicas: 2
  maxReplicas: 10
  metrics:
  - type: Resource
    resource:
      name: cpu
      target:
        type: Utilization
        averageUtilization: 70
  behavior:
    scaleDown:
      stabilizationWindowSeconds: 300
```

### Phase 5: Terraform for Cloud Resources (3 days)

IaC for cloud infrastructure (PostgreSQL, Redis, networking).

**Directory Structure:**
```
deploy/terraform/
├── modules/
│   ├── postgresql/
│   ├── redis/
│   ├── networking/
│   └── kubernetes/
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── terraform.tfvars
│   ├── staging/
│   └── prod/
```

**Tasks:**
- [ ] Create PostgreSQL module (Azure Database for PostgreSQL or AWS RDS)
- [ ] Create Redis module (Azure Cache for Redis or AWS ElastiCache)
- [ ] Create networking module (VNet/VPC, subnets, security groups)
- [ ] Create K8s module (AKS or EKS)
- [ ] Environment-specific tfvars

**PostgreSQL Module Example:**
```hcl
resource "azurerm_postgresql_flexible_server" "main" {
  name                = "${var.project}-${var.environment}"
  resource_group_name = var.resource_group
  location            = var.location

  sku_name   = var.sku
  storage_mb = var.storage_mb
  version    = "16"

  authentication {
    active_directory_auth_enabled = true
  }
}
```

### Phase 6: Deployment Runbook (1 day)

Document the deployment process.

**Tasks:**
- [ ] Create `deploy/RUNBOOK.md`
- [ ] Document prerequisites (kubectl, helm, terraform)
- [ ] Step-by-step deployment instructions
- [ ] Rollback procedures
- [ ] Troubleshooting guide
- [ ] Secrets rotation procedure

**Runbook Sections:**
1. Prerequisites
2. Initial Setup (first-time deployment)
3. Routine Deployment (updates)
4. Rollback Procedure
5. Scaling Operations
6. Secret Rotation
7. Disaster Recovery
8. Troubleshooting

---

## Success Criteria

| Metric | Target |
|--------|--------|
| Deploy to new cluster | < 30 minutes |
| Zero-downtime deployment | Rolling updates work |
| HPA response time | Scale up within 2 minutes |
| Documentation completeness | All procedures documented |

---

## Risks & Mitigations

| Risk | Mitigation |
|------|------------|
| Cloud provider lock-in | Use Terraform modules, abstract provider specifics |
| Secrets exposure | Use external secrets operator or vault |
| Cost overruns | Set resource limits, use spot instances for workers |

---

## Dependencies

- Dockerfiles need application to be packageable
- Terraform needs cloud provider access
- Helm needs K8s cluster for testing

---

## Effort Estimate

| Phase | Effort |
|-------|--------|
| Phase 1: Dockerfiles | 1 day |
| Phase 2: K8s manifests | 3 days |
| Phase 3: Helm chart | 2 days |
| Phase 4: HPA/PDB | 1 day |
| Phase 5: Terraform | 3 days |
| Phase 6: Runbook | 1 day |
| **Total** | **11 days** |
