# Personal Finance Dashboard ("Monarch-style") — Plan

**Status:** Planning only. This doc lives in `nl2api` temporarily; it moves to a new
repo (working name `finboard`) once we start building.

**Goal:** Replace Credit Karma with a private, self-owned dashboard: all accounts in
one place, automatic transaction import, good categorization, cash-flow and
net-worth views, and per-property P&L pages. Data stays on hardware/cloud I control.

**Inspiration:** r/ClaudeAI "I built my own Monarch-style finance dashboard" (SharkFin):
SimpleFIN → Actual Budget (system of record) → custom React front end, built
page-by-page starting with Cash Flow.

---

## 1. What the thread recommends (and what holds up)

| Component | Role | Notes (verified Sept 2026) |
|---|---|---|
| **SimpleFIN Bridge** | Bank aggregation | $1.50/mo or $15/yr, up to 25 institutions, US/Canada, ~daily refresh, up to 90 days history. Simple read-only HTTP API. No approval process. |
| **Plaid** | Bank aggregation | New teams (since Apr 2026) get a free **Trial** plan: 10 production Items, then pay-as-you-go. Re-linking an institution burns an Item. Better coverage/freshness than SimpleFIN, but approval friction reported and costs grow. |
| **Actual Budget** | Budget engine / system of record | OSS, local-first, envelope budgeting, built-in SimpleFIN sync, Node API (`@actual-app/api`). What SharkFin used as its backend. |
| **Sure** (we-promise/sure) | Complete self-hosted app | Community fork of Maybe Finance. Rails + Postgres + Redis, Docker, AGPLv3, ~10k stars, very active. Accounts, transactions, investments, net worth, AI assistant. The "just use this" answer from the thread. |
| **KevFin** | Complete self-hosted app | React/Vite + Express/TS + SQLite, MIT. Plaid + SimpleFIN, property valuation (Zillow APIs), Monte Carlo retirement, Docker/NAS. Closest to a codebase we could fork. |
| Firefly III (not in thread) | Complete self-hosted app | Mature double-entry ledger, PHP. Powerful rules engine, dated UI. Mentioned for completeness. |

Hosting suggestions from the thread: mini-PC/NAS at home + Tailscale; Oracle/AWS
free tier VM; Cloudflare Workers behind Google OAuth.

---

## 2. Decision: build vs. adopt

Three realistic paths:

| Path | Time to useful | Control / customization | Maintenance |
|---|---|---|---|
| **A. Adopt Sure** (+ SimpleFIN) | An evening | Low–medium (Rails, AGPL) | Upstream does it |
| **B. SharkFin pattern**: SimpleFIN → Actual → custom UI | A weekend + page-by-page | High for UI, budget logic owned by Actual | Two systems to run |
| **C. Own stack**: SimpleFIN/Plaid → our DB → our API/UI | 2–4 weekends | Full | All ours |

**Recommendation: A as a one-week spike, then C.**

1. **Spike (week 1):** Run Sure in Docker with a SimpleFIN token. Cost: $1.50.
   This gets Credit Karma replaced immediately and teaches what I actually use
   (and what annoys me) before writing code. If Sure is good enough, stop here.
2. **Build (C) if the spike shows gaps** (property P&L, custom cash-flow views,
   LLM categorization I can tune, FIRE projections). Keep the ingest layer
   provider-agnostic so SimpleFIN and Plaid are interchangeable.

Why not B: Actual is an excellent *budgeting* app, but using it as a backend means
syncing through its Node API and living within its data model. Worth it only if
I want envelope budgeting specifically. Revisit if I do.

---

## 3. Target architecture (Path C)

```
 SimpleFIN ─┐                            ┌─ React dashboard (Vite, Recharts)
 Plaid* ────┼─► ingest worker ─► Postgres/SQLite ─► FastAPI ─┤
 CSV/OFX ───┘   (dedupe, normalize,  (accounts, txns,        └─ CLI / MCP server (ask Claude
                 categorize)         balances, rules,             "what did the rental cost
                     │               properties, snapshots)       me in Q3?")
                     └─► rules engine → LLM fallback (Claude Haiku) for uncategorized
```
`*` optional, only for institutions SimpleFIN doesn't cover.

**Stack choice:** Python (FastAPI, SQLAlchemy/asyncpg, Pydantic) + React. Reuses
patterns I already have here: async repositories, frozen Pydantic models, and
evalkit for measuring categorization accuracy.

### Core data model

- `institutions`, `accounts` (type: checking/credit/loan/investment/property/manual)
- `balances` (daily snapshot per account → net worth history)
- `transactions` (provider id, posted/pending, amount, merchant raw/clean,
  category, tags, `entity` e.g. *Household*, *Property: 12 Oak St*, `is_transfer`)
- `categorization_rules` (merchant pattern → category/entity; learned from edits)
- `entities` (household + each property; lets property pages roll up only net
  cash flow into the main budget — SharkFin's best idea)
- `budgets`, `recurring` (detected subscriptions/bills)

### Ingestion rules

- Idempotent upsert keyed on `(provider, provider_txn_id)`; handle pending→posted.
- Transfer detection (matching amounts across own accounts within ±3 days) so
  credit-card payments don't count as spending.
- Categorization order: user rule → merchant history → LLM (with category list
  and a few examples) → "Uncategorized" for review. Every manual correction
  becomes a rule.

---

## 4. MVP scope and phases

Thread advice worth taking: **make one page excellent, then expand.**

| Phase | Deliverable | Done when |
|---|---|---|
| 0 | Sure spike + SimpleFIN account; export Credit Karma / bank CSVs for history | I've used it for a week and listed gaps |
| 1 | New repo, schema, SimpleFIN ingest + CSV import, CLI to list txns | All accounts sync nightly, no duplicates over 7 days |
| 2 | Categorization (rules + LLM) + review queue | ≥90% correct on labeled set (see §6) |
| 3 | **Cash Flow page** (income vs spend by month, by category, drill-down) | Replaces what I checked in Credit Karma |
| 4 | Net worth over time, accounts page | Daily snapshots graphing correctly |
| 5 | Property pages (per-property P&L; only net flows into household) | Each property shows monthly NOI + top expenses |
| 6 | Budgets, recurring/subscriptions, alerts (email/push on big or odd txns) | — |
| 7 | Nice-to-haves: investments/allocation, FIRE projection, MCP server for Q&A | — |

---

## 5. Hosting and security

Financial data + a SimpleFIN access URL (a bearer credential to all my accounts)
means **no public endpoint by default**.

**Recommended:** small always-on box at home (used mini-PC ~$75–150, or existing
NAS) running Docker Compose, reachable only via **Tailscale**. Nightly encrypted
backup (restic → Backblaze B2 / S3, pennies/month).

**Cloud alternative:** Oracle Cloud free-tier ARM VM (or a $5 VPS) with the same
Compose file, Tailscale-only access. Cloudflare Workers + D1 + Access (Google
login) also works but forces a TS/Workers rewrite; not worth it for one user.

Security checklist:
- SimpleFIN/Plaid secrets in env/secret store, never in repo or logs.
- DB at rest encryption (disk-level) + encrypted backups.
- LLM categorization sends only merchant string, amount, date — no account numbers.
  (Option: local model via Ollama for fully offline.)
- App auth even behind Tailscale (single user, passkey or OAuth).

---

## 6. Evaluation (per repo rule: every capability needs evaluation)

- **Categorization:** hand-label ~300 of my own transactions → fixture set.
  Metric: accuracy by category + % needing review. Baseline rules-only vs.
  rules+LLM. Re-run on every prompt/rule change (evalkit pack or simple pytest).
- **Ingest correctness:** duplicate rate, transfer-detection precision/recall on
  a labeled month, balance reconciliation (sum of txns vs. reported balance).

---

## 7. Costs (monthly, steady state)

| Item | Cost |
|---|---|
| SimpleFIN | $1.25–1.50 |
| Plaid (optional, ≤10 items on Trial) | $0 → ~$0.30/account after |
| LLM categorization (Haiku, ~300 txns/mo) | < $0.10 |
| Hosting | $0 (home box/free tier) to ~$5 VPS |
| Backups | < $0.50 |
| **Total** | **~$2–7/mo** (+ one-time mini-PC if needed) |

---

## 8. Risks / open questions

- **Institution coverage:** check each of my banks/cards on SimpleFIN before
  committing; Plaid as fallback.
- **History:** SimpleFIN gives ~90 days; older history needs CSV exports now
  (grab from Credit Karma and each bank before leaving).
- **Credit score:** Credit Karma's score monitoring isn't replaced by any of this —
  keep the free CK account (or card-issuer FICO) just for that.
- **Maintenance creep:** Path C is a hobby project forever. The Sure spike is the
  guard against building something I don't need.
- Open: do I want envelope budgeting (→ reconsider Actual) or just tracking + cash flow?
- Open: which properties/entities, and should mortgage principal count as expense
  or equity?

---

## 9. Next steps

1. Sign up for SimpleFIN; verify every institution connects.
2. `docker compose up` Sure locally with the SimpleFIN token; use it for a week.
3. Export all available history from Credit Karma and banks (CSV/OFX).
4. Create new repo `finboard` (private), move this doc to `docs/PLAN.md`, add
   `BACKLOG.md` with the phases above.
5. Start Phase 1.

## Sources

- Reddit thread (SharkFin) — r/ClaudeAI, Sept 2026
- SimpleFIN + Actual: https://actualbudget.org/docs/advanced/bank-sync/simplefin/
- Plaid free Trial plan: https://plaid.com/docs/account/billing/
- Sure: https://github.com/we-promise/sure · https://docs.sure.am/
- KevFin: https://github.com/kxl3785/KevFin
