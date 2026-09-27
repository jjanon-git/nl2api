# finboard — Plan

**Status:** Planning. No code yet.

**Goal:** Replace Credit Karma with a private, self-owned dashboard covering all
accounts: net worth and investment balances first, then cash flow, budgets, and
per-property P&L. Data stays on hardware/cloud I control.

**Inspiration:** r/ClaudeAI "I built my own Monarch-style finance dashboard" (SharkFin):
SimpleFIN → Actual Budget (system of record) → custom React front end, built
page-by-page.

---

## 1. What I track today (Credit Karma, Sept 2026)

The portfolio is **investment-heavy**: about 95% sits in retirement and brokerage
accounts, and cash is at Chase and Marcus. So net worth, balance history and
allocation matter more than transaction budgeting.

| Institution | Accounts | Login / portal |
|---|---|---|
| Fidelity (retail) | 2 brokerage (TOD), 2 HSA | fidelity.com |
| Fidelity (workplace) | 3 employer 401(k)s | NetBenefits |
| Chase | Self-Directed Investing, Self-Directed retirement, checking/savings | chase.com |
| Vanguard | 1 joint brokerage | vanguard.com |
| Other recordkeeper | 1 employer 401(k) | *TBD — confirm which portal* |
| Marcus (Goldman Sachs) | Savings | marcus.com |

Credit Karma problems visible today, which this design has to fix:
- **Stale duplicates:** the Vanguard account is listed twice, and one copy
  hasn't synced in 11 months ("needs attention"). One 401(k) also appears twice.
- **Broken connections are hidden:** the only signal is a banner.
- **Balance-only:** it shows no holdings or allocation.

## 2. Institution coverage by aggregator

| Institution | SimpleFIN (MX underneath) | Plaid | Risk | Fallback |
|---|---|---|---|---|
| Chase (bank + Self-Directed) | Supported (OAuth) | Supported (OAuth) | Low | — |
| Marcus | Supported | Supported | Low | — |
| Fidelity retail + NetBenefits | Probably. Fidelity allows only token-based access through Akoya; MX has access, but reliability complaints are common across apps | Needs a separate access request (automatic on Growth/Custom, manual on Pay-as-you-go); **may not be available on the free Trial** | **High** | Positions/transactions CSV download |
| Vanguard | Hit-or-miss. Vanguard dropped OFX and breaks aggregators often (already broken in CK) | Supported but flaky | **Medium–High** | Positions CSV download |
| Other 401(k) recordkeeper | Depends on the recordkeeper | Depends | Medium | Statement PDF/CSV, or manual balance |

**Takeaways**
1. **Verify first ($1.50):** subscribe to SimpleFIN for one month and connect every
   login, before building anything. Its institution search is the source of truth.
2. **Plaid is a backup:** about 6 logins fits the Trial's 10 Items, but Fidelity
   (roughly half the net worth) likely needs an extra access request.
3. **Manual import is a core feature:** for Fidelity/Vanguard/401(k)s, a monthly CSV
   positions import is enough. Retirement accounts don't need daily transactions.
4. **Show freshness everywhere:** every account shows "last synced", stale
   accounts are flagged loudly, and each account has exactly one live source.

## 3. Components considered

| Component | Role | Notes |
|---|---|---|
| **SimpleFIN Bridge** | Aggregator | $15/yr, 25 institutions, ~daily refresh, 90 days history, simple read-only API, holdings in beta |
| **Plaid** | Aggregator | Free Trial: 10 production Items (teams created after Apr 2026); Investments product included; Fidelity gated |
| **Actual Budget** | Budget engine | OSS, local-first, envelope budgeting, SimpleFIN sync. Weak on investments |
| **Sure** (we-promise/sure) | Complete app | Maybe Finance fork, Rails/Postgres, AGPLv3, ~10k stars, investments + net worth |
| **KevFin** | Complete app / fork base | React + Express + SQLite, MIT, Plaid + SimpleFIN, property values, Monte Carlo retirement |
| **Wealthfolio** | Investment tracker | Local desktop app, CSV import of holdings; relevant because the portfolio is investment-heavy |

## 4. Build vs. adopt

| Path | Time to useful | Fit for investment-heavy | Maintenance |
|---|---|---|---|
| A. Sure + SimpleFIN | An evening | Good (net worth, holdings) | Upstream |
| B. SimpleFIN → Actual → custom UI | A weekend | Poor (Actual is budgeting-first) | Two systems |
| C. Own stack | 2–4 weekends | Tailored | All mine |

**Decision: run A as a one-week spike, then build C if the spike shows gaps.**
Drop B: Actual's strengths (envelopes) aren't what I need.

## 5. Target architecture (Path C)

```
 SimpleFIN ─┐                               ┌─ React dashboard (Vite, Recharts)
 Plaid* ────┼─► ingest worker ─► Postgres ─► FastAPI ─┤
 CSV/OFX ───┘   (normalize, dedupe,  (accounts, balances,   └─ MCP server / CLI
 manual ────┘    categorize)          holdings, txns, ...)      ("how's my allocation?")
```
`*` optional.

**Stack:** Python 3.12 (FastAPI, asyncpg/SQLAlchemy, Pydantic) + React/Vite +
Postgres (SQLite acceptable for v0). Docker Compose.

### Data model

- `institutions`, `connections` (provider, status, last_success_at, last_error)
- `accounts` (type, subtype: brokerage/401k/hsa/ira/checking/savings/property/loan;
  **one** active `source` per account; owner(s); `is_hidden`)
- `balance_snapshots` (account, date, balance). This feeds net-worth history, and
  gaps render as stale rather than as zero
- `holdings` (account, date, security, qty, value, cost basis), `securities`
  (ticker, name, asset class)
- `transactions` (provider id, pending/posted, amount, merchant, category,
  `entity`, `is_transfer`)
- `entities` (household, each property); property P&L rolls only net cash flow
  into household
- `categorization_rules`, `budgets`, `recurring`

### Ingestion rules
- Idempotent upsert on `(source, external_id)`; pending→posted reconciliation.
- Account matching by institution + mask + type to prevent CK-style duplicates.
- Transfer detection across own accounts (±3 days, equal amounts).
- Categorization: user rule → merchant history → LLM (Claude Haiku) → review queue.

## 6. Phases (reprioritized for investment-heavy profile)

| Phase | Deliverable | Done when |
|---|---|---|
| 0 | SimpleFIN trial: connect all logins, record coverage in table §2; Sure spike for a week; export CK + bank history | Coverage table filled with facts |
| 1 | Repo scaffold, schema, SimpleFIN ingest, CSV positions import, CLI | All accounts present, one source each, nightly sync, zero dupes over 7 days |
| 2 | **Net Worth + Accounts page**: total, by account/type/owner, history, freshness badges | Replaces the CK screen above |
| 3 | Holdings + allocation (asset class, fund look-through later) | Allocation matches Fidelity/Vanguard within 1% |
| 4 | Cash flow (Chase + Marcus + cards): categorization + review queue | ≥90% categorization accuracy (§8) |
| 5 | Property pages (per-property P&L) | Monthly NOI + top expenses per property |
| 6 | Alerts (stale connection, large txn), budgets, recurring | — |
| 7 | FIRE / retirement projection, MCP server for Q&A | — |

## 7. Hosting and security

The SimpleFIN access URL is a bearer credential to every account, so there is
**no public endpoint**.

- **Recommended:** home mini-PC/NAS + Docker Compose, reachable only via Tailscale.
  Nightly encrypted backups (restic → B2/S3).
- **Alternative:** Oracle free-tier ARM VM or $5 VPS, same Compose file, Tailscale only.
- Secrets in `.env` (gitignored) or a secret store, never in logs.
- **Never commit real financial data**: no exports, screenshots, account
  numbers, or names. `data/` and `exports/` are gitignored. Test fixtures are synthetic.
- LLM calls send only merchant, amount, and date. Option: local model via Ollama.
- App-level auth even behind Tailscale.

## 8. Evaluation

- **Ingest:** duplicate rate = 0; every account's latest balance within $1 of the
  institution site on a spot-check; stale-account detection covered by tests.
- **Categorization:** ~300 hand-labeled transactions (kept local, not in git) plus
  a synthetic fixture set in git. Accuracy by category; baseline rules-only vs.
  rules+LLM; re-run on every rule/prompt change.
- **Allocation:** matches each institution's own allocation view within 1%.

## 9. Costs (monthly)

SimpleFIN ~$1.25 · Plaid $0 (Trial, optional) · LLM < $0.10 · hosting $0–5 ·
backups < $0.50 → **~$2–7/mo**.

## 10. Open questions

- Which recordkeeper runs the non-Fidelity 401(k)?
- Joint accounts: how to show ownership (mine/spouse/joint) in net worth?
- Mortgage principal: expense or equity on property pages?
- Keep CK (or card-issuer FICO) for credit-score monitoring only.

## Sources

- SimpleFIN + Actual: https://actualbudget.org/docs/advanced/bank-sync/simplefin/
- SimpleFIN institutions: https://beta-bridge.simplefin.org/search-institutions
- Plaid billing/Trial: https://plaid.com/docs/account/billing/
- Plaid OAuth institutions (Fidelity access): https://plaid.com/docs/link/oauth/
- Fidelity/Akoya: https://riabiz.com/a/2023/10/19/fidelity-just-dropped-the-hammer-on-screen-scrapers-to-cheers-but-some-firms-like-plaid-are-holdouts-and-the-cfpb-may-wield-the-final-gavel
- Sure: https://github.com/we-promise/sure
- KevFin: https://github.com/kxl3785/KevFin
- Wealthfolio SimpleFIN discussion: https://github.com/afadil/wealthfolio/issues/197
