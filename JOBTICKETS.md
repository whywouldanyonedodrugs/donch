# Donch Job Tickets

This file is the single place to track operational and engineering jobs.

## Rules
- Keep statuses current: `todo`, `in_progress`, `blocked`, `done`.
- Update `Last Updated` whenever ticket statuses change.
- Use one ticket id per work item (example: `JT-005`, `ENG-002`).
- Do not remove completed tickets; mark them `done`.

## Constraint
- `CODEOWNERS` is intentionally out of scope right now (free GitHub tier constraint).
- Do not open tickets that depend on `CODEOWNERS` until this constraint changes.

## Last Updated
- `2026-02-15` (UTC)

## Active Tickets
| ID | Title | Priority | Status | Notes |
|---|---|---|---|---|
| JT-002 | Warmup-aware schema-fail classification | High | done | Implemented in live export with `schema_fail_class`, listing-age fields, and per-class diagnostics. |
| JT-003 | Full live sizing parity with backtester | High | blocked | Core logic aligned; waiting for 5 real runtime sizing lines from actual entries. |
| JT-004 | Decision reason taxonomy normalization | High | done | `reason_raw` + `reason_canonical` exported; mapping and agreement artifacts added. |
| ENG-001 | Split large `live/live_trader.py` into modules | Medium | todo | Target modules: decision, sizing, execution, risk, data context. |
| ENG-002 | Reduce legacy config toggles and ambiguous branches | High | todo | Add strict config validation and approved run profiles. |
| OPS-001 | CI gates and release checks | High | todo | Lint, tests, parity fixtures, artifact/schema checks. |
| EXE-001 | Add TCA/slippage tracking | Medium | todo | Persist decision vs fill timing/price and slippage metrics. |
| RISK-001 | Independent pre-trade risk-control layer | High | todo | Hard risk limits and separate gate before order placement. |

## Backlog
| ID | Title | Priority | Status | Notes |
|---|---|---|---|---|
| OPS-002 | Daily autopar delivery health monitor | Medium | todo | Alert when export package missing/late or schema-fail spikes. |
| OPS-003 | Shared-path hardening for autopar packages | Medium | todo | Add retention policy and integrity check on published packages. |

## Ticket Template
Use this when adding new work:

```md
| ID | Title | Priority | Status | Notes |
|---|---|---|---|---|
| XXX-000 | Short title | High/Medium/Low | todo | One-line scope + acceptance criteria summary. |
```

## Detailed Ticket Specs

### JT-002 - Warmup-aware schema-fail classification
- `Status`: `done`
- `Owner`: Live/Autopar
- `Objective`: Separate expected warmup schema failures from integration defects without weakening strict schema checks.
- `Specs`:
- Keep strict schema fail-closed behavior unchanged.
- Use exchange instrument metadata for listing context.
- Classify schema failures into `warmup_expected`, `integration_defect`, `unknown`.
- `Deliverables`:
- `live_decisions.csv` fields: `schema_fail_class`, `symbol_listed_at_utc`, `symbol_age_days`.
- `symbol_metadata.csv` keyed by symbol with listing metadata source.
- `schema_diagnostics.json` with class counts and top missing/symbols by class.
- `Acceptance Tests / Evidence`:
- Export package contains all fields/files above.
- 20-row schema fail sample includes class and listing-age fields.
- Diagnostics JSON includes class breakdown and per-class top lists.
- `Current Evidence`:
- `results/autopar_exports/autopar_2026-02-09/schema_fail_sample_20.csv`
- `results/autopar_exports/autopar_2026-02-09/symbol_metadata.csv`
- `results/autopar_exports/autopar_2026-02-09/schema_diagnostics.json`

### JT-003 - Full live sizing parity with backtester
- `Status`: `blocked`
- `Owner`: Live Sizing
- `Objective`: Match backtester sizing semantics 1:1 for regime handling, multiplier chain, probe cap, risk conversion, and notional cap.
- `Specs`:
- Regime handling first, including down-regime equity downscale path.
- Sizing precedence: `risk_scale` override else dynamic map+ETH downsize+clamp.
- Risk-off probe cap on `risk_on==0`.
- Cash/percent conversion exactly as offline rules.
- Qty from risk per unit with notional/leverage cap.
- No hidden probability veto in order-open path.
- `Deliverables`:
- Code references for each chain step.
- 100-row live-vs-offline parity fixture CSV.
- Summary stats JSON (`mean`/`p90` abs errors).
- Runtime sizing log lines including `p_cal`, `risk_on`, `size_mult`, `risk_usd`, `qty`.
- `Acceptance Tests / Evidence`:
- Fixture abs errors near zero for `size_mult` and `risk_usd`.
- Unit tests for fixture parity tool pass.
- At least 5 runtime sizing lines from real entries (not synthetic fixture).
- `Current Evidence`:
- `results/autopar_exports/sizing_parity_fixture_20260215T095756Z.csv`
- `results/autopar_exports/sizing_parity_fixture_20260215T095756Z.summary.json`
- `results/autopar_exports/sizing_parity_fixture_20260215T095756Z.logsample.txt`
- `Blocker`: insufficient real entry events to produce 5 runtime sizing lines.

### JT-004 - Decision reason taxonomy normalization
- `Status`: `done`
- `Owner`: Live/Autopar
- `Objective`: Stabilize reason agreement metrics across versions with canonical mapping while preserving raw reasons.
- `Specs`:
- Export both `reason_raw` and `reason_canonical`.
- Canonical reason map maintained in exporter.
- Keep raw reason unchanged for forensics.
- `Deliverables`:
- `live_decisions.csv` with raw and canonical reason columns.
- `reason_mapping.csv` with daily counts by mapping.
- `reason_agreement.json` with canonical agreement metrics.
- `Acceptance Tests / Evidence`:
- Sample export rows include both columns.
- Mapping table file present and populated.
- Agreement metric generated in each run and included in `run_context.json`.
- `Current Evidence`:
- `results/autopar_exports/autopar_2026-02-09/reason_sample_20.csv`
- `results/autopar_exports/autopar_2026-02-09/reason_mapping.csv`
- `results/autopar_exports/autopar_2026-02-09/reason_agreement.json`

### ENG-001 - Split large `live/live_trader.py` into modules
- `Status`: `todo`
- `Owner`: Core Eng
- `Objective`: Reduce maintainability risk and branch complexity by separating orchestration from domain logic.
- `Specs`:
- Extract pure-domain modules: decision, sizing, execution, risk, data context.
- Keep runtime behavior unchanged via parity tests before/after refactor.
- Decrease direct business logic in `live/live_trader.py`.
- `Deliverables`:
- New module files and integration wiring.
- Refactor plan doc with extraction order.
- Regression/parity test updates.
- `Acceptance Tests / Evidence`:
- Existing parity tests remain green.
- New module unit tests added for extracted logic.
- `live/live_trader.py` reduced in scope and responsibilities.

### ENG-002 - Reduce legacy config toggles and branch ambiguity
- `Status`: `todo`
- `Owner`: Core Eng
- `Objective`: Remove conflicting toggle combinations and enforce valid runtime profiles.
- `Specs`:
- Define config schema and incompatible combinations.
- Add startup validation with explicit fail message.
- Introduce named profiles: `parity`, `canary`, `production`.
- `Deliverables`:
- Config schema validator module.
- Profile presets and migration notes.
- Deprecation list for obsolete toggles.
- `Acceptance Tests / Evidence`:
- Invalid config matrix tests fail as expected.
- Profile load tests pass.
- Startup logs print active profile and resolved critical knobs.

### OPS-001 - CI gates and release checks
- `Status`: `todo`
- `Owner`: Ops/Eng
- `Objective`: Prevent unsafe changes from reaching live by enforcing automated checks.
- `Specs`:
- CI stages: lint, static checks, unit tests, parity fixture tests, export schema checks.
- Release pipeline requires successful CI and parity artifacts.
- Do not include `CODEOWNERS`-dependent controls.
- `Deliverables`:
- CI workflow files.
- Release check script and docs.
- Failure triage guide.
- `Acceptance Tests / Evidence`:
- PR pipeline runs all required jobs.
- CI fails on injected parity/schema regression.
- Release command fails when required artifact/check is missing.

### EXE-001 - Add TCA/slippage tracking
- `Status`: `todo`
- `Owner`: Execution
- `Objective`: Measure decision-to-fill quality and feed execution reality back into risk/backtest assumptions.
- `Specs`:
- Persist decision price, submit/ack/fill timestamps, fill price, spread, slippage bps.
- Produce daily TCA report by symbol and side.
- Add configurable slippage assumptions to parity/backtest comparison.
- `Deliverables`:
- DB schema extension for execution metrics.
- TCA daily report job and output files.
- Dashboard/CSV summary for ops review.
- `Acceptance Tests / Evidence`:
- New fields present for recent trades.
- Daily TCA report generated automatically.
- Slippage assumptions demonstrably applied in offline evaluation path.

### RISK-001 - Independent pre-trade risk-control layer
- `Status`: `todo`
- `Owner`: Risk/Infra
- `Objective`: Enforce hard risk limits independently from strategy logic.
- `Specs`:
- Pre-trade gate checks: daily loss cap, max exposure, per-symbol cap, kill switch.
- Strategy cannot bypass gate on order path.
- Gate emits structured allow/deny decisions with reason codes.
- `Deliverables`:
- Risk-gate module/service and integration in order-open flow.
- Configurable risk policy file.
- Incident playbook for auto-halt and resume.
- `Acceptance Tests / Evidence`:
- Negative tests prove blocked orders under limit breaches.
- Gate decision logs appear for each attempted order.
- Manual kill switch test blocks and then allows after reset.

### OPS-002 - Daily autopar delivery health monitor
- `Status`: `todo`
- `Owner`: Ops
- `Objective`: Detect missing exports and abnormal schema-fail behavior automatically.
- `Specs`:
- Alert when daily package missing/late.
- Alert on schema-fail-rate spike vs rolling baseline.
- Include top non-warmup fields in alert payload.
- `Deliverables`:
- Monitor script/service.
- Alert channel integration.
- Baseline configuration file.
- `Acceptance Tests / Evidence`:
- Simulated missing package triggers alert.
- Simulated schema spike triggers alert with diagnostics context.

### OPS-003 - Shared-path hardening for autopar packages
- `Status`: `todo`
- `Owner`: Ops
- `Objective`: Improve integrity and operability of published package path used by external consumers.
- `Specs`:
- Retention policy for old packages.
- Integrity validation of copied package (hash/size/file set).
- Atomic publish semantics (`tmp -> final rename`).
- `Deliverables`:
- Publish hardening script updates.
- Retention cleanup job.
- Integrity report per publish run.
- `Acceptance Tests / Evidence`:
- Publish writes complete/consistent package atomically.
- Retention job removes only policy-eligible old packages.
- Corrupted copy simulation is detected and reported.
