# Append-only out-of-sample evaluation design

## Decision

This PR defines an OOS contract and a proposed manifest only. It does not activate a cohort, generate historical or live OOS observations, download market data, rerun Backtest Only, change a strategy, or reinterpret the existing in-sample outputs.

Repository evidence identifies one current production primary without inference: the unique `Qualified` row at `qualification_rank = 1` in the Backtest Only-owned `docs/data/backtest_summary.json` is `score_bo_l40_rm002_erp010__signal_3d_confirm__ma50`. The same key is rank 1 in `docs/data/backtest_strategy_summary.csv`, is published with rank 1 by `docs/data/backtest_portfolio_curve_manifest.json`, and is independently named as the production primary in the PR #16 analysis. The selector itself is `src/backtest.py::rank_strategy_summary`; the strategy registry is `src/backtest.py::STRATEGY_RULES`.

The manifest is therefore `proposed`, not `active`. Merge facts, the first post-merge completed trade, the collector, and ex-ante maturity thresholds do not exist yet and are recorded as blockers rather than invented.

## Repository authority audit

| Question | Authoritative source and finding |
| --- | --- |
| Canonical / production candidate | `docs/data/backtest_summary.json`, generated and owned by Backtest Only, contains the unique rank-1 Qualified key. `src/backtest.py::rank_strategy_summary` defines the selector. The strategy summary CSV and portfolio curve manifest corroborate the same key. |
| Strategy key and parameters | `src/backtest.py` defines `SignalRule`, `EntryRule`, `ExitRule`, the parameter grids, `STRATEGY_RULES`, entry index functions, exit stop functions, `MAX_HOLDING_DAYS`, and key composition. The generated summary serializes the resolved `signal_key`, `entry_key`, `exit_key`, and `signal_params`. |
| Universe | `config/universe.yml`, `config/aum.csv`, `config/exclusions.yml`, `config/manual_overrides.csv`, `src/universe.py::build_base_universe`, and `src/backtest.py::build_historical_features`. The definition is dynamic by date; the rules are frozen, not a historical constituent list. |
| Transaction cost | `src/backtest.py::ROUND_TRIP_COST = 0.002`. Completed-event net return subtracts the full round trip. `src/portfolio.py::simulate_canonical_portfolio` charges half, 0.001, on each turnover side. |
| Benchmark | `src/run_backtest_only.py` includes SPY as benchmark-only. `src/portfolio.py::build_spy_benchmark` constructs the USD 1,000 SPY proxy. `src/prices.py::download_ohlcv` uses Yahoo Finance with `auto_adjust=True` and `actions=False`. |
| Portfolio model | `src/portfolio.py`, especially `PORTFOLIO_MODEL_NAME` and `simulate_canonical_portfolio`; the committed name is `canonical_equal_weight_active_v1`. `docs/data/backtest_portfolio_curve_manifest.json` is the authoritative generated publication manifest. |
| Daily Scan ownership | `.github/workflows/daily_scan.yml` runs the scan and owns `config/aum.csv`, `docs/data/latest.{json,csv}`, `universe_current.csv`, `excluded_etfs_summary.csv`, and `docs/data/history`. Although the command currently calls `run_backtests`, the workflow restores Backtest Only-owned outputs before committing. |
| Backtest Only ownership | `.github/workflows/backtest-only.yml` runs `src.run_backtest_only` and owns the bounded backtest summaries, strategy-year aggregates, recent/diagnostic/skipped summaries, portfolio summary, portfolio curve manifest and files, SPY benchmark, and compressed daily-return matrix. Raw event files are prohibited. |
| Generated artifact referenced by this OOS manifest | `docs/data/backtest_summary.json` is authoritative for the selected rank, resolved strategy identity, serialized rules, costs, max hold, gates, and portfolio model. `docs/data/backtest_portfolio_curve_manifest.json` is authoritative for the bounded canonical portfolio publication and corroborates the primary. |
| Semantics since PR #16 | No strategy or portfolio-economic semantics changed from PR #16 (`75d9561`) through current main (`f719886`). PR #17 changed boolean-shift spelling to the output-equivalent `shift(..., fill_value=False)` and made price-panel arrays explicit writable copies. Its regression tests guarantee identical signal/entry/exit outputs; costs, rules, grids, universe, ranking, gates, and economics were untouched. |

Source-of-truth precedence is: executable code and checked-in configuration for semantics; Backtest Only's committed bounded artifact for the current selector result; the OOS manifest for the frozen cohort snapshot. A research narrative is corroborating evidence, never a candidate selector.

## Purpose and non-purpose

The purpose is to collect genuinely forward observations for a preregistered candidate and evaluation contract after the contract exists on `main`. It separates what was known when a decision was made from what was realized later, preserves input and code provenance, detects drift, and makes every correction auditable.

It is not parameter optimization, candidate mining, a new gate, a ranking change, an execution system, a promise of profitability, a reconstruction of past OOS results, or a replacement for the current production outputs. Interim OOS results must not be used to tune this cohort.

## Cohort and frozen definitions

`oos-0001` contains exactly the strategy key recorded in the manifest. The strategy parameter fingerprint covers the resolved strategy key, signal key and parameters, entry key, exit key, and 63-session maximum hold. Callable semantics are additionally pinned by source Git blob IDs. A later change to any frozen parameter or semantic source does not update this cohort; it creates a new manifest and cohort ID.

The universe freeze is a rule freeze. Daily membership remains dynamic under the existing AUM, exclusion, manual-override, history, price, liquidity, and rank rules. Every decision snapshot records the realized eligible-symbol set or a deterministic content hash plus an immutable referenced snapshot. A definition change requires a new cohort; an ordinary membership change under unchanged rules does not.

The transaction-cost, SPY benchmark, data conventions, and `canonical_equal_weight_active_v1` definitions are frozen exactly as recorded in the manifest. A vendor refresh is not silently treated as a semantic change, but its input identity and any later correction must be recorded.

## Activation and no-backfill rule

The contract becomes eligible for activation only after this manifest is merged to `main`. The OOS start is the economic date of the first completed trade observed after that merge. The activation ledger must record the actual merge commit, merge timestamp, and first eligible completed-trade record; the manifest deliberately leaves the latter facts null.

No decision, position, trade, mark, or outcome from before merge may be inserted. A position initiated before merge is not OOS-eligible even if it exits later, because no immutable ex-ante snapshot exists. Collection may record it only as an excluded transition diagnostic. If collection starts late, the gap is recorded; it is never filled retrospectively.

## Observation model

Raw observations and derived metrics are separate.

An ex-ante decision record is written from information available at the decision time and includes:

- cohort, manifest, strategy fingerprint, collector commit, economic date, and observation timestamp;
- input artifact hashes, vendor convention, and realized universe snapshot hash;
- symbol eligibility and frozen signal/entry state;
- intended entry, exit, stop, membership, or no-action decision;
- the known portfolio state needed to reproduce the next action; and
- a deterministic record ID and schema version.

An ex-post outcome record is a different record that references one or more decision IDs. It records realized entry/exit or daily mark facts, transaction costs, completion reason and time, and the source revision. Outcome data never mutate the decision snapshot.

Derived records contain portfolio marks, trade aggregates, benchmark comparisons, evidence counts, and evaluation statistics. Every derived snapshot names the complete raw/correction cutoff, computation schema, and code commit. Recalculation produces a new derived snapshot; it does not replace an earlier one.

## Append-only layout

The implementation PR should create, without historical files:

```text
docs/data/oos/oos-0001/
  activation/events.jsonl
  raw/decisions/YYYY/YYYY-MM-DD.jsonl
  raw/outcomes/YYYY/YYYY-MM-DD.jsonl
  raw/corrections/YYYY/YYYY-MM-DD.jsonl
  derived/snapshots/YYYY/YYYY-MM-DD.json
```

Existing records are never edited or deleted. New lines or new date-partitioned files are allowed. An immutable activation event binds the committed manifest to the actual merge fact. A future candidate or definition change adds a new cohort manifest and a separate directory; it does not repoint or revise `oos-0001`.

## Idempotency, duplicates, and corrections

The collector computes `record_id` as SHA-256 over the canonical identity fields for cohort, record type, economic date, strategy, symbol or portfolio, decision reference, and event phase. Replaying an identical record is a no-op. The same ID with a different payload is a hard failure and emits no partial commit.

Two different IDs that claim the same unique economic event are also a conflict. The workflow fails closed until an explicit correction record is approved. Duplicate-date portfolio marks are not resolved by last-write-wins.

Vendor revisions, bad symbols, corporate-action corrections, and parsing fixes never overwrite raw data. A correction record references the superseded record ID, states the reason, records old and new input provenance, supplies the corrected payload, and has its own ID and timestamp. Derived metrics follow the latest valid correction chain while retaining the complete history.

## Provenance and drift detection

Every raw observation records the collector commit SHA, manifest version, strategy fingerprint, input hashes, universe snapshot hash, and vendor/data convention. The baseline manifest also records Git blob IDs for all frozen semantic, configuration, workflow-ownership, and authoritative generated sources.

At runtime the collector must:

1. resolve every manifest key exactly once from `STRATEGY_RULES`;
2. rebuild and compare the parameter fingerprint;
3. compare the frozen semantic/configuration blob IDs or an explicitly versioned successor fingerprint;
4. verify the candidate still matches the frozen identity without rerunning selection;
5. verify the activation event and monotonic raw ledger; and
6. record the current collector commit and input hashes.

The repository HEAD is expected to advance for documentation and generated data. HEAD inequality alone is not drift. A mismatch in a frozen semantic or definition fingerprint is drift and requires a new cohort or an explicit non-semantic migration reviewed before collection resumes.

## Fail-closed conditions

The future workflow must stop without committing observations when:

- manifest status is not active, a blocking unresolved item remains, or activation facts are absent;
- a strategy key is missing, duplicated, ambiguous, or its fingerprint differs;
- a frozen strategy, universe, cost, benchmark, portfolio, or data-convention fingerprint differs;
- required input provenance, market date, universe snapshot, decision snapshot, or portfolio state is missing;
- a record predates activation or would constitute historical backfill;
- dates regress, a record ID conflicts, a unique event is duplicated, or a correction chain is invalid;
- input data are incomplete or internally inconsistent for an intended decision;
- the append would edit/delete an existing raw record; or
- raw data were written but validation or derived computation failed.

Failure must leave the tracked ledger unchanged and surface a non-zero workflow result. It must never fall back to a different candidate, parameter, universe, price convention, or benchmark.

## Evaluation maturity and minimum evidence

Maturity is evidence availability, not a performance gate. The states are `proposed`, `collecting`, `minimum_evidence_met`, `evaluation_due`, and `closed`. Interim metrics remain descriptive and cannot authorize tuning.

Before activation, a separate approval must preregister:

- minimum completed trades;
- minimum elapsed trading sessions;
- minimum distinct calendar-month coverage;
- treatment of open trades and missing sessions; and
- the evaluation date or deterministic evaluation trigger.

Those values must be chosen without reading OOS outcomes. The decision may use operational cadence and the already-known expected trade frequency solely to estimate how long collection will take; it may not optimize a threshold against realized OOS performance. The evaluation becomes mature only when all preregistered evidence dimensions pass. Any later threshold change creates a new evaluation-protocol version and does not retroactively mature the old snapshot.

The exact thresholds remain a blocking item because this design PR was not authorized to select them silently.

## Future implementation PR: exact bounded scope

The next PR should:

1. add a candidate-only append ledger and canonical JSON/JSONL serializer;
2. add manifest loading, activation-event creation, fingerprint checks, provenance capture, idempotency, conflict detection, and correction-chain validation;
3. expose the already-computed Daily Scan decision/lifecycle state to a read-only OOS collector without changing strategy calculations;
4. make Daily Scan the sole owner of `docs/data/oos/**` and stage only validated append-only changes; Backtest Only must neither generate nor rewrite OOS data;
5. add derived snapshot generation from the immutable effective raw view;
6. add a fail-closed workflow path on `main` only, with concurrency compatible with `data-publish-main`;
7. preregister the approved maturity thresholds and then write the activation event using the real merge facts; and
8. add deterministic fixtures and tests without downloading data or running the ten-year backtest.

It must not change signals, entry/exit/stop/hold logic, costs, universe rules, gates, ranking, candidate selection, UI, or portfolio economics. It must begin with empty OOS storage and cannot import any existing `docs/data` observation as OOS.

## Test plan

This design PR validates required fields, boolean invariants, identifier formats, unique strategy keys, blocker/status consistency, commit SHA shape, parameter fingerprint integrity, and deterministic JSON serialization.

The implementation PR must additionally test:

- exact ex-ante/ex-post separation and schema validation;
- first-run activation and rejection of pre-merge data;
- identical retry no-op and conflicting retry failure;
- duplicate economic event rejection;
- append-only filesystem diff enforcement;
- correction chains and deterministic effective views;
- parameter, callable-source, universe, cost, benchmark, and portfolio drift failures;
- missing/partial input failure with no ledger mutation;
- stable derived metrics from shuffled input order;
- new-cohort creation without old-cohort mutation; and
- Daily Scan ownership with Backtest Only non-ownership.

## Operations and rollback

Activation requires a reviewed implementation PR, real merge facts, approved maturity thresholds, and a green empty-ledger test. Normal operation appends one validated batch, derives a versioned snapshot, commits both atomically, and reports counts, hashes, and cutoff IDs.

To suspend collection, disable the collector and append a suspension event; do not delete observations. A code rollback may stop future appends but may not rewrite history. Resume the same cohort only when all frozen fingerprints still match and the collection gap is recorded. Otherwise close the cohort and create a new one.

For a bad observation, append a correction. For compromised provenance, append an incident/closure event and exclude affected records through the effective view. Repository recovery may revert collector code, but any revert that would remove committed OOS records is prohibited; the records must remain and be superseded by auditable corrections.

## Scope confirmation

This PR adds only this design, `config/oos_evaluation_manifest.json`, and its schema test. The existing PR #16 immutability guard is narrowed only to permit that explicitly authorized new manifest while continuing to protect every other config file, production output, UI path, and data workflow. It changes no production code, workflow, UI, strategy, cost, universe, gate, ranking, generated `docs/data`, or portfolio economics, and it produces no OOS observation.
