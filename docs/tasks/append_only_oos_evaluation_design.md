# Append-only out-of-sample evaluation design

## Decision and interpretation

This PR proposes an OOS preregistration contract and an immutable candidate-selection snapshot. It does not activate a cohort, implement a collector, create OOS observations, download market data, rerun Backtest Only, or change any strategy or portfolio behavior.

At the pinned selection snapshot, the **Current Backtest Qualified rank-1 candidate at the pinned selection snapshot** is:

`score_bo_l40_rm002_erp010__signal_3d_confirm__ma50`

For this contract it is the **Frozen OOS evaluation candidate**. It is **Not production-approved** and **Not validated for out-of-sample profitability**. It was **Selected by the Backtest ranking function** `src/backtest.py::rank_strategy_summary`. The unique Qualified rank-1 result is evidence for choosing a candidate for proposed forward evaluation; it is not evidence that the strategy is ready for live deployment.

The immutable selection facts are:

- artifact `as_of`: `2026-07-29`;
- generated-data commit: `e844f557820c0987eeea96424e261c6fde085a51`;
- calculation source commit recorded by `Generated-From`: `5b23b5d6070f4924e1afc53e7561c007663a0f0b`;
- selected key: `score_bo_l40_rm002_erp010__signal_3d_confirm__ma50`;
- qualification tier: `Qualified`; and
- qualification rank: `1`.

The generated-data commit is a direct child of the calculation source commit. The proposed PR #18 contract merge is a different, later provenance fact and does not exist yet. No future merge commit or timestamp is synthesized.

## Candidate identity

The candidate parameter fingerprint covers this complete resolved identity:

| Field | Frozen value |
| --- | --- |
| `strategy_key` | `score_bo_l40_rm002_erp010__signal_3d_confirm__ma50` |
| `signal_key` | `score_bo_l40_rm002_erp010` |
| `score_lookback` | `40` |
| `r20_min` | `-0.02` |
| `er20_min` | `0.10` |
| `close_filter` | `close > ma50` |
| `entry_key` | `signal_3d_confirm` |
| `exit_key` | `ma50` |
| `max_holding_days` | `63` |
| `round_trip_cost` | `0.002` |

The canonical parameter fingerprint is stored in the manifest. It is separate from the selected-row fingerprint: the former identifies the candidate definition, while the latter identifies the historical ranking evidence.

A later Backtest Only publication may produce different metrics, ranks, or even a different rank-1 key. That does not update, invalidate, or repoint `oos-0001`. A future candidate requires a new cohort; a historical selection snapshot is never silently replaced with the newest mutable output.

## Four provenance categories

The manifest separates provenance into four categories so definition drift is not confused with expected input change or operational maintenance.

### A. Frozen semantic definitions

Frozen semantic definitions are the economic and decision-level meanings of the cohort. A material change normally creates a new cohort. An explicitly reviewed migration may continue the cohort only when it proves semantic equivalence.

The frozen definitions include:

- the exact strategy key and resolved signal parameters;
- the three-session confirmation entry rule and next-session-open execution;
- the MA50 trailing exit rule and 63-session maximum holding period;
- the round-trip and portfolio-turnover transaction-cost semantics;
- feature and signal calculations;
- universe construction and daily eligibility rules;
- configured exclusions and manual overrides;
- the SPY benchmark convention;
- canonical equal-weight-active portfolio economic semantics; and
- the Yahoo Finance daily adjusted-price convention.

The manifest records a semantic snapshot, its canonical SHA-256 fingerprint, and exact source paths and Git blob IDs from calculation source commit `5b23b5d6070f4924e1afc53e7561c007663a0f0b`.

Universe semantics are frozen rules, not a permanently frozen constituent list. Membership remains dynamic by date under the frozen AUM-ranking, exclusion, override, history, price, liquidity, and liquidity-rank rules.

### B. Immutable candidate-selection evidence

Immutable candidate-selection evidence records what selected this candidate at preregistration:

- generated-data commit `e844f557820c0987eeea96424e261c6fde085a51`;
- `Generated-From` calculation source commit `5b23b5d6070f4924e1afc53e7561c007663a0f0b`;
- artifact `as_of` `2026-07-29`;
- exact Git blob IDs and content SHA-256 fingerprints for:
  - `docs/data/backtest_summary.json`;
  - `docs/data/backtest_strategy_summary.csv`; and
  - `docs/data/backtest_portfolio_curve_manifest.json`;
- the selected-row snapshot, including rank and qualification;
- the selected-row fingerprint; and
- the candidate parameter fingerprint.

These references are immutable historical evidence. Routine tests validate the snapshot embedded in the manifest and never derive the frozen candidate from the mutable current `docs/data` files.

### C. Dynamic observation inputs

Dynamic inputs are expected to change. Their content or blob changes alone are not definition drift:

- `config/aum.csv`;
- downloaded price data;
- the realized daily eligible universe;
- daily universe membership;
- daily decision inputs; and
- vendor data revisions.

`config/aum.csv` is not a hard-frozen semantic Git blob. For the pinned selection run, the manifest records its exact Git blob and content hash at the calculation source commit as historical selection-input provenance. Future observations must record the applicable AUM/input hashes and realized universe snapshot.

The same rule applies to price and vendor revisions. They are identified per observation, corrections remain append-only, and their ordinary evolution does not mutate the frozen cohort.

### D. Operational baselines

Operational baselines include:

- the Daily ETF Scan workflow and scan entry point;
- the Backtest Only workflow and calculation entry point;
- the completed-daily-bar guard;
- the publication provenance guard; and
- the future collector workflow and collector implementation.

These files are audited and versioned, but a workflow or runner blob change alone is not strategy-economic drift. The same cohort may continue through a compatibility or operational update only after an explicit compatibility check confirms that frozen semantics remain unchanged.

## Current repository operation after PRs #19 through #23

The present repository behavior is:

- Daily ETF Scan performs scan-only work and no longer imports or calls `run_backtests`.
- Daily ETF Scan owns `config/aum.csv`, `docs/data/latest.json`, `docs/data/latest.csv`, `docs/data/universe_current.csv`, `docs/data/excluded_etfs_summary.csv`, and `docs/data/history/`.
- Backtest Only is the sole producer of Backtest-owned artifacts.
- Backtest Only filters incomplete current-date daily bars before deriving `as_of`; the completion decision uses New York time and the conservative 8:15 p.m. cutoff.
- Both data-publishing workflows capture the calculation source revision before calculation.
- A generated commit records `Generated-From: <source commit>` and must be a direct child of that calculation source.
- Publication fails closed if `main` advances. Stale calculated outputs are never pulled, rebased, merged, autostashed, retried, or force-pushed onto newer code.
- Daily ETF Scan and Backtest Only retain separate output ownership.

These operational facts are current baselines, not candidate-selection evidence. Operational compatibility must be checked separately from frozen economic semantics.

## Activation model

PR #18 is design-only. Merging it cannot begin genuine OOS collection because no append-only collector or activation mechanism exists.

The activation sequence is:

1. PR #18 merges a proposed preregistration contract and immutable selection snapshot.
2. The manifest remains `proposed`, not `active`.
3. A later reviewed implementation adds the append-only collector, approved maturity thresholds, activation mechanism, and empty-ledger validation.
4. A future activation event records:
   - the actual PR #18 contract merge commit and timestamp;
   - the collector implementation/activation commit and timestamp;
   - the approved evaluation-protocol version; and
   - the first eligible ex-ante decision captured after activation.
5. OOS eligibility begins only with the first eligible ex-ante decision immutably recorded after collector activation.
6. An outcome is OOS-eligible only when it references an eligible post-activation decision record.
7. A trade or position initiated before activation is never admitted merely because it exits afterward.
8. No decision, trade, position, mark, or outcome may be retrospectively backfilled.

All activation facts remain null or unresolved in this PR. A post-activation collector may record a pre-existing position only as an excluded transition diagnostic, never as an eligible OOS record.

## Observation model

Raw observations and derived metrics are separate.

An ex-ante decision record is written only from information available at decision time and includes:

- cohort, manifest, protocol, strategy fingerprint, and collector commit;
- economic date and immutable observation timestamp;
- applicable input hashes and vendor convention;
- the realized eligible-universe snapshot hash;
- symbol eligibility and frozen signal/entry state;
- intended entry, exit, stop, membership, or no-action decision;
- the portfolio state needed to reproduce the next action; and
- a deterministic record ID and schema version.

An ex-post outcome record references one or more eligible decision IDs. It records realized entry, exit, daily mark, cost, completion reason, and source revision. Outcome information never mutates its decision snapshot.

Derived records contain portfolio marks, aggregates, benchmark comparisons, evidence counts, and evaluation statistics. Each derived snapshot identifies the complete raw/correction cutoff, computation schema, and code commit. Recalculation creates a new derived snapshot.

## Append-only storage, idempotency, and corrections

A future implementation may create only empty OOS storage:

```text
docs/data/oos/oos-0001/
  activation/events.jsonl
  raw/decisions/YYYY/YYYY-MM-DD.jsonl
  raw/outcomes/YYYY/YYYY-MM-DD.jsonl
  raw/corrections/YYYY/YYYY-MM-DD.jsonl
  derived/snapshots/YYYY/YYYY-MM-DD.json
```

Existing records are never edited or deleted. New lines and new date partitions are append-only.

The collector computes `record_id` from canonical identity fields. Replaying an identical record is a no-op. Reusing an ID with different content, or assigning different IDs to the same unique economic event, is a hard failure.

Vendor revisions, symbol corrections, corporate actions, and parsing fixes append correction records. Each correction identifies the superseded record, reason, old and new provenance, corrected payload, and its own immutable identity. Derived views follow the latest valid correction chain while retaining all history.

## Drift and compatibility

At collection time the future implementation must:

1. resolve the manifest strategy key exactly once;
2. rebuild and compare the candidate parameter fingerprint;
3. verify the frozen semantic fingerprint or an approved semantic-migration record;
4. verify the activation event, protocol version, and append-only ledger;
5. capture current dynamic input hashes and the realized universe snapshot; and
6. run an explicit compatibility check for operational-baseline changes.

Repository HEAD is expected to advance. A new generated-data blob, a new Backtest rank, a refreshed AUM file, a vendor revision, or an operational blob change is not by itself frozen-definition drift.

A changed strategy parameter, rule, cost, signal calculation, universe definition, benchmark economic convention, portfolio economic convention, or price-adjustment convention is semantic drift. Collection then fails closed until a new cohort or explicitly reviewed migration is approved.

## Fail-closed conditions

The future collector must stop without writing when:

- the manifest is not active or activation facts are unresolved;
- a strategy key is missing, duplicated, ambiguous, or fingerprint-mismatched;
- frozen semantics differ without an approved migration;
- operational compatibility has not been established;
- required input hashes, universe snapshot, market date, or decision state are missing;
- an observation predates activation or lacks an eligible ex-ante decision;
- a retry conflicts, a unique event is duplicated, dates regress, or a correction chain is invalid;
- an append would edit or delete an existing raw record; or
- validation or derived computation fails after staging.

Failure leaves the tracked ledger unchanged. The collector must never fall back to another strategy, parameter set, universe rule, price convention, or benchmark.

## Evaluation maturity

Maturity is evidence availability, not a performance gate. Before activation, a separate review must approve:

- minimum completed trades;
- minimum elapsed trading sessions;
- minimum distinct calendar-month coverage;
- treatment of open positions and missing sessions; and
- an evaluation date or deterministic evaluation trigger.

These values must be chosen without consulting OOS outcomes. The approved set receives an evaluation-protocol version. A threshold change creates a new protocol version and never retroactively matures an earlier snapshot.

This PR deliberately leaves all threshold values unresolved.

## Future implementation scope

A later reviewed implementation must:

1. add canonical JSON/JSONL serialization and an append-only candidate ledger;
2. implement manifest loading, empty-ledger validation, activation, fingerprints, provenance, idempotency, conflicts, and corrections;
3. expose already-computed decision/lifecycle state without changing strategy calculations;
4. capture dynamic input hashes and realized universe snapshots per observation;
5. build versioned derived snapshots from the immutable effective view;
6. add a fail-closed collector path compatible with existing publication concurrency;
7. register approved maturity thresholds and the real activation facts; and
8. add deterministic fixtures with no market-data download or full backtest.

It must not change signals, entries, exits, stops, holding rules, costs, universe rules, gates, ranking, selection, UI, or portfolio economics. It starts with an empty ledger and cannot import existing observations as OOS.

## Test and scope contract

This design PR tests:

- required schema and four provenance categories;
- exact pinned generated-data and calculation-source commit identities;
- unique candidate identity and internal consistency;
- deterministic canonical serialization;
- candidate parameter, semantic, and selected-row fingerprints;
- proposed status, blockers, and unresolved activation fields;
- dynamic AUM classification and absence from hard-frozen sources;
- immutable historical selection artifacts;
- snapshot independence from mutable current Backtest files; and
- terminology that prevents live-deployment implications.

The PR changes only:

- `config/oos_evaluation_manifest.json`;
- `docs/tasks/append_only_oos_evaluation_design.md`;
- `tests/test_oos_evaluation_manifest.py`; and
- the existing bounded exclusion in `tests/test_skew_aware_robustness.py`.

It changes no production code, workflow, generated `docs/data`, strategy, signal, universe, rank, qualification, transaction cost, portfolio behavior, UI, or dependency. It creates no collector, activation event, threshold values, OOS records, or generated market-data output.
