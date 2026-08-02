# Trend Strategy v2 Foundation 6

## Scope and version boundaries

Foundation 6 adds `controlled_strategy_option_catalog_v2`,
`candidate_space_estimate_v2`, and `persisted_local_execution_manager_v1`.
It broadens only the explicit price-only library: no trend filter, close above
MA200, close above rising MA200, prior-price-high breakout with L20--L55,
next-open entry, Low20 stop/ratcheting exit, equal active weights, bounded
cost/slippage, and declared portfolio constraints. The catalog is the sole
source for API/UI option labels, units, ranges, parameter form, Korean names,
explanation references, compatibility rules, identity participation, and
adapter support status. It never accepts Python, formulas, imports, paths, or
remote URLs.

The Phase A adapter accepts the three price-only trend filters and L20--L55
prior-high lookbacks. Constraints not independently enacted by the current
canonical portfolio simulator are labelled `metadata_only`; they are not
silently translated to a different rule.

## Compatibility and estimation

The versioned, machine-readable rules reject a missing breakout lookback,
Low20 rules with fewer than 20 sessions, maximum positions above the universe,
and a sub-100% asset-group cap without group data. Errors retain a stable code,
exact Korean reason, English diagnostic, identity, recoverability and suggested
action at the API boundary.

Estimation reports raw Cartesian combinations, invalid incompatibilities,
canonical duplicates, valid economic candidates, reusable completed candidates,
new candidates, evaluation-only applications, robustness workload, total work,
and reasons grouped by rule. `EvaluationProfile` remains outside economic
identity. A catalog hash participates in every normalized construction and
estimate, so a catalog change requires a new estimate and confirmation.

## Persisted local execution manager

The manager stores immutable canonical request records and append-only,
content-hashed event records under a fixed local ResultStore-relative root.
Atomic replace is used for every write. The rebuildable projection is not a
source of truth: startup reloads and validates every request/event record.
Corrupt records fail closed.

Candidate state is separate from aggregate progress and contains deterministic
ordinal, economic specification hash, timestamps, lease, artifact references,
failure information, provenance, and one of `pending`, `reused`, `running`,
`succeeded`, `failed`, `cancelled`, `skipped`, or `blocked`. Queue order is
request candidate order. Worker ownership includes worker ID, PID, host,
timestamps/heartbeat, attempt/candidate, engine, and source identity; PID
alone is never treated as proof of ownership.

An exclusive append-only candidate lease prevents two local workers from taking
the same pending candidate. The manager neither kills PIDs nor starts remote
workers. It is bounded to one local host and conservative about liveness after
restart.

## Recovery, resume, retry, cancellation, and reconciliation

On reconstruction, a persisted `running` candidate has no trustworthy live
in-process owner and is recorded as `blocked` with `attempt_interrupted`; it is
not represented as running or successful. Reconciliation records every
decision. Resume creates new pending state events only for pending, failed,
cancelled, or interrupted/blocked candidates; completed and reused candidates
are retained. Retry remains a new Foundation 5 `ExecutionAttempt` linked to
the prior terminal attempt. Restart reconstructs state; reconcile classifies
it; resume schedules incomplete work. Cancellation remains cooperative and is
persisted by the existing attempt repository.

Artifact reconciliation is fail closed: missing/corrupt/provenance-invalid
economic evidence is never promoted to success. Valid exact `StrategyRun`
artifacts may be reused, and Foundation 2 may regenerate deterministic derived
metrics or create a missing `EvaluationRun` without an economic rerun. Partial
economic results require review.

## API and UI

New local endpoints are `GET /construction/options` (Foundation 5 payload plus
the Foundation 6 catalog), `POST /construction/compatibility`, Foundation 6
`POST /construction/estimate`, `GET /execution-manager`, `GET /workers`,
`POST /execution-requests/{id}/resume`, `POST
/execution-attempts/{id}/reconcile`, and `GET
/execution-attempts/{id}/candidates`. Existing Foundation 3--5 routes remain
compatible. Writes remain localhost-only, bounded and schema checked.

The Korean UI renders the catalog from the API with engine state and constraints,
and shows persisted manager source-of-truth, worker/recovery evidence, and the
live/stale/interrupted/recovered/reconciled distinction rather than presenting a
stale worker as active.

## Deferred

No unrestricted search, optimizer, arbitrary formula, distributed queue,
remote worker, market-data download, OOS collection, or production approval is
implemented. Walk-forward and robustness execution adapters remain Foundation
7 work.
