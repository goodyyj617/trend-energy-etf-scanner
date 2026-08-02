# Trend Strategy v2 Foundation 7

## Scope

Foundation 7 makes four selected, finite robustness checks executable over an
already completed `StrategyRun`: fixed-strategy walk-forward, leave-one-year-out
(LOYO), paired moving-block bootstrap, and canonical cost stress. It adds no
optimizer, parameter reselection, arbitrary formula, data download, remote
worker, production approval, White Reality Check, Hansen SPA, DSR, PBO, or
unrestricted simulation grid.

## Contracts and bounds

`robustness_execution_plan_v1` is normalized from a completed base run and
records the source snapshot, economic/benchmark artifact hashes, engine and
commit, catalog/policy hashes, methods, exact scenarios, seed, settings and
work estimate. Its content-derived plan hash is immutable. Operational work is
separate in `robustness_execution_manifest_v1`, `robustness_scenario_v1`, and
`robustness_scenario_result_v1`; a completed economic manifest is never
changed. Generated evidence is `robustness_summary_v2`, with an explicit
`robustness_summary_v1` compatibility section instead of a silent v1 change.

`robustness_execution_policy_v1` limits folds (8), LOYO scenarios (12),
bootstrap samples (400), block length (2--20), cost scenarios (4), economic
reruns (64), combined units (512), date span (4,000 days), body size (256 KiB)
and local concurrency (1). Confirmation is hash-bound at 64 units; a hard
limit cannot be confirmed.

The option catalog is `robustness_option_catalog_v1`, the single API/UI
definition for Korean and English names, method versions, supported veto
fields, defaults, workload units and deferred methods.

## Method semantics

Walk-forward uses explicit ordered folds with exact economic-date training and
test ranges, nonnegative embargo, rolling or expanding label, minimum training
and test observations, and an explicit `incomplete` or `skip` boundary policy.
It evaluates the fixed normalized strategy; it is not nested model selection.
The evidence includes fold count, completed/passed/incomplete counts, pass
ratio, worst and median fold, ranges and per-fold result records.

LOYO derives only explicit calendar years. First/last observed years are
flagged partial; partial years are excluded unless the plan explicitly permits
flagged use. Removing a year that leaves insufficient observations is
`incomplete`, not a pass. Evidence retains exclusion ranges, year outcome,
reversing years (opposite sign versus the full period), stability ratio, and
incomplete years.

Bootstrap aligns strategy and benchmark on exact common economic dates and
resamples those paired daily-return tuples with a deterministic circular moving
block. The effect is the paired mean daily-return difference. It stores seed,
sample count, block length/method, two-sided raw p-value and percentile CI. An
adjusted p-value remains null until an explicit correction; `holm_adjust`
records the finite declared hypothesis family, ordering, raw and adjusted
values, and correction hash.

Cost stress creates only explicit nonnegative multipliers. It calls the
registered canonical economic runner; no generic CAGR haircut is used. The
stored result retains the multiplier, engine-produced stressed metrics and
survival result.

## Work, lifecycle, recovery and cache

The estimator reports separate economic, fold, LOYO, cost-stress, bootstrap
resample, deterministic-metric and EvaluationProfile units. Bootstrap samples
are not economic backtests. Evidence identity includes base run, economic and
benchmark hashes, plan hash, method/version/settings, seed and engine. Changing
only an EvaluationProfile threshold does not change that evidence identity.

Scenario state is explicit: `pending`, `running`, `reused`, `succeeded`,
`failed`, `cancelled`, `skipped`, `blocked`, or `incomplete`. Each record has
ordinal, settings hash, seed, ownership, timestamps, provenance, artifact
references, failure and incomplete reasons. Restart reconciliation converts a
running scenario without a trustworthy owner to `blocked`; resume restarts the
deterministic bootstrap from its seed rather than claiming sample checkpoints.
Corrupt, missing, incomplete, unsupported or provenance-mismatched evidence is
not treated as passing evidence.

## API and UI

The local API exposes `GET /api/v1/robustness/options`, `POST
/api/v1/robustness/normalize`, `/estimate`, and `/plans`, `POST
/api/v1/robustness/plans/{id}/start`, and `GET
/api/v1/robustness/plans/{id}/evidence`. Write routes remain bounded, JSON-only,
loopback-only through the existing server boundary, and return Korean-first
stable errors. The existing Korean robustness view remains method-separated;
the catalog/evidence payload supplies fold, LOYO, bootstrap and cost-stress
tables rather than a hidden combined score.

## Next phase

Foundation 8 may add only statistically justified additional diagnostics and
end-to-end user-acceptance work, without enabling unrestricted optimization.
