# Trend Strategy v2 Foundation 10A

## Evaluation Profile Studio

Foundation 10A adds a Korean-first local Evaluation Profile Studio. It reads
saved profiles, clones a selected profile, validates only bounded allow-listed
settings, and saves a new immutable `evaluation_profile_v2` record. Existing
`evaluation_profile_v1` bytes, hashes, profile IDs, and EvaluationRuns remain
unchanged.

Each Studio revision records its root profile, parent profile, generated
technical revision name, revision number, timestamp, Korean change summary,
and changed field groups. The ResultStore derives history from immutable
`evaluation_profiles` records; it never rewrites an existing profile.

Metric choices are supplied by the existing Metric Registry and retain its
direction, representation, and suitability limits. Behavior deduplication is
limited to the existing correlation/Jaccard/path-distance diagnostics and the
existing simplicity fields. Arbitrary formulas, code, paths, imports, and URLs
are not accepted.

The Studio validates before saving and requires an idempotency key for save or
apply operations. Applying a profile targets one stored StrategyRun and uses
the established stored-artifact evaluation path. It can reuse or calculate
derived metrics from saved evidence, but it never starts an economic backtest
and does not change StrategyRun identity.

The default decision pipeline remains mandatory gates, epsilon-Pareto,
robustness vetoes, lexicographic tie-breaking, and behavior deduplication.
Exploratory metric weights are visually and semantically separate; their
output cannot override the default pipeline.

## Boundaries

The Studio remains loopback-only and same-origin. It adds no remote hosting,
authentication, strategy family, optimization, market-data request, generated
data update, or Foundation 10B/10C/10D implementation.
