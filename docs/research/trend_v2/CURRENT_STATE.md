# Trend Strategy v2 Current State

## Current status

- PR #18 froze a v1 OOS candidate as baseline evidence.
- PR #24 defined its maturity protocol.
- The v1 OOS collector was not implemented or activated; no OOS collection is active.
- Phase A1 implemented the comparison architecture.
- Phase A2 froze adjusted daily OHLCV and completed the empirical comparison and robustness analysis.
- Legacy v1 scanner and backtest behavior remain unchanged.
- The trend-energy score and score-breakout trigger are retired from the primary Trend v2 research path.
- The primary product objective is now a reusable Korean-first web backtest and strategy-comparison tool.
- Foundation 1 implements immutable strategy-run, evaluation-profile,
  evaluation-run, metric-registry, terminology, retention-policy, and bounded
  local result-store contracts without changing the legacy execution path.
- Foundation 2 makes those contracts operational from stored artifacts. It
  validates versioned daily, yearly, rolling, robustness, and behavior schemas;
  calculates and content-addresses reusable metrics; aligns benchmark metrics
  on exact common economic dates; ingests robustness evidence without rerunning
  simulations; and applies behavior clustering before final representative
  selection while preserving every StrategyRun.
- Implemented version boundaries are
  `trend_v2_stored_curve_metric_engine_v1`,
  `legacy_portfolio_metric_parity_v1`,
  `trend_v2_behavior_fingerprint_v1`, and `metric_registry_v2`.
- `StrategyRunManifest` records terminal outcomes only. A separate
  append-only `ExecutionAttempt` contract now records pending, queued, running,
  cancelling, cancelled, failed, and completed operational states without
  changing immutable economic-run status.
- Foundation 3 reconstructs `saved_run_registry_v1` deterministically from
  ResultStore and execution-attempt metadata; classifies available, missing,
  corrupt, pruned, never-generated, unsupported, and integrity-failed evidence;
  and persists only a disposable canonical index.
- Foundation 3 exposes `trend_v2_local_read_api_v1` through a bounded,
  localhost-first, read-only standard-library API. It serves saved runs,
  provenance, direct and derived artifacts, profiles, EvaluationRuns, stored
  behavior diagnostics, and execution attempts without invoking an economic
  backtest.
- Foundation 4 exposes `trend_v2_korean_saved_run_ui_v1` as a dependency-free,
  same-origin Korean-first web interface. It adds saved-run filtering and
  bounded charts, same-run evaluation-profile comparison, explicitly separated
  non-compensatory and exploratory stages, stored robustness and behavior
  inspection, operational-attempt separation, and a complete centralized
  explanation area without adding strategy execution.
- Foundation 5 adds versioned construction, decimal-safe finite parameter
  spaces, exact candidate/workload estimates, versioned local limits,
  hash-bound one-time confirmation, immutable execution requests, and a
  conservative one-worker lifecycle over append-only `execution_attempt_v1`.
- Foundation 5 executes only the established score-independent Phase A
  baseline: rising MA200 trend filter, prior-price-high L20 signal, next-open
  entry, Low20 initial/ratcheting exit, and canonical cash-constrained equal
  weight. Valid economic and Foundation 2 artifacts are reused; no unrestricted
  search or new strategy research is enabled.
- Foundation 6 adds a versioned Korean controlled option catalog, exact
  compatibility-pruned candidate estimation, and a local append-only execution
  manager with rebuildable candidate projections, conservative interruption
  recovery, leases, resume records, worker ownership, and API/UI inspection.

 - Foundation 7 adds bounded selected robustness plans and execution records for
  fixed-strategy walk-forward, LOYO, paired moving-block bootstrap, and
  canonical cost stress. Plans and evidence are separate from immutable
  StrategyRun manifests; missing or incomplete evidence remains explicit.

## Current phase

Product foundation -- canonical cost-stress execution

## Phase A2 record

- Formal empirical score-breakout classification: `Inconclusive`.
- Operational decision: do not retain the score or score-breakout trigger in the primary research path.
- Frozen economic dates: 2016-08-01 through 2026-07-30.
- Frozen snapshot SHA-256: `b37025ba30f5ecc2d32c53797201b8fcf43c06f539e6414ed4d4cdd9f70b82bd`.
- This remains provisional in-sample research, not production approval or genuine OOS evidence.

## Exact next task

Foundation 9B -- local acceptance and startup/recovery hardening, without
unrestricted optimization.

## Required product direction

- The user will configure signals and backtest rules from the web UI.
- The user will configure what strategic superiority means, including mandatory gates, Pareto objectives, epsilon tolerances, robustness requirements, tie-break order, and optional exploratory metric weights.
- The default decision mode remains non-compensatory; weighted comparison is a separate user-defined view.
- The visible UI will be Korean-first.
- A dedicated explanation tab will document every acronym and metric with exact formulas and numerical examples.
- An unchanged `StrategyRun` must be reusable under multiple saved evaluation profiles without rerunning the backtest.

## Blocking limitations

- The historical universe uses present-day AUM and present-day product availability, creating current-universe and survivorship bias.
- Final performance conclusions still require point-in-time universe reconstruction or a survivorship-sensitivity analysis.
- Storage limits and the boundary between Git-tracked summaries and external/local large artifacts remain an open design decision.
- First and last stored calendar years are conservatively marked incomplete
  because Foundation 2 does not introduce an exchange-calendar service.
- Robustness simulations are not recomputed; missing evidence fails referenced
  vetoes closed with a stored reason.
- External object storage and distributed calculation remain deferred.
- The local API deliberately has no authentication and must remain loopback-only
  until a later remote-access security design is approved.
- Foundation 3 records retention-pruned state but does not automatically delete
  retained artifact bytes.

## Explicitly deferred

- new signal-family screening;
- entry, stop, and exit optimization;
- position-sizing optimization;
- point-in-time universe reconstruction;
- new OOS cohort creation;
- OOS collector implementation or activation;
- production deployment.
- unrestricted strategy execution, distributed queues, remote workers, and
  cloud scheduling.
- mid-candidate process interruption; Foundation 5 cancellation is cooperative
  at safe local boundaries.
- generated walk-forward and robustness simulations in the controlled adapter;
  requested work is estimated for preview, nonzero execution requests are
  refused, and missing evidence is not fabricated.
