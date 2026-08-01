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
  execution-attempt or job-status contract for pending and running work is
  required before the web API starts background backtests.

## Current phase

Product foundation -- local web API and saved-run registry

## Phase A2 record

- Formal empirical score-breakout classification: `Inconclusive`.
- Operational decision: do not retain the score or score-breakout trigger in the primary research path.
- Frozen economic dates: 2016-08-01 through 2026-07-30.
- Frozen snapshot SHA-256: `b37025ba30f5ecc2d32c53797201b8fcf43c06f539e6414ed4d4cdd9f70b82bd`.
- This remains provisional in-sample research, not production approval or genuine OOS evidence.

## Exact next task

Foundation 3 -- local web API and saved-run registry:

1. expose local API endpoints for saved StrategyRuns, derived artifacts,
   EvaluationProfiles, EvaluationRuns, provenance, and retention status;
2. add a saved-run registry with filtering, pagination, immutable history, and
   explicit execution-attempt/job status separate from terminal manifests;
3. expose Foundation 2 calculation and evaluation as a local job without
   coupling it to a new economic backtest;
4. keep the API Korean-first ready, but do not build the final web UI in the
   API phase.

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

## Explicitly deferred

- new signal-family screening;
- entry, stop, and exit optimization;
- position-sizing optimization;
- point-in-time universe reconstruction;
- new OOS cohort creation;
- OOS collector implementation or activation;
- production deployment.
