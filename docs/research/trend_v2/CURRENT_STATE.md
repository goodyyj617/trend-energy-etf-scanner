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

## Current phase

Product foundation -- reusable metric and selection engine

## Phase A2 record

- Formal empirical score-breakout classification: `Inconclusive`.
- Operational decision: do not retain the score or score-breakout trigger in the primary research path.
- Frozen economic dates: 2016-08-01 through 2026-07-30.
- Frozen snapshot SHA-256: `b37025ba30f5ecc2d32c53797201b8fcf43c06f539e6414ed4d4cdd9f70b82bd`.
- This remains provisional in-sample research, not production approval or genuine OOS evidence.

## Exact next task

Foundation 2 -- complete the reusable metric and selection engine integration
on the Foundation 1 contracts:

1. calculate the configured registry metrics from stored daily, yearly, and
   rolling artifacts instead of accepting precomputed summary inputs only;
2. implement reusable rolling and robustness calculations;
3. produce behavior-path deduplication inputs from stored portfolio curves;
4. integrate calculated metrics and robustness artifacts with the Foundation 1
   gate, Pareto, epsilon, veto, tie-break, and exploratory-weighted pipeline;
5. expose stable calculation-engine version boundaries and parity tests against the
   existing reliable portfolio metrics.

Foundation 2 must continue to preserve legacy v1 behavior and must not run a
new strategy search or add signal, entry, stop, or exit rules.

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

## Explicitly deferred

- new signal-family screening;
- entry, stop, and exit optimization;
- position-sizing optimization;
- point-in-time universe reconstruction;
- new OOS cohort creation;
- OOS collector implementation or activation;
- production deployment.
