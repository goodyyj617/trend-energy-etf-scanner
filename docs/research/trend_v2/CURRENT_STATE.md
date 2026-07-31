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

## Current phase

Product foundation -- strategy runs, evaluation profiles, reusable result storage, and configurable comparison contracts

## Phase A2 record

- Formal empirical score-breakout classification: `Inconclusive`.
- Operational decision: do not retain the score or score-breakout trigger in the primary research path.
- Frozen economic dates: 2016-08-01 through 2026-07-30.
- Frozen snapshot SHA-256: `b37025ba30f5ecc2d32c53797201b8fcf43c06f539e6414ed4d4cdd9f70b82bd`.
- This remains provisional in-sample research, not production approval or genuine OOS evidence.

## Exact next task

Foundation 1 -- define and implement versioned contracts for:

1. `StrategyRun`;
2. `EvaluationProfile`;
3. `EvaluationRun`;
4. compact persistent result storage;
5. hash-based cache identity;
6. configurable metric, gate, Pareto, epsilon, robustness, and optional weighted-view settings.

This phase must not run a new strategy search. It must first create the reusable tool architecture described in `PRODUCT_REQUIREMENTS.md`.

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