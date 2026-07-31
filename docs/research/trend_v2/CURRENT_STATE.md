# Trend Strategy v2 Current State

## Current status

- PR #18 froze a v1 OOS candidate as baseline evidence.
- PR #24 defined its maturity protocol.
- The v1 OOS collector was not implemented or activated; no OOS collection is active.
- Phase A1 implemented the comparison architecture.
- Phase A2 froze adjusted daily OHLCV and completed the empirical comparison and robustness analysis.
- Legacy v1 scanner and backtest behavior remain unchanged.

## Current phase

Phase A2 complete -- empirical score-breakout classification: **Inconclusive**

## Phase A2 result

Complete evidence was supplied, but the preregistered portfolio, robustness, and control-comparability conditions do not jointly support Retain or Reject.

- Frozen economic dates: 2016-08-01 through 2026-07-30.
- Frozen snapshot SHA-256: `b37025ba30f5ecc2d32c53797201b8fcf43c06f539e6414ed4d4cdd9f70b82bd`.
- Score breakout status for Phase B: `exploratory_and_unresolved`.
- This is provisional in-sample research, not production approval or genuine OOS evidence.

## Exact next task

Phase B1 -- entry-family screening using exactly: `score_breakout_l20`, `prior_price_high_l20`.

Maximum Phase B1 signal candidates: 2.

## Blocking limitations

- The historical universe uses present-day AUM and present-day product availability, creating current-universe and survivorship bias.
- Final performance conclusions still require point-in-time universe reconstruction or a survivorship-sensitivity analysis.

## Explicitly deferred

- broad exit-family expansion beyond the Phase B sequence;
- position-sizing optimization;
- point-in-time universe reconstruction;
- new OOS cohort creation;
- OOS collector implementation or activation;
- scanner UI redesign.
