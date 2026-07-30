# Trend Strategy v2 Current State

## Current status

- PR #18 froze a v1 OOS candidate as baseline evidence.
- PR #24 defined its maturity protocol.
- The v1 OOS collector was not implemented or activated.
- Trend Strategy v2 research now takes priority.
- No OOS collection is active.

## Current phase

Phase 0 — persistent repository context bootstrap

## Next phase

Phase A — refactor research architecture and test score-breakout incremental value

## Phase A must accomplish

- introduce first-class FilterRule and InitialStopRule concepts;
- preserve legacy v1 behavior;
- decompose the scanner signal;
- simplify v2 numeric parameters;
- implement score-breakout incremental-value testing;
- introduce SPY-relative portfolio objectives;
- produce a Retain, Inconclusive, or Reject result for score breakout.

## Explicitly deferred

- broad entry/exit expansion;
- position-sizing optimization;
- point-in-time universe reconstruction;
- new OOS cohort creation;
- OOS collector implementation or activation;
- scanner UI redesign.

## Known limitation

The current historical universe uses present-day AUM and present-day product
availability. Final performance conclusions require a point-in-time or
survivorship-sensitivity analysis.
