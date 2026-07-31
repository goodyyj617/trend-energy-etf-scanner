# Trend Strategy v2 Search Protocol

## Phase A — Signal validity and architecture

- Separate filter, signal, entry, initial stop, trailing exit, sizing, and
  portfolio components.
- Decompose the existing scanner signal.
- Simplify the score-breakout parameter grid.
- Determine whether score breakout provides incremental value over:
  - trend filter only;
  - price breakout;
  - frequency-matched random events;
  - shifted-signal placebo.
- Keep legacy v1 behavior runnable and unchanged.

## Phase B — Behaviorally distinct trading rules

Evaluate distinct entry, initial-stop, and trailing-exit families.

Do not use fixed holding-period exits.

Use sequential screening:

1. Signal screen with common trading rules.
2. Entry screen for surviving signals.
3. Initial-stop and trailing-exit screen.
4. Portfolio and robustness screen.

Do not build the complete Cartesian product.

## Phase C — Candidate reduction

Use:

- minimum objective constraints;
- Pareto filtering;
- behavior-path deduplication;
- successive halving;
- walk-forward stability;
- leave-one-year-out stability;
- transaction-cost stress;
- asset-group stability.

Reduce the final research set to approximately three to five behaviorally
distinct candidates.

## Phase D — Decision

Only after the prior phases:

- decide whether a new OOS cohort is warranted;
- archive or retain v1 as baseline;
- preregister the selected v2 candidate before collection.
