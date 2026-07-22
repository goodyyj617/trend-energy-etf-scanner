# Bounded Strategy-Year Aggregate Design

## Status and scope

This change adds instrumentation for a later redesign of the Time Robustness Gate. It does not change signal, entry, exit, stop, transaction-cost, universe, backtest-period, qualification-gate, ranking, or dashboard behavior.

The generated output is:

`docs/data/backtest_strategy_year_summary.csv`

It is produced from completed-trade records already held in memory after simulation. No strategy is simulated again, and no raw event-level trade file is committed.

## Row grain and expected size

The row grain is one observed:

`strategy_key × entry_year`

Only combinations with at least one completed trade are emitted. Empty strategy-year combinations are not materialized.

The current grid has 540 strategies and approximately ten observed entry years, so the expected maximum is about 5,400 rows. This is bounded by the strategy grid and the downloaded backtest horizon rather than by the millions of underlying completed-trade events.

## Entry-year assignment

`year_basis` is always `entry_year`.

Each completed trade is assigned to the calendar year of `entry_date`. `exit_date` does not determine the cohort. A trade entered on 2025-12-31 and exited in 2026 remains in entry year 2025. This matches the existing Entry-Year Stability convention and allows fully realized exits after the cohort boundary.

## Schema

### Identification

| Column | Definition |
|---|---|
| `strategy_key` | Stable strategy identifier. |
| `strategy_label` | Existing human-readable strategy label. |
| `score_lookback` | Score Breakout lookback parameter. |
| `r20_min` | R20 minimum parameter. |
| `er20_min` | ER20 minimum parameter. |
| `entry_rule` | Existing entry-rule key. |
| `exit_rule` | Existing exit-rule key. |
| `entry_year` | Calendar year derived from the completed trade's entry date. |
| `year_basis` | Constant `entry_year`. |

### Year coverage

| Column | Definition |
|---|---|
| `year_period_start` | Later of January 1 for `entry_year` and the earliest completed-trade entry date available in the run. |
| `year_period_end` | Earlier of December 31 for `entry_year` and the latest completed-trade entry date available in the run. |
| `first_entry_date` | Earliest actual entry date in this strategy-year row. |
| `last_entry_date` | Latest actual entry date in this strategy-year row. |
| `first_exit_date` | Earliest realized exit date among trades assigned to this entry year. |
| `last_exit_date` | Latest realized exit date among trades assigned to this entry year. |
| `is_partial_year` | True when the clipped available entry period does not cover January 1 through December 31 of the entry year. |
| `is_full_calendar_year` | True only when `year_period_start` is January 1 and `year_period_end` is December 31. It is the inverse of `is_partial_year`. |

The coverage calculation uses the available completed-trade entry period for the run, matching the existing bounded period-analysis semantics. It does not infer completeness from the number of trades. A partial year can contain many trades and still remain partial.

For an ordinary multi-year run, interior years are full calendar years and the first and last observed entry years are partial unless their available entry period covers the full calendar boundaries.

### Counts and return aggregates

All returns are completed net trade returns after the existing round-trip transaction cost.

| Column | Definition |
|---|---|
| `completed_trades` | Number of completed trade records in the strategy-entry-year cohort. |
| `winning_trades` | Count with `net_return > 0`. |
| `losing_trades` | Count with `net_return < 0`. |
| `flat_trades` | Count with `net_return == 0`. |
| `gross_profit` | Sum of positive completed net trade returns. |
| `gross_loss_abs` | Absolute value of the sum of negative completed net trade returns. |
| `sum_trade_returns` | Arithmetic sum of completed net trade returns. It is not a portfolio return. |
| `avg_trade_return` | Arithmetic mean of completed net trade returns. |
| `median_trade_return` | Median completed net trade return. Diagnostic only. |
| `profit_factor` | `gross_profit / gross_loss_abs`. |
| `win_rate` | `winning_trades / completed_trades`; flat trades remain in the denominator. |
| `p10_trade_return` | Pandas linear-interpolation 10th percentile of completed net trade returns, matching the existing `tail_return_10` convention. |
| `worst_trade_return` | Minimum completed net trade return. |
| `best_trade_return` | Maximum completed net trade return. |
| `avg_holding_days` | Arithmetic mean of existing completed-trade holding days. |

### Zero-loss Profit Factor

The annual output uses the same convention as the overall strategy summary:

- if `gross_loss_abs > 0`, Profit Factor is `gross_profit / gross_loss_abs`;
- if there are positive returns and no negative returns, Profit Factor is missing/NaN rather than infinity;
- if there are neither positive nor negative returns, Profit Factor is `0.0`.

CSV serialization leaves the NaN zero-loss value blank, consistent with other generated pandas CSV outputs.

## Reconstruction guarantees

For each strategy, aggregating all emitted annual rows must reconstruct the existing overall completed-trade population:

- annual `completed_trades` sums to overall `completed_trades`;
- annual winning, losing, and flat counts sum to the overall completed count;
- annual `sum_trade_returns` sums to overall `sum_trade_returns`;
- annual `gross_profit` and `gross_loss_abs` reconstruct the numerator and denominator of overall Profit Factor;
- total annual sum divided by total annual completed trades reconstructs overall `avg_trade_return`;
- reconstructed Profit Factor matches overall `profit_factor`, including the zero-loss convention.

Floating-point reconstruction tests use relative tolerance `1e-12` and absolute tolerance `1e-10`. The tolerance permits harmless changes in summation order from grouped aggregation without weakening the accounting identity.

Median, percentiles, minimum, and maximum are cohort-level distribution statistics and are not additively reconstructable. Deterministic fixture tests verify their direct calculation.

The aggregation operates on a copy of the required completed-trade columns. Tests verify that trade rows, skipped rows, strategy counts, and existing summary values are unchanged.

## Summary metadata

`backtest_summary.json` receives additive bounded metadata:

- `strategy_year_summary_row_count`
- `strategy_year_min`
- `strategy_year_max`
- `strategy_year_basis`
- `partial_years`
- `full_calendar_years`
- `timing_metadata.strategy_year_aggregation_sec`

The existing `period_analysis[].yearly_details`, Entry-Year Stability calculations, gates, tiers, and schema version remain unchanged.

The runtime is also logged separately as:

`backtest_timing strategy_year_aggregation_sec=<seconds>`

## Performance design

The implementation selects only the required columns from the existing completed-trade DataFrame, computes simple vector columns once, and uses a grouped aggregation over `strategy_key` and `entry_year`. The 10th percentile is joined from a grouped quantile calculation. Strategy parameters are parsed only from one identity row per strategy, not once per trade.

This adds one bounded aggregation pass and no simulation pass. Development validation uses deterministic fixtures and unit tests; the full 10-year Backtest Only workflow is intentionally not run in this implementation PR.

## Publication and storage safety

The Backtest Only workflow explicitly stages `backtest_strategy_year_summary.csv`. The existing preflight rejects raw event files and any `docs/data` file larger than 45 MiB. The Daily Scan workflow does not publish this file because Backtest Only owns it.

At roughly 5,400 rows and 32 columns, the output is bounded, reviewable, and suitable for Git history. It contains aggregate statistics only and cannot expand with each underlying trade event.

One full Backtest Only run is required after this implementation is merged to generate and publish the real 10-year CSV and its metadata.

## Future Time Gate use

This output enables a later analysis to compare:

- partial versus full calendar entry years;
- annual trade-count eligibility thresholds;
- average-return and Profit-Factor definitions of a positive year;
- leave-one-year-out robustness;
- behaviorally redundant parameter states using annual outcome vectors.

This task does not approve any of those future gate definitions. In particular, annual `median_trade_return` remains a diagnostic distribution statistic and is **not** approved as the future positive-year definition.
