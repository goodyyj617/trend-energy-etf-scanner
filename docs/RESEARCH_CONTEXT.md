# Research Context

## Project Goal

Build a GitHub Pages ETF scanner and event backtester that identifies robust trend-following rules, rather than one lucky high backtest result. The research should determine which signal to follow, which entry rule is robust, which price-based exit protects downside while preserving trend alpha, how much alpha exists versus a benchmark, and whether results remain strong across nearby parameter values.

## How Classic and Codex Should Collaborate

- ChatGPT Classic is mainly for strategy interpretation, research design, and deciding the next experiment.
- Codex is mainly for implementation, refactoring, local checks, workflow fixes, and multi-file edits.
- The repository and this file are the shared source of truth between sessions.
- After Codex work, provide a handoff summary that can be pasted back into Classic.
- Update this file when an implemented decision, constraint, or research priority changes.

## Current Implemented State

- Scanner, Active Signals, and Backtest tabs are implemented.
- The Backtest tab focuses on Score Breakout strategies evaluated from completed, exit-applied trades after entry, price stop, maximum holding period, and round-trip cost.
- Signal Forward Return Diagnostics are no longer rendered in the default Backtest UI. Their generator remains temporarily for compatibility but does not affect the Candidate Snapshot, ranking, gates, or Robustness Tier.
- The default Backtest UI includes an event-level warning, entry-period presets/status, Candidate Snapshot, Exit-Applied Strategy Leaderboard, gate/component summaries, and collapsed detail tables.
- Backtest return and excursion fields are displayed as percentages; for example, `0.023` is shown as `2.3%`.
- Phase 1 supports static entry-date presets (All available, recent 1/3/5 years) instead of arbitrary dates so the site does not need an unbounded raw trade file.
- `src/run_backtest_only.py` runs the backtest without the daily scanner.
- `.github/workflows/backtest-only.yml` runs the backtest-only job and commits bounded summary outputs.
- Backtest logic currently lives in `src/backtest.py`; there is no separate `src/signal_rules.py` in the current repository state.

## Current Strategy Hypothesis

A breakout in a blended trend-energy score may identify ETFs entering a persistent trend. The signal is only a candidate/entry trigger. Entry confirmation variants should be compared independently, and every simulated position must use a price-based stop plus a maximum holding cap. The desired result is a stable neighborhood of useful parameters with controlled downside, not the single highest row.

## Signal Definitions

- `TE = return * efficiency ratio`.
- `score = 0.65 * TE63 + 0.35 * TE126`.
- Score Breakout becomes true when today's score exceeds its highest value over the prior `score_lookback` trading days, score is positive, `R20 >= r20_min`, `ER20 >= er20_min`, and `Close > MA50`.
- Current grid: `score_lookback = [10, 20, 40]`, `r20_min = [-0.02, 0.00, 0.02]`, and `er20_min = [0.05, 0.10, 0.15]`.

## Entry Rule Definitions

- **First signal:** enter at the next trading day's open after the signal first changes from false to true.
- **Signal 2D confirm:** enter at the next open after two consecutive true signal days.
- **Signal 3D confirm:** enter at the next open after three consecutive true signal days.
- **5D breakout after signal:** within five trading days after the first signal, require a close above the prior five-day high, then enter at the next open.

## Exit Rule Definitions

All exit variants use an initial price stop, update it as a trailing stop that does not move downward, and retain a 63-trading-day maximum holding cap.

- Signal-day Low10, trailed by Low10.
- Signal-day Low20, trailed by Low20.
- Low20 minus `0.5 * ATR20`, trailed on the same basis.
- Chandelier stop: `HHV20 - 2.5 * ATR20`.
- MA50 trailing stop.

## Backtest Interpretation Rules

Use this order of evaluation:

1. Choose the requested entry-period preset and confirm the included trade entry period and realized exit period.
2. Require enough completed trades after the selected signal, entry, price stop/exit, maximum holding period, and cost.
3. Compare completed-trade Performance and descriptive Downside.
4. Prefer candidates that remain acceptable across direct parameter neighbors and entry-calendar-year cohorts.
5. Compare entry and exit rules within those stable neighborhoods.
6. Later validate benchmark alpha and portfolio behavior with overlapping positions, capital allocation, exposure, cash, portfolio CAGR, and portfolio drawdown.

The default date semantics include completed trades whose `entry_date` is inside the requested inclusive period. Their fully realized results remain included even when `exit_date` is after the requested end date. Open/incomplete trades are excluded.

Default event-level labels are explicit: Avg Trade Ret, Median Trade Ret, Trade Win Rate, Sum of Trade Returns, and Trade-Sequence DD. Sum of Trade Returns is the arithmetic sum of completed net trade returns. Trade-Sequence DD is calculated from the entry-date-ordered compounded trade sequence. Neither is a portfolio performance metric. The legacy compounded `total_return` may remain in compatibility data but is not shown as Sum of Trade Returns.

### Phase 1 Gate Configuration and Tiers

Gate thresholds are defined once in `src/backtest.py`, serialized in `backtest_summary.json`, and displayed in the UI. Starting decision thresholds are:

- `min_completed_trades = 100`
- `min_bucket_trades = 10`
- `min_eligible_neighbors = 2`
- `min_profit_factor = 1.0`
- `min_median_trade_return = 0.0`
- `min_neighbor_pass_ratio = 0.60`
- `min_positive_year_ratio = 0.60`
- `min_positive_regime_ratio = 0.50`

These are transparent, revisable research decisions, not statistical proof. Phase 1 uses sample size, Median Trade Ret, and Profit Factor as mandatory gates. Trade-Sequence DD, Worst Trade Ret, and 10th-percentile Trade Ret are descriptive rather than hard-fail gates.

Candidate selection is deterministic: higher provisional tier, more mandatory gates passed, higher Median Trade Ret, higher Profit Factor, lower absolute Trade-Sequence DD, then more completed trades. The old weighted `robust_score` is not used.

Parameter Stability uses direct adjacent Score Breakout grid neighbors while holding entry and exit rules fixed. Time Stability uses completed outcomes grouped by entry calendar year. Regime Stability is `Not available` because SPY history is not already present in the static pipeline; adding a new benchmark dependency is deferred.

## Data / GitHub Storage Constraints

- Keep GitHub Pages outputs bounded and reviewable: JSON summaries, aggregated CSV summaries, and a limited recent-trades sample.
- Generate detailed event/trade data transiently when needed, but do not commit large raw backtest CSV files.
- Workflows must use explicit allowlists for committed outputs and a preflight file-size check.
- Full backtests may be expensive; use syntax/unit checks locally and run the full data download/backtest only when reasonable.

## Do Not Do

- Do not use signal FALSE as an exit.
- Do not reintroduce MaxHold-only / no-stop strategies.
- Do not judge strategy quality by `total_return` alone.
- Do not optimize for one best parameter only; evaluate robustness neighborhoods.
- Do not store huge raw CSV files in the repo.
- Do not assume event-backtest `total_return` is portfolio CAGR.

## Next Research Questions

1. Do Score Breakout forward returns remain positive across neighboring parameter combinations, adequate sample sizes, and different market regimes?
2. Which entry confirmation improves median return and MAE without discarding too much trend participation?
3. Which stop family best limits sudden losses while retaining MFE, after costs?
4. Are results stable across ETF groups, liquidity tiers, and time subperiods?
5. How much benchmark-relative alpha remains after realistic portfolio construction, overlapping signals, and capital constraints?

## Suggested Handoff Format

1. Files changed
2. Core changes
3. Tests/checks run
4. Risks or remaining issues
5. Next decisions for Classic ChatGPT
