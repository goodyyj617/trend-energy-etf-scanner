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
- The Backtest tab currently focuses on the Score Breakout signal and includes a Robustness Dashboard, forward-return diagnostics, strategy summaries, and recent trades.
- Backtest return and excursion fields are displayed as percentages; for example, `0.023` is shown as `2.3%`.
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

1. Signal forward-return diagnostics: determine whether the signal has useful forward expectancy before judging trade construction.
2. Parameter robustness: prefer neighborhoods where nearby lookback, R20, and ER20 settings also perform well.
3. Entry rule comparison: compare timing/confirmation rules within robust signal neighborhoods.
4. Exit rule comparison: evaluate downside protection, return distribution, drawdown, stop-hit behavior, and preservation of favorable excursions.
5. Benchmark alpha and portfolio validation: later test benchmark-relative alpha, overlapping positions, capital allocation, exposure, and portfolio CAGR/drawdown.

Event-level compounded `total_return` is a diagnostic convenience, not final portfolio performance. Consider sample size, median and average returns, win rate, MFE/MAE, drawdown, parameter stability, costs, and regime coverage together.

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
