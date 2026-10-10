# Trend Strategy v2 Decisions

## Settled decisions

All decisions in this section have status `Accepted`.

1. Existing collector implementation and activation work is paused.
2. PR #18 and PR #24 remain as baseline preregistration records and are not deleted.
3. The existing v1 cohort will not be activated during Trend Strategy v2 research.
4. The existing scanner Boolean is considered too entangled to serve as the canonical v2 architecture.
5. The trend-energy score, score-breakout trigger, and score-lookback grid are retired from the primary Trend v2 research path.
6. R20 and ER20 threshold grids will not be retained as independent v2 grids.
7. Fixed holding-period exits are prohibited in v2.
8. Complete Cartesian parameter search is prohibited.
9. SPY-relative portfolio return and downside risk are primary research objectives.
10. Codex must use repository context files instead of relying on long chat history.
11. The primary deliverable is a reusable web backtest and strategy-comparison tool, not a one-off preferred strategy result.
12. The user must be able to configure signals and backtest rules from the web UI.
13. Strategy execution results and evaluation criteria must be separate so unchanged backtests can be re-evaluated without rerunning.
14. The default comparison method is non-compensatory: configurable gates, Pareto selection, epsilon tolerance, robustness vetoes, and lexicographic tie-breaking.
15. The UI may provide user-adjustable metric weights only as a separately labeled exploratory comparison mode; weighted rankings must not override mandatory gates or the default Pareto/robustness result.
16. The visible web interface is Korean-first.
17. Every acronym and metric must have a dedicated explanation with formulas, variable definitions, numerical examples, interpretation, assumptions, and limitations.
18. Strategy runs, evaluation profiles, and evaluation runs must be versioned, hashed, and preserved in history to expose threshold or weight changes made after observing results.

### Research lab decisions (2026-10-08)

19. Build a new single-screen research lab (`lab/`, Streamlit) instead of extending the Foundation UI. The Foundation code is preserved but no longer the primary product path.
20. Portfolio-level metrics compared with SPY on the same dates are primary. Trade-level metrics (t-statistic of mean trade return, Profit Factor, win rate, median trade return) are shown as secondary evidence of a per-trade statistical edge.
21. Supersedes the Cartesian-product prohibition in decision 8 for the lab: a two-dimensional parameter grid (normally one signal parameter x one exit parameter, at most 12 values per axis) is allowed for robustness analysis. Unrestricted multi-dimensional search remains prohibited.
22. Strategy blocks start from price and volume with minimal derived indicators (moving average, prior N-day high/low, average volume), one numeric parameter per block. Finer refinement comes later.
23. ETFs first; individual stocks are allowed through the yfinance path, with an explicit survivorship-bias warning.
24. Lab robustness uses no weighted score: pass/fail gates, neighbor survival, connected passing regions, leave-one-year-out re-checks, then lexicographic ordering (region size, neighbor survival, LOYO pass share, Calmar).
25. Default lab gates come from CHARTER.md: CAGR >= 0.80 x SPY CAGR, |MDD| <= 0.75 x |SPY MDD|, at least 30 completed trades. Gates are not lowered automatically when no candidate passes.

### Research lab test bench (2026-10-09)

26. Uninvested cash earns a T-bill return by default: BIL total return, and before BIL existed (2007-05) the 13-week T-bill rate (^IRX) / 252. A 0% option remains for comparison.
27. ETF universes go through a recorded filter funnel: cash-like ETFs (Morningstar "Ultrashort Bond" / money market) are never traded and serve only as the cash proxy; default asset classes exclude bonds; AUM >= $1B; expense ratio <= 0.75%; near-duplicates (daily-return correlation >= 0.98, computed only on pre-holdout data) keep the larger fund. Metadata is fetched once into `lab/etf_meta.csv` and is current-day data (survivorship bias acknowledged).
28. A holdout period (default start 2024-01-01) is excluded from all research runs. Only saved strategies can be evaluated on it, unchanged; every evaluation and every change to the holdout setting is logged.
29. Every distinct strategy evaluated (single runs, grid cells, checklist variants) is logged; the count and the spread of their Sharpe ratios feed the Deflated Sharpe Ratio.
30. Every backtest is shown next to three baselines on the same dates, costs and cash: SPY buy-and-hold, SPY 200-day moving-average timing, and an equal-weight daily-rebalanced universe.
31. Single-strategy robustness is a 7-item pass/fail checklist with no weights: gates, neighbor parameters (> 50% of +/-1-step variants pass), 2x cost, LOYO, both halves, Calmar above SPY 200-day timing, DSR >= 0.95.
32. Strategy blocks may be fixed rules with no numeric parameter (moving-average stack, up candle); such blocks are excluded from grids.

### Strategy-family comparison (2026-10-09)

33. Pre-registered family menu `family-menu-v1`, fixed before any results were seen:
    entries breakout N {20, 30, 50, 80, 120, 200}, above_ma N {50, 80, 100, 150, 200, 250},
    momentum N {21, 42, 63, 126, 189, 252}; exits low_break N {10, 15, 20, 30, 50, 80},
    below_ma N {20, 30, 50, 100, 150, 200}, atr_trail k {1.5, 2, 2.5, 3, 4, 5}.
    Values cover short/medium/long horizons with roughly geometric spacing. Changing the menu after
    seeing results is a new pre-registration (new version), not an edit.
34. Families are compared by whole-grid behaviour, ordered lexicographically by Gate pass share,
    largest connected passing region, share of cells with Calmar above SPY 200-day timing, then
    median Calmar. Lower-quartile Calmar, median CAGR/MDD and median trade count are shown but do
    not order. Every cell is logged as a trial.

### Confirmation-condition test (2026-10-09)

35. A family is refined by adding ONE entry block at a time at that block's default value
    (no tuning of the added parameter), rerunning the family's 6x6 grid. A variant is an
    improvement only if it is a Pareto improvement on the four family ordering metrics.
36. An added condition is adopted only if it is an improvement on every universe tested
    (here: ETF snapshot 2017-2023 and long-history multi-asset 2000-2023), and only if the
    improvement is material (simpler rules preferred, CHARTER principle 8).
    Finalists for refinement are the families with the lowest sum of family-comparison ranks
    across universes. Results: `docs/research/lab/2026-10-09_family_comparison_and_refinement.md`.
37. New block: market trend (SPY close > SPY N-day SMA), a regime filter for new entries.

### Holdout, criteria review, sizing (2026-10-10)

38. F1 (120d breakout + 126d momentum / 80d low exit, equal sizing, long-history universe)
    was evaluated once on the holdout (2024-01-02 - 2026-10-08) under the unchanged criteria and
    failed (return 0.71 x SPY, drawdown 0.82 x SPY). F1 is not adopted.
39. The return gate (CAGR >= 0.80 x SPY CAGR) is retained by user decision.
40. Checklist "halves" v2: in each half of the research period, strategy Calmar >= SPY Calmar.
    v1 (full gate in each half) passed 0 of 49 gate-passing cells and re-imposed the
    full-period return requirement on bull-only halves.
41. DSR uses the effective number of trials N_eff = rho + (1 - rho) * N, where rho is the mean
    pairwise correlation of the trials' monthly returns (pairs with >= 24 overlapping months).
    Monthly trial returns are stored in lab_results/trial_returns.pkl. Threshold 0.95 unchanged.
42. The holdout verdict uses the return and drawdown gates only; the trade count of a short
    holdout is shown as reliability information.
43. Inverse-volatility sizing (target = equity / K x min(median vol / own vol, 2), 60-day vol)
    met its pre-declared adoption rule (Pareto improvement for >= 5 of 9 families in both
    universes: 7 and 6) and is the app's default sizing; equal sizing remains selectable.
    Record: docs/research/lab/2026-10-10_holdout_criteria_sizing.md.

### Legacy operations (2026-10-10)

44. The v1 Backtest Only workflow no longer runs on a schedule; it is manual-only
    (workflow_dispatch). v1 is frozen and no longer on the research path, each run took 3-5 hours,
    and runs that overlapped a merge to main correctly refused to publish (failed 2026-10-09 and
    2026-10-10). The published v1 outputs in docs/data stay as they are. The OOS manifest
    records the PR #18 baseline blob of the workflow as history; the OOS collector was never
    activated, so no cohort is affected. Daily ETF Scan is unchanged.

### Wider family menu and common add-on test (2026-10-10, registered before running)

The research goal is breadth: test many filters, entries and exits and judge which kinds work.
Components are judged by whole-grid behaviour and consistency across universes; the DSR stays
the bar for a final single strategy only.

45. Pre-registered menu `family-menu-v2`, fixed before any v2 result was seen. v1 values unchanged, plus
    entries ma_cross N {40, 60, 100, 150, 200, 250}, bollinger k {1.0, 1.25, 1.5, 2.0, 2.5, 3.0},
    rsi_min X {50, 55, 60, 65, 70, 75}, williams X {-50, -40, -30, -20, -10, -5}; exit trailing_pct
    X {5, 8, 10, 15, 20, 30}. 7 entries x 4 exits = 28 families, 36 cells each. Ordering as in 34;
    families are ranked across universes by the sum of their ranks.
46. New block ma_cross: SMA(round(N/4)) > SMA(N). The short window is fixed at a quarter of the long
    one (50/200 is the classic golden cross) to keep one parameter per block.
47. Common add-on test: one block at its default value is added to every family (a family built
    on that same block is skipped) and each family's grid is compared with the plain family by the
    Pareto verdict of 35. A component is "generally helpful" when it improves a majority of the
    tested families in every universe, "generally harmful" when it worsens a majority in every
    universe, otherwise inconclusive. First component tested: initial stop (stop_loss_pct 10%),
    the open "initial-stop families" question. Universes and settings: those of 43 (inverse-vol
    sizing, ETF snapshot 2017-2023 and long-history multi-asset 2000-2023).

## Open decisions

- final trend-filter definition;
- first non-score signal families exposed in the web strategy builder;
- Phase B entry families;
- Phase B initial-stop families;
- Phase B trailing-exit families;
- exact research and final evaluation-profile defaults;
- result-retention limits and external artifact storage;
- position-sizing and concentration-control design;
- point-in-time universe remediation.
