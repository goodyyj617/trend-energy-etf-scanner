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
