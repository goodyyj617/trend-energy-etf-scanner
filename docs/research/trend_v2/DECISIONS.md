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
48. Results of 45 and 47: momentum x trailing_pct has the lowest rank sum (1 + 2); the
    trailing_pct exit family ranks first on average in both universes and below_ma last. The 10%
    initial stop is inconclusive (improves 18/28 families on the long history, worsens 19/28 on the
    snapshot) and is not added by default. Trials 6,393, N_eff ~1,402.
    Record: docs/research/lab/2026-10-10_family_menu_v2.md.

### Macro regime filters (2026-10-19, registered before running)

Question: do market-regime and macro indicators improve trade timing beyond the price-only
rules? Each indicator is tested as a component across every family, as in 47, not tuned.

49. Macro data are FRED daily series that are market observations and are not revised later:
    T10Y3M (10-year minus 3-month Treasury), BAA10Y (Moody's Baa corporate minus 10-year
    Treasury), VIXCLS (VIX close). A decision on day t uses only values dated on or before the
    previous trading day (publication lag); a value older than 10 calendar days counts as missing,
    and a missing value means "condition false" (no buy for a filter, no sale for an exit).
    Research runs read the frozen file lab/macro_snapshot.csv.gz. Revised or monthly series
    (NFCI, unemployment, PMI, GDP) are excluded: FRED serves only the latest vintage, so a
    backtest would see revisions that were not known at the time.
50. New regime blocks, one parameter each, defaults fixed now: curve_ok T10Y3M > 0 (curve not
    inverted); credit_calm BAA10Y <= its 200-observation SMA (spread not above trend);
    vix_calm VIX < 25. Together with the existing market_trend (SPY > 200-day SMA) these are
    the four regimes. Each has an exact complement exit (market_off, curve_inverted,
    credit_stress, vix_high) that sells every holding when the regime turns off.
51. Test: for each of the four regimes, two forms, each as the common add-on test of 47 on the
    universes and settings of 43 (8 tests x 2 universes): (a) entry filter only (blocks new
    buys); (b) regime switch = the filter plus its complement exit (moves to cash). Verdicts per
    form follow 47: generally helpful / generally harmful / inconclusive. A macro indicator is
    said to add timing value only if one of its forms is generally helpful; whether it beats the
    price-only market_trend in the same form is reported by comparing their improve counts and
    is descriptive. A descriptive table of SPY next-day returns and 63-day forward returns in
    each regime state is shown for context and decides nothing. Thresholds and windows are not
    tuned in this round; any later tuning is a new registration. Every cell is logged as a trial.
52. Results of 51 (improve / mixed / worse of 28 families, snapshot | long): no macro indicator adds
    timing value. Yield curve and VIX are generally harmful in both forms (curve filter 6/1/21 |
    0/5/23, VIX filter 0/0/28 | 6/4/18). Credit spread is inconclusive with opposite directions
    (filter 6/0/22 | 17/11/0), the same pattern as the price-only market_trend filter (3/1/24 |
    19/7/2). The market_trend switch is the most consistent but still inconclusive (17/5/6 |
    8/20/0, no family worse on the long history). Nothing is added by default. Trials 22,521,
    N_eff ~7,049. Record: docs/research/lab/2026-10-19_macro_regime_filters.md.
53. Daily alert (operations, not research): lab/daily_alert.py, run by the Daily Alert workflow
    after each US close, reports the four regimes and the orders queued by the model portfolio
    of config/alert.yml for the next open, as a comment on a GitHub issue (Telegram optional). It
    reports signals only, no performance, so the holdout period stays unseen for research.

### Confirmation filters as common add-ons (2026-10-22, registered before running)

54. Question: do the volume and trend-confirmation filters generally help the 28 families? Three
    entry blocks are tested, each at its existing default and not tuned, as the common add-on test
    of 47 on the universes and settings of 43: volume_surge (volume > 1.5 x prior 20-day average),
    obv_up (OBV > its 10-day SMA), ma_stack (SMA20 > SMA60 > SMA120 > SMA200, no parameter).
    The fourth filter proposed with them, market_trend (SPY > 200-day SMA), was already tested in
    51-52 and is not rerun. Verdicts follow 47. A filter that is generally helpful is still not
    added by default without a separate decision. Every cell is logged as a trial.
55. Results of 54 (improve / mixed / worse of 28 families, snapshot | long): no filter is generally
    helpful; all three are inconclusive. volume_surge 7/2/19 | 15/9/4 and ma_stack 4/5/19 | 21/7/0
    point in opposite directions, the same pattern as the market_trend filter in 52; obv_up is the
    least harmful (16/8/4 | 12/10/6) but misses a majority on the long history and its median
    Calmar change is near zero. Nothing is added by default. Trials 28,569, N_eff ~8,662.
    Record: docs/research/lab/2026-10-22_confirmation_filters.md.

### Market-trend switch on the top family (2026-10-22, registered before running)

56. Candidate: the top v2 family momentum x trailing_pct with the market_trend switch of 51
    (entry filter market_trend 200 + exit market_off 200, both at their defaults), settings and
    universes of 43. In each universe the family's 6x6 grid is run with the switch and the cell
    ranked first by the grid ordering (largest region, neighbor survival, LOYO, Calmar among
    Gate-passing cells) is chosen; if no cell passes, that is the result for that universe. The
    chosen cell is run through the 7-item checklist of 31 (halves v2 of 40, DSR with N_eff of 41)
    in its own universe. Descriptive only, deciding nothing: the same cell without the switch, and
    each universe's chosen cell run on the other universe. The holdout is not used. A cell that
    passes all 7 items is reported as a holdout candidate for the user to decide on; nothing is
    adopted by this test.
57. Results of 56: snapshot 2017-2023 has no Gate-passing cell (all 36 fail the return gate,
    median CAGR 0.42 x SPY), so no candidate. Long 2000-2023: 7 of 36 pass; the top cell
    momentum 189 / trailing 10% passes 3 of 7 checklist items (gate, halves, beats SPY 200-day)
    and fails neighbors (3/8), 2x cost, LOYO (0.67) and DSR (0.26). The switch cuts the drawdown
    (-34% to -24%) but also the return, leaving the cell on the edge of the return gate. Not a
    holdout candidate; C2 (6/7) remains the strongest. Trials 28,581, N_eff ~8,665.
    Record: docs/research/lab/2026-10-22_market_switch_checklist.md.

### Exploration gate and family menu v3 (2026-10-22, registered before running)

58. Two gates, both fixed now. Exploration gate (for comparing families and components):
    strategy Calmar >= 1.0 x SPY Calmar on the same dates, |MDD| <= 0.75 x |SPY MDD|, completed
    trades >= 30; no return gate, so strategies that earn less than SPY but lose much less
    survive. LOYO under this gate re-checks the Calmar and drawdown conditions. Final gate
    (unchanged, 39): CAGR >= 0.80 x SPY, |MDD| <= 0.75 x |SPY MDD|, trades >= 30. A holdout
    candidate must pass the final gate and the 7-item checklist of 31 evaluated under the final
    gate; the final gate is not changed after these results are seen. Results show both gates.
59. Pre-registered menu `family-menu-v3`: v2 unchanged plus new price/volume blocks, one parameter
    each, other windows fixed. Entries: near_high (close >= (1 - X%) x 252-day highest close)
    X {2, 5, 8, 12, 16, 20}; adx_trend (Wilder ADX(14) >= X and +DI > -DI) X {15, 20, 25, 30, 35, 40};
    keltner (close > EMA20 + k x ATR14) k {0.5, 1.0, 1.5, 2.0, 2.5, 3.0}; cmf_min (Chaikin Money
    Flow 20 >= X) X {-0.05, 0, 0.05, 0.10, 0.15, 0.20}. Exits: momentum_neg (N-day return < 0)
    N {21, 42, 63, 126, 189, 252}; rsi_below (RSI14 < X) X {30, 35, 40, 45, 50, 55}; cmf_below
    (CMF20 < X) X {-0.25, -0.20, -0.15, -0.10, -0.05, 0}. 11 entries x 7 exits = 77 families.
60. Test: the 77-family comparison under the exploration gate, on the universes and settings of
    43. Families are ordered as in 34 and ranked across universes by rank sum. Descriptive:
    average family rank of each entry and each exit block per universe (as in 48), and each
    family's final-gate pass share. For the three families with the lowest rank sum, the top cell
    of each universe's grid (ordering of 34 under the exploration gate) is run through the
    checklist under both gates. A cell passing all 7 items under the final gate is reported as a
    holdout candidate for the user to decide on; nothing is adopted by this test.
61. Results of 60 (exploration gate): breakout x atr_trail has the lowest rank sum (1 + 6), then
    breakout x below_ma (9 + 1) and breakout x rsi_below (8 + 3). breakout is the best entry in
    both universes and trailing_pct the best exit, as in 48. Among new blocks keltner is the best
    entry (mid-upper in both), near_high and adx_trend are mid, cmf_min is near the bottom; the
    three new exits are all in the lower half. No checklist cell passes 7/7 under the final
    gate; the best is breakout 80 / below_ma 200 on the long history (5/7), below C2 (6/7). Under
    the exploration gate breakout 20 / rsi_below 50 on the snapshot passes 6/7 (fails only DSR,
    0.50). No holdout candidate. Trials 32,139, N_eff ~9,572.
    Record: docs/research/lab/2026-10-22_menu_v3_exploration_gate.md.

### Absolute drawdown limit (2026-10-22, user decision, registered before running)

62. By user decision, both gates add an absolute limit |MDD| <= 15% on the research period
    (grid.EXPLORE_GATES and grid.FINAL_GATES; LOYO re-checks it with each year removed). The
    relative limit 0.75 x |SPY MDD| and the other conditions of 58 are unchanged. This makes both
    gates stricter, not looser, and is the user's stated goal rather than a reaction to which
    families passed. The test of 60 is rerun unchanged on menu v3 under the new gates.
    The fixed-rule blocks ma_stack and up_candle were never part of the family runs (their base
    configuration has no entries); ma_stack was only tested as a common add-on in 54.
63. Results of 62 (menu v3 under the 15% drawdown limit): exploration-gate passing cells fall to
    45 (snapshot) and 62 (long) of 2,772; final-gate passing cells to 0 and 3. bollinger entries
    take the seven lowest rank sums, and every top cell is bollinger k = 3.0, the largest menu
    value (rarely invested). Their checklists pass at most 6/7 under the exploration gate (all
    fail DSR, 0.41-0.58) and at most 2/7 under the final gate. Descriptive: a daily-rebalanced
    SPY/T-bill mix scaled to a 15% drawdown returns 6.6% (snapshot, 41% SPY) and 3.3% (long, 25%
    SPY); the top cells beat it on Calmar but not on snapshot CAGR. No holdout candidate.
    Trials 32,161, N_eff ~9,586. Proposals for period bias (rolling windows, universe/period
    split, CPCV/PBO, block bootstrap) and multi-variable search (Sobol sampling with PBO) are in
    docs/research/lab/2026-10-22_drawdown_limit_and_next_methods.md; none is adopted yet.

### Goal-based criterion, rolling windows, universe/period split (2026-10-22, registered before running)

The user's goal is the highest return with the smallest drawdown, without relying on leverage.

64. Goal criterion (lab/target.py). Each strategy is blended daily with T-bills at the largest
    weight w in [0, 1] (0.01 steps) that keeps its research-period |MDD| <= 15%; w = 1 when the
    drawdown is already within 15% (no leverage). The same scaling of SPY gives the reference
    "SPY + T-bill mix". A cell passes when its scaled CAGR >= the mix's scaled CAGR and it has
    >= 30 completed trades. This replaces the gates of 58 and 62 for exploration; those gates
    stay defined and are still reported.
65. Time robustness: rolling windows of three full calendar years stepping one year inside each
    research period; in each window the strategy and the mix keep their full-period weights,
    and the window counts as won when the strategy's CAGR >= the mix's. window_share = share of
    windows won. Universe/period split: a third sample, the long-history universe on the
    snapshot dates (2017-08-01 - 2023-12-31), so that "same ETFs, different period" and "same
    period, different ETFs" can be told apart.
    Test: menu v3 (77 families) in the three samples. Family order (lexicographic): share of
    passing cells, largest connected passing region, median window_share, median excess CAGR
    over the mix; families ranked across samples by rank sum. The top cell of a family is the
    passing cell ordered by region size, neighbor survival, window_share, excess CAGR. Reported
    for the three families with the lowest rank sum, with DSR. Holdout-candidate rules are
    registered after the multi-variable search (Sobol + PBO), not here; nothing is adopted.

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
