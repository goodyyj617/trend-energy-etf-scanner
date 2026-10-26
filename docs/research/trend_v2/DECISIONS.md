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
66. Multi-variable search (lab/explore.py), by user decision replacing "no exhaustive search
    beyond two parameters" with a fixed-budget sample: 2,048 points of an unscrambled Sobol
    sequence over 8 variables, the same points in the three samples of 65, settings otherwise of
    43. Variables: entry block (the 11 menu-v3 entries) and its value inside its menu range,
    snapped to the parameter step; exit block (the 7 menu-v3 exits) and its value likewise;
    filter (none, market_trend, ma_stack, obv_up, volume_surge, adx_trend, near_high at their
    defaults); initial stop (none for half the points, otherwise 5-20%); max positions 5-20;
    sizing equal or inverse-vol. Every point is evaluated with 64 and 65 and logged as a trial.
    Reported: share of passing points; per-variable summaries (descriptive); PBO by CSCV with 12
    contiguous blocks of monthly returns, ranking by min(CAGR, 0.15 x Calmar) in each half; and
    the robust point = passing in all three samples with window_share >= 0.5 in each, maximizing
    the smallest excess over the mix across samples, with its DSR. A point is proposed as a
    holdout candidate only if it passes in all three samples, has window_share >= 2/3 in each,
    and PBO < 0.5 in every sample; the user decides whether to evaluate it, and the holdout
    verdict would use the criterion of 64.

### Strategy-only test (2026-10-22, user decision, registered before running)

67. By user decision the primary test is now the rule itself, without portfolio construction
    (replacing "portfolio metrics are primary" for exploration). lab/single.py: every ETF is
    traded alone, 100% in while a position is open and in T-bills otherwise (one engine run with
    one slot per symbol and entries not limited by cash; each symbol's trades equal a stand-alone
    run, checked by a test). Costs, price and liquidity filters as in 43; max positions and
    sizing play no role. A symbol is judged if it has >= 3 years of data in the period. Symbol
    excess = 15%-budget CAGR of the strategy (64) - 15%-budget CAGR of that symbol's buy-and-hold.
    The sleeve portfolio, built from the trade records, is the daily equal-weight average of the
    stand-alone accounts; it is evaluated with 64 (against the SPY + T-bill mix) and 65. A cell
    passes when >= 50% of judged symbols have excess >= 0 and the sleeve passes 64. Family order:
    share of passing cells, largest connected passing region, median symbol pass share, median
    sleeve excess; ranked across the three samples of 65 by rank sum, menu v3. The portfolio-level
    run of 65 is kept as a comparison. Every cell is logged as a trial (sleeve curve).
68. The Sobol search of 66 is run in the strategy-only mode: the same 2,048 points, variables 1-6
    (max positions and sizing are dropped as portfolio settings). Pass, PBO (on sleeve monthly
    returns), robust point and holdout-candidate rule as in 66, with the pass of 67 and the
    sleeve's window_share.
69. Results of 65, 67 and 68. Portfolio level (65): passing cells 48 / 888 / 105 (snapshot / long /
    long 2017-2023); best families beat the 15%-budget SPY+T-bill mix by 0.2-1.9 points a year,
    DSR 0.16-0.54. Strategy-only (67): passing cells 0 / 220 / 28; no family beats buy-and-hold
    (same budget) on a majority of snapshot ETFs; trailing_pct families lead on the long history,
    and the same ETFs on 2017-2023 mostly fail, so the split points to the period, not the
    universe. Sobol (68): pass 0% / 10.7% / 0.2%, no point passes all three samples; PBO 0.68 /
    0.25 / 0.53. No holdout candidate. Trials 41,049, N_eff ~11,956.
    Record: docs/research/lab/2026-10-22_strategy_only_and_sobol.md.

### Individual stocks from CRSP, point-in-time universe (2026-10-23, registered before running)

The user wants the test on individual stocks, with a universe wide enough not to be a
large-cap proxy of SPY, and the goal stated as "CAGR above SPY with MDD below SPY" without
leverage. The user's CRSP daily stock file (1991-2024, delisted stocks included) is used
locally or from the user's Google Drive in Colab; the data is never committed.

70. Data and universe (lab/crsp.py). Common stocks (SHRCD 10, 11) on NYSE, AMEX and Nasdaq.
    Prices are total-return adjusted (chain of 1 + RET anchored at the last split-adjusted
    price, like yfinance Adj Close); volume x CFACPR keeps close x volume the day's dollar
    volume. Point-in-time universe: at each year-end the TOP_N = 1,000 stocks by market value
    (|PRC| x SHROUT) are the entry candidates of the following year (Panel.eligible); a
    holding that drops out is kept until its exit rule fires. Delisting: one bar on the next
    trading day at last close x (1 + DLRET), DLRET missing -> -30% for codes 500 and 520-584
    (Shumway 1997), else 0; the engine's forced "delist" exit sells at that bar
    (Panel.delist_exit). A stock's last two bars are never entry candidates. Price >= $5 and
    20-day dollar volume >= $5M filters of 43 still apply; 10 bp per side. Research samples:
    1993-02-01 - 2023-12-31 (SPY's history) and 2017-08-01 - 2023-12-31; 2024 stays unseen.
    Top 3,000 is a later registration if the top 1,000 shows anything.
71. Test (lab/crsp_study.py, strategy-only mode of 67). Menu v3 families and 1,024 Sobol
    points over variables 1-6 of 66, in both samples. A cell passes when its sleeve portfolio
    (equal weight over that day's eligible and held stand-alone accounts, 100% invested, no
    leverage) has CAGR >= SPY CAGR, |MDD| <= |SPY MDD| and >= 30 completed trades. Family
    order: pass share, largest connected passing region, median CAGR excess over SPY, median
    share of stocks beating their own buy-and-hold at the 15% budget (67); ranked across the
    two samples by rank sum. PBO by CSCV (12 blocks) on the Sobol sleeves per sample. The
    15%-budget metrics of 64-65 are reported alongside. A point or cell is proposed as a
    holdout candidate only if it passes in both samples with window_share >= 2/3 in each and
    PBO < 0.5 in both; the user decides on the holdout look (verdict: the goal of this item).
    Every cell is logged as a trial. Results and the record go into a dated note under
    docs/research/lab/ when the run finishes.

### Per-position risk management layer (2026-10-23, components registered before any run)

The user wants strategies built on the risk management of individual positions, not on asset
allocation. Review: docs/research/lab/2026-10-23_quant_models_small_account.md.

72. Engine components, all off by default (existing configs keep their keys and results, checked
    by tests): exit block atr_stop (close < entry - k x ATR14 of the signal day; k default 2,
    range 0.5-5); sizing "risk" (shares = risk_pct% of equity / initial stop distance, risk_pct
    default 1, with an optional cap max_weight on one position; a symbol without an initial
    stop is not bought); heat (no buy that lifts the sum of open initial risks above this
    fraction of equity); breakeven_r (after a gain of X R the stop moves to the entry price,
    exit reason "손익분기 이동"). Pyramiding (adding units as a position gains) is not built yet.
    Time stops and fixed or partial profit targets stay excluded by the CHARTER unless the user
    changes it. Trade-level R-multiple statistics (expectancy, win rate, SQN) are the secondary
    readout of this layer.
    Test (to run after 71): base rules = the families that pass 71 in both samples or, if none,
    breakout x trailing_pct and momentum x trailing_pct (the two-sample leaders of 48 and 61).
    On the CRSP universe of 70, account level, no leverage, max_positions 20: a Sobol sample of
    1,024 points over atr_stop k {0.5-5}, risk_pct {0.25-2}, max_weight {5-25%}, heat {2-20%
    or off}, breakeven_r {off, 0.5-2}, trailing exit value within its menu range, entry value
    within its menu range; verdict as in 71 (CAGR >= SPY, |MDD| <= |SPY MDD|, both samples,
    window_share, PBO), plus after-cost and after-tax CAGR (round trip 0.5% unless the user
    gives the broker's terms; Korean capital-gains tax 22% above 2.5M won a year).
73. CHARTER amendment by user decision (2026-10-23): two risk-management exits are allowed.
    time_stop N (exit when a position has been held N trading days and its close is at or below
    the entry price; profitable positions are not touched; default 20, range 5-120) and
    take_profit_pct X (on the first close at or above entry x (1 + X%), sell 50% at the next open,
    once; the rest follows the other exits; default 20, range 5-100). An unconditional holding
    period and a target that closes the whole position stay prohibited. A position with a partial
    sale is one row in the trade list, its return combining both sales (exit reason suffixed
    "(부분 익절 50% 후)"); the sales are in BacktestResult.partial_exits and lab/single.py rebuilds
    the part-stock, part-cash stand-alone account from them (tested against the one-slot engine).
    Both join the variables of the risk-management test of 72.
74. Long runs on the CRSP panel: the first Colab run stopped after 4 families when a worker ran
    out of memory (block outputs and exit arrays of every family stayed in the cache). The study
    now frees rule-specific arrays when the family changes (SignalCache.drop_rule_data), stores
    the next-exit array as int32, and computes the per-symbol metrics as matrices
    (target.fit_weights; identical weights, per-symbol buy-and-hold now from float64 prices,
    differences < 1e-6). Results of the four finished families stand; the run resumes from them.
    A second stop (8 workers on 51 GB, after 15 more families) came from memory peaks of
    workers coinciding: the notebook now allows about 8 GB per worker, and the runner restarts
    the pool with one worker fewer and resumes when a worker is killed. Finished families are
    unaffected (each is written only when complete).
75. The daily alert of 53 drops the model portfolio and its next-open orders: the user does not
    need them, and no strategy has passed the final gate (the alert strategy was only the v2
    leader's defaults). It now reports the four regimes of 50 with, for each: the current value,
    the on rule, how long the current state has held, a plain-language reading (level band,
    one-month change, percentile since 1990, distance to the 200-day reference), a flag on the
    day a regime flips, and a fixed explanation of what the indicator measures. The research
    verdict of 52 (no regime improved timing; macro adds nothing beyond SPY's 200-day line) is
    repeated in every comment so the alert is read as context, not as a trading signal.
    Bug fixed with it: yfinance stamps the current day's bar at an intraday time (tz-aware,
    differing per ticker), so the wide pivot put it in a separate row that was NaN for SPY and
    most holdings while a few tickers' signals still fired. lab/data.normalize_dates maps every
    stamp to the New York trading date; the alert additionally drops trailing rows with no SPY
    close (daily_alert.trim_incomplete_tail). Research results are unaffected: the snapshot
    files are midnight-stamped, and a yfinance cache file could carry such a stamp only on its
    last row (the download day), which lies after every research end date (holdout start - 1).

76. Results of 71, families part (CRSP point-in-time top 1,000, strategy-only, menu v3, 5,544
    cells): no cell meets the goal in either sample (0 / 2,772 each). Nearly every cell has a
    smaller drawdown than SPY (best CAGR among them 8.8% vs SPY 9.9% on 1993-2023, 10.4% vs 12.7%
    on 2017-2023). At the 15% drawdown budget 19.7% of 1993-2023 cells beat the SPY + T-bill mix
    (best +2.2 points), none on 2017-2023. The rule beats holding the stock on only 15-34% of
    stocks (family medians). trailing_pct exits and long entry windows lead both samples. The
    Sobol part of 71 was not run (Colab TPU quota). Trials 46,593 (Colab log merged), N_eff
    ~13,459. Record: docs/research/lab/2026-10-24_crsp_individual_stocks.md.

### Daily alert redesign (2026-10-24, user request)

### Selection, then trading (2026-10-24, user direction, registered before running)

The user's direction after 76: first screen out stocks that must not be bought, then pick stocks
and entry timing with swappable filters, widen the universe to small caps, keep the existing exits,
and run on the lab computer via Google Drive. Review: docs/research/lab/2026-10-24_selection_then_trade_design.md.

77. Screens (lab/screens.py), all price/volume/size/dividend based (CRSP has no accounting data;
    Compustat can be added later). Exclusion screens (True = do not buy): x_downtrend (12-1 return
    < X%), x_drawdown (close < 52w high x (1 - X%)), x_volatility (60d vol in top X%), x_lottery
    (max daily return over 21d >= X%), x_overextended (36m return in top X%), x_small (cap < $X M),
    x_thin (21d dollar volume < $X M). Selection scores (higher = better, combined by mean
    percentile rank, top K stay buyable): s_momentum (M-1 return), s_high52, s_smooth (share of up
    days over M months), s_quality (M-1 return / M-month vol), s_volume_trend, s_small, s_large,
    s_dividend (M-month dividend yield), s_reversal (-M-month return), s_lowvol. Screens are
    decided on each month's last trading day and hold for the next month (Panel.eligible); the
    engine buys eligible symbols in selection-rank order (Panel.rank). Universe: CRSP top 3,000 by
    year-end market value (as 70 with TOP_N = 3,000), price >= $5, 21-day dollar volume >= $1M.
    Stage A test (lab/screen_study.py): every exclusion screen (and none) x every selection score,
    grid = exclusion value {4 registered values} x K {20, 50, 100, 200}, two entry variants
    (always, above_ma 100), fixed trading layer: trailing_pct 30 + atr_stop 3 (risk unit), sizing
    risk 1% / position cap 10%, max 20 positions, 20 bp per side; samples of 71; account level.
    Verdict per cell: CAGR >= SPY and |MDD| <= |SPY MDD| and >= 30 trades. Family order: pass
    share, largest connected passing region, median CAGR excess, median window share; rank sum
    across samples. Reported alongside: 15%-budget metrics, buyable count. Stage B (entries,
    exits, risk layer by Sobol around the best screens) is registered after Stage A. Every cell
    is a trial. Runs on the lab computer through scripts/crsp_remote.py (panel copied from Drive
    to a temporary folder, results written back, folder deleted).
78. Results of 77, Stage A (CRSP top 3,000, 4,640 cells, lab computer): 94 / 2,320 cells pass on
    1993-2023 and 106 / 2,320 on 2017-2023, but only 2 cells pass both samples and neither sits in
    a region that passes in both. The selection score decides; exclusion screens move pass shares
    only within 2-6%. Winners flip between samples: dividend yield (18.1% vs 0.4%), small caps
    (6.5% vs 0.4%), 36-month reversal (6.0% vs 0%) on 1993-2023; large caps (4.3% vs 40.1%) on
    2017-2023. Momentum selection passes no cell in either sample under the fixed trading layer.
    No holdout candidate. Trials 51,233 (lab-computer log merged), N_eff ~16,515.
    Record: docs/research/lab/2026-10-25_screen_comparison_top3000.md.
79. Representative-filter comparison of the selection scores (screens v2, registered before any
    run). Every selection score of lab/screens.py (42, tiers 1-3, each at its registered default
    lookback; exact values are left to later robustness tests) is tested alone: hygiene defaults
    always on (history 12 m, price >= $5, dollar volume >= $1M, Amihud bottom 20% out; no size
    floor), exclusion {none, Minervini trend template 25} x top K {20, 50, 100} = a 2 x 3 grid per
    score. Two entries (always, above_ma 100) x two trading layers: hold (trailing_pct 30 +
    atr_stop 3) and replace (new exit block left_list: sold the day after the stock leaves the
    monthly list, + atr_stop 3). Sizing risk 1% / cap 10%, max 20 positions, 20 bp per side,
    samples of 71, CRSP top 3,000, account level. Verdict and family order as 77 (pass share,
    largest region, median CAGR excess, median window share); scores ranked by rank sum across
    samples per (entry, layer), with tier averages. 2,016 cells, every cell a trial. Runner:
    lab/score_study.py through scripts/crsp_remote.py (study "scores"; a Drive panel without SIC
    codes is rebuilt). Entry-timing filters are redesigned after this result.
    Design: docs/research/lab/2026-10-25_screens_v2_design.md.
80. Stage B: blends of the consistent top selection scores x a Sobol sample of the trading layer
    (user approved 2026-10-26, registered before any run). Selection: every 2- and 3-combination
    of {dollar volume, trend efficiency, large cap, smoothness, accumulation/distribution, OBV
    slope} (15 + 20), each at its default lookback, combined by the mean percentile rank, plus
    12-1 momentum alone (36). Trading layer: the same 32 Sobol points (lab.explore.sobol, 7
    variables) for every combination and both samples: exclusion {hygiene, + tt_fail 25}, top K
    {20, 50, 100}, layer {hold (trailing_pct 15-40), replace (left_list, no trailing)}, atr_stop
    {off, 2, 3, 4}, risk_pct 0.5-2.0, max_positions {10, 15, 20, 30}. Fixed: entry always, cap
    10%, 20 bp per side, hygiene defaults (no size floor), no leverage. Risk sizing needs an
    initial stop; a point without an ATR stop would buy nothing, so it is sized with equal slots
    (1 / max positions) and its risk % is unused. 36 x 32 x 2 = 2,304 runs, every run a trial.
    Pass as 79 (CAGR >= SPY, |MDD| <= |SPY MDD|, >= 30 trades); dd_excess and window share kept.
    Judging: per combination, pass share of the 32 points then median CAGR excess per sample;
    rank = sum of the two sample ranks; a combination passing in one sample only is never
    promoted. For the top combinations, where the both-sample passes sit in layer space (no
    single-point regions). PBO (CSCV) over all runs is information only; DSR only for 1-3
    finalists if such a region exists. No holdout. Runner: lab/combo_study.py through
    scripts/crsp_remote.py (study "combos"); it writes the memo figures itself.
    Design: docs/research/lab/2026-10-26_stage_b_design.md.
    Result (2026-10-26, run 19:51-22:18 KST): 197 / 203 of 1,152 runs pass (1993-2023 /
    2017-2023); 34 of 36 combinations have a both-sample pass, 73 both-sample (combination,
    point) passes, 67 of them in the hold layer. Top by rank sum: efficiency + large (pass share
    53% / 56%, median excess +0.4 / +0.5 pp), dollar volume + efficiency + large (50% / 50%).
    Replace layer passes 1-7% at every ATR setting; momentum alone never passes (the replace
    layer without an ATR stop does not revive it). Rank correlation of combinations across
    samples 0.30; PBO 30% / 60%; DSR of three representative points 0.85-0.97 (1993-2023) and
    0.57-0.74 (2017-2023) at N 635-17,000. Read as index-like (large, liquid names), not a
    selection edge. No holdout candidate. Trials +2,304 (lab-computer log on Drive, not merged).
    Record: docs/research/lab/2026-10-26_stage_b_results.md.
81. Stage C: entry strategies x exit strategies on a daily universe (user requests 2026-10-26,
    registered before any run). Universe: each trading day the top 3,000 common stocks by the
    21-session average market value up to that day (lab.crsp --daily; the engine reads the mask
    on the signal day and buys at the next open), delisted names included with the forced
    delisting exit; panel = every name ever in that set. Hygiene screens evaluated daily (12-month
    history >= 90%, price >= $5, 21-day dollar volume >= $1M, Amihud top 20% out); no size floor,
    no trend template, no month-end list. Entries (E1-E9 fire only on the first day the condition
    is true; canonical value in brackets): E0 always (control); E1 close > prior N-day high, N
    20/[55]/100; E2 SMA(fast) crosses above SMA200, fast 20/[50]/100; E3 close > prior 252-day
    high and volume >= m x prior 50-day average, m 1.2/[1.5]/2.0; E4 yesterday's Bollinger(20, 2)
    bandwidth in the lowest p% of 126 days and close > upper band, p 10/[20]/30; E5 close > SMA200
    and RSI(2) < r, r 5/[10]/15; E6 MACD crosses above signal with MACD > 0, (8,17,9)/[(12,26,9)]/
    (19,39,9); E7 +DI(14) crosses above -DI(14) with ADX(14) >= a, a 15/[20]/25; E8 L-month
    return > T-bill return over the same span, L 3/6/[12]; E9 open >= prior high x (1 + g),
    volume >= 3 x prior 50-day average, close > open, g 2/[4]/6%. Exits (full sale at the next
    open; delisting exit common; no extra common stop): X1 close < peak close x (1 - x), x
    15/[25]/35%; X2 chandelier close < peak close - k x ATR(22), k 2/[3]/4 (new path exit; the
    existing atr_trail uses ATR(14)); X3 close < prior N-day low, N 10/[20]/50; X4 close < SMA(N),
    N 20/[50]/100; X5 N-day return < 0, N 21/[63]/126; X6 Parabolic SAR (step a, max 0.2) flips
    above the price, a 0.01/[0.02]/0.03 (fires on the flip day; a position bought in a downtrend
    waits for the next flip); X7 close > SMA5 or RSI(2) > r, r 60/[70]/80; X8 SPY < SPY SMA(N),
    N 100/150/[200]; X9 stop at entry - k x ATR(14) of the signal day, breakeven after +2R, time
    stop after 60 sessions if close <= entry, k 1.5/[2]/3. 3 x 3 cells per pair (E0: 3), 756
    cells per sample, samples as 71. C1 every signal (lab/single.py, trade lists kept); C2 a
    20-slot account, 1/20 of equity at purchase, no rebalancing, signals while full dropped,
    priority (a) 20-day dollar volume on every cell, and on each pair's canonical cell also (b)
    mean percentile of trend efficiency (6 months) and market value among that day's eligible
    names and (c) random (seed 81). Cost 20 bp per side, T-bill cash, no leverage. Gate as 79
    (C2: CAGR > SPY, |MDD| < |SPY MDD|, >= 30 trades). Per-signal table (E1-E9): every signal
    traded on its own with the same exits, bought at the next open, overlapping signals of one
    symbol allowed, cost included (engine run_backtest(outcomes=True), reusing first_exit); per
    cell: signals, expectancy, win rate, payoff, average holding days, share of big winners
    (return >= +50%). Missed vs taken: C2(a) trades matched to that table by (symbol, signal day);
    counts, per-signal expectancy and big-winner share of taken vs missed, signal-day
    market-value rank quartiles. (Matching to C1 trades was dropped after the dry run: C1 holds
    one position per symbol and ignores signals while held, so most account buys had no C1 twin.) Judging: entry rank = median
    over every exit x cell, exit rank likewise, per sample; pass share and neighbours within 3 x
    3; Spearman of ranks across samples; entries below E0 flagged; index check (beta, down
    capture, excess by period) on the top pairs; PBO information only; DSR only for 1-3
    finalists; holdout unused; nothing changed after results. Runs 1,512 C1 + 1,512 C2 + 360
    priority (all 90 pairs incl. E0), each a trial. Runner lab/stage_c_study.py through
    scripts/crsp_remote.py --study stagec (panel lab_crsp/panel_daily3000, results
    lab_crsp/stagec_daily3000). Design: docs/research/lab/2026-10-26_stage_c_design.md.

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
