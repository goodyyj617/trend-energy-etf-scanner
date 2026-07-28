# Phase B skew-aware robustness and crash-avoidance analysis

## Decision

The bounded canonical portfolio data are internally valid and statistically informative, but they cover one in-sample history, only eight full calendar years, and a small number of nested historical SPY crash episodes. The results support provisional *diagnostic ranges*, not mandatory production portfolio-risk gates. No strategy, portfolio allocation rule, qualification gate, or ranking was changed.

## 1. Input validation

| Check | Result |
| --- | ---: |
| Unique daily strategy series | 540 |
| Qualified strategies / published curves | 42 / 42 |
| Economic observations per strategy | 2,332 |
| Economic date range | 2017-04-17 to 2026-07-27 |
| Explicit t0 rows excluded from estimation | 1 |
| Duplicate date x strategy rows | 0 |
| Non-finite returns | 0 |
| Returns <= -100% | 0 |
| Maximum ending-equity reconstruction error | 1.37e-10 |
| Independently reproduced production primary | `score_bo_l40_rm002_erp010__signal_3d_confirm__ma50` |
| Portfolio statistics in production rank | no |
| Material validation failures | 0 |

The all-strategy input is the documented `backtest_portfolio_daily_returns.csv.gz`, not an assumed Parquet file. The schema-v2 initialization row remains available for USD 1,000 normalization but is excluded from every estimate. Strategy and SPY economic dates are joined by exact equality; no forward fill is used. The final canonical audit script also passes.

## 2. Populations and baseline results

The primary population is the 42 current Qualified strategies. The multiple-testing population is all 540 tested strategies; no poor strategy was removed. All 540 receive baseline return, distribution, drawdown, tail, PSR/DSR, behavioral-deduplication, and crash-return diagnostics. Computationally intensive bootstrap and detailed exposure/cost diagnostics use the 42 published Qualified curves.

Qualified versus non-Qualified medians:

| Metric | Qualified | Non-Qualified |
| --- | ---: | ---: |
| CAGR | 4.75% | 4.23% |
| Maximum drawdown | -29.83% | -29.43% |
| CDaR 95 | -26.50% | -26.34% |
| Daily skewness | 0.228 | -0.435 |
| Annualized Sharpe | 0.437 | 0.436 |

The robustness CSV reports arithmetic and geometric growth, volatility, unbiased skewness/excess kurtosis, downside/upside semideviation, Omega, gain/loss, positive-day/month/year rates, Sharpe/Sortino/Calmar/Martin, three largest drawdowns, CDaR 90/95/99, recovery statistics, drawdown-frequency thresholds, rolling losses, ES, and the five largest distinct drawdown paths. Exposure and implementation fields are populated only for published Qualified curves.

The positive-calendar-year ratio in this Phase B descriptive table includes every observed calendar-year bucket, including partial 2017 and 2026. It is not the production Time Gate ratio, which continues to use only eligible full entry years.

## 3. Time concentration and removal diagnostics

Contributions use additive log returns because every canonical equity path remains positive. Ratios are contribution divided by total observed log growth; negative or near-zero total growth is reported without converting the ratio into a gate. The long-form concentration table includes baseline, best-day removal, five/ten best-day zeroing, best-month/year zeroing, leave-one-full-year-out, and every distinct drawdown-episode removal, each with ending equity, CAGR, MDD, CDaR 95, Sharpe, and Calmar.

For the primary candidate, the best day contributes 19.72% of total log growth, the best five days 63.31%, the best ten days 96.18%, the best month 46.76%, and the best year 66.23%. These are fragility diagnostics, not pass conditions; positive skew naturally concentrates some growth in a minority of positive periods.

Raw trade-level winner concentration and ETF-level profit attribution are unavailable and were not fabricated.

## 4. Stationary block bootstrap

The stationary bootstrap uses economic daily returns, geometric block lengths with restart probability `1 / mean_block_length`, fixed root seed 20260728, observed path length 2332, and percentile intervals. Behavioral equivalence is established before bootstrap computation. The 42 Qualified labels reduce to 27 unique Qualified return paths, and each path is bootstrapped exactly once for the comparison. Every path uses the same block-20 index matrix (`a20988da73bd1a04c843855e91b3a958683e1f2cc5ec7a90bdb8ef62156e595f`), and strategy and SPY use those same indices within every relative-performance path, preserving contemporaneous dependence. Missing observations are not filled because validation requires a complete common matrix. Partial calendar years remain in the daily bootstrap as observed economic returns; annual gate eligibility is not reused here. CAGR uses the observed calendar span. Block-boundary splicing and regime non-stationarity remain limitations, and reported probabilities are not guaranteed future probabilities.

The cross-candidate comparison uses 5,000 paths at mean block length 20 for every Qualified behavioral path and is the only bootstrap input to Pareto analysis. The production primary retains a separate 10,000-path diagnostic: its first 5,000 rows are the common comparison matrix, and the additional 5,000 rows use fixed seed 20260729. The detailed diagnostic is not a Pareto input. Primary sensitivity:

| Mean block | Paths | P(CAGR > 0) | 5th pct CAGR | 95th pct MDD magnitude |
| ---: | ---: | ---: | ---: | ---: |
| 5 | 5,000 | 87.90% | -1.72% | 40.75% |
| 20 | 5,000 | 88.74% | -1.63% | 39.23% |
| 60 | 5,000 | 87.16% | -1.91% | 40.56% |

Primary detailed block-20 results: P(ending equity > USD 1,000) 88.20%, P(CAGR > 0) 88.20%, jointly resampled P(CAGR > SPY) 3.11%, 5th-percentile ending equity USD 855.42, 5th-percentile CAGR -1.67%, 95th-percentile MDD magnitude 39.36%, and 5th-percentile Calmar -0.048. Its common-comparison values are P(CAGR > 0) 88.74%, P(CAGR > SPY) 3.20%, 5th-percentile CAGR -1.63%, and 95th-percentile MDD magnitude 39.23%.

## 5. PSR, DSR, and selection bias

Daily Sharpe is the sample mean divided by sample standard deviation; annualized values multiply by sqrt(252). PSR uses the finite-sample skewness/kurtosis-adjusted denominator `sqrt(1 - skew*SR + ((kurtosis-1)/4)*SR^2)`, with unbiased sample skewness and excess kurtosis converted to ordinary kurtosis. Benchmarks are annual Sharpe 0, observed SPY Sharpe 0.849, and the predeclared positive annual Sharpe 0.50, each divided by sqrt(252) before evaluation. PSR is a probability that the population Sharpe exceeds the benchmark under the formula's assumptions; it is not labeled a generic p-value.

DSR uses the same PSR adjustment against the expected maximum Sharpe, cross-sectional observed-Sharpe standard deviation, Euler-Mascheroni expected-maximum approximation, and two trial treatments: raw N=540 and exact/numerically equivalent daily paths hashed after deterministic 1e-12 quantization. The exact-deduplicated count is 360; raw and deduplicated expected maximum annual Sharpe are 0.690 and 0.662. Correlation clusters are not called independent trials.

Primary: annualized Sharpe 0.439, PSR vs zero 91.01%, PSR vs SPY 10.54%, PSR vs 0.50 42.65%, DSR N=540 22.22%, and exact-deduplicated DSR 24.81%. PSR/DSR reduce neither data reuse nor model-selection history to out-of-sample evidence.

## 6. Multiple-testing feasibility

The common 540-column daily matrix is sufficient in principle for a stationary-bootstrap White Reality Check or Hansen SPA loss-differential design against cash (`strategy_return - 0`) and SPY (`strategy_return - SPY_return`) with shared block indices. It is not implemented here because a defensible SPA requires an explicitly selected studentization and weak-model trimming rule, while a 540-strategy Reality Check would answer a different familywise-null question than the requested per-candidate bootstrap and DSR. Adding a superficially precise p-value without those predeclared choices would be less reliable than the supplied feasibility design.

PBO/CSCV is not authoritative here: only eight full calendar years are available, strategy returns share regimes, and symmetric temporal partitions would be few and highly dependent. A later frozen walk-forward protocol is recommended.

## 7. Historical SPY crash episodes

Episodes are independently constructed at 10%, 15%, and 20%. Onset is the first threshold crossing from above, each threshold is non-overlapping until recovery of its prior peak, severe episodes may nest within lower-threshold episodes, and unrecovered episodes remain open. t0 is excluded.

| Episode | Threshold | Prior peak | Onset | Trough | Recovery | Nested within |
| --- | ---: | --- | --- | --- | --- | --- |
| SPY10_01 | 10% | 2018-01-26 | 2018-02-08 | 2018-02-08 | 2018-08-06 | none |
| SPY10_02 | 10% | 2018-09-20 | 2018-12-14 | 2018-12-24 | 2019-04-12 | none |
| SPY10_03 | 10% | 2020-02-19 | 2020-02-27 | 2020-03-23 | 2020-08-10 | none |
| SPY10_04 | 10% | 2022-01-03 | 2022-02-22 | 2022-10-12 | 2023-12-13 | none |
| SPY10_05 | 10% | 2025-02-19 | 2025-03-13 | 2025-04-08 | 2025-06-26 | none |
| SPY15_01 | 15% | 2018-09-20 | 2018-12-20 | 2018-12-24 | 2019-04-12 | SPY10_02 |
| SPY15_02 | 15% | 2020-02-19 | 2020-03-09 | 2020-03-23 | 2020-08-10 | SPY10_03 |
| SPY15_03 | 15% | 2022-01-03 | 2022-05-09 | 2022-10-12 | 2023-12-13 | SPY10_04 |
| SPY15_04 | 15% | 2025-02-19 | 2025-04-04 | 2025-04-08 | 2025-06-26 | SPY10_05 |
| SPY20_01 | 20% | 2020-02-19 | 2020-03-12 | 2020-03-23 | 2020-08-10 | SPY10_03;SPY15_02 |
| SPY20_02 | 20% | 2022-01-03 | 2022-06-13 | 2022-10-12 | 2023-12-13 | SPY10_04;SPY15_03 |

There are 5 10%, 4 15%, and 2 20% episodes. This small, nested sample cannot establish general crash protection.

Crash loss capture is `max(-strategy onset-to-trough return, 0) / max(-SPY onset-to-trough return, epsilon)`; lower is better. Relative protection is the signed strategy return minus SPY return. The episode CSV also reports peak-to-trough/onset-to-recovery windows, local MDD/CDaR/rolling losses, recovery, and relative outcomes for all 540 strategies. Exposure decay, positions, and cost are limited to the 42 published curves.

Exposure below 100% at onset is evidence of avoiding some risk before the threshold crossing. The change from onset to days +5/+10/+20 describes reaction after the crash begins. Peak-to-trough versus onset-to-trough loss separates losses already suffered before the threshold from losses after it. A low onset exposure can therefore coexist with a slow post-onset reduction, and neither is silently interpreted as a stop-system pass.

## 8. Current primary candidate

`score_bo_l40_rm002_erp010__signal_3d_confirm__ma50` remains production rank 1; Phase B does not change that rank.

- Ending equity USD 1,535.19; CAGR 4.73%.
- MDD -26.44%; CDaR 95 -23.69%; ES95 -1.90%; longest time under water 1603 days.
- Daily skewness 0.394; excess kurtosis 13.841.
- Behavioral group `B280` contains 2 numerically equivalent parameter paths.
- Cost sensitivity: observed CAGR 4.73%, 1.5x-cost CAGR 2.91%, 2x-cost CAGR 1.11%; 2x ending equity USD 1,107.94, MDD -32.86%, and Calmar 0.034. This is a deterministic fixed-membership/turnover-path reconstruction using the exact daily cost load, not a new portfolio simulation.
- Strict Pareto frontier: no; tolerance frontier: no.

Primary crash table:

| Episode | Onset | Trough | Strategy onset-to-trough | SPY onset-to-trough | Loss capture | Exposure onset | Exposure +10d |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| SPY10_01 | 2018-02-08 | 2018-02-08 | -1.48% | -3.75% | 0.40 | 66.31% | 100.00% |
| SPY10_02 | 2018-12-14 | 2018-12-24 | 0.29% | -11.18% | 0.00 | 100.00% | 100.00% |
| SPY10_03 | 2020-02-27 | 2020-03-23 | -3.16% | -28.01% | 0.11 | 97.83% | 55.13% |
| SPY10_04 | 2022-02-22 | 2022-10-12 | -11.46% | -16.94% | 0.68 | 100.00% | 100.00% |
| SPY10_05 | 2025-03-13 | 2025-04-08 | -2.70% | -10.90% | 0.25 | 98.61% | 100.00% |
| SPY15_01 | 2018-12-20 | 2018-12-24 | -0.05% | -6.19% | 0.01 | 100.00% | 100.00% |
| SPY15_02 | 2020-03-09 | 2020-03-23 | -4.38% | -24.61% | 0.18 | 93.01% | 100.00% |
| SPY15_03 | 2022-05-09 | 2022-10-12 | -15.11% | -12.59% | 1.20 | 100.00% | 100.00% |
| SPY15_04 | 2025-04-04 | 2025-04-08 | -2.41% | -7.49% | 0.32 | 86.87% | 100.00% |
| SPY20_01 | 2020-03-12 | 2020-03-23 | -1.99% | -18.26% | 0.11 | 55.13% | 100.00% |
| SPY20_02 | 2022-06-13 | 2022-10-12 | -12.31% | -7.75% | 1.59 | 100.00% | 100.00% |

It remains attractive where its deterministic qualification evidence, positive historical growth, skew-aware Sharpe probability, and relative crash behavior agree. It may be unsafe where bootstrap lower tails, prolonged underwater periods, winner-period concentration, turnover/cost sensitivity, or slow exposure decay remain material. Its first-place qualification rank is supported by event-level gate evidence, but portfolio-risk evidence is mixed and was never part of that rank.

## 9. Turnover and cost robustness

The analysis preserves the high observed turnover/cost values. For each Qualified curve it reports annual turnover, total and annual cost, cost per full portfolio year, cost relative to initial capital/ending profit/de-costed same-path growth, turnover-bearing rebalance days, and average turnover per rebalance. The 1x/1.5x/2x scenarios reconstruct daily gross return as observed net return plus exact daily cost divided by prior equity, then scale that same daily cost load. They do not claim that higher costs would leave future membership or equal-weight solver allocations unchanged.

No cost threshold is introduced and no production cost assumption changes.

## 10. Candidate comparison and Pareto analysis

Separate leaders are determined at the behavioral-path level. When a winning path has duplicate parameter labels, the table lists the deterministic representative and all Qualified member labels:

| Behavior group | Representative | Qualified member labels | Leader objective(s) |
| --- | --- | --- | --- |
| `B007` | `score_bo_l10_rm002_erp005__first_signal__low10` | score_bo_l10_rm002_erp005__first_signal__low10<br>score_bo_l10_rp000_erp005__first_signal__low10 | largest_behavioral_equivalence_group |
| `B027` | `score_bo_l10_rm002_erp010__first_signal__low10` | score_bo_l10_rm002_erp010__first_signal__low10<br>score_bo_l10_rp000_erp010__first_signal__low10 | best_worst_crash_drawdown, largest_behavioral_equivalence_group |
| `B047` | `score_bo_l10_rm002_erp015__first_signal__low10` | score_bo_l10_rm002_erp015__first_signal__low10<br>score_bo_l10_rp000_erp015__first_signal__low10 | largest_behavioral_equivalence_group, lowest_es95 |
| `B127` | `score_bo_l20_rm002_erp005__first_signal__low10` | score_bo_l20_rm002_erp005__first_signal__low10<br>score_bo_l20_rp000_erp005__first_signal__low10 | largest_behavioral_equivalence_group |
| `B147` | `score_bo_l20_rm002_erp010__first_signal__low10` | score_bo_l20_rm002_erp010__first_signal__low10<br>score_bo_l20_rp000_erp010__first_signal__low10 | largest_behavioral_equivalence_group, lowest_bootstrap_95pct_mdd, lowest_cdar95 |
| `B167` | `score_bo_l20_rm002_erp015__first_signal__low10` | score_bo_l20_rm002_erp015__first_signal__low10<br>score_bo_l20_rp000_erp015__first_signal__low10 | largest_behavioral_equivalence_group, lowest_mdd |
| `B207` | `score_bo_l20_rp002_erp010__first_signal__low10` | score_bo_l20_rp002_erp010__first_signal__low10 | shortest_longest_time_under_water |
| `B258` | `score_bo_l40_rm002_erp005__signal_3d_confirm__low20` | score_bo_l40_rm002_erp005__signal_3d_confirm__low20<br>score_bo_l40_rp000_erp005__signal_3d_confirm__low20 | largest_behavioral_equivalence_group |
| `B259` | `score_bo_l40_rm002_erp005__signal_3d_confirm__low20_minus_0_5atr` | score_bo_l40_rm002_erp005__signal_3d_confirm__low20_minus_0_5atr<br>score_bo_l40_rp000_erp005__signal_3d_confirm__low20_minus_0_5atr | largest_behavioral_equivalence_group |
| `B260` | `score_bo_l40_rm002_erp005__signal_3d_confirm__ma50` | score_bo_l40_rm002_erp005__signal_3d_confirm__ma50<br>score_bo_l40_rp000_erp005__signal_3d_confirm__ma50 | largest_behavioral_equivalence_group |
| `B278` | `score_bo_l40_rm002_erp010__signal_3d_confirm__low20` | score_bo_l40_rm002_erp010__signal_3d_confirm__low20<br>score_bo_l40_rp000_erp010__signal_3d_confirm__low20 | largest_behavioral_equivalence_group |
| `B279` | `score_bo_l40_rm002_erp010__signal_3d_confirm__low20_minus_0_5atr` | score_bo_l40_rm002_erp010__signal_3d_confirm__low20_minus_0_5atr<br>score_bo_l40_rp000_erp010__signal_3d_confirm__low20_minus_0_5atr | largest_behavioral_equivalence_group |
| `B280` | `score_bo_l40_rm002_erp010__signal_3d_confirm__ma50` | score_bo_l40_rm002_erp010__signal_3d_confirm__ma50<br>score_bo_l40_rp000_erp010__signal_3d_confirm__ma50 | largest_behavioral_equivalence_group |
| `B298` | `score_bo_l40_rm002_erp015__signal_3d_confirm__low20` | score_bo_l40_rm002_erp015__signal_3d_confirm__low20<br>score_bo_l40_rp000_erp015__signal_3d_confirm__low20 | largest_behavioral_equivalence_group |
| `B299` | `score_bo_l40_rm002_erp015__signal_3d_confirm__low20_minus_0_5atr` | score_bo_l40_rm002_erp015__signal_3d_confirm__low20_minus_0_5atr<br>score_bo_l40_rp000_erp015__signal_3d_confirm__low20_minus_0_5atr | largest_behavioral_equivalence_group, lowest_cost_sensitivity |
| `B300` | `score_bo_l40_rm002_erp015__signal_3d_confirm__ma50` | score_bo_l40_rm002_erp015__signal_3d_confirm__ma50<br>score_bo_l40_rp000_erp015__signal_3d_confirm__ma50 | largest_behavioral_equivalence_group, lowest_worst_crash_loss_capture |
| `B318` | `score_bo_l40_rp002_erp005__signal_3d_confirm__low20` | score_bo_l40_rp002_erp005__signal_3d_confirm__low20 | fastest_mean_exposure_reduction |
| `B338` | `score_bo_l40_rp002_erp010__signal_3d_confirm__low20` | score_bo_l40_rp002_erp010__signal_3d_confirm__low20 | fastest_mean_exposure_reduction |
| `B358` | `score_bo_l40_rp002_erp015__signal_3d_confirm__low20` | score_bo_l40_rp002_erp015__signal_3d_confirm__low20 | fastest_mean_exposure_reduction |
| `B359` | `score_bo_l40_rp002_erp015__signal_3d_confirm__low20_minus_0_5atr` | score_bo_l40_rp002_erp015__signal_3d_confirm__low20_minus_0_5atr | best_worst_full_year_loo, highest_bootstrap_5pct_cagr, highest_bootstrap_prob_cagr_positive, highest_cagr, highest_calmar, highest_dsr_raw, highest_psr_vs_zero, lowest_best_day_concentration |

Strict dominance requires no worse performance in all ten declared dimensions and strict improvement in at least one. Parameter labels within one group cannot dominate each other. The strict frontier contains 17 behavioral paths:

`B027` (`score_bo_l10_rm002_erp010__first_signal__low10`, 2 labels), `B047` (`score_bo_l10_rm002_erp015__first_signal__low10`, 2 labels), `B147` (`score_bo_l20_rm002_erp010__first_signal__low10`, 2 labels), `B167` (`score_bo_l20_rm002_erp015__first_signal__low10`, 2 labels), `B207` (`score_bo_l20_rp002_erp010__first_signal__low10`, 1 label), `B298` (`score_bo_l40_rm002_erp015__signal_3d_confirm__low20`, 2 labels), `B299` (`score_bo_l40_rm002_erp015__signal_3d_confirm__low20_minus_0_5atr`, 2 labels), `B300` (`score_bo_l40_rm002_erp015__signal_3d_confirm__ma50`, 2 labels), `B318` (`score_bo_l40_rp002_erp005__signal_3d_confirm__low20`, 1 label), `B319` (`score_bo_l40_rp002_erp005__signal_3d_confirm__low20_minus_0_5atr`, 1 label), `B320` (`score_bo_l40_rp002_erp005__signal_3d_confirm__ma50`, 1 label), `B338` (`score_bo_l40_rp002_erp010__signal_3d_confirm__low20`, 1 label), `B339` (`score_bo_l40_rp002_erp010__signal_3d_confirm__low20_minus_0_5atr`, 1 label), `B340` (`score_bo_l40_rp002_erp010__signal_3d_confirm__ma50`, 1 label), `B358` (`score_bo_l40_rp002_erp015__signal_3d_confirm__low20`, 1 label), `B359` (`score_bo_l40_rp002_erp015__signal_3d_confirm__low20_minus_0_5atr`, 1 label), `B360` (`score_bo_l40_rp002_erp015__signal_3d_confirm__ma50`, 1 label)

The tolerance frontier uses 1 bp for CAGR/bootstrap CAGR/cost sensitivity, 0.001 for DSR and portfolio/crash-return risk rates, 0.01 for crash capture, and 5 days for time under water. It contains 16 behavioral paths:

`B027` (`score_bo_l10_rm002_erp010__first_signal__low10`, 2 labels), `B047` (`score_bo_l10_rm002_erp015__first_signal__low10`, 2 labels), `B147` (`score_bo_l20_rm002_erp010__first_signal__low10`, 2 labels), `B167` (`score_bo_l20_rm002_erp015__first_signal__low10`, 2 labels), `B207` (`score_bo_l20_rp002_erp010__first_signal__low10`, 1 label), `B298` (`score_bo_l40_rm002_erp015__signal_3d_confirm__low20`, 2 labels), `B299` (`score_bo_l40_rm002_erp015__signal_3d_confirm__low20_minus_0_5atr`, 2 labels), `B300` (`score_bo_l40_rm002_erp015__signal_3d_confirm__ma50`, 2 labels), `B318` (`score_bo_l40_rp002_erp005__signal_3d_confirm__low20`, 1 label), `B319` (`score_bo_l40_rp002_erp005__signal_3d_confirm__low20_minus_0_5atr`, 1 label), `B320` (`score_bo_l40_rp002_erp005__signal_3d_confirm__ma50`, 1 label), `B338` (`score_bo_l40_rp002_erp010__signal_3d_confirm__low20`, 1 label), `B339` (`score_bo_l40_rp002_erp010__signal_3d_confirm__low20_minus_0_5atr`, 1 label), `B358` (`score_bo_l40_rp002_erp015__signal_3d_confirm__low20`, 1 label), `B359` (`score_bo_l40_rp002_erp015__signal_3d_confirm__low20_minus_0_5atr`, 1 label), `B360` (`score_bo_l40_rp002_erp015__signal_3d_confirm__ma50`, 1 label)

The Pareto CSV retains every Qualified strategy label for compatibility, but `pareto_unit` is `behavioral_path`. Group flags, dominator counts, dimension-removal diagnostics, leader objectives, and comparison-bootstrap fields are mapped exactly to all member labels. Duplicate labels are parameter descriptions, not independent economic candidates. No weighted or lexicographic robustness score is constructed.

The crash dimensions use the aggregate 10% SPY-episode population: mean relative protection is maximized and worst loss capture is minimized. This avoids counting nested 15% and 20% observations again in the same Pareto dimension. Block-length sensitivity is nevertheless reported for the union of strict and tolerance frontier strategies plus the production primary.

### Provisional diagnostic ranges, not gates

| Qualified-strategy diagnostic | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| CAGR | 2.10% | 4.75% | 9.94% |
| MDD magnitude | 18.86% | 29.83% | 40.97% |
| CDaR 95 magnitude | 15.43% | 26.50% | 35.18% |
| Bootstrap 5th-percentile CAGR | -3.36% | -1.63% | 0.64% |
| Raw-N DSR | 10.31% | 22.03% | 43.87% |
| Worst 10% crash loss capture | 0.505 | 0.812 | 1.738 |
| 2x-cost CAGR drop | 3.26% | 3.63% | 6.36% |

These observed ranges can frame later preregistration, but none is proposed as a mandatory cutoff from this in-sample run.

## 11. Descriptive interpretation categories

| Strategy | Evidence-based categories |
| --- | --- |
| `score_bo_l40_rm002_erp010__signal_3d_confirm__ma50` | slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rp000_erp010__signal_3d_confirm__ma50` | slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rm002_erp005__signal_3d_confirm__ma50` | slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rp000_erp005__signal_3d_confirm__ma50` | slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rm002_erp015__signal_3d_confirm__ma50` | statistically credible positive-skew trend follower, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rp000_erp015__signal_3d_confirm__ma50` | statistically credible positive-skew trend follower, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rp002_erp010__signal_3d_confirm__ma50` | credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, insufficient crash observations |
| `score_bo_l40_rm002_erp015__signal_3d_confirm__low20` | statistically credible positive-skew trend follower, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rp000_erp015__signal_3d_confirm__low20` | statistically credible positive-skew trend follower, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rp002_erp015__signal_3d_confirm__ma50` | credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, insufficient crash observations |
| `score_bo_l10_rm002_erp010__first_signal__low10` | slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l10_rp000_erp010__first_signal__low10` | slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l10_rm002_erp015__first_signal__low10` | slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l10_rp000_erp015__first_signal__low10` | slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l10_rm002_erp005__first_signal__low10` | slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l10_rp000_erp005__first_signal__low10` | slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l20_rp002_erp010__first_signal__low10` | slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, insufficient crash observations |
| `score_bo_l20_rp000_erp010__first_signal__low10` | slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l20_rp002_erp005__first_signal__low10` | credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, insufficient crash observations |
| `score_bo_l20_rm002_erp010__first_signal__low10` | slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l20_rp000_erp005__first_signal__low10` | slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l20_rp000_erp015__first_signal__low10` | slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l20_rm002_erp005__first_signal__low10` | slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l20_rm002_erp015__first_signal__low10` | slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rp002_erp005__signal_3d_confirm__ma50` | credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, insufficient crash observations |
| `score_bo_l40_rm002_erp010__signal_3d_confirm__low20` | credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rp000_erp010__signal_3d_confirm__low20` | credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rm002_erp015__signal_3d_confirm__low20_minus_0_5atr` | statistically credible positive-skew trend follower, credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rp000_erp015__signal_3d_confirm__low20_minus_0_5atr` | statistically credible positive-skew trend follower, credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rm002_erp010__signal_3d_confirm__low20_minus_0_5atr` | credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rp000_erp010__signal_3d_confirm__low20_minus_0_5atr` | credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rm002_erp005__signal_3d_confirm__low20` | credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rp000_erp005__signal_3d_confirm__low20` | credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rm002_erp005__signal_3d_confirm__low20_minus_0_5atr` | credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rp000_erp005__signal_3d_confirm__low20_minus_0_5atr` | credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, behaviorally redundant parameter variant, insufficient crash observations |
| `score_bo_l40_rp002_erp010__signal_3d_confirm__low20_minus_0_5atr` | statistically credible positive-skew trend follower, credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, insufficient crash observations |
| `score_bo_l40_rp002_erp015__signal_3d_confirm__low20` | statistically credible positive-skew trend follower, credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, insufficient crash observations |
| `score_bo_l40_rp002_erp010__signal_3d_confirm__low20` | credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, insufficient crash observations |
| `score_bo_l40_rp002_erp015__signal_3d_confirm__low20_minus_0_5atr` | statistically credible positive-skew trend follower, credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, insufficient crash observations |
| `score_bo_l40_rp002_erp005__signal_3d_confirm__low20` | credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, insufficient crash observations |
| `score_bo_l40_rp002_erp005__signal_3d_confirm__low20_minus_0_5atr` | credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, insufficient crash observations |
| `score_bo_l20_rp002_erp015__first_signal__low10` | credible edge but severe drawdown risk, slow crash-response strategy, high winner-period concentration, high turnover/cost fragility, likely selection-bias concern, insufficient crash observations |

These categories are overlapping diagnostics, not production qualification tiers.

Category evidence rules are explicit: PSR-vs-zero at least 95% plus positive skew; MDD magnitude at least 30%; worst 10% crash loss capture below 0.5; mean time to exposure below 50% above ten sessions or unreached; best-ten-day share above 50% of log growth; 2x-cost CAGR drop above two percentage points; raw-N DSR below 95%; behavioral group size above one; and the universal small-crash-sample warning. They are descriptive labels only.

## 12. Out-of-sample limitations and next protocol

- The same historical data influenced strategy design and selection; these are not pure out-of-sample results.
- Stationary bootstrap assumes a sufficiently stable dependent return process and can create artificial block boundaries; it cannot manufacture unseen regimes.
- PSR and DSR reduce but do not eliminate selection bias.
- Historical SPY crashes are a small, nested episode sample.
- Long-only ETF trend following is not diversified long/short managed-futures trend following.
- Future crash gaps can exceed historical stop behavior.
- Yahoo/yfinance-quality adjusted prices and vendor revisions impose research limitations.
- Detailed accounting is bounded to published curves; raw trade and ETF attribution are unavailable.

A later protocol should freeze signals, gates, portfolio rules, candidate set, and code hash before a shadow period; record every daily decision; prohibit retrospective parameter replacement; and evaluate predeclared walk-forward windows only after sufficient new regimes accumulate. That protocol is recommended but not implemented here.

## 13. Reproducibility and boundaries

All result tables are produced by `scripts/analyze_skew_aware_robustness.py` from committed bounded outputs. Deterministic tests cover common bootstrap indices and hashes, shared strategy/SPY sampling, exact group-result mapping, key-renaming and column-order invariance, primary-detail separation, duplicate-count-invariant group Pareto, group-level leaders, t0 exclusion, equity/MDD/CDaR/ES/time-under-water, drawdown/crash episodes and nesting, loss capture, exposure decay, concentration/removal, PSR/DSR, cost sensitivity, deterministic CSV ordering, production metric parity, and production-file immutability.

No full Backtest Only workflow was run. No raw event-level data were published. No signal, parameter grid, entry, exit, stop, max-hold, transaction-cost assumption, universe, allocation model, backtest period, Sample/Edge/Time/Parameter Gate, qualification tier, production ranking, UI, or risk gate changed.

## 14. Methodological correction: behavior-path-invariant bootstrap and Pareto

The prior implementation was reproducible for a given label, but it was label-dependent: `strategy_key` entered the RNG seed, so reproducibility alone did not guarantee invariance to renaming, duplicate labels, or column order. The correction establishes the same 360 deterministic 1e-12-quantized behavioral groups before bootstrap, selects the lexicographically smallest key as representative, computes each selected path once, and maps the exact serialized result back to every member.

| Before/after item | Prior PR #16 method | Corrected method |
| --- | ---: | ---: |
| Behavioral groups | 360 | 360 |
| Strict frontier groups represented | 17 | 17 |
| Tolerance frontier groups represented | 16 | 16 |
| Qualified comparison bootstrap computations | 42 label-level runs | 27 behavior-path runs |
| Production-primary strict / tolerance status | no / no | no / no |
| Final research outcome | Additional out-of-sample data required | Additional out-of-sample data required |

The group counts are unchanged because the normalized-return definition was already correct; the timing and statistical unit changed. Under the prior label output, groups `B027`, `B147`, `B167`, `B298`, `B300` had at least one strict or tolerance flag split between identical members. All corrected group members now have identical flags and dominator counts.

Label-level frontier changes:

| Strategy | Group | Prior strict / tolerance | Corrected strict / tolerance |
| --- | --- | ---: | ---: |
| `score_bo_l40_rp002_erp010__signal_3d_confirm__ma50` | `B340` | yes / yes | yes / no |
| `score_bo_l10_rm002_erp010__first_signal__low10` | `B027` | no / no | yes / yes |
| `score_bo_l20_rm002_erp015__first_signal__low10` | `B167` | no / no | yes / yes |
| `score_bo_l20_rp000_erp010__first_signal__low10` | `B147` | yes / no | yes / yes |
| `score_bo_l40_rm002_erp015__signal_3d_confirm__ma50` | `B300` | yes / no | yes / yes |
| `score_bo_l40_rp000_erp015__signal_3d_confirm__low20` | `B298` | yes / no | yes / yes |
| `score_bo_l40_rp002_erp005__signal_3d_confirm__ma50` | `B320` | yes / no | yes / yes |

Using “any prior member was on the frontier” only to compare group presence, strict group membership did not change. Two true tolerance-frontier group statuses changed while the total remained 16: `B320` changed from no to yes and `B340` changed from yes to no. The other prior split groups retained their group-level presence but now map one coherent result to every member.

B280 before and after:

| Result | P(CAGR > 0) | P(CAGR > SPY) | 5th pct CAGR | 95th pct MDD magnitude |
| --- | ---: | ---: | ---: | ---: |
| Prior: production primary (10,000 label-seeded paths) | 88.78% | 3.37% | -1.63% | 39.66% |
| Prior: equivalent label (5,000 label-seeded paths) | 89.08% | 3.04% | -1.62% | 39.72% |
| Corrected common comparison (both labels, 5,000 paths) | 88.74% | 3.20% | -1.63% | 39.23% |

Both B280 labels now receive the same comparison bootstrap statistics, strict/tolerance flags (no/no), dominator counts (1/1), and comparison-derived descriptive categories. The production primary's separate 10,000-path detailed diagnostic remains separately labeled and does not enter Pareto analysis. Its frontier status did not change.

No separate objective changed its winning behavioral group. Some label-level leader tags changed because every member of a winning group now inherits the same group result; this removes arbitrary label selection without changing the economic leader. Block-sensitivity common-matrix hashes are block 5 `c039b388e2e47cfb84e2807f288ea4ff38c4249222953b654d80416fd206b4da`, block 20 `a20988da73bd1a04c843855e91b3a958683e1f2cc5ec7a90bdb8ef62156e595f`, and block 60 `7b4690980f594fb13664995b04ca862b605d958ecd12e9fc14319550eb0bd643`; every matrix uses seed 20260728 and 5,000 paths.

The correction changes Monte Carlo estimates and two tolerance-group memberships, but it does not change production logic, the production rank, or the research conclusion.


Additional out-of-sample data required
