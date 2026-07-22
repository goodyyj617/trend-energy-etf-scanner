# Trend-Following Time Gate Analysis

## Decision

Use a conservative provisional Time Robustness Gate:

1. assign trades by **entry year**;
2. use only `is_full_calendar_year = true` years;
3. require at least `100` completed trades for an eligible year and at least `5` eligible years;
4. define an annual edge year as both `avg_trade_return > 0` and `profit_factor > 1.0`;
5. require `joint_positive_year_ratio >= 0.60`; and
6. require leave-one-eligible-year-out (`LOYO`) pass ratio `>= 0.80`, where each fold must retain at least 100 completed trades, positive pooled average return, and pooled profit factor `>= 1.2`.

With the eight full years currently available, the ratio cutoffs are discrete: `0.60` means at least 5/8 positive years (`0.625` observed), and `0.80` LOYO means at least 7/8 passing folds (`0.875` observed). The recommended Time Gate leaves 42 of the 368 base candidates. All 42 also pass the separately proposed effective-neighbor parameter gate.

The recommendation is suitable for a provisional production implementation because it is explicit, reproducible, non-degenerate, and stable to one-year removal for at least seven of eight folds. It is not evidence of out-of-sample durability. Bootstrap results remain diagnostic and must not become an approval threshold from this backtest alone.

**Outcome: Ready with conservative provisional thresholds**

## Scope and reproducibility

This analysis used only the bounded generated outputs:

- `docs/data/backtest_summary.json`
- `docs/data/backtest_strategy_summary.csv`
- `docs/data/backtest_strategy_year_summary.csv`

The full backtest was not rerun. No raw event-level trades were used or committed. No strategy, signal, entry, exit, stop, cost, universe, period, production gate, ranking, or UI logic was changed.

The bounded result tables are:

- [`trend_following_time_gate_scenarios.csv`](trend_following_time_gate_scenarios.csv): all 16 annual-definition/threshold scenarios, family distributions, concentration measures, and robustness intersections.
- [`trend_following_time_gate_top_candidates.csv`](trend_following_time_gate_top_candidates.csv): up to 20 deterministically ranked candidates per scenario, including annual, LOYO, bootstrap, and raw/effective neighbor fields. Scenarios with fewer than 20 passing candidates contain all passers.

## 1. Aggregate validation

Validation passed before any gate analysis continued.

| Check | Result |
| --- | ---: |
| Annual rows | 5,400 |
| Strategies | 540 |
| Minimum / maximum entry year | 2017 / 2026 |
| Partial years | 2017, 2026 |
| Full calendar years | 2018–2025 |
| Annual aggregation runtime | 3.720249 seconds |
| Sort order | deterministic by `strategy_key`, then `entry_year` |
| Duplicate `strategy_key × entry_year` rows | 0 |
| Missing required columns | 0 |
| Year basis | `entry_year` only |

The published metadata and annual file agree on row count, year range, partial years, full years, and year basis. Validation used `rtol = 1e-12` and `atol = 1e-10`, consistent with the aggregate design's floating-point guidance.

Every strategy reconstructed successfully. Annual completed trades exactly equal overall completed trades; annual winning, losing, and flat counts exactly partition completed trades; and all annual flat counts sum to zero. The overall file does not publish separate winning/losing/flat columns, so winning trades were checked against `completed_trades × trade_win_rate`, losing trades were the remaining non-flat trades, and flat trades were confirmed as zero. The largest implied winning-count difference was `1.82e-12`.

The largest absolute reconstruction errors were:

| Reconstructed metric | Maximum absolute error |
| --- | ---: |
| `sum_trade_returns` | 4.26e-14 |
| weighted `avg_trade_return` | 9.97e-17 |
| `gross_profit` | 2.79e-11 |
| `gross_loss_abs` | 2.79e-11 |
| `profit_factor` | 6.66e-16 |

Because the overall summary does not directly publish gross-profit and gross-loss numerators, they were inferred from overall `sum_trade_returns` and `profit_factor`, then matched to the sums of the annual numerators. All errors are within tolerance.

## 2. Eligible-year definition

The primary analysis uses entry year, not exit year. A year is eligible only when:

- `is_full_calendar_year = true`; and
- annual `completed_trades >= 100`.

At least five eligible years are required. In the current data, all 540 strategies have all eight full years eligible at each tested annual minimum:

| Annual minimum trades | Strategies with 8 eligible years | Minimum eligible years | Base candidates |
| ---: | ---: | ---: | ---: |
| 20 | 540 | 8 | 368 |
| 50 | 540 | 8 | 368 |
| 100 | 540 | 8 | 368 |

Therefore 20, 50, and 100 have no current selection sensitivity. Retaining 100 is still preferable as a conservative definition for future runs: it supplies the requested annual sample floor without changing this result. It must remain an explicit configuration rather than an inferred property of this dataset.

The base set applies the proposed overall edge conditions before annual ratio testing:

- overall `completed_trades >= 100`;
- overall `profit_factor >= 1.2`;
- overall `avg_trade_return > 0`; and
- at least 5 eligible full years.

This yields 368 base candidates.

### Partial years, descriptive only

Partial 2017 and 2026 are excluded from every primary Time Gate, LOYO, and bootstrap calculation.

| Entry year | Covered period | Base rows | Completed trades min / median / max | Avg > 0 | PF > 1 | Strong joint | Median annual avg | Median annual PF |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 2017 | 2017-04-10–2017-12-31 | 368 | 135 / 428 / 1,142 | 368 | 368 | 368 | 0.011781 | 2.3083 |
| 2026 | 2026-01-01–2026-07-15 | 368 | 208 / 510 / 1,292 | 265 | 265 | 246 | 0.010455 | 1.6292 |

Their favorable results illustrate why partial years must not be silently treated as full calendar years.

## 3. Annual edge definitions

For every strategy, the requested annual counts, ratios, annual medians, and annual minima were calculated before forming the base set. They are included for each reported candidate in the top-candidate CSV.

| Definition | Annual pass condition | Result in full-year data |
| --- | --- | --- |
| A — Average edge | `avg_trade_return > 0` | Identical to B and C |
| B — PF edge | `profit_factor > 1.0` | Identical to A and C |
| C — Joint edge | both A and B | Identical to A and B |
| D — Strong joint | A and `profit_factor >= 1.2` | Stricter |

A, B, and C disagree on zero of the 4,320 full-calendar strategy-year rows. This is not three independent confirmations: with positive trade count and gross loss, positive average return, positive summed return, and profit factor above one are algebraically equivalent. Definition C is recommended for semantic clarity and to make both intended edge conditions auditable.

Definition D differs from C on 191 full-calendar strategy-year rows across 152 strategies. Within the 368-candidate base set, it differs on 53 rows across 49 strategies. Requiring annual PF `>= 1.2` therefore adds meaningful severity, but the sensitivity results below show that it over-concentrates the surviving family.

The full-year regimes are highly shared across candidates:

| Year | Median trades | Joint-positive strategies | Strong-joint strategies | Median annual avg | Median annual PF |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 2018 | 448.5 | 11 | 2 | -0.007855 | 0.5305 |
| 2019 | 751.0 | 368 | 368 | 0.023673 | 4.0587 |
| 2020 | 651.0 | 368 | 366 | 0.037553 | 3.1552 |
| 2021 | 611.0 | 0 | 0 | -0.009588 | 0.4926 |
| 2022 | 733.0 | 0 | 0 | -0.024023 | 0.2326 |
| 2023 | 873.0 | 368 | 368 | 0.024060 | 3.3578 |
| 2024 | 871.0 | 54 | 12 | -0.005765 | 0.6931 |
| 2025 | 1,036.5 | 368 | 368 | 0.020629 | 2.7573 |

This common regime structure is an important limitation: a positive-year ratio is partly a market-period test, not purely a strategy-specific stability test.

## 4. Threshold sensitivity

All 16 requested scenarios were evaluated. “Effective results” deduplicate identical aggregate-result configurations; concentration is based on the largest share on any single exit, entry, or signal-parameter axis, plus the effective count.

| Definition | Ratio | Required years | Passing | Effective results | Parameter pass | LOYO >= 0.80 | Time + parameter + LOYO | Concentration |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| A | 0.50 | 4/8 | 368 | 244 | 361 | 313 | 313 | broad |
| A | 0.60 | 5/8 | 61 | 38 | 56 | 42 | 42 | moderate |
| A | 0.70 | 6/8 | 4 | 2 | 3 | 0 | 0 | very narrow |
| A | 0.75 | 6/8 | 4 | 2 | 3 | 0 | 0 | very narrow |
| B | 0.50 | 4/8 | 368 | 244 | 361 | 313 | 313 | broad |
| B | 0.60 | 5/8 | 61 | 38 | 56 | 42 | 42 | moderate |
| B | 0.70 | 6/8 | 4 | 2 | 3 | 0 | 0 | very narrow |
| B | 0.75 | 6/8 | 4 | 2 | 3 | 0 | 0 | very narrow |
| C | 0.50 | 4/8 | 368 | 244 | 361 | 313 | 313 | broad |
| C | 0.60 | 5/8 | 61 | 38 | 56 | 42 | 42 | moderate |
| C | 0.70 | 6/8 | 4 | 2 | 3 | 0 | 0 | very narrow |
| C | 0.75 | 6/8 | 4 | 2 | 3 | 0 | 0 | very narrow |
| D | 0.50 | 4/8 | 366 | 243 | 361 | 313 | 313 | broad |
| D | 0.60 | 5/8 | 14 | 9 | 12 | 12 | 12 | very narrow |
| D | 0.70 | 6/8 | 0 | 0 | 0 | 0 | 0 | very narrow |
| D | 0.75 | 6/8 | 0 | 0 | 0 | 0 | 0 | very narrow |

The complete entry, exit, lookback, `r20_min`, and `er20_min` distributions are serialized in the scenario CSV. The main family effects are:

- C at 0.50 performs no time filtering: all 368 base candidates pass.
- C at 0.60 leaves 61 candidates: exit rules are Low10 34, Low20 9, Low20 - 0.5ATR 9, and MA50 9; entries are 3D confirm 27, First signal 15, 2D confirm 13, and 5D breakout 6. The largest single-axis share is 55.7%, so selection is moderately concentrated but not a single-cell result.
- C at 0.70 and 0.75 are identical because six of eight years is 0.75. Only four rows pass, representing two effective behaviors; all use L20 and Low10, and the `r20_min = -0.02` / `0.00` pairs are identical. No candidate also reaches LOYO 0.80.
- D at 0.60 leaves 14 rows and only 9 effective results; 12 of 14 are 3D-confirm/L40 configurations. Its 85.7% dominant-axis share is too narrow for a primary production definition.

Threshold 0.60 is recommended because it creates a real time filter while retaining multiple entry, exit, and parameter families. It is not selected to target a convenient candidate count: 0.50 is non-discriminating, while 0.70/0.75 collapse to a behaviorally redundant corner and fail the requested LOYO sensitivity.

## 5. Leave-one-year-out robustness

For every base candidate and omitted eligible year, pooled completed trades, summed returns, gross profit, and gross loss were reconstructed from annual numerators and denominators. Pooled average return and profit factor were then recomputed; annual ratios were never averaged.

A fold passes when pooled completed trades are at least 100, pooled average trade return is positive, and pooled profit factor is at least 1.2.

| LOYO threshold | Required folds | Base pass | C at 0.60 pass |
| ---: | ---: | ---: | ---: |
| 0.75 | 6/8 | 337 | 48 |
| 0.80 | 7/8 | 313 | 42 |
| 1.00 | 8/8 | 270 | 24 |

The base LOYO pass-ratio distribution is: 0.375 (4), 0.500 (11), 0.625 (16), 0.750 (24), 0.875 (43), and 1.000 (270). A threshold of 1.00 remains a strict diagnostic because it gives any single influential positive year an absolute veto.

Within the 42 recommended Time Gate candidates, 24 pass all eight folds and 18 fail exactly one fold. Their omitted-year pass counts are:

| Omitted year | Base pass | C at 0.60 pass | Recommended-set pass |
| ---: | ---: | ---: | ---: |
| 2018 | 364 / 368 | 57 / 61 | 42 / 42 |
| 2019 | 340 / 368 | 42 / 61 | 42 / 42 |
| 2020 | 286 / 368 | 36 / 61 | 25 / 42 |
| 2021 | 368 / 368 | 61 / 61 | 42 / 42 |
| 2022 | 368 / 368 | 61 / 61 | 42 / 42 |
| 2023 | 328 / 368 | 41 / 61 | 41 / 42 |
| 2024 | 368 / 368 | 61 / 61 | 42 / 42 |
| 2025 | 319 / 368 | 48 / 61 | 42 / 42 |

Removing 2020 most often produces the worst pooled average return (223 candidates) and worst pooled PF (169); 2023 and 2025 are the other observed worst omissions. The table makes clear that LOYO tests dependence on unusually helpful years: removing losing years such as 2021 or 2022 naturally improves the pooled result.

LOYO `>= 0.80` is recommended as a conservative companion to the annual ratio. It rejects C-at-0.60 candidates whose overall edge depends too heavily on more than one favorable year while avoiding the absolute-veto behavior of 1.00.

## 6. Year-block bootstrap

The bootstrap used only the eight eligible full-year aggregate blocks. For each strategy, 10,000 samples of eight years with replacement were drawn with NumPy's `default_rng` seed `20260723`. Each sampled year preserves its completed-trade count, summed return, gross profit, and gross loss; pooled average return and PF are reconstructed from those totals.

| Candidate set | Count | Median P(avg > 0 and PF > 1) | Median P(PF >= 1.2) | Median 5th pct avg | Median 5th pct PF | 5th pct avg > 0 | 5th pct PF > 1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Base | 368 | 0.8814 | 0.7626 | -0.003359 | 0.8262 | 6 | 6 |
| C at 0.60 | 61 | 0.8418 | 0.7194 | -0.003058 | 0.7759 | 0 | 0 |
| Recommended Time Gate | 42 | 0.8431 | 0.7326 | -0.005875 | 0.7173 | 0 | 0 |

For these aggregates, P(`avg > 0`) equals P(`PF > 1`) and their joint probability because the conditions are algebraically equivalent within a pooled sample. The separate fields are retained in the candidate CSV for auditability.

No recommended candidate has a positive 5th-percentile average return or a 5th-percentile PF above one. This is a material warning about shared bad-year regimes and the small number of independent annual blocks. It does not invalidate the deterministic gate, but it rules out using a bootstrap confidence cutoff as a production approval rule from this one in-sample period.

## 7. Parameter robustness interaction

Parameter robustness uses direct adjacent signal parameters within the same entry/exit family. A parameter gate passes when there are at least two effective neighbors and the effective neighbor edge-pass ratio is at least 0.60. A neighbor's edge passes with overall completed trades `>= 100`, PF `>= 1.2`, and average trade return `> 0`.

Raw neighbors count parameter labels. Effective neighbors remove configurations whose full overall-plus-annual aggregate result is identical to the candidate and deduplicate identical neighbor result signatures. This explicitly handles the confirmed behaviorally identical `r20_min = -0.02` and `0.00` pairs.

| Result | Base candidates |
| --- | ---: |
| Raw-neighbor parameter gate pass | 364 / 368 |
| Effective-neighbor parameter gate pass | 361 / 368 |
| Raw/effective ratio changed | 20 |
| Effective neighbors = 2 / 3 / 4 / 5 | 53 / 161 / 125 / 29 |

At C 0.60, 56 of 61 pass the effective parameter gate. After adding recommended LOYO `>= 0.80`, all 42 Time Gate candidates pass it. Thus the revised Time Gate does not leave parameter-fragile candidates in the selected set, while raw and effective evidence remain separately visible in the top-candidate CSV.

Across all base candidates, 368 labeled configurations reduce to 244 unique aggregate-result signatures. This is why raw row counts must not be interpreted as independent behavioral evidence.

## 8. Candidate leaders and deterministic ranking

Gate qualification is separate from ranking. No weighted composite score is used. The primary candidate is selected lexicographically in the requested order:

1. recommended Time Gate pass;
2. effective parameter gate pass;
3. LOYO pass ratio;
4. annual joint-positive ratio;
5. bootstrap probability of average return above zero and PF above one;
6. overall PF;
7. overall average trade return;
8. completed trades; and
9. `strategy_key` ascending as the deterministic final tie-breaker.

The primary candidate is:

`score_bo_l20_rm002_erp010__first_signal__low10`

Score BO L20 R20>=-0.02 ER20>=0.10 / First signal / Low10 trailing

- completed trades: 13,689;
- overall PF: 1.394261;
- overall average trade return: 0.004210;
- joint-positive ratio: 5/8 = 0.625;
- LOYO: 8/8 = 1.000;
- bootstrap joint probability: 0.9455;
- bootstrap 5th-percentile average / PF: -0.000124 / 0.9895; and
- effective neighbors: 4, with edge-pass ratio 0.75.

Its `r20_min = 0.00` counterpart has identical aggregate behavior. The `-0.02` key wins only the final lexical tie-break and should be treated as a representative of one effective result, not as independent superiority.

Separate leaders within the C-at-0.60 qualifying set are:

| Objective | Leader(s) | Value |
| --- | --- | ---: |
| Highest overall PF | L40, R20 -0.02/0.00, ER20 0.10, 3D confirm, MA50 | 1.439783 |
| Highest overall average return | L40, R20 0.02, ER20 0.10, 3D confirm, MA50 | 0.008026 |
| Best annual joint-positive ratio | four L20 Low10 rows: R20 -0.02/0.00 crossed with 5D ER20 0.05 or 2D ER20 0.15 | 0.750 |
| Best LOYO ratio | 24 tied rows | 1.000 |
| Best bootstrap downside | primary candidate and its R20 0.00 twin | 5th pct avg -0.000124; PF 0.9895 |
| Best effective parameter robustness | 37 tied rows | 1.000 |
| Best median trade return, diagnostic | L40, R20 -0.02/0.00, ER20 0.15, 3D confirm, MA50 | -0.002459 |

The primary is not the annual-ratio leader because LOYO is ranked first after the gates. The four 0.75 annual-ratio rows do not reach the recommended LOYO threshold.

## 9. Rejected alternatives

- **Include partial 2017/2026:** rejected because coverage differs from full years and both partial periods are unusually favorable in this sample.
- **Annual minimum 20 or 50:** currently equivalent to 100, so they add no evidence and provide a weaker future sample floor.
- **Definition A or B alone:** currently equivalent to C, but less explicit about the intended joint edge semantics.
- **Definition D at 0.60:** rejected as the primary gate because only 14 labeled rows/9 effective results survive and 12 rows occupy the same 3D-confirm/L40 region.
- **Ratio 0.50:** rejected because it passes all 368 base candidates under A/B/C and therefore adds no Time Gate discrimination.
- **Ratio 0.70 or 0.75:** rejected because both mean six of eight years, leave four labeled/two effective results, and leave no candidate with LOYO `>= 0.80`.
- **LOYO 1.00:** retained as a strict diagnostic, not recommended as an automatic veto.
- **Bootstrap approval threshold:** rejected for this task because there are only eight in-sample annual blocks and no recommended candidate has positive 5th-percentile edge under resampling.
- **Annual median trade return:** diagnostic only; it is not used to declare a positive year or to gate candidates.

## 10. Remaining limitations

- The analysis contains only eight full calendar blocks and one market history. Annual observations are not independent experiments.
- Candidate families share the same broad winning and losing years, so positive-year counts partly measure common regime exposure.
- The bootstrap is an in-sample year-block resampling, not a walk-forward or out-of-sample test.
- The annual minimum sensitivity is uninformative in this run because every strategy exceeds 100 trades in every full year.
- Many labeled configurations are behaviorally identical, especially `r20_min = -0.02` and `0.00`; effective deduplication reduces but cannot create independent market evidence.
- A provisional production implementation should log the exact eligible years, annual pass counts, LOYO folds, and effective-neighbor evidence so the thresholds can be revisited as new complete calendar years accumulate.

Within those limits, production implementation of the explicitly defined conservative provisional gate is justified. It should not be described as out-of-sample validation, and this analysis does not authorize any strategy or ranking change.
