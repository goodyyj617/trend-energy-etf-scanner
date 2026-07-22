# Trend-Following Gate Definition Validation

Date: 2026-07-22

Scope: bounded validation of Draft PR #10 only

## 1. Decision

**Outcome: Additional aggregate data required**

Model 2 is not ready for production implementation as written. Its current 58-strategy result depends on two definitions that are not stated tightly enough in the proposal:

1. `positive_year_ratio` currently means the share of eligible entry years whose **median completed-trade return is positive**, even though the proposed trend-following rationale removes median return from the candidate-level mandatory gate.
2. The partial entry years 2017 and 2026 are counted exactly like the eight complete calendar years from 2018 through 2025.

Under the current definition, the maximum observed positive-year ratio is 0.60. All 58 Model 2 candidates have exactly six positive years out of ten. Excluding 2017 and 2026 reduces every strategy's maximum median-positive ratio to 0.50, so Models 2–4 have zero passing strategies at the proposed 0.60 Time Gate.

The bounded output is sufficient to identify this problem and to test simple alternatives, but it is not sufficient to choose a coverage-aware or rolling time-cohort definition confidently. A future bounded output should add explicit year coverage and rolling-period aggregates before the production gate is implemented.

No full backtest was run. No production Python, strategy, configuration, workflow, generated data, or UI file was changed.

## 2. Sources and validation boundary

This review uses only the currently committed bounded outputs and PR #10 analysis files:

- `docs/data/backtest_summary.json`
- `docs/data/backtest_strategy_summary.csv`
- `docs/tasks/trend_following_gate_refactor.md`
- `docs/tasks/trend_following_gate_what_if_top20.csv`

The calculations use the 540 strategy rows and the 5,400 annual strategy rows in `period_analysis[all].yearly_details`. No raw trade reconstruction was attempted.

The companion `trend_following_gate_validation_summary.csv` contains the bounded pass-count sensitivity table.

## 3. Time Gate definition

### 3.1 Exact current calculation

The bounded payload establishes the following current behavior:

- Trades are assigned to a calendar cohort by **entry year**, not exit year.
- 2017 and 2026 are included.
- The available entry interval is 2017-04-10 through 2026-07-15, so both boundary years are partial years.
- Realized exits through 2026-07-21 are included for trades whose entry dates are inside the selected period.
- A year is eligible when it has at least `min_bucket_trades = 10` completed trades for the strategy.
- An eligible year is positive when its annual `median_trade_return > 0`.
- `positive_year_ratio = positive eligible years / eligible years`.
- The current Time Stability status requires at least two eligible years and `positive_year_ratio >= 0.60`.

All 540 strategies have ten eligible years under the current minimum. Partial years receive the same one-year weight as full calendar years; there is no coverage weighting or completeness flag.

### 3.2 Annual trade-count sensitivity

Testing minimum annual completed-trade thresholds of 20, 50, and 100 does not change any ratio or Model 1–4 pass count:

| Annual minimum | Eligible years per strategy | Model 1 | Model 2 | Model 3 | Model 4 |
|---:|---:|---:|---:|---:|---:|
| 10 (current) | 10 | 235 | 58 | 58 | 56 |
| 20 | 10 | 235 | 58 | 58 | 56 |
| 50 | 10 | 235 | 58 | 58 | 56 |
| 100 | 10 | 235 | 58 | 58 | 56 |

The smallest strategy-year count in the bounded data is 135. Even the partial years are well above 100 completed trades for every strategy:

| Entry year | Minimum | Median | Maximum | Calendar coverage |
|---:|---:|---:|---:|---|
| 2017 | 135 | 405 | 1,326 | Partial from 2017-04-10 |
| 2018 | 137 | 431 | 1,219 | Full |
| 2019 | 241 | 776 | 1,926 | Full |
| 2020 | 252 | 665 | 1,959 | Full |
| 2021 | 216 | 594 | 2,242 | Full |
| 2022 | 456 | 734 | 1,328 | Full |
| 2023 | 486 | 898 | 2,011 | Full |
| 2024 | 251 | 891 | 3,021 | Full |
| 2025 | 429 | 1,063 | 3,159 | Full |
| 2026 | 208 | 551 | 1,646 | Partial through 2026-07-15 |

**Annual-count recommendation:** use 100 as a provisional eligibility floor because it is conservative, aligns with the candidate-level sample floor, and is non-binding throughout this dataset. This is not an empirically optimized threshold: the observed distribution cannot distinguish 20, 50, and 100. The threshold must remain serialized and revisable.

Trade count does not solve partial-year comparability. The 2026 half-year cohort has ample trades but is still not a full calendar regime observation.

### 3.3 Full-calendar-years-only sensitivity

Removing partial 2017 and 2026 and retaining the current annual median-positive definition gives:

| Definition | Model 1 | Model 2 | Model 3 | Model 4 | Maximum ratio |
|---|---:|---:|---:|---:|---:|
| Current: 2017–2026, median-positive | 235 | 58 | 58 | 56 | 0.60 |
| Full years: 2018–2025, median-positive | 305 | 0 | 0 | 0 | 0.50 |

Model 1 rises because four positive years out of eight equals 0.50, while the same pattern plus one positive and one negative partial year is five out of ten. Models 2–4 fall to zero because no strategy has five median-positive full years out of eight.

This means the proposed 0.60 result is not invariant to calendar coverage.

### 3.4 Positive-year definition sensitivity

The proposed candidate Edge Gate uses Profit Factor and average trade return, but the existing Time Gate still uses annual median return. This reintroduces the same trend-following distribution concern at the annual level.

The bounded annual rows permit a diagnostic alternative: define a positive year as `avg_trade_return > 0` and `profit_factor >= 1.0`. In this dataset those two annual conditions select exactly the same strategy-year rows.

Using a provisional annual minimum of 100:

| Positive-year definition | Years | Model 1 | Model 2 | Model 3 | Model 4 | Maximum ratio |
|---|---|---:|---:|---:|---:|---:|
| Median return > 0 | 2017–2026 | 235 | 58 | 58 | 56 | 0.60 |
| Median return > 0 | 2018–2025 | 305 | 0 | 0 | 0 | 0.50 |
| Avg return > 0 and PF >= 1 | 2017–2026 | 407 | 292 | 266 | 169 | 0.80 |
| Avg return > 0 and PF >= 1 | 2018–2025 | 407 | 60 | 41 | 0 | 0.75 |

The expectancy-based full-year alternative produces a plausible nonzero set, but it is not a harmless clarification: the Model 2 family changes materially, and the lexicographic top candidate becomes `score_bo_l20_rp000_erp015__signal_2d_confirm__low10`. This is evidence that the Time Gate definition is a substantive research decision, not an implementation detail.

## 4. Time robustness sensitivity

### 4.1 Proposed Model 2 edge population

There are 368 strategies passing the proposed Model 2 Sample and Edge gates before applying Time:

- `completed_trades >= 100`
- `profit_factor >= 1.2`
- `avg_trade_return > 0`

Under the current annual median-positive definition:

| Positive-year threshold | Passing strategies |
|---:|---:|
| 0.50 | 235 |
| 0.60 | 58 |
| 0.70 | 0 |

**A 0.70 threshold is infeasible under the current definition because no strategy reaches 0.70.** The maximum is 0.60.

Under the diagnostic full-calendar-year expectancy definition:

| Positive-year threshold | Passing strategies |
|---:|---:|
| 0.50 | 368 |
| 0.60 | 60 |
| 0.70 | 2 |

Therefore 0.70 is not universally infeasible; it is infeasible only under the current median-positive definition. That contrast further demonstrates that the positive-year metric must be fixed before selecting a ratio threshold.

### 4.2 Leave-one-entry-year-out sensitivity

The annual bounded rows allow exclusion of one eligible entry year at a time without reconstructing raw trades.

For the current Model 2 definition:

- All 58 candidates have exactly six positive years out of ten.
- Excluding one of their six positive years changes the ratio to 5/9 = 0.556 and fails 0.60.
- Excluding one of their four negative years changes the ratio to 6/9 = 0.667 and passes.
- Every candidate retains qualification in exactly 4 of the 10 leave-one-year-out cases.
- No candidate passes all ten leave-one-year-out cases.
- Omitting 2017, 2019, 2020, 2023, 2025, or 2026 produces zero Model 2 candidates.
- Omitting 2018, 2021, 2022, or 2024 leaves all 58 candidates.

The qualifying set therefore moves as a single block rather than demonstrating independent family-level resilience. The 58-candidate result is a six-year regime pattern, not a stable supermajority across years.

For the diagnostic full-year expectancy definition:

- 60 strategies pass Model 2.
- 58 pass only 3 of 8 leave-one-full-year-out cases.
- Two strategies pass all 8 cases; both are the behaviorally identical R20 -0.02/0.00 versions of L20 / ER20 0.15 / Signal 2D confirm / Low10.

This alternative is more leave-one-out selective, but it also elevates a different family. Production implementation should not embed that change without an explicit research decision.

## 5. Parameter-neighbor redundancy

### 5.1 Raw neighbor definition

PR #10 defines direct neighbors by holding entry and exit fixed and changing exactly one Score Breakout parameter to an adjacent grid value:

- lookback: 10, 20, 40
- R20 minimum: -0.02, 0.00, 0.02
- ER20 minimum: 0.05, 0.10, 0.15

An eligible neighbor has at least 100 completed trades. For this validation, a neighbor passes Model 2 edge when:

```text
completed_trades >= 100
and profit_factor >= 1.2
and avg_trade_return > 0
```

The raw edge pass ratio is passing eligible direct neighbors divided by eligible direct neighbors.

### 5.2 Effective deduplication used in this validation

The bounded output is sufficient to deduplicate exact observed behaviors. An effective state was defined by exact equality of:

- aggregate completed trades, average return, median return, win rate, and Profit Factor; and
- the complete ten-year vector of annual completed trades, average return, median return, and Profit Factor.

A direct neighbor with the same observed signature as the candidate was removed, and remaining neighbors with duplicate signatures were counted once.

Results:

| Population | Raw Parameter pass | Effective Parameter pass | Notes |
|---|---:|---:|---|
| All 540 candidate states | 364 | 362 | 42 ratios changed |
| 368 Model 2 edge-pass states | 364 | 361 | 248 contain a removed redundant neighbor |
| 58 current Model 2 qualified states | 58 | 58 | All remain at ratio 1.00 |

Across the full grid, 360 candidate states have one redundant direct neighbor removed, corresponding to the confirmed exact R20 -0.02/0.00 duplication. The deduplication changes only a few pass/fail decisions at the proposed 0.60 Parameter threshold, but it materially changes the claimed neighborhood breadth.

For the 58 current Model 2 candidates:

- raw neighbor edge pass ratio: 1.00 for all 58;
- effective neighbor edge pass ratio: 1.00 for all 58;
- effective neighbor counts: 2 neighbors for 2 candidates, 3 for 21, 4 for 27, and 5 for 8.

The current What-if top candidate has three raw and three effective neighbors, all passing edge. The previous dashboard leader has three raw neighbors but only two effective neighbors after its R20-duplicate state is removed; its ratio remains 1.00.

**Recommendation:** Parameter Stability must report both pass ratio and effective eligible-neighbor count. A 100% ratio over two effective states is not the same evidence as 100% over five.

For reproducible future deduplication, add a bounded `effective_signal_state_id` or `outcome_fingerprint` to each strategy row. It should be derived from the aggregate and time-cohort outcome vector, not from the parameter label. No new raw event file is required.

## 6. Candidate qualification and ranking

### 6.1 Ranking used by PR #10

PR #10 first filters strategies through each model's gates, then ranks only the passing set by:

1. higher `positive_year_ratio`;
2. higher `profit_factor`;
3. higher `avg_trade_return`;
4. more `completed_trades`;
5. lexicographically smaller `strategy_key`.

Qualification and ranking are separate. A ranking field does not compensate for a failed gate. No weighted composite score is used.

### 6.2 Recommended deterministic ranking

After the Time definition is resolved, use this lexicographic order:

1. pass all mandatory Sample, Edge, and Time gates;
2. higher defined Time robustness, including positive-year ratio and a serialized leave-one-period-out diagnostic;
3. higher effective neighbor edge pass ratio, then more effective eligible neighbors;
4. higher Profit Factor;
5. higher average trade return;
6. more completed trades;
7. lexicographically smaller `strategy_key` as the final stable tie-break.

Mandatory gate count should not be used to rank failed strategies as though two partial failures were investable candidates. Apply the gates as a filter, then rank qualified candidates.

Using the current ten-year median-positive definition, this ranking does **not** change the top candidate for any of Models 1–4. Time robustness and effective Parameter ratio are tied at the top, so Profit Factor still selects:

`score_bo_l10_rp002_erp005__first_signal__low20`

Using the diagnostic full-year expectancy definition does change the Model 2 top candidate to:

`score_bo_l20_rp000_erp015__signal_2d_confirm__low10`

It has six positive full years out of eight and survives all eight leave-one-full-year-out checks. Its R20 -0.02 twin has the same observed behavior. This ranking reversal is another reason not to implement until the time-cohort definition is approved.

## 7. Exit-family interpretation

### 7.1 Why the leaders differ

The previous dashboard leader is:

`score_bo_l10_rm002_erp005__breakout_5d_after_signal__low20_minus_0_5atr`

- 5D breakout after signal / Low20 - 0.5ATR
- median trade return: -0.0900%
- Profit Factor: 1.6304
- average trade return: 1.0153%
- current positive-year ratio: 0.50
- raw/effective neighbor edge ratios: 1.00 / 1.00

The PR #10 What-if leader is:

`score_bo_l10_rp002_erp005__first_signal__low20`

- First signal / Low20
- median trade return: -0.3018%
- Profit Factor: 1.7291
- average trade return: 1.2354%
- current positive-year ratio: 0.60
- raw/effective neighbor edge ratios: 1.00 / 1.00

The difference is attributable to:

- **Removal of the median mandatory gate:** necessary for either negative-median trend-following candidate to qualify under the new design.
- **Time Gate:** decisive against the previous leader, which has 0.50 versus the new leader's 0.60.
- **Ranking order:** decisive among qualified strategies. The old dashboard prioritizes median return before Profit Factor; PR #10 prioritizes Time and then Profit Factor.
- **Profit-Factor preference:** directly selects First signal / Low20 as the What-if top candidate.
- **Average-return preference:** not the main cause. First signal / Low20 - 0.5ATR has a higher average return of 1.3195% but a lower Profit Factor of 1.7050.
- **Parameter Stability:** not a differentiator; both leaders have raw and effective ratios of 1.00.

### 7.2 Leaders within the current 58-strategy Model 2 set

| Criterion | Leader | Value |
|---|---|---:|
| Highest Profit Factor | L10 / R20 0.02 / ER20 0.05 / First signal / Low20 | 1.7291 |
| Highest average trade return | L10 / R20 0.02 / ER20 0.05 / First signal / Low20 - 0.5ATR | 1.3195% |
| Best median trade return | L10 / ER20 0.15 / 5D breakout / Low20 - 0.5ATR, tied R20 -0.02 and 0.00 | -0.1027% |
| Best current Time robustness | All 58 tie | 0.60; each passes 4/10 leave-one-year-out cases |
| Best effective Parameter ratio | All 58 tie | 1.00 |
| Widest effective neighborhood among ratio leaders | Eight strategies tie | 5 effective neighbors |

No single strategy dominates every criterion. The What-if top is the Profit Factor leader after Time qualification, not the median, average-return, or uniquely most stable leader.

## 8. Required definition changes and additional bounded data

Before production implementation, approve and serialize all of the following:

1. **Cohort assignment:** continue using entry date; state this explicitly.
2. **Calendar coverage:** decide whether incomplete boundary years are excluded, weighted, or reported separately. Do not count them silently as full years.
3. **Positive-year metric:** choose annual median return or a trend-following-compatible annual expectancy definition. The latter should be tested as `avg_trade_return > 0` and `profit_factor >= 1.0`.
4. **Annual eligibility:** provisionally use at least 100 completed trades, while documenting that 20/50/100 are indistinguishable in this sample.
5. **Time sensitivity:** serialize leave-one-period-out retention rather than relying only on one ratio.
6. **Parameter breadth:** serialize effective neighbor count and an effective state identifier.

The next bounded backtest output should add, per strategy and time cohort:

- `cohort_start` and `cohort_end`;
- `calendar_year_complete`;
- observed and expected trading-day coverage, or an equivalent coverage ratio;
- completed trades, average return, median return, and Profit Factor, which already exist for calendar-year rows;
- bounded fixed or rolling 12-month cohort summaries with the same metrics;
- a strategy-level leave-one-cohort-out pass count or minimum ratio;
- `effective_signal_state_id` / `outcome_fingerprint` for neighbor deduplication.

These are aggregate additions. Millions of raw trades do not need to be published.

## 9. Final recommendation

**Additional aggregate data required**

Do not implement Model 2 production gates from PR #10 yet. The Sample and candidate-level Edge gates are sufficiently defined, and a provisional annual minimum of 100 is reasonable. The Time Gate is not: its 58 passes depend on a median-positive annual metric and on counting two partial years as full observations, and none of the 58 survives every one-year exclusion.

Add coverage-aware and rolling time aggregates in the next planned backtest-output revision, then rerun this bounded validation. Keep PR #10 as a design and validation PR; do not open a production implementation PR from this task.
