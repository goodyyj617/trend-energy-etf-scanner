# Trend-Following Gate Refactor Design and What-If Analysis

## Status and Scope

- Type: research design note and bounded what-if analysis.
- Source commit: `e431a61` (`Update backtest data`).
- Source files: `docs/data/backtest_summary.json` and `docs/data/backtest_strategy_summary.csv`.
- Analysis period: completed trades entered from 2017-04-10 through 2026-07-15, with realized exits through 2026-07-21.
- Source population: 540 strategies, 4,130,637 completed trade rows, schema v2, diagnostics disabled, and `diagnostic_event_count = 0`.
- No backtest was rerun. This note does not change production gates, tiers, UI, strategy logic, signals, entries, exits, stops, costs, universe rules, or period logic.
- The bounded 80-row top-candidate extract is in `docs/tasks/trend_following_gate_what_if_top20.csv`.

## 1. Current Problem

The current mandatory gate model is not compatible with the observed positive-skew trend-following return profile.

- All 540 strategies fail `median_trade_return >= 0` and therefore receive `Not qualified`.
- 458 strategies have Profit Factor at or above 1.0.
- The same 458 strategies have positive average trade return.
- A negative median alongside positive average return and Profit Factor above 1.0 is consistent with a distribution in which frequent small losses are offset by fewer, larger winners. It is not sufficient evidence that the strategy lacks edge.
- Current Parameter Stability is downstream-distorted. `neighbor_pass_ratio` requires neighbors to pass the full mandatory gate set, so the universal median failure forces every current neighbor ratio to zero and every Parameter Stability status to `Fail`.
- Median Trade Return remains useful for diagnosing win/loss shape, typical-trade experience, and dependence on large winners. It should not remain an automatic disqualifier.

The current labels should therefore be interpreted as a gate-model failure, not as evidence that all 540 strategies are unusable.

## 2. Proposed Trend-Following Gate Model

### Mandatory gates

| Component | Proposed rule | Rationale |
|---|---|---|
| Sample Gate | `completed_trades >= 100` | Retains the existing minimum sample requirement. |
| Edge Gate | `profit_factor >= 1.2` and `avg_trade_return > 0` | Requires positive aggregate trade edge without assuming that the typical trade must win. |
| Time Gate | `eligible_years >= 5` and `positive_year_ratio >= 0.60` | Requires the edge to occur across multiple entry-year cohorts. |
| Risk Gate | Descriptive initially | Existing event-level downside metrics are not portfolio risk metrics, and no approved threshold currently exists. |

`median_trade_return` should stay visible but move from mandatory gate to diagnostic. The UI and data contract should continue to report it with Win Rate, Profit Factor, average return, tail return, Worst Trade Ret, and stop behavior.

### Proposed qualification and tiers

- **Qualified edge:** Sample, Edge, and Time gates pass.
- **Broadly stable candidate:** Qualified edge plus proposed Parameter Stability pass.
- **Mixed neighborhood:** Qualified edge with insufficient or sub-threshold Parameter Stability.
- **Not qualified:** Sample, Edge, or Time gate fails.
- **Risk unverified:** Explicit suffix or badge until a risk gate based on an approved metric is implemented.

The deterministic what-if ranking used in this note is:

1. Higher `positive_year_ratio`.
2. Higher `profit_factor`.
3. Higher `avg_trade_return`.
4. More completed trades.
5. Lexicographically smaller `strategy_key` for a final stable tie-break.

This ranking is for analysis only and is not a proposed production-code change yet.

## 3. Proposed Parameter Stability Model

Parameter Stability should not inherit the full candidate qualification result.

For each direct grid neighbor, holding entry and exit rules fixed, define:

```text
neighbor_edge_pass =
    completed_trades >= min_completed_trades
    and profit_factor >= min_profit_factor
    and avg_trade_return > 0
```

Then report:

- `neighbor_edge_pass_ratio`: eligible direct neighbors passing the edge definition divided by eligible direct neighbors.
- `neighbor_median_profit_factor`: median Profit Factor across eligible direct neighbors.
- `neighbor_avg_trade_return_min`: minimum average trade return across eligible direct neighbors.
- `neighbor_time_pass_ratio`: share of eligible direct neighbors with `eligible_years >= 5` and `positive_year_ratio >= 0.60`.

Proposed starting Parameter Stability gate:

- At least two eligible direct neighbors.
- `neighbor_edge_pass_ratio >= 0.60`.
- Keep `neighbor_time_pass_ratio` descriptive until its interaction with the candidate Time Gate is reviewed.

This recalculation is feasible using the existing 540-row strategy summary. Under Model 2, all 58 passing candidates have `neighbor_edge_pass_ratio = 1.0` and would pass this proposed edge-neighborhood gate. The top candidate has:

- 3 direct neighbors.
- Proposed neighbor edge pass ratio: 100%.
- Neighbor median Profit Factor: 1.7290.
- Neighbor minimum average trade return: 1.066%.
- Neighbor time pass ratio: 66.7%.

The R20 `-0.02` and `0.00` settings are outcome-redundant in this run. Before relying heavily on neighbor ratios, the implementation should decide whether identical effective filters count as separate neighbors or as one effective parameter state; otherwise redundancy can overstate neighborhood breadth.

## 4. Proposed UI Changes

- Show a **Trend-following profile** badge when `median_trade_return < 0`, `profit_factor > 1`, and `avg_trade_return > 0`.
- Keep Median Trade Ret prominent, but label it as a distribution diagnostic rather than a disqualifying gate.
- Add a positive-skew / big-winner-dependence warning explaining that average profitability depends on fewer large winners overcoming frequent small losses.
- Continue showing Time Stability and Parameter Stability as separate components.
- Replace the current neighbor mandatory-pass ratio with the proposed edge-neighborhood fields.
- Keep risk metrics visible, but clearly distinguish event-level tail loss and Trade-Sequence DD from portfolio drawdown.
- De-emphasize Trade-Sequence DD in candidate ranking. For thousands of overlapping independent trade events it approaches -100% and is not an investable portfolio drawdown measure.

## 5. What-If Gate Analysis

All source strategies have 10 eligible entry years, so an additional `eligible_years >= 5` check would not change these what-if counts.

| Model | Profit Factor | Positive-year ratio | Passing strategies | Share of grid |
|---|---:|---:|---:|---:|
| Model 1 | >= 1.1 | >= 0.50 | 235 | 43.5% |
| Model 2 | >= 1.2 | >= 0.60 | 58 | 10.7% |
| Model 3 | >= 1.3 | >= 0.60 | 58 | 10.7% |
| Model 4 | >= 1.5 | >= 0.60 | 56 | 10.4% |

Models 2 and 3 select exactly the same 58 strategies. Raising Profit Factor from 1.3 to 1.5 removes only two strategies, both using Signal 3D confirm. The principal discriminator is the 60% positive-year requirement, not the difference between Profit Factor 1.2 and 1.3.

### Top candidate under every model

All four models select the same top candidate under the stated ranking:

| Field | Value |
|---|---|
| Strategy | `score_bo_l10_rp002_erp005__first_signal__low20` |
| Label | Score BO L10 R20>=0.02 ER20>=0.05 / First signal / Low20 trailing |
| Completed trades | 9,117 |
| Avg Trade Ret | 1.235% |
| Median Trade Ret | -0.302% |
| Trade Win Rate | 47.64% |
| Profit Factor | 1.7291 |
| Positive-year ratio | 60% |
| Eligible years | 10 |
| Current Parameter Stability | Fail (`neighbor_pass_ratio = 0`) |
| Proposed neighbor edge pass ratio | 100% |
| Proposed neighbor time pass ratio | 66.7% |
| Current Time Stability | Pass |
| Current old-gate tier | Not qualified |

The complete Top 20 for each model, including every requested current parameter-stability field and the proposed recalculated neighbor fields, is recorded in `trend_following_gate_what_if_top20.csv`.

### Distributions by model

#### Exit rule

| Exit | Model 1 | Model 2 | Model 3 | Model 4 |
|---|---:|---:|---:|---:|
| Low20 - 0.5ATR trailing | 103 | 26 | 26 | 26 |
| Low20 trailing | 96 | 20 | 20 | 18 |
| MA50 trailing | 36 | 12 | 12 | 12 |
| Low10 trailing | 0 | 0 | 0 | 0 |
| Chandelier20 2.5ATR | 0 | 0 | 0 | 0 |

#### Entry rule

| Entry | Model 1 | Model 2 | Model 3 | Model 4 |
|---|---:|---:|---:|---:|
| 5D breakout after signal | 63 | 22 | 22 | 22 |
| First signal | 52 | 17 | 17 | 17 |
| Signal 2D confirm | 69 | 17 | 17 | 17 |
| Signal 3D confirm | 51 | 2 | 2 | 0 |

#### Score lookback

| Lookback | Model 1 | Model 2 | Model 3 | Model 4 |
|---:|---:|---:|---:|---:|
| 10 | 83 | 32 | 32 | 30 |
| 20 | 83 | 26 | 26 | 26 |
| 40 | 69 | 0 | 0 | 0 |

#### R20 minimum

| R20 minimum | Model 1 | Model 2 | Model 3 | Model 4 |
|---:|---:|---:|---:|---:|
| -0.02 | 68 | 4 | 4 | 4 |
| 0.00 | 68 | 4 | 4 | 4 |
| 0.02 | 99 | 50 | 50 | 48 |

#### ER20 minimum

| ER20 minimum | Model 1 | Model 2 | Model 3 | Model 4 |
|---:|---:|---:|---:|---:|
| 0.05 | 75 | 16 | 16 | 15 |
| 0.10 | 79 | 19 | 19 | 18 |
| 0.15 | 81 | 23 | 23 | 23 |

## 6. Check of Earlier Observations

### Low20 - 0.5ATR is strongest by median/time consistency: confirmed

- Least-negative median of strategy Median Trade Ret: -0.359%, versus -0.377% for Low20 and -0.486% for MA50.
- Highest mean positive-year ratio: 51.9%.
- Most current Time Stability passes: 26.
- Most Model 2 passes: 26.

### MA50 is strongest by Profit Factor: confirmed

- Highest median strategy Profit Factor: 1.5945.
- This comes with a more-negative median trade return and only 12 Time Stability / Model 2 passes, so it is an edge-strength result rather than the strongest overall consistency result.

### Signal 3D confirm is weakest: confirmed

- Lowest median average trade return: 0.657%.
- Lowest median Profit Factor: 1.3736.
- Most-negative median of strategy medians: -0.541%.
- Only 2 Model 2/3 passes and no Model 4 passes.

Signal 2D confirm has the highest median Profit Factor and median average return, while 5D breakout has the most Model 2 and Time Stability passes.

### L40 is weak: confirmed

- Zero Time Stability passes.
- Zero Model 2, 3, or 4 passes.
- Lower median Profit Factor and more-negative median trade return than L10 and L20.

### R20 -0.02 and R20 0.00 are redundant: confirmed for observed outcomes

All 180 matched parameter/entry/exit pairs are identical for completed trades, average return, median return, win rate, Profit Factor, positive-year ratio, eligible years, and Time Stability status. Neighbor metadata can differ because the two values occupy different grid positions.

### L20 / ER20 0.15 under loose R20 thresholds: partially confirmed

At R20 `-0.02` and `0.00`, the L20 / ER20 0.15 groups retain strong edge statistics:

- Median Profit Factor: 1.5172.
- Median average trade return: 0.793%.
- Median Median Trade Ret: -0.368%.

However, each 20-strategy loose-R20 group produces only one Model 2 pass and has a mean positive-year ratio of 36%. The stricter R20 `0.02` version produces 9 Model 2 passes, the highest signal-group count. The earlier observation therefore holds for edge strength, but not for time consistency under the proposed Time Gate.

## 7. Open Questions and Recommendation

### Profit Factor threshold: 1.1, 1.2, or 1.3

- 1.1 with a 50% year ratio is permissive: 235 strategies pass.
- 1.2 with a 60% year ratio produces a focused 58-strategy set.
- 1.3 produces the identical set, so current data cannot distinguish 1.2 from 1.3.
- Recommendation: start with **1.2**, retain 1.3 as a sensitivity view, and avoid implying that the indistinguishable current outcome proves 1.3 is intrinsically better.

### Positive-year ratio: 0.60 or 0.70

- The maximum observed ratio is 0.60; a 0.70 gate would qualify zero strategies.
- Recommendation: start with **0.60** and review the year-level definition before considering 0.70.

### Risk gate: descriptive or mandatory

- Recommendation: keep risk descriptive until thresholds are approved and portfolio exposure/overlap is modeled.
- Candidate risk review should still display Worst Trade Ret, 10th-percentile Trade Ret, stop-hit behavior, and relevant excursion metrics.

### Median threshold: remove or allow -0.25%

- Recommendation: remove Median Trade Ret from mandatory qualification entirely.
- A -0.25% threshold would still reject many profitable positive-skew candidates and would remain an arbitrary distribution-shape gate.

### Trade-Sequence DD

- Recommendation: de-emphasize it in qualification and ranking.
- It compounds thousands of independent, overlapping event trades and is explicitly not portfolio drawdown. A later portfolio backtest should supply investable drawdown and capital-at-risk measures.

## Recommended Starting Model

Adopt Model 2 as the implementation candidate:

```text
Sample: completed_trades >= 100
Edge: profit_factor >= 1.2 and avg_trade_return > 0
Time: eligible_years >= 5 and positive_year_ratio >= 0.60
Parameter: eligible_neighbors >= 2 and neighbor_edge_pass_ratio >= 0.60
Risk: descriptive / unverified
Median: visible diagnostic, not mandatory
```

Before implementation, confirm how outcome-identical parameter values should count in neighborhood breadth and approve the revised tier names. Production code and UI should change only in a separate implementation task with regenerated summaries and focused gate/tier tests.
