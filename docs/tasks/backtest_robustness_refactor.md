# Backtest Robustness Refactor — Implementation Spec

## Status

- Type: implementation task specification
- Scope: Backtest tab presentation, completed-trade aggregation, date filtering, robustness reporting, and supporting output contracts
- Strategy status: existing signal, entry, exit, stop, maximum-holding, and cost logic must remain unchanged
- Implementation status: Phase 1 implemented; arbitrary date controls, strict entry-and-exit mode, and benchmark regime analysis remain deferred

## Objective

Refactor the Backtest tab so its default view evaluates robustness from completed trades after the full strategy path has been applied: signal selection, entry rule, price-based exit/stop, maximum holding period, and round-trip cost. The page must clearly distinguish event-level trade analysis from a portfolio equity backtest and must favor transparent, inspectable robustness gates over a single arbitrary weighted score.

The refactor must not add signals, alter strategy rules, or turn the event backtest into a portfolio simulator.

## Research Intent

The current page gives prominent placement to fixed-window Signal Forward Return Diagnostics. Those diagnostics can help isolate signal behavior, but they do not include the selected exit rule, stop behavior, maximum holding cap, or realized trade cost. The redesigned default UI should therefore answer this question first:

> Which nearby signal parameters, entry rules, and exit rules produce consistently acceptable completed-trade outcomes across parameter neighborhoods, time periods, and market regimes?

Fixed-window signal diagnostics are removed from the rendered Backtest tab. Their generator may remain temporarily for compatibility, but they do not drive ranking, Candidate Snapshot, Robustness Tier, or robustness gates.

## In Scope

1. Reorganize the Backtest tab around exit-applied completed-trade results.
2. Remove fixed-window Signal Forward Diagnostics from the rendered Backtest tab by default; the underlying generator may remain temporarily for compatibility.
3. Add requested start/end date controls and report the effective included completed-trade period.
4. Rename event-level result labels to prevent portfolio-level interpretation.
5. Add a prominent event-level analysis warning.
6. Add compact page-reading guidance and detailed metric definitions.
7. Replace or demote the current weighted `robust_score` with transparent robustness components, gates, and tiers.
8. Add time-stability and regime-stability summaries based on completed trades.
9. Add a compact Candidate Snapshot.
10. Keep long result tables collapsed by default.
11. Update bounded summary outputs as needed without committing raw event-level CSV files.
12. Update `docs/RESEARCH_CONTEXT.md` after implementation and validation.

## Out of Scope

- New signal families or new Score Breakout parameters.
- Changes to Score Breakout, entry timing, stop calculation, trailing-stop behavior, maximum holding period, transaction cost, or trade simulation semantics.
- Signal FALSE exits.
- MaxHold-only or other stopless strategies.
- Capital allocation, position sizing, concurrent-position limits, cash accounting, portfolio equity curves, portfolio CAGR, or portfolio drawdown.
- Benchmark-relative alpha implementation unless separately specified later.
- Persisting or publishing complete trade/event-level datasets.

## Required Page Structure

The default Backtest tab should use the following order.

### 1. Event-Level Analysis Warning

Display a visually distinct warning near the top of the Backtest tab:

> This page analyzes completed strategy trades as independent events. It is not a portfolio equity backtest. Sum of Trade Returns and Trade-Sequence DD are not portfolio return, CAGR, or portfolio drawdown because overlapping trades, capital allocation, exposure, and cash constraints are not modeled.

The warning must remain visible without expanding a disclosure.

### 2. How to Read This Page

Add a compact section, visible by default, with no more than five short steps:

1. Set the requested trade period and confirm the effective included period.
2. Check that a candidate passes minimum sample and completed-trade requirements.
3. Compare Performance and Downside after the selected exit, stop, maximum holding period, and cost.
4. Prefer candidates that remain acceptable across neighboring parameters, time buckets, and regimes.
5. Use detailed tables for verification; reserve portfolio conclusions for a later portfolio backtest.

### 3. Date Controls and Included Period

Provide compact controls for:

- Requested Start Date
- Requested End Date
- Apply
- Reset to All Available

Display the following status values beside or directly below the controls:

- Requested entry period: user-entered start through end, inclusive.
- Included trade entry period: minimum and maximum `entry_date` among included completed trades.
- Realized exit period of included trades: minimum and maximum `exit_date` among those trades, including exits after the requested end date.
- Included completed trades: count after filtering.
- Available trade entry period: earliest through latest available `entry_date` in the bounded analysis source.
- A clear empty-state message when no completed trades match.

#### Date Filtering Semantics

For the first implementation, filter completed trades by `entry_date` inside the requested inclusive date range. Show each included trade's fully realized result using its actual `exit_date` and exit price even when the exit occurs after the requested end date. This answers the primary research question: "How did trades initiated during this period perform?"

Do not exclude a valid completed trade merely because it was opened near the end of the requested entry period and exited later. This is especially important with the existing maximum holding cap of 63 trading days. A stricter optional mode requiring both entry and exit inside the requested period may be considered later, but it is not part of the default implementation.

Rules:

- Blank start means no lower bound.
- Blank end means no upper bound.
- Start after end is invalid and must not apply.
- Invalid or unparsable dates must show an inline validation message.
- The controls must not silently clamp the requested dates; requested and effective periods must be shown separately.
- Filtering must be applied before all Candidate Snapshot, tier, gate, time-stability, regime-stability, and detail-table calculations.
- Open or incomplete trades must be excluded from completed-trade metrics.
- Included trades may have realized exits after the requested end date; the UI must make this visible through the separate Realized Exit Period status.
- Any later strict entry-and-exit mode must be an explicit user-selected mode and must not silently replace the entry-date default.

### 4. Candidate Snapshot

Add a compact, visible-by-default snapshot for the currently selected or top qualifying candidate. It must not present a candidate when mandatory gates fail without clearly showing that status.

Suggested fields:

- Robustness Tier and gate status
- Signal parameter tuple: Lookback, R20 Min, ER20 Min
- Entry rule
- Exit rule
- Completed trades
- Effective entry/exit period
- Avg Trade Ret
- Median Trade Ret
- Trade Win Rate
- Sum of Trade Returns
- Trade-Sequence DD
- Profit Factor
- Stop Hit Rate
- Average Holding Days
- Parameter Stability status
- Time Stability status
- Regime Stability status

Candidate selection must be explainable and deterministic. Apply this exact order:

1. Higher Robustness Tier.
2. More mandatory gates passed.
3. Higher Median Trade Ret.
4. Higher Profit Factor.
5. Lower absolute Trade-Sequence DD.
6. Larger completed-trade count.

Do not use the existing weighted `robust_score` as the primary selector or as an earlier tie-break than any item above.

### 5. Exit-Applied Robustness Overview

Show five labeled components. Each component must expose its inputs, gate outcomes, and missing-data state.

#### Performance

Use completed net trade returns after round-trip cost. At minimum report:

- Completed trades
- Avg Trade Ret
- Median Trade Ret
- Trade Win Rate
- Sum of Trade Returns
- Profit Factor

Performance gates should emphasize adequate sample size and a positive central tendency rather than one extreme aggregate result. Proposed configurable gates:

- Minimum completed trades
- Median Trade Ret greater than zero
- Avg Trade Ret greater than zero
- Profit Factor greater than one

Thresholds must be defined in one configuration/data-contract location rather than duplicated in Python and JavaScript.

#### Downside

At minimum report:

- Trade-Sequence DD
- Worst Trade Ret
- Lower-tail trade return statistic, preferably 10th percentile
- Stop Hit Rate
- Average MAE if retained in the completed-trade summary and calculated consistently

Trade-Sequence DD must be explicitly defined as drawdown on the ordered sequence of completed trade returns under the existing event-level compounding convention. It is not chronological portfolio drawdown and does not resolve overlapping trades.

Proposed configurable downside gates:

- Trade-Sequence DD no worse than a stated limit
- Lower-tail return no worse than a stated limit
- No missing downside metrics when sufficient trades exist

Stop Hit Rate is descriptive and should not automatically fail a candidate without a separately approved threshold; a high rate can reflect either effective protection or an overly tight stop.

#### Parameter Stability

Evaluate a candidate against its nearby Score Breakout parameter neighborhood while holding entry and exit rules constant.

Neighbor definition:

- Parameters are `score_lookback`, `r20_min`, and `er20_min`.
- A direct neighbor differs by one adjacent grid step in exactly one parameter and matches the other two parameters.
- Edge values naturally have fewer neighbors.
- Entry rule and exit rule must be identical when comparing neighbors.

At minimum report:

- Available direct neighbors
- Neighbors meeting the minimum completed-trade requirement
- Neighbors passing mandatory Performance and Downside gates
- Neighbor pass ratio
- Median and range of neighbor Median Trade Ret
- Median and range of neighbor Trade-Sequence DD

Parameter Stability should fail or be marked insufficient when too few eligible direct neighbors exist. The minimum eligible-neighbor count and pass ratio must be explicit configurable gates.

### Time Stability

Split filtered completed trades into calendar-based subperiods using `entry_date` for cohort attribution, while measuring each cohort with fully realized completed-trade results. Prefer calendar years when each year has enough trades; otherwise use deterministic contiguous entry-date buckets such as halves or thirds of the requested entry period.

At minimum report:

- Bucket label and date range
- Completed trades per bucket
- Median Trade Ret per bucket
- Avg Trade Ret per bucket
- Trade Win Rate per bucket
- Profit Factor per bucket
- Trade-Sequence DD per bucket
- Count/ratio of eligible buckets passing the time-stability gates

Buckets below the minimum trade count must be labeled insufficient rather than passed or failed. Time Stability must not pass when only one eligible bucket exists. Gate thresholds and fallback bucketing rules must be centralized and documented.

### Regime Stability

Evaluate completed trades across a small, transparent set of market regimes. Regime labels must be assigned using information available at the entry date to avoid look-ahead bias.

Preferred initial regime definitions, evaluated using SPY values known on each trade's `entry_date`:

- **SPY MA200 regime:** SPY close above SPY MA200 versus SPY close at or below SPY MA200.
- **SPY 63-day return regime:** SPY 63-trading-day return positive versus zero or negative.

Both labels must use only benchmark information available as of the trade entry date. Do not use future returns, full-period outcomes, ex-post drawdown labels, or any other hindsight-based regime assignment.

If SPY or equivalent benchmark history is not already available in the current static pipeline, display Regime Stability as `Not available`. Do not add a new benchmark download or static-data dependency as part of this refactor.

At minimum report when available:

- Regime definition and benchmark
- Completed trades by regime
- Median Trade Ret
- Avg Trade Ret
- Trade Win Rate
- Profit Factor
- Trade-Sequence DD
- Eligible-regime pass ratio

Regimes below the minimum trade count must be marked insufficient. Regime Stability must not pass based on a single eligible regime.

## Centralized Gate Configuration

All robustness thresholds must come from one versioned gate configuration included in the generated data contract and exposed in the UI's Metric Definitions or gate details. Python aggregation and JavaScript rendering must not maintain separate threshold copies. JavaScript should render the gate values supplied by the payload rather than re-declaring them.

Use the following initial configurable defaults as starting decision thresholds:

| Configuration key | Initial default | Meaning |
|---|---:|---|
| `min_completed_trades` | `100` | Minimum completed trades required for a candidate to be evaluated. |
| `min_bucket_trades` | `10` | Minimum completed trades for a time or regime bucket to be eligible. |
| `min_eligible_neighbors` | `2` | Minimum direct parameter neighbors with adequate completed trades. |
| `min_profit_factor` | `1.0` | Minimum completed-trade profit factor. |
| `min_median_trade_return` | `0.0` | Minimum Median Trade Ret. |
| `min_neighbor_pass_ratio` | `0.60` | Minimum share of eligible direct neighbors passing mandatory Performance and Downside gates. |
| `min_positive_year_ratio` | `0.60` | Minimum share of eligible entry-year buckets with positive Median Trade Ret. |
| `min_positive_regime_ratio` | `0.50` | Minimum share of eligible regimes with positive Median Trade Ret. |
| `max_trade_sequence_dd` | `0.25` | Maximum allowed absolute Trade-Sequence DD magnitude; equivalent to no worse than `-25%`. |
| `min_tail_return` | `-0.10` | Minimum allowed 10th-percentile completed Trade Ret. |

These values are explicit, revisable research decisions, not statistically proven constants. They must be easy to change in one place, must be serialized with the backtest output, and must be displayed with the result so historical output can be interpreted using the thresholds that produced it. Revising a threshold requires regenerating the affected tier/gate summaries; the UI must not silently apply new thresholds to old aggregates.

The configuration should include a schema/version identifier and may later be moved to a dedicated config file. The first implementation may define it as one Python constant/dictionary if that is the smallest safe change, provided the generated payload is the only threshold source used by the UI.

## Robustness Tier and Gate Model

Do not rank robustness primarily by an arbitrary weighted composite score. Replace or clearly demote `robust_score` in the default UI.

Use transparent mandatory gates and tiers. Exact numerical thresholds should be centralized and approved before implementation. The initial structure should be:

- **Tier A — Broadly robust:** mandatory sample, Performance, and Downside gates pass; Parameter Stability passes; Time Stability passes; Regime Stability passes when available. If regime data is unavailable, label the result `Tier A (regime unverified)` rather than silently treating it as passed.
- **Tier B — Promising:** mandatory sample, Performance, and Downside gates pass; Parameter Stability passes; Time or Regime Stability is insufficient or mixed.
- **Tier C — Fragile:** the candidate has acceptable headline completed-trade results but fails Parameter Stability, Time Stability, or Regime Stability.
- **Not qualified:** mandatory sample, Performance, or Downside gates fail.
- **Insufficient data:** the minimum completed-trade requirement is not met.

Every tier display must include a compact gate matrix with `Pass`, `Fail`, `Insufficient`, or `Not available`. Users must be able to see why a candidate received its tier.

Avoid adding an overall numeric score unless a later research decision defines and justifies it. Individual metrics may still be used for deterministic tie-breaking after tier and gate status.

## Metric Label Changes

Change default Backtest UI labels as follows. Internal field names may remain temporarily for compatibility, but new output contracts should prefer explicit event-level names.

| Current field | Required UI label | Preferred explicit output alias |
|---|---|---|
| `total_return` | Sum of Trade Returns | `sum_trade_returns` |
| `max_drawdown` | Trade-Sequence DD | `trade_sequence_drawdown` |
| `avg_return` | Avg Trade Ret | `avg_trade_return` |
| `median_return` | Median Trade Ret | `median_trade_return` |
| `win_rate` | Trade Win Rate | `trade_win_rate` |

`Sum of Trade Returns` should be calculated as the arithmetic sum of completed net trade returns. If the existing `total_return` is compounded across event rows, do not merely relabel it: add the arithmetic sum explicitly and either remove the compounded value from the default UI or label it separately as an event-sequence diagnostic. This distinction must be verified during implementation.

All return, rate, excursion, and drawdown values must display as percentages with one decimal place unless extra precision is required in a detailed export.

## Metric Definitions Section

Add a visible heading with definitions inside a collapsed-by-default disclosure. Definitions must be plain-language, concise, and match the actual code.

### Signal Inputs

- **R20:** close-to-close return over the prior 20 trading days.
- **ER20:** 20-trading-day efficiency ratio, measuring net directional movement relative to the sum of absolute daily movements; higher values indicate a cleaner directional path.
- **TE63:** 63-day return multiplied by the corresponding 63-day efficiency ratio.
- **TE126:** 126-day return multiplied by the corresponding 126-day efficiency ratio.
- **Score:** `0.65 * TE63 + 0.35 * TE126`.
- **Score Breakout:** true when the current Score exceeds its highest value over the prior configured lookback, Score is positive, R20 and ER20 meet their minimum thresholds, and Close is above MA50.

Before publishing these definitions, verify the exact return and efficiency-ratio formulas in `src/features.py`; the UI wording must follow code if it differs from this shorthand.

### Entry Rules

- **First signal:** next-open entry after Score Breakout changes from false to true.
- **Signal 2D confirm:** next-open entry after two consecutive true signal days.
- **Signal 3D confirm:** next-open entry after three consecutive true signal days.
- **5D breakout after signal:** after the first signal, require the configured five-day price-breakout confirmation within five trading days, then enter at the next open.

### Exit Rules

Explain that every variant uses a valid initial price stop, updates the stop according to its rule without moving it downward, and retains the maximum holding cap. Define:

- Low10 trailing
- Low20 trailing
- Low20 minus 0.5 ATR20 trailing
- Chandelier20 at 2.5 ATR
- MA50 trailing
- Stop execution behavior for gaps, based on the actual simulator
- Maximum holding exit
- Round-trip cost and when it is deducted

### Result Metrics

Define at minimum:

- Completed Trades
- Avg Trade Ret
- Median Trade Ret
- Trade Win Rate
- Sum of Trade Returns
- Trade-Sequence DD
- Worst Trade Ret
- 10th Percentile Trade Ret
- Profit Factor
- Stop Hit Rate
- Max Hold Exit Rate
- Average Holding Days
- MAE and MFE if displayed
- Parameter Stability
- Time Stability
- Regime Stability
- Robustness Tier

Every definition must state whether it is computed per completed trade, across an ordered event sequence, across parameter neighbors, or across time/regime buckets.

## Fixed-Window Signal Diagnostics

Remove the `Signal Forward Return Diagnostics` table from the rendered Backtest tab in the default implementation. Do not place Fwd20 diagnostics in a default or collapsed Backtest UI section as part of this refactor.

The underlying diagnostic generator and compatibility fields may remain temporarily if removing them would unnecessarily expand implementation risk. Their continued generation must not imply that they are part of the redesigned page.

Do not use fixed-window Avg Fwd20, Median Fwd20, Win20, MFE20, MAE20, or the existing `robust_score` in Candidate Snapshot selection or exit-applied robustness gates.

## Long Tables and Progressive Disclosure

All long tables must be collapsed by default using accessible disclosure controls, preferably native `<details>`/`<summary>` or equivalent controls with keyboard support and `aria-expanded` state.

Recommended collapsed sections:

- All Candidate Results
- Parameter Neighborhood Details
- Time-Bucket Details
- Regime Details
- Strategy Summary
- Recent Completed Trades
- Signal-only Diagnostics, if retained
- Metric Definitions

The default visible page should remain compact: warning, reading guide, date controls, Candidate Snapshot, robustness tier/gates, and five component summaries.

## Data and Output Contract

The page must be able to recalculate all displayed period-filtered results from an appropriately bounded data source. The implementation must choose and document one of these approaches:

### Preferred: Pre-Aggregated Bounded Analysis Cubes

Generate compact completed-trade aggregates by strategy, approved time bucket, and approved regime bucket. Publish only the aggregates required by the UI. This best satisfies repository-size constraints but permits only supported date/bucket selections.

### Alternative: Bounded Completed-Trade Sample for Client Filtering

Publish a deliberately bounded completed-trade dataset only if its maximum row count and file size are explicitly enforced and it is sufficient for the requested date controls. This must not become an unbounded raw event-level CSV.

### Server/Workflow Recalculation

If arbitrary start/end dates cannot be supported safely in static GitHub Pages without raw trades, date controls may be limited to precomputed period boundaries or a separately triggered workflow. The UI must not imply arbitrary filtering if only preset periods are supported.

Regardless of approach:

- Do not commit `backtest_trades.csv`, `signal_diagnostics.csv`, `backtest_skipped.csv`, or equivalent unbounded raw files.
- Keep an explicit workflow allowlist for committed artifacts.
- Retain and extend preflight file-size checks.
- Record schema/version metadata in the primary backtest JSON payload.
- Preserve backward compatibility for one release where practical, but remove ambiguous labels from the default UI immediately.
- Document whether date filtering is exact or limited to precomputed buckets.

## Anticipated Implementation Areas

The implementation will likely touch:

- `src/backtest.py`: completed-trade metrics, arithmetic Sum of Trade Returns, gate inputs, parameter neighbors, time buckets, and optional regime aggregation.
- `src/run_backtest_only.py`: no strategy changes; only output-generation integration if required.
- `docs/index.html`: page structure, warnings, controls, disclosures, and definitions.
- `docs/app.js`: labels, date-control state, completed-trade table rendering, and period metadata.
- `docs/backtest_dashboard.js`: Candidate Snapshot, gate/tier rendering, parameter/time/regime stability, and removal of the weighted score from the default path.
- `docs/style.css`: compact warning, snapshot, gate matrix, disclosure, and responsive styles.
- `.github/workflows/backtest-only.yml`: bounded output allowlist and size validation if new summary files are introduced.
- `docs/RESEARCH_CONTEXT.md`: implemented state, interpretation priority, date semantics, tier model, and remaining portfolio-validation limitations.

This list is guidance, not permission to change strategy logic.

## Implementation Sequence

1. Verify current formulas and simulator behavior in `src/features.py` and `src/backtest.py`.
2. Decide the static-site date-filtering data contract and confirm whether arbitrary dates are feasible without raw data.
3. Add unambiguous completed-trade metrics and backward-compatible aliases where required.
4. Implement period filtering and requested/effective period metadata.
5. Implement Performance and Downside aggregates.
6. Implement direct-neighbor Parameter Stability.
7. Implement deterministic Time Stability buckets.
8. Implement Regime Stability only after the benchmark/regime source is approved and available; otherwise expose `Not available`.
9. Implement transparent gates and Robustness Tiers.
10. Refactor the default Backtest UI and collapse long tables.
11. Remove fixed-window diagnostics from default ranking and snapshot logic.
12. Verify repository output sizes and workflow allowlists.
13. Run syntax, data-contract, metric, filtering, tier, and UI checks.
14. Update `docs/RESEARCH_CONTEXT.md` with the final implemented decisions.

## Acceptance Criteria

### Behavior

- The default Backtest tab is driven by completed trades after entry, exit, stop, maximum holding, and cost.
- Fixed-window Signal Forward Diagnostics are not rendered on the Backtest tab and have no effect on exit-applied robustness results.
- Requested start/end dates are validated and displayed.
- Requested Entry Period, Included Trade Entry Period, and Realized Exit Period of Included Trades are separately visible.
- Completed trades are included when `entry_date` is inside the requested inclusive entry period, even when the realized `exit_date` is later than the requested end date.
- Candidate Snapshot and all robustness components respond consistently to the applied period.
- Empty and insufficient-data states do not produce misleading tiers.

### Terminology

- The five required metric labels are updated everywhere in the default UI.
- Sum of Trade Returns is an arithmetic sum, not a relabeled compound event sequence.
- The event-level warning is visible without expansion.
- No UI text describes event metrics as portfolio return, CAGR, equity drawdown, or investable performance.

### Robustness

- Performance, Downside, Parameter Stability, Time Stability, and Regime Stability are separately visible.
- Each component exposes its inputs and `Pass`, `Fail`, `Insufficient`, or `Not available` status.
- Robustness Tier follows documented gates.
- The default candidate selection does not depend on an arbitrary weighted composite score.
- Candidate tie-breaking follows the exact tier, mandatory-gate count, Median Trade Ret, Profit Factor, absolute Trade-Sequence DD, and completed-trade-count order.
- All gate thresholds are supplied by one versioned configuration and displayed with the result.
- Parameter neighbors hold entry and exit rules constant and differ by one adjacent parameter grid step.
- Time and regime buckets enforce minimum completed-trade counts.

### UI

- Warning, reading guide, period controls, Candidate Snapshot, tier/gates, and component summaries are compact and visible by default.
- Long tables and definitions are collapsed by default and keyboard accessible.
- Percentage metrics retain one-decimal percentage formatting.
- The layout remains usable on narrow screens.

### Safety and Storage

- No signal, entry, exit, stop, maximum-holding, or transaction-cost logic changes.
- Signal FALSE is never used as an exit.
- No MaxHold-only/no-stop strategy is added.
- No raw event-level CSV is added or committed.
- No portfolio CAGR or portfolio maximum-drawdown claim is introduced.
- Workflow output paths remain explicitly allowlisted and size checked.
- `docs/RESEARCH_CONTEXT.md` is updated after implementation, not during this spec-only step.

## Test Plan

At minimum, implementation should include or run:

- Python syntax checks.
- JavaScript syntax checks, including completeness checks for `docs/backtest_dashboard.js`.
- HTML parse/render smoke check.
- Metric unit checks confirming arithmetic Sum of Trade Returns and Trade-Sequence DD semantics.
- Date-boundary checks for inclusive start/end, blank bounds, invalid order, and no matching trades.
- Checks that incomplete trades are excluded, entry dates are filtered inclusively, and valid included trades retain realized exits after the requested end date.
- Check that any future strict entry-and-exit mode cannot silently become the default.
- Neighbor-construction checks at middle and edge grid points.
- Tier/gate checks covering pass, fail, insufficient, and regime-unavailable cases.
- Time-bucket and regime-bucket minimum-sample checks.
- Formatting check that `0.023` renders as `2.3%` for all renamed return/rate fields.
- UI check that long tables are collapsed on initial load.
- Output-size and prohibited-raw-file checks.
- A full backtest only if runtime and network access are reasonable; otherwise document that it remains to be run in GitHub Actions.

## Decisions to Confirm Before Implementation

1. **Static-site filtering:** decide whether arbitrary requested entry dates are required or whether bounded presets/precomputed entry-period buckets are acceptable under the no-raw-data constraint.
2. **Gate defaults:** approve or revise the centralized starting values before treating the resulting tiers as research decisions.
3. **Regime availability:** verify whether SPY history already exists in the static pipeline. If it does not, ship Regime Stability as `Not available` without adding the dependency.
4. **Legacy compounded metric:** decide whether to keep the current compounded event-sequence result under an explicit diagnostic label or remove it from the UI entirely.

## Required Research Context Update After Implementation

After code, data-contract, workflow, and UI validation are complete, update `docs/RESEARCH_CONTEXT.md` to record:

- The Backtest tab's completed-trade-first interpretation order.
- Final date-filtering semantics and any static-site limitations.
- Final event-level metric names and definitions.
- Final robustness gates, thresholds, tiers, and insufficient-data rules.
- Final parameter-neighbor, time-bucket, and regime definitions.
- The status of fixed-window Signal Forward Diagnostics.
- The continuing warning that this is not a portfolio equity backtest.
- Any deferred benchmark-alpha or portfolio-validation work.
