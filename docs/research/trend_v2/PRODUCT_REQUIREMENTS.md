# Trend Strategy v2 Product Requirements

## Product objective

The primary deliverable is a reusable web-based backtest and strategy-comparison tool.
The tool must allow the user to define and run strategy rules, preserve economic
backtest results, and evaluate the same stored results under different comparison
criteria without rerunning the strategy when the economic path is unchanged.

The project is not primarily a request for Codex to choose and run one preferred
strategy on the user's behalf.

## Required separation

The implementation must keep these concepts separate:

1. `StrategyRun`
   - data snapshot;
   - universe;
   - filter and signal definition;
   - entry, initial stop, trailing exit, and sizing rules;
   - execution and cost assumptions;
   - generated signals, trades, positions, and daily portfolio curve.
2. `EvaluationProfile`
   - selected metrics;
   - metric directions;
   - feasibility thresholds;
   - Pareto objectives and epsilon tolerances;
   - robustness requirements;
   - optional user-defined comparison weights;
   - tie-breaking and display preferences.
3. `EvaluationRun`
   - one immutable application of an evaluation profile to one or more stored
     strategy runs;
   - profile version and hash;
   - resulting gates, Pareto status, robustness status, ranking view, and
     decision labels.

Changing only an evaluation profile must not require rerunning an unchanged
strategy path.

## Web backtest controls

The web application must let the user configure and execute backtests rather
than requiring source-code edits.

At minimum, the user must be able to select or configure:

- backtest date range and frozen data snapshot;
- universe and benchmark;
- trend filter;
- signal family and its permitted parameters;
- entry rule;
- initial stop;
- trailing exit;
- position sizing and portfolio constraints;
- transaction cost and slippage assumptions;
- walk-forward and robustness settings.

The UI must prevent unrestricted parameter explosions by showing the resulting
candidate count and requiring explicit confirmation for large searches.
Fixed holding-period exits and fixed profit targets remain prohibited unless a
future explicit user decision changes the research charter.

## Strategy-comparison controls

The user must be able to decide what constitutes strategic superiority.
The UI must support two distinct modes.

### Default mode: non-compensatory selection

This is the recommended research mode and must be the default:

1. configurable feasibility gates;
2. configurable Pareto objectives and objective directions;
3. configurable epsilon tolerances;
4. robustness vetoes;
5. lexicographic tie-breaking;
6. behavior-path deduplication and simplicity preference.

A strong value in one metric must not automatically compensate for failure of a
mandatory risk or robustness condition.

### Optional mode: user-weighted comparison

The user may optionally create an exploratory weighted comparison view.
The UI must allow direct adjustment of metric weights, including zero weight,
and must display:

- the exact normalization method;
- the exact metric direction and transformation;
- the weight assigned to every metric;
- the normalized total weight;
- sensitivity of the ranking to weight changes;
- warnings when a candidate fails a mandatory gate despite a high weighted
  value.

A weighted comparison must never silently replace the default Pareto and
robustness decision. It must be labeled as a user-defined exploratory view, and
every weight profile must be versioned and hashed.

The discarded trend-energy signal score and score-breakout trigger must not be
reintroduced through this weighted evaluation feature. Signal construction and
strategy evaluation are separate concepts.

## Metrics available to the user

The configurable metric library should include, at minimum:

- net CAGR and CAGR relative to SPY;
- annualized volatility;
- conventional and autocorrelation-adjusted Sharpe ratio;
- Sortino ratio;
- maximum drawdown and relative maximum drawdown;
- CDaR 95 and relative CDaR 95;
- Calmar ratio and relative Calmar ratio;
- longest time under water and recovery duration;
- worst rolling 3-, 12-, and 36-month performance;
- expected shortfall and downside deviation;
- turnover, cost drag, exposure, cash ratio, and position concentration;
- walk-forward pass ratio and worst-fold result;
- leave-one-year-out stability and reversing years;
- paired block-bootstrap effect and confidence interval;
- multiple-testing-adjusted evidence;
- Deflated Sharpe Ratio and PBO when the required trial history exists;
- asset-group concentration and transaction-cost stress survival.

Trade-level metrics such as Profit Factor, win rate, and median trade return are
diagnostics and must not become mandatory primary objectives by default.

## Korean-first interface

The visible web interface should be Korean-first.

- Navigation, controls, table headers, validation messages, status labels, and
  report summaries should use Korean by default.
- The original English abbreviation may be shown in parentheses where useful.
- A language architecture that permits later English support is preferred, but
  Korean is the required default presentation.

## Metric and terminology guide

The web application must include a dedicated explanation tab for every acronym,
metric, validation method, strategy component, and decision label used by the
tool.

Each explanation must include:

1. Korean term and English name;
2. abbreviation;
3. exact formula with variable definitions;
4. a numerical worked example;
5. interpretation of high and low values;
6. units and annualization convention;
7. assumptions and limitations;
8. when the metric can be misleading;
9. how the metric is used in feasibility, Pareto, weighted, and robustness
   decisions;
10. links from every table header or UI label to the relevant explanation.

The guide must cover at least CAGR, volatility, Sharpe, adjusted Sharpe,
Sortino, MDD, CDaR, Calmar, recovery duration, turnover, exposure, walk-forward,
LOYO, block bootstrap, adjusted p-values, Deflated Sharpe Ratio, PBO, Pareto
dominance, epsilon dominance, and behavior deduplication.

## Result persistence and reuse

The design must support storing one economic strategy result and evaluating it
later without rerunning the backtest.

Persist compact, reusable artifacts such as:

- run manifest and hashes;
- daily portfolio curve;
- sparse signal and execution events;
- trade lifecycles;
- yearly and rolling metrics;
- robustness summaries;
- evaluation-profile history.

Avoid dense candidate-by-symbol-by-date matrices for every candidate. Use
columnar compression, sparse events, content hashes, deduplication, and tiered
retention. Large common snapshots and caches should not be duplicated per
strategy run.

The storage implementation must expose estimated artifact size and retention
status to the user before large runs are saved.

## Reproducibility and overfitting controls

Every strategy run and evaluation run must record:

- data snapshot hash;
- engine and code version;
- full strategy specification;
- full evaluation profile;
- all tested candidates, including rejected candidates;
- random seeds and bootstrap index hashes;
- created time and economic date range.

Changing thresholds, epsilon values, or weights after seeing results creates a
new evaluation profile and must remain visible in history. The UI must not allow
users to overwrite prior evaluation profiles silently.

## Delivery sequence

Build the product incrementally:

1. strategy-run, evaluation-profile, evaluation-run, and result-store contracts;
2. reusable metric and selection engine;
3. local web API and saved-run registry;
4. Korean-first comparison and configuration UI;
5. metric/terminology explanation tab;
6. signal, entry, stop, and exit builders;
7. scalable caching and retention controls;
8. later OOS and production workflows only after explicit approval.
