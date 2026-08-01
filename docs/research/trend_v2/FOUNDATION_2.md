# Trend Strategy v2 Foundation 2

## Version boundaries

- metric calculation engine: `trend_v2_stored_curve_metric_engine_v1`;
- metric definitions: `legacy_portfolio_metric_parity_v1`;
- behavior engine: `trend_v2_behavior_fingerprint_v1`;
- derived metric manifest: `derived_metric_manifest_v1`;
- metric registry: `metric_registry_v2`.

Changing a formula version, calculation setting, source artifact hash,
benchmark identity, or benchmark artifact hash creates a new derived-metric
identity. Changing only an `EvaluationProfile` does not.

## Stored schemas

`daily_portfolio_curve_v1` stores sorted, unique economic dates, positive net
portfolio value, daily net return, gross and net exposure, cash weight, daily
turnover, transaction cost, and optional position count and gross path values.

`yearly_metrics_v1` stores calendar-year return, sample volatility annualized
with 252 sessions, within-year maximum drawdown, turnover, observations, and
the exact start/end economic dates. Only years strictly between the first and
last stored calendar years are marked complete. This deliberately fails safe
when no exchange calendar is available to prove boundary completeness.

`rolling_metrics_v1` supports full-window 63, 252, and 756-session values.
Each row is indexed by its trailing economic date and contains cumulative and
annualized return, annualized volatility, conventional Sharpe, and maximum
drawdown. A row is emitted only after the complete trailing window exists, so
no forward value is used.

`robustness_summary_v1` preserves walk-forward, LOYO, paired block-bootstrap,
multiple-testing, transaction-cost stress, asset-group concentration, seeds,
sample counts, method names, and source hashes. Required evidence fields may
be null only with a precise unavailable reason. Missing artifacts never create
fabricated values.

`behavior_metadata_v1` stores separate SHA-256 fingerprints for daily returns,
active exposure, entry dates, exit dates, and optional symbol lifecycles, plus
the compact pairwise comparison inputs.

All schemas reject missing required fields, duplicate or unsorted dates,
non-finite numbers, invalid binary values, impossible non-positive portfolio
values, inconsistent date ranges, and manifest row-count mismatches.

## Metric definitions

The parity metrics intentionally call the existing reliable portfolio-summary
semantics. CAGR uses elapsed calendar days divided by 365.25; volatility and
conventional Sharpe use daily sample standard deviation and 252 sessions;
Sortino uses the root mean square of `min(daily_return, 0)`; MDD and recovery
duration use the stored portfolio-value path; CDaR95 uses the existing fixed
negative-drawdown tail count; Calmar is CAGR divided by absolute MDD; annual
turnover is summed daily turnover divided by elapsed calendar years.

Additional metrics include current unfinished time under water, worst full
63/252/756-session returns, average exposure and cash weight, optional position
counts, and gross-CAGR minus net-CAGR transaction-cost drag. A missing gross
path is reported as unavailable and is not replaced with a different formula.

Benchmark-relative metrics are calculated only after intersecting exact
economic dates. Provenance records both input ranges, the retained range,
dropped dates, and common observation count. Fewer than the configured minimum
common observations fails every benchmark-relative metric closed.

## Selection and behavior

The stored-artifact operation keeps mandatory gates, epsilon-Pareto membership,
robustness vetoes, lexicographic ordering, and the optional weighted view
separate. Missing metrics retain their artifact or calculation reason. The
weighted value cannot change gates, Pareto membership, veto status, or final
constraint-Pareto labels.

Behavior diagnostics remain separate: daily-return correlation, active-date
Jaccard, entry-date Jaccard, exit-date Jaccard, and normalized path distance.
Configured conditions must all pass before two candidates are linked into a
cluster. Every original `StrategyRun` remains stored; the representative is
selected deterministically from configured simplicity metadata.

## Known limitations

- Metric parity preserves the existing zero-risk behavior: Sharpe, Sortino,
  and Calmar are unavailable when their denominators are zero.
- First and last calendar years are conservatively incomplete without a
  point-in-time exchange calendar.
- Robustness evidence is ingested and validated but expensive simulations are
  not rerun in this foundation.
- The historical universe still has present-day AUM/product-availability and
  survivorship limitations recorded in `CURRENT_STATE.md`.
