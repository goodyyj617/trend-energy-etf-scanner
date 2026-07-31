# Trend Strategy v2 Phase A2 Result

## Empirical classification: Inconclusive

Complete evidence was supplied, but the preregistered portfolio, robustness, and control-comparability conditions do not jointly support Retain or Reject.
Inconclusive means score breakout remains exploratory and unresolved: the evidence does not justify either retaining it as a validated incremental rule or removing it as rejected. The transition rule therefore proceeds to Phase B1 with one representative score lookback and the best simple deterministic comparator.
This is provisional in-sample empirical research, not production approval and not genuine OOS evidence.

## Frozen input

- Economic dates: 2016-08-01 through 2026-07-30
- Snapshot SHA-256: `b37025ba30f5ecc2d32c53797201b8fcf43c06f539e6414ed4d4cdd9f70b82bd`
- Collector source commit: `bbf7c5669238c6543954b0613f163508c3caa27c`
- Empirical analysis source commit: `1b441a6d5beb57cfac51d36354168932bc2a41df`
- Retained/requested symbols: 470/470
- Failed or empty symbols: none

## Portfolio comparison

| Signal | CAGR | CAGR / SPY | MDD | abs(MDD) / SPY | CDaR95 | abs(CDaR95) / SPY | Calmar | Calmar / SPY | Recovery days | Turnover | Costs | Raw signals | Executable triggers | Completed trades |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| score_breakout_l10 | 3.11% | 0.209 | -28.10% | 0.833 | -25.34% | 1.233 | 0.111 | 0.251 | 1724 | 28.988 | 329.68 | 88258 | 47799 | 10361 |
| score_breakout_l20 | 4.40% | 0.295 | -24.31% | 0.721 | -22.46% | 1.093 | 0.181 | 0.409 | 1540 | 27.268 | 341.50 | 61694 | 33365 | 8154 |
| score_breakout_l40 | 2.90% | 0.194 | -26.65% | 0.790 | -25.54% | 1.243 | 0.109 | 0.246 | 1724 | 26.598 | 306.62 | 42548 | 23315 | 6094 |
| trend_filter_only | 2.20% | 0.148 | -32.17% | 0.954 | -29.88% | 1.454 | 0.068 | 0.155 | 1633 | 32.689 | 330.10 | 535292 | 8794 | 5879 |
| prior_price_high_l20 | 4.07% | 0.273 | -20.02% | 0.594 | -18.88% | 0.919 | 0.203 | 0.460 | 1402 | 26.203 | 310.00 | 77711 | 45644 | 8497 |
| signal_surge_v0 | 5.41% | 0.363 | -25.82% | 0.766 | -21.09% | 1.026 | 0.210 | 0.474 | 1073 | 30.708 | 382.03 | 45159 | 11030 | 4541 |

The representative score_breakout_l20 produced a 4.40% CAGR versus 2.20% for trend-only and 4.07% for the prior-price-high comparator. It improved return over both, but its drawdown, CDaR95, Calmar, and recovery were worse than prior-price-high, and its CAGR retained only 0.295 of SPY rather than the provisional 0.80 objective.
The legacy signal is shown only as a historical diagnostic baseline.

## Robustness evidence

| Score | Stronger comparator | Walk-forward improved | LOYO stable | Reversing years | Annualized paired effect | 95% CI | Raw one-sided p | Holm-adjusted p | Pass | Dominant group/share | Missing/unclassified share |
|---|---|---:|---:|---|---:|---|---:|---:|---|---|---:|
| score_breakout_l10 | prior_price_high_l20 | 2/6 (0.333) | 9/9 (1.000) | [] | -0.92% | [-2.51%, 0.53%] | 0.890622 | 1.000000 | false | Commodity / 0.471 | 0.167 |
| score_breakout_l20 | prior_price_high_l20 | 3/6 (0.500) | 7/9 (0.778) | [2020, 2023] | 0.31% | [-0.89%, 1.53%] | 0.305739 | 0.917217 | false | Industry / Theme / 0.359 | 0.168 |
| score_breakout_l40 | prior_price_high_l20 | 3/6 (0.500) | 9/9 (1.000) | [] | -1.13% | [-3.01%, 0.79%] | 0.869226 | 1.000000 | false | Bond / 0.930 | 0.171 |

Bootstrap uses 5,000 stationary-bootstrap paths with mean block length 20. The representative seed is 260732 and index hash is `ae4ac14f3d3d06d77db66b1c123c56b8a65c582f45923b7871eefeadfaa4c991`; every candidate's seed and index hash are recorded in `robustness_evidence.json`.
Holm correction covers all 3 tested score lookbacks at alpha 0.05; no unadjusted result is used as final evidence.
Asset-group concentration is `non_additive_group_restricted_portfolio_annualized_mean_daily_return_effect`. Group-restricted effects are a non-additive concentration diagnostic, not additive contributions.
The circular eligible-session-index placebo uses requested offset 63; per-symbol executable-trigger counts are preserved=true.
Every walk-forward fold and LOYO omission is reported separately in `robustness_evidence.json`.

## Phase B1 transition

Exact signal set: score_breakout_l20, prior_price_high_l20.
Score breakout status: exploratory_and_unresolved.

## Limitations

The historical panel uses the current frozen AUM ranking and present-day product availability. It therefore has survivorship and current-universe bias. The result is suitable only for provisional Phase A2 comparison and does not establish production readiness or genuine out-of-sample performance.
