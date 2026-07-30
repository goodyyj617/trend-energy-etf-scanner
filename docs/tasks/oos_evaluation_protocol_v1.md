# OOS evaluation maturity protocol v1

## Decision

This document approves `oos-eval-v1.0.0` for cohort `oos-0001` before any collector, activation event, eligible ex-ante decision, or OOS outcome exists. Fixing the rules before collection prevents observed outcomes from influencing the amount of evidence, the cutoff date, or the reporting set.

PR #18 merged as `f60f46e9c7bb4006ea8be22e76b5230b71dde1d5` at `2026-07-30T15:47:40Z`. That merge made the preregistration contract authoritative. It did not activate OOS collection, is not the collector activation commit, and did not create an eligible OOS decision. OOS has still not started.

The known PR #18 merge fact is recorded identically in both the manifest and this protocol. Recording it resolves only the contract-merge provenance blocker. The protocol status is `approved_pre_activation` and the cohort manifest remains `proposed`; it does not activate the collector. OOS remains inactive until a separately reviewed collector is activated, and the first eligible ex-ante decision remains unresolved.

## Conjunctive maturity conditions

The first official OOS evaluation snapshot is allowed only when all four conditions are true at the same calendar month-end cutoff:

| Condition | Locked threshold |
| --- | ---: |
| Completed trades | At least 100 |
| Eligible sessions | At least 252 |
| Elapsed time | At least 12 calendar months |
| Valid session coverage | At least 0.98 |

`all_conditions_required` is `true`. Maturity is labeled only `Immature` or `Mature for review`.

One hundred trades alone are insufficient because many correlated trades can occur in a short market regime. A very high trade count cannot substitute for calendar exposure or session coverage.

Two hundred fifty-two eligible sessions require approximately one trading year of immutable decisions and portfolio state. Twelve elapsed calendar months is separately required so rapid session/trade accumulation cannot avoid real time across the calendar. Neither requirement substitutes for the other.

Coverage protects the interpretation of the observed path. At least 98% of expected market sessions from the first eligible decision through the cutoff must have valid eligible-session records. Perfect coverage is not required because an operational outage must not make the cohort permanently unevaluable, but every missing session and the longest missing gap remain visible.

Unresolved duplicate, correction, identity, or record conflicts prevent maturity regardless of the numeric thresholds.

## What counts

A completed trade:

- belongs to `oos-0001`;
- originates from an eligible post-activation ex-ante decision;
- has a valid append-only outcome record;
- appears once in the effective correction view; and
- realizes an exit under the frozen strategy rules.

Skipped entries, pre-activation positions, cancelled intentions, duplicates, open positions, reconstructed trades, and unresolved records do not count.

An eligible session is one distinct post-activation market session with an immutable decision snapshot, realized-universe hash, dynamic-input hashes, valid portfolio state or mark, and no unresolved conflict. Symbols, signals, and trades on the same economic date cannot increase the count above one.

The elapsed-time clock begins on the economic date of the first eligible post-activation ex-ante decision. The cutoff must be at least 12 calendar months after that date.

Before activation, the collector implementation must freeze the market-calendar source and version used for the expected-session denominator.

## Deterministic first evaluation

The trigger is `first_month_end_after_all_maturity_conditions_are_met`.

Maturity is evaluated only at calendar month-end. The economic cutoff is the last completed market session on or before that month-end. If the final threshold is reached during a month, evaluation waits until that month-end.

This rule removes discretionary cutoff choice. The first mature snapshot is immutable and cannot be replaced by a later, more favorable snapshot. Later monitoring snapshots must identify themselves as follow-ups and cannot overwrite the primary snapshot.

## Maturity is not an approval gate

`Mature for review` means only that sufficient evidence is available for the locked review. It is not a profitability result, qualification, production approval, deployment decision, or rejection decision. The reporting metrics are descriptive definitions rather than automatic gates.

Before maturity, monthly snapshots may be produced only after collector activation and must be labeled:

`Immature — descriptive monitoring only`

Interim results cannot change the candidate, protocol, thresholds, strategy semantics, or cutoff. They cannot be presented as OOS validation or trigger live-use approval.

## Open positions

At an evaluation cutoff, open positions remain open. The protocol creates no synthetic or forced exit.

Open positions:

- do not count as completed trades;
- are excluded from completed-trade diagnostics;
- remain in canonical portfolio mark-to-market metrics; and
- report count, gross and net exposure, unrealized P&L, oldest age, and capital allocated.

A later realized exit appends a later outcome record. It never rewrites the earlier snapshot.

## Missing sessions and corrections

A missing collector session is not eligible and is never reconstructed as an ex-ante decision. It remains in the expected-session denominator and is listed in data-quality diagnostics. Prices, signals, decisions, and portfolio marks are not interpolated.

A later outage record may describe the gap, but it cannot pretend that a contemporaneous decision was captured. Vendor corrections follow the PR #18 append-only correction contract: append a correction record and retain the original observation.

## Locked reporting set

The mature snapshot reports the following fixed definitions.

### Data sufficiency and integrity

- completed-trade count;
- eligible-session count;
- elapsed calendar months;
- expected-session count;
- valid-session coverage ratio;
- missing-session count;
- longest missing-session gap;
- correction count;
- unresolved-conflict count; and
- open-position count.

### Canonical portfolio metrics

- initial and ending equity;
- net total return;
- CAGR only with at least 252 eligible sessions;
- annualized volatility;
- Sharpe ratio under the existing repository convention;
- maximum drawdown;
- the existing CDaR 95 definition;
- Calmar ratio when mathematically defined;
- turnover and total transaction costs;
- average and maximum gross exposure;
- SPY return on exact common economic dates; and
- net excess return versus SPY.

### Completed-trade diagnostics

- completed trades;
- trade win rate;
- mean and median net trade return;
- Profit Factor;
- worst trade and 10th-percentile trade return;
- average holding period;
- stop-exit rate; and
- maximum-hold exit rate.

Metrics are not added after outcomes are observed. A later exploratory or sensitivity analysis requires a separately versioned secondary analysis protocol.

## Change control

Before activation, any change requires a reviewed PR, a new protocol version, and a documented rationale.

After activation, the `oos-eval-v1.0.0` thresholds and primary trigger are immutable for `oos-0001`. Changes cannot retroactively alter the primary evaluation. Exploratory work requires a new protocol version, and a material strategy or economic-definition change requires a new cohort. Later Backtest ranks cannot modify this cohort or protocol.

## Scope

This PR adds design and configuration only. It does not add a collector, workflow, activation event, ledger, decision, outcome, portfolio mark, snapshot, production code, UI, market-data operation, or behavioral change. OOS collection remains inactive.
