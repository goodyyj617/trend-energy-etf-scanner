# Trend Strategy v2 Foundation 5

## Scope and versions

Foundation 5 adds controlled strategy construction and bounded local execution.
It does not add an optimizer, unrestricted search, remote execution, market-data
download, distributed workers, or new signal research.

Version boundaries are `strategy_construction_request_v1`,
`normalized_strategy_construction_v1`, `candidate_space_estimate_v1`,
`execution_confirmation_v1`, `execution_request_v1`,
`local_execution_policy_v1`, `trend_v2_phase_a_controlled_adapter_v1`,
`trend_v2_controlled_write_api_v1`, and
`trend_v2_korean_controlled_strategy_ui_v1`.

Existing `StrategyRunSpec`, `StrategyRunManifest`, `EvaluationProfile`,
`EvaluationRun`, `ResultStore`, `execution_attempt_v1`, and
`saved_run_registry_v1` remain separate. Construction is not a completed run,
an execution request is not an attempt, and an attempt is not an immutable
economic result. Operational states remain absent from `StrategyRunManifest`.

## Construction and supported options

The request covers snapshot, inclusive dates, universe, benchmark, trend
filter, signal, entry, initial stop, trailing exit, sizing, portfolio
constraints, transaction cost, slippage, walk-forward work, robustness work,
and evaluation profiles.

The initial allow-list is deliberately narrow:

- validated `phase_a2_frozen_2026_07_30` snapshot;
- `phase_a2_historical_eligible_v1` universe and `spy_adjusted_close_v1`;
- `price_above_rising_ma200_v0`;
- the established `prior_price_high_l20_v1` baseline only;
- `first_event_next_open_v1`;
- `signal_day_low20_v1` and `ratcheting_low20_v1`;
- `canonical_equal_weight_active_v1`;
- `long_only_cash_constrained_v1`;
- finite round-trip transaction-cost and slippage values from 0 to 50 bp.

There is no fixed holding-period exit or fixed profit target. The retired
internal trend-energy composite and its breakout trigger are not construction
options. Legacy source options are not enabled automatically.

## Parameter normalization

A numeric parameter accepts one fixed value, an explicit finite list, an
inclusive integer range, or an inclusive decimal range with a positive explicit
step. Decimal parsing uses `Decimal`; floats, non-finite values, nonpositive
steps, reversed ranges, inexact endpoints, duplicate normalized values, unknown
fields, and out-of-schema values fail closed.

Exact decimals use their shortest base-10 string. Equivalent fixed,
singleton-list, and singleton-range inputs produce the same hash. Candidate
order is ascending `strategy_run_id`; equivalent specifications deduplicate by
that identity. Policy permits at most eight values per dimension and two
varying dimensions. No arbitrary signal grid is exposed.

## Candidate estimate

Let `E` be distinct economic candidates, `P` profiles, `F` walk-forward folds,
and `R` robustness scenarios:

- raw Cartesian candidates = product of declared dimension counts;
- evaluation applications = `E × P`;
- fold executions = `E × F`;
- robustness scenarios = `E × R`;
- benchmark calculations = `E`;
- derived calculations = `E`;
- total units = `E + E×P + E×F + E×R + E + E`.

Evaluation applications are not called economic backtests. Existing valid runs
are reported as reuse and reduce estimated new backtests. For example, two
economic candidates, two profiles, three folds, and two robustness scenarios
produce 2 economic, 4 evaluation, 6 fold, 4 robustness, 2 benchmark, and 2
derived units: 20 total.

## Policy and confirmation

`config/trend_v2/local_execution_policy_v1.json` is the single policy source.
It sets informational work at 8 units, confirmation at 24, hard refusal at 256,
maximums of 64 economic candidates, 128 evaluation applications, 256 total
units, 4,000 calendar days, 500 universe members, eight values per dimension,
two varying dimensions, one worker, a 256 KiB body, and a 15-minute one-time
confirmation. Every estimate returns each limit, observed value, severity, and
trigger result. Confirmation cannot bypass a hard violation.

`execution_confirmation_v1` binds normalized construction hash, estimate hash,
policy version/hash, threshold-result hash, creation, expiry, and one-time use.
Any strategy, date, snapshot, universe, benchmark, cost, slippage, profile
workload, walk-forward/robustness setting, policy, or estimate change makes it
stale. Usage is persisted against exactly one immutable request.

## ExecutionRequest and ExecutionAttempt

`execution_request_v1` records normalized construction and estimate, required
confirmation identity, timestamp, ordered StrategyRun candidates, profiles,
policy, source commit, engine, snapshot, and expected outputs. It is immutable.

Starting creates one `execution_attempt_v1` per candidate because that existing
contract intentionally identifies one StrategyRun. Request status aggregates
the attempts. Later attempts for the same economic identity get a new attempt
number and retry-parent ID; prior events remain append-only. Same-status
nonterminal progress snapshots preserve stages without changing identity.

Create, start, read, cooperative cancel, and retry are supported. Start is
idempotent. Queued cancellation is immediate. Running cancellation moves to
`cancelling` and stops at a safe candidate/pre-commit boundary; it cannot
interrupt arbitrary Python in the middle of one synchronous calculation.
Retry is allowed only for failed or cancelled attempts and preserves the
immutable request ID in provenance.

## Engine adapter and outputs

The adapter reuses `build_historical_features`,
`evaluate_signal_observations(make_prior_price_high_rule(20))`,
`simulate_signal_lifecycles`, and `simulate_canonical_portfolio`. This is the
existing score-independent Phase A path with next-open entry and Low20 stops,
explicitly without a fixed holding exit. It validates the frozen snapshot and
every option. Cost and slippage are combined as round-trip bp assumptions.
On Windows, the adapter accepts only the proven CRLF-to-canonical-LF checkout
equivalence for `universe_snapshot.csv`; the normalized bytes must match the
recorded member hash, every gzip price shard remains byte-strict, and the
complete snapshot hash remains strict.

Only produced evidence is stored: `daily_portfolio_curve_v1`, SPY curve,
`trade_lifecycles_v1`, sparse `signal_execution_events_v1`, and a successful
terminal manifest. Foundation 2 creates content-addressed derived, yearly,
rolling, behavior, and EvaluationRun outputs. Missing robustness simulations
are not fabricated and enabled vetoes fail closed.
Nonzero walk-forward fold or robustness scenario counts remain available for
exact workload preview, but execution-request creation fails with
`engine_unsupported` until an adapter can produce those artifacts.

Daily ETF Scan remains scan-only. Backtest Only remains the legacy full-grid
publisher. Foundation 5 invokes neither entrypoint and changes neither path.

## Cache and duplicate policy

- valid successful equivalent with daily and benchmark artifacts: reuse;
- profile-only change: reuse economic/derived artifacts and equivalent
  EvaluationRuns;
- absent run: execute once;
- active equivalent: reject `duplicate_active_execution`;
- corrupt, provenance-invalid, missing, partial, failed, or pruned equivalent:
  reject `stored_equivalent_run_corrupt`.

A failed candidate remains an attempt failure and never becomes a successful
manifest. Successful siblings remain stored. Any orphan bytes from an
unexpected pre-manifest failure remain discoverable, not promoted.

## API, UI, idempotency, and errors

Existing read routes remain compatible. Added routes are:

- `GET /api/v1/construction/options`;
- `POST /api/v1/construction/normalize` and `/estimate`;
- `POST /api/v1/construction/confirm`;
- `POST /api/v1/execution-requests`;
- `GET /api/v1/execution-requests/{id}`;
- `POST /api/v1/execution-requests/{id}/start`;
- `POST /api/v1/execution-attempts/{id}/cancel` and `/retry`.

Normalize/estimate are pure. Confirmation, request creation, start, cancel, and
retry require persistent canonical idempotency keys. Repetition returns the
original identity; a key reused with different content fails.

The Korean UI adds `전략 구성` and `실행 요청`, explanations for every control,
normalized preview, raw/economic/evaluation/robustness/total counts, every
threshold, hard failure, deliberate unselected confirmation, request/attempt
identities, per-candidate progress, reuse/failure/cancellation counts, stages,
timestamps, failures, artifacts, cancel, and retry. Text and symbols accompany
color. Operational progress is not presented as StrategyRun success.

Errors include stable construction, option, range, overflow, confirmation,
hard-limit, duplicate, snapshot, benchmark, universe, engine, lifecycle,
corrupt-cache, request-size, and internal-execution codes. Each response has a
Korean message, English diagnostic, request ID, object identity, recoverability,
and optional next action, without stacks or absolute paths.

## Security and provenance

The server remains loopback-first with no remote fetch, user filesystem path,
traversal, shell input, command construction, dynamic Python/import, untrusted
pickle, or unbounded body/response. CORS is disabled unless an explicit local
origin is configured. Concurrency is one. All inputs use allow-listed schemas.

Canonical hashes reconstruct user input, normalization, estimate, policy,
confirmation, request, attempts, candidate order, cache decisions, stored
artifacts, derived calculations, and evaluations.

## Limitations and deferred work

- cancellation is cooperative at safe boundaries;
- fold and robustness work is counted for preview but nonzero execution
  requests are refused by this first adapter;
- only the established L20 baseline and fixed Phase A execution rules exist;
- requested endpoints must be available frozen economic observations;
- authentication, remote access, distributed scheduling, unrestricted search,
  new strategy research, market download, point-in-time universe repair, and
  OOS activation remain deferred.

The exact next task is Foundation 6: broaden the controlled strategy library
and add persisted execution management after the Foundation 5 contracts and
bounded execution path are proven stable.
