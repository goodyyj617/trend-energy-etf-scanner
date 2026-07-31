# Trend Strategy v2 Current State

## Current status

- PR #18 froze a v1 OOS candidate as baseline evidence.
- PR #24 defined its maturity protocol.
- The v1 OOS collector was not implemented or activated.
- No OOS collection is active.
- Phase A1 architecture and the deterministic comparison framework are
  implemented.
- Legacy v1 scanner and backtest behavior remain unchanged.
- Phase A has not completed because Phase A2 has not run.

## Current phase

Phase A1 -- comparison framework implemented

## Completed Phase A1 work

- Added first-class boundaries for universe eligibility, trend filter, signal,
  entry, initial stop, trailing exit, fixed canonical equal-weight sizing, and
  portfolio construction.
- Added a fixed, score-independent primary trend filter: close above MA200 and
  positive MA200 slope over 20 sessions. MA200 and its slope are computed only
  inside the Trend v2 research path and are not optimized.
- Kept `legacy_scanner_trend_filter_control_v0` as a separately named
  sensitivity comparator rather than the primary Phase A filter.
- Decomposed `signal_surge_v0` into separately observable universe, legacy
  trend filter, trigger, continuous ranking, and risk diagnostic fields while
  retaining exact Boolean reconstruction.
- Retained `signal_surge_v0` only as a legacy baseline in the v2 comparison.
- Reduced the v2 score-breakout family to `score_lookback` as its only
  optimized numeric parameter. R20 and ER20 remain diagnostics and ranking
  values and are not v2 thresholds.
- Added deterministic comparisons against trend-filter-only, prior-price-high,
  per-symbol executable-trigger-frequency-matched random events, and a
  within-symbol shifted placebo.
- Added separate raw Boolean signal, first-event executable-trigger, and
  completed-lifecycle counts. Random controls fail clearly when exact
  per-symbol executable-trigger matching is impossible; shifted controls
  report edge loss and comparability.
- Applied one common next-open entry, signal-day Low20 initial stop,
  ratcheting Low20 trailing exit, transaction cost, fixed canonical equal
  weight, and canonical portfolio construction path to every signal
  comparison.
- Added signal-only predeclared forward returns plus 20-session MFE and MAE.
  These diagnostics do not create an exit.
- Added exact-common-date SPY-relative CAGR, maximum drawdown, CDaR95, Calmar,
  recovery duration, turnover, and cost results. Profit Factor, win rate, and
  trade-return statistics remain diagnostics.
- Added deterministic Retain/Inconclusive/Reject mechanics that fail closed
  unless complete walk-forward, leave-one-year-out, paired-bootstrap,
  multiple-testing, asset-group, and event/executable-trigger comparability
  evidence is supplied.
- Added no fixed holding-period exit, fixed profit target, Phase A2 empirical
  run, broad Phase B rule expansion, OOS collector, or OOS protocol/manifests.

## Score-breakout empirical classification

Score breakout has **not yet been empirically classified**. The framework's
current Inconclusive path reflects unavailable empirical and robustness
evidence. It is not evidence for or against score breakout, and synthetic
fixtures validate mechanics only.

The repository does not contain a frozen multi-year raw adjusted-OHLCV panel
from which every required comparator path and robustness result can be
reproduced. The committed `docs/data` outputs contain aggregate v1 results and
recent/generated artifacts, not the complete Phase A2 inputs and evidence.
This correction does not download market data or run empirical research-data
generation.

## Blocking limitations

- Phase A2 requires one approved, frozen, reproducible adjusted-OHLCV research
  snapshot.
- The historical universe still uses present-day AUM and present-day product
  availability. Final performance conclusions require point-in-time universe
  reconstruction or survivorship-sensitivity analysis.
- Retain or Reject cannot be produced without the preregistered robustness and
  control-comparability evidence.

## Exact next task

Phase A2 -- freeze one reproducible adjusted-OHLCV research snapshot, execute
the preregistered score-breakout comparison, and produce an empirical
Retain/Inconclusive/Reject result.

## Explicitly deferred

- Phase B1 entry-family screening;
- broad entry/exit expansion;
- position-sizing optimization;
- point-in-time universe reconstruction;
- new OOS cohort creation;
- OOS collector implementation or activation;
- scanner UI redesign.
