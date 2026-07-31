# Trend Strategy v2 Current State

## Current status

- PR #18 froze a v1 OOS candidate as baseline evidence.
- PR #24 defined its maturity protocol.
- The v1 OOS collector was not implemented or activated.
- No OOS collection is active.
- Phase A architecture and deterministic comparison methodology are implemented.
- Legacy v1 scanner and backtest behavior remain unchanged.

## Current phase

Phase A -- implementation complete; empirical score-breakout result is
Inconclusive for the blocking evidence reason below.

## Completed Phase A work

- Added first-class boundaries for universe eligibility, trend filter, signal,
  entry, initial stop, trailing exit, position sizing, and portfolio
  construction.
- Decomposed `signal_surge_v0` into separately observable universe, trend
  filter, trigger, continuous ranking, and risk diagnostic fields while
  retaining exact Boolean reconstruction.
- Retained `signal_surge_v0` only as a legacy baseline in the v2 comparison.
- Reduced the v2 score-breakout family to `score_lookback` as its only
  optimized numeric parameter. R20 and ER20 remain diagnostics and ranking
  values and are not v2 thresholds.
- Added deterministic comparisons against trend-filter-only, prior-price-high,
  per-symbol frequency-matched random events, and a within-symbol shifted
  placebo.
- Applied one common next-open entry, signal-day Low20 initial stop,
  ratcheting Low20 trailing exit, transaction cost, equal-weight sizing, and
  canonical portfolio construction path to every signal comparison.
- Added signal-only predeclared forward returns plus 20-session MFE and MAE.
  These diagnostics do not create an exit.
- Added exact-common-date SPY-relative CAGR, maximum drawdown, CDaR95, Calmar,
  recovery duration, turnover, and cost results. Profit Factor, win rate, and
  trade-return statistics remain diagnostics.
- Added deterministic Retain/Inconclusive/Reject rules. The rules require at
  least five exact common SPY years and 100 score events before Retain or
  Reject can be returned.
- Added no fixed holding-period exit, fixed profit target, broad Phase B rule
  expansion, OOS collector, or OOS protocol/manifests.

## Score-breakout classification

**Inconclusive.** The repository does not contain a frozen multi-year raw
adjusted-OHLCV panel from which the four required comparator event paths can be
reconstructed. The committed `docs/data` outputs contain aggregate v1 results
and recent/generated artifacts, but not the trend-only, prior-price-high,
frequency-matched-random, and shifted-placebo lifecycles on identical economic
dates and trading rules. Phase A deliberately did not download market data or
regenerate `docs/data`, so an empirical Retain or Reject result cannot be
supported in this phase. The implemented classifier returns Inconclusive when
the predeclared history or event-count evidence minimum is absent.

## Blocking limitations

- An approved frozen raw adjusted-OHLCV research panel is required to run the
  Phase A comparison and resolve the score-breakout classification empirically.
- The historical universe still uses present-day AUM and present-day product
  availability. Final performance conclusions require point-in-time universe
  reconstruction or survivorship-sensitivity analysis.
- The provisional Phase A trend filter is the decomposed legacy scanner trend
  control; the final v2 trend-filter definition remains an open decision.

## Exact next Phase B task

Phase B1 -- after the Phase A comparison is run on an approved frozen panel,
preregister and sequentially screen behaviorally distinct entry families for
the Phase A-surviving signals while holding universe, trend filter, initial
stop, trailing exit, transaction costs, position sizing, and portfolio
construction fixed. Do not expand initial-stop or trailing-exit families in
that first Phase B1 screen.

## Explicitly deferred

- broad entry/exit expansion beyond the exact Phase B1 task;
- position-sizing optimization;
- point-in-time universe reconstruction;
- new OOS cohort creation;
- OOS collector implementation or activation;
- scanner UI redesign.
