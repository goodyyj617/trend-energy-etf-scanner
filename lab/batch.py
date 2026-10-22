"""Run lab tests from the command line with the app's default universes and settings.

Everything is logged exactly as when the test is run in the app: every backtested cell goes
to lab_results/trials.csv (DSR bookkeeping) and each result is saved under lab_results/ so
it shows in the app's "저장된 결과" page.

    python -m lab.batch addon switch market_trend       # one common add-on test, both universes
    python -m lab.batch regime                          # the pre-registered macro regime set (DECISIONS 51)

Universes (the app's defaults, DECISIONS 43):
  snapshot  ETF snapshot after the default filter funnel, 2017-08-01 - 2023-12-31
  long      long-history multi-asset ETFs after the same funnel, 2000-01-01 - 2023-12-31
The end date is the day before the holdout start; holdout data is never used here.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

from . import research_log as rlog
from . import store
from .blocks import ENTRY_BLOCKS, REGIME_PAIRS
from .data import BENCHMARK, LONG_HISTORY_ETFS, Panel, download_panel, load_snapshot
from .engine import StrategyConfig
from .families import AddonTest, FamilyComparison, run_addon_test, run_family_comparison
from .grid import Gates
from .universe import UniverseFilter, apply_filter, candidate_table

UNIVERSE_NAMES = {"snapshot": "ETF 스냅샷 2017–2023", "long": "장기 멀티에셋 2000–2023"}
START = {"snapshot": "2017-08-01", "long": "2000-01-01"}
# DECISIONS 51: the four regimes, each as an entry filter and as a switch (filter + complement exit)
REGIME_TESTS = [(kind, key) for key in REGIME_PAIRS for kind in ("entry", "switch")]


@dataclass
class Universe:
    name: str
    panel: Panel
    cfg: StrategyConfig


def research_end() -> str:
    hold = rlog.load_holdout()
    if not hold["enabled"]:
        raise SystemExit("보류 구간이 꺼져 있습니다. 연구 종료일을 정할 수 없어 중단합니다.")
    return str(date.fromisoformat(hold["start"]) - timedelta(days=1))


def build_universe(kind: str) -> Universe:
    hold = rlog.load_holdout()
    end = research_end()
    if kind == "snapshot":
        base = load_snapshot()
        candidates = base.symbols
    else:
        syms = [s for v in LONG_HISTORY_ETFS.values() for s in v]
        base, _ = download_panel(syms, end, source="장기 멀티에셋 ETF (yfinance)")
        candidates = base.tradable_symbols
    table, _ = apply_filter(candidate_table(candidates, base.info), UniverseFilter(min_aum_musd=1000.0),
                            base.close, hold["start"])
    chosen = sorted(table.loc[table["status"] == "선택", "symbol"])
    cfg = StrategyConfig(entries={}, exits={}, start=START[kind], end=end, sizing="inverse_vol")
    return Universe(UNIVERSE_NAMES[kind], base.subset(chosen), cfg)


def _collector(u: Universe, kind: str, sink: list):
    return lambda c, r: sink.append((kind, c, u.panel.tradable_symbols, r.equity))


def plain_comparison(u: Universe) -> FamilyComparison:
    rows: list = []
    fc = run_family_comparison(u.panel, u.cfg, Gates(), on_result=_collector(u, "family", rows))
    rlog.log_trials(rows)  # identical cells already in the log are not new trials
    return fc


def addon(u: Universe, kind: str, key: str, plain: FamilyComparison, save: bool = True) -> AddonTest:
    rows: list = []
    at = run_addon_test(u.panel, u.cfg, Gates(), kind, key, plain=plain, on_result=_collector(u, "addon", rows))
    rlog.log_trials(rows)
    if save:
        store.save_addon(at, f"{at.label} · {u.name}", u.panel.source, u.panel.tradable_symbols)
    return at


# --------------------------------------------------------------------------- descriptive timing table

def regime_states(panel: Panel) -> pd.DataFrame:
    """Market-wide on/off state of each regime entry filter at its default (True = buying allowed)."""
    out = {}
    for key in REGIME_PAIRS:
        b = ENTRY_BLOCKS[key]
        out[key] = b.compute(panel, b.param.default).iloc[:, 0].astype(bool)
    return pd.DataFrame(out)


def timing_table(panel: Panel, start: str, end: str) -> pd.DataFrame:
    """SPY's next-day returns split by each regime state (descriptive only, not a verdict).
    A state decided on day t is paired with SPY's return from the t close to the t+1 close."""
    spy = panel.close[BENCHMARK]
    nxt = spy.pct_change(fill_method=None).shift(-1)
    fwd63 = spy.shift(-63) / spy - 1
    states = regime_states(panel)
    mask = (spy.index >= pd.Timestamp(start)) & (spy.index <= pd.Timestamp(end))
    rows = []
    for key in states:
        for on in (True, False):
            sel = mask & (states[key] == on).to_numpy() & nxt.notna().to_numpy()
            r = nxt[sel]
            rows.append({
                "regime": ENTRY_BLOCKS[key].label, "state": "켜짐(매수 허용)" if on else "꺼짐",
                "share_of_days": float(sel.sum() / max(1, (mask & nxt.notna().to_numpy()).sum())),
                "spy_ann_return": float(r.mean() * 252), "spy_ann_vol": float(r.std() * np.sqrt(252)),
                "spy_sharpe": float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else np.nan,
                "spy_fwd63_median": float(fwd63[sel].median()),
                "spells": int(((states[key] == on) & (states[key].shift(1) != on))[mask].sum()),
            })
    return pd.DataFrame(rows)


def main(argv: list[str]) -> None:
    if not argv:
        raise SystemExit(__doc__)
    if argv[0] == "addon":
        tests = [(argv[1], argv[2])]
    elif argv[0] == "regime":
        tests = REGIME_TESTS
    else:
        raise SystemExit(__doc__)
    for kind in ("snapshot", "long"):
        u = build_universe(kind)
        print(f"== {u.name}: {len(u.panel.tradable_symbols)} symbols, {u.cfg.start} - {u.cfg.end}", flush=True)
        if argv[0] == "regime":
            print(timing_table(u.panel, u.cfg.start, u.cfg.end).to_string(index=False), flush=True)
        plain = plain_comparison(u)
        for k, key in tests:
            at = addon(u, k, key, plain)
            print(f"{at.label}: {at.counts()}", flush=True)
    n, _ = rlog.trial_stats()
    n_eff, rho, _ = rlog.effective_trials()
    print(f"trials {n}, effective {n_eff:.0f} (rho {rho:.2f})")


if __name__ == "__main__":
    main(sys.argv[1:])
