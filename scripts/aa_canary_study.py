"""Canary signals on B2–B5: move nasdaq-100 when a signal warns (allocation DECISIONS A40; memo 2026-10-28_canary.md).

    .venv/Scripts/python.exe scripts/aa_canary_study.py --out docs/research/allocation/2026-10-28_canary
    .venv/Scripts/python.exe scripts/aa_canary_study.py --quick --out %TEMP%\\aa_canary_quick     # ~1 min check

Design (fixed before the run): https://claude.ai/artifact/7AyPtvA6372TwXaSJFYxPq, lab/aa_canary.py.

  base       B2–B5 as registered / found (static, quarter-end rebalance, 10 bp, real prices after listing)
  signals    11 canaries that are never bought (K1–K11: US ETFs in USD, FRED series, USD/KRW) and 3 rules on the
             held products' KRW prices (D1–D3); each with its centre and two neighbour settings
  actions    K and D1: nasdaq → defensive equity (staples · health care) or → the safe sleeve; D2 / D3 tilt 10 %p
  trading    the state is read every close; a change trades at the next close (바로 매매), or with no new signal
             trade for 21 rows after one (1개월 금지); the quarter-end rebalance brings the held state's target back
  periods    2005-01-03 – 2026-10-16 and 2000-09-01 – 2026-10-16 (A37 ④ proxies, model before 2005). D rules read
             the 2000-09 KRW panel (the longest history) in both periods; canaries read their own history
  checks     currency (K1 · K3 · K4 on KRW prices), 20 bp one way, rebalance-date offsets 0/1/5/10/15 (rows that
             pass 1 and 2), the equal-cost static allocation, 200 placebo signals per rule (circular shifts ≥ 1 year)
  verdict    1 2005 MDD ≥ −15 %   2 2000-09 MDD ≥ −15 % (from the signal start when it starts late)
             3 both neighbours pass 1 · 2, and the worst MDD over the offsets (both periods) ≥ −20 %
             4 2000-09 MDD ≥ 1 %p shallower than the equal-cost static   5 deeper placebos ≥ 90 %

Placebos follow the 바로 매매 rule (the design's count); a 1개월 금지 row is read against the same placebo set.
Writes small tables and figures into --out; nothing goes to lab_results/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import aa_global_search as G  # noqa: E402
import aa_local_study as L  # noqa: E402
import aa_trend_filter_study as T  # noqa: E402
from lab import aa_canary as K  # noqa: E402
from lab import aa_composition as C  # noqa: E402
from lab import aa_local as loc  # noqa: E402
from lab import aa_search as srch  # noqa: E402
from lab import aa_search_results as sr  # noqa: E402
from lab import figures as figs  # noqa: E402
from lab.aa_assets import CASH, PRESETS  # noqa: E402
from lab.aa_engine import AllocConfig, run_allocation, simulate  # noqa: E402
from lab.aa_korea import load_usdkrw  # noqa: E402
from lab.aa_validate import TIMING_OFFSETS  # noqa: E402
from lab.data import download_panel  # noqa: E402

SEED = 4040
HALF = "2016-01-01"
ERAS = {"2005": sr.START, "2000-09": loc.DOTCOM_START}
VERSIONS = {"바로 매매": 0, "1개월 금지": K.COOLDOWN}
SETTINGS = ("중심", "이웃 1", "이웃 2")
REFERENCE = "stable40"
TIMING_LINE = -0.20  # A30's date-robust line
EQ_MARGIN = 0.01  # verdict 4: at least 1 %p shallower than the equal-cost static
PLACEBO_SHARE = 0.90  # verdict 5
CHUNK = 1500
FULL = dict(placebos=200, offsets=TIMING_OFFSETS)
QUICK = dict(placebos=4, offsets=(0, 5))
NEEDED = ("SPY", "VWO", "BND", "EFA", "EEM", "AGG", "TIP", *K.SECTORS)

# loaders: tests swap them for made-up prices
LOADERS = {"2005": lambda prices: G.load(prices), "2000-09": lambda prices: L.load_extended(prices)[0]}


def load_canaries() -> dict:
    """USD closes of every canary (carried back with older funds), the USD cash return and USD/KRW."""
    usd, failed = download_panel(K.CANARY_TICKERS, end=G.DATA_END)
    if failed:
        print("받지 못한 신호 자산:", failed)
    closes, chains = K.chained(usd.close)
    return {"usd": closes, "usd_cash": usd.cash, "fx": load_usdkrw(), "macro_df": None, "chains": chains, "failed": failed}


def pairs() -> list[tuple[str, str]]:
    return [(r.key, a) for r in K.RULES.values() for a in r.actions]


def columns() -> list[str]:
    held = {s for w in C.CENTRES.values() for s in w} | set(K.DEFENSIVE) | set(PRESETS[REFERENCE].weights)
    return [s for s in sr.SYMBOLS if s in held]


def windows(era: str) -> dict:
    return {"dotcom": loc.DOTCOM_WINDOW, **sr.WINDOWS} if era == "2000-09" else dict(sr.WINDOWS)


def spec(era: str, name: str, rule: str = "", action: str = "", version: str = "바로 매매", setting: str = "중심",
         cur: str = "usd", cost: int = 10, offset: int = 0, kind: str = "신호", **kw) -> dict:
    param = K.RULES[rule].params[SETTINGS.index(setting)] if rule else None
    return {"종류": kind, "기간": era, "배분": name, "신호": rule, "행동": action, "판": version, "설정": setting, "값": param,
            "통화": cur, "비용": cost, "날짜 이동": offset, **kw}


class Study:
    def __init__(self, krw: dict, can: dict):
        self.syms = columns()
        self.columns = self.syms + [CASH]
        self.krw = krw
        self.sims = {(era, cost): K.DynamicSim(krw[era], self.syms, start, sr.END, cost_bps=cost)
                     for era, start in ERAS.items() for cost in (10, 20)}
        self.inp = {cur: K.Inputs(can["usd"], can["usd_cash"], can["fx"], krw["2000-09"], can["macro_df"], cur)
                    for cur in ("usd", "krw")}
        self.full_dates = krw["2000-09"].close.index.union(can["usd"].index).union(self.sims["2005", 10].dates)
        self._raw: dict = {}
        self._tab: dict = {}

    def raw_full(self, rule: str, param, cur: str, name: str) -> pd.Series:
        """A rule's state on every date of the 2000-09 KRW panel (its whole history, so the first day of a
        period already has yesterday's FRED value)."""
        key = (rule, param, cur, name if rule in ("d2", "d3") else None)
        if key not in self._raw:
            base = C.CENTRES[name] if rule in ("d2", "d3") else None
            self._raw[key] = K.raw_states(rule, self.inp[cur], param, self.full_dates, base)
        return self._raw[key]

    def raw(self, era: str, rule: str, param, cur: str, name: str) -> pd.Series:
        return self.raw_full(rule, param, cur, name).reindex(self.sims[era, 10].dates)

    def table(self, name: str, rule: str, action: str) -> np.ndarray:
        key = (name, rule, action)
        if key not in self._tab:
            t = K.tables(C.CENTRES[name], rule, action, self.columns)
            self._tab[key] = np.vstack([t, np.repeat(t[:1], 3 - len(t), axis=0)])  # every path gets three states
        return self._tab[key]

    def vector(self, weights: dict) -> np.ndarray:
        v = np.zeros(len(self.columns))
        for k, x in weights.items():
            v[self.columns.index(k)] = x
        return v / v.sum()

    def start_of(self, s: dict) -> pd.Timestamp:
        """Where MDD is measured: the period's first day, or the rule's first readable day when later."""
        dates = self.sims[s["기간"], 10].dates
        if s.get("측정 시작") is not None:
            return pd.Timestamp(s["측정 시작"])
        if not s["신호"]:
            return dates[0]
        first = K.signal_start(self.raw(s["기간"], s["신호"], s["값"], s["통화"], s["배분"]))
        return dates[0] if first is None or first <= dates[0] else first

    def build(self, s: dict):
        sim = self.sims[s["기간"], s["비용"]]
        q = K.quarter_rows(sim.dates, s["날짜 이동"])
        if s.get("w") is not None:  # a static allocation (the comparators and the equal-cost rows)
            tab, st = np.repeat(np.asarray(s["w"], dtype=float)[None], 3, axis=0), np.zeros(sim.n_days, dtype=int)
        else:
            st = self.raw(s["기간"], s["신호"], s["값"], s["통화"], s["배분"]).fillna(0).astype(int).to_numpy()
            if s["종류"] == "가짜":
                st = np.roll(st, -int(s["밀기"]))
            tab = self.table(s["배분"], s["신호"], s["행동"])
        return tab, K.schedule(st, q, VERSIONS.get(s["판"], 0)), self.start_of(s)

    def run(self, specs: list[dict], label: str) -> pd.DataFrame:
        groups: dict = {}
        for i, s in enumerate(specs):
            groups.setdefault((s["기간"], s["비용"]), []).append(i)
        parts = []
        done = 0
        for (era, cost), idx in groups.items():
            sim = self.sims[era, cost]
            for a in range(0, len(idx), CHUNK):
                sub = idx[a:a + CHUNK]
                built = [self.build(specs[i]) for i in sub]
                paths, turn, trades = sim.run(np.stack([b[0] for b in built]), np.vstack([b[1].decide for b in built]),
                                              np.vstack([b[1].choice for b in built]), np.array([b[1].first for b in built]))
                m = self.metrics(paths, turn, trades, sim, era, cost, [b[1] for b in built], [b[2] for b in built])
                m.insert(0, "_i", sub)
                parts.append(m)
                done += len(sub)
                if len(specs) > CHUNK:
                    print(f"  {label}: {done:,}/{len(specs):,}", flush=True)
        df = pd.concat(parts, ignore_index=True).sort_values("_i").drop(columns="_i").reset_index(drop=True)
        meta = pd.DataFrame([{k: v for k, v in s.items() if k != "w"} for s in specs])
        meta = meta.drop(columns=[c for c in meta.columns if c in df.columns])  # 측정 시작 is reported by metrics
        return pd.concat([meta, df], axis=1)

    @staticmethod
    def metrics(paths, turn, trades, sim, era, cost, scheds, starts) -> pd.DataFrame:
        df = srch.summarize(paths, turn, sim.dates, sim.cash, windows=windows(era), halves=(HALF,) if era == "2005" else ())
        i0 = np.array([int(sim.dates.searchsorted(s)) for s in starts])
        mdd_m, cagr_m = np.empty(len(paths)), np.empty(len(paths))
        for k in np.unique(i0):
            rows = np.flatnonzero(i0 == k)
            mdd_m[rows] = srch._mdd(paths[rows, k:])
            cagr_m[rows] = srch._cagr(paths[rows, max(k - 1, 0):])
        years = sim.n_days / 252.0
        df["측정 시작"] = [s.date().isoformat() for s in starts]
        df["늦게 시작"] = i0 > 0
        df["mdd_meas"], df["cagr_meas"] = mdd_m, cagr_m
        df["신호 매매"] = [s.n_signal for s in scheds]
        df["연 신호 매매"] = df["신호 매매"] / years
        df["21일 안 되돌림"] = [s.whipsaws() for s in scheds]
        df["옮긴 비율"] = [float((s.choice[k:] != 0).mean()) for s, k in zip(scheds, i0)]
        df["상태 2 비율"] = [float((s.choice[k:] == 2).mean()) for s, k in zip(scheds, i0)]
        df["매매 수"] = trades
        df["연 비용"] = np.asarray(turn) * cost / 1e4
        return df


# --------------------------------------------------------------------------- the runs

def main_specs() -> list[dict]:
    out = []
    for era in ERAS:
        for name in C.NAMES:
            for rule, action in pairs():
                for version in VERSIONS:
                    for setting in SETTINGS:
                        out.append(spec(era, name, rule, action, version, setting))
                    out.append(spec(era, name, rule, action, version, cost=20))
            for rule in K.CURRENCY_RULES:
                for action in K.RULES[rule].actions:
                    out.append(spec(era, name, rule, action, cur="krw"))
    return out


def static_specs(study: Study) -> list[dict]:
    named = {**{n: C.CENTRES[n] for n in C.NAMES}, PRESETS[REFERENCE].name: PRESETS[REFERENCE].weights}
    return [spec(era, n, kind="정적", version="정적 분기", setting="", w=study.vector(w)) for era in ERAS for n, w in named.items()]


def a38_rows(krw: dict) -> pd.DataFrame:
    """A38 (equities below their 10-month mean → CD금리, monthly) through the engine, as a comparator."""
    rows = []
    for era, start in ERAS.items():
        res = [run_allocation(krw[era], T.config(C.CENTRES[n], "필터 10", start)) for n in C.NAMES]
        paths = np.vstack([r.equity.to_numpy(dtype=float) for r in res])
        df = srch.summarize(paths, np.array([r.turnover for r in res]), res[0].equity.index, res[0].cash, windows=windows(era))
        df.insert(0, "배분", list(C.NAMES))
        df.insert(0, "기간", era)
        df.insert(0, "종류", "A38 주식 10개월선")
        df["mdd_meas"], df["cagr_meas"] = df["mdd"], df["cagr"]
        rows.append(df)
    return pd.concat(rows, ignore_index=True)


def centre_key(df: pd.DataFrame) -> pd.Series:
    return (df["종류"] == "신호") & (df["설정"] == "중심") & (df["통화"] == "usd") & (df["비용"] == 10) & (df["날짜 이동"] == 0)


def offset_specs(runs: pd.DataFrame, offsets) -> list[dict]:
    c = runs[centre_key(runs)]
    piv = c.pivot_table(index=["배분", "신호", "행동", "판"], columns="기간", values="mdd_meas")
    ok = piv[(piv["2005"] >= sr.MDD_LINE) & (piv["2000-09"] >= sr.MDD_LINE)].index
    return [spec(era, n, r, a, v, offset=int(o)) for (n, r, a, v) in ok for era in ERAS for o in offsets if o]


def placebo_specs(study: Study, n: int) -> list[dict]:
    out = []
    keys = list(K.RULES)
    for e, era in enumerate(ERAS):
        T_ = study.sims[era, 10].n_days
        for rule, action in pairs():
            seed = SEED + 100 * keys.index(rule) + e
            _, shifts = K.placebo_states(np.zeros(T_, dtype=int), n, seed)
            for name in C.NAMES:
                real = spec(era, name, rule, action)
                start = study.start_of(real)
                for k, sh in enumerate(shifts):
                    out.append({**real, "종류": "가짜", "가짜 번호": k, "밀기": int(sh), "측정 시작": start})
    return out


def equal_cost(study: Study, runs: pd.DataFrame) -> tuple[list[dict], pd.DataFrame]:
    """For every centre row: the static allocation moving x of the full move for good, x matching the row's
    2005–2026 CAGR (on the 2005 period, quarterly static)."""
    sim = study.sims["2005", 10]
    cache: dict = {}

    def cagr_of(W):
        key = W.tobytes()
        if key not in cache:
            p, t, _ = sim.run(*K.static_rows(W, sim))
            cache[key] = srch._cagr(p)
        return cache[key]

    c = runs[centre_key(runs)]
    piv = c.pivot_table(index=["배분", "신호", "행동", "판"], columns="기간", values=["cagr", "측정 시작"], aggfunc="first")
    specs, rows = [], []
    for (n, r, a, v), row in piv.iterrows():
        x, w = K.equal_cost_static(C.CENTRES[n], r, a, study.columns, float(row[("cagr", "2005")]), cagr_of)
        rows.append({"배분": n, "신호": r, "행동": a, "판": v, "같은 비용 x": x})
        for era in ERAS:
            specs.append(spec(era, n, kind="같은 비용 정적", version=v, setting="", w=w,
                              **{"측정 시작": row[("측정 시작", era)], "짝 신호": r, "짝 행동": a}))
    return specs, pd.DataFrame(rows)


def verdicts(runs: pd.DataFrame, offs: pd.DataFrame | None, plac: pd.DataFrame, eq: pd.DataFrame,
             eq_x: pd.DataFrame) -> pd.DataFrame:
    key = ["배분", "신호", "행동", "판"]
    sig = runs[(runs["종류"] == "신호") & (runs["통화"] == "usd") & (runs["날짜 이동"] == 0)]
    c = sig[(sig["설정"] == "중심") & (sig["비용"] == 10)]
    v = c.pivot_table(index=key, columns="기간", values=["cagr", "mdd_meas", "mdd", "연 신호 매매", "옮긴 비율",
                                                          "21일 안 되돌림", "연 비용"], aggfunc="first")
    v.columns = [f"{a}_{b}" for a, b in v.columns]
    starts = c[c["기간"] == "2000-09"].set_index(key)[["측정 시작", "늦게 시작"]]
    v = v.join(starts.rename(columns={"측정 시작": "2000-09 측정 시작", "늦게 시작": "2000-09 늦게 시작"}))
    v["1 2005 −15% 안"] = v["mdd_meas_2005"] >= sr.MDD_LINE
    v["2 2000-09 −15% 안"] = v["mdd_meas_2000-09"] >= sr.MDD_LINE
    nb = sig[(sig["설정"] != "중심") & (sig["비용"] == 10)]
    nb_ok = nb.assign(ok=nb["mdd_meas"] >= sr.MDD_LINE).groupby(key)["ok"].all()
    v["이웃 둘 다 1·2"] = nb_ok.reindex(v.index).fillna(False).astype(bool)
    worst = c.groupby(key)["mdd_meas"].min()
    if offs is not None and len(offs):
        worst = pd.concat([worst, offs.groupby(key)["mdd_meas"].min()], axis=1).min(axis=1)
    ran = v["1 2005 −15% 안"] & v["2 2000-09 −15% 안"]
    v["날짜 이동 최악 MDD"] = worst.reindex(v.index).where(ran)
    v["3 이웃·날짜 견고"] = v["이웃 둘 다 1·2"] & (v["날짜 이동 최악 MDD"] >= TIMING_LINE)
    e = eq.pivot_table(index=["배분", "짝 신호", "짝 행동", "판"], columns="기간", values=["mdd_meas", "cagr"], aggfunc="first")
    e.index.names = key
    v["같은 비용 x"] = eq_x.set_index(key)["같은 비용 x"].reindex(v.index)
    v["같은 비용 cagr_2005"] = e[("cagr", "2005")].reindex(v.index)
    v["같은 비용 mdd_2000-09"] = e[("mdd_meas", "2000-09")].reindex(v.index)
    v["4 같은 비용보다 1%p 얕음"] = v["mdd_meas_2000-09"] - v["같은 비용 mdd_2000-09"] >= EQ_MARGIN
    real = v["mdd_meas_2000-09"].rename("진짜").reset_index()
    pm = plac.loc[plac["기간"] == "2000-09", ["배분", "신호", "행동", "mdd_meas"]].merge(real, on=["배분", "신호", "행동"])
    v["가짜보다 얕은 비율"] = (pm.assign(deeper=pm["mdd_meas"] < pm["진짜"]).groupby(key)["deeper"].mean()
                          .reindex(v.index))
    v["5 가짜 90% 이상"] = v["가짜보다 얕은 비율"] >= PLACEBO_SHARE
    b20 = sig[(sig["설정"] == "중심") & (sig["비용"] == 20)].pivot_table(index=key, columns="기간", values=["cagr", "mdd_meas"],
                                                                         aggfunc="first")
    for (a, b) in b20.columns:
        v[f"20bp {a}_{b}"] = b20[(a, b)].reindex(v.index)
    checks = ["1 2005 −15% 안", "2 2000-09 −15% 안", "3 이웃·날짜 견고", "4 같은 비용보다 1%p 얕음", "5 가짜 90% 이상"]
    v["통과 수"] = v[checks].sum(axis=1)
    v["판정"] = np.where(v[checks].all(axis=1), "후보", "탈락")
    v = v.reset_index()
    v.insert(2, "신호 이름", v["신호"].map(lambda k: K.RULES[k].label))
    v.insert(4, "행동 이름", v["행동"].map(K.ACTIONS))
    return v


def placebo_summary(plac: pd.DataFrame, runs: pd.DataFrame) -> pd.DataFrame:
    real = runs[centre_key(runs) & (runs["판"] == "바로 매매")].set_index(["기간", "배분", "신호", "행동"])["mdd_meas"]
    g = plac.groupby(["기간", "배분", "신호", "행동"])["mdd_meas"]
    out = pd.DataFrame({"가짜 수": g.size(), "가짜 5%": g.quantile(0.05), "가짜 중앙": g.median(), "가짜 95%": g.quantile(0.95)})
    out["진짜 MDD"] = real.reindex(out.index)
    pm = plac.join(out["진짜 MDD"], on=["기간", "배분", "신호", "행동"])
    out["가짜보다 얕은 비율"] = (pm["mdd_meas"] < pm["진짜 MDD"]).groupby([pm[k] for k in ("기간", "배분", "신호", "행동")]).mean()
    return out.reset_index()


def rules_table(study: Study) -> pd.DataFrame:
    rows = []
    for r in K.RULES.values():
        for i, p in enumerate(r.params):
            first = K.signal_start(study.raw_full(r.key, p, "usd", "B2"))
            rows.append({"신호": r.key, "이름": r.label, "출처": r.source, "계산과 위험 조건": r.what, "설정": SETTINGS[i], "값": p,
                         "행동": " / ".join(K.ACTIONS[a] for a in r.actions), "종류": "카나리아" if r.kind == "canary" else "보유 상품",
                         "신호 시작": None if first is None else first.date().isoformat(),
                         "2000-09보다 늦음": bool(first is None or first > pd.Timestamp(loc.DOTCOM_START))})
    return pd.DataFrame(rows)


def bands(study: Study) -> pd.DataFrame:
    """Daily state of the 14 centre rules on the 2000-09 dates (D2 · D3 on B2), for the time-axis figure."""
    era = "2000-09"
    cols = {r.label: study.raw(era, r.key, r.params[0], "usd", "B2").fillna(0).astype(int) for r in K.RULES.values()}
    return pd.DataFrame(cols, index=study.sims[era, 10].dates)


# --------------------------------------------------------------------------- self-checks

def engine_check(study: Study) -> float:
    """DynamicSim against run_allocation (static) and against aa_engine.simulate (three signal rows)."""
    worst = 0.0
    for era, start in ERAS.items():
        sim = study.sims[era, 10]
        named = {**{n: C.CENTRES[n] for n in C.NAMES}, "ref": PRESETS[REFERENCE].weights}
        W = np.vstack([study.vector(w) for w in named.values()])
        paths, turn, _ = sim.run(*K.static_rows(W, sim))
        for k, w in enumerate(named.values()):
            res = run_allocation(study.krw[era], AllocConfig(f"check{k}", {s: float(x) for s, x in w.items()}, start, sr.END,
                                                             rebalance=G.REBALANCE, cost_bps=G.COST_BPS))
            worst = max(worst, float(np.max(np.abs(paths[k] / res.equity.to_numpy() - 1.0))),
                        abs(turn[k] - res.turnover) / max(res.turnover, 1e-12))
        close = study.krw[era].close.loc[:sim.dates[-1], study.syms].ffill()
        rets = close.pct_change(fill_method=None).loc[sim.dates]
        for s in (spec(era, "B4", "k4", "safe"), spec(era, "B2", "k1", "defensive", "1개월 금지"), spec(era, "B3", "d3", "tilt_es")):
            tab, sched, _ = study.build(s)
            p, t, _ = sim.run(tab[None], sched.decide[None], sched.choice[None], np.array([sched.first]))
            targets = {int(i): dict(zip(study.columns, tab[sched.choice[i]])) for i in np.flatnonzero(sched.decide)}
            eq, _, t_eng, _ = simulate(rets, sim.cash, targets, 10, 0.0, first=dict(zip(study.columns, tab[sched.first])))
            worst = max(worst, float(np.max(np.abs(p[0] / eq.to_numpy() - 1.0))), abs(t[0] - t_eng) / max(t_eng, 1e-12))
    print(f"엔진 대조: 정적 {len(C.NAMES) + 1}개 · 신호 3개 × 2기간, 최대 상대 오차 {worst:.2e}")
    if worst > 1e-6:
        raise SystemExit("시뮬레이터가 엔진과 다릅니다. 멈춥니다.")
    return worst


# --------------------------------------------------------------------------- figures

ERA_LABEL = {"2005": "2005–2026", "2000-09": "2000-09–2026 (닷컴 포함)"}
ACTION_COLOR = {"defensive": figs.COLORS[0], "safe": figs.COLORS[1], "tilt_nd": figs.COLORS[2], "tilt_es": figs.COLORS[3]}


def figures(out: Path, v: pd.DataFrame, static: pd.DataFrame, a38: pd.DataFrame, ps: pd.DataFrame, band: pd.DataFrame,
            runs: pd.DataFrame, krw_static: pd.DataFrame) -> list[str]:
    if not figs.available():
        print("matplotlib 없음: 그림을 건너뜁니다.")
        return []
    plt = figs._plt()
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D
    fdir = out / "figures"
    made = []

    # 1 every signal × action against the static B, per allocation and period
    fig = plt.figure(figsize=(15, 8.6))
    gs = fig.add_gridspec(3, len(C.NAMES), height_ratios=[1, 1, 0.17])
    axes = np.array([[fig.add_subplot(gs[i, j]) for j in range(len(C.NAMES))] for i in range(2)])
    for i, era in enumerate(ERAS):
        for j, name in enumerate(C.NAMES):
            ax = axes[i, j]
            d = v[v["배분"] == name]
            for version, filled in (("바로 매매", True), ("1개월 금지", False)):
                e = d[d["판"] == version]
                for action, color in ACTION_COLOR.items():
                    f = e[e["행동"] == action]
                    if len(f):
                        ax.scatter(100 * f[f"mdd_meas_{era}"], 100 * f[f"cagr_{era}"], s=26,
                                   facecolors=color if filled else "none", edgecolors=color, lw=1.0,
                                   label=f"{K.ACTIONS[action]} · {version}" if (i, j) == (0, 0) else None)
            for k, r in d[d["판정"] == "후보"].iterrows():
                ax.annotate(f"{r['신호']} {r['판']}", (100 * r[f"mdd_meas_{era}"], 100 * r[f"cagr_{era}"]), fontsize=7,
                            xytext=(3, 3), textcoords="offset points")
            s = static[(static["기간"] == era) & (static["배분"] == name)].iloc[0]
            ax.scatter(100 * s["mdd"], 100 * s["cagr"], marker="*", s=160, color="#111", zorder=4, label="원래 B (정적)" if (i, j) == (0, 0) else None)
            a = a38[(a38["기간"] == era) & (a38["배분"] == name)].iloc[0]
            ax.scatter(100 * a["mdd"], 100 * a["cagr"], marker="^", s=70, color="#718096", zorder=4, label="A38 주식 10개월선" if (i, j) == (0, 0) else None)
            ax.axvline(100 * sr.MDD_LINE, color="#9b2c2c", lw=0.8, ls="--")
            ax.grid(alpha=0.3)
            ax.set_title(f"{name} · {ERA_LABEL[era]}", fontsize=9)
            if i == 1:
                ax.set_xlabel("MDD (%, 신호 시작부터)")
            if j == 0:
                ax.set_ylabel("CAGR (%)")
    leg = fig.add_subplot(gs[2, :])
    leg.axis("off")
    leg.legend(*axes[0, 0].get_legend_handles_labels(), loc="center", ncol=5, fontsize=8, frameon=False)
    made.append(figs._save(fig, fdir / "1_signals.png").name)

    # 2 placebo: the real MDD against the 5–95 % range of its 200 placebo signals (2000-09, 바로 매매)
    p = ps[ps["기간"] == "2000-09"]
    labels = [f"{K.RULES[r].label} · {K.ACTIONS[a]}" for r, a in pairs()]
    fig, axes = plt.subplots(1, len(C.NAMES), figsize=(15, 0.28 * len(labels) + 1.8), sharey=True, squeeze=False)
    for j, name in enumerate(C.NAMES):
        ax = axes[0, j]
        d = p[p["배분"] == name].set_index(["신호", "행동"]).reindex(pairs())
        y = np.arange(len(d))
        ax.hlines(y, 100 * d["가짜 5%"], 100 * d["가짜 95%"], color="#a0aec0", lw=3)
        ax.scatter(100 * d["가짜 중앙"], y, marker="|", color="#4a5568", s=60)
        good = d["가짜보다 얕은 비율"] >= PLACEBO_SHARE
        ax.scatter(100 * d["진짜 MDD"], y, color=np.where(good, figs.COLORS[2], figs.COLORS[3]), s=22, zorder=3)
        ax.axvline(100 * sr.MDD_LINE, color="#9b2c2c", lw=0.8, ls="--")
        ax.set_yticks(y, labels, fontsize=7)
        ax.invert_yaxis()
        ax.grid(axis="x", alpha=0.3)
        ax.set_title(f"{name}: 진짜 신호(점, 초록 = 가짜 90% 이상보다 얕음)\n가짜 신호 5–95%(막대)·중앙(눈금)", fontsize=9)
        ax.set_xlabel("2000-09–2026 MDD (%)")
    made.append(figs._save(fig, fdir / "2_placebo.png").name)

    # 3 when each signal warns, against the drawdowns of the static allocations
    fig, (top, bot) = plt.subplots(2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [1, 2.2]})
    for k, name in enumerate(C.NAMES):
        eq = krw_static[name]
        top.plot(eq.index, 100 * (eq / eq.cummax() - 1), lw=0.9, color=figs.COLORS[k], label=name)
    top.axhline(100 * sr.MDD_LINE, color="#9b2c2c", lw=0.8, ls="--")
    top.set_ylabel("정적 B 낙폭 (%)")
    top.legend(fontsize=8, ncol=4, loc="lower left")
    top.grid(alpha=0.3)
    M = band.to_numpy().T.astype(float)
    x = band.index
    bot.imshow(M, aspect="auto", interpolation="nearest", cmap=ListedColormap(["#ffffff", "#f6ad55", "#c53030"]), vmin=0, vmax=2,
               extent=(x[0], x[-1], len(band.columns) - 0.5, -0.5))
    bot.set_yticks(range(len(band.columns)), list(band.columns), fontsize=8)
    bot.set_title("위험 신호가 켜진 날 (주황 = 켜짐 또는 절반, 빨강 = K1 둘 다 / D2·D3 반대쪽, D2·D3은 B2 기준)", fontsize=9)
    made.append(figs._save(fig, fdir / "3_signal_bands.png").name)

    # 4 currency: K1 · K3 · K4 on USD prices against KRW prices
    cur = runs[(runs["종류"] == "신호") & (runs["설정"] == "중심") & (runs["비용"] == 10) & (runs["판"] == "바로 매매")
               & (runs["날짜 이동"] == 0) & runs["신호"].isin(K.CURRENCY_RULES)]
    w = cur.pivot_table(index=["기간", "배분", "신호", "행동"], columns="통화", values="mdd_meas").reset_index()
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    for ax, era in zip(axes, ERAS):
        d = w[w["기간"] == era]
        handles = []
        for rule, marker in zip(K.CURRENCY_RULES, ("o", "s", "D")):
            f = d[d["신호"] == rule]
            ax.scatter(100 * f["usd"], 100 * f["krw"], marker=marker, s=34, color=[ACTION_COLOR[a] for a in f["행동"]])
            handles.append(Line2D([], [], marker=marker, ls="", color="#4a5568", label=K.RULES[rule].label))
        for action in ("defensive", "safe"):
            handles.append(Line2D([], [], marker="o", ls="", color=ACTION_COLOR[action], label=K.ACTIONS[action]))
        lo = 100 * min(d["usd"].min(), d["krw"].min()) - 1
        ax.plot([lo, 0], [lo, 0], color="#444", lw=0.8)
        ax.set_xlabel("달러 가격 신호 MDD (%)")
        ax.set_ylabel("원화 환산 신호 MDD (%)")
        ax.set_title(f"{ERA_LABEL[era]}: 대각선 위 = 원화 판이 얕음", fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(handles=handles, fontsize=8)
    made.append(figs._save(fig, fdir / "4_currency.png").name)
    return made


# --------------------------------------------------------------------------- main

def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--prices", choices=["real", "model"], default="real")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args(argv)
    P = QUICK if args.quick else FULL
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    krw = {era: LOADERS[era](args.prices) for era in ERAS}
    can = load_canaries()
    missing = [s for s in NEEDED if s not in can["usd"].columns or can["usd"][s].notna().sum() == 0]
    if missing:
        raise SystemExit(f"신호 자산 가격이 없습니다: {missing}")
    can["chains"].to_csv(out / "chains.csv", index=False)
    study = Study(krw, can)
    err = engine_check(study)
    rules = rules_table(study)
    rules.to_csv(out / "rules.csv", index=False)

    st_specs = static_specs(study)
    static = study.run(st_specs, "정적")
    a38 = a38_rows(krw)
    comp = pd.concat([static.assign(종류="정적 분기"), a38], ignore_index=True)
    comp.to_csv(out / "comparators.csv", index=False)

    specs = main_specs()
    print(f"신호 규칙: {len(specs):,}개 경로")
    runs = study.run(specs, "신호")
    offs_specs = offset_specs(runs, P["offsets"])
    print(f"날짜 이동: 기준 1·2를 통과한 줄 → {len(offs_specs):,}개")
    offs = study.run(offs_specs, "날짜 이동") if offs_specs else None
    eq_specs, eq_x = equal_cost(study, runs)
    eq = study.run(eq_specs, "같은 비용 정적")
    pl_specs = placebo_specs(study, P["placebos"])
    print(f"가짜 신호: {len(pl_specs):,}개")
    plac = study.run(pl_specs, "가짜 신호")

    all_runs = pd.concat([runs, offs] if offs is not None else [runs], ignore_index=True)
    all_runs.to_csv(out / "runs.csv.gz", index=False)
    eq.to_csv(out / "equal_cost.csv", index=False)
    plac[["기간", "배분", "신호", "행동", "가짜 번호", "밀기", "mdd_meas", "cagr"]].to_csv(out / "placebo.csv.gz", index=False)
    ps = placebo_summary(plac, runs)
    ps.to_csv(out / "placebo_summary.csv", index=False)
    v = verdicts(runs, offs, plac, eq, eq_x)
    v.to_csv(out / "verdicts.csv", index=False)
    band = bands(study)
    band.to_csv(out / "bands.csv.gz")

    sim = study.sims["2000-09", 10]
    W = np.vstack([study.vector(C.CENTRES[n]) for n in C.NAMES])
    p, _, _ = sim.run(*K.static_rows(W, sim))
    krw_static = pd.DataFrame(p.T, index=sim.dates, columns=list(C.NAMES))
    made = figures(out, v, static, a38, ps, band, runs, krw_static)

    summary = {
        "prices": args.prices, "quick": bool(args.quick), "placebos": P["placebos"], "offsets": list(P["offsets"]),
        "paths": {"신호": len(specs), "날짜 이동": len(offs_specs), "같은 비용": len(eq_specs), "가짜": len(pl_specs), "정적": len(st_specs)},
        "engine_check_max_rel": err, "combos": int(len(v)), "candidates": int((v["판정"] == "후보").sum()),
        "pass_counts": {c: int(v[c].sum()) for c in v.columns if c[:1].isdigit() and c[1:2] == " "},
        "failed_tickers": list(can.get("failed") or []), "figures": made, "minutes": round((time.time() - t0) / 60, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    main()
