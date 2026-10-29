"""Stage L: the same buy sets under many exits (DECISIONS 90).

    python -m lab.exit_study --panel DIR --eligible FILE --stage-h DIR --stage-i DIR --v2 DIR --events DIR
                             --r0x1 DIR --out DIR [--stage 1|2|all] [--workers N] [--table-workers N] [--dry]

Stage 1 (engine): per-signal tables (stagec_signals_v2 format) of the 15 buy rules at their middle
value and of the random buys E0 (seeds 1-5) under the new exits PS, LB, MA, RS, MN, ST, and of E0
under CH. The other exits reuse stored tables (X1, H1 k3, H9 k3; CH = Stage C X2 at k 3 for the 15
rules) and TP is H1 k3 with a half sale at +50% added from the prices (as the engine's
take_profit_pct). Stage 2: singles, pairs and triples (lab/pair_study.py's latest-signal rows), E0
thinned to 8 frequency bins and pseudo-triples, each under all 11 exits: 10-stock 1%-rule accounts
(100 random orders), no-limit contributions, per-signal return histograms and tests; verdicts and
chance counts per exit, candidates over exits, exit-character tables and figures.
Output (aggregates only): tables/ (stage 1), sets/*.csv, verdicts.csv, stats.csv, hist.csv,
unlimited.csv, accounts_quantiles.csv, chance.json, exit_character.csv, singles_by_exit.csv,
baselines.csv, figures/.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import entry_study as es, event_study as ev, figures, pair_study as pj, pick_study as ph, stage_c_study as sc, tail_study as ts
from .blocks import atr
from .crsp_study import SAMPLES, load_panel_with_cash
from .engine import EXIT_BLOCKS

# ---- registered (DECISIONS 90): fixed before any result
# key -> label, stored table ('new' = built in stage 1), engine exit block (new tables), extra ATR x3
# stop, 1%-rule stop (ATR kind, k; None = fixed 10% weight), partial take-profit (+X%, fraction)
EXITS_L = {
    "X1": {"label": "25% 트레일링만", "table": "x1", "stop": None},
    "H1": {"label": "25% 트레일링 + ATR(14)×3 손절", "table": "h1", "stop": ("atr14", 3.0)},
    "CH": {"label": "샹들리에 (최고 종가 − ATR(22)×3)", "table": "x2", "stop": ("atr22", 3.0)},
    "PS": {"label": "파라볼릭 SAR 반전 + ATR×3", "table": "new", "block": ("psar_flip", 0.02), "extra": True, "stop": ("atr14", 3.0)},
    "LB": {"label": "20일 신저가 이탈 + ATR×3", "table": "new", "block": ("low_break", 20), "extra": True, "stop": ("atr14", 3.0)},
    "MA": {"label": "50일선 이탈 + ATR×3", "table": "new", "block": ("below_ma", 50), "extra": True, "stop": ("atr14", 3.0)},
    "H9": {"label": "ATR×3 + 2R 손익분기 + 60일 시간 손절", "table": "x9", "stop": ("atr14", 3.0)},
    "TP": {"label": "+50%에 절반 매도, 나머지 H1", "table": "h1", "stop": ("atr14", 3.0), "partial": (50.0, 0.5)},
    "ST": {"label": "넓은 손절만 (ATR×5)", "table": "new", "block": ("atr_stop", 5.0), "extra": False, "stop": ("atr14", 5.0)},
    "RS": {"label": "단기 반등 청산 (종가 > SMA5 또는 RSI2 > 90) + ATR×3", "table": "new", "block": ("rsi2_exit", 90), "extra": True,
           "stop": ("atr14", 3.0)},
    "MN": {"label": "126일 수익률 마이너스 + ATR×3", "table": "new", "block": ("momentum_neg", 126), "extra": True, "stop": ("atr14", 3.0)},
}
EXTRA_STOP = {"atr_stop": 3.0}
H_KI = 2  # H1 / H9 tables at k = 3 (pick_study.K_VALUES)
MIN_EXITS = 3  # a candidate passes the market test under at least this many exits
N_PLACEBO_PER_BIN = 40
N_TRIALS_BEFORE = pj.N_TRIALS_BEFORE + pj.N_NEW + pj.N_NEW_J3  # Stages B-J3 (approx.)
N_NEW = 4_600  # (15 + 105 + 455) sets x 8 new exits
SEED = 90
_W: dict = {}


def new_exits() -> list[str]:
    return [x for x, d in EXITS_L.items() if d["table"] == "new"]


# --------------------------------------------------------------------------- stage 1: engine tables

def use_table_menu(keys: list[str], exits: list[str], seeds: list[int]) -> None:
    """Stage C's signals-only runner on the given buy rules (middle value; 'R0' = random buys of
    the seeds) and Stage L exits (one value each, + ATR x3 stop where registered)."""
    sc.ENTRIES.clear()
    for e in keys:
        if e == "R0":
            sc.ENTRIES["R0"] = ("g_random", list(seeds))
        else:
            key, vals = ph._C_ENTRIES[e] if e in ph._C_ENTRIES else ev.EVENTS[e]
            sc.ENTRIES[e] = (key, [vals[ph._C_CANON.get(e, 1)]])
        sc.ENTRY_CANON[e] = 0
    sc.EXITS.clear()
    for x in exits:
        block, value = EXITS_L[x]["block"] if x != "CH" else ("chandelier", 3.0)
        sc.EXITS[x] = (block, [value])
        sc.EXIT_CANON[x] = 0
        if EXITS_L[x].get("extra"):
            sc.EXIT_EXTRA[x] = dict(EXTRA_STOP)
    sc.SAMPLES = {"full": SAMPLES["full"]}


def _t_init(panel_dir: str, eligible: str, keys: list[str], exits: list[str], seeds: list[int]):
    use_table_menu(keys, exits, seeds)
    sc._setup(load_panel_with_cash(Path(panel_dir)), pd.read_parquet(eligible))


def _timed_signals(job):
    t0 = time.time()
    job, sig = sc._run_signals(job)
    return job, sig, time.time() - t0


def build_tables(out: Path, panel: str, eligible: str, keys: list[str], exits: list[str], seeds: list[int], workers: int) -> None:
    """Engine tables in out/signals/full/<buy>__<exit>_signals.parquet (finished files skipped) and
    out/timing.csv (seconds and signals per table)."""
    use_table_menu(keys, exits, seeds)
    jobs = [("full", e, x) for e in keys for x in exits if not sc.signals_file(out, "full", e, x).exists()]
    ts.log(f"엔진 신호표 {len(jobs)}개 남음 (매수 {len(keys)} × 매도 {len(exits)})")
    if not jobs:
        return
    t0 = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=workers, initializer=_t_init, initargs=(panel, eligible, keys, exits, seeds)) as ex:
        futs = [ex.submit(_timed_signals, j) for j in jobs]
        for n_done, f in enumerate(as_completed(futs), 1):
            job, sig, sec = f.result()
            dst = sc.signals_file(out, *job)
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_suffix(".tmp")
            sig.astype({"return": "float32", "holding_days": "int32"}).to_parquet(tmp, compression="zstd")
            tmp.replace(dst)
            rows.append({"entry": job[1], "exit": job[2], "signals": len(sig), "seconds": sec})
            ts.log(f"  [{n_done}/{len(jobs)}] {job[1]}×{job[2]}: 신호 {len(sig):,} · {sec:.0f}s · 경과 {(time.time() - t0) / 60:.0f}분")
    t = pd.DataFrame(rows)
    f = out / "timing.csv"
    (pd.concat([pd.read_csv(f), t]) if f.exists() else t).to_csv(f, index=False)


# --------------------------------------------------------------------------- stage 2: tables, weights

def table_of(e: str, x: str) -> tuple[Path, int, int]:
    """(signal table, entry cell, exit cell) of rule e or E0 seed 'R0s<k>' under Stage L exit x ('H1k2'
    = Stage I's E0 x H1 k 2, the base the frequency bins are thinned from, as in Stage J)."""
    d = _W["dirs"]
    if e.startswith("R0s"):
        si = pj.E0_SEEDS.index(int(e[3:]))
        if x == "H1k2":
            return d["stage_i"] / "r0" / "signals" / "full" / "R0__H1_signals.parquet", si, 1
        src = EXITS_L[x]["table"]
        if src in ("h1", "x9"):
            return d["stage_i"] / "r0" / "signals" / "full" / f"R0__{'H1' if src == 'h1' else 'X9'}_signals.parquet", si, H_KI
        if src == "x1":
            return d["r0x1"] / "signals" / "full" / "R0__X1_signals.parquet", si, 0
        return d["tables"] / "signals" / "full" / f"R0__{x}_signals.parquet", si, 0
    src = EXITS_L[x]["table"]
    if src == "h1":
        return d["stage_h"] / "h1" / "signals" / "full" / f"{e}__H1_signals.parquet", 0, H_KI
    if src == "new":
        return d["tables"] / "signals" / "full" / f"{e}__{x}_signals.parquet", 0, 0
    root = d["v2"] if e.startswith("E") else d["events"]
    stage_c = {"x1": ("X1", 1), "x2": ("X2", 1), "x9": ("X9", H_KI)}[src]
    return root / "signals" / "full" / f"{e}__{stage_c[0]}_signals.parquet", pj._C_CANON.get(e, 1), stage_c[1]


def partial_sales(C: np.ndarray, O: np.ndarray, ei: np.ndarray, sj: np.ndarray, f_excl: np.ndarray, last: int,
                  up: float) -> np.ndarray:
    """Day of the partial sale (panel index, -1 if none) of each position, as the engine's
    take_profit_pct: the first close >= entry open x (1 + up) from the entry day up to (not
    including) the full exit's signal day f; sold at the next open with a price, which must come no
    later than f (on f + 1 the full sale goes first) or, for a position still open, the window end."""
    out = np.full(len(ei), -1, dtype=np.int64)
    for r in range(len(ei)):
        e, j, f = int(ei[r]), int(sj[r]), int(f_excl[r])
        entry = O[e, j]
        if not entry > 0:
            continue
        c = C[e:f, j]
        hit = np.flatnonzero(c >= entry * (1 + up))  # NaN closes compare False
        if not len(hit):
            continue
        d = e + int(hit[0])
        lim = min(f, last)
        o = O[d + 1:lim + 1, j]
        ok = np.flatnonzero(o == o)
        if len(ok):
            out[r] = d + 1 + int(ok[0])
    return out


def table_post(s: pd.DataFrame, e: str, x: str) -> pd.DataFrame:
    """TP: the stored H1 k3 rows with the half sale at +50% (sale date and combined return)."""
    if x not in EXITS_L or not EXITS_L[x].get("partial"):
        return s
    up, frac = EXITS_L[x]["partial"]
    px = _W["px"]
    ei = px.dates.get_indexer(pd.DatetimeIndex(s["entry_date"]))
    sj = px.cols.get_indexer(s["symbol"].astype(str))
    f = px.dates.get_indexer(pd.DatetimeIndex(s["exit_signal_date"]))
    last = _W["last"]
    f_excl = np.where(f >= 0, f, last + 1)
    ps = partial_sales(_W["C"], px.O, ei, sj, f_excl, last, up / 100.0)
    c = sc.COMMON["cost_bps"] / 10_000
    entry = px.O[ei, sj].astype(float)
    r_p = np.where(ps >= 0, px.O[np.maximum(ps, 0), sj].astype(float) * (1 - c) / (entry * (1 + c)) - 1.0, np.nan)
    s = s.copy()
    full = s["return"].to_numpy(dtype=float)
    s["return"] = np.where(ps >= 0, frac * r_p + (1 - frac) * full, full)
    s["psale_date"] = px.dates[np.maximum(ps, 0)].where(ps >= 0)
    return s


def _init(panel_dir: str, eligible: str, dirs: dict, cfg: dict):
    pj.BINS, pj.E0_SEEDS = list(cfg["bins"]), list(cfg["seeds"])
    panel = load_panel_with_cash(Path(panel_dir))
    elig = pd.read_parquet(eligible)
    px = ts.Prices(panel, elig)
    idx = panel.close.index
    win = (idx >= pd.Timestamp(es.START)) & (idx <= pd.Timestamp(es.END))
    w0, n = int(np.flatnonzero(win)[0]), int(win.sum())
    common = dict(px=px, w0=w0, n=n, wdates=idx[win], cash=panel.cash.reindex(idx[win]).fillna(0.0).to_numpy(),
                  O=px.O[w0:w0 + n].astype(float), LC=px.LC[w0:w0 + n].astype(float))
    pj._W.update(common, dirs=dirs, tables={}, refs={}, table_of=table_of, table_post=table_post)
    _W.update(common, dirs=dirs, ATR22=atr(panel, 22).to_numpy(dtype="float32"), C=panel.close.to_numpy(dtype="float32"),
              last=int(idx.searchsorted(pd.Timestamp(es.END), side="right")) - 1, refs={})


def signals_l(spec: tuple, x: str) -> pd.DataFrame:
    if spec[0] == "e0full":  # all of E0's 1% random buys of one seed
        return pj.rule_table(f"R0s{spec[1]}", x)
    return pj.signals_for(spec, x)


def weights_l(m: pd.DataFrame, x: str) -> pd.DataFrame:
    """1%-rule weight (10% cap) from the registered stop, or a fixed 10% without a stop."""
    stop = EXITS_L[x]["stop"]
    if stop is None:
        w = pd.DataFrame({"w": np.full(len(m), ph.CAP), "capped": True})
        w["contrib"] = w["w"] * m["ret"].to_numpy()
        return w
    kind, k = stop
    if kind == "atr14":
        return ph.weights(_W["px"], m, k)
    a = _W["ATR22"][m["si"].to_numpy(), m["sj"].to_numpy()].astype(float)
    entry = _W["px"].O[m["ei"].to_numpy(), m["sj"].to_numpy()].astype(float)
    stop_pct = k * a / entry
    raw = np.where(stop_pct > 0, ph.RISK / stop_pct, np.nan)
    w = pd.DataFrame({"w": np.minimum(raw, ph.CAP), "capped": raw >= ph.CAP})
    w["contrib"] = w["w"] * m["ret"].to_numpy()
    return w


def prepared_l(spec: tuple, x: str) -> pd.DataFrame:
    """The set's filled signals inside the window with weight, contribution, return and (TP) the
    window day of the partial sale."""
    px, w0, n = _W["px"], _W["w0"], _W["n"]
    s = signals_l(spec, x)
    m = px.measures(s, es.END)
    w = weights_l(m, x)
    ps = (px.dates.get_indexer(pd.DatetimeIndex(s["psale_date"])) if "psale_date" in s else np.full(len(s), -1))
    d = pd.concat([m[["si", "ei", "sj", "sale", "closed", "x_spy", "ret"]], w[["w", "contrib"]],
                   s[["signal_date", "holding_days", "exit_reason"]]], axis=1)
    d["psale"] = np.where(ps >= 0, ps - w0, -1)
    ok = (d["ei"].to_numpy() >= w0) & (d["ei"].to_numpy() < w0 + n) & np.isfinite(d["w"].to_numpy())
    return d[ok].reset_index(drop=True)


def e0_reference(bin_: float, x: str, without: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    key = (bin_, x, without)
    if key not in _W["refs"]:
        r = np.concatenate([prepared_l(("e0", s, bin_, 0), x)["ret"].to_numpy() for s in pj.E0_SEEDS if s != without])
        r = r[np.isfinite(r)]
        rng = np.random.default_rng([SEED, 3, int(round(bin_ * 1e7)), list(EXITS_L).index(x), without or 0])
        _W["refs"][key] = (r, pj.boot_means(r, rng))
    return _W["refs"][key]


STOP_LABEL = EXIT_BLOCKS["atr_stop"].label


def run_set(job):
    spec, n_orders, cost_bps, exits = job
    w0, n, wdates = _W["w0"], _W["n"], _W["wdates"]
    name = pj.set_name(spec)
    accs, uns, hists = [], [], []
    for x in exits:
        d = prepared_l(spec, x)
        ei, sale = d["ei"].to_numpy() - w0, d["sale"].to_numpy() - w0
        closed = d["closed"].to_numpy(dtype=bool) & (sale < n)
        psale = d["psale"].to_numpy() if EXITS_L[x].get("partial") else None
        frac = EXITS_L[x]["partial"][1] if psale is not None else 0.5
        rng = np.random.default_rng([SEED, zlib.crc32(f"{name}|{x}|{cost_bps}".encode())])
        for o in range(n_orders):
            eq, tr = ph.replay_risk(ei - 1, ei, d["sj"].to_numpy(), sale, closed, d["w"].to_numpy(), rng.random(len(d)),
                                    _W["O"], _W["LC"], _W["cash"], cost_bps / 10_000, psale=psale, pfrac=frac)
            accs.append({"set": name, "kind": spec[0], "exit": x, "order": o, "trades": len(tr),
                         **es.period_stats(pd.Series(eq, index=wdates))})
        yr = pd.DatetimeIndex(d["signal_date"]).year.to_numpy()
        ret = d["ret"].to_numpy(dtype=float)
        cl = d["closed"].to_numpy(dtype=bool)
        un = {"set": name, "kind": spec[0], "exit": x, "signals": len(d), "freq": len(d) / pj.STOCK_DAYS, "days": n,
              "hold_median": float(np.median(d["holding_days"].to_numpy()[cl])) if cl.any() else np.nan,
              "stop_share": float((d["exit_reason"].to_numpy()[cl] == STOP_LABEL).mean()) if cl.any() else np.nan,
              "loss_share": float((ret[cl] < 0).mean()) if cl.any() else np.nan,
              "big_contrib_share": float((d["contrib"].to_numpy() >= ph.BIG_CONTRIB).mean()) if len(d) else np.nan,
              "partial_share": float((d["psale"].to_numpy() >= 0).mean()) if psale is not None and len(d) else np.nan}
        for p, (a, b) in es.PERIODS.items():
            mm = (yr >= a) & (yr <= b)
            r = ret[mm & np.isfinite(ret)]
            un[f"n_{p}"] = int(mm.sum())
            un[f"mean_contrib_{p}"] = float(d["contrib"].to_numpy()[mm].mean()) if mm.any() else np.nan
            un[f"mean_x_spy_{p}"] = float(d["x_spy"].to_numpy()[mm].mean()) if mm.any() else np.nan
            un[f"mean_ret_{p}"] = float(r.mean()) if len(r) else np.nan
            un[f"median_ret_{p}"] = float(np.median(r)) if len(r) else np.nan
            un[f"win_{p}"] = float((r > 0).mean()) if len(r) else np.nan
            un[f"big50_{p}"] = float((r >= 0.5).mean()) if len(r) else np.nan
            un[f"loss20_{p}"] = float((r <= -0.2).mean()) if len(r) else np.nan
            if p in pj.HIST_PERIODS:
                hists.append({"set": name, "kind": spec[0], "exit": x, "period": p,
                              **{f"h{i}": int(c) for i, c in enumerate(np.histogram(r, pj.RET_EDGES)[0])}})
        if spec[0] not in ("e0", "e0full") and cost_bps == pj.COST_BPS:
            b = pj.nearest_bin(len(d) / pj.STOCK_DAYS)
            ref, ref_boot = e0_reference(b, x, spec[1] if spec[0] == "placebo" else None)
            rng = np.random.default_rng([SEED, zlib.crc32(f"{name}|{x}|boot".encode())])
            un.update({"st_bin": b, **pj.ret_stats(ret[np.isfinite(ret)], ref, ref_boot, rng)})
        uns.append(un)
    return spec, pd.DataFrame(accs), pd.DataFrame(uns), pd.DataFrame(hists)


def run_sets(todo: list, sets_dir: Path, a, suffix: str = "") -> None:
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.panel, a.eligible, a.dirs, a.cfg)) as ex:
        for i, (spec, acc, un, hist) in enumerate(ex.map(run_set, todo, chunksize=1)):
            nm = pj.set_name(spec) + suffix
            acc.to_csv(sets_dir / f"{nm}_acc.csv", index=False)
            un.to_csv(sets_dir / f"{nm}_un.csv", index=False)
            hist.to_csv(sets_dir / f"{nm}_hist.csv", index=False)
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                ts.log(f"  [{i + 1}/{len(todo)}] 남은 예상 {(len(todo) - i - 1) * (time.time() - t0) / (i + 1) / 60:.0f}분")


# --------------------------------------------------------------------------- verdicts per exit

def chance_by_exit(v: pd.DataFrame, kinds=("triple", "pair", "single")) -> dict:
    """Per exit: pass counts of the sets with >= 30 trades in (a) and (b) (both halves) vs the
    pseudo-triples' pass rate p0 and the 95th percentile of Binomial(N, p0)."""
    out = {}
    for x, g in v.groupby("exit", sort=False):
        pl = g[g["kind"] == "placebo"]
        out[x] = {}
        for kind in kinds:
            s = g[(g["kind"] == kind) & (g["trades"] >= pj.MIN_TRADES)]
            n = len(s)
            row = {"n_judged": n, "excluded_few_trades": int(((g["kind"] == kind) & (g["trades"] < pj.MIN_TRADES)).sum())}
            for k, col in (("a", "a_halves"), ("b", "b_halves")):
                p0 = float(pl[col].mean()) if len(pl) else np.nan
                obs = int(s[col].sum())
                cdf, q95 = 0.0, n
                for c in range(n + 1):
                    cdf += math.comb(n, c) * p0 ** c * (1 - p0) ** (n - c)
                    if cdf >= 0.95:
                        q95 = c
                        break
                row[k] = {"observed": obs, "p0": p0, "expected": p0 * n, "q95": q95, "information": obs > q95}
            out[x][kind] = row
    return out


def market_pass(v: pd.DataFrame, base: pd.DataFrame) -> pd.Series:
    spy = base.set_index("baseline").loc["SPY"]
    ok = pd.Series(True, index=v.index)
    for p in ("1993-2023", "2017-2023"):
        ok &= (v[f"cagr_{p}"] > spy[f"cagr_{p}"]) & (v[f"mdd_{p}"].abs() < abs(spy[f"mdd_{p}"]))
    return ok & (v["trades"] >= pj.MIN_TRADES)


def exit_character(v: pd.DataFrame, un: pd.DataFrame, acc: pd.DataFrame) -> pd.DataFrame:
    """Per exit, E0 (all 1% random buys, 5 seeds): 10-stock accounts (median over 500) and signal
    statistics pooled over the seeds; with the median over the 15 singles of their 10-stock CAGR."""
    rows = []
    for x in EXITS_L:
        a = acc[(acc["kind"] == "e0full") & (acc["exit"] == x)]
        u = un[(un["kind"] == "e0full") & (un["exit"] == x)]
        s = v[(v["kind"] == "single") & (v["exit"] == x)]
        nn = u["n_1993-2023"]
        row = {"exit": x, "label": EXITS_L[x]["label"], "weight": "1% 룰" if EXITS_L[x]["stop"] else "10% 고정"}
        for p in es.PERIODS:
            row[f"e0_cagr_{p}"], row[f"e0_mdd_{p}"] = float(a[f"cagr_{p}"].median()), float(a[f"mdd_{p}"].median())
        row["e0_trades"] = float(a["trades"].median())
        for c in ("hold_median", "stop_share", "loss_share", "big_contrib_share", "partial_share"):
            row[f"e0_{c}"] = float((u[c] * u["signals"]).sum() / u["signals"].sum()) if u[c].notna().any() else np.nan
        for c in ("mean_ret", "median_ret", "win", "big50", "loss20"):
            row[f"e0_{c}"] = float((u[f"{c}_1993-2023"] * nn).sum() / nn.sum())
        for p in ("1993-2023",) + es.HALVES + ("2017-2023",):
            row[f"singles_cagr_median_{p}"] = float(s[f"cagr_{p}"].median())
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures

def _grid(plt, k: int, ncol: int = 4, w: float = 4.2, h: float = 3.2):
    nrow = int(np.ceil(k / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(w * ncol, h * nrow), squeeze=False)
    for ax in axes.ravel()[k:]:
        ax.set_visible(False)
    return fig, axes.ravel()


def fig_e0_returns(out: Path, un: pd.DataFrame, hist: pd.DataFrame) -> Path:
    """① E0's per-signal returns under each exit (1993-2023, 5 seeds pooled)."""
    plt = figures._plt()
    hc = [c for c in hist.columns if c.startswith("h") and c[1:].isdigit()]
    fig, axes = _grid(plt, len(EXITS_L))
    for ax, x in zip(axes, EXITS_L):
        h = hist[(hist["kind"] == "e0full") & (hist["exit"] == x) & (hist["period"] == "1993-2023")][hc].to_numpy().sum(axis=0)
        u = un[(un["kind"] == "e0full") & (un["exit"] == x)]
        nn = u["n_1993-2023"]
        pooled = lambda c: float((u[c] * nn).sum() / nn.sum())  # noqa: E731
        pj._bar(ax, pj._density(h), color=figures.COLORS[0], lw=1.4)
        ax.axvline(0, color="#bbb", lw=0.8)
        ax.set_title(f"{x}: {EXITS_L[x]['label']}", fontsize=8.5)
        ax.text(0.98, 0.97, f"평균 {pooled('mean_ret_1993-2023') * 100:+.1f}%, 승률 {pooled('win_1993-2023') * 100:.0f}%\n"
                            f"+50% 이상 {pooled('big50_1993-2023') * 100:.1f}%, −20% 이하 {pooled('loss20_1993-2023') * 100:.1f}%\n"
                            f"보유 중앙값 {float((u['hold_median'] * u['signals']).sum() / u['signals'].sum()):.0f}일",
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    fig.suptitle("무작위 매수 E0 신호 하나의 수익률 분포, 매도별 (1993–2023, 편도 20bp, 5%p 칸, 양 끝 칸은 −60% 미만 / +200% 초과)", fontsize=11)
    return figures._save(fig, out / "figures" / "e0_returns_by_exit.png")


def fig_excess_by_exit(out: Path, v: pd.DataFrame, h: str) -> Path:
    """② Per exit: 10-stock CAGR in excess of the same-frequency E0 median, triples and pairs vs the
    pseudo-triples, one half."""
    plt = figures._plt()
    fig, axes = _grid(plt, len(EXITS_L))
    for ax, x in zip(axes, EXITS_L):
        g = v[v["exit"] == x]
        t = pj._excess(g[(g["kind"] == "triple") & (g["trades"] >= pj.MIN_TRADES)], h)
        p = pj._excess(g[(g["kind"] == "pair") & (g["trades"] >= pj.MIN_TRADES)], h)
        b = pj._excess(g[g["kind"] == "placebo"], h)
        allx = np.concatenate([t, p, b])
        if not len(allx) or not len(b):
            continue
        edges = np.linspace(np.floor(np.nanmin(allx)) - 0.5, np.ceil(np.nanmax(allx)) + 0.5, 31)
        ax.hist(b, edges, density=True, color="#bbbbbb", label="가짜 묶음")
        if len(t):
            ax.hist(t, edges, density=True, histtype="step", lw=1.6, color=figures.COLORS[0], label="3개 묶음")
        if len(p):
            ax.hist(p, edges, density=True, histtype="step", lw=1.2, color=figures.COLORS[1], label="쌍")
        q = np.percentile(b, 95)
        ax.axvline(q, color="#555", ls="--", lw=1)
        ax.axvline(0, color="#999", lw=0.8)
        _, pv = pj.mann_whitney(t, b) if len(t) else (np.nan, np.nan)
        ax.text(0.02, 0.97, f"묶음 평균 {np.mean(t) if len(t) else np.nan:+.1f}%p (가짜 {b.mean():+.1f})\n"
                            f"95% 선 넘음 {np.mean(t > q) * 100 if len(t) else np.nan:.0f}%, p = {pv:.3f}",
                transform=ax.transAxes, va="top", fontsize=7.5, bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"})
        ax.set_title(f"{x}: {EXITS_L[x]['label']}", fontsize=8.5)
        ax.grid(alpha=0.3)
        ax.tick_params(labelsize=7)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper right", bbox_to_anchor=(0.99, 0.995), fontsize=8, frameon=False)
    fig.suptitle(f"매도별 10종목 CAGR − 같은 빈도 무작위 중앙값 (%p), {h.replace('-', '–')}: 묶음 vs 가짜 묶음 (점선 = 가짜 95%)", fontsize=11)
    return figures._save(fig, out / "figures" / f"excess_by_exit_{h}.png")


def fig_single_heatmap(out: Path, v: pd.DataFrame) -> Path:
    """③ Exit x single rule: 10-stock CAGR in excess of the same-frequency E0 median (pp), two halves."""
    plt = figures._plt()
    rules = pj.RULES
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    s = v[v["kind"] == "single"]
    mats = []
    for h in es.HALVES:
        mat = np.full((len(EXITS_L), len(rules)), np.nan)
        for _, r in s.iterrows():
            e = r["set"].split("__")[1]
            if e in rules and r["exit"] in EXITS_L:
                mat[list(EXITS_L).index(r["exit"]), rules.index(e)] = (r[f"cagr_{h}"] - r[f"e0_cagr_{h}"]) * 100
        mats.append(mat)
    lim = max(1.0, float(np.nanmax(np.abs(np.concatenate([m.ravel() for m in mats])))) if np.isfinite(mats[0]).any() else 1.0)
    for ax, h, mat in zip(axes, es.HALVES, mats):
        im = ax.imshow(mat, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=6,
                            color="white" if abs(mat[i, j]) > 0.6 * lim else "black")
        ax.set_xticks(range(len(rules)), rules, fontsize=7)
        ax.set_yticks(range(len(EXITS_L)), list(EXITS_L), fontsize=8)
        ax.set_title(h.replace("-", "–"), fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8)
    fig.suptitle("매수 규칙 단독 × 매도: 10종목 CAGR − 같은 빈도 무작위 중앙값 (%p), 파랑 = 무작위보다 높음", fontsize=11)
    return figures._save(fig, out / "figures" / "singles_by_exit.png")


def fig_candidates(out: Path, v: pd.DataFrame, un: pd.DataFrame, hist: pd.DataFrame, picks: list[tuple[str, str]], title: str) -> Path:
    """④ Per-signal returns of the picked (set, exit) vs the same-frequency E0."""
    plt = figures._plt()
    hc = [c for c in hist.columns if c.startswith("h") and c[1:].isdigit()]
    fig, axes = plt.subplots(max(len(picks), 1), 1, figsize=(11, 2.9 * max(len(picks), 1)), sharex=True, squeeze=False)
    he = hist[(hist["kind"] == "e0") & (hist["period"] == "1993-2023")]
    he = he.assign(bin=he["set"].str.split("__").str[2].astype(float))
    for ax, (s, x) in zip(axes[:, 0], picks):
        u = un[(un["set"] == s) & (un["exit"] == x)].iloc[0]
        r = v[(v["set"] == s) & (v["exit"] == x)].iloc[0]
        hs = hist[(hist["set"] == s) & (hist["exit"] == x) & (hist["period"] == "1993-2023")][hc].to_numpy().sum(axis=0)
        ref = he[(he["bin"] == r["bin"]) & (he["exit"] == x)][hc].to_numpy().sum(axis=0)
        pj._bar(ax, pj._density(ref), color="#777", lw=1.5, label="같은 빈도 무작위 매수")
        pj._bar(ax, pj._density(hs), color=figures.COLORS[0], lw=1.6, label="이 묶음")
        ax.axvline(0, color="#bbb", lw=0.8)
        st = (f"평균 차 {u['st_diff'] * 100:+.1f}%p (95% CI {u['st_ci_lo'] * 100:+.1f} – {u['st_ci_hi'] * 100:+.1f}), "
              f"순위 검정 p = {u['st_mw_p']:.3f}") if pd.notna(u.get("st_n")) else "신호 10개 미만: 검정 생략"
        ax.set_title(f"{s.split('__', 1)[1].replace('__', ' × ')}, 매도 {x}: 신호 {int(u['signals']):,}개, 평균 {u['mean_ret_1993-2023'] * 100:+.1f}%, "
                     f"10종목 CAGR {r['cagr_1993-2023'] * 100:.1f}% (2017–2023 {r['cagr_2017-2023'] * 100:.1f}%)\n{st}", fontsize=8.5)
        ax.grid(alpha=0.3)
    axes[0, 0].legend(loc="upper right", fontsize=8, frameon=False)
    axes[-1, 0].set_xlabel("신호 하나의 수익률 (%, 1993–2023, 편도 20bp 반영, 양 끝 칸은 −60% 미만 / +200% 초과)")
    fig.suptitle(title, fontsize=11)
    return figures._save(fig, out / "figures" / "candidates_returns.png")


# --------------------------------------------------------------------------- main

def stage2(a) -> None:
    out = Path(a.out)
    kinds = ["single", "pair", "triple"]
    # references first: a run cut short (each set is a checkpoint) still leaves judgeable sets
    specs = ([(("e0", s, b, 0), a.n_orders, pj.COST_BPS, list(EXITS_L)) for s in pj.E0_SEEDS for b in pj.BINS]
             + [(("e0full", s), a.n_orders, pj.COST_BPS, list(EXITS_L)) for s in pj.E0_SEEDS]
             + [(("placebo", pj.E0_SEEDS[d % len(pj.E0_SEEDS)], b, 1 + d), a.p_orders, pj.COST_BPS, list(EXITS_L))
                for b in pj.BINS for d in range(a.n_placebo)]
             + [(("single", r), a.n_orders, pj.COST_BPS, list(EXITS_L)) for r in pj.RULES]
             + [(("pair",) + c, a.n_orders, pj.COST_BPS, list(EXITS_L)) for c in pj.combos(2)]
             + [(("triple",) + c, a.n_orders, pj.COST_BPS, list(EXITS_L)) for c in pj.combos(3)])
    sets_dir = out / "sets"
    sets_dir.mkdir(parents=True, exist_ok=True)
    todo = [j for j in specs if not (sets_dir / f"{pj.set_name(j[0])}_acc.csv").exists()]
    ts.log(f"묶음 {len(todo)}개 남음 × 매도 {len(EXITS_L)}")
    run_sets(todo, sets_dir, a)
    read = lambda pat: pd.concat([pd.read_csv(f) for f in sorted(sets_dir.glob(pat))], ignore_index=True)  # noqa: E731
    acc, un, hist = pj._with_bin(read("*_acc.csv")), pj._with_bin(read("*_un.csv")), read("*_hist.csv")
    _init(a.panel, a.eligible, a.dirs, a.cfg)
    base = es.baselines(_W["px"], _W["wdates"], _W["w0"], _W["cash"])
    base.to_csv(out / "baselines.csv", index=False)
    judged = ~acc["kind"].isin(["e0"])
    v = pj.judge(acc[judged], un[~un["kind"].isin(["e0"])], acc[acc["kind"] == "e0"], un[un["kind"] == "e0"])
    v["kind"] = v["set"].str.split("__").str[0]
    v["market_pass"] = market_pass(v, base) & v["kind"].isin(kinds)
    v["candidate_20bp"] = v["market_pass"] & (v.groupby("set")["market_pass"].transform("sum") >= MIN_EXITS)
    v["candidate_40bp"] = False
    cand = v[v["candidate_20bp"]]
    if len(cand):  # the candidate rows again at 40 bp
        d40 = out / "sets_40bp"
        d40.mkdir(exist_ok=True)
        by_name = {pj.set_name(j[0]): j[0] for j in specs}
        jobs = [(by_name[s], a.n_orders, pj.CHECK_COST_BPS, sorted(g["exit"].tolist())) for s, g in cand.groupby("set")
                if not (d40 / f"{s}_acc.csv").exists()]
        run_sets(jobs, d40, a)
        med40 = pd.concat([pd.read_csv(d40 / f"{s}_acc.csv") for s in cand["set"].unique()]).groupby(["set", "exit"]).median(numeric_only=True)
        spy = base.set_index("baseline").loc["SPY"]
        for i, r in cand.iterrows():
            m = med40.loc[(r["set"], r["exit"])]
            v.loc[i, "candidate_40bp"] = all(m[f"cagr_{p}"] > spy[f"cagr_{p}"] and abs(m[f"mdd_{p}"]) < abs(spy[f"mdd_{p}"])
                                             for p in ("1993-2023", "2017-2023"))
    v["candidate"] = v["candidate_40bp"] & (v.groupby("set")["candidate_40bp"].transform("sum") >= MIN_EXITS)
    v.to_csv(out / "verdicts.csv", index=False)
    acc.groupby(["set", "kind", "exit"]).quantile([0.05, 0.5, 0.95], numeric_only=True).to_csv(out / "accounts_quantiles.csv")
    un.to_csv(out / "unlimited.csv", index=False)
    hist.to_csv(out / "hist.csv", index=False)
    st_cols = ["set", "kind", "exit", "signals", "st_bin"] + [f"{c}_1993-2023" for c in ("mean_ret", "median_ret", "win", "big50", "loss20")] \
        + [c for c in un.columns if c.startswith("st_") and c != "st_bin"]
    st = un[~un["kind"].isin(["e0", "e0full"])].reindex(columns=st_cols)
    st.to_csv(out / "stats.csv", index=False)
    character = exit_character(v, un, acc)
    character.to_csv(out / "exit_character.csv", index=False)
    v[v["kind"] == "single"].to_csv(out / "singles_by_exit.csv", index=False)
    ch = {"by_exit": chance_by_exit(v), "stats_by_exit": {x: pj.stats_summary(g) for x, g in st.groupby("exit", sort=False)},
          "candidates_20bp": sorted(v.loc[v["candidate_20bp"], "set"].unique().tolist()),
          "candidates": sorted(v.loc[v["candidate"], "set"].unique().tolist()),
          "trials": {"before": N_TRIALS_BEFORE, "new": N_NEW}}
    (out / "chance.json").write_text(json.dumps(ch, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    write_figures(out, v, un, hist)
    t = ch["by_exit"]
    ts.log("완료 · 매도별 (b) 통과 묶음 / 우연 95% 분위: " + ", ".join(
        f"{x} {t[x]['triple']['b']['observed']}/{t[x]['triple']['b']['q95']}" for x in t) + f" · 후보 {len(ch['candidates'])}")


def write_figures(out: Path, v: pd.DataFrame, un: pd.DataFrame, hist: pd.DataFrame) -> list[Path]:
    if not figures.available():
        return []
    made = [fig_e0_returns(out, un, hist)] + [fig_excess_by_exit(out, v, h) for h in es.HALVES] + [fig_single_heatmap(out, v)]
    cands = v[v["candidate"]] if "candidate" in v else v.iloc[:0]
    if len(cands):
        picks = [(s, g.sort_values("cagr_1993-2023").iloc[-1]["exit"]) for s, g in cands.groupby("set")][:6]
        title = "후보 묶음의 신호 단위 수익률 (가장 좋은 매도) vs 같은 빈도 무작위"
    else:  # no candidate: the sets passing the market test under the most exits
        mp = v[v["market_pass"]].groupby("set").agg(n=("exit", "size"), cagr=("cagr_1993-2023", "mean"))
        top = mp.sort_values(["n", "cagr"], ascending=False).head(5).index.tolist()
        picks = [(s, v[(v["set"] == s) & v["market_pass"]].sort_values("cagr_1993-2023").iloc[-1]["exit"]) for s in top]
        title = "후보 없음: SPY 기준을 가장 많은 매도에서 통과한 묶음의 신호 단위 수익률 vs 같은 빈도 무작위"
    if picks:
        made.append(fig_candidates(out, v, un, hist, picks, title))
    return made


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("panel", "eligible", "stage_h", "stage_i", "v2", "events", "r0x1", "out"):
        ap.add_argument(f"--{k.replace('_', '-')}", required=True)
    ap.add_argument("--stage", default="all", choices=["1", "2", "all"])
    ap.add_argument("--workers", type=int, default=1, help="2단계 작업 수")
    ap.add_argument("--table-workers", type=int, default=4, help="1단계(엔진) 작업 수: 작업당 메모리 약 20GB")
    ap.add_argument("--dry", action="store_true", help="E2·E9·V3, 씨앗 2, 순서 5, 가짜 구간당 2")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    a.dirs = {"stage_h": Path(a.stage_h), "stage_i": Path(a.stage_i), "v2": Path(a.v2), "events": Path(a.events),
              "r0x1": Path(a.r0x1), "tables": out / "tables"}
    seeds = pj.E0_SEEDS[:2] if a.dry else list(pj.E0_SEEDS)
    if a.dry:
        pj.RULES = ["E2", "E9", "V3"]
    pj.E0_SEEDS = seeds
    a.cfg = {"bins": pj.BINS_J3, "seeds": seeds}
    pj.BINS = list(pj.BINS_J3)
    a.n_orders, a.p_orders = (5, 3) if a.dry else (pj.N_ORDERS, pj.PLACEBO_ORDERS)
    a.n_placebo = 2 if a.dry else N_PLACEBO_PER_BIN
    if a.stage in ("1", "all"):
        build_tables(out / "tables", a.panel, a.eligible, pj.RULES + ["R0"], new_exits(), seeds, a.table_workers)
        build_tables(out / "tables", a.panel, a.eligible, ["R0"], ["CH"], seeds, min(a.table_workers, 1))
    if a.stage in ("2", "all"):
        stage2(a)


if __name__ == "__main__":
    main(sys.argv[1:])
