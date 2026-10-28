"""Stage J: buy rules in pairs, against random buys of the same frequency (DECISIONS 88).

    python -m lab.pair_study --panel DIR --eligible FILE --stage-h DIR --stage-i DIR --v2 DIR --events DIR
                             --out DIR [--workers N] [--dry]

A pair (A, B) buys at the next open after the later of two first-day signals of A and B on the same
stock within PAIR_WINDOW trading days (either order, same day included). The later signal's stored
table row carries the exit, so pairs are built from the saved per-signal tables. Exits: H1 k 2,
H1 k 3 (25% trailing or ATR stop) and X1 (25% trailing only, fixed 10% weight). Baseline: random
buys E0 thinned to the nearest frequency bin; chance count from thinned random pseudo-pairs.
Output (aggregates only): sets/*.csv, accounts.csv, unlimited.csv, verdicts.csv, chance.json,
baselines.csv, figures/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import entry_study as es, event_study as ev, figures, pick_study as ph, stage_c_study as sc, tail_study as ts
from .crsp_study import SAMPLES, load_panel_with_cash
from .engine import CANCELLED

# ---- registered (DECISIONS 88): fixed before any result
RULES = [f"E{i}" for i in range(1, 10)] + list(ev.EVENTS)
PAIR_WINDOW = 10
EXITS_J = {"H1k2": ("H1", 1), "H1k3": ("H1", 2), "X1": ("X1", None)}
BINS = [0.0002, 0.0005, 0.001, 0.002, 0.004]  # share of eligible stock-days
E0_SEEDS = [1, 2, 3, 4, 5]
N_ORDERS = 100
N_PLACEBO_PER_BIN, PLACEBO_ORDERS = 40, 20
MIN_TRADES = 30
STOCK_DAYS = 15_680_749  # hygiene-eligible stock-days 1993-02-01 - 2023-12-31 (Stage I)
COST_BPS, CHECK_COST_BPS = 20.0, 40.0
N_TRIALS_BEFORE = es.N_TRIALS_BEFORE + es.N_NEW  # Stages B-I (approx.)
N_NEW = 315  # 105 pairs x 3 exits (singles, E0 and pseudo-pairs not counted)
SEED = 88
_C_CANON = dict(sc.ENTRY_CANON)
_W: dict = {}


def pairs() -> list[tuple[str, str]]:
    return list(itertools.combinations(RULES, 2))


def set_name(s: tuple) -> str:
    return "__".join(map(str, s))


def nearest_bin(freq: float) -> float:
    return min(BINS, key=lambda b: abs(np.log(max(freq, 1e-9)) - np.log(b)))


# --------------------------------------------------------------------------- E0 x X1 tables

def use_random_x1_menu(dry: bool = False) -> None:
    sc.ENTRIES.clear()
    sc.ENTRIES["R0"] = ("g_random", list(E0_SEEDS if not dry else E0_SEEDS[:2]))
    sc.EXITS.clear()
    sc.EXITS["X1"] = ("trailing_pct", [25])
    sc.SAMPLES = {"full": SAMPLES["full"]}


def _r_init(panel_dir: str, eligible: str, dry: bool):
    use_random_x1_menu(dry)
    sc._setup(load_panel_with_cash(Path(panel_dir)), pd.read_parquet(eligible))


# --------------------------------------------------------------------------- signal sets

def rule_table(e: str, exit_key: str) -> pd.DataFrame:
    """Filled first-day signals of rule e (middle value) with their exit under exit_key."""
    key = (e, exit_key)
    if key in _W["tables"]:
        return _W["tables"][key]
    d = _W["dirs"]
    x, yi = EXITS_J[exit_key]
    if e.startswith("R0s"):
        seed_i = E0_SEEDS.index(int(e[3:]))
        f = d["stage_i"] / "r0" / "signals" / "full" / "R0__H1_signals.parquet" if x == "H1" else d["r0x1"] / "signals" / "full" / "R0__X1_signals.parquet"
        xi, yi = seed_i, (yi if x == "H1" else 0)
    elif x == "H1":
        f, xi = d["stage_h"] / "h1" / "signals" / "full" / f"{e}__H1_signals.parquet", 0
    else:
        root = d["v2"] if e.startswith("E") else d["events"]
        f, xi, yi = root / "signals" / "full" / f"{e}__X1_signals.parquet", _C_CANON.get(e, 1), 1
    s = pd.read_parquet(f, columns=["xi", "yi", "symbol", "signal_date", "entry_date", "exit_signal_date", "return",
                                    "holding_days", "closed", "exit_reason"])
    s = s[(s["xi"] == xi) & (s["yi"] == yi) & (s["exit_reason"] != CANCELLED)].drop(columns=["xi", "yi"])
    s = s.drop_duplicates(["symbol", "signal_date"]).reset_index(drop=True)
    s["si"] = _W["px"].dates.get_indexer(pd.DatetimeIndex(s["signal_date"]))
    _W["tables"][key] = s
    return s


def pair_rows(a: str, b: str, exit_key: str) -> pd.DataFrame:
    """Rows of the pair signal: days of A with a B signal in [d - W, d] and days of B with an A signal
    in [d - W, d], each taken from its own rule's table (same stock-day -> one row)."""
    ta, tb = rule_table(a, exit_key), rule_table(b, exit_key)
    out = []
    for t_late, t_early in ((ta, tb), (tb, ta)):
        late = t_late.sort_values("si")
        early = t_early[["symbol", "si"]].rename(columns={"si": "si_early"}).sort_values("si_early")
        m = pd.merge_asof(late, early, left_on="si", right_on="si_early", by="symbol", direction="backward",
                          tolerance=PAIR_WINDOW)
        out.append(m[m["si_early"].notna()].drop(columns="si_early"))
    return pd.concat(out, ignore_index=True).drop_duplicates(["symbol", "si"]).reset_index(drop=True)


def thinned(seed: int, bin_: float, draw: int, exit_key: str) -> pd.DataFrame:
    """Random buys of seed `seed` thinned to share `bin_` of stock-days (E0 is 1%); the same stock-days
    for every exit (draw 0..: independent thinnings)."""
    base = rule_table(f"R0s{seed}", "H1k2")[["symbol", "si"]].sort_values(["si", "symbol"]).reset_index(drop=True)
    u = np.random.default_rng([SEED, seed, int(bin_ * 1e6), draw]).random(len(base))
    keep = base[u < bin_ / 0.01]
    t = rule_table(f"R0s{seed}", exit_key)
    return t.merge(keep, on=["symbol", "si"], how="inner")


def signals_for(spec: tuple, exit_key: str) -> pd.DataFrame:
    kind = spec[0]
    if kind == "single":
        return rule_table(spec[1], exit_key)
    if kind == "pair":
        return pair_rows(spec[1], spec[2], exit_key)
    _, seed, bin_, draw = spec  # "e0" (draw 0) or "placebo" (draw >= 1)
    return thinned(seed, bin_, draw, exit_key)


# --------------------------------------------------------------------------- per set: accounts

def _init(panel_dir: str, eligible: str, dirs: dict):
    es._init(panel_dir, eligible, "", dirs)
    _W.update({k: es._W[k] for k in ("px", "w0", "n", "wdates", "cash", "O", "LC")})
    es._W.pop("dv", None)  # dollar-volume order is not used here
    _W["dirs"] = dirs
    _W["tables"] = {}


def run_set(job):
    spec, n_orders, cost_bps = job
    px = _W["px"]
    w0, n, wdates = _W["w0"], _W["n"], _W["wdates"]
    accs, uns = [], []
    for exit_key, (x, ki) in EXITS_J.items():
        s = signals_for(spec, exit_key)
        m = px.measures(s, es.END)
        if x == "H1":
            w = ph.weights(px, m, ph.K_VALUES[ki])
        else:  # no stop: fixed 10% weight
            w = pd.DataFrame({"w": np.full(len(m), ph.CAP), "capped": True})
            w["contrib"] = w["w"] * m["ret"].to_numpy()
        d = pd.concat([m[["si", "ei", "sj", "sale", "closed", "x_spy"]], w[["w", "contrib"]], s[["signal_date"]]], axis=1)
        ok = (d["ei"].to_numpy() >= w0) & (d["ei"].to_numpy() < w0 + n) & np.isfinite(d["w"].to_numpy())
        d = d[ok].reset_index(drop=True)
        ei, sale = d["ei"].to_numpy() - w0, d["sale"].to_numpy() - w0
        closed = d["closed"].to_numpy(dtype=bool) & (sale < n)
        rng = np.random.default_rng([SEED, zlib.crc32(f"{set_name(spec)}|{exit_key}|{cost_bps}".encode())])
        for o in range(n_orders):
            eq, tr = ph.replay_risk(ei - 1, ei, d["sj"].to_numpy(), sale, closed, d["w"].to_numpy(), rng.random(len(d)),
                                    _W["O"], _W["LC"], _W["cash"], cost_bps / 10_000)
            accs.append({"set": set_name(spec), "kind": spec[0], "exit": exit_key, "order": o, "trades": len(tr),
                         **es.period_stats(pd.Series(eq, index=wdates))})
        yr = pd.DatetimeIndex(d["signal_date"]).year.to_numpy()
        un = {"set": set_name(spec), "kind": spec[0], "exit": exit_key, "signals": len(d), "freq": len(d) / STOCK_DAYS,
              "days": n}
        for p, (a, b) in es.PERIODS.items():
            mm = (yr >= a) & (yr <= b)
            un[f"n_{p}"] = int(mm.sum())
            un[f"mean_contrib_{p}"] = float(d["contrib"].to_numpy()[mm].mean()) if mm.any() else np.nan
            un[f"mean_x_spy_{p}"] = float(d["x_spy"].to_numpy()[mm].mean()) if mm.any() else np.nan
        uns.append(un)
    return spec, pd.DataFrame(accs), pd.DataFrame(uns)


# --------------------------------------------------------------------------- verdicts

def judge(acc: pd.DataFrame, un: pd.DataFrame, ref_acc: pd.DataFrame, ref_un: pd.DataFrame) -> pd.DataFrame:
    """Per set and exit: medians vs the reference (same-frequency E0) and the (a) / (b) half tests."""
    rows = []
    med = acc.groupby(["set", "exit"]).median(numeric_only=True)
    for (st, x), r in med.iterrows():
        u = un[(un["set"] == st) & (un["exit"] == x)].iloc[0]
        b = nearest_bin(float(u["freq"]))
        ra = ref_acc[(ref_acc["bin"] == b) & (ref_acc["exit"] == x)]
        ru = ref_un[(ref_un["bin"] == b) & (ref_un["exit"] == x)]
        row = {"set": st, "exit": x, "freq": u["freq"], "signals": u["signals"], "days": u["days"],
               "per_day": u["signals"] / u["days"], "bin": b, "trades": r["trades"]}
        for p in es.PERIODS:
            row[f"cagr_{p}"], row[f"mdd_{p}"] = r[f"cagr_{p}"], r[f"mdd_{p}"]
            row[f"e0_cagr_{p}"] = float(ra[f"cagr_{p}"].median())
            row[f"pct_in_e0_{p}"] = float((ra[f"cagr_{p}"] < r[f"cagr_{p}"]).mean())
            row[f"contrib_{p}"] = float(u[f"mean_contrib_{p}"])
            # E0 mean contribution pooled over the seeds (weighted by the period's signal counts)
            row[f"e0_contrib_{p}"] = float((ru[f"mean_contrib_{p}"] * ru[f"n_{p}"]).sum() / ru[f"n_{p}"].sum())
            row[f"x_spy_{p}"] = float(u[f"mean_x_spy_{p}"])
        row["a_halves"] = all(row[f"contrib_{h}"] > row[f"e0_contrib_{h}"] for h in es.HALVES)
        row["b_halves"] = all(row[f"cagr_{h}"] > row[f"e0_cagr_{h}"] for h in es.HALVES)
        rows.append(row)
    v = pd.DataFrame(rows)
    g = v.groupby("set")
    v["enough_trades"] = g["trades"].transform("min") >= MIN_TRADES
    v["meaningful_a"] = g["a_halves"].transform("sum") >= 2
    v["meaningful_b"] = g["b_halves"].transform("sum") >= 2
    return v


def candidates(v: pd.DataFrame, base: pd.DataFrame) -> pd.Series:
    spy = base.set_index("baseline").loc["SPY"]
    ok = pd.Series(True, index=v.index)
    for p in ("1993-2023", "2017-2023"):
        ok &= (v[f"cagr_{p}"] > spy[f"cagr_{p}"]) & (v[f"mdd_{p}"].abs() < abs(spy[f"mdd_{p}"]))
    ok &= v["trades"] >= MIN_TRADES
    return ok & (ok.groupby(v["set"]).transform("sum") >= 2)


def chance(vp: pd.DataFrame, n_pairs: int, passed: dict) -> dict:
    """Pass rate of the random pseudo-pairs and the 95th percentile of the pass count expected by chance."""
    from math import comb

    out = {"n_pairs_judged": n_pairs, "observed": passed}
    one = vp.drop_duplicates("set")
    for k in ("meaningful_a", "meaningful_b"):
        p0 = float(one[k].mean())
        cdf, q95 = 0.0, n_pairs
        for x in range(n_pairs + 1):
            cdf += comb(n_pairs, x) * p0 ** x * (1 - p0) ** (n_pairs - x)
            if cdf >= 0.95:
                q95 = x
                break
        out[k] = {"p0": p0, "expected": p0 * n_pairs, "q95": q95, "information": passed[k] > q95}
        if "bin" in one:  # information only: the pass rate within each frequency bin
            out[k]["p0_by_bin"] = {f"{b:g}": float(g[k].mean()) for b, g in one.groupby("bin")}
    return out


def write_figures(out: Path, v: pd.DataFrame, ref_acc: pd.DataFrame, base: pd.DataFrame) -> list[Path]:
    if not figures.available():
        return []
    plt = figures._plt()
    made = []
    spy = base.set_index("baseline").loc["SPY"]
    pv = v[v["kind"] == "pair"]
    days = float(v["days"].iloc[0])
    ticks = [0.1, 0.2, 0.5, 1, 2, 4, 8, 16]
    for x in EXITS_J:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
        for ax, h in zip(axes, es.HALVES):
            for b in BINS:
                ra = ref_acc[(ref_acc["bin"] == b) & (ref_acc["exit"] == x)][f"cagr_{h}"] * 100
                per_day = b * STOCK_DAYS / days
                ax.vlines(per_day, ra.quantile(0.05), ra.quantile(0.95), color="#999", lw=6, alpha=0.5)
                ax.plot(per_day, ra.median(), "_", color="#333", ms=14)
            g = pv[pv["exit"] == x]
            ax.scatter(g["per_day"], g[f"cagr_{h}"] * 100, s=14, color=figures.COLORS[0], alpha=0.8, zorder=3)
            ax.axhline(spy[f"cagr_{h}"] * 100, color="#c0392b", ls="--", lw=1)
            ax.set_xscale("log")
            lo_, hi_ = ax.get_xlim()
            t_ = [t for t in ticks if lo_ <= t <= hi_]
            ax.set_xticks(t_, [f"{t:g}" for t in t_])
            ax.minorticks_off()
            ax.set_xlabel("하루 평균 신호 수 (로그)")
            ax.set_title(h.replace("-", "–"))
            ax.grid(alpha=0.3)
        axes[0].set_ylabel("10종목 계좌 CAGR (%)")
        fig.suptitle(f"매수 2개 묶기 105쌍 (점) vs 같은 빈도 무작위 매수 (회색 5–95%, 가로선 중앙값), 매도 {x}; 빨간 점선 SPY", fontsize=11)
        made.append(figures._save(fig, out / "figures" / f"pairs_{x}.png"))
    sv = v[v["kind"] == "single"]
    for x in EXITS_J:
        g = pv[pv["exit"] == x]
        gs = sv[sv["exit"] == x]
        fig, axes = plt.subplots(2, 2, figsize=(13, 12))
        for c, h in enumerate(es.HALVES):
            for r_, (col, scale, cmap, lim, title) in enumerate((
                    ("cagr", 100, "viridis", None, "10종목 CAGR (%), 대각선 = 단독"),
                    ("pct_in_e0", 100, "RdBu", (0, 100), "같은 빈도 무작위 매수 안 분위 (%)"))):
                ax = axes[r_, c]
                mat = np.full((len(RULES), len(RULES)), np.nan)
                for _, r in g.iterrows():
                    a, b = r["set"].split("__")[1:3]
                    i, j = RULES.index(a), RULES.index(b)
                    mat[i, j] = mat[j, i] = r[f"{col}_{h}"] * scale
                if col == "cagr":
                    for _, r in gs.iterrows():
                        i = RULES.index(r["set"].split("__")[1])
                        mat[i, i] = r[f"cagr_{h}"] * scale
                lo, hi = lim if lim else (np.nanmin(mat), np.nanmax(mat))
                im = ax.imshow(mat, cmap=cmap, vmin=lo, vmax=hi)
                for i in range(len(RULES)):
                    for j in range(len(RULES)):
                        if np.isfinite(mat[i, j]):
                            t = (mat[i, j] - lo) / max(hi - lo, 1e-9)
                            dark = t < 0.45 if cmap == "viridis" else (t < 0.2 or t > 0.8)
                            ax.text(j, i, f"{mat[i, j]:.0f}" if col == "pct_in_e0" else f"{mat[i, j]:.1f}", ha="center",
                                    va="center", fontsize=5.5, color="white" if dark else "black")
                ax.set_xticks(range(len(RULES)), RULES, fontsize=7)
                ax.set_yticks(range(len(RULES)), RULES, fontsize=7)
                ax.set_title(f"{h.replace('-', '–')}: {title}", fontsize=10)
                fig.colorbar(im, ax=ax, shrink=0.7)
        fig.suptitle(f"쌍별 10종목 계좌 (무작위 순 100번 중앙값), 매도 {x}; 아래 50 = 같은 빈도 무작위와 같음", fontsize=11)
        made.append(figures._save(fig, out / "figures" / f"pairs_matrix_{x}.png"))
    fig, axes = plt.subplots(1, len(EXITS_J), figsize=(15, 5), sharex=True, sharey=True)
    for ax, x in zip(axes, EXITS_J):
        g = pv[pv["exit"] == x]
        lo = np.nanmin(pv[["best_single_cagr", "cagr_1993-2023"]].to_numpy()) * 100 - 0.5
        hi = np.nanmax(pv[["best_single_cagr", "cagr_1993-2023"]].to_numpy()) * 100 + 0.5
        ax.plot([lo, hi], [lo, hi], color="#555", lw=1)
        ax.axhline(spy["cagr_1993-2023"] * 100, color="#c0392b", ls="--", lw=1)
        up = g["stacks"].to_numpy(dtype=bool)
        ax.scatter(g["best_single_cagr"][~up] * 100, g["cagr_1993-2023"][~up] * 100, s=14, color="#999", label="더 나은 단독 이하")
        ax.scatter(g["best_single_cagr"][up] * 100, g["cagr_1993-2023"][up] * 100, s=14, color=figures.COLORS[0],
                   label="더 나은 단독보다 높음")
        ax.set_title(f"매도 {x}: {int(up.sum())} / {len(g)}쌍이 더 나은 단독보다 높음", fontsize=10)
        ax.set_xlabel("구성 단독 두 개 중 더 나은 쪽 CAGR (%)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("쌍 CAGR (%)")
    axes[-1].legend(loc="lower right", fontsize=8, frameon=False)
    fig.suptitle("쌓임 효과: 10종목 계좌 1993–2023 CAGR (무작위 순 중앙값), 대각선 위 = 쌍이 낫다, 빨간 점선 SPY", fontsize=11)
    made.append(figures._save(fig, out / "figures" / "pairs_stacking.png"))
    return made


def run_sets(todo: list, sets_dir: Path, a) -> None:
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.panel, a.eligible, a.dirs)) as ex:
        for i, (spec, acc, un) in enumerate(ex.map(run_set, todo, chunksize=1)):
            acc.to_csv(sets_dir / f"{set_name(spec)}_acc.csv", index=False)
            un.to_csv(sets_dir / f"{set_name(spec)}_un.csv", index=False)
            if (i + 1) % 10 == 0 or i + 1 == len(todo):
                ts.log(f"  [{i + 1}/{len(todo)}] 남은 예상 {(len(todo) - i - 1) * (time.time() - t0) / (i + 1) / 60:.0f}분")


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for k in ("panel", "eligible", "stage_h", "stage_i", "v2", "events", "out"):
        ap.add_argument(f"--{k.replace('_', '-')}", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = {"stage_h": Path(a.stage_h), "stage_i": Path(a.stage_i), "v2": Path(a.v2), "events": Path(a.events), "r0x1": out / "r0x1"}
    a.dirs = dirs
    use_random_x1_menu(a.dry)
    with ProcessPoolExecutor(max_workers=2, initializer=_r_init, initargs=(a.panel, a.eligible, a.dry)) as ex:
        sc.run_signal_tables(ex, out / "r0x1", ts.log)
    global RULES, E0_SEEDS, N_PLACEBO_PER_BIN
    if a.dry:
        RULES, E0_SEEDS, N_PLACEBO_PER_BIN = ["E2", "E9", "V3"], E0_SEEDS[:2], 2
    n_orders = 5 if a.dry else N_ORDERS
    p_orders = 3 if a.dry else PLACEBO_ORDERS
    specs = ([(("single", r), n_orders, COST_BPS) for r in RULES] + [(("pair", x, y), n_orders, COST_BPS) for x, y in pairs()]
             + [(("e0", s, b, 0), n_orders, COST_BPS) for s in E0_SEEDS for b in BINS]
             + [(("placebo", E0_SEEDS[d % len(E0_SEEDS)], b, 1 + d), p_orders, COST_BPS) for b in BINS for d in range(N_PLACEBO_PER_BIN)])
    sets_dir = out / "sets"
    sets_dir.mkdir(exist_ok=True)
    todo = [j for j in specs if not (sets_dir / f"{set_name(j[0])}_acc.csv").exists()]
    ts.log(f"묶음 {len(todo)}개 남음 (단독 {len(RULES)}, 쌍 {len(pairs())}, E0 {len(E0_SEEDS) * len(BINS)}, 위약 {N_PLACEBO_PER_BIN * len(BINS)})")
    run_sets(todo, sets_dir, a)
    acc = pd.concat([pd.read_csv(f) for f in sorted(sets_dir.glob("*_acc.csv"))], ignore_index=True)
    un = pd.concat([pd.read_csv(f) for f in sorted(sets_dir.glob("*_un.csv"))], ignore_index=True)
    for t in (acc, un):
        r0 = t["kind"].isin(["e0", "placebo"])
        t["bin"] = np.nan
        t.loc[r0, "bin"] = t.loc[r0, "set"].str.split("__").str[2].astype(float)
    ref_acc, ref_un = acc[acc["kind"] == "e0"], un[un["kind"] == "e0"]
    _init(a.panel, a.eligible, dirs)
    base = es.baselines(_W["px"], _W["wdates"], _W["w0"], _W["cash"])
    base.to_csv(out / "baselines.csv", index=False)
    v = judge(acc[acc["kind"] != "e0"], un[un["kind"] != "e0"], ref_acc, ref_un)
    v["kind"] = v["set"].str.split("__").str[0]
    v["candidate_20bp"] = candidates(v, base) & (v["kind"] != "placebo")
    v["candidate_40bp"] = False
    cand = v[v["candidate_20bp"]]
    if len(cand):  # candidates only: the same accounts at 40 bp
        d40 = out / "sets_40bp"
        d40.mkdir(exist_ok=True)
        by_name = {set_name(j[0]): j[0] for j in specs}
        run_sets([(by_name[s], n_orders, CHECK_COST_BPS) for s in sorted(cand["set"].unique())
                  if not (d40 / f"{s}_acc.csv").exists()], d40, a)
        med40 = pd.concat([pd.read_csv(d40 / f"{s}_acc.csv") for s in cand["set"].unique()]).groupby(["set", "exit"]).median(numeric_only=True)
        spy = base.set_index("baseline").loc["SPY"]
        for i, r in cand.iterrows():
            m = med40.loc[(r["set"], r["exit"])]
            v.loc[i, "candidate_40bp"] = all(m[f"cagr_{p}"] > spy[f"cagr_{p}"] and abs(m[f"mdd_{p}"]) < abs(spy[f"mdd_{p}"])
                                             for p in ("1993-2023", "2017-2023"))
    # stacking: pair vs the better of its two singles (same exit, median CAGR; 1993-2023 and each half)
    single = v[v["kind"] == "single"].set_index(["set", "exit"])
    for p in ("1993-2023",) + es.HALVES:
        def better_single(r, p=p):
            if r["kind"] != "pair":
                return np.nan
            a_, b_ = r["set"].split("__")[1:3]
            return max(single[f"cagr_{p}"].get((f"single__{a_}", r["exit"]), np.nan),
                       single[f"cagr_{p}"].get((f"single__{b_}", r["exit"]), np.nan))
        v[f"best_single_cagr_{p}"] = v.apply(better_single, axis=1)
    v["best_single_cagr"] = v["best_single_cagr_1993-2023"]
    v["stacks"] = v["cagr_1993-2023"] > v["best_single_cagr"]
    v["stacks_halves"] = np.logical_and.reduce([v[f"cagr_{h}"] > v[f"best_single_cagr_{h}"] for h in es.HALVES])
    v.to_csv(out / "verdicts.csv", index=False)
    acc.groupby(["set", "kind", "exit"]).quantile([0.05, 0.5, 0.95], numeric_only=True).to_csv(out / "accounts_quantiles.csv")
    un.to_csv(out / "unlimited.csv", index=False)
    pv = v[(v["kind"] == "pair") & v["enough_trades"]].drop_duplicates("set")
    passed = {"meaningful_a": int(pv["meaningful_a"].sum()), "meaningful_b": int(pv["meaningful_b"].sum())}
    ch = chance(v[v["kind"] == "placebo"], len(pv), passed)
    ch["candidates_20bp"] = v.loc[v["candidate_20bp"], ["set", "exit"]].to_dict("records")
    ch["candidates_40bp"] = v.loc[v["candidate_40bp"], ["set", "exit"]].to_dict("records")
    ch["excluded_few_trades"] = sorted(v.loc[(v["kind"] == "pair") & ~v["enough_trades"], "set"].unique().tolist())
    ch["trials"] = {"before": N_TRIALS_BEFORE, "new": N_NEW}
    (out / "chance.json").write_text(json.dumps(ch, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    write_figures(out, v, ref_acc, base)
    ts.log(f"완료 · 통과 쌍 (a) {passed['meaningful_a']} · (b) {passed['meaningful_b']} / {len(pv)} · "
           f"우연 95% 분위 {ch['meaningful_a']['q95']} · {ch['meaningful_b']['q95']} · 후보 {len(ch['candidates_20bp'])}")


if __name__ == "__main__":
    main(sys.argv[1:])
