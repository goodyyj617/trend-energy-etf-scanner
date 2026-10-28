"""Stage I: each buy strategy alone in the 1%-rule account, against random buys (DECISIONS 87).

    python -m lab.entry_study --panel DIR --eligible FILE --stage-h DIR --v2 DIR --events DIR --out DIR
                              [--workers N] [--dry]

1. E0 tables: random buys (`blocks.g_random`, each stock-day with probability 1%, seeds 1-5) x exits
   H1 (25% trailing or ATR k stop) and X9 (= H9), built like `stage_c_study --signals-only`.
2. Per buy (E1-E9, V1-V6 at their middle value; E0 seed 1-5) x exit (H1, H9) x k (1.5 / 2 / 3):
   the filled signals with their 1%-rule weight w = min(1% / stop%, 10%) and contribution w x return;
   - no stock limit: contribution stats by period and a calendar-time portfolio scaled to 100%;
   - account A (10 stocks, 20-session re-entry ban, 20 bp): 100 random buying orders + dollar volume.
3. Verdicts (registered): meaningful buy vs E0 (a: no limit, b: 10 stocks) and candidates vs SPY.
Output (aggregates only): jobs/*.csv, accounts.csv, unlimited.csv, verdicts.csv, baselines.csv,
summary.json, figures/.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from . import event_study as ev, figures, pick_study as ph, stage_c_study as sc, tail_study as ts
from .crsp_study import SAMPLES, load_panel_with_cash
from .metrics import deflated_sharpe

# ---- registered (DECISIONS 87): fixed before any result
ENTRY_KEYS = [f"E{i}" for i in range(1, 10)] + list(ev.EVENTS)
RANDOM_SEEDS = [1, 2, 3, 4, 5]
EXITS_I = {"H1": "H1", "H9": "X9"}  # exit -> table suffix
K_VALUES = ph.K_VALUES
N_ORDERS = 100
START, END = "1993-02-01", SAMPLES["full"][1]
PERIODS = {"1993-2023": (1993, 2023), "1993-2007": (1993, 2007), "2008-2023": (2008, 2023), "2017-2023": (2017, 2023),
           "2000-2023": (2000, 2023)}
HALVES = ("1993-2007", "2008-2023")
COST_BPS, CHECK_COST_BPS = 20.0, 40.0
N_TRIALS_BEFORE = 63_400 + 48  # Stages B-H (approx.)
N_NEW = 90
SEED = 87
_C_CANON = dict(sc.ENTRY_CANON)
_W: dict = {}


def buys() -> list[str]:
    return ENTRY_KEYS + [f"E0s{s}" for s in RANDOM_SEEDS]


def job_name(b: str, x: str, ki: int) -> str:
    return f"{b}__{x}_k{K_VALUES[ki]:g}"


# --------------------------------------------------------------------------- 1. E0 tables

def use_random_menu(dry: bool = False) -> None:
    sc.ENTRIES.clear()
    sc.ENTRIES["R0"] = ("g_random", list(RANDOM_SEEDS if not dry else RANDOM_SEEDS[:2]))
    sc.EXITS.clear()
    sc.EXITS["H1"] = ("atr_stop", list(K_VALUES))
    sc.EXITS["X9"] = ("atr_stop", list(K_VALUES))
    sc.EXIT_EXTRA["H1"] = {"trailing_pct": ph.H1_TRAIL}
    sc.SAMPLES = {"full": SAMPLES["full"]}


def _r_init(panel_dir: str, eligible: str, dry: bool):
    use_random_menu(dry)
    sc._setup(load_panel_with_cash(Path(panel_dir)), pd.read_parquet(eligible))


def table(dirs: dict, b: str, x: str) -> tuple[Path, int]:
    suffix = EXITS_I[x]
    if b.startswith("E0s"):
        return dirs["r0"] / "signals" / "full" / f"R0__{suffix}_signals.parquet", RANDOM_SEEDS.index(int(b[3:]))
    if x == "H1":
        return dirs["h1"] / "signals" / "full" / f"{b}__H1_signals.parquet", 0
    root = dirs["v2"] if b.startswith("E") else dirs["events"]
    return root / "signals" / "full" / f"{b}__X9_signals.parquet", _C_CANON.get(b, 1)


# --------------------------------------------------------------------------- 2. per job

def period_stats(eq: pd.Series) -> dict:
    out = {}
    for name, (a, b) in PERIODS.items():
        e = eq.loc[f"{a}":f"{b}"]
        st = ph.curve_stats(e / e.iloc[0])
        out[f"cagr_{name}"], out[f"mdd_{name}"] = st["cagr"], st["mdd"]
    return out


def _init(panel_dir: str, eligible: str, out: str, dirs: dict):
    panel = load_panel_with_cash(Path(panel_dir))
    elig = pd.read_parquet(eligible)
    px = ts.Prices(panel, elig)
    idx = panel.close.index
    win = (idx >= pd.Timestamp(START)) & (idx <= pd.Timestamp(END))
    w0 = int(np.flatnonzero(win)[0])
    n = int(win.sum())
    _W.update(px=px, w0=w0, n=n, wdates=idx[win], cash=panel.cash.reindex(idx[win]).fillna(0.0).to_numpy(),
              dv=panel.close.mul(panel.volume).rolling(20, min_periods=10).mean().to_numpy(dtype="float32"),
              O=px.O[w0:w0 + n].astype(float), LC=px.LC[w0:w0 + n].astype(float), out=Path(out), dirs=dirs)


def signals_of(b: str, x: str, ki: int) -> pd.DataFrame:
    px: ts.Prices = _W["px"]
    f, xi = table(_W["dirs"], b, x)
    s = pd.read_parquet(f)
    s = s[(s["xi"] == xi) & (s["yi"] == ki) & (s["exit_reason"] != ph.CANCELLED)].reset_index(drop=True)
    s = s.drop_duplicates(["symbol", "signal_date"]).reset_index(drop=True)
    m = px.measures(s, END)
    w = ph.weights(px, m, K_VALUES[ki])
    return pd.concat([s[["symbol", "signal_date", "return", "closed", "exit_reason"]], m[["si", "ei", "sj", "sale", "x_spy"]], w], axis=1)


def run_job(job):
    b, x, ki, n_orders, cost_bps, tag_ = job
    t0 = time.time()
    s = signals_of(b, x, ki)
    w0, n, wdates = _W["w0"], _W["n"], _W["wdates"]
    ok = (s["ei"].to_numpy() >= w0) & (s["ei"].to_numpy() < w0 + n) & np.isfinite(s["w"].to_numpy())
    s = s[ok].reset_index(drop=True)
    ei = s["ei"].to_numpy() - w0
    sale = s["sale"].to_numpy() - w0
    closed = s["closed"].to_numpy(dtype=bool) & (sale < n)
    sj, w = s["sj"].to_numpy(), s["w"].to_numpy()
    cost = cost_bps / 10_000
    rng = np.random.default_rng([SEED, zlib.crc32(f"{b}|{x}|{ki}".encode()), int(cost_bps)])
    rows = []
    keys = [("dollar_volume", np.nan_to_num(_W["dv"][s["si"].to_numpy(), sj], nan=-1.0))] if not tag_ else []
    keys += [(f"random_{o}", rng.random(len(s))) for o in range(n_orders)]
    for name, key in keys:
        eq, tr = ph.replay_risk(ei - 1, ei, sj, sale, closed, w, key, _W["O"], _W["LC"], _W["cash"], cost)
        e = pd.Series(eq, index=wdates)
        rows_ = tr["row"].to_numpy()
        contrib = tr["weight"].to_numpy() * tr["ret"].to_numpy()
        rows.append({"buy": b, "exit": x, "k": K_VALUES[ki], "order": name, "trades": len(tr), **period_stats(e),
                     "big_share": float((contrib >= ph.BIG_CONTRIB).mean()) if len(tr) else np.nan,
                     "capped_share": float(s["capped"].to_numpy()[rows_].mean()) if len(tr) else np.nan,
                     "mean_x_spy": float(s["x_spy"].to_numpy()[rows_].mean()) if len(tr) else np.nan})
    acc = pd.DataFrame(rows)
    yr = pd.DatetimeIndex(s["signal_date"]).year.to_numpy()
    un = {"buy": b, "exit": x, "k": K_VALUES[ki], "signals": len(s)}
    for name, (a, bb) in PERIODS.items():
        m = (yr >= a) & (yr <= bb)
        g = s[m]
        un[f"mean_contrib_{name}"] = float(g["contrib"].mean()) if len(g) else np.nan
        un[f"big_{name}"] = float((g["contrib"] >= ph.BIG_CONTRIB).mean()) if len(g) else np.nan
        un[f"mean_x_spy_{name}"] = float(g["x_spy"].mean()) if len(g) else np.nan
        un[f"n_{name}"] = int(m.sum())
    un["capped_share"] = float(s["capped"].mean())
    if not tag_:
        ceq, gross = ph.calendar_weighted(ei, sale, closed, sj, w, _W["O"], _W["LC"], _W["cash"], cost)
        un.update({f"cal_{k}": v for k, v in period_stats(pd.Series(ceq, index=wdates)).items()})
        un["gross_median"] = float(np.median(gross[gross > 0])) if (gross > 0).any() else 0.0
    d = _W["out"] / "jobs"
    d.mkdir(parents=True, exist_ok=True)
    acc.to_csv(d / f"{job_name(b, x, ki)}{tag_}_accounts.csv", index=False)
    pd.DataFrame([un]).to_csv(d / f"{job_name(b, x, ki)}{tag_}_unlimited.csv", index=False)
    return b, x, ki, time.time() - t0


# --------------------------------------------------------------------------- 3. verdicts

def collect(out: Path, tag_: str = "") -> tuple[pd.DataFrame, pd.DataFrame]:
    acc = pd.concat([pd.read_csv(f) for f in sorted((out / "jobs").glob(f"*{tag_}_accounts.csv"))
                     if tag_ or not f.name.endswith("_40bp_accounts.csv")], ignore_index=True)
    un = pd.concat([pd.read_csv(f) for f in sorted((out / "jobs").glob(f"*{tag_}_unlimited.csv"))
                    if tag_ or not f.name.endswith("_40bp_unlimited.csv")], ignore_index=True)
    return acc, un


def verdicts(acc: pd.DataFrame, un: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    rnd = acc[acc["order"].str.startswith("random_")].copy()
    rnd["is_e0"] = rnd["buy"].str.startswith("E0s")
    un = un.copy()
    un["is_e0"] = un["buy"].str.startswith("E0s")
    spy = base.set_index("baseline").loc["SPY"]
    rows = []
    for (x, k), g in rnd.groupby(["exit", "k"]):
        e0 = g[g["is_e0"]]
        u0 = un[(un["exit"] == x) & (un["k"] == k) & un["is_e0"]]
        # E0 mean contribution pooled over the seeds (weighted by signal counts)
        e0_contrib = {h: float((u0[f"mean_contrib_{h}"] * u0[f"n_{h}"]).sum() / u0[f"n_{h}"].sum()) for h in PERIODS}
        for b in ENTRY_KEYS:
            gb = g[g["buy"] == b]
            ub = un[(un["exit"] == x) & (un["k"] == k) & (un["buy"] == b)].iloc[0]
            row = {"buy": b, "exit": x, "k": k, "signals": int(ub["signals"]), "capped_share": ub["capped_share"]}
            for h in PERIODS:
                med = float(gb[f"cagr_{h}"].median())
                row[f"cagr_{h}"] = med
                row[f"mdd_{h}"] = float(gb[f"mdd_{h}"].median())
                row[f"e0_cagr_{h}"] = float(e0[f"cagr_{h}"].median())
                row[f"pct_in_e0_{h}"] = float((e0[f"cagr_{h}"] < med).mean())
                row[f"contrib_{h}"] = float(ub[f"mean_contrib_{h}"])
                row[f"e0_contrib_{h}"] = e0_contrib[h]
                row[f"big_{h}"] = float(ub[f"big_{h}"])
                row[f"x_spy_{h}"] = float(ub[f"mean_x_spy_{h}"])
                if f"cal_cagr_{h}" in ub:
                    row[f"cal_cagr_{h}"], row[f"cal_mdd_{h}"] = float(ub[f"cal_cagr_{h}"]), float(ub[f"cal_mdd_{h}"])
            row["trades"] = float(gb["trades"].median())
            row["a_halves"] = all(row[f"contrib_{h}"] > row[f"e0_contrib_{h}"] for h in HALVES)
            row["b_halves"] = all(row[f"cagr_{h}"] > row[f"e0_cagr_{h}"] for h in HALVES)
            row["market_pass"] = all(row[f"cagr_{p}"] > spy[f"cagr_{p}"] and abs(row[f"mdd_{p}"]) < abs(spy[f"mdd_{p}"])
                                     for p in ("1993-2023", "2017-2023")) and row["trades"] >= 30
            rows.append(row)
    v = pd.DataFrame(rows)
    grp = v.groupby(["buy", "exit"])
    v["meaningful_a"] = grp["a_halves"].transform("sum") >= 2
    v["meaningful_b"] = grp["b_halves"].transform("sum") >= 2
    v["candidate_20bp"] = v["market_pass"] & (grp["market_pass"].transform("sum") >= 2)
    return v


def baselines(px: ts.Prices, wdates: pd.DatetimeIndex, w0: int, cash: np.ndarray) -> pd.DataFrame:
    n = len(wdates)
    spy = pd.Series(px.spy_c[w0:w0 + n], index=wdates)
    r = spy.pct_change(fill_method=None).fillna(0.0)
    full_spy = pd.Series(px.spy_c, index=px.dates)
    on = (full_spy > full_spy.rolling(200).mean()).shift(1, fill_value=False).to_numpy()[w0:w0 + n]
    ma = pd.Series(np.cumprod(1 + np.where(on, r.to_numpy(), cash)), index=wdates)
    ew = pd.Series(px.ew[w0:w0 + n] / px.ew[w0], index=wdates)
    return pd.DataFrame([{"baseline": k, **period_stats(v)} for k, v in
                         (("SPY", spy / spy.iloc[0]), ("SPY 200일선", ma), ("동일비중 유니버스", ew))])


def write_figures(out: Path, acc: pd.DataFrame, v: pd.DataFrame, base: pd.DataFrame, k: float = 2.0) -> list[Path]:
    """E0's 500 accounts as a band, each buy's median account as a dot: per exit, both halves (k = 2)."""
    if not figures.available():
        return []
    plt = figures._plt()
    made = []
    rnd = acc[acc["order"].str.startswith("random_") & (acc["k"] == k)]
    for x in EXITS_I:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
        g = v[(v["exit"] == x) & (v["k"] == k)].set_index("buy").reindex(ENTRY_KEYS)
        e0 = rnd[(rnd["exit"] == x) & rnd["buy"].str.startswith("E0s")]
        y = np.arange(len(ENTRY_KEYS))
        for ax, h in zip(axes, HALVES):
            lo, mid, hi = (np.percentile(e0[f"cagr_{h}"], q) * 100 for q in (5, 50, 95))
            ax.axvspan(lo, hi, color="#bbbbbb", alpha=0.5, label="무작위 매수 E0 (5–95%)")
            ax.axvline(mid, color="#555", lw=1.2, label="E0 중앙값")
            ax.axvline(float(base.set_index("baseline").loc["SPY", f"cagr_{h}"]) * 100, color="#c0392b", ls="--", lw=1.2, label="SPY")
            ax.scatter(g[f"cagr_{h}"] * 100, y, color=figures.COLORS[0], zorder=3, label="매수 전략 (무작위 순 중앙값)")
            ax.set_title(f"{h.replace('-', '–')}")
            ax.set_xlabel("CAGR (%)")
            ax.grid(axis="x", alpha=0.3)
        axes[0].set_yticks(y, [f"{b} {ev.NAMES[b]}" for b in ENTRY_KEYS])
        axes[0].invert_yaxis()
        axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
        fig.suptitle(f"10종목 1% 룰 계좌, 매도 {x}, k {k:g}: 매수 전략 vs 무작위 매수", fontsize=12)
        made.append(figures._save(fig, out / "figures" / f"entries_{x}_account10.png"))
        fig, axes = plt.subplots(1, 2, figsize=(12, 6.5), sharey=True)
        for ax, h in zip(axes, HALVES):
            ax.axvline(g[f"e0_contrib_{h}"].iloc[0] * 100, color="#555", lw=1.2, label="E0 평균")
            ax.barh(y, g[f"contrib_{h}"] * 100, color=figures.COLORS[0], label="매수 전략")
            ax.set_title(f"{h.replace('-', '–')}")
            ax.set_xlabel("거래 기여 평균 (자산 대비 %)")
            ax.grid(axis="x", alpha=0.3)
        axes[0].set_yticks(y, [f"{b} {ev.NAMES[b]}" for b in ENTRY_KEYS])
        axes[0].invert_yaxis()
        axes[1].legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), fontsize=8, frameon=False)
        fig.suptitle(f"종목 수 제한 없음, 매도 {x}, k {k:g}: 신호당 거래 기여 vs 무작위 매수", fontsize=12)
        made.append(figures._save(fig, out / "figures" / f"entries_{x}_unlimited.png"))
    return made


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--eligible", required=True)
    ap.add_argument("--stage-h", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--events", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry", action="store_true", help="E1·E9 + 씨앗 2개, k 2, 순서 3번")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dirs = {"r0": out / "r0", "h1": Path(a.stage_h) / "h1", "v2": Path(a.v2), "events": Path(a.events)}
    use_random_menu(a.dry)
    with ProcessPoolExecutor(max_workers=max(1, min(a.workers, 4)), initializer=_r_init, initargs=(a.panel, a.eligible, a.dry)) as ex:
        sc.run_signal_tables(ex, out / "r0", ts.log)
    global ENTRY_KEYS, RANDOM_SEEDS
    if a.dry:
        ENTRY_KEYS, RANDOM_SEEDS = ["E1", "E9"], RANDOM_SEEDS[:2]
    n_orders = 3 if a.dry else N_ORDERS
    kis = [1] if a.dry else range(len(K_VALUES))
    jobs = [(b, x, ki, n_orders, COST_BPS, "") for b in buys() for x in EXITS_I for ki in kis
            if not (out / "jobs" / f"{job_name(b, x, ki)}_accounts.csv").exists()]
    ts.log(f"작업 {len(jobs)}개 (매수 × 매도 × k)")
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.panel, a.eligible, str(out), dirs)) as ex:
        for i, (b, x, ki, sec) in enumerate(ex.map(run_job, jobs, chunksize=1)):
            ts.log(f"  [{i + 1}/{len(jobs)}] {job_name(b, x, ki)} {sec:.0f}s · 남은 예상 {(len(jobs) - i - 1) * (time.time() - t0) / (i + 1) / 60:.0f}분")
    _init(a.panel, a.eligible, str(out), dirs)
    base = baselines(_W["px"], _W["wdates"], _W["w0"], _W["cash"])
    base.to_csv(out / "baselines.csv", index=False)
    acc, un = collect(out)
    acc.to_csv(out / "accounts.csv", index=False)
    un.to_csv(out / "unlimited.csv", index=False)
    if a.dry:
        ts.log("점검 실행 완료")
        return
    v = verdicts(acc, un, base)
    v["candidate_40bp"] = False
    v["dsr_raw"] = v["dsr_new"] = np.nan
    cands = v[v["candidate_20bp"]]
    if len(cands):
        jobs40 = [(r.buy, r.exit, K_VALUES.index(r.k), N_ORDERS, CHECK_COST_BPS, "_40bp") for r in cands.itertuples()]
        with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.panel, a.eligible, str(out), dirs)) as ex:
            list(ex.map(run_job, jobs40, chunksize=1))
        acc40, _ = collect(out, "_40bp")
        spy = base.set_index("baseline").loc["SPY"]
        for i in cands.index:
            g = acc40[(acc40["buy"] == v.loc[i, "buy"]) & (acc40["exit"] == v.loc[i, "exit"]) & (acc40["k"] == v.loc[i, "k"])]
            v.loc[i, "candidate_40bp"] = all(g[f"cagr_{p}"].median() > spy[f"cagr_{p}"] and abs(g[f"mdd_{p}"].median()) < abs(spy[f"mdd_{p}"])
                                             for p in ("1993-2023", "2017-2023"))
        ts.log("후보 DSR은 대표 무작위 순 곡선이 필요해 결과 메모에서 따로 계산합니다(후보가 있을 때만).")
    v.to_csv(out / "verdicts.csv", index=False)
    write_figures(out, acc, v, base)
    summ = {"meaningful_a": v.loc[v["meaningful_a"], ["buy", "exit"]].drop_duplicates().to_dict("records"),
            "meaningful_b": v.loc[v["meaningful_b"], ["buy", "exit"]].drop_duplicates().to_dict("records"),
            "candidates_20bp": v.loc[v["candidate_20bp"], ["buy", "exit", "k"]].to_dict("records"),
            "candidates_40bp": v.loc[v["candidate_40bp"], ["buy", "exit", "k"]].to_dict("records")}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1), encoding="utf-8")
    ts.log(f"완료 · 의미 있는 매수 (a) {len(summ['meaningful_a'])} · (b) {len(summ['meaningful_b'])} · 후보 {len(summ['candidates_20bp'])}")


if __name__ == "__main__":
    main(sys.argv[1:])
