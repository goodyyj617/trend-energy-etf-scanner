"""Stage D timing diagnosis: why did the 20-slot account's missed signals earn more than its bought ones?

    python scripts/stage_d_timing.py --panel DIR --signals DIR --eligible FILE --out DIR [--workers N]

Only the saved per-signal tables (stagec_signals_v2) and replays of the 20-slot account
(lab/signal_replay.py) on each pair's canonical cell, both samples, band 'all', 20 bp, buying order
dollar volume (Stage C) and smallest first. No new strategy, no engine run. Per cell:
  1. signals split into bought / symbol already held / no free slot (cancelled orders left out)
  2. days bucketed by their number of signals (deciles of days with a signal)
  3. bought vs missed inside the same day bucket: how much of the gap remains
  4. SPY's drawdown from its 52-week high and 20-day return on the signal day, terciles
Writes cells.csv, buckets.csv, summary.json and figures to `out` (aggregates only).
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab import figures, missed_study as md, signal_replay as sr, stage_c_study as sc  # noqa: E402
from lab.crsp_study import SAMPLES, load_panel_with_cash  # noqa: E402
from lab.data import BENCHMARK  # noqa: E402
from lab.engine import CANCELLED, SignalCache  # noqa: E402

BIG = 0.5
_W: dict = {}


def _init(panel_dir: str, eligible: str, signals: str):
    panel = load_panel_with_cash(Path(panel_dir))
    panel = replace(panel, eligible=pd.read_parquet(eligible))
    _W.update(panel=panel, cache=SignalCache(panel), signals=Path(signals),
              rank={"dv": None, "small": md.priority_frame(panel, "small")})
    spy = panel.close[BENCHMARK]
    _W["spy_dd"] = spy / spy.rolling(252, min_periods=60).max() - 1.0
    _W["spy_r20"] = spy / spy.shift(20) - 1.0


def stats(r: np.ndarray) -> dict:
    return {"n": int(len(r)), "mean": float(r.mean()) if len(r) else np.nan,
            "win": float((r > 0).mean()) if len(r) else np.nan, "big": float((r >= BIG).mean()) if len(r) else np.nan}


def _run(job):
    sample, e, x, how = job
    panel, cache = _W["panel"], _W["cache"]
    xi, yi = sc.canonical(e, x)
    cfg = replace(sc.config(sample, e, x, xi, yi), max_positions=sc.SLOTS)
    sig = pd.read_parquet(_W["signals"] / "signals" / sample / f"{e}__{x}_signals.parquet")
    sig = sig[(sig["xi"] == xi) & (sig["yi"] == yi)].reset_index(drop=True)
    log: dict = {}
    sr.replay(replace(panel, rank=_W["rank"][how]), cfg, sig, cache=cache, log=log)
    dates = panel.close.index[(panel.close.index >= pd.Timestamp(cfg.start)) & (panel.close.index <= pd.Timestamp(cfg.end))]
    di = dates.get_indexer(pd.DatetimeIndex(sig["signal_date"]))
    dj = panel.close.columns.get_indexer(sig["symbol"].astype(str))
    status = np.array([log["status"].get((i, j), "out") for i, j in zip(di.tolist(), dj.tolist())], dtype=object)
    filled = (sig["exit_reason"] != CANCELLED).to_numpy() & (status != "out")
    s = pd.DataFrame({"day": di, "ret": sig["return"].to_numpy(dtype=float), "status": status})[filled]
    s["status"] = s["status"].replace({"queued": "bought"})
    s["missed"] = s["status"] != "bought"
    out = {"sample": sample, "entry": e, "exit": x, "priority": how}
    for g in ("bought", "held", "full"):
        for k, v in stats(s.loc[s["status"] == g, "ret"].to_numpy()).items():
            out[f"{g}_{k}"] = v
    for k, v in stats(s.loc[s["missed"], "ret"].to_numpy()).items():
        out[f"missed_{k}"] = v
    # 2. day buckets by the day's number of signals
    per_day = s.groupby("day").size()
    dec = pd.qcut(per_day.rank(method="first"), 10, labels=False) + 1 if len(per_day) >= 10 else pd.Series(1, index=per_day.index)
    s["bucket"] = s["day"].map(dec)
    free = np.asarray(log["free"])
    top_days = dec.index[dec == dec.max()]
    out["top_decile_signal_share"] = float((s["bucket"] == dec.max()).mean())
    out["top_decile_full_day_share"] = float((free[top_days] <= 0).mean()) if len(top_days) else np.nan
    out["all_days_full_share"] = float((free[per_day.index] <= 0).mean())
    b = s.groupby(["bucket", "missed"])["ret"].agg(["mean", "size"]).unstack("missed")
    brows = []
    for bk, row in b.iterrows():
        brows.append({**{k: out[k] for k in ("sample", "entry", "exit", "priority")}, "bucket": int(bk),
                      "taken_mean": row.get(("mean", False), np.nan), "taken_n": row.get(("size", False), 0),
                      "missed_mean": row.get(("mean", True), np.nan), "missed_n": row.get(("size", True), 0),
                      "all_mean": float(s.loc[s["bucket"] == bk, "ret"].mean()),
                      "full_day_share": float((free[dec.index[dec == bk]] <= 0).mean())})
    bb = pd.DataFrame(brows)
    both = bb.dropna(subset=["taken_mean", "missed_mean"])
    gap = out["missed_mean"] - out["bought_mean"]
    within = float(np.average(both["missed_mean"] - both["taken_mean"], weights=both["taken_n"])) if both["taken_n"].sum() else np.nan
    out.update({"gap": gap, "within_gap": within, "gap_remaining": within / gap if gap and np.isfinite(within) else np.nan})
    # 4. market state on the signal day
    sd = dates[s["day"].to_numpy()]
    for name, ser in (("spy_dd", _W["spy_dd"]), ("spy_r20", _W["spy_r20"])):
        v = ser.reindex(sd).to_numpy()
        t = pd.qcut(pd.Series(v).rank(method="first"), 3, labels=False).to_numpy() + 1
        for k in (1, 2, 3):
            m = t == k
            out[f"{name}_t{k}_mean"] = float(s["ret"].to_numpy()[m].mean()) if m.any() else np.nan
            out[f"{name}_t{k}_bought_share"] = float((~s["missed"].to_numpy()[m]).mean()) if m.any() else np.nan
    return out, bb


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--signals", required=True)
    ap.add_argument("--eligible", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    load_panel_with_cash(Path(a.panel))  # fills the price cache once, so the workers only read it
    jobs = [(s, e, x, how) for s in SAMPLES for e, x in md.pairs() for how in ("dv", "small")]
    rows, buckets = [], []
    with ProcessPoolExecutor(max_workers=a.workers, initializer=_init, initargs=(a.panel, a.eligible, a.signals)) as ex:
        for n, (r, b) in enumerate(ex.map(_run, jobs, chunksize=1)):
            rows.append(r)
            buckets.append(b)
            if (n + 1) % 20 == 0:
                print(f"{n + 1}/{len(jobs)}", flush=True)
    cells = pd.DataFrame(rows)
    bk = pd.concat(buckets, ignore_index=True)
    cells.to_csv(out / "cells.csv", index=False)
    bk.to_csv(out / "buckets.csv", index=False)
    med = cells.groupby(["priority", "sample"]).median(numeric_only=True)
    med.to_csv(out / "medians.csv")
    bmed = bk.groupby(["priority", "sample", "bucket"])[["all_mean", "taken_mean", "missed_mean", "full_day_share"]].median()
    bmed.to_csv(out / "bucket_medians.csv")
    (out / "summary.json").write_text(json.dumps({"medians": med.round(4).reset_index().to_dict("records")},
                                                 ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(med.T.round(4).to_string())
    print(bmed.round(4).to_string())
    if not figures.available():
        return
    names = {k: SAMPLES[k][2] for k in SAMPLES}
    d = out / "figures"
    m = med.loc["dv"]
    labels = ["산 신호", "이미 보유 중이라 건너뜀", "자리가 없어 못 삼"]
    figures.paired_bars(labels, {names[s]: np.array([m.loc[s, f"{g}_mean"] for g in ("bought", "held", "full")]) for s in SAMPLES},
                        "놓친 신호를 이유별로 나누면: 신호당 평균 수익 (대표 칸 81개 중앙값, 거래대금 순)", "신호당 평균 수익 (%)",
                        d / "groups.png")
    plt = figures._plt()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, s in zip(axes, SAMPLES):
        g = bmed.loc[("dv", s)]
        ax.bar(g.index, g["all_mean"] * 100, color=figures.COLORS[0], label="신호당 평균 수익 (%)")
        ax2 = ax.twinx()
        ax2.plot(g.index, g["full_day_share"] * 100, color=figures.COLORS[1], marker="o", label="계좌가 꽉 찬 날 비율 (%)")
        ax.set_xlabel("그날 신호 수 10분위 (1 = 적음, 10 = 많음)")
        ax.set_ylabel("신호당 평균 수익 (%)")
        ax2.set_ylabel("꽉 찬 날 비율 (%)")
        ax.set_title(names[s])
        ax.grid(alpha=0.3)
        h1, l1 = ax.get_legend_handles_labels()
        h2, l2 = ax2.get_legend_handles_labels()
        ax.legend(h1 + h2, l1 + l2, fontsize=8, loc="upper left")
    fig.suptitle("신호가 몰린 날일수록 수익이 높고 계좌는 꽉 차 있는가 (거래대금 순, 81개 쌍 중앙값)")
    figures._save(fig, d / "day_buckets.png")


if __name__ == "__main__":
    main(sys.argv[1:])
