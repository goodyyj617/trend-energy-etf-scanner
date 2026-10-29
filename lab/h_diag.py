"""Stage H follow-up: why the 10-stock account lags in bull markets (descriptive, no new trials).

    python -m lab.h_diag --panel DIR --eligible FILE --stage-h DIR --events DIR --v2 DIR --out DIR

Rebuilds the H1 settings of Stage H (k 1.5 / 2 / 3; the main model's walk-forward scores only, the
fit is deterministic) and replays the 10-stock 1%-rule account with a day log for three buying
orders: model score, dollar volume and one random path (seed DIAG_SEED). The model and dollar-volume
accounts must reproduce Stage H's accounts.csv. Per period (2000-2011, 2012-2023, 2017-2023):
exposure and positions, days full with signals left / days idle, market-cap rank of the holdings,
beta / correlation to SPY and excess in months with SPY above its 200-day average, yearly big
winners and the top-5 trades' share, and exits by reason. Aggregates only.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from . import figures, pick_study as ph, tail_study as ts
from .crsp_study import SAMPLES, load_panel_with_cash

PERIODS = {"2000-2011": (2000, 2011), "2012-2023": (2012, 2023), "2017-2023": (2017, 2023)}
DIAG_SEED = 2029
TOP_RANK = 500


def account_logs(px, panel, s, scores, w0, n, wdates, cash):
    yr = pd.DatetimeIndex(s["signal_date"]).year.to_numpy()
    sel = np.flatnonzero((yr >= ph.FIRST_TEST) & (yr <= ph.LAST_TEST) & (s["ei"].to_numpy() >= w0) & (s["ei"].to_numpy() < w0 + n))
    ss = s.iloc[sel].reset_index(drop=True)
    ei = ss["ei"].to_numpy() - w0
    sale = ss["sale"].to_numpy() - w0
    closed = ss["closed"].to_numpy(dtype=bool) & (sale < n)
    sj = ss["sj"].to_numpy()
    O, LC = px.O[w0:w0 + n].astype(float), px.LC[w0:w0 + n].astype(float)
    dv = panel.close.mul(panel.volume).rolling(20, min_periods=10).mean().to_numpy(dtype="float32")[ss["si"].to_numpy(), sj]
    keys = {"model": np.nan_to_num(scores[sel], nan=-1e18), "dollar_volume": np.nan_to_num(dv, nan=-1.0),
            "random": np.random.default_rng(DIAG_SEED).random(len(ss))}
    out = {}
    for name, key in keys.items():
        log: dict = {}
        eq, tr = ph.replay_risk(ei - 1, ei, sj, sale, closed, ss["w"].to_numpy(), key, O, LC, cash, ph.COST_BPS / 10_000, log=log)
        rows = tr["row"].to_numpy()
        tr = tr.assign(contrib=tr["weight"] * tr["ret"], capped=ss["capped"].to_numpy()[rows],
                       reason=np.where(ss["closed"].to_numpy()[rows] & (tr["exit_i"].to_numpy() < n - 1),
                                       ss["exit_reason"].to_numpy()[rows], "기간 끝까지 보유"),
                       exit_year=wdates[tr["exit_i"].to_numpy()].year)
        out[name] = (pd.Series(eq, index=wdates), tr, log)
    return out


def period_mask(wdates, a, b):
    y = wdates.year
    return (y >= a) & (y <= b)


def diagnose(tag, logs, wdates, cap_rank, spy, spy_on_month):
    rows, years = [], []
    for name, (eq, tr, log) in logs.items():
        r = eq.pct_change(fill_method=None).fillna(0.0)
        rs = spy.pct_change(fill_method=None).fillna(0.0)
        for p, (a, b) in PERIODS.items():
            m = period_mask(wdates, a, b)
            idx = np.flatnonzero(m)
            held_ranks = [cap_rank[i, j] for i in idx for j in log["held"][i]]
            held_ranks = np.asarray([x for x in held_ranks if np.isfinite(x)])
            e = eq[m]
            st = ph.curve_stats(e / e.iloc[0])
            me = e.resample("ME").last().pct_change(fill_method=None)
            ms = spy[m].resample("ME").last().pct_change(fill_method=None)
            on = spy_on_month.reindex(me.index).fillna(False).to_numpy(dtype=bool)
            t = tr[(tr["exit_year"] >= a) & (tr["exit_year"] <= b)]
            row = {"setting": tag, "account": name, "period": p, "cagr": st["cagr"], "mdd": st["mdd"],
                   "invested_mean": float(log["invested"][m].mean()), "positions_mean": float(log["positions"][m].mean()),
                   "positions_p10": float(np.percentile(log["positions"][m], 10)),
                   "days_full_left": float((log["left"][m] > 0).mean()), "days_idle": float(log["idle"][m].mean()),
                   "banned_per_year": float(log["banned"][m].sum() / (b - a + 1)),
                   "cap_rank_median": float(np.median(held_ranks)) if len(held_ranks) else np.nan,
                   "top500_share": float((held_ranks <= TOP_RANK).mean()) if len(held_ranks) else np.nan,
                   "beta": float(np.cov(r[m], rs[m])[0, 1] / np.var(rs[m], ddof=1)), "corr": float(r[m].corr(rs[m])),
                   "excess_bull_months": float((me - ms)[on].mean()), "excess_bear_months": float((me - ms)[~on].mean()),
                   "bull_month_share": float(on[1:].mean()), "trades": len(t),
                   "capped_share": float(t["capped"].mean()) if len(t) else np.nan,
                   "big": int((t["contrib"] >= ph.BIG_CONTRIB).sum()), "big_contrib_sum": float(t.loc[t["contrib"] >= ph.BIG_CONTRIB, "contrib"].sum()),
                   "mean_weight": float(t["weight"].mean()) if len(t) else np.nan}
            for reason, g in t.groupby("reason"):
                row[f"exit_share::{reason}"] = len(g) / len(t)
                row[f"exit_contrib::{reason}"] = float(g["contrib"].mean())
            rows.append(row)
        yr = eq.groupby(eq.index.year).last()
        prev = pd.concat([pd.Series([eq.iloc[0]], index=[yr.index[0] - 1]), yr.iloc[:-1]]).to_numpy()
        yret = yr / prev - 1
        for y, ret in yret.items():
            t = tr[tr["exit_year"] == y]
            top5 = np.sort(t["contrib"].to_numpy())[::-1][:5].sum() if len(t) else 0.0
            years.append({"setting": tag, "account": name, "year": int(y), "return": float(ret), "trades": len(t),
                          "big": int((t["contrib"] >= ph.BIG_CONTRIB).sum()),
                          "big_contrib_sum": float(t.loc[t["contrib"] >= ph.BIG_CONTRIB, "contrib"].sum()),
                          "top5_contrib_sum": float(top5), "spy": float(spy[spy.index.year == y].iloc[-1] / spy[spy.index.year < y].iloc[-1] - 1)
                          if (spy.index.year < y).any() else np.nan})
    return pd.DataFrame(rows), pd.DataFrame(years)


def write_figures(out: Path, d: pd.DataFrame) -> list[Path]:
    if not figures.available():
        return []
    made = []
    d = d[d["setting"] == "H1_k2"]
    names = {"model": "모형 순", "dollar_volume": "거래대금 순", "random": "무작위(경로 하나)"}
    for stat, title, unit, pct in (("invested_mean", "H1 k 2 10종목 계좌: 평균 주식 비중", "자산 중 주식 (%)", True),
                                   ("cap_rank_median", "H1 k 2 10종목 계좌: 보유 종목 시총 순위 중앙값 (1 = 최대)", "순위", False),
                                   ("excess_bull_months", "H1 k 2 10종목 계좌: SPY 200일선 위인 달의 월 초과수익 평균", "%p", True),
                                   ("big_contrib_sum", "H1 k 2 10종목 계좌: 큰 승자(기여 ≥ +5%) 기여 합", "자산 대비 (%)", True)):
        m = d.pivot(index="period", columns="account", values=stat).reindex(list(PERIODS))
        made.append(figures.paired_bars([p.replace("-", "–") for p in PERIODS], {names[k]: m[k].to_numpy(dtype=float) for k in names if k in m},
                                        title, unit, out / "figures" / f"diag_{stat}.png", pct=pct, legend_out=True))
    return made


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--eligible", required=True)
    ap.add_argument("--stage-h", required=True, help="Stage H 결과 폴더(h1/, settings/)")
    ap.add_argument("--events", required=True)
    ap.add_argument("--v2", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    panel = load_panel_with_cash(Path(a.panel))
    eligible = pd.read_parquet(a.eligible)
    panel.eligible = eligible
    px = ts.Prices(panel, eligible)
    dirs = {"h1": Path(a.stage_h) / "h1", "v2": Path(a.v2), "events": Path(a.events)}
    sets = {ph.tag("H1", ki): ph.build_setting(px, dirs, "H1", ki, ph.ENTRY_KEYS) for ki in range(len(ph.K_VALUES))}
    base = pd.concat([v[["si", "sj"]] for v in sets.values()]).drop_duplicates().reset_index(drop=True)
    ts.attach_features(panel, eligible, {"base": base})
    idx = panel.close.index
    win = (idx >= pd.Timestamp(f"{ph.FIRST_TEST}-01-01")) & (idx <= pd.Timestamp(SAMPLES["full"][1]))
    w0, wdates = int(np.flatnonzero(win)[0]), idx[win]
    n = len(wdates)
    cash = panel.cash.reindex(wdates).fillna(0.0).to_numpy()
    cap_rank = panel.extra["cap_rank"].reindex(index=wdates, columns=panel.close.columns).to_numpy(dtype="float32")
    spy = pd.Series(px.spy_c[w0:w0 + n], index=wdates)
    full_spy = pd.Series(px.spy_c, index=px.dates)
    above = (full_spy > full_spy.rolling(200).mean())
    spy_on_month = above.resample("ME").last().shift(1).astype("boolean").fillna(False).astype(bool)  # previous month-end
    diags, years, repro = [], [], {}
    for t, s in sets.items():
        s = s.merge(base, on=["si", "sj"], how="left")
        for e in ph.ENTRY_KEYS:
            if f"is_{e}" not in s:
                s[f"is_{e}"] = 0
        scores, _, _ = ph.walk_forward(s, px.dates, np.random.default_rng(0), 0, only={ph.MAIN})
        logs = account_logs(px, panel, s, scores[ph.MAIN], w0, n, wdates, cash)
        ref = pd.read_csv(Path(a.stage_h) / "settings" / t / "accounts.csv").set_index("account")
        repro[t] = {k: {"diag": float(ph.period_stats(logs[k][0])["cagr_all"]), "stage_h": float(ref.loc[r, "cagr_all"])}
                    for k, r in (("model", "logit_atr"), ("dollar_volume", "dollar_volume"))}
        repro[t]["random_path_cagr"] = float(ph.period_stats(logs["random"][0])["cagr_all"])
        repro[t]["stage_h_random_median"] = float(ref[ref.index.str.startswith("random_")]["cagr_all"].median())
        d, y = diagnose(t, logs, wdates, cap_rank, spy, spy_on_month)
        diags.append(d)
        years.append(y)
        ts.log(f"  {t}: 재현 {repro[t]}")
    d = pd.concat(diags, ignore_index=True)
    d.to_csv(out / "diag.csv", index=False)
    pd.concat(years, ignore_index=True).to_csv(out / "diag_years.csv", index=False)
    (out / "reproduction.json").write_text(json.dumps(repro, indent=1), encoding="utf-8")
    write_figures(out, d)
    ts.log(f"완료 · 결과: {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
