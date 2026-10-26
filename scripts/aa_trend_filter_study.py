"""B2 · B3 · B4 with a 10-month trend filter on their equities (allocation DECISIONS A38; memo 2026-10-27_trend_filter.md).

    .venv/Scripts/python.exe scripts/aa_trend_filter_study.py --out docs/research/allocation/2026-10-27_trend_filter
    .venv/Scripts/python.exe scripts/aa_trend_filter_study.py --quick --out %TEMP%\\aa_filter_quick     # ~1 min check

A37 found that B2–B4 hold MDD −15 % from 2005 but break it from 2000-09 (dot-com, model before 2005). This asks
whether Faber's (2007) 10-month moving-average filter, applied to the equity products only, keeps them inside the
line in both periods. The rule and the reading were fixed in the design doc before the run:

  filter     at each month end, an equity product whose KRW month-end close is below the mean of its last 10
             month-end closes moves its base weight to KODEX CD금리액티브 (459580) for the month; back when above.
             Gold, bonds, REITs, CD and dollar futures are never switched. Fewer than 10 month ends: stay invested.
             Decided on the month-end close, traded at the next close, 10 bp one way (the engine's rules); with a
             filter the engine re-weights monthly, so '정적 월' (static, monthly) is the like-for-like reference.
  settings   정적 분기 (the registered setting), 정적 월, 필터 8 / 10 / 12 (equities only), Faber 원형 10 (every
             risky product, reference only)
  rows       B2, B3, B4, 안정형 40, S&P500 보유 with every setting and the A30 date offsets; the first 2,000 members of
             A37's structure family and of its random control with 정적 분기 / 정적 월 / 필터 10
  periods    2005-01-03 – 2026-10-16 (A37 ①–③ panel) and 2000-09-01 – 2026-10-16 (A37 ④ panel, model before 2005).
             The signal reads each product's history before the start where the panel has it.
  verdict    point: 필터 10 keeps MDD ≥ −15 % in both periods; robust: also with 8 and 12; structure: ≥ 50 % of the
             family sample inside in both periods with 필터 10. CAGR is shown, never judged.

Every run goes through lab.aa_engine.run_allocation, split over processes (--workers). Writes small tables into --out;
nothing goes to lab_results/.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import aa_global_search as G  # noqa: E402
import aa_local_study as L  # noqa: E402
from lab import aa_local as loc  # noqa: E402
from lab import aa_search as srch  # noqa: E402
from lab import aa_search_results as sr  # noqa: E402
from lab.aa_assets import CASH, PRESETS  # noqa: E402
from lab.aa_engine import (  # noqa: E402
    AllocConfig, Overlay, _normalise, _Signals, decision_rows, run_allocation, target_weights,
)
from lab.aa_validate import TIMING_OFFSETS  # noqa: E402

SAFE = loc.CD  # KODEX CD금리액티브(합성)
FAMILY_SEED, CONTROL_SEED = L.SEED + 10, L.SEED + 11  # the A37 ② samples
SETTINGS = {
    "정적 분기": dict(rebalance="Q"),
    "정적 월": dict(rebalance="M"),
    "필터 8": dict(rebalance="M", sma=8, only="equity"),
    "필터 10": dict(rebalance="M", sma=10, only="equity"),
    "필터 12": dict(rebalance="M", sma=12, only="equity"),
    "Faber 원형 10": dict(rebalance="M", sma=10, only="all"),
}
SAMPLE_SETTINGS = ("정적 분기", "정적 월", "필터 10")
PRIMARY, NEIGHBOURS = "필터 10", ("필터 8", "필터 12")
NAMED = {**loc.CENTRES, "안정형 40": PRESETS["stable40"].weights, "S&P500 보유": PRESETS["sp500"].weights}
ERAS = {"2005": sr.START, "2000-09": loc.DOTCOM_START}
CURVE_END = "2006-12-31"
STRUCTURE_SHARE = 0.5

FULL = dict(n_sample=2000, offsets=TIMING_OFFSETS, chunk=100)
QUICK = dict(n_sample=12, offsets=(0, 5), chunk=6)

# panels and loaders: tests swap the loaders for made-up prices
LOADERS = {"2005": lambda prices: G.load(prices), "2000-09": lambda prices: L.load_extended(prices)[0]}
_PANELS: dict[str, object] = {}


def config(weights: dict, setting: str, start: str, offset: int = 0) -> AllocConfig:
    s = SETTINGS[setting]
    overlay = Overlay()
    if "sma" in s:
        only = () if s["only"] == "all" else tuple(c for c in loc.EQUITY if weights.get(c))
        if s["only"] == "all" or only:  # an allocation without equities has nothing to filter
            overlay = Overlay("sma", float(s["sma"]), SAFE, only=only)
    return AllocConfig(setting, {k: float(v) for k, v in weights.items() if v}, start, sr.END, rebalance=s["rebalance"],
                       cost_bps=G.COST_BPS, overlay=overlay, rebalance_offset=offset)


def _windows(era: str) -> dict:
    return {"dotcom": loc.DOTCOM_WINDOW, **sr.WINDOWS} if era == "2000-09" else dict(sr.WINDOWS)


def _extra_starts(era: str) -> tuple[str, ...]:
    return (sr.START, "2016-01-01") if era == "2000-09" else ("2016-01-01",)


def summarize_runs(results: list, era: str) -> pd.DataFrame:
    """srch.summarize over engine runs of one period (they share the dates)."""
    paths = np.vstack([r.equity.to_numpy(dtype=float) for r in results])
    turnover = np.array([r.turnover for r in results], dtype=float)
    df = srch.summarize(paths, turnover, results[0].equity.index, results[0].cash, windows=_windows(era),
                        extra_starts=_extra_starts(era))
    return df


def filter_activity(panel, cfg: AllocConfig, dates: pd.DatetimeIndex) -> dict:
    """How often the filter switched equity off, read from the engine's own month-end decisions: the mean share of
    the base equity weight that sat in CD, the share of months with anything off, and the month ends where the
    switched-off set changed."""
    if cfg.overlay.kind == "none":
        return {"꺼진 비중": 0.0, "꺼진 달": 0.0, "전환": 0}
    close = panel.close.loc[:dates[-1], cfg.assets].ffill()
    sig = _Signals(close, panel.cash.reindex(panel.close.index).fillna(0.0).loc[close.index])
    base = _normalise(cfg.weights)
    equity = [s for s in base if s in loc.EQUITY]
    eq_w = sum(base[s] for s in equity) or 1.0
    shares, prev, switches, any_off = [], None, 0, 0
    for i in decision_rows(dates, "M"):
        tw = target_weights(base, cfg.overlay, sig, dates[int(i)])
        off = frozenset(s for s in equity if s not in tw)
        shares.append(sum(base[s] for s in off) / eq_w)
        any_off += bool(off)
        switches += prev is not None and off != prev
        prev = off
    n = max(len(shares), 1)
    return {"꺼진 비중": float(np.mean(shares)) if shares else 0.0, "꺼진 달": any_off / n, "전환": int(switches)}


# --------------------------------------------------------------------------- workers

def _init(panels: dict) -> None:
    _PANELS.update(panels)


def _run_chunk(task: tuple) -> pd.DataFrame:
    era, setting, rows, labels = task
    panel = _PANELS[era]
    results = [run_allocation(panel, config(w, setting, ERAS[era])) for w in rows]
    df = summarize_runs(results, era)
    df.insert(0, "설정", setting)
    df.insert(0, "기간", era)
    df.insert(0, "행", labels)
    return df


def _map(tasks: list, panels: dict, workers: int):
    if workers <= 1:
        _init(panels)
        return [_run_chunk(t) for t in tasks]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(panels,)) as ex:
        return list(ex.map(_run_chunk, tasks))


def _weights(row: np.ndarray) -> dict:
    return {s: float(v) for s, v in zip(sr.SYMBOLS, row) if v}


# --------------------------------------------------------------------------- study

def named_study(panels: dict, P: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Every setting on the named rows, with the worst MDD over the A30 date offsets and the filter's activity."""
    parts, curves = [], {}
    for era, panel in panels.items():
        start = ERAS[era]
        for setting in SETTINGS:
            results, meta = [], []
            for name, w in NAMED.items():
                cfg = config(w, setting, start)
                res = run_allocation(panel, cfg)
                worst = min(float((eq / eq.cummax() - 1.0).min()) for eq in
                            [res.equity, *(run_allocation(panel, config(w, setting, start, o)).equity for o in P["offsets"] if o)])
                results.append(res)
                meta.append({"이름": name, "기간": era, "설정": setting, "날짜 최악 MDD": worst,
                             **filter_activity(panel, cfg, res.equity.index)})
                if era == "2000-09" and setting in ("정적 분기", PRIMARY):
                    curves[f"{name} · {setting}"] = res.equity.loc[:CURVE_END]
            parts.append(pd.concat([pd.DataFrame(meta), summarize_runs(results, era)], axis=1))
            print(f"  이름 붙은 배분 · {era} · {setting}", flush=True)
    cur = pd.DataFrame(curves)
    cur.index.name = "date"
    return pd.concat(parts, ignore_index=True), cur.iloc[::5]


def sample_rows(n: int) -> dict[str, np.ndarray]:
    fam = loc.sample_family(20_000, seed=FAMILY_SEED)[:n]  # the A37 ② members, in their order
    ctl = loc.sample_control(2000, 10, seed=CONTROL_SEED)[:n]
    return {"구조 가족": fam, "무작위": ctl}


def sample_study(panels: dict, P: dict, workers: int) -> pd.DataFrame:
    groups = sample_rows(P["n_sample"])
    tasks = []
    for era in panels:
        for setting in SAMPLE_SETTINGS:
            for g, W in groups.items():
                for a in range(0, len(W), P["chunk"]):
                    block = W[a:a + P["chunk"]]
                    tasks.append((era, setting, [_weights(r) for r in block], [f"{g} {a + j}" for j in range(len(block))]))
    t0 = time.time()
    out = []
    for k, df in enumerate(_map(tasks, panels, workers)):
        out.append(df)
        if (k + 1) % max(len(tasks) // 10, 1) == 0:
            print(f"  표본 {k + 1}/{len(tasks)} 묶음 ({time.time() - t0:.0f}s)", flush=True)
    s = pd.concat(out, ignore_index=True)
    s.insert(0, "묶음", s["행"].str.split(" ").str[0].replace({"구조": "구조 가족"}))
    return s


def group_table(s: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (g, setting), part in s.groupby(["묶음", "설정"], sort=False):
        wide = part.pivot(index="행", columns="기간", values="mdd")
        both = (wide >= sr.MDD_LINE).all(axis=1)
        row = {"묶음": g, "설정": setting, "수": len(wide), "두 기간 모두 선 안": float(both.mean())}
        for era in ERAS:
            e = part[part["기간"] == era]
            row[f"{era} 선 안"] = float((e["mdd"] >= sr.MDD_LINE).mean())
            row[f"{era} cagr 중앙"] = float(e["cagr"].median())
            row[f"{era} mdd 중앙"] = float(e["mdd"].median())
            if "cagr_2016" in e:
                row[f"{era} 2016~ cagr 중앙"] = float(e["cagr_2016"].median())
        rows.append(row)
    return pd.DataFrame(rows)


def verdict(named: pd.DataFrame, groups: pd.DataFrame) -> dict:
    def inside(name: str, setting: str) -> bool:
        part = named[(named["이름"] == name) & (named["설정"] == setting)]
        return bool(len(part) == len(ERAS) and (part["mdd"] >= sr.MDD_LINE).all())

    point = {c: inside(c, PRIMARY) for c in loc.CENTRES}
    robust = {c: point[c] and all(inside(c, f) for f in NEIGHBOURS) for c in loc.CENTRES}
    fam = groups[(groups["묶음"] == "구조 가족") & (groups["설정"] == PRIMARY)]
    share = float(fam["두 기간 모두 선 안"].iloc[0]) if len(fam) else float("nan")
    return {"점 판정 (필터 10, 두 기간 −15% 안)": point, "필터 견고 (8·12도)": robust,
            "구조 가족 두 기간 선 안 비율 (필터 10)": share, "구조 판정 (≥ 50%)": bool(share >= STRUCTURE_SHARE),
            "제안할 필터 판 후보": [c for c in loc.CENTRES if point[c] and robust[c]]}


def warmup_table(panel) -> pd.DataFrame:
    rows = []
    for c in loc.EQUITY:
        if c not in panel.close.columns:
            continue
        first = panel.close[c].first_valid_index()
        me = panel.close[c].loc[:pd.Timestamp(loc.DOTCOM_START)].dropna().resample("ME").last().dropna()
        rows.append({"상품": c, "첫 가격": None if first is None else first.date().isoformat(), "2000-09 전 월말 수": len(me),
                     "2000-09에 필터 작동": len(me) >= 10})
    return pd.DataFrame(rows)


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prices", choices=("real", "model"), default="real")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    args = ap.parse_args(argv)
    P = QUICK if args.quick else FULL
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    panels = {era: LOADERS[era](args.prices) for era in ERAS}
    warm = warmup_table(panels["2000-09"])
    warm.to_csv(out / "warmup.csv", index=False)
    print(warm.to_string())

    print("이름 붙은 배분")
    named, curves = named_study(panels, P)
    named.to_csv(out / "named.csv", index=False)
    curves.to_csv(out / "curves.csv.gz")
    check = named[(named["설정"] == "정적 분기") & named["이름"].isin(list(loc.CENTRES))].set_index(["이름", "기간"])["mdd"]
    print("정적 분기 MDD (A37과 같아야 함):", check.round(4).to_dict())

    print(f"표본 {P['n_sample']}개 × 2묶음 × {len(SAMPLE_SETTINGS)}설정 × 2기간, 작업자 {args.workers}")
    samples = sample_study(panels, P, args.workers)
    keep = ["묶음", "행", "기간", "설정", "cagr", "mdd", "cagr_2016", "turnover"] + [c for c in samples.columns if c.startswith("dd_")]
    samples[[c for c in keep if c in samples.columns]].to_csv(out / "samples.csv.gz", index=False, float_format="%.6g")
    groups = group_table(samples)
    groups.to_csv(out / "groups.csv", index=False)

    v = verdict(named, groups)
    summary = {"prices": args.prices, "quick": args.quick, "eras": ERAS, "end": sr.END, "safe": SAFE,
               "settings": SETTINGS, "sample_settings": SAMPLE_SETTINGS, "n_sample": P["n_sample"],
               "offsets": list(P["offsets"]), "verdict": v, "seconds": round(time.time() - t_all, 1)}
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    cols = ["이름", "기간", "설정", "cagr", "mdd", "cagr_2016", "날짜 최악 MDD", "꺼진 비중", "전환"]
    print(named[cols].to_string())
    print(groups.to_string())
    print(json.dumps(v, ensure_ascii=False, indent=2))
    print(f"끝 ({time.time() - t_all:.0f}s) → {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
