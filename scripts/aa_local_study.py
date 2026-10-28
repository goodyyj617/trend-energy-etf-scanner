"""Local structure study around B2 · B3 · B4 (allocation DECISIONS A37; memo 2026-10-27_local_study.md).

    .venv/Scripts/python.exe scripts/aa_local_study.py --out docs/research/allocation/2026-10-27_local_study
    .venv/Scripts/python.exe scripts/aa_local_study.py --quick --out %TEMP%\\aa_local_quick     # ~1 min check
    .venv/Scripts/python.exe scripts/aa_local_study.py --dotcom --out docs/research/allocation/2026-10-27_local_study

Asks whether the structure B2–B4 share (nasdaq-led equity, gold, cash and short bonds instead of long
treasuries) is a wide region and a cause, not a lucky point (lab/aa_local.py):

  ① neighbours    5,000 random neighbours per centre at L1 distance 2–30 %p, plus same-role swaps
  ② structure     20,000 allocations built from rules R1–R3, each with one feature taken out (X1–X4),
                  and 20,000 unconstrained allocations from the same 21 products
  ③ time          crisis drawdowns, 2005–15 vs 2016–26 ranks, CSCV inside the family, 1-year block bootstrap
  ④ dot-com       (--dotcom) the US proxies carried back to 2000-09 with substitutes, the same allocations
                  from 2000-09 to 2026-10: all model before 2005, the only test on a period the search never saw

Same setting as stages 1–2 (2005-01-03 – 2026-10-16, quarterly, 10 bp, whole percents, min 3 %), prices
real after listing by default (A34). The only cut is MDD ≥ −15 %; the reading rules were fixed in the
design before the run. Writes small tables into --out; nothing goes to lab_results/.
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
from lab import aa_local as loc  # noqa: E402
from lab import aa_search as srch  # noqa: E402
from lab import aa_search_results as sr  # noqa: E402
from lab.aa_assets import PRESETS, PROXY_SYMBOLS  # noqa: E402
from lab.aa_engine import AllocConfig, run_allocation  # noqa: E402
from lab.aa_korea import krw_panel, load_bok_rate, load_usdkrw  # noqa: E402
from lab.aa_real import apply_source  # noqa: E402
from lab.data import download_panel  # noqa: E402

SEED = 2027
HALF = "2016-01-01"  # ③: 2005–2015 against 2016–2026
BOOT_BLOCK = 252
COMPARE = ("stable40", "aaa_2026")  # presets shown beside the centres in ③
# ④ also runs every other registered preset (gs_b2 / gs_b4 are B2 / B4); the first four were the comparisons fixed before
# the run, the rest were added after it so that each preset's note can say whether it survives 2000–2002. a41_ga (A41) holds
# KRX gold spot, which is outside this universe; its own 2000-09 result is in the A41 memo
DOTCOM_COMPARE = ("stable40", "aaa_2026", "sixty_forty", "sp500",
                  *[k for k in PRESETS if k not in ("stable40", "aaa_2026", "sixty_forty", "sp500", "gs_b2", "gs_b4", "a41_ga")])

FULL = dict(n_nb=5000, n_family=20_000, n_class=2000, picks=10, n_boot=1000, n_boot_rows=200, n_check=30, chunk=6000)
QUICK = dict(n_nb=150, n_family=600, n_class=60, picks=10, n_boot=40, n_boot_rows=20, n_check=4, chunk=3000)

METRIC_COLS = ["cagr", "mdd", "sharpe_excess", "turnover", "cagr_h1", "mdd_h1", "cagr_h2", "mdd_h2",
               "dd_2008", "dd_2011", "dd_2018", "dd_2020", "dd_2022"]


def evaluate(sim: srch.StaticSim, W: np.ndarray, chunk: int, label: str, monthly: bool = False
             ) -> tuple[pd.DataFrame, np.ndarray | None]:
    parts, months = [], []
    t0 = time.time()
    for a in range(0, len(W), chunk):
        paths, turnover = sim.run(G.to_fraction(W[a:a + chunk]))
        parts.append(srch.summarize(paths, turnover, sim.dates, sim.cash, windows=sr.WINDOWS, halves=(HALF,))[METRIC_COLS])
        if monthly:
            months.append(srch.monthly_returns(paths, sim.dates).astype(np.float32))
        print(f"  {label}: {min(a + chunk, len(W)):,}/{len(W):,} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.concat(parts, ignore_index=True)
    shares = sr.class_shares(W)
    df["주식"], df["금"], df["나스닥"] = shares["주식"].to_numpy(), W[:, [loc.COL[s] for s in loc.GOLD]].sum(axis=1), \
        W[:, loc.COL[loc.NASDAQ]]
    df["장기채"], df["현금"] = W[:, loc.COL[loc.LONG]], W[:, loc.COL[loc.CD]]
    df["상품 수"] = (W > 0).sum(axis=1)
    return df, (np.vstack(months) if monthly else None)


def engine_check(krw, sim: srch.StaticSim, W: np.ndarray, start: str = sr.START) -> float:
    paths, turnover = sim.run(G.to_fraction(W))
    worst = 0.0
    for k, w in enumerate(W):
        weights = sr.weights_of(w)
        res = run_allocation(krw, AllocConfig(f"check{k}", weights, start, sr.END, rebalance=G.REBALANCE,
                                              cost_bps=G.COST_BPS))
        worst = max(worst, float(np.max(np.abs(paths[k] / res.equity.to_numpy() - 1.0))),
                    abs(turnover[k] - res.turnover) / max(res.turnover, 1e-12))
    print(f"엔진 대조: {len(W)}개, 최대 상대 오차 {worst:.2e}")
    if worst > 1e-6:
        raise SystemExit("시뮬레이터가 엔진과 다릅니다. 멈춥니다.")
    return worst


def within(s: pd.Series) -> float:
    return float((s >= sr.MDD_LINE).mean()) if len(s) else float("nan")


def q(s: pd.Series, p: float) -> float:
    return float(s.quantile(p)) if len(s) else float("nan")


# --------------------------------------------------------------------------- ① neighbours

def neighbour_study(sim, P, centres: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows, meta = [], []
    for k, (name, c) in enumerate(zip(loc.CENTRES, centres)):
        rows.append(c[None, :])
        meta.append(pd.DataFrame({"중심": name, "종류": "중심", "바꾸기": "", "거리": 0}, index=[0]))
        nb = loc.perturb(c, P["n_nb"], seed=SEED + k)
        rows.append(nb)
        meta.append(pd.DataFrame({"중심": name, "종류": "이웃", "바꾸기": "", "거리": loc.l1(nb, c)}))
        subs = loc.substitutions(c)
        if subs:
            rows.append(np.vstack([v for _, v in subs]))
            meta.append(pd.DataFrame({"중심": name, "종류": "바꾸기", "바꾸기": [s for s, _ in subs],
                                      "거리": [int(loc.l1(v, c)[0]) for _, v in subs]}))
    W = np.vstack(rows)
    df, _ = evaluate(sim, W, P["chunk"], "① 이웃")
    df = pd.concat([pd.concat(meta, ignore_index=True), df], axis=1)
    df["거리 구간"] = loc.dist_bin(df["거리"].to_numpy())
    df["비중"] = [sr.code_text(w) for w in W]
    nb = df[df["종류"] == "이웃"]
    bins = (nb.groupby(["중심", "거리 구간"], sort=False)
            .agg(수=("mdd", "size"), 선안=("mdd", within), cagr_중앙=("cagr", "median"), cagr_하위10=("cagr", lambda s: q(s, 0.1)),
                 mdd_최악=("mdd", "min"), mdd_중앙=("mdd", "median"))
            .reset_index())
    order = {b: i for i, b in enumerate(loc.dist_bin(np.array(loc.DIST_BINS[1:])))}
    bins = bins.sort_values(["중심", "거리 구간"], key=lambda s: s.map(order) if s.name == "거리 구간" else s,
                            ignore_index=True)
    bins["선안"] = bins["선안"].astype(float)
    near = (nb[nb["거리"] <= 10].groupby("중심").agg(수=("mdd", "size"), 선안=("mdd", within),
                                                    cagr_중앙=("cagr", "median")).reset_index())
    near["거리 구간"] = "0–10 (읽는 기준)"
    bins = pd.concat([bins, near], ignore_index=True)
    subs = df[df["종류"].isin(["중심", "바꾸기"])][["중심", "종류", "바꾸기", "cagr", "mdd", "dd_2008", "dd_2020", "dd_2022"]]
    return df, bins, subs


# --------------------------------------------------------------------------- ② structure

def structure_study(sim, P) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    F = loc.sample_family(P["n_family"], seed=SEED + 10)
    Wc = loc.sample_control(P["n_class"], P["picks"], seed=SEED + 11)
    blocks, meta = [F], [pd.DataFrame({"묶음": "구조 가족", "원본": np.arange(len(F))})]
    for x in loc.ABLATIONS:
        out = [(i, loc.ablate(w, x)) for i, w in enumerate(F)]
        out = [(i, v) for i, v in out if v is not None]
        blocks.append(np.vstack([v for _, v in out]))
        meta.append(pd.DataFrame({"묶음": x, "원본": [i for i, _ in out]}))
    blocks.append(Wc)
    meta.append(pd.DataFrame({"묶음": "무작위", "원본": -1}, index=range(len(Wc))))
    W = np.vstack(blocks)
    df, monthly = evaluate(sim, W, P["chunk"], "② 구조", monthly=True)
    df = pd.concat([pd.concat(meta, ignore_index=True), df], axis=1)
    r = loc.rules(W)
    df["규칙 모두"] = r["all"].to_numpy()
    df["비중"] = [sr.code_text(w) if g in ("구조 가족", "무작위") else "" for w, g in zip(W, df["묶음"])]
    return df, W, monthly


def group_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    order = ["구조 가족", *loc.ABLATIONS, "무작위", "무작위 중 규칙 만족", "무작위 중 규칙 불만족"]
    for g in order:
        if g.startswith("무작위 중"):
            part = df[(df["묶음"] == "무작위") & (df["규칙 모두"] == (not g.endswith("불만족")))]
        else:
            part = df[df["묶음"] == g]
        rows.append({"묶음": g, "설명": {"구조 가족": "R1–R3 모두", **loc.ABLATIONS}.get(g, "같은 21개 상품, 규칙 없음"),
                     "수": len(part), "선안": within(part["mdd"]),
                     "cagr_중앙": q(part["cagr"], 0.5), "cagr_하위10": q(part["cagr"], 0.1), "cagr_상위10": q(part["cagr"], 0.9),
                     "mdd_중앙": q(part["mdd"], 0.5), "mdd_하위10": q(part["mdd"], 0.1),
                     **{f"{w}_선안": within(part[f"dd_{w}"]) for w in ("2008", "2020", "2022")},
                     **{f"{w}_중앙": q(part[f"dd_{w}"], 0.5) for w in ("2008", "2020", "2022")}})
    return pd.DataFrame(rows)


def paired_table(df: pd.DataFrame) -> pd.DataFrame:
    """For each ablation: how many family members that were inside the line fall out, and the CAGR change."""
    fam = df[df["묶음"] == "구조 가족"].set_index("원본")
    rows = []
    for x in loc.ABLATIONS:
        part = df[df["묶음"] == x].set_index("원본")
        base = fam.loc[part.index]
        inside = base["mdd"] >= sr.MDD_LINE
        rows.append({"묶음": x, "설명": loc.ABLATIONS[x], "짝 수": len(part),
                     "원본 선안": float(inside.mean()), "바꾼 뒤 선안": float((part["mdd"] >= sr.MDD_LINE).mean()),
                     "선 안→밖": float(((part["mdd"] < sr.MDD_LINE) & inside).sum() / max(inside.sum(), 1)),
                     "cagr 변화 중앙(%p)": float((part["cagr"] - base["cagr"]).median() * 100),
                     "mdd 변화 중앙(%p)": float((part["mdd"] - base["mdd"]).median() * 100)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- ③ time

def spearman(a: pd.Series, b: pd.Series) -> float:
    return float(a.rank().corr(b.rank())) if len(a) > 2 else float("nan")


def time_study(sim, P, sdf: pd.DataFrame, W: np.ndarray, monthly: np.ndarray, centres: np.ndarray
               ) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    halves = []
    for g in ["구조 가족", *loc.ABLATIONS, "무작위"]:
        part = sdf[sdf["묶음"] == g]
        halves.append({"묶음": g, "수": len(part), "CAGR 순위 상관": spearman(part["cagr_h1"], part["cagr_h2"]),
                       "MDD 순위 상관": spearman(part["mdd_h1"], part["mdd_h2"]),
                       "앞 선안": within(part["mdd_h1"]), "뒤 선안": within(part["mdd_h2"]),
                       "앞 cagr 중앙": q(part["cagr_h1"], 0.5), "뒤 cagr 중앙": q(part["cagr_h2"], 0.5)})
    halves = pd.DataFrame(halves)
    pbo = {}
    for g in ("구조 가족", "무작위"):
        m = (sdf["묶음"] == g).to_numpy()
        res = srch.pbo_selection(monthly[m])
        pbo[g] = {"pbo": res["pbo"], "splits": res["splits"], "oos_rank_median": float(np.median(res["oos_rank"])),
                  "pick_oos_within": float((res["pick_oos_mdd"] >= sr.MDD_LINE).mean()),
                  "distinct_picks": int((res["picked"] > 0).sum())}
        print(f"  PBO {g}: {res['pbo']:.2f} ({res['splits']} 분할)")
    # bootstrap rows: centres, comparison presets, a random sample of the family and of the control
    rng = np.random.default_rng(SEED + 20)
    fam_idx = np.where(sdf["묶음"] == "구조 가족")[0]
    ctl_idx = np.where(sdf["묶음"] == "무작위")[0]
    pick_f = rng.choice(fam_idx, size=min(P["n_boot_rows"], len(fam_idx)), replace=False)
    pick_c = rng.choice(ctl_idx, size=min(P["n_boot_rows"], len(ctl_idx)), replace=False)
    pres = np.vstack([loc.vector(PRESETS[k].weights) for k in COMPARE])
    Wb = np.vstack([centres, pres, W[pick_f], W[pick_c]])
    labels = [*loc.CENTRES, *[PRESETS[k].name for k in COMPARE], *["구조 가족"] * len(pick_f), *["무작위"] * len(pick_c)]
    paths, _ = sim.run(G.to_fraction(Wb))
    t0 = time.time()
    B = loc.block_bootstrap_mdd(paths, P["n_boot"], block=BOOT_BLOCK, seed=SEED + 21)
    print(f"  부트스트랩 {P['n_boot']}회 × {len(Wb)}개 ({time.time() - t0:.0f}s)")
    boot = pd.DataFrame({"이름": labels, "선 넘을 확률": (B < sr.MDD_LINE).mean(axis=1), "mdd_중앙": np.median(B, axis=1),
                         "mdd_하위10": np.quantile(B, 0.1, axis=1), "mdd_하위5": np.quantile(B, 0.05, axis=1)})
    boot_hist = pd.DataFrame(B[:len(loc.CENTRES) + len(COMPARE)].T, columns=labels[:len(loc.CENTRES) + len(COMPARE)])
    # crisis table for the centres and the comparison presets
    cdf, _ = evaluate(sim, np.vstack([centres, pres]), P["chunk"], "③ 위기")
    cdf.insert(0, "이름", [*loc.CENTRES, *[PRESETS[k].name for k in COMPARE]])
    return halves, pbo, boot, boot_hist, cdf


# --------------------------------------------------------------------------- ④ dot-com

def load_extended(prices: str):
    """The A31 KRW panel with every proxy carried back to DOTCOM_START (lab.aa_local.extend_history)."""
    usd, failed = download_panel(sorted(set(PROXY_SYMBOLS) | {"IEF", "TLT"}), end=G.DATA_END)
    subs, failed_s = download_panel(sorted({c for v in loc.SUBSTITUTES.values() for c in v}), end=G.DATA_END)
    if failed or failed_s:
        print("받지 못한 종목:", failed + failed_s)
    ext, table = loc.extend_history(usd, subs, loc.DOTCOM_START)
    krw, unavailable = krw_panel(ext, load_usdkrw(), load_bok_rate(), products=sr.ALL_PRODUCTS)
    if unavailable:
        raise SystemExit(f"대용 데이터가 없는 상품: {unavailable}")
    if prices == "real":
        krw, info, msg = apply_source(krw, "real")
        if info is None:
            raise SystemExit(msg)
        print(msg)
    first = krw.close[sr.SYMBOLS].apply(lambda s: s.first_valid_index())
    late = first[first > pd.Timestamp(loc.DOTCOM_START)]
    if len(late):
        raise SystemExit(f"{loc.DOTCOM_START} 이후에 데이터가 시작되는 상품: {late.to_dict()}  (이음 표: {table.to_dict('records')})")
    return krw, table


def dotcom_study(P, prices: str, out: Path, loader=load_extended) -> dict:
    krw, table = loader(prices)
    table.to_csv(out / "dotcom_proxies.csv", index=False)
    sim = srch.StaticSim(krw, sr.SYMBOLS, loc.DOTCOM_START, sr.END, rebalance=G.REBALANCE, cost_bps=G.COST_BPS)
    centres = np.vstack([loc.vector(w) for w in loc.CENTRES.values()])
    err = engine_check(krw, sim, np.vstack([centres, loc.sample_family(P["n_check"], seed=SEED + 99)]), start=loc.DOTCOM_START)
    windows = {"dotcom": loc.DOTCOM_WINDOW, **sr.WINDOWS}

    def run(W: np.ndarray, label: str) -> pd.DataFrame:
        parts = []
        for a in range(0, len(W), P["chunk"]):
            paths, turnover = sim.run(G.to_fraction(W[a:a + P["chunk"]]))
            df = srch.summarize(paths, turnover, sim.dates, sim.cash, windows=windows, extra_starts=(sr.START,))
            parts.append(df[["cagr", "mdd", "dd_dotcom", "dd_2008", "dd_2020", "dd_2022", "cagr_2005", "mdd_2005"]])
            print(f"  ④ {label}: {min(a + P['chunk'], len(W)):,}/{len(W):,}", flush=True)
        return pd.concat(parts, ignore_index=True)

    names = [*loc.CENTRES, *[PRESETS[k].name for k in DOTCOM_COMPARE]]
    Wn = np.vstack([centres, *[loc.vector(PRESETS[k].weights) for k in DOTCOM_COMPARE]])
    named = run(Wn, "중심·프리셋")
    named.insert(0, "이름", names)
    paths, _ = sim.run(G.to_fraction(Wn))
    trough = [sim.dates[int(np.argmin(p / np.maximum.accumulate(p)))] for p in paths]
    named["최저점"] = [d.date().isoformat() for d in trough]
    named.to_csv(out / "dotcom.csv", index=False)
    keep = sim.dates <= pd.Timestamp("2006-12-31")
    curves = pd.DataFrame(paths[:, keep].T, index=sim.dates[keep], columns=names)
    curves.index.name = "date"
    curves.iloc[::5].to_csv(out / "dotcom_paths.csv.gz")

    F = loc.sample_family(P["n_family"], seed=SEED + 10)  # the same members as ②
    groups = {"구조 가족": F}
    for x in loc.ABLATIONS:
        v = [loc.ablate(w, x) for w in F]
        groups[x] = np.vstack([w for w in v if w is not None])
    groups["무작위"] = loc.sample_control(P["n_class"], P["picks"], seed=SEED + 11)
    rows = []
    for g, W in groups.items():
        df = run(W, g)
        rows.append({"묶음": g, "수": len(df), "2000~ 선안": within(df["mdd"]), "닷컴 창 선안": within(df["dd_dotcom"]),
                     "2005~ 선안": within(df["mdd_2005"]), "닷컴 창 낙폭 중앙": q(df["dd_dotcom"], 0.5),
                     "닷컴 창 낙폭 하위10": q(df["dd_dotcom"], 0.1), "2000~ cagr 중앙": q(df["cagr"], 0.5),
                     "2000~ mdd 중앙": q(df["mdd"], 0.5)})
    gdf = pd.DataFrame(rows)
    gdf.to_csv(out / "dotcom_groups.csv", index=False)
    summary = {"prices": prices, "start": loc.DOTCOM_START, "window": loc.DOTCOM_WINDOW, "engine_check_max_rel": err,
               "named": named[["이름", "cagr", "mdd", "dd_dotcom", "mdd_2005", "최저점"]].to_dict("records"),
               "groups": gdf.to_dict("records"), "proxies": table.to_dict("records")}
    (out / "dotcom_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(named[["이름", "cagr", "mdd", "dd_dotcom", "mdd_2005", "최저점"]].to_string())
    print(gdf.to_string())
    return summary


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prices", choices=("real", "model"), default="real")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--dotcom", action="store_true", help="④만: 2000-09부터 늘린 대용 데이터로")
    args = ap.parse_args(argv)
    P = QUICK if args.quick else FULL
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    if args.dotcom:
        dotcom_study(P, args.prices, out)
        print(f"끝 ({time.time() - t_all:.0f}s) → {out}")
        return
    krw = G.load(args.prices)
    sim = srch.StaticSim(krw, sr.SYMBOLS, sr.START, sr.END, rebalance=G.REBALANCE, cost_bps=G.COST_BPS)
    centres = np.vstack([loc.vector(w) for w in loc.CENTRES.values()])
    check_rows = np.vstack([centres, loc.sample_family(P["n_check"], seed=SEED + 99)])
    err = engine_check(krw, sim, check_rows)

    print("① 이웃 지도")
    nb, bins, subs = neighbour_study(sim, P, centres)
    nb.to_csv(out / "neighbours.csv.gz", index=False)
    bins.to_csv(out / "neighbour_bins.csv", index=False)
    subs.to_csv(out / "substitutions.csv", index=False)

    print("② 구조 규칙")
    sdf, W, monthly = structure_study(sim, P)
    sdf.to_csv(out / "structure.csv.gz", index=False)
    groups = group_table(sdf)
    groups.to_csv(out / "groups.csv", index=False)
    paired = paired_table(sdf)
    paired.to_csv(out / "ablations.csv", index=False)

    print("③ 시간")
    halves, pbo, boot, boot_hist, crisis = time_study(sim, P, sdf, W, monthly, centres)
    halves.to_csv(out / "halves.csv", index=False)
    boot.to_csv(out / "bootstrap.csv", index=False)
    boot_hist.to_csv(out / "bootstrap_mdd.csv.gz", index=False)
    crisis.to_csv(out / "crisis.csv", index=False)

    model_rows = None
    if args.prices == "real":  # the centres on the model only, for the real-vs-model line in the memo
        kmodel = G.load("model")
        msim = srch.StaticSim(kmodel, sr.SYMBOLS, sr.START, sr.END, rebalance=G.REBALANCE, cost_bps=G.COST_BPS)
        mdf, _ = evaluate(msim, centres, P["chunk"], "모형만 중심")
        mdf.insert(0, "이름", list(loc.CENTRES))
        mdf.to_csv(out / "centres_model.csv", index=False)
        model_rows = mdf[["이름", "cagr", "mdd"]].to_dict("records")

    g = groups.set_index("묶음")
    summary = {
        "params": {**P, "seed": SEED, "prices": args.prices, "start": sr.START, "end": sr.END, "half": HALF,
                   "rebalance": G.REBALANCE, "cost_bps": G.COST_BPS, "boot_block": BOOT_BLOCK, "universe": loc.UNIVERSE},
        "engine_check_max_rel": err,
        "centres": crisis.head(len(loc.CENTRES))[["이름", "cagr", "mdd"]].to_dict("records"),
        "centres_model": model_rows,
        "reading": {
            "① 0–10 선안": bins[bins["거리 구간"].str.startswith("0–10")].set_index("중심")["선안"].to_dict(),
            "② 가족 선안": float(g.loc["구조 가족", "선안"]), "② 무작위 선안": float(g.loc["무작위", "선안"]),
            "② 제거 후 선안": {x: float(g.loc[x, "선안"]) for x in loc.ABLATIONS},
            "③ 가족 PBO": pbo["구조 가족"]["pbo"], "③ 무작위 PBO": pbo["무작위"]["pbo"],
            "③ 가족 CAGR 순위 상관": float(halves.set_index("묶음").loc["구조 가족", "CAGR 순위 상관"]),
        },
        "pbo": pbo,
        "seconds": round(time.time() - t_all, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps(summary["reading"], ensure_ascii=False, indent=2, default=float))
    print(f"끝 ({summary['seconds']:.0f}s) → {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
