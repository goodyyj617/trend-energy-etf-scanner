"""Is the B2–B5 composition itself good? (allocation DECISIONS A39; memo 2026-10-27_composition.md)

    .venv/Scripts/python.exe scripts/aa_composition_study.py --out docs/research/allocation/2026-10-27_composition
    .venv/Scripts/python.exe scripts/aa_composition_study.py --quick --out %TEMP%\\aa_comp_quick      # ~2 min check

The four search allocations all hold nasdaq-100 and differ in everything else (lab/aa_composition.py):

  ① leave one out  each product out; its weight pro rata to the rest or to CD금리; both halves 2005–15 / 2016–26
  ② swap the rest  4×4: the row allocation's nasdaq weight + the column allocation's other products
  ③ simpler        the mean of the four (B 평균) and the nasdaq-100 × KRX gold × CD금리 grid (24 allocations)
  ④ choose on half the A31 sample ("B" subset: no leverage, no synthetic) re-scored on each half; the top 100
                   by CAGR inside −15 % on one half, judged on the other; where B2–B5 rank on each half
  ⑤ model vs real  the A33 table (2026-10-26_real_prices/products.csv) for the 13 products, no new run

①–③ also run from 2000-09 on the extended proxies (as A37 ④; all model before 2005). Same setting as stages
1–2: quarterly, 10 bp, prices real after listing (A34). Reading rules were fixed in the design before the
run. Writes small tables and figures into --out; nothing goes to lab_results/.
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
from lab import aa_composition as C  # noqa: E402
from lab import aa_local as loc  # noqa: E402
from lab import aa_search as srch  # noqa: E402
from lab import aa_search_results as sr  # noqa: E402
from lab import figures as figs  # noqa: E402
from lab.aa_assets import PRESETS  # noqa: E402

HALF = "2016-01-01"  # 2005–2015 against 2016–2026, as A37 ③
REFERENCE = "stable40"
PRODUCTS_TABLE = sr.REAL_DIR / "products.csv"
METRICS = ["cagr", "mdd", "cagr_h1", "mdd_h1", "cagr_h2", "mdd_h2", "dd_2008", "dd_2020", "dd_2022"]
METRICS_2000 = ["cagr", "mdd", "dd_dotcom"]
CHUNK = 6000


def named_sets() -> tuple[pd.DataFrame, np.ndarray]:
    """Every named allocation of ①–③ plus the centres and the reference, one row each."""
    meta, rows = [], []

    def add(test: str, name: str, variant: str, w: np.ndarray, **extra) -> None:
        meta.append({"시험": test, "배분": name, "변형": variant, **extra})
        rows.append(np.asarray(w, dtype=int))

    vec = {k: C.vector(v) for k, v in C.CENTRES.items()}
    for name, w in vec.items():
        add("중심", name, "등록 비중 그대로", w)
    add("참고", PRESETS[REFERENCE].name, "프리셋", C.vector(PRESETS[REFERENCE].weights))
    for name, w in vec.items():
        for mode, text in C.LOO_MODES.items():
            for code, v in C.leave_one_out(w, mode):
                add("①", name, f"{sr.name_of(code)} 빼기 ({text})", v, 뺀상품=code, 방식=mode)
    for r, c, v in C.swap_matrix():
        if r != c:
            add("②", r, f"나스닥은 {r}, 나머지는 {c}", v, 행=r, 열=c)
    add("③", "B 평균", "네 배분 비중의 평균", C.b_average())
    for label, v in C.skeleton_grid():
        add("③", "뼈대", label, v)
    W = np.vstack(rows)
    assert (W.sum(axis=1) == 100).all()
    df = pd.DataFrame(meta)
    df["비중"] = [sr.code_text(w) for w in W]
    return df, W


def evaluate(sim: srch.StaticSim, W: np.ndarray, label: str, cols: list[str], **kw) -> pd.DataFrame:
    parts = []
    for a in range(0, len(W), CHUNK):
        paths, turnover = sim.run(G.to_fraction(W[a:a + CHUNK]))
        parts.append(srch.summarize(paths, turnover, sim.dates, sim.cash, **kw)[cols])
        if len(W) > CHUNK:
            print(f"  {label}: {min(a + CHUNK, len(W)):,}/{len(W):,}", flush=True)
    return pd.concat(parts, ignore_index=True)


# --------------------------------------------------------------------------- reading ①–③

def loo_table(named: pd.DataFrame) -> pd.DataFrame:
    centre = named[named["시험"] == "중심"].set_index("배분")
    loo = named[named["시험"] == "①"].copy()
    for col in ("cagr", "mdd", "cagr_h1", "mdd_h1", "cagr_h2", "mdd_h2", "cagr_2000", "mdd_2000"):
        loo[f"Δ{col}"] = loo[col] - loo["배분"].map(centre[col])
    loo["판정"] = [C.loo_verdict((r["Δcagr_h1"], r["Δcagr_h2"]), (r["Δmdd_h1"], r["Δmdd_h2"])) for _, r in loo.iterrows()]
    loo["상품"] = loo["뺀상품"].map(sr.name_of)
    return loo


def swap_table(named: pd.DataFrame) -> tuple[pd.DataFrame, dict, dict]:
    centre = named[named["시험"] == "중심"].assign(행=lambda d: d["배분"], 열=lambda d: d["배분"])
    sw = pd.concat([centre, named[named["시험"] == "②"]], ignore_index=True)[["행", "열", "cagr", "mdd", "cagr_2000", "mdd_2000", "비중"]]
    v05 = C.swap_verdict(sw[["행", "열", "cagr", "mdd"]])
    v00 = C.swap_verdict(sw[["행", "열", "cagr_2000", "mdd_2000"]].rename(columns={"cagr_2000": "cagr", "mdd_2000": "mdd"}))
    return sw, v05, v00


def simple_table(named: pd.DataFrame) -> pd.DataFrame:
    centre_cagrs = named.loc[named["시험"] == "중심", "cagr"].to_numpy()
    s = named[named["시험"] == "③"].copy()
    s["판정"] = [C.simple_verdict(r.cagr, r.mdd, centre_cagrs) for r in s.itertuples()]
    return s


# --------------------------------------------------------------------------- ④ choose on one half

def reselect_study(sim: srch.StaticSim, P: dict, named: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    W1, meta = srch.sample_two_level(sr.CLASSES, n_class=P["n_class"], picks=P["picks"], seed=G.SEED, lev_class=sr.LEV_CLASS,
                                     lev_cap=G.LEV_CAP, equity_classes=sr.EQUITY_CLASSES, min_pct=G.MIN_PCT,
                                     max_products=G.MAX_PRODUCTS)
    assert list(meta["symbols"]) == sr.SYMBOLS
    W2 = srch.sample_sparse(sr.SYMBOLS, n=P["n_sparse"], seed=G.SEED + 1, k_range=(3, G.MAX_PRODUCTS), min_pct=G.MIN_PCT)
    W = np.unique(np.vstack([W1, W2]), axis=0)
    W = W[C.subset_b(W)]
    print(f"④ 표본: A31 표본 중 레버리지·가상 없는 {len(W):,}개")
    df = evaluate(sim, W, "④ 반기 평가", ["cagr", "mdd", "cagr_h1", "mdd_h1", "cagr_h2", "mdd_h2"], halves=(HALF,))
    shares = C.role_shares(W)
    tops, verdicts = [], {}
    for train, test, label in (("h1", "h2", "2005–2015에서 고름"), ("h2", "h1", "2016–2026에서 고름")):
        idx = C.reselect(df, train, test)
        t = pd.concat([df.iloc[idx].reset_index(drop=True), shares.iloc[idx].reset_index(drop=True)], axis=1)
        t.insert(0, "고른 기간", label)
        t.insert(1, "순위", np.arange(1, len(t) + 1))
        t["비중"] = [sr.code_text(w) for w in W[idx]]
        tops.append(t)
        verdicts[label] = {"판정": C.reselect_verdict(shares.iloc[idx]), "고른 수": int(len(idx)),
                           "나스닥 중앙": float(shares.iloc[idx]["나스닥"].median()), "금 중앙": float(shares.iloc[idx]["금"].median()),
                           "방어 주식 중앙": float(shares.iloc[idx]["방어 주식"].median()),
                           "현금·달러 중앙": float(shares.iloc[idx]["현금·달러"].median()),
                           f"다른 반기 MDD 중앙": float(df.iloc[idx][f"mdd_{test}"].median()),
                           f"다른 반기 −15% 안": float((df.iloc[idx][f"mdd_{test}"] >= sr.MDD_LINE).mean()),
                           f"다른 반기 CAGR 중앙": float(df.iloc[idx][f"cagr_{test}"].median())}
    top = pd.concat(tops, ignore_index=True)
    ranks = []
    centre = named[named["시험"].isin(["중심", "참고"])]
    for r in centre.itertuples():
        row = {"배분": r.배분}
        for h in ("h1", "h2"):
            ok = df[f"mdd_{h}"] >= sr.MDD_LINE
            row[f"{h} 선 안"] = bool(getattr(r, f"mdd_{h}") >= sr.MDD_LINE)
            row[f"{h} 순위"] = int(1 + (df.loc[ok, f"cagr_{h}"] > getattr(r, f"cagr_{h}")).sum())
            row[f"{h} 선 안 수"] = int(ok.sum())
        ranks.append(row)
    return top, pd.DataFrame(ranks), {"표본 수": int(len(W)), **verdicts}


# --------------------------------------------------------------------------- figures

def figures(out: Path, loo: pd.DataFrame, sw: pd.DataFrame, simple: pd.DataFrame, named: pd.DataFrame,
            top: pd.DataFrame | None, flags: pd.DataFrame) -> list[str]:
    if not figs.available():
        print("matplotlib 없음: 그림을 건너뜁니다.")
        return []
    plt = figs._plt()
    fdir = out / "figures"
    made = []

    # 1 leave one out: change in CAGR and MDD per half when each product is taken out (pro rata)
    pro = loo[loo["방식"] == "pro"]
    fig, axes = plt.subplots(len(C.NAMES), 2, figsize=(11, 2.0 + 0.62 * len(pro)), squeeze=False)
    for i, name in enumerate(C.NAMES):
        d = pro[pro["배분"] == name]
        y = np.arange(len(d))
        for j, (col, title) in enumerate((("cagr", "빼면 CAGR 변화 (%p)"), ("mdd", "빼면 MDD 변화 (%p, +가 얕아짐)"))):
            ax = axes[i, j]
            ax.barh(y - 0.2, 100 * d[f"Δ{col}_h1"], height=0.4, color=figs.COLORS[0], label="2005–2015")
            ax.barh(y + 0.2, 100 * d[f"Δ{col}_h2"], height=0.4, color=figs.COLORS[1], label="2016–2026")
            ax.axvline(0, color="#444", lw=0.8)
            tol = 0.3 if col == "cagr" else 0.5
            for x in (-tol, tol):
                ax.axvline(x, color="#999", lw=0.6, ls=":")
            ax.set_yticks(y, [f"{p} ({v})" for p, v in zip(d["상품"], d["판정"])], fontsize=7)
            ax.invert_yaxis()
            ax.grid(axis="x", alpha=0.3)
            ax.set_title(f"{name}: {title}", fontsize=9)
            if i == 0 and j == 0:
                ax.legend(fontsize=7, loc="lower right")
    axes[0, 0].set_title(f"① {C.NAMES[0]}: 빼면 CAGR 변화 (%p, 비례 배분, 점선 = 판정 기준)", fontsize=9)
    made.append(figs._save(fig, fdir / "1_leave_one_out.png").name)

    # 2 swap: 4×4 CAGR with MDD in each cell, 2005 and 2000-09
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))
    for ax, (c, m, title) in zip(axes, (("cagr", "mdd", "2005–2026"), ("cagr_2000", "mdd_2000", "2000-09–2026 (닷컴 포함)"))):
        grid = sw.pivot(index="행", columns="열", values=c).loc[list(C.NAMES), list(C.NAMES)] * 100
        mdd = sw.pivot(index="행", columns="열", values=m).loc[list(C.NAMES), list(C.NAMES)] * 100
        im = ax.imshow(grid.to_numpy(), cmap="Blues")
        for a in range(len(C.NAMES)):
            for b in range(len(C.NAMES)):
                out_line = mdd.iat[a, b] < 100 * sr.MDD_LINE
                ax.text(b, a, f"{grid.iat[a, b]:.1f}%\n{mdd.iat[a, b]:.1f}%", ha="center", va="center", fontsize=8,
                        color="#9b2c2c" if out_line else "#111", fontweight="bold" if a == b else "normal")
        ax.set_xticks(range(len(C.NAMES)), [f"나머지 {n}" for n in C.NAMES])
        ax.set_yticks(range(len(C.NAMES)), [f"나스닥 {n}" for n in C.NAMES])
        ax.set_title(f"② {title}: 위 CAGR, 아래 MDD (빨강 = −15% 밖)", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, label="CAGR (%)")
    made.append(figs._save(fig, fdir / "2_swap.png").name)

    # 3 simpler allocations against the four
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    sk = simple[simple["배분"] == "뼈대"]
    ax.scatter(100 * sk["mdd"], 100 * sk["cagr"], s=30, color="#a0aec0", label="뼈대: 나스닥100 × KRX금 × CD금리 (24개)")
    for r in sk.itertuples():
        n, g = r.변형.split(" · ")[0].split()[-1], r.변형.split(" · ")[1].split()[-1]
        ax.annotate(f"{n}/{g}", (100 * r.mdd, 100 * r.cagr), fontsize=6, color="#718096", xytext=(2, 2), textcoords="offset points")
    cen = named[named["시험"].isin(["중심", "참고"])]
    for k, r in enumerate(cen.itertuples()):
        ax.scatter(100 * r.mdd, 100 * r.cagr, s=90, marker="*" if r.시험 == "중심" else "D", color=figs.COLORS[k % 4] if r.시험 == "중심" else "#444",
                   zorder=3)
        ax.annotate(r.배분, (100 * r.mdd, 100 * r.cagr), fontsize=8, xytext=(4, 4), textcoords="offset points")
    av = simple[simple["배분"] == "B 평균"].iloc[0]
    ax.scatter(100 * av["mdd"], 100 * av["cagr"], s=90, marker="P", color="#2f855a", zorder=3, label="B 평균")
    ax.axvline(100 * sr.MDD_LINE, color="#c53030", ls="--", lw=1)
    ax.set_xlabel("MDD 2005–2026 (%)")
    ax.set_ylabel("CAGR 2005–2026 (%)")
    ax.set_title("③ 더 단순한 배분 (뼈대 점 이름 = 나스닥/금 %)", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.3)
    made.append(figs._save(fig, fdir / "3_simple.png").name)

    # 4 what the half-period picks hold
    if top is not None and len(top):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
        roles = ["나스닥", "방어 주식", "다른 주식", "채권", "금", "현금·달러"]
        for ax, (label, d) in zip(axes, top.groupby("고른 기간", sort=False)):
            ax.boxplot([d[r] for r in roles], showfliers=False)
            ax.set_xticks(range(1, len(roles) + 1), roles, fontsize=8)
            for r, (lo, hi) in C.RESELECT_RANGES.items():
                i = roles.index(r) + 1
                ax.fill_between([i - 0.4, i + 0.4], lo, hi, color="#f6ad55", alpha=0.25, lw=0)
            b = C.role_shares(np.vstack([C.vector(v) for v in C.CENTRES.values()]))
            for i, r in enumerate(roles, start=1):
                ax.scatter([i] * len(b), b[r], marker="*", color="#2b6cb0", s=30, zorder=3)
            ax.set_ylabel("비중 (%)")
            ax.set_title(f"④ {label} 상위 100개의 역할별 비중 (별 = B2–B5, 주황 = B 범위)", fontsize=9)
            ax.grid(axis="y", alpha=0.3)
        made.append(figs._save(fig, fdir / "4_reselect.png").name)

    # 5 real vs model per product
    f = flags.dropna(subset=["차이 %p"])
    fig, ax = plt.subplots(figsize=(8, 0.4 * len(f) + 1.5))
    y = np.arange(len(f))
    col = ["#c53030" if v == "모형 경고" else "#718096" if v == "비교 불가" else "#2f855a" for v in f["판정"]]
    ax.barh(y, f["차이 %p"], xerr=2 * f["차이 표준오차 %p"], color=col, alpha=0.85, error_kw={"lw": 0.8, "capsize": 2})
    ax.axvline(0, color="#444", lw=0.8)
    for x in (-1, 1):
        ax.axvline(x, color="#999", lw=0.6, ls=":")
    ax.set_yticks(y, [f"{p} ({h:.1f}년, 상관 {c:.2f})" for p, h, c in zip(f["상품"], f["겹친 해"], f["월 상관"])], fontsize=7)
    ax.invert_yaxis()
    ax.set_xlabel("실제 − 모형, 연 수익률 (%p, 막대 끝 = ±2 표준오차)")
    ax.set_title("⑤ 상장 후 실제 가격과 모형의 차이 (빨강 = 모형 경고)", fontsize=10)
    ax.grid(axis="x", alpha=0.3)
    made.append(figs._save(fig, fdir / "5_model_gap.png").name)
    return made


# --------------------------------------------------------------------------- main

def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prices", choices=("real", "model"), default="real")
    ap.add_argument("--quick", action="store_true", help="④ 표본만 작게 (나머지는 그대로)")
    ap.add_argument("--no-dotcom", action="store_true", help="2000-09 실행을 건너뜀 (점검용)")
    args = ap.parse_args(argv)
    P = G.QUICK if args.quick else G.FULL
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()

    print("⑤ 실제 가격과 모형 (A33 표)")
    flags = C.model_flags(pd.read_csv(PRODUCTS_TABLE), C.products_of())
    flags.to_csv(out / "model_flags.csv", index=False)

    meta, W = named_sets()
    krw = G.load(args.prices)
    sim = srch.StaticSim(krw, sr.SYMBOLS, sr.START, sr.END, rebalance=G.REBALANCE, cost_bps=G.COST_BPS)
    centres = np.vstack([C.vector(v) for v in C.CENTRES.values()])
    err = L.engine_check(krw, sim, np.vstack([centres, W[-3:]]))
    m05 = evaluate(sim, W, "①–③ 2005", METRICS, windows=sr.WINDOWS, halves=(HALF,))
    named = pd.concat([meta, m05], axis=1)

    err_x = None
    if args.no_dotcom:
        for c in METRICS_2000:
            named[f"{c}_2000"] = np.nan
    else:
        krwx, proxies = L.load_extended(args.prices)
        proxies.to_csv(out / "dotcom_proxies.csv", index=False)
        simx = srch.StaticSim(krwx, sr.SYMBOLS, loc.DOTCOM_START, sr.END, rebalance=G.REBALANCE, cost_bps=G.COST_BPS)
        err_x = L.engine_check(krwx, simx, centres, start=loc.DOTCOM_START)
        m00 = evaluate(simx, W, "①–③ 2000-09", METRICS_2000, windows={"dotcom": loc.DOTCOM_WINDOW})
        named = pd.concat([named, m00.add_suffix("_2000")], axis=1)
    named.to_csv(out / "named.csv", index=False)

    loo = loo_table(named)
    loo.drop(columns=["비중"]).to_csv(out / "loo.csv", index=False)
    sw, v05, v00 = swap_table(named)
    sw.to_csv(out / "swap.csv", index=False)
    simple = simple_table(named)
    simple.to_csv(out / "simple.csv", index=False)

    print("④ 반쪽 기간에서 다시 고르기")
    top, ranks, v4 = reselect_study(sim, P, named)
    top.to_csv(out / "reselect_top.csv.gz", index=False)
    ranks.to_csv(out / "reselect_ranks.csv", index=False)

    made = figures(out, loo, sw, simple, named, top, flags)
    pro = loo[loo["방식"] == "pro"]
    summary = {
        "params": {"prices": args.prices, "start": sr.START, "end": sr.END, "dotcom_start": loc.DOTCOM_START, "half": HALF,
                   "rebalance": G.REBALANCE, "cost_bps": G.COST_BPS, "sample": {k: P[k] for k in ("n_class", "picks", "n_sparse")},
                   "seed": G.SEED, "quick": args.quick},
        "engine_check_max_rel": err, "engine_check_max_rel_2000": err_x,
        "centres": named[named["시험"].isin(["중심", "참고"])][["배분", "cagr", "mdd", "cagr_2000", "mdd_2000"]].to_dict("records"),
        "① 판정 (비례 배분)": {name: pro[pro["배분"] == name].set_index("상품")["판정"].to_dict() for name in C.NAMES},
        "② 판정 2005": v05, "② 판정 2000-09 (참고)": v00,
        "③": simple[["배분", "변형", "cagr", "mdd", "cagr_2000", "mdd_2000", "판정"]].to_dict("records"),
        "③ 같은 일을 하는 뼈대 수": int((simple[simple["배분"] == "뼈대"]["판정"] == "같은 일을 한다").sum()),
        "④": v4, "④ B 순위": ranks.to_dict("records"),
        "⑤": flags[["상품", "판정"]].set_index("상품")["판정"].to_dict(),
        "figures": made, "seconds": round(time.time() - t_all, 1),
    }
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print(json.dumps({k: summary[k] for k in ("① 판정 (비례 배분)", "② 판정 2005", "③ 같은 일을 하는 뼈대 수", "④", "⑤")},
                     ensure_ascii=False, indent=2, default=float))
    print(f"끝 ({summary['seconds']:.0f}s) → {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
