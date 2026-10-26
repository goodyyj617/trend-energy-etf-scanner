"""Global static search over 30 Korean-listed products (DECISIONS A31; memo 2026-10-25_wider_search_design.md §2).

    .venv/Scripts/python.exe scripts/aa_global_search.py                       # full run (~15 min, 4 cores)
    .venv/Scripts/python.exe scripts/aa_global_search.py --quick               # small smoke run (~1 min)
    .venv/Scripts/python.exe scripts/aa_global_search.py --out DIR --full-out DIR
    .venv/Scripts/python.exe scripts/aa_global_search.py --app-files FULL.csv.gz  # only rebuild the app page's files

The only mandatory cut is the full-period MDD line (−15 %). Everything else (half periods, crisis
windows, holdout, rolling windows, neighbours, timing, PBO) is written as a column or a ⚠ flag so the
user can judge CAGR against MDD themselves.

Reads the yfinance cache in lab_data/yf (downloads what is missing), lab/aa_usdkrw.csv.gz and
lab/aa_bok_rate.csv. Writes tables, a scatter and a summary into --out (small files, meant for the
repo) and the full candidate table into --full-out (large; project files or a local folder). The app page
'전역 탐색 결과' reads passing.csv.gz (every allocation within the line) and cloud.csv.gz (a sample of the rest).
Nothing is written to lab_results/ and nothing is added to the lab's run log.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import aa_search as srch  # noqa: E402
from lab.aa_assets import CASH, PRESETS, PROXY_SYMBOLS  # noqa: E402
from lab.aa_engine import AllocConfig, run_allocation  # noqa: E402
from lab.aa_korea import krw_panel, load_bok_rate, load_usdkrw  # noqa: E402
from lab.aa_real import apply_source  # noqa: E402
from lab.aa_search_results import (  # noqa: E402  (the A31 universe, shared with the app page)
    ALL_PRODUCTS, CLASSES, END, EQUITY_CLASSES, LEV_CLASS, MDD_LINE, START, SYMBOLS, WINDOWS, app_tables, class_shares,
    describe, flags, name_of,
)
from lab.aa_validate import TIMING_OFFSETS  # noqa: E402
from lab.data import download_panel  # noqa: E402

# --------------------------------------------------------------------------- registered parameters (A31)

# START, END = 2005-01-03, 2026-10-16 (lab.aa_search_results; the USD/KRW snapshot ends 2026-10-16)
DATA_END = "2026-10-23"
SEED = 2026
REBALANCE, COST_BPS = "Q", 10.0
HALF = "2015-01-01"
EXTRA_STARTS = ("2008-01-01", "2010-01-01")
HOLDOUT_SHARE = 0.15
ROLL_YEARS = 5  # ROLL_MIN_SHARE 0.80, MDD_LINE −15%, TIMING_LINE −20% (A30) in lab.aa_search_results
LEV_CAP = 0.20
MIN_PCT, MAX_PRODUCTS = 3, 10

FULL = dict(n_class=20_000, picks=10, n_sparse=50_000, top_refine=300, top_roll=2000, refine_rounds=3, chunk=8000)
QUICK = dict(n_class=400, picks=4, n_sparse=1500, top_refine=30, top_roll=200, refine_rounds=2, chunk=4000)


def pct(x: float, digits: int = 1) -> str:
    return "nan" if x != x else f"{x * 100:.{digits}f}"


# --------------------------------------------------------------------------- data and self-check

def load(prices: str = "model"):
    """The A31 KRW panel; prices="real" swaps in real Korean prices after listing (A33, lab/aa_real.py)."""
    usd, failed = download_panel(sorted(set(PROXY_SYMBOLS) | {"IEF", "TLT"}), end=DATA_END)
    if failed:
        print("받지 못한 종목:", failed)
    krw, unavailable = krw_panel(usd, load_usdkrw(), load_bok_rate(), products=ALL_PRODUCTS)
    if unavailable:
        raise SystemExit(f"대용 데이터가 없는 상품: {unavailable}")
    if prices == "real":
        krw, info, msg = apply_source(krw, "real")
        if info is None:
            raise SystemExit(msg)
        print(msg)
    first = krw.close[SYMBOLS].apply(lambda s: s.first_valid_index())
    late = first[first > pd.Timestamp(START)]
    if len(late):
        raise SystemExit(f"{START} 이후에 데이터가 시작되는 상품: {late.to_dict()}")
    return krw


def preset_matrix() -> tuple[np.ndarray, list[str]]:
    rows, names = [], []
    for p in PRESETS.values():
        if p.rule is not None:  # the search is static; presets with a signal (A38) are not points on it
            continue
        if any(s not in SYMBOLS for s in p.weights):  # a preset holding CASH or a product outside the universe is skipped
            print(f"프리셋 건너뜀(유니버스 밖 상품 또는 현금): {p.name}")
            continue
        w = np.zeros(len(SYMBOLS) + 1)
        for s, v in p.weights.items():
            w[SYMBOLS.index(s)] = v
        rows.append(w / w.sum())
        names.append(p.name)
    return np.vstack(rows), names


def self_check(krw, sim: srch.StaticSim, W: np.ndarray, names: list[str], tol: float = 1e-6) -> float:
    """The vectorised simulator must reproduce the engine on every registered preset."""
    paths, turnover = sim.run(W)
    worst = 0.0
    for k, name in enumerate(names):
        weights = {s: v * 100 for s, v in zip(SYMBOLS + [CASH], W[k]) if v > 0}
        res = run_allocation(krw, AllocConfig(name, weights, START, END, rebalance=REBALANCE, cost_bps=COST_BPS))
        rel = float(np.max(np.abs(paths[k] / res.equity.to_numpy() - 1.0)))
        rel_t = abs(turnover[k] - res.turnover) / max(res.turnover, 1e-12)
        worst = max(worst, rel, rel_t)
    print(f"자체 점검: 프리셋 {len(names)}개, 엔진 대비 최대 상대 오차 {worst:.2e} (허용 {tol:.0e})")
    if worst > tol:
        raise SystemExit("시뮬레이터가 엔진과 다릅니다. 탐색을 멈춥니다.")
    return worst


# --------------------------------------------------------------------------- evaluation

def to_fraction(W: np.ndarray) -> np.ndarray:
    """Whole-percent product weights → fractions with the (always zero) cash column appended."""
    F = np.zeros((W.shape[0], W.shape[1] + 1))
    F[:, :-1] = W / 100.0
    return F


def evaluate(sim: srch.StaticSim, W: np.ndarray, chunk: int, label: str) -> tuple[pd.DataFrame, np.ndarray]:
    """Metrics for every row of W (whole percents) plus float32 monthly returns (rows × months)."""
    parts, monthly = [], []
    n_hold = int(round(sim.n_days * HOLDOUT_SHARE))
    t0 = time.time()
    for a in range(0, len(W), chunk):
        F = to_fraction(W[a:a + chunk])
        paths, turnover = sim.run(F)
        df = srch.summarize(paths, turnover, sim.dates, sim.cash, windows=WINDOWS, halves=(HALF,),
                            extra_starts=EXTRA_STARTS, holdout_share=HOLDOUT_SHARE)
        df["cagr_sel"], df["mdd_sel"] = srch.cagr_mdd(paths[:, :sim.n_days - n_hold])
        parts.append(df)
        monthly.append(srch.monthly_returns(paths, sim.dates).astype(np.float32))
        done = min(a + chunk, len(W))
        print(f"  {label}: {done:,}/{len(W):,} ({time.time() - t0:.0f}s)", flush=True)
    df = pd.concat(parts, ignore_index=True)
    df = pd.concat([df, class_shares(W)], axis=1)
    df["통과"] = df["mdd"] >= MDD_LINE
    df["비중"] = [describe(w) for w in W]
    return df, np.vstack(monthly)


def refine(sim: srch.StaticSim, seeds: np.ndarray, rounds: int, chunk: int) -> tuple[np.ndarray, pd.DataFrame]:
    """Greedy 5 %p moves on each seed (best CAGR among neighbours within the MDD line), at most `rounds`
    rounds. Returns the refined weights and a table with the seed's neighbour worst MDD (round 1),
    rounds used and the CAGR gain."""
    cur = seeds.copy()
    cur_cagr = np.full(len(cur), -np.inf)
    nb_worst = np.full(len(cur), np.nan)
    rounds_used = np.zeros(len(cur), dtype=int)
    active = np.ones(len(cur), dtype=bool)
    F0 = to_fraction(cur)
    p0, _ = sim.run(F0)
    cur_cagr, _ = srch.cagr_mdd(p0)
    start_cagr = cur_cagr.copy()
    for r in range(rounds):
        idx = np.where(active)[0]
        if not len(idx):
            break
        groups = [srch.neighbours(cur[i], step=5, min_pct=MIN_PCT) for i in idx]
        owner = np.concatenate([np.full(len(g), i) for i, g in zip(idx, groups)])
        Wn = np.vstack([v for g in groups for v in g])
        cagr_n = np.empty(len(Wn))
        mdd_n = np.empty(len(Wn))
        for a in range(0, len(Wn), chunk):
            p, _ = sim.run(to_fraction(Wn[a:a + chunk]))
            cagr_n[a:a + chunk], mdd_n[a:a + chunk] = srch.cagr_mdd(p)
        print(f"  다듬기 {r + 1}회: 후보 {len(idx)}개, 이웃 {len(Wn):,}개", flush=True)
        for i in idx:
            m = owner == i
            if r == 0:
                nb_worst[i] = mdd_n[m].min()
            ok = m & (mdd_n >= MDD_LINE)
            if not ok.any():
                active[i] = False
                continue
            j = np.where(ok)[0][np.argmax(cagr_n[ok])]
            if cagr_n[j] > cur_cagr[i] + 1e-12:
                cur[i], cur_cagr[i] = Wn[j], cagr_n[j]
                rounds_used[i] += 1
            else:
                active[i] = False
    table = pd.DataFrame({"nb_worst_mdd": nb_worst, "refine_rounds": rounds_used, "refine_gain": cur_cagr - start_cagr})
    return cur, table


def timing_worst(sim_factory, W: np.ndarray, chunk: int) -> np.ndarray:
    """Worst MDD over the A30 rebalance-date offsets for each row."""
    worst = np.full(len(W), np.inf)
    for off in TIMING_OFFSETS:
        sim = sim_factory(off)
        for a in range(0, len(W), chunk):
            p, _ = sim.run(to_fraction(W[a:a + chunk]))
            _, m = srch.cagr_mdd(p)
            worst[a:a + chunk] = np.minimum(worst[a:a + chunk], m)
    return worst


def rolling(sim: srch.StaticSim, W: np.ndarray, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    share = np.empty(len(W))
    med = np.empty(len(W))
    for a in range(0, len(W), chunk):
        p, _ = sim.run(to_fraction(W[a:a + chunk]))
        share[a:a + chunk], med[a:a + chunk] = srch.rolling_window_share(p, sim.dates, years=ROLL_YEARS, mdd_line=MDD_LINE)
    return share, med


# --------------------------------------------------------------------------- outputs

TABLE_COLS = ["cagr", "mdd", "sharpe_excess", "calmar", "turnover", "상품 수", "주식", "채권", "금·실물", "현금·달러", "레버리지",
              "환노출", "cagr_h1", "mdd_h1", "cagr_h2", "mdd_h2", "dd_2008", "dd_2011", "dd_2018", "dd_2020", "dd_2022",
              "cagr_2008", "mdd_2008", "cagr_2010", "mdd_2010", "cagr_holdout", "mdd_holdout", "cagr_sel", "mdd_sel",
              "roll_share", "roll_med_cagr", "nb_worst_mdd", "timing_worst", "refine_rounds", "refine_gain", "pbo_oos_within",
              "출처", "⚠", "비중"]


def band_tables(df: pd.DataFrame, presets: pd.DataFrame, top: int = 10, tie: float = 0.003, n_seeds: int | None = None) -> str:
    """Top candidates per 1 %p MDD band (ties within `tie` CAGR of the last shown row are included)."""
    lines = ["# MDD 구간별 상위 후보 (A31 1단계)", "",
             f"기간 {START} – {END}, 분기 리밸런싱, 편도 {COST_BPS:.0f}bp, 원화. 필수 조건은 전체 기간 MDD ≥ {MDD_LINE * 100:.0f}% 하나이고, "
             "⚠ 표시는 정보일 뿐 후보를 떨어뜨리지 않는다: 반기 = 2005–14 또는 2015–26 구간 MDD가 −15% 아래, 위기 = 2008/2011/2018/2020/2022 "
             "창 안 낙폭이 −15% 아래, 보류 = 마지막 15% 구간 MDD가 −15% 아래, 창 = 5년 롤링 창 중 −15% 안에 든 비율 < 80%, "
             "이웃 = ±5%p 이웃 중 최악 MDD가 −15% 아래, 날짜 = 리밸런싱 날짜 0/1/5/10/15일 이동 시 최악 MDD가 −20% 아래. "
             "환노출 = 달러 노출 상품 합(%). * = 가상 상품(실제 상품 확인 전).", ""]
    refined = df[df["출처"] == "다듬음"]
    n_seeds = n_seeds or int(refined["seed_rank"].max()) if "seed_rank" in refined and len(refined) else n_seeds
    groups = [("## 무작위 표본 (계층·희소)", df[df["출처"] != "다듬음"]),
              (f"## 다듬은 후보 (상위 {n_seeds or '?'}개의 ±5%p 이웃 이동, 최대 3회) — 표본과 따로 본다", refined)]
    edges = np.arange(-0.15, 0.0, 0.01)
    for title, part in groups:
        passing = part[part["mdd"] >= MDD_LINE]
        if passing.empty:
            continue
        lines += [title, ""]
        for lo in edges:  # deepest band first: the highest CAGR comes first
            hi = lo + 0.01
            band = passing[(passing["mdd"] > lo - 1e-12) & (passing["mdd"] <= hi + 1e-12)]
            if band.empty:
                continue
            band = band.sort_values("cagr", ascending=False)
            cut = band["cagr"].iloc[min(top, len(band)) - 1] - tie
            shown = band[band["cagr"] >= cut].head(top + 10)
            lines.append(f"### MDD −{abs(hi) * 100:.0f} ~ −{abs(lo) * 100:.0f}%  (후보 {len(band):,}개)")
            lines.append("")
            lines.append("| CAGR | MDD | 초과샤프 | 칼마 | 회전 | 수 | 주식 | 채권 | 금 | 현금·달러 | 레버 | 환노출 | 출처 | ⚠ | 비중 |")
            lines.append("|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|---|")
            for _, r in shown.iterrows():
                warn = "" if pd.isna(r["⚠"]) else r["⚠"]
                lines.append(f"| {pct(r['cagr'])} | {pct(r['mdd'])} | {r['sharpe_excess']:.2f} | {r['calmar']:.2f} | {pct(r['turnover'], 0)}% | "
                             f"{int(r['상품 수'])} | {int(r['주식'])} | {int(r['채권'])} | {int(r['금·실물'])} | {int(r['현금·달러'])} | "
                             f"{int(r['레버리지'])} | {int(r['환노출'])} | {r['출처']} | {warn} | {r['비중']} |")
            lines.append("")
    lines.append("## 등록된 프리셋 (같은 조건)")
    lines.append("")
    lines.append("| 프리셋 | CAGR | MDD | 초과샤프 | 칼마 | 통과 | ⚠ |")
    lines.append("|---|---:|---:|---:|---:|:--:|---|")
    for _, r in presets.sort_values("cagr", ascending=False).iterrows():
        warn = "" if pd.isna(r["⚠"]) else r["⚠"]
        lines.append(f"| {r['이름']} | {pct(r['cagr'])} | {pct(r['mdd'])} | {r['sharpe_excess']:.2f} | {r['calmar']:.2f} | "
                     f"{'✓' if r['mdd'] >= MDD_LINE else '✗'} | {warn} |")
    return "\n".join(lines) + "\n"


def scatter(df: pd.DataFrame, presets: pd.DataFrame, path: Path, font_path: Path | None, x_min: float = -0.35) -> bool:
    """CAGR × MDD of every candidate (colour = equity share, marker = leverage) with the presets numbered."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import LinearSegmentedColormap
        from matplotlib.lines import Line2D
    except ImportError:
        print("matplotlib이 없어 산점도를 건너뜁니다.")
        return False
    if font_path and Path(font_path).exists():
        from matplotlib import font_manager as fm
        fm.fontManager.addfont(str(font_path))
        plt.rcParams["font.family"] = fm.FontProperties(fname=str(font_path)).get_name()
    plt.rcParams["axes.unicode_minus"] = False
    ink, muted, grid, line_c = "#1f1f1e", "#6b6a66", "#e6e5e1", "#b4452c"
    ramp = LinearSegmentedColormap.from_list("blue", ["#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])  # ordinal range: stays visible
    shown = df[df["mdd"] >= x_min]
    n_out = len(df) - len(shown)
    sample = shown.sample(min(len(shown), 80_000), random_state=0) if len(shown) > 80_000 else shown
    plain, lev = sample[sample["레버리지"] == 0], sample[sample["레버리지"] > 0]
    fig, ax = plt.subplots(figsize=(11, 7.5), dpi=130)
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    kw = dict(cmap=ramp, vmin=0, vmax=100, s=5, linewidths=0, alpha=0.5, rasterized=True)
    sc = ax.scatter(plain["mdd"] * 100, plain["cagr"] * 100, c=plain["주식"], marker="o", **kw)
    if len(lev):
        ax.scatter(lev["mdd"] * 100, lev["cagr"] * 100, c=lev["주식"], marker="^", **kw)
    ax.axvline(MDD_LINE * 100, color=line_c, lw=1.2, ls="--")
    pres = presets.sort_values("cagr", ascending=False).reset_index(drop=True)
    ax.scatter(pres["mdd"] * 100, pres["cagr"] * 100, marker="D", s=52, c="#ffffff", edgecolors=ink, linewidths=1.4, zorder=5)
    for i, r in pres.iterrows():
        ax.annotate(str(i + 1), (r["mdd"] * 100, r["cagr"] * 100), xytext=(5, 3), textcoords="offset points", fontsize=8, color=ink)
    key = "\n".join(f"{i + 1:>2} {r['이름'].split(' (')[0]}  {r['cagr'] * 100:.1f} / {r['mdd'] * 100:.1f}" for i, r in pres.iterrows())
    ax.text(0.985, 0.975, "등록 프리셋 (CAGR / MDD, %)\n" + key, transform=ax.transAxes, va="top", ha="right", fontsize=7.6, color=ink,
            family=plt.rcParams["font.family"], bbox=dict(boxstyle="round,pad=0.4", fc="white", ec=grid))
    ax.set_xlim(x_min * 100, 0.5)
    ax.text(MDD_LINE * 100 + 0.3, ax.get_ylim()[0] + 0.3, "MDD −15% (필수 조건)", color=line_c, fontsize=9, va="bottom")
    ax.set_xlabel("최대 낙폭 MDD (%)", color=ink)
    ax.set_ylabel("연복리 수익률 CAGR (%)", color=ink)
    title = f"정적 배분 {len(df):,}개: CAGR × MDD ({START[:4]}–{END[:4]}, 원화, 분기 리밸런싱)"
    if n_out:
        title += f"  — MDD {x_min * 100:.0f}% 아래 {n_out:,}개는 그림 밖"
    ax.set_title(title, color=ink, loc="left", fontsize=11)
    ax.grid(True, color=grid, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(grid)
    ax.tick_params(colors=muted)
    cb = fig.colorbar(sc, ax=ax, pad=0.01)
    cb.set_label("주식 비중 (%)", color=ink)
    cb.ax.tick_params(colors=muted)
    cb.outline.set_edgecolor(grid)
    handles = [Line2D([], [], marker="o", color="#3987e5", ls="", ms=6, label="레버리지 없음"),
               Line2D([], [], marker="^", color="#3987e5", ls="", ms=6, label="레버리지 포함(≤ 20%)"),
               Line2D([], [], marker="D", color="white", markeredgecolor=ink, ls="", ms=6, label="등록 프리셋 (번호는 오른쪽 표)")]
    ax.legend(handles=handles, loc="lower left", frameon=False, fontsize=9, labelcolor=ink)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True


SUBSETS = [  # (label, products that must be absent) — the same frontier without the pieces a buyer may not want
    ("전체", ()),
    ("레버리지 없음", ("레버리지",)),
    ("가상 상품 없음", ("*",)),
    ("레버리지·가상 없음", ("레버리지", "*")),
    ("레버리지·가상·섹터 없음", ("레버리지", "*", "유틸리티", "필수소비재", "헬스케어", "에너지")),
    ("… 나스닥100도 없음", ("레버리지", "*", "유틸리티", "필수소비재", "헬스케어", "에너지", "나스닥")),
]


def frontier_table(df: pd.DataFrame, presets: pd.DataFrame) -> str:
    """Best CAGR per 1 %p MDD band among passing raw samples, for the whole set and for buyable subsets."""
    passing = df[(df["mdd"] >= MDD_LINE) & (df["출처"] != "다듬음")].copy()
    passing["band"] = np.ceil(passing["mdd"] * 100).astype(int)
    bands = list(range(-14, 0, 1))
    head = "| 부분집합 | " + " | ".join(f"({b - 1}, {b}]" for b in bands) + " |"
    lines = ["# MDD 구간별 최고 CAGR (%) — 부분집합별 경계선", "",
             f"통과 표본({len(passing):,}개) 중 각 1 %p MDD 구간에서 가장 높은 CAGR. 행은 특정 상품을 뺀 부분집합이고, "
             "열 머리글은 구간 (하한, 상한]. 레버리지 = 2배 상품 6종, 가상 = 실제 상품 확인 전인 환노출 미국채(*), "
             "섹터 = S&P500 유틸리티·필수소비재·헬스케어·에너지. 다듬은 후보는 제외.", "", head,
             "|---|" + "---:|" * len(bands)]
    for label, absent in SUBSETS:
        mask = np.ones(len(passing), dtype=bool)
        for token in absent:
            col = passing["레버리지"] > 0 if token == "레버리지" else passing["비중"].str.contains(token, regex=False)
            mask &= ~col.to_numpy()
        best = passing[mask].groupby("band")["cagr"].max()
        cells = [f"{best[b] * 100:.1f}" if b in best.index else "–" for b in bands]
        lines.append(f"| {label} ({int(mask.sum()):,}) | " + " | ".join(cells) + " |")
    lines += ["", "등록 프리셋의 자리 (CAGR / MDD):", ""]
    for _, r in presets.sort_values("cagr", ascending=False).iterrows():
        lines.append(f"- {r['이름']}: {r['cagr'] * 100:.1f} / {r['mdd'] * 100:.1f}" + ("" if r["mdd"] >= MDD_LINE else " (−15% 밖)"))
    return "\n".join(lines) + "\n"


def replot(out: Path, full_out: Path, font: Path | None) -> None:
    """Rebuild bands.md, frontier.md and scatter.png from the saved tables (no simulation)."""
    df = pd.read_csv(full_out / "candidates_full.csv.gz")
    presets = pd.read_csv(out / "presets.csv")
    refined_path = out / "top300_refined.csv"
    both = pd.concat([df, pd.read_csv(refined_path)], ignore_index=True) if refined_path.exists() else df
    (out / "bands.md").write_text(band_tables(both, presets), encoding="utf-8")
    (out / "frontier.md").write_text(frontier_table(df, presets), encoding="utf-8")
    scatter(df, presets, out / "scatter.png", font)
    print(f"다시 만들었습니다: {out / 'bands.md'}, {out / 'frontier.md'}, {out / 'scatter.png'}")


# --------------------------------------------------------------------------- main

def write_app_files(full: pd.DataFrame, out: Path) -> None:
    """passing.csv.gz and cloud.csv.gz for the app page '전역 탐색 결과' (lab.aa_search_results)."""
    passing, cloud = app_tables(full, seed=SEED)
    out.mkdir(parents=True, exist_ok=True)
    passing.to_csv(out / "passing.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    cloud.to_csv(out / "cloud.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    print(f"앱 파일: {out}/passing.csv.gz ({len(passing):,}행), cloud.csv.gz ({len(cloud):,}행)")


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--prices", choices=("model", "real"), default="model",
                    help="model = A31 as registered; real = real Korean prices after listing (A33, needs lab_data/kr)")
    ap.add_argument("--out", default="docs/research/allocation/2026-10-25_global_search")
    ap.add_argument("--full-out", default=None, help="folder for the full candidate table (default: --out)")
    ap.add_argument("--font", default=None, help="TTF with Hangul for the scatter (optional)")
    ap.add_argument("--no-pbo", action="store_true")
    ap.add_argument("--replot", action="store_true", help="only rebuild bands.md, frontier.md and scatter.png from the saved tables")
    ap.add_argument("--app-files", default=None, metavar="FULL_CSV",
                    help="only rebuild passing.csv.gz and cloud.csv.gz in --out from a saved candidates_full.csv.gz")
    args = ap.parse_args(argv)
    P = QUICK if args.quick else FULL
    out = Path(args.out)
    full_out = Path(args.full_out) if args.full_out else out
    if args.replot:
        replot(out, full_out, Path(args.font) if args.font else None)
        return
    if args.app_files:
        write_app_files(pd.read_csv(args.app_files), out)
        return
    out.mkdir(parents=True, exist_ok=True)
    full_out.mkdir(parents=True, exist_ok=True)
    t_all = time.time()
    summary: dict = {"params": {**P, "seed": SEED, "start": START, "end": END, "rebalance": REBALANCE, "cost_bps": COST_BPS,
                                "mdd_line": MDD_LINE, "universe": SYMBOLS, "prices": args.prices}}

    krw = load(args.prices)
    sim = srch.StaticSim(krw, SYMBOLS, START, END, rebalance=REBALANCE, cost_bps=COST_BPS)
    print(f"유니버스 {len(SYMBOLS)}종, 거래일 {sim.n_days:,}일 ({sim.dates[0].date()} – {sim.dates[-1].date()})")
    PW, pnames = preset_matrix()
    summary["self_check_max_rel_error"] = self_check(krw, sim, PW, pnames)

    # --- sampling
    W1, meta = srch.sample_two_level(CLASSES, n_class=P["n_class"], picks=P["picks"], seed=SEED, lev_class=LEV_CLASS,
                                     lev_cap=LEV_CAP, equity_classes=EQUITY_CLASSES, min_pct=MIN_PCT, max_products=MAX_PRODUCTS)
    assert list(meta["symbols"]) == SYMBOLS
    W2 = srch.sample_sparse(SYMBOLS, n=P["n_sparse"], seed=SEED + 1, k_range=(3, MAX_PRODUCTS), min_pct=MIN_PCT)
    W = np.vstack([W1, W2])
    source = np.array(["계층"] * len(W1) + ["희소"] * len(W2))
    W, first = np.unique(W, axis=0, return_index=True)
    source = source[first]
    print(f"표본: 계층 {len(W1):,} + 희소 {len(W2):,} → 중복 제거 후 {len(W):,}개")
    summary["n_candidates"] = int(len(W))

    # --- evaluation
    df, monthly = evaluate(sim, W, P["chunk"], "평가")
    df["출처"] = source
    n_pass = int(df["통과"].sum())
    print(f"MDD ≥ {MDD_LINE * 100:.0f}% 통과: {n_pass:,}개 ({n_pass / len(df) * 100:.1f}%)")
    summary["n_pass"] = n_pass

    # --- presets in the same table
    pres_df = evaluate(sim, (PW[:, :-1] * 100).round().astype(int), P["chunk"], "프리셋")[0]
    pres_df["이름"] = pnames
    pres_df["출처"] = "프리셋"

    # --- information columns for the top lists (raw samples)
    order = df[df["통과"]].sort_values("cagr", ascending=False).index.to_numpy()
    top_roll = order[:P["top_roll"]]
    top_ref = order[:P["top_refine"]]
    for col in ("roll_share", "roll_med_cagr", "nb_worst_mdd", "timing_worst", "refine_rounds", "refine_gain", "pbo_oos_within"):
        df[col] = np.nan
        pres_df[col] = np.nan
    if len(top_roll):
        df.loc[top_roll, "roll_share"], df.loc[top_roll, "roll_med_cagr"] = rolling(sim, W[top_roll], P["chunk"])
    pres_df["roll_share"], pres_df["roll_med_cagr"] = rolling(sim, (PW[:, :-1] * 100).round().astype(int), P["chunk"])

    def sim_at(off: int) -> srch.StaticSim:
        return srch.StaticSim(krw, SYMBOLS, START, END, rebalance=REBALANCE, cost_bps=COST_BPS, rebalance_offset=off)

    if len(top_ref):
        df.loc[top_ref, "timing_worst"] = timing_worst(sim_at, W[top_ref], P["chunk"])
    pres_df["timing_worst"] = timing_worst(sim_at, (PW[:, :-1] * 100).round().astype(int), P["chunk"])

    # --- refinement (listed apart, flagged 다듬음)
    refined_df = pd.DataFrame(columns=df.columns)
    if len(top_ref):
        Wr, rtab = refine(sim, W[top_ref], P["refine_rounds"], P["chunk"])
        df.loc[top_ref, "nb_worst_mdd"] = rtab["nb_worst_mdd"].to_numpy()
        moved = rtab["refine_rounds"].to_numpy() > 0
        if moved.any():
            Wr_u, idx_u = np.unique(Wr[moved], axis=0, return_index=True)
            refined_df, _ = evaluate(sim, Wr_u, P["chunk"], "다듬은 후보")
            refined_df["출처"] = "다듬음"
            refined_df["refine_rounds"] = rtab["refine_rounds"].to_numpy()[moved][idx_u]
            refined_df["refine_gain"] = rtab["refine_gain"].to_numpy()[moved][idx_u]
            refined_df["roll_share"], refined_df["roll_med_cagr"] = rolling(sim, Wr_u, P["chunk"])
            refined_df["timing_worst"] = timing_worst(sim_at, Wr_u, P["chunk"])
            refined_df["seed_rank"] = (np.arange(len(top_ref))[moved][idx_u] + 1)
            nbw = np.full(len(Wr_u), np.nan)
            for i, w in enumerate(Wr_u):
                p, _ = sim.run(to_fraction(np.vstack(srch.neighbours(w, step=5, min_pct=MIN_PCT))))
                nbw[i] = srch.cagr_mdd(p)[1].min()
            refined_df["nb_worst_mdd"] = nbw
        summary["refined"] = {"seeds": int(len(top_ref)), "moved": int(moved.sum()), "unique": int(len(refined_df)),
                              "median_gain_pp": float(np.median(rtab["refine_gain"][moved]) * 100) if moved.any() else 0.0}
        print(f"다듬기: 씨앗 {len(top_ref)}개 중 {moved.sum()}개가 움직임, 고유 {len(refined_df)}개")

    # --- campaign-level dials
    sel = df[df["mdd_sel"] >= MDD_LINE].sort_values("cagr_sel", ascending=False).index.to_numpy()[:P["top_refine"]]
    overlap = len(set(sel) & set(top_ref))
    summary["holdout_dial"] = {
        "top_n": int(P["top_refine"]), "overlap": int(overlap),
        "sel_top_holdout_cagr_median": float(df.loc[sel, "cagr_holdout"].median()) if len(sel) else None,
        "sel_top_holdout_mdd_median": float(df.loc[sel, "mdd_holdout"].median()) if len(sel) else None,
        "sel_top_holdout_within_share": float((df.loc[sel, "mdd_holdout"] >= MDD_LINE).mean()) if len(sel) else None,
        "sel_top_full_pass_share": float(df.loc[sel, "통과"].mean()) if len(sel) else None,
    }
    print(f"보류 구간 대조: 선택 구간 상위 {P['top_refine']}개와 전체 기간 상위의 겹침 {overlap}개, "
          f"선택 구간 상위의 보류 구간 MDD 중앙값 {pct(summary['holdout_dial']['sel_top_holdout_mdd_median'])}%")
    if not args.no_pbo:
        t0 = time.time()
        pbo = srch.pbo_selection(monthly.astype(np.float64), mdd_line=MDD_LINE, blocks=10)
        df["pbo_oos_within"] = pbo["oos_within"]
        top_picked = np.argsort(-pbo["picked"])[:5]
        summary["pbo"] = {"splits": pbo["splits"], "pbo": pbo["pbo"],
                          "pick_oos_cagr_median": float(np.median(pbo["pick_oos_cagr"])),
                          "pick_oos_mdd_median": float(np.median(pbo["pick_oos_mdd"])),
                          "pick_oos_within_share": float((pbo["pick_oos_mdd"] >= MDD_LINE).mean()),
                          "top_picked": [{"share": float(pbo["picked"][i]), "weights": describe(W[i]),
                                          "cagr": float(df.loc[i, "cagr"]), "mdd": float(df.loc[i, "mdd"])} for i in top_picked],
                          "seconds": round(time.time() - t0)}
        print(f"CSCV: PBO {pbo['pbo']:.2f}, 고른 후보의 표본 외 MDD 중앙값 {pct(summary['pbo']['pick_oos_mdd_median'])}%, "
              f"표본 외에서 −15% 안에 든 비율 {summary['pbo']['pick_oos_within_share']:.2f} ({summary['pbo']['seconds']}s)")

    # --- flags and files
    df["⚠"] = flags(df)
    pres_df["⚠"] = flags(pres_df)
    if len(refined_df):
        refined_df["⚠"] = flags(refined_df)
    cols = [c for c in TABLE_COLS if c in df.columns]
    df[cols].to_csv(full_out / "candidates_full.csv.gz", index=False, float_format="%.6g", compression="gzip")
    write_app_files(df[cols], out)
    df.loc[top_roll, cols].to_csv(out / "top2000.csv", index=False, float_format="%.6g")
    df.loc[top_ref, cols].to_csv(out / "top300.csv", index=False, float_format="%.6g")
    if len(refined_df):
        refined_df.sort_values("cagr", ascending=False)[[c for c in cols + ["seed_rank"] if c in refined_df.columns]].to_csv(
            out / "top300_refined.csv", index=False, float_format="%.6g")
    pres_df[["이름"] + cols].to_csv(out / "presets.csv", index=False, float_format="%.6g")
    both = pd.concat([df, refined_df], ignore_index=True) if len(refined_df) else df
    (out / "bands.md").write_text(band_tables(both, pres_df, n_seeds=len(top_ref)), encoding="utf-8")
    (out / "frontier.md").write_text(frontier_table(df, pres_df), encoding="utf-8")
    summary["scatter"] = scatter(df, pres_df, out / "scatter.png", Path(args.font) if args.font else None)
    summary["seconds"] = round(time.time() - t_all)
    summary["pass_by_source"] = {k: int(v) for k, v in df[df["통과"]]["출처"].value_counts().items()}
    summary["best_by_band"] = [{"band_hi": round(hi, 2), "cagr": float(g["cagr"].max()), "n": int(len(g))}
                               for hi, g in both[both["통과"]].groupby(np.ceil(both[both["통과"]]["mdd"] * 100) / 100)]
    (out / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"완료 ({summary['seconds']}s). 결과: {out}/ (bands.md, top300.csv, top300_refined.csv, top2000.csv, presets.csv, "
          f"scatter.png, frontier.md, summary.json), 전체 표: {full_out}/candidates_full.csv.gz")


if __name__ == "__main__":
    main(sys.argv[1:])
