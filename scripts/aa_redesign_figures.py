"""Figures for A41 (allocation DECISIONS A41) from the committed result tables only.

    .venv/Scripts/python.exe scripts/aa_redesign_figures.py --out docs/research/allocation/2026-10-28_a41
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab import aa_landscape as LS  # noqa: E402
from lab import figures as figs  # noqa: E402

ZONE_COLORS = {"안": "#2b6cb0", "경계": "#dd6b20", "밖": "#b0b0b0"}


def _fig(w=9, h=6):
    plt = figs._plt()
    fig, ax = plt.subplots(figsize=(w, h))
    return plt, fig, ax


def fig_map(out: Path, dest: Path) -> Path:
    mp = pd.read_csv(out / "map.csv.gz")
    named = pd.read_csv(out / "named.csv")
    cen = pd.read_csv(out / "centres.csv") if (out / "centres.csv").stat().st_size > 5 else pd.DataFrame()
    plt, fig, ax = _fig()
    for z in ("밖", "경계", "안"):
        p = mp[mp["zone"] == z]
        ax.scatter(p["mdd"] * 100, p["cagr"] * 100, s=2, alpha=0.25 if z == "밖" else 0.35, color=ZONE_COLORS[z],
                   label=f"{z} ({len(p):,}점{' 표본' if z == '밖' else ''})", rasterized=True)
    ax.scatter(named["mdd"] * 100, named["cagr"] * 100, marker="D", s=50, facecolor="none", edgecolor="black", label="비교 배분")
    for _, r in named.iterrows():
        ax.annotate(r["이름"], (r["mdd"] * 100, r["cagr"] * 100), fontsize=8, xytext=(4, 3), textcoords="offset points")
    if len(cen):
        full = cen[cen["고른 구간"] == "전체"]
        ax.scatter(full["mdd"] * 100, full["cagr"] * 100, marker="*", s=160, color="#9b2c2c", label="평균 중심 (전체 구간, 결과를 본 뒤 보완)")
        for _, r in full.iterrows():
            ax.annotate(f"{r['묶음'].split()[0]} 사후", (r["mdd"] * 100, r["cagr"] * 100), fontsize=8, xytext=(4, -10),
                        textcoords="offset points", color="#9b2c2c")
    ax.axvline(-15, color="gray", ls="--", lw=1)
    ax.axvline(-17, color="gray", ls=":", lw=1)
    ax.set_xlim(max(-45, mp["mdd"].min() * 100), 0)
    ax.set_xlabel("MDD (%, 2000-09 – 2024-12)")
    ax.set_ylabel("CAGR (%)")
    ax.set_title("A41 ① 비중 지도: 13개 상품, 2^20점 (원화, 분기, 10bp)")
    leg = ax.legend(fontsize=8, loc="lower left")
    for h in list(getattr(leg, "legend_handles", getattr(leg, "legendHandles", [])))[:3]:
        h.set_sizes([20])
    return figs._save(fig, dest / "1_map.png")


def fig_sobol(out: Path, dest: Path) -> Path:
    s = pd.read_csv(out / "sobol.csv")
    plt = figs._plt()
    outs = ["cagr", "mdd", "rank", "dd_dotcom"]
    titles = {"cagr": "CAGR", "mdd": "MDD", "rank": "순위 값 (시작 시점 CAGR 하위 10%)", "dd_dotcom": "닷컴 창 낙폭"}
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
    for ax, o in zip(axes.ravel(), outs):
        t = s[s["출력"] == o].sort_values("ST")
        y = np.arange(len(t))
        ax.barh(y - 0.2, t["ST"], height=0.4, color="#2b6cb0", label="총 지수")
        ax.barh(y + 0.2, t["S1"].clip(lower=0), height=0.4, color="#dd6b20", label="1차 지수")
        ax.errorbar(t["ST"], y - 0.2, xerr=[t["ST"] - t["ST_lo"], t["ST_hi"] - t["ST"]], fmt="none", ecolor="black", lw=0.7)
        ax.set_yticks(y)
        ax.set_yticklabels(t["이름"], fontsize=8)
        ax.set_title(titles[o], fontsize=10)
    axes[0, 0].legend(fontsize=8)
    fig.suptitle("A41 ② 상품별 전역 민감도 (Sobol 지수, 실제 시뮬레이션)")
    return figs._save(fig, dest / "2_sobol.png")


def fig_include(out: Path, dest: Path) -> Path:
    t = pd.read_csv(out / "include_exclude.csv").sort_values("넣으면 안 비율 변화 중앙")
    lo = pd.read_csv(out / "leave_one_out.csv").set_index("상품")
    plt = figs._plt()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.5), sharey=True)
    y = np.arange(len(t))
    axes[0].barh(y, t["넣으면 안 비율 변화 중앙"] * 100, color=np.where(t["넣으면 안 비율 변화 중앙"] >= 0, "#2b6cb0", "#9b2c2c"))
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(t["상품"], fontsize=8)
    axes[0].set_xlabel("그 상품을 넣은 집합 − 뺀 집합: '안' 비율 차 (%p, 집합 짝의 중앙)")
    axes[0].axvline(0, color="gray", lw=0.8)
    l2 = lo.reindex(t["상품"])
    axes[1].barh(y, l2["MDD변화_중앙"] * 100, color="#dd6b20")
    axes[1].set_xlabel("점에서 그 상품을 빼고 비례로 맞출 때 MDD 변화 (%p, 중앙)")
    axes[1].axvline(0, color="gray", lw=0.8)
    fig.suptitle("A41 ③ 상품 넣기·빼기")
    return figs._save(fig, dest / "3_include.png")


def _targets(R: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(R["대상"]))


def fig_rebalance(out: Path, dest: Path) -> Path:
    R = pd.read_csv(out / "robustness.csv.gz")
    reb = R[R["시험"] == "리밸런싱"]
    names = _targets(reb)
    plt, fig, ax = _fig(max(9, 0.6 * len(names)), 6)
    ax.boxplot([reb[reb["대상"] == n]["mdd"] * 100 for n in names], showfliers=True)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([n.replace("중심 ", "") for n in names], rotation=60, ha="right", fontsize=7)
    ax.axhline(-15, color="gray", ls="--", lw=1)
    ax.set_ylabel("MDD (%)")
    ax.set_title("A41 ④ 리밸런싱 528가지 변형의 MDD (주기 × 시기 × 일 이동 × 밴드 × 비용)")
    return figs._save(fig, dest / "4_rebalance.png")


def fig_starts(out: Path, dest: Path) -> Path:
    R = pd.read_csv(out / "robustness.csv.gz")
    st = R[R["시험"] == "시작 시점"].copy()
    st["시작"] = pd.to_datetime(st["변형"])
    plt, fig, ax = _fig(11, 6)
    for k, n in enumerate(_targets(st)):
        p = st[st["대상"] == n].sort_values("시작")
        style = "-" if n.startswith("중심 전체") else ":"
        if n.startswith("중심") and not n.startswith("중심 전체"):
            continue
        ax.plot(p["시작"], p["cagr"] * 100, style, lw=1.2, label=n.replace("중심 ", ""))
    ax.set_ylabel("시작일 → 2024-12 CAGR (%)")
    ax.set_title("A41 ⑤ 시작 시점별 CAGR (매달 시작, 끝 2024-12-31)")
    ax.legend(fontsize=7, ncol=2)
    return figs._save(fig, dest / "5_starts.png")


def fig_exchange(out: Path, dest: Path) -> Path:
    e = pd.read_csv(out / "exchange.csv")
    e = e[e["단계"] == "1%p"]
    names = [LS.SHORT[c] for c in LS.UNIVERSE]
    M = e.pivot_table(index="줄이는 상품", columns="늘리는 상품", values="MDD변화_중앙").reindex(index=names, columns=names)
    plt, fig, ax = _fig(10, 8)
    v = np.nanmax(np.abs(M.to_numpy())) * 100 if np.isfinite(M.to_numpy()).any() else 1.0
    im = ax.imshow(M.to_numpy() * 100, cmap="RdBu", vmin=-v, vmax=v)
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.set_xlabel("1%p 늘리는 상품")
    ax.set_ylabel("1%p 줄이는 상품")
    fig.colorbar(im, ax=ax, label="MDD 변화 중앙 (%p, + = 얕아짐)")
    ax.set_title("A41 ⑥ 교환 비율: '안' 구역 점에서 1%p를 옮길 때 MDD 변화")
    return figs._save(fig, dest / "6_exchange.png")


def fig_halves(out: Path, dest: Path) -> Path:
    cen = pd.read_csv(out / "centres.csv") if (out / "centres.csv").stat().st_size > 5 else pd.DataFrame()
    rules = pd.read_csv(out / "rules.csv")
    plt, fig, ax = _fig(9, 6)
    if len(cen):
        for tag, col_m, col_r, mk in (("앞 절반", "mdd_h2", "rank_h2", "o"), ("뒤 절반", "mdd_h1", "rank_h1", "s")):
            p = cen[cen["고른 구간"] == tag]
            ax.scatter(p[col_m] * 100, p[col_r] * 100, marker=mk, s=60, label=f"{tag}에서 고른 중심 → 반대쪽 절반")
    r = rules[rules["상품 집합"] == "13개 전부"]
    for col_m, col_r, mk, lab in (("mdd_h2", "rank_h2", "^", "규칙형(13개) 뒤 절반"), ("mdd_h1", "rank_h1", "v", "규칙형(13개) 앞 절반")):
        ax.scatter(r[col_m] * 100, r[col_r] * 100, marker=mk, s=60, label=lab)
        for _, x in r.iterrows():
            ax.annotate(x["규칙"], (x[col_m] * 100, x[col_r] * 100), fontsize=7, xytext=(3, 3), textcoords="offset points")
    ax.axvline(-15, color="gray", ls="--", lw=1)
    ax.set_xlabel("반대쪽 절반 MDD (%)")
    ax.set_ylabel("반대쪽 절반 순위 값 (%)")
    ax.set_title("A41 ⑦ 절반 교차: 한쪽에서 고른 것의 반대쪽 성적")
    ax.legend(fontsize=7)
    return figs._save(fig, dest / "7_halves.png")


def fig_holdout(out: Path, dest: Path) -> Path:
    h = pd.read_csv(out / "holdout.csv")
    plt, fig, ax = _fig(10, 6.5)
    late = h["실행"].fillna("첫 실행").ne("첫 실행") if "실행" in h else pd.Series(False, index=h.index)
    for flag, mk, lab in ((False, "o", "첫 실행"), (True, "s", "추가 (첫 실행에서 빠진 대상)")):
        p = h[late == flag]
        if p.empty:
            continue
        ax.scatter(p["실제 MDD"] * 100, p["실제 누적 수익"] * 100, marker=mk, s=55, facecolor="none" if flag else "#2b6cb0",
                   edgecolor="#2b6cb0", label=lab)
        for _, r in p.iterrows():
            name = r["대상"].replace("중심 전체 ", "").replace("중심 ", "").replace(" 평균·보완 1", " 평균 중심 (사후)")
            ax.annotate(name, (r["실제 MDD"] * 100, r["실제 누적 수익"] * 100), fontsize=7, xytext=(4, 3), textcoords="offset points")
    ax.axvline(-15, color="gray", ls="--", lw=1)
    ax.set_xlabel(f"보류 구간 MDD (%, 실제 가격 우선, {h['실제 시작'].iloc[0]} – {h['실제 끝'].iloc[0]})")
    ax.set_ylabel("누적 수익 (%)")
    ax.set_title("A41 ⑧ 보류 구간 한 번: 연구에 쓰지 않은 22개월 (기록만, 후보·판정은 그대로)")
    ax.legend(fontsize=8)
    return figs._save(fig, dest / "8_holdout.png")


def main(argv):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    out = Path(args.out)
    dest = out / "figures"
    dest.mkdir(exist_ok=True)
    if not figs.available():
        raise SystemExit("matplotlib이 없습니다")
    made = []
    for f in (fig_map, fig_sobol, fig_include, fig_rebalance, fig_starts, fig_exchange, fig_halves, fig_holdout):
        try:
            made.append(f(out, dest).name)
        except FileNotFoundError as exc:
            print("건너뜀:", f.__name__, exc)
    print(made)


if __name__ == "__main__":
    main(sys.argv[1:])
