"""Memo figures (PNG, aggregates only) for finished CRSP studies, from their result folders on Drive.

    python scripts/make_study_figures.py --drive "H:\\내 드라이브\\CRSP" [--study scores|screens|all]

Reads the per-cell CSVs and summaries of lab_crsp/<study>_top3000 (never prices) and writes
docs/research/lab/figures/<study>_top3000/*.png. New studies write their own figures at the end of
the run (e.g. lab/combo_study.py); this script covers the studies finished before that rule.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab import figures, screens  # noqa: E402

FIG = ROOT / "docs" / "research" / "lab" / "figures"
NAMES = {"full": "CRSP 1993–2023", "recent": "CRSP 2017–2023"}
LAYERS = {"hold": "보유형", "replace": "교체형"}


def label(key: str) -> str:
    s = screens.ALL_SCREENS.get(key)
    return s.label if s else key


def cells(folder: Path) -> pd.DataFrame:
    """Every cell CSV under families/<sample>/<entry>/[<layer>/], with sample/entry/layer columns."""
    parts = []
    for f in sorted((folder / "families").rglob("*.csv")):
        rel = f.relative_to(folder / "families").parts
        g = pd.read_csv(f)
        g["sample"], g["entry"] = rel[0], rel[1]
        g["layer"] = rel[2] if len(rel) == 4 else "hold"
        parts.append(g)
    return pd.concat(parts, ignore_index=True)


def spy_of(c: pd.DataFrame) -> dict:
    return {NAMES[k]: (float(g["spy_cagr"].iloc[0]), float(g["spy_mdd"].iloc[0])) for k, g in c.groupby("sample")}


def scores_figures(folder: Path, out: Path) -> dict:
    s = pd.read_csv(folder / "summary.csv")
    # excess per score = mean over the four (entry, layer) groups of the median excess of its 6 cells
    ex = s.groupby(["score", "sample"])["median_goal_excess"].mean().unstack("sample")
    ex = ex.sort_values("full", ascending=False)
    labels = [label(k) for k in ex.index]
    figures.paired_bars(labels, {NAMES[k]: ex[k].to_numpy() for k in ("full", "recent")},
                        "선별 점수별 SPY 대비 초과 CAGR (진입 2 × 매매층 2 묶음의 중앙값 평균)", "SPY 대비 초과 CAGR (%p)",
                        out / "score_excess.png")
    rho = float(ex["full"].rank().corr(ex["recent"].rank()))
    figures.scatter_labeled(ex["full"].to_numpy(), ex["recent"].to_numpy(), labels, "두 표본의 선별 점수 순위 일관성",
                            "1993–2023 초과 (%p)", "2017–2023 초과 (%p)", out / "score_consistency.png",
                            note=f"순위 상관(Spearman) {rho:.2f}")
    lay = s.groupby(["score", "sample", "layer"])["median_goal_excess"].mean().unstack("layer")
    diff = (lay["replace"] - lay["hold"]).unstack("sample").reindex(ex.index)
    figures.paired_bars(labels, {NAMES[k]: diff[k].to_numpy() for k in ("full", "recent")},
                        "교체형 − 보유형: SPY 대비 초과 CAGR 차이 (진입 2가지 평균)", "차이 (%p, 오른쪽 = 교체형이 나음)",
                        out / "hold_vs_replace.png")
    c = cells(folder)
    groups = {NAMES[k]: g.assign(group=g["layer"].map(LAYERS)) for k, g in c.groupby("sample")}
    figures.cagr_mdd(groups, spy_of(c), f"모든 칸의 CAGR과 MDD ({len(c):,}칸, 별 = SPY)", out / "cagr_mdd.png")
    return {"spearman": rho, "cells": len(c),
            "median_diff_replace_minus_hold": {k: float(diff[k].median()) for k in ("full", "recent")}}


def screens_figures(folder: Path, out: Path) -> dict:
    c = cells(folder)
    share = c.groupby(["select", "sample"])["goal_pass"].mean().unstack("sample").sort_values("full", ascending=False)
    figures.paired_bars([label(k) for k in share.index], {NAMES[k]: share[k].to_numpy() for k in ("full", "recent")},
                        "선별 점수별 통과 칸 비율 (제외 8 × 진입 2 × 16칸)", "통과 칸 비율 (%)", out / "select_pass_share.png")
    groups = {NAMES[k]: g.assign(group=g["entry"].map({"always": "조건 없음", "above_ma100": "100일선 위"}))
              for k, g in c.groupby("sample")}
    figures.cagr_mdd(groups, spy_of(c), f"모든 칸의 CAGR과 MDD ({len(c):,}칸, 별 = SPY)", out / "cagr_mdd.png")
    return {"cells": len(c), "pass": {k: int(g["goal_pass"].sum()) for k, g in c.groupby("sample")}}


STUDIES = {"scores": scores_figures, "screens": screens_figures}


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--drive", required=True)
    ap.add_argument("--study", choices=[*STUDIES, "all"], default="all")
    ap.add_argument("--top", type=int, default=3000)
    a = ap.parse_args(argv)
    if not figures.available():
        sys.exit("matplotlib가 없습니다: pip install -r requirements-lab.txt")
    for name in STUDIES if a.study == "all" else [a.study]:
        folder = Path(a.drive) / "lab_crsp" / f"{name}_top{a.top}"
        info = STUDIES[name](folder, FIG / f"{name}_top{a.top}")
        print(name, info, flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
