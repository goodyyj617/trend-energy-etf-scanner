"""Stage G-2: extreme-event entries V1-V6, judged per signal (DECISIONS 85), resumable.

    python -m lab.event_study --panel DIR --eligible FILE --out DIR [--g1 DIR] [--workers N] [--dry]

Builds the per-signal tables of V1-V6 x X1-X9 (3 x 3 cells, both samples) exactly like
`stage_c_study --signals-only` (same universe, hygiene, exits, costs), then measures them with the
Stage G-1 per-signal measures (lab/tail_study.py: tail shares, big winners B50 / B5R / B2S, excess
over SPY and the equal-weight universe). With --g1 (the G-1 output folder) the E1-E9 rows are added
for comparison. Output: signals/, cells.csv, compare.csv, compare_canonical.csv, figures/.
"""
from __future__ import annotations

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

import pandas as pd

from . import figures, stage_c_study as sc, tail_study as ts
from .crsp_study import SAMPLES, load_panel_with_cash

# ---- registered (DECISIONS 85): fixed before any result
EVENTS: dict[str, tuple[str, list]] = {
    "V1": ("g_ret_z1", [2.0, 2.5, 3.0]),
    "V2": ("g_ret_z5", [2.0, 2.5, 3.0]),
    "V3": ("g_volume_z", [3, 4, 5]),
    "V4": ("g_range_expansion", [2.0, 2.5, 3.0]),
    "V5": ("g_high52_volume", [1.5, 2, 3]),
    "V6": ("g_contraction_break", [10, 20, 30]),
}
NAMES = {"V1": "하루 급등", "V2": "5일 급등", "V3": "거래량 폭발", "V4": "변동폭 급확장", "V5": "52주 신고가 종가 + 거래량",
         "V6": "수축 뒤 돌파", "E1": "N일 고가 돌파", "E2": "골든크로스", "E3": "52주 신고가 + 거래량", "E4": "볼린저 수축 돌파",
         "E5": "눌림목", "E6": "MACD", "E7": "+DI 교차", "E8": "시계열 모멘텀", "E9": "갭 상승"}
COMPARE = ["signals", "mean", "median", "mean_wo_top5", "top5_share", "skew", "b50", "b5r", "b2s", "median_x_spy",
           "mean_x_spy", "beat_spy", "median_x_ew", "big_mean_x_spy"]


def use_menu(dry: bool = False) -> None:
    """Stage C's runner with the V1-V6 entries (and, dry, only V1 x X1, X9)."""
    sc.ENTRIES.clear()
    sc.ENTRIES.update({k: v for k, v in EVENTS.items() if not dry or k == "V1"})
    sc.ENTRY_CANON.update({k: 1 for k in EVENTS})
    if dry:
        for k in [k for k in sc.EXITS if k not in ("X1", "X9")]:
            del sc.EXITS[k]


def _init(panel_dir: str, eligible: str, dry: bool):
    use_menu(dry)
    sc._setup(load_panel_with_cash(Path(panel_dir)), pd.read_parquet(eligible))


def compare(c: pd.DataFrame, g1: pd.DataFrame | None) -> tuple[pd.DataFrame, pd.DataFrame]:
    allc = pd.concat([c, g1], ignore_index=True) if g1 is not None else c
    allc = allc[allc["sample"].isin(list(SAMPLES))]
    by_entry = allc.groupby(["sample", "entry"])[COMPARE].median().reset_index()
    by_entry["signals"] = allc.groupby(["sample", "entry"])["signals"].median().to_numpy()
    can = allc[allc["canonical"].astype(bool)].groupby(["sample", "entry", "exit"])[COMPARE].first().reset_index()
    return by_entry, can


def write_figures(out: Path, by_entry: pd.DataFrame) -> list[Path]:
    if not figures.available():
        return []
    made = []
    order = [k for k in list(EVENTS) + [f"E{i}" for i in range(1, 10)] if k in set(by_entry["entry"])]
    for stat, title, unit in (("b5r", "매수별 큰 승자(≥ 5R) 비율 (전 칸 중앙값)", "%"),
                              ("b50", "매수별 큰 승자(≥ +50%) 비율 (전 칸 중앙값)", "%"),
                              ("median_x_spy", "매수별 신호당 SPY 대비 초과 중앙값 (전 칸 중앙값)", "%p"),
                              ("mean", "매수별 신호당 평균 수익 (전 칸 중앙값)", "%")):
        m = by_entry.pivot(index="entry", columns="sample", values=stat).reindex(order)
        made.append(figures.paired_bars([f"{k} {NAMES[k]}" for k in order], {SAMPLES[s][2]: m[s].to_numpy(dtype=float) for s in SAMPLES},
                                        title, unit, out / "figures" / f"events_{stat}.png", pct=True, legend_out=True))
    return made


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--eligible", required=True, help="Stage C의 eligible_daily.parquet")
    ap.add_argument("--out", required=True)
    ap.add_argument("--g1", default=None, help="Stage G-1 결과 폴더(cells.csv): E1–E9 비교용")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--dry", action="store_true", help="V1 × X1, X9만")
    a = ap.parse_args(argv)
    use_menu(a.dry)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "menu.json").write_text(json.dumps({"events": EVENTS, "exits": sc.EXITS, "hygiene": sc.HYGIENE, "common": sc.COMMON},
                                              ensure_ascii=False, default=str), encoding="utf-8")
    workers = a.workers
    while True:
        try:
            with ProcessPoolExecutor(max_workers=workers, initializer=_init, initargs=(a.panel, a.eligible, a.dry)) as ex:
                sc.run_signal_tables(ex, out, ts.log)
            break
        except (BrokenProcessPool, MemoryError):
            if workers <= 1:
                raise
            workers = max(1, workers * 3 // 4)
            ts.log(f"메모리가 부족했습니다. 작업 {workers}개로 줄여 이어서 실행합니다(끝난 부분은 건너뜀).")
    ts.log("신호 단위 측정(꼬리 · SPY 대비)")
    panel = load_panel_with_cash(Path(a.panel))
    eligible = pd.read_parquet(a.eligible)
    panel.eligible = eligible
    c = ts.run_cells(ts.Prices(panel, eligible), out, out)
    g1 = pd.read_csv(Path(a.g1) / "cells.csv") if a.g1 else None
    by_entry, can = compare(c, g1)
    by_entry.to_csv(out / "compare.csv", index=False)
    can.to_csv(out / "compare_canonical.csv", index=False)
    write_figures(out, by_entry)
    ts.log(f"완료 · 결과: {out}")


if __name__ == "__main__":
    import sys

    main(sys.argv[1:])
