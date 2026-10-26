"""The experiment log: docs/research/lab/experiments.json, read by the app's '실험 기록' page.

Every experiment leaves (1) why it was run, (2) a short summary of the key results and (3) figures
(PNG of aggregated results, committed under docs/research/lab/figures/<study>/). When the study's
result folder is reachable (Drive CRSP/<results_dir>), `chart_frame` turns its summary into a
table for an interactive chart. Newest first.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REGISTRY = ROOT / "docs" / "research" / "lab" / "experiments.json"
DRIVE_GUESSES = [Path(f"{d}:/내 드라이브/CRSP") for d in "GHIJ"] + [Path.home() / "Google Drive" / "CRSP"]
SAMPLE_NAMES = {"full": "1993–2023", "recent": "2017–2023"}
REQUIRED = ("id", "title", "date", "why", "results", "figures")


def load(path: Path | None = None) -> list[dict]:
    data = json.loads((path or REGISTRY).read_text(encoding="utf-8"))
    rows = data["experiments"]
    for e in rows:
        missing = [k for k in REQUIRED if k not in e]
        if missing:
            raise ValueError(f"experiments.json: {e.get('id', '?')}에 {missing} 없음")
    return sorted(rows, key=lambda e: e["date"], reverse=True)


def default_drive() -> str:
    for p in DRIVE_GUESSES:
        if (p / "lab_crsp").exists():
            return str(p)
    return ""


def results_folder(e: dict, drive: str) -> Path | None:
    if not drive or not e.get("results_dir"):
        return None
    p = Path(drive) / e["results_dir"]
    return p if p.exists() else None


def _label(key: str) -> str:
    from . import screens
    s = screens.ALL_SCREENS.get(key)
    return s.label if s else key


def chart_frame(e: dict, folder: Path) -> tuple[pd.DataFrame, str, str] | None:
    """(frame with columns 이름 / 표본 / 값, value kind 'excess' | 'pass', title) for one experiment,
    from its summary files; None when the folder has none of them."""
    kind = e.get("kind")
    if kind == "scores" and (folder / "summary.csv").exists():
        s = pd.read_csv(folder / "summary.csv")
        g = s.groupby(["score", "sample"])["median_goal_excess"].mean().reset_index()
        return (pd.DataFrame({"이름": g["score"].map(_label), "표본": g["sample"].map(SAMPLE_NAMES),
                              "값": g["median_goal_excess"]}), "excess", "선별 점수별 SPY 대비 초과 CAGR (중앙값)")
    if kind == "screens":
        files = sorted(folder.glob("summary_*_*.csv"))
        if files:
            parts = []
            for f in files:
                g = pd.read_csv(f)
                g["sample"] = f.stem.split("_")[1]
                parts.append(g)
            s = pd.concat(parts, ignore_index=True)
            g = s.groupby(["select", "sample"])["pass_share"].mean().reset_index()
            return (pd.DataFrame({"이름": g["select"].map(_label), "표본": g["sample"].map(SAMPLE_NAMES),
                                  "값": g["pass_share"]}), "pass", "선별 점수별 통과 칸 비율 (제외 × 진입 평균)")
    if kind == "combos" and (folder / "summary.csv").exists():
        s = pd.read_csv(folder / "summary.csv")
        return (pd.DataFrame({"이름": s["combo"].str.replace("+", " + "), "표본": s["sample"].map(SAMPLE_NAMES),
                              "값": s["median_goal_excess"]}), "excess", "선별 조합별 SPY 대비 초과 CAGR (32점 중앙값)")
    return None
