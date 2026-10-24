"""Plain-language notes shown in the allocation lab and shared with other research threads.

docs/research/allocation/NOTES.md holds three sections; each entry is one bullet
"- YYYY-MM-DD · text". The app renders them and can append entries. Other threads edit the
same file with any editor, so the format stays plain Markdown.

lab_results/aa_runs.csv logs every run made in the app (what was tried, with its headline
numbers) and feeds the "campaign" tab. It is the user's research record and is not committed.
"""
from __future__ import annotations

import json
import re
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from .data import ROOT

NOTES_PATH = ROOT / "docs" / "research" / "allocation" / "NOTES.md"
RUNS_PATH = ROOT / "lab_results" / "aa_runs.csv"
SECTIONS = {
    "findings": "실무에 바로 쓰는 발견",
    "warnings": "정직한 경고",
    "campaign": "여기까지 오게 된 캠페인",
}
SECTION_HELP = {
    "findings": "백테스트에서 반복해 확인돼 실제 배분 결정에 바로 쓸 수 있는 것.",
    "warnings": "모든 결과에 공통으로 붙는 한계. 숫자를 믿기 전에 읽을 것.",
    "campaign": "어떤 질문에서 출발해 무엇을 시험했고 어디까지 왔는지. 다음 세션이 이어 가는 곳.",
}
_ENTRY = re.compile(r"^- (\d{4}-\d{2}-\d{2}) · (.*)$")
RUN_COLUMNS = ["time", "name", "description", "period", "cagr", "mdd", "calmar", "sharpe_excess", "turnover", "kind",
               "config"]  # config: JSON of the AllocConfig (or {"mix": MixConfig}) so a run can be re-run later


def load_notes(path: Path = NOTES_PATH) -> dict[str, list[tuple[str, str]]]:
    """{section key: [(date, text), ...]} in file order."""
    out: dict[str, list[tuple[str, str]]] = {k: [] for k in SECTIONS}
    if not path.exists():
        return out
    titles = {v: k for k, v in SECTIONS.items()}
    current = None
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.startswith("## "):
            current = titles.get(line[3:].strip())
            continue
        if current is None:
            continue
        m = _ENTRY.match(line.rstrip())
        if m:
            out[current].append((m.group(1), m.group(2).strip()))
        elif line.startswith("  ") and out[current]:  # continuation line of the previous entry
            d, t = out[current][-1]
            out[current][-1] = (d, t + " " + line.strip())
    return out


def add_note(section: str, text: str, day: str | None = None, path: Path = NOTES_PATH) -> None:
    """Append one entry under a section (creating the file with all three sections if needed)."""
    if section not in SECTIONS:
        raise KeyError(section)
    text = " ".join(text.split())
    if not text:
        raise ValueError("빈 메모는 추가하지 않습니다.")
    day = day or str(date.today())
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# 자산배분 연구 노트\n\n" + "".join(f"## {t}\n\n" for t in SECTIONS.values()), encoding="utf-8")
    lines = path.read_text(encoding="utf-8").splitlines()
    title = SECTIONS[section]
    try:
        start = next(i for i, l in enumerate(lines) if l.strip() == f"## {title}")
    except StopIteration:
        lines += ["", f"## {title}", ""]
        start = len(lines) - 2
    end = next((i for i in range(start + 1, len(lines)) if lines[i].startswith("## ")), len(lines))
    insert = end
    while insert > start + 1 and not lines[insert - 1].strip():
        insert -= 1
    lines.insert(insert, f"- {day} · {text}")
    if insert == end or (insert < len(lines) - 1 and lines[insert + 1].startswith("## ")):
        lines.insert(insert + 1, "")
    path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")


def log_run(name: str, description: str, period: tuple, metrics: dict, kind: str = "alloc",
            path: Path | None = None, config: dict | None = None) -> None:
    path = path or RUNS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)
    row = {
        "time": datetime.now().isoformat(timespec="seconds"), "name": name, "description": description,
        "period": f"{period[0]} → {period[1]}",
        **{k: metrics.get(k) for k in ("cagr", "mdd", "calmar", "sharpe_excess", "turnover")}, "kind": kind,
        "config": json.dumps(config, ensure_ascii=False, default=str) if config is not None else "",
    }
    if path.exists():
        header = pd.read_csv(path, nrows=0, encoding="utf-8").columns.tolist()
        if header != RUN_COLUMNS:  # a log from an older version: rewrite it with the current columns
            old = pd.read_csv(path, encoding="utf-8").reindex(columns=RUN_COLUMNS)
            old.to_csv(path, index=False, encoding="utf-8")
    pd.DataFrame([row], columns=RUN_COLUMNS).to_csv(path, mode="a", header=not path.exists(), index=False, encoding="utf-8")


def load_runs(path: Path | None = None) -> pd.DataFrame:
    path = path or RUNS_PATH
    if not path.exists():
        return pd.DataFrame(columns=RUN_COLUMNS)
    return pd.read_csv(path, encoding="utf-8").reindex(columns=RUN_COLUMNS)
