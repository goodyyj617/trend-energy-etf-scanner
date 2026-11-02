"""Monthly real-price record of the A42 candidates (allocation DECISIONS A42 §5, sealed window ②).

Once a month (GitHub Actions, `.github/workflows/aa_a42_track.yml`) this downloads the real closes and distributions
of the tracked products (`aa_real.fetch`), computes last month's total return of each tracked allocation (weights
held through the month, no trading) and posts it with the 'expected range' (the 5–95 % quantiles of the allocation's
monthly returns in the research window 2000-09 – 2024-12, representative price set) as a comment on the issue
"📈 A42 실제 가격 기록". It writes nothing to the repository and never changes a weight; the first evaluation is due
after twelve months (2027-11).

    python -m lab.aa_track            # print last month's comment
    python -m lab.aa_track --deliver  # also post it (needs GITHUB_TOKEN and GITHUB_REPOSITORY)
"""
from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import pandas as pd

from . import aa_real

TRACK = Path(__file__).resolve().parents[1] / "docs" / "research" / "allocation" / "2026-11-02_a42_stage5" / "track.csv"
ISSUE_TITLE = "📈 A42 실제 가격 기록"
ISSUE_BODY = ("A42 5단계의 봉인 구간 ②: 후보와 비교 배분의 지난달 실제 가격 수익을 매달 첫 거래일에 남깁니다. "
              "'예상 범위'는 연구 구간(2000-09 – 2024-12) 월간 수익의 5–95% 분위입니다. 첫 평가는 12개월 뒤(2027-11)이고, "
              "이 기록으로 비중을 바꾸지 않습니다. 동적 층이 있는 후보는 신호 없이 기본 비중으로 기록합니다.")


def load_track(path: Path = TRACK) -> pd.DataFrame:
    t = pd.read_csv(path, dtype=str)
    t["weights"] = [{c: float(x) for c, x in (p.split(":") for p in s.split(";"))} for s in t["비중"]]
    for c in ("예상 5%", "예상 95%"):
        t[c] = t[c].astype(float)
    return t


def month_returns(series: dict[str, pd.Series], month: pd.Period) -> dict[str, float]:
    """Total return of each product over `month` (last close of the month over the last close before it)."""
    out = {}
    for code, s in series.items():
        s = s.dropna()
        before = s[s.index < month.start_time]
        inside = s[(s.index >= month.start_time) & (s.index <= month.end_time)]
        if len(before) and len(inside):
            out[code] = float(inside.iloc[-1] / before.iloc[-1] - 1.0)
    return out


def report(month: pd.Period | None = None, folder: Path | None = None) -> str:
    t = load_track()
    codes = sorted({c for w in t["weights"] for c in w})
    folder = Path(folder or tempfile.mkdtemp(prefix="aa_track_"))
    aa_real.fetch(codes, folder)
    files = aa_real.load_files(folder)
    series = aa_real.real_series(*files, quirks=aa_real.QUIRKS)
    month = month or (pd.Timestamp.today().to_period("M") - 1)
    r = month_returns({c: series[c] for c in codes if c in series}, month)
    lines = [f"**{month} 실제 가격 수익** (분배금 포함, 한 달 동안 비중 그대로)", "",
             "| 배분 | 지난달 | 예상 범위 (5–95%) | |", "|---|---|---|---|"]
    for _, row in t.iterrows():
        w = row["weights"]
        miss = [c for c in w if c not in r]
        if miss:
            lines.append(f"| {row['이름']} | 자료 없음 ({', '.join(miss)}) | {row['예상 5%']:+.1%} – {row['예상 95%']:+.1%} | |")
            continue
        ret = sum(w[c] / 100.0 * r[c] for c in w)
        inside = row["예상 5%"] <= ret <= row["예상 95%"]
        lines.append(f"| {row['이름']} | {ret:+.2%} | {row['예상 5%']:+.1%} – {row['예상 95%']:+.1%} | {'안' if inside else '**밖**'} |")
    lines += ["", f"<!-- a42-track {month} -->"]
    return "\n".join(lines)


def post(text: str, month: str) -> str:
    from .daily_alert import _github
    token, repo = os.environ["GITHUB_TOKEN"], os.environ["GITHUB_REPOSITORY"]
    api = f"https://api.github.com/repos/{repo}"
    issues = _github("GET", f"{api}/issues?state=open&per_page=100", token)
    issue = next((i for i in issues if i["title"] == ISSUE_TITLE), None)
    if issue is None:
        issue = _github("POST", f"{api}/issues", token, {"title": ISSUE_TITLE, "body": ISSUE_BODY})
    comments = _github("GET", f"{api}/issues/{issue['number']}/comments?per_page=100", token)
    if any(f"<!-- a42-track {month} -->" in (c.get("body") or "") for c in comments):
        return "already posted"
    c = _github("POST", f"{api}/issues/{issue['number']}/comments", token, {"body": text})
    return c.get("html_url", "")


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deliver", action="store_true")
    ap.add_argument("--month", default=None, help="YYYY-MM (default: last month)")
    a = ap.parse_args(argv)
    month = pd.Period(a.month, "M") if a.month else pd.Timestamp.today().to_period("M") - 1
    text = report(month)
    print(text)
    if a.deliver:
        print("issue comment:", post(text, str(month)))


if __name__ == "__main__":
    main()
