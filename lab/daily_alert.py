"""Daily alert: where the four market regimes of DECISIONS 50 stand, explained in plain words.

    python -m lab.daily_alert              # print the message (markdown)
    python -m lab.daily_alert --deliver    # also post it (GitHub issue comment, Telegram if configured)

Settings: config/alert.yml. SPY comes fresh from yfinance and the macro series from FRED, so
this needs network access; it runs in GitHub Actions after each US close.

The message carries signals and context only (no performance): the period after the holdout
start stays unseen for research (DECISIONS 28). The research verdict on these indicators is
in docs/research/lab/2026-10-19_macro_regime_filters.md: none of the macro ones improved
trade timing, so they are shown to describe the situation, not to trigger trades.
"""
from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import macro
from .blocks import ENTRY_BLOCKS, EXIT_BLOCKS, REGIME_PAIRS, sma
from .data import BENCHMARK, FIELDS, Panel, download_panel

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "alert.yml"
ON, OFF, UNKNOWN = "🟢 켜짐", "🔴 꺼짐", "⚪ 데이터 없음"
HISTORY_START = "1990-01-01"  # VIX starts 1990; the same window is used for every indicator's percentile
MONTH = 21  # trading days

SERIES_OF = {"curve_ok": "T10Y3M", "credit_calm": "BAA10Y", "vix_calm": "VIXCLS"}
SHORT = {"market_trend": "시장 추세", "curve_ok": "장단기 금리차", "credit_calm": "신용 스프레드", "vix_calm": "VIX"}
ICON = {"market_trend": "📈", "curve_ok": "📉", "credit_calm": "💳", "vix_calm": "🌡️"}

# Plain-language reading of each indicator's level: (upper bound, wording). Conventional reference
# points for description only; the regime rule (DECISIONS 50) is the only on/off criterion.
BANDS: dict[str, list[tuple[float, str]]] = {
    "curve_ok": [(0.0, "역전 (단기 금리가 장기보다 높음)"), (0.5, "거의 평탄"), (1.0, "완만한 우상향"), (np.inf, "뚜렷한 우상향")],
    "credit_calm": [(1.5, "매우 좁음 (위험 선호가 강한 상태)"), (2.5, "보통"), (3.5, "넓음 (신용 경계)"), (np.inf, "위기 수준")],
    "vix_calm": [(13.0, "매우 낮음 (과도한 안도감 경계)"), (20.0, "평온"), (25.0, "경계"), (35.0, "불안"), (np.inf, "공황 수준")],
}

EXPLAIN = {
    "market_trend": (
        "S&P 500을 따르는 ETF(SPY)의 종가가 최근 200거래일(약 10개월) 평균 가격보다 위에 있는지 봅니다. "
        "위에 있으면 주식 시장이 중기 상승 추세, 아래면 하락 추세로 읽습니다. 가장 단순하고 널리 쓰이는 추세 판단 기준입니다."
    ),
    "curve_ok": (
        "미국 10년 국채 금리에서 3개월 국채 금리를 뺀 값(%p)입니다. 보통은 돈을 오래 빌릴수록 금리가 높아 양수입니다. "
        "이 값이 음수(장단기 역전)가 되면 시장이 앞으로 금리 인하, 즉 경기 둔화를 예상한다는 뜻이고, 역사적으로 침체 6개월–2년 전에 역전이 나타났습니다. "
        "단, 2022–23년처럼 역전 중에도 주가가 오른 기간이 길어 매매 시점 지표로는 맞지 않았습니다."
    ),
    "credit_calm": (
        "신용등급이 낮은 편(Baa, 투자적격 최하단) 회사채 금리에서 10년 국채 금리를 뺀 값(%p)입니다. "
        "기업이 국가보다 얼마나 더 높은 이자를 내야 돈을 빌릴 수 있는지, 즉 시장이 기업 부도 위험을 얼마나 걱정하는지를 나타냅니다. "
        "값이 커지면(스프레드 확대) 신용 경색, 작아지면 위험 선호입니다. 여기서는 최근 200관측일 평균보다 높은지 낮은지로 방향을 봅니다."
    ),
    "vix_calm": (
        "S&P 500 옵션 가격에서 계산한, 앞으로 30일 동안 시장이 예상하는 연율 변동성입니다(공포 지수). "
        "VIX 20이면 시장이 연 ±20% 수준의 출렁임을 예상한다는 뜻입니다. 급락 때 치솟고 평온할 때 15 안팎입니다. "
        "주의: 높은 VIX는 대개 이미 떨어진 뒤에 나오므로, 연구에서는 VIX로 매수를 막는 것이 반등을 놓쳐 오히려 해로웠습니다."
    ),
}
RESEARCH_NOTE = (
    "**연구에서 확인된 것** — 28개 전략 계열에 각 지표를 필터로 붙여 두 기간(2000–2023, 2017–2023)에서 시험했을 때, "
    "금리차와 VIX는 매매 시점을 오히려 나쁘게 했고 신용 스프레드는 기간마다 방향이 반대였습니다. "
    "SPY 200일선만 가장 일관됐습니다. 그래서 매크로 세 지표는 '지금 시장이 어떤 상태인가'를 읽는 용도이고, 매매 신호가 아닙니다. "
    "자세한 내용은 `docs/research/lab/2026-10-19_macro_regime_filters.md`."
)


def load_config(path: Path = CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


# --------------------------------------------------------------------------- data

def trim_incomplete_tail(panel: Panel) -> tuple[Panel, int]:
    """Drop trailing rows where the benchmark close is missing (a half-built last bar).
    Returns (panel, rows dropped)."""
    spy = panel.close[BENCHMARK]
    last_ok = spy.last_valid_index()
    if last_ok is None or last_ok == panel.close.index[-1]:
        return panel, 0
    keep = panel.close.index <= last_ok
    cut = Panel(*(getattr(panel, f)[keep] for f in FIELDS), info=panel.info, source=panel.source,
                tradable=panel.tradable, cash=None if panel.cash is None else panel.cash[keep], kind=panel.kind)
    return cut, int((~keep).sum())


def load_benchmark_panel() -> tuple[Panel, int]:
    panel, failed = download_panel([BENCHMARK], str(date.today()), source="일일 알림 (yfinance)")
    if failed:
        raise RuntimeError(f"가격을 받지 못했습니다: {failed}")
    return trim_incomplete_tail(panel)


# --------------------------------------------------------------------------- regimes

@dataclass
class RegimeRow:
    key: str
    label: str
    value: str  # current value, formatted
    rule: str  # on condition, formatted
    on_today: bool | None
    on_before: bool | None
    since: str = ""  # "YYYY-MM-DD부터 N거래일": how long the current state has held
    context: list[str] = field(default_factory=list)  # plain-language lines about the current level
    on_share: float = float("nan")  # share of days since HISTORY_START in the on state

    @property
    def short(self) -> str:
        return SHORT[self.key]

    @property
    def flipped(self) -> bool:
        return None not in (self.on_today, self.on_before) and self.on_today != self.on_before


def _states(on: pd.DataFrame, off: pd.DataFrame) -> pd.Series:
    """1 / 0 / NaN (data missing) per day, from the entry filter and its complement exit."""
    a, b = on.iloc[:, 0].astype(bool).to_numpy(), off.iloc[:, 0].astype(bool).to_numpy()
    return pd.Series(np.where(a, 1.0, np.where(b, 0.0, np.nan)), index=on.index)


def _state_at(states: pd.Series, i: int) -> bool | None:
    v = states.iloc[i]
    return None if np.isnan(v) else bool(v)


def _since(states: pd.Series) -> str:
    s = states.dropna()
    if s.empty:
        return ""
    cur = s.iloc[-1]
    changed = s[s != cur]
    start = s.index[0] if changed.empty else s.index[s.index.get_loc(changed.index[-1]) + 1]
    return f"{start.date().isoformat()}부터 {int((s.index >= start).sum())}거래일"


def _on_share(states: pd.Series) -> float:
    s = states.dropna()
    s = s[s.index >= pd.Timestamp(HISTORY_START)]
    return float(s.mean()) if len(s) else float("nan")


def _band(key: str, x: float) -> str:
    for upper, text in BANDS[key]:
        if x < upper:
            return text
    return BANDS[key][-1][1]


def _percentile_text(hist: pd.Series, x: float, high_means: str) -> str:
    """'1990년 이후 상위 12% (높은 편: ...)' — where the current value sits in its own history."""
    h = hist.dropna()
    h = h[h.index >= pd.Timestamp(HISTORY_START)]
    if not len(h):
        return ""
    p = float((h < x).mean())
    if p >= 0.8:
        side = f"높은 편: {high_means}"
    elif p <= 0.2:
        side = "낮은 편"
    else:
        side = "중간 수준"
    return f"1990년 이후 상위 {1 - p:.0%} ({side})."


def market_trend_row(panel: Panel) -> RegimeRow:
    b = ENTRY_BLOCKS["market_trend"]
    n = int(b.param.default)
    st = _states(b.compute(panel, n), EXIT_BLOCKS[REGIME_PAIRS["market_trend"]].compute(panel, n))
    spy = panel.close[BENCHMARK]
    ma = sma(spy.to_frame(), n).iloc[:, 0]
    px, m = float(spy.iloc[-1]), float(ma.iloc[-1])
    ctx: list[str] = []
    if np.isfinite(px) and np.isfinite(m):
        gap = px / m - 1
        ctx.append(f"SPY {px:,.2f}는 200일선 {m:,.2f}보다 **{abs(gap):.1%} {'위' if gap >= 0 else '아래'}**에 있습니다.")
    if len(spy) > MONTH and np.isfinite(spy.iloc[-1 - MONTH]):
        ctx.append(f"최근 1개월 수익률 {spy.iloc[-1] / spy.iloc[-1 - MONTH] - 1:+.1%}.")
    hi = spy.iloc[-252:].max()
    if np.isfinite(hi) and np.isfinite(px):
        ctx.append(f"52주 최고 종가 대비 {px / hi - 1:+.1%}.")
    value = f"SPY {px:,.2f} / 200일선 {m:,.2f}" if np.isfinite(px) and np.isfinite(m) else "데이터 없음"
    return RegimeRow("market_trend", b.label, value, f"SPY 종가 > {n}일 이동평균", _state_at(st, -1), _state_at(st, -2),
                     _since(st), ctx, _on_share(st))


def macro_row(panel: Panel, key: str) -> RegimeRow:
    b = ENTRY_BLOCKS[key]
    v = b.param.default
    idx = panel.close.index
    st = _states(b.compute(panel, v), EXIT_BLOCKS[REGIME_PAIRS[key]].compute(panel, v))
    sid = SERIES_OF[key]
    window = int(v) if key == "credit_calm" else None
    val, avg = macro.series(sid, idx, window)
    raw = macro.data()[sid].dropna()
    seen = raw[raw.index < idx[-1]]
    when = seen.index[-1].strftime("%m-%d") if len(seen) else "?"
    unit = "" if key == "vix_calm" else "%p"
    cur = float(val.iloc[-1])
    ctx: list[str] = []
    if np.isfinite(cur):
        ctx.append(f"현재 **{cur:.2f}{unit}**: {_band(key, cur)}.")
        prev = float(val.iloc[-1 - MONTH]) if len(val) > MONTH else np.nan
        if np.isfinite(prev):
            ctx.append(f"1개월 전 {prev:.2f}{unit} → 지금 {cur:.2f}{unit} ({cur - prev:+.2f}).")
        high_means = {"curve_ok": "금리차가 큰 쪽", "credit_calm": "신용 불안이 큰 쪽", "vix_calm": "불안이 큰 쪽"}[key]
        ctx.append(_percentile_text(raw, cur, high_means))
        if avg is not None and np.isfinite(avg.iloc[-1]):
            a = float(avg.iloc[-1])
            ctx.append(f"{window}관측일 평균 {a:.2f}{unit}보다 {'낮아' if cur <= a else '높아'} 추세는 {'안정' if cur <= a else '확대'} 쪽입니다.")
        hist = raw[raw.index >= pd.Timestamp(HISTORY_START)]
        if len(hist):
            ctx.append(f"참고: 1990년 이후 최고 {hist.max():.2f}{unit} ({hist.idxmax().strftime('%Y-%m')}), 중앙값 {hist.median():.2f}{unit}.")
        value = f"{cur:.2f}" + (f" / {window}일 평균 {avg.iloc[-1]:.2f}" if avg is not None else "") + f" ({when} 값)"
    else:
        value = "데이터 없음"
    rule = {"curve_ok": f"금리차 > {v:g}%p", "credit_calm": f"스프레드 ≤ 최근 {window}관측일 평균", "vix_calm": f"VIX < {v:g}"}[key]
    return RegimeRow(key, b.label, value, rule, _state_at(st, -1), _state_at(st, -2), _since(st), ctx, _on_share(st))


def regime_rows(panel: Panel) -> list[RegimeRow]:
    return [market_trend_row(panel)] + [macro_row(panel, k) for k in REGIME_PAIRS if k != "market_trend"]


# --------------------------------------------------------------------------- message

def _mark(on: bool | None) -> str:
    return UNKNOWN if on is None else (ON if on else OFF)


def headline(regimes: list[RegimeRow]) -> str:
    known = [r for r in regimes if r.on_today is not None]
    if not known:
        return "국면 데이터를 읽지 못했습니다."
    off = [r.short for r in known if not r.on_today]
    if not off:
        return f"{len(known)}개 국면 모두 켜짐. 네 지표 어디에도 위험 신호가 없는 상태입니다."
    if len(off) == len(known):
        return f"{len(known)}개 국면 모두 꺼짐. 네 지표가 모두 위험을 가리키는 상태입니다."
    return f"{len(known)}개 중 {len(known) - len(off)}개 켜짐. 꺼진 국면: {', '.join(off)}."


def render(regimes: list[RegimeRow], as_of: str, mention: str | None = None, dropped_rows: int = 0) -> str:
    flips = [r for r in regimes if r.flipped]
    lines = [f"## 📊 일일 알림 · {as_of} 종가 기준", "", f"**요약:** {headline(regimes)}"]
    if flips:
        lines.append("**오늘 바뀐 국면:** " + ", ".join(f"{r.short} {_mark(r.on_before)} → {_mark(r.on_today)}" for r in flips))
    lines += ["", "### 시장 국면", "",
              "켜짐 = 그 지표로 보면 새로 매수해도 되는 국면, 꺼짐 = 조심할 국면. 값의 뜻은 아래 '지금 상황 읽기'에 풀어 썼습니다.", "",
              "| 지표 | 현재 값 | 켜짐 기준 | 상태 | 현재 상태 지속 |", "|---|---|---|---|---|"]
    for r in regimes:
        flag = " ⚠️ 오늘 바뀜" if r.flipped else ""
        lines.append(f"| {r.label} | {r.value} | {r.rule} | {_mark(r.on_today)}{flag} | {r.since} |")
    lines += ["", "### 지금 상황 읽기", ""]
    for r in regimes:
        share = f" 1990년 이후 이 국면이 켜져 있던 날은 {r.on_share:.0%}입니다." if np.isfinite(r.on_share) else ""
        body = " ".join(r.context) if r.context else "데이터가 없어 읽을 수 없습니다."
        lines.append(f"- {ICON[r.key]} **{r.short}** · {_mark(r.on_today)} — {body}{share}")
    lines += ["", "<details><summary>지표 해설 (처음 보는 분을 위해)</summary>", ""]
    for r in regimes:
        lines += [f"**{r.label}**  ", EXPLAIN[r.key], ""]
    lines += [RESEARCH_NOTE, "", "</details>", ""]
    notes = ["매크로 값은 FRED 공표 시점을 고려해 전일 값(괄호 안 날짜)을 씁니다. 연구 기록과 같은 규칙입니다."]
    if dropped_rows:
        notes.append(f"가격 데이터의 마지막 {dropped_rows}행은 미완성 봉이라 제외했습니다.")
    lines.append("<sub>" + " ".join(notes) + "</sub>")
    if mention:
        lines += ["", f"cc @{mention}"]
    return "\n".join(lines)


def has_news(regimes: list[RegimeRow]) -> bool:
    return any(r.flipped for r in regimes)


def plain_text(markdown: str) -> str:
    """Telegram version: no markdown emphasis or HTML; table rows become ' · ' separated lines."""
    t = re.sub(r"</?(details|summary|sub)>", "", markdown)
    t = t.replace("**", "").replace("`", "")
    t = re.sub(r"^\|---.*$", "", t, flags=re.M)
    t = re.sub(r"^\| ?(.*?) ?\|$", lambda m: " · ".join(c.strip() for c in m.group(1).split("|")), t, flags=re.M)
    return re.sub(r"\n{3,}", "\n\n", t).strip()


# --------------------------------------------------------------------------- delivery

ISSUE_BODY = ("매일 미국 장 마감 후, 시장 국면 4개(SPY 200일선, 장단기 금리차, 신용 스프레드, VIX)의 상태와 그 뜻을 댓글로 남깁니다. "
              "설정: `config/alert.yml`. 코드: `lab/daily_alert.py`.")


def _github(method: str, url: str, token: str, body: dict | None = None):
    import requests

    r = requests.request(method, url, json=body, timeout=30, headers={
        "Authorization": f"Bearer {token}", "Accept": "application/vnd.github+json"})
    r.raise_for_status()
    return r.json()


def post_issue_comment(text: str, title: str) -> str:
    token, repo = os.environ["GITHUB_TOKEN"], os.environ["GITHUB_REPOSITORY"]
    api = f"https://api.github.com/repos/{repo}"
    issues = _github("GET", f"{api}/issues?state=open&per_page=100", token)
    issue = next((i for i in issues if i["title"] == title), None)
    if issue is None:
        issue = _github("POST", f"{api}/issues", token, {"title": title, "body": ISSUE_BODY})
    elif (issue.get("body") or "") != ISSUE_BODY:
        _github("PATCH", f"{api}/issues/{issue['number']}", token, {"body": ISSUE_BODY})
    c = _github("POST", f"{api}/issues/{issue['number']}/comments", token, {"body": text})
    return c["html_url"]


def post_telegram(text: str) -> bool:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return False
    import requests

    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", timeout=30,
                      data={"chat_id": chat, "text": plain_text(text)[:4000], "disable_web_page_preview": "true"})
    r.raise_for_status()
    return True


def main(argv: list[str]) -> None:
    conf = load_config()
    macro.use(macro.fetch_fred())
    panel, dropped = load_benchmark_panel()
    print(f"SPY bars: {len(panel.close)}, last {panel.last_date.date()}, incomplete tail rows dropped: {dropped}")
    regimes = regime_rows(panel)
    text = render(regimes, panel.last_date.date().isoformat(), conf.get("mention"), dropped)
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    if "--deliver" in argv:
        if conf.get("only_on_change") and not has_news(regimes):
            print("변화 없음: 알림을 보내지 않습니다.")
            return
        print("issue comment:", post_issue_comment(text, conf.get("issue_title", "📊 일일 알림")))
        print("telegram:", post_telegram(text))
    print(json.dumps({"as_of": panel.last_date.date().isoformat(), "regimes": {r.key: r.on_today for r in regimes}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
