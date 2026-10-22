"""Daily alert: market regimes and the model portfolio's orders for the next open.

    python -m lab.daily_alert              # print the message (markdown)
    python -m lab.daily_alert --deliver    # also post it (GitHub issue comment, Telegram if configured)

Settings: config/alert.yml. Prices come fresh from yfinance and macro series from FRED, so
this needs network access; it runs in GitHub Actions after each US close.

Only signals are reported, not performance: the period after the holdout start stays unseen
for research (DECISIONS 28).
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

from . import macro
from .blocks import ENTRY_BLOCKS, EXIT_BLOCKS, REGIME_PAIRS, sma
from .data import BENCHMARK, Panel, download_panel
from .engine import StrategyConfig, run_backtest

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "config" / "alert.yml"
ON, OFF = "🟢 켜짐", "🔴 꺼짐"


def load_config(path: Path = CONFIG) -> dict:
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def universe_symbols(kind: str) -> list[str]:
    """Tradable symbols of the research universe of that name (same filter funnel as the lab)."""
    from .batch import UNIVERSE_NAMES, build_universe

    if kind not in UNIVERSE_NAMES:
        raise ValueError(f"unknown universe: {kind}")
    return build_universe(kind).panel.tradable_symbols


# --------------------------------------------------------------------------- regimes

@dataclass
class RegimeRow:
    key: str
    label: str
    value: str
    rule: str
    on_today: bool | None
    on_before: bool | None

    @property
    def flipped(self) -> bool:
        return None not in (self.on_today, self.on_before) and self.on_today != self.on_before


def _state(frame: pd.DataFrame, exit_frame: pd.DataFrame, i: int) -> bool | None:
    on, off = bool(frame.iloc[i, 0]), bool(exit_frame.iloc[i, 0])
    return True if on else (False if off else None)  # neither = data missing


def regime_rows(panel: Panel) -> list[RegimeRow]:
    idx = panel.close.index
    rows = []
    for key, out in REGIME_PAIRS.items():
        b = ENTRY_BLOCKS[key]
        v = b.param.default
        on = b.compute(panel, v)
        off = EXIT_BLOCKS[out].compute(panel, v)
        if key == "market_trend":
            spy = panel.close[BENCHMARK]
            value = f"SPY {spy.iloc[-1]:.2f} / {int(v)}일선 {sma(spy.to_frame(), int(v)).iloc[-1, 0]:.2f}"
        else:
            sid = {"curve_ok": "T10Y3M", "credit_calm": "BAA10Y", "vix_calm": "VIXCLS"}[key]
            val, avg = macro.series(sid, idx, int(v) if key == "credit_calm" else None)
            raw = macro.data()[sid].dropna()
            seen = raw[raw.index < idx[-1]]
            when = seen.index[-1].strftime("%m-%d") if len(seen) else "?"
            value = f"{val.iloc[-1]:.2f}" + (f" / {int(v)}일 평균 {avg.iloc[-1]:.2f}" if avg is not None else "") + f" ({when})"
        rows.append(RegimeRow(key, b.label, value, f"{b.rule} · {b.param.label.split(' ')[0]} = {v:g}", _state(on, off, -1), _state(on, off, -2)))
    return rows


# --------------------------------------------------------------------------- model portfolio

def model_portfolio(panel: Panel, cfg: StrategyConfig) -> dict:
    res = run_backtest(panel, cfg)
    trail = cfg.exits.get("trailing_pct")
    holdings = []
    for _, p in res.open_positions.iterrows():
        closes = panel.close[p["symbol"]].loc[p["entry_date"]:]
        peak = float(closes.max())
        holdings.append({"symbol": p["symbol"], "entry_date": p["entry_date"].date().isoformat(),
                         "from_peak": float(closes.iloc[-1] / peak - 1), "trail": trail})
    sells = {s: EXIT_BLOCKS[k].label for s, k in res.pending_orders.get("sell", {}).items()}
    return {"as_of": res.equity.index[-1].date().isoformat(), "holdings": holdings,
            "buy": list(res.pending_orders.get("buy", [])), "sell": sells, "describe": cfg.describe(),
            "max_positions": cfg.max_positions}


# --------------------------------------------------------------------------- message

def _mark(on: bool | None) -> str:
    return "⚪ 데이터 없음" if on is None else (ON if on else OFF)


def render(regimes: list[RegimeRow], book: dict, mention: str | None = None) -> str:
    flips = [r for r in regimes if r.flipped]
    head = f"## 📊 일일 알림 · {book['as_of']} 종가 기준"
    lines = [head, "", "주문은 다음 거래일 시가 기준입니다. 연구 검증을 통과한 전략이 아닌 표본 내 연구 후보입니다.", ""]
    if flips:
        lines.append("**국면 변화:** " + ", ".join(f"{r.label} {_mark(r.on_before)} → {_mark(r.on_today)}" for r in flips))
        lines.append("")
    lines += ["### 시장 국면 (켜짐 = 새 매수 허용 국면)", "", "| 지표 | 현재 값 | 켜짐 조건 | 상태 |", "|---|---|---|---|"]
    lines += [f"| {r.label} | {r.value} | {r.rule} | {_mark(r.on_today)}{' ⚠️ 바뀜' if r.flipped else ''} |" for r in regimes]
    lines += ["", "<sub>매크로 값은 FRED 공표 시점을 고려해 전일 값(괄호 안 날짜)을 씁니다. 연구 기록과 같은 규칙입니다.</sub>", ""]
    lines += ["### 모델 포트폴리오", "", f"전략: {book['describe']}", ""]
    lines.append("- **내일 매수:** " + (", ".join(book["buy"]) if book["buy"] else "없음"))
    lines.append("- **내일 매도:** " + (", ".join(f"{s} ({why})" for s, why in book["sell"].items()) if book["sell"] else "없음"))
    held = book["holdings"]
    lines.append(f"- **보유 {len(held)} / {book['max_positions']}종목**")
    if held:
        lines += ["", "| 종목 | 매수일 | 보유 중 최고 종가 대비 |", "|---|---|---|"]
        for h in sorted(held, key=lambda h: h["from_peak"]):
            stop = f" (청산선 −{h['trail']:g}%)" if h["trail"] else ""
            lines.append(f"| {h['symbol']} | {h['entry_date']} | {h['from_peak']:+.1%}{stop} |")
    if mention:
        lines += ["", f"cc @{mention}"]
    return "\n".join(lines)


def has_news(regimes: list[RegimeRow], book: dict) -> bool:
    return any(r.flipped for r in regimes) or bool(book["buy"]) or bool(book["sell"])


# --------------------------------------------------------------------------- delivery

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
        issue = _github("POST", f"{api}/issues", token, {
            "title": title, "body": "매일 장 마감 후 시장 국면과 모델 포트폴리오 주문을 댓글로 남깁니다. 설정: `config/alert.yml`."})
    c = _github("POST", f"{api}/issues/{issue['number']}/comments", token, {"body": text})
    return c["html_url"]


def post_telegram(text: str) -> bool:
    token, chat = os.environ.get("TELEGRAM_BOT_TOKEN"), os.environ.get("TELEGRAM_CHAT_ID")
    if not (token and chat):
        return False
    import requests

    plain = text.replace("**", "").replace("<sub>", "").replace("</sub>", "")
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage", timeout=30,
                      data={"chat_id": chat, "text": plain[:4000], "disable_web_page_preview": "true"})
    r.raise_for_status()
    return True


def main(argv: list[str]) -> None:
    conf = load_config()
    macro.use(macro.fetch_fred())
    symbols = universe_symbols(conf.get("universe", "snapshot"))
    panel, failed = download_panel(symbols, str(date.today()), source="일일 알림 (yfinance)")
    s = conf["strategy"]
    cfg = StrategyConfig(entries=dict(s["entries"]), exits=dict(s["exits"]), start=str(conf.get("start", "2017-08-01")),
                         end=str(panel.last_date.date()), max_positions=int(s.get("max_positions", 10)),
                         cost_bps=float(s.get("cost_bps", 10)), sizing=str(s.get("sizing", "inverse_vol")))
    regimes = regime_rows(panel)
    book = model_portfolio(panel, cfg)
    text = render(regimes, book, conf.get("mention"))
    if failed:
        text += f"\n\n<sub>가격을 받지 못한 종목 {len(failed)}개: {', '.join(failed[:20])}</sub>"
    print(text)
    summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary:
        with open(summary, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    if "--deliver" in argv:
        if conf.get("only_on_change") and not has_news(regimes, book):
            print("변화 없음: 알림을 보내지 않습니다.")
            return
        print("issue comment:", post_issue_comment(text, conf.get("issue_title", "📊 일일 알림")))
        print("telegram:", post_telegram(text))
    print(json.dumps({"regimes": {r.key: r.on_today for r in regimes}, "buy": book["buy"],
                      "sell": list(book["sell"])}, ensure_ascii=False))


if __name__ == "__main__":
    main(sys.argv[1:])
