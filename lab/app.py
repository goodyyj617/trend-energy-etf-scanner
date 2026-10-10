"""추세추종 전략 연구실 — Streamlit app.

Run:  .venv\\Scripts\\python.exe -m streamlit run lab/app.py
"""
from __future__ import annotations

import sys
from dataclasses import replace
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import research_log as rlog  # noqa: E402
from lab import store  # noqa: E402
from lab.blocks import ALL_BLOCKS, ENTRY_BLOCKS, EXIT_BLOCKS  # noqa: E402
from lab.data import BENCHMARK, LONG_HISTORY_ETFS, Panel, download_panel, load_snapshot, snapshot_available  # noqa: E402
from lab.families import (  # noqa: E402
    ENTRY_MENU, EXIT_MENU, MENU_VERSION, ORDER, SUMMARY_TEXT, addon_choices, addon_label, cell_count, family_pairs,
    run_addon_test, run_family_comparison,
)
from lab.refine import VERDICT_TEXT, addon_candidates, addon_value, family_label, run_refinement  # noqa: E402
from lab.engine import (  # noqa: E402
    BASELINE_HELP, BASELINE_LABELS, RULES_KO, SIZING_LABELS, StrategyConfig, baselines, run_backtest,
)
from lab.grid import (  # noqa: E402
    CHECK_TEXT, GATE_TEXT, MAX_AXIS_VALUES, ROBUST_TEXT, Gates, axis_values, check_gates, run_grid, strategy_checklist,
)
from lab.metrics import DEFINITIONS, all_metrics, annual_table, curve_metrics, deflated_sharpe, drawdown, fmt_value  # noqa: E402
from lab.universe import (  # noqa: E402
    CASH_CLASS, CLASS_HELP, CLASSES, DEDUP_CORR, DEFAULT_CLASSES, UniverseFilter, apply_filter, candidate_table,
)

st.set_page_config(page_title="추세추종 전략 연구실", page_icon="📈", layout="wide")

BLUE, ORANGE, GRAY, AQUA = "#2a78d6", "#eb6834", "#8a8984", "#1baf7a"
BASE_STYLE = {"spy": dict(color=ORANGE, width=2), "spy_ma200": dict(color=GRAY, width=2, dash="dash"),
              "equal_weight": dict(color=AQUA, width=1.5, dash="dot")}
SEQ_BLUE = [[0, "#cde2fb"], [0.5, "#5598e7"], [1, "#0d366b"]]
BR = "  " + chr(10)  # markdown hard line break
LONG_COUNT = sum(len(v) for v in LONG_HISTORY_ETFS.values())
SOURCES = {
    "snapshot": "ETF 스냅샷 · 466개 · 2016-08 – 2026-07 · 인터넷 불필요",
    "long": f"장기 멀티에셋 ETF · {LONG_COUNT}개 · 1993 – 현재 (2000년·2008년 하락장 포함) · yfinance",
    "custom": "직접 입력 · ETF·개별 종목 · yfinance",
}
FAMILY_ORDER = ["가격", "거래량", "보조지표"]
SCORECARD = ["total_return", "cagr", "mdd", "sharpe", "sortino", "calmar", "worst_12m", "volatility", "month_win",
             "underwater_days"]


def tip(key: str) -> str:
    d = DEFINITIONS[key]
    return f"{d.short}\n\n계산: {d.formula}"


def param_label(key: str) -> str:
    if key == "max_positions":
        return "최대 보유 종목 수"
    b = ALL_BLOCKS[key]
    return f"{b.label} · {b.param.label}"


def init(key: str, value) -> str:
    """Set a widget's starting value once; afterwards the session state owns it."""
    if key not in st.session_state:
        st.session_state[key] = value
    return key


def estimate_seconds(panel: Panel, cfg: StrategyConfig, n_backtests: int) -> float:
    """Rough run time; calibrated on a 24-year, 33-ETF family comparison (~0.3 s per backtest)."""
    days = int(((panel.close.index >= pd.Timestamp(cfg.start)) & (panel.close.index <= pd.Timestamp(cfg.end))).sum())
    return n_backtests * (0.05 + days * len(panel.tradable_symbols) * 1.5e-6)


def gates_now() -> Gates:
    return Gates(float(st.session_state.get("gate_cagr", 0.80)), float(st.session_state.get("gate_mdd", 0.75)),
                 int(st.session_state.get("gate_trades", 30)))


# =========================================================================== data (cached)

@st.cache_resource(show_spinner="ETF 스냅샷을 불러오는 중… (처음 한 번만 몇 초 걸립니다)")
def snapshot() -> Panel:
    return load_snapshot()


@st.cache_resource(show_spinner="장기 ETF 가격을 불러오는 중… (처음 한 번은 30초 정도, 이후에는 저장된 파일 사용)")
def long_panel(day: str) -> Panel:
    syms = [s for v in LONG_HISTORY_ETFS.values() for s in v]
    panel, _ = download_panel(syms, day, source="장기 멀티에셋 ETF (yfinance)")
    return panel


def base_panel(source: str) -> Panel:
    return snapshot() if source == "snapshot" else long_panel(str(date.today()))


@st.cache_resource(show_spinner="유니버스 필터 적용 중…", max_entries=16)
def universe_table(source: str, day: str, classes: tuple, min_aum: float, max_er: float, dedup: bool,
                   corr_end: str | None):
    base = base_panel(source)
    candidates = base.tradable_symbols if source == "long" else base.symbols
    table = candidate_table(candidates, base.info)
    flt = UniverseFilter(classes=classes, min_aum_musd=min_aum, max_expense_pct=max_er, dedup=dedup)
    return apply_filter(table, flt, base.close, corr_end)


@st.cache_resource(show_spinner=False, max_entries=8)
def subset_panel(source: str, day: str, symbols: tuple) -> Panel:
    return base_panel(source).subset(list(symbols))


def rebuild_panel(meta: dict) -> Panel:
    """Rebuild the exact universe of a saved result (used by the holdout page)."""
    if meta["data_kind"] == "snapshot":
        return snapshot().subset(meta["universe"])
    panel, _ = download_panel(meta["universe"], str(date.today()), source=meta.get("data_source", "yfinance"))
    return panel


# =========================================================================== ① universe and period

def funnel_line(funnel: list[tuple[str, int]]) -> str:
    return " → ".join(f"**{n}** <small>{label}</small>" for label, n in funnel)


def universe_section() -> Panel | None:
    st.subheader("① 유니버스와 기간")
    source = st.radio("데이터와 거래 대상", list(SOURCES), format_func=SOURCES.get, key=init("uni_source", "snapshot"),
                      help="ETF 스냅샷: 저장소에 고정 저장된 466개 ETF(레버리지·인버스·옵션형 등은 이미 제외).\n\n"
                           f"장기 멀티에셋: 2004년 이전 상장 대표 ETF {LONG_COUNT}개. 2000–02년·2008년 하락장까지 시험할 수 있습니다.\n\n"
                           "직접 입력: 원하는 ETF·개별 종목. 아래 ETF 필터는 적용되지 않고 유동성 필터만 적용됩니다.")
    hold = rlog.load_holdout()
    corr_end = hold["start"] if hold["enabled"] else None
    today = str(date.today())

    if source in ("snapshot", "long"):
        if source == "snapshot" and not snapshot_available():
            st.error("저장소에서 ETF 스냅샷 폴더를 찾지 못했습니다.")
            return None
        try:
            base_panel(source)
        except Exception as exc:  # network errors for the yfinance-backed list
            st.error(f"가격 데이터를 불러오지 못했습니다: {exc}")
            return None
        c1, c2, c3, c4 = st.columns([3, 1.3, 1.3, 1.2])
        classes = c1.multiselect(
            "자산군", CLASSES, key=init("uni_classes", list(DEFAULT_CLASSES)),
            help="\n\n".join(f"**{k}**: {v}" for k, v in CLASS_HELP.items()),
        )
        min_aum = c2.number_input("최소 운용 규모 ($백만)", 0.0, 500000.0, step=250.0, key=init("uni_min_aum", 1000.0),
                                  help="ETF 운용 자산(AUM) 하한. 작은 ETF는 상장폐지·호가 공백 위험이 큽니다. "
                                       "현재 시점 값이라 과거에 작았던 ETF도 통과할 수 있습니다(생존 편향).")
        max_er = c3.number_input("최대 총보수 (%/년)", 0.0, 2.0, step=0.05, key=init("uni_max_er", 0.75),
                                 help="연간 운용 보수 상한. 보수는 가격에 이미 반영되어 있지만, 비싼 ETF는 대개 테마·액티브형입니다. "
                                      "보수 정보가 없는 ETF는 통과시킵니다.")
        dedup = c4.checkbox("중복 ETF 정리", key=init("uni_dedup", True),
                            help=f"일간 수익률 상관계수가 {DEDUP_CORR} 이상인 ETF 쌍(예: SPY·VOO·IVV)은 사실상 같은 상품이므로 "
                                 "운용 규모가 큰 쪽만 남깁니다. 같은 베팅을 두 번 보유하는 것을 막습니다. "
                                 "상관은 보류 구간 이전 데이터로만 계산합니다.")
        table, funnel = universe_table(source, today, tuple(classes), float(min_aum), float(max_er), bool(dedup), corr_end)
        st.markdown(funnel_line(funnel), unsafe_allow_html=True)
        chosen = sorted(table.loc[table["status"] == "선택", "symbol"])
        with st.expander(f"종목 목록 보기 (선택 {len(chosen)}개 · 제외 {len(table) - len(chosen)}개)"):
            shown = table.assign(_sel=table["status"] != "선택").sort_values(["_sel", "asset_class", "aum"], ascending=[True, True, False])
            st.dataframe(shown.drop(columns="_sel"), hide_index=True, width="stretch", column_config={
                "symbol": "티커", "name": "이름", "category": "분류 (Morningstar)", "asset_class": "자산군",
                "aum": st.column_config.NumberColumn("운용 규모 ($)", format="compact"),
                "expense_ratio": st.column_config.NumberColumn("총보수 (%)", format="%.2f"),
                "status": "상태",
            })
        if not chosen:
            st.warning("거래 대상 종목이 없습니다. 자산군이나 필터를 조정하세요.")
            return None
        panel = subset_panel(source, today, tuple(chosen))
    else:
        tickers = st.text_area("티커 (쉼표 또는 줄바꿈으로 구분)", key=init("yf_tickers", "SPY, QQQ, IWM, EFA, EEM, GLD, TLT, XLE"),
                               help="ETF와 개별 종목 모두 가능합니다. 비교 기준 SPY와 현금 수익률용 BIL은 자동으로 받습니다.")
        refresh = st.checkbox("캐시 무시하고 새로 받기", value=False)
        if st.button("가격 데이터 불러오기"):
            syms = [t for t in tickers.replace("\n", ",").split(",") if t.strip()]
            with st.spinner(f"{len(syms)}개 티커를 내려받는 중…"):
                try:
                    p, failed = download_panel(syms, today, refresh=refresh)
                    st.session_state["yf_panel"] = p
                    if failed:
                        st.warning("받지 못한 티커: " + ", ".join(failed))
                except Exception as exc:  # network / yfinance errors
                    st.error(f"다운로드 실패: {exc}")
        panel = st.session_state.get("yf_panel")
        if panel is None:
            st.info("‘가격 데이터 불러오기’를 누르세요.")
            return None
        st.caption("⚠️ 개별 종목은 현재 상장 종목만으로 과거를 테스트하므로 생존 편향이 ETF보다 큽니다.")

    return period_section(panel, source, hold)


def period_section(panel: Panel, source: str, hold: dict) -> Panel | None:
    # SPY is the comparison, so the selectable range starts where SPY data starts
    first = max(panel.first_date, panel.close[BENCHMARK].first_valid_index()).date()
    last = panel.last_date.date()
    hs = date.fromisoformat(hold["start"]) if hold["enabled"] else None
    research_last = last if hs is None or hs > last else hs - timedelta(days=1)
    if research_last <= first:
        st.error("보류 구간 시작일이 데이터 시작보다 앞서 연구할 기간이 없습니다. 보류 구간 설정을 확인하세요.")
        return None
    default_start = max(first, date(first.year + 1, first.month, 1))
    if source == "long":
        default_start = max(default_start, date(2000, 1, 1))
    if st.session_state.get("_data_range") != (first, research_last, source):  # new data: reset dates to defaults
        st.session_state["_data_range"] = (first, research_last, source)
        st.session_state.pop("start", None)
        st.session_state.pop("end", None)
    c1, c2, c3 = st.columns([1, 1, 2])
    c1.date_input("시작일", min_value=first, max_value=research_last, key=init("start", default_start))
    c2.date_input("종료일", min_value=first, max_value=research_last, key=init("end", research_last))
    c3.markdown(
        f"**거래 대상 {len(panel.tradable_symbols)}종목** · 데이터 {first} – {last}  \n"
        "<small>지표(이동평균 등)는 시작일 이전 데이터로 미리 계산합니다.</small>",
        unsafe_allow_html=True,
    )
    holdout_controls(hold, last)
    if st.session_state["start"] >= st.session_state["end"]:
        st.error("시작일은 종료일보다 앞이어야 합니다.")
        return None
    return panel


def holdout_controls(hold: dict, last: date) -> None:
    n_eval = len(hold["evaluations"])
    if hold["enabled"]:
        st.info(f"🔒 **보류 구간: {hold['start']} 이후** — 연구(백테스트·격자)에 쓰지 않고, 왼쪽 ‘최종 검증’ 페이지에서만 "
                f"평가합니다. 지금까지 {n_eval}번 평가했습니다.")
    else:
        st.warning("🔓 보류 구간이 꺼져 있습니다. 모든 데이터를 연구에 쓰므로 나중에 확인할 ‘처음 보는 데이터’가 남지 않습니다.")
    with st.expander("보류 구간 설정 바꾸기"):
        st.caption("변경 내역은 모두 기록됩니다. 결과를 본 뒤 보류 구간을 줄이거나 끄면, 그 구간은 더 이상 검증 데이터가 아닙니다.")
        c1, c2, c3 = st.columns([1, 1, 1])
        en = c1.checkbox("보류 구간 사용 (권장)", value=hold["enabled"], key="holdout_enabled_w")
        d = c2.date_input("보류 구간 시작일", value=date.fromisoformat(hold["start"]), key="holdout_start_w")
        if c3.button("변경 적용"):
            rlog.set_holdout(bool(en), str(d))
            st.rerun()
        if hold["changes"]:
            st.caption(f"변경 기록 {len(hold['changes'])}건 · 마지막: {hold['changes'][-1]['time']}")


# =========================================================================== ②–④ strategy builder

def block_inputs(blocks: dict, prefix: str, defaults: dict) -> dict[str, float | None]:
    chosen: dict[str, float | None] = {}
    for fam in FAMILY_ORDER:
        items = [(k, b) for k, b in blocks.items() if b.family == fam]
        if not items:
            continue
        st.markdown(f"<small><b>{fam}</b></small>", unsafe_allow_html=True)
        for key, b in items:
            c1, c2, c3 = st.columns([2.2, 1.2, 3])
            on = c1.checkbox(b.label, key=init(f"{prefix}_on_{key}", key in defaults), help=b.help)
            p = b.param
            if p is None:
                c2.markdown("<small>고정 규칙</small>", unsafe_allow_html=True)
                val = None
            else:
                cast = int if p.is_int else float
                val = c2.number_input(p.label, min_value=cast(p.minimum), max_value=cast(p.maximum), step=cast(p.step),
                                      key=init(f"{prefix}_val_{key}", cast(defaults.get(key, p.default))),
                                      disabled=not on, help=p.help, label_visibility="collapsed")
            c3.caption(b.rule if p is None else f"{p.label} · {b.rule}")
            if on:
                chosen[key] = None if p is None else float(val)
    return chosen


def strategy_section() -> StrategyConfig | None:
    left, right = st.columns(2, gap="large")
    with left:
        st.subheader("② 진입 조건")
        st.caption("선택한 조건이 **모두** 참인 날 종가 기준으로 신호 → **다음 거래일 시가**에 매수")
        entries = block_inputs(ENTRY_BLOCKS, "entry", {"breakout": 55, "above_ma": 200})
    with right:
        st.subheader("③ 청산 조건")
        st.caption("선택한 조건 중 **하나라도** 참인 날 종가 기준으로 신호 → **다음 거래일 시가**에 매도")
        exits = block_inputs(EXIT_BLOCKS, "exit", {"low_break": 20})

    st.subheader("④ 포트폴리오 · 비용 · 현금")
    c1, c2, c3, c4, c5 = st.columns(5)
    k = c1.number_input("최대 보유 종목 수", 1, 50, key=init("max_positions", 10),
                        help="동시에 보유할 수 있는 최대 종목 수. 새 종목에는 (총자산 ÷ 이 값)만큼, 현금이 부족하면 남은 현금만큼 배정합니다.")
    cost = c2.number_input("편도 거래 비용 (bp)", 0.0, 100.0, step=1.0, key=init("cost_bps", 10.0),
                           help="매수·매도 각각에 부과. 증권사 수수료 + 슬리피지(호가 차이). 10bp = 0.10%. "
                                "본인 계좌의 해외주식 수수료율을 넣으세요.")
    min_price = c3.number_input("최소 주가 ($)", 0.0, 100.0, step=1.0, key=init("min_price", 5.0),
                                help="신호일 종가가 이보다 낮으면 매수하지 않습니다.")
    min_dv = c4.number_input("최소 거래대금 ($백만)", 0.0, 10000.0, step=1.0, key=init("min_dv", 5.0),
                             help="신호일까지 N거래일 평균 (종가 × 거래량)이 이보다 작으면 매수하지 않습니다. "
                                  "그날까지의 데이터만 쓰므로 미래 정보가 섞이지 않습니다.")
    liq = c5.selectbox("거래대금 평균 기간", [20, 60], key=init("liq_days", 20), format_func=lambda d: f"{d}거래일")
    c1, c2 = st.columns([1, 2])
    sizing = c1.selectbox(
        "포지션 크기", list(SIZING_LABELS), format_func=SIZING_LABELS.get, key=init("sizing", "inverse_vol"),
        help="동일 금액: 새 종목마다 총자산 ÷ 최대 보유 수.\n\n"
             "변동성 역가중: 새 종목마다 (총자산 ÷ 최대 보유 수) × min(유니버스 중앙 변동성 ÷ 그 종목 변동성, 2). "
             "변동성 = 신호일까지 60거래일 일간 수익률의 표준편차. 잔잔한 종목은 더 많이(최대 2배), 출렁이는 종목은 적게 사서 "
             "종목마다 위험을 비슷하게 맞춥니다. 진입 후 비중은 다시 맞추지 않습니다.")
    cash = c2.checkbox("남는 현금에 단기국채 수익률 적용 (BIL, 2007년 이전은 13주 국채금리)", key=init("cash_yield", True),
                       help="추세추종은 하락장에서 현금으로 피하는 것이 핵심이라, 현금 수익률 0%는 전략을 부당하게 불리하게 만듭니다. "
                            "실제로는 SGOV·BIL 같은 단기국채 ETF나 증권사 달러 예수금 이자로 비슷한 수익을 얻을 수 있습니다.")
    if not entries or not exits:
        st.warning("진입 조건과 청산 조건을 각각 하나 이상 선택하세요.")
        return None
    return StrategyConfig(entries=entries, exits=exits, start=str(st.session_state["start"]),
                          end=str(st.session_state["end"]), max_positions=int(k), cost_bps=float(cost),
                          min_price=float(min_price), min_dollar_volume=float(min_dv) * 1e6,
                          liquidity_days=int(liq), cash_yield=bool(cash), sizing=str(sizing))


# =========================================================================== backtest view

def run_and_store(panel: Panel, cfg: StrategyConfig) -> None:
    res = run_backtest(panel, cfg)
    base = baselines(panel, cfg, res.equity.index)
    rlog.log_trials([("backtest", cfg, panel.tradable_symbols, res.equity)])
    st.session_state["bt"] = {"res": res, "base": base, "panel": panel, "check": None}


def kpi_row(m: dict, keys: list[str]) -> None:
    cols = st.columns(len(keys))
    for col, key in zip(cols, keys):
        col.metric(DEFINITIONS[key].label, fmt_value(key, m[key]), help=tip(key))
        if f"spy_{key}" in m:
            col.caption(f"SPY {fmt_value(key, m['spy_' + key])}")


def scorecard(res, base: dict) -> None:
    curves = {"전략": curve_metrics(res.equity), **{BASELINE_LABELS[k]: curve_metrics(v.dropna()) for k, v in base.items()}}
    rows = []
    for key in SCORECARD:
        d = DEFINITIONS[key]
        vals = {name: c[key] for name, c in curves.items()}
        finite = {n: v for n, v in vals.items() if v is not None and np.isfinite(v)}
        best = None
        if d.better and finite:
            best = (max if d.better == "high" else min)(finite, key=finite.get)
        row = {"지표": d.label}
        for name, v in vals.items():
            row[name] = fmt_value(key, v) + (" ★" if name == best else "")
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch", column_config={
        "전략": st.column_config.TextColumn("전략", help="지금 설정한 전략"),
        **{BASELINE_LABELS[k]: st.column_config.TextColumn(BASELINE_LABELS[k], help=BASELINE_HELP[k]) for k in base},
    })
    st.caption("★ = 그 지표에서 가장 좋은 값. 기준선은 같은 기간·같은 비용·같은 현금 수익률로 계산합니다. "
               "복잡한 전략이 ‘SPY 200일선’(규칙 1개)보다 못하면 복잡하게 만들 이유가 없습니다.")


def equity_chart(res, base: dict, log: bool) -> go.Figure:
    fig = go.Figure()
    for k, s in base.items():
        fig.add_scatter(x=s.index, y=s.values, name=BASELINE_LABELS[k], line=BASE_STYLE[k],
                        hovertemplate="%{x|%Y-%m-%d}<br>" + BASELINE_LABELS[k] + " %{y:.3f}<extra></extra>")
    fig.add_scatter(x=res.equity.index, y=res.equity.values, name="전략", line=dict(color=BLUE, width=2.5),
                    hovertemplate="%{x|%Y-%m-%d}<br>전략 %{y:.3f}<extra></extra>")
    fig.update_layout(height=360, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      title="자산 곡선 (시작 = 1.0)", legend=dict(orientation="h", y=1.14, x=1, xanchor="right"),
                      yaxis_type="log" if log else "linear")
    return fig


def drawdown_chart(eq: pd.Series, bench: pd.Series) -> go.Figure:
    fig = go.Figure()
    for s, name, color in ((bench, BENCHMARK, ORANGE), (eq, "전략", BLUE)):
        dd = drawdown(s) * 100
        fig.add_scatter(x=dd.index, y=dd.values, name=name, line=dict(color=color, width=2),
                        hovertemplate="%{x|%Y-%m-%d}<br>" + name + " %{y:.1f}%<extra></extra>")
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      title="낙폭 (직전 최고점 대비, %)", legend=dict(orientation="h", y=1.18, x=1, xanchor="right"),
                      yaxis_ticksuffix="%")
    return fig


def annual_chart(ann: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    years = [f"{y}{'*' if p else ''}" for y, p in zip(ann["연도"], ann["부분 연도"])]
    fig.add_bar(x=years, y=ann["SPY"] * 100, name="SPY", marker_color=ORANGE,
                hovertemplate="%{x}<br>SPY %{y:.1f}%<extra></extra>")
    fig.add_bar(x=years, y=ann["전략"] * 100, name="전략", marker_color=BLUE,
                hovertemplate="%{x}<br>전략 %{y:.1f}%<extra></extra>")
    fig.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), barmode="group", bargap=0.25, bargroupgap=0.08,
                      title="연도별 수익률 (* = 일부 기간만 포함)", yaxis_ticksuffix="%",
                      legend=dict(orientation="h", y=1.16, x=1, xanchor="right"))
    return fig


def gate_check(m: dict) -> None:
    g = gates_now()
    failed = check_gates(m, g)
    items = [
        (GATE_TEXT["min_cagr_ratio"][0], f"CAGR {fmt_value('cagr', m['cagr'])} vs 기준 {g.min_cagr_ratio:.2f}×SPY {fmt_value('cagr', m['spy_cagr'])}"),
        (GATE_TEXT["max_mdd_ratio"][0], f"|MDD| {fmt_value('mdd', abs(m['mdd']))} vs 기준 {g.max_mdd_ratio:.2f}×|SPY MDD| {fmt_value('mdd', abs(m['spy_mdd']))}"),
        (GATE_TEXT["min_trades"][0], f"완료 거래 {int(m['n_trades'])} vs 기준 {g.min_trades}"),
    ]
    st.markdown(BR.join(f"{'❌' if name in failed else '✅'} **{name}** — {detail}" for name, detail in items))
    st.caption("기준값은 ‘🧭 강건성 격자’ 탭에서 바꿀 수 있습니다.")


def checklist_view(bt: dict, dsr: float) -> None:
    st.markdown("#### 강건성 점검")
    st.caption("한 전략이 ‘우연히 좋은 한 점’인지 확인합니다. 항목마다 통과/미달만 표시하고 점수로 합치지 않습니다. "
               "점검에 쓰인 추가 백테스트도 시험 횟수에 기록됩니다.")
    if st.button("🔍 강건성 점검 실행", help="주변 파라미터·비용 2배 등 수 회의 추가 백테스트를 돌립니다 (수 초)."):
        res, panel = bt["res"], bt["panel"]
        collected = []
        with st.spinner("점검 중…"):
            bt["check"] = strategy_checklist(
                panel, res.config, gates_now(), res, bt["base"]["spy_ma200"], dsr,
                on_result=lambda c, r: collected.append(("check", c, panel.tradable_symbols, r.equity)))
        rlog.log_trials(collected)
    if bt["check"]:
        applicable = [c for c in bt["check"] if c["passed"] is not None]
        st.markdown(f"**{len(applicable)}개 항목 중 {sum(c['passed'] for c in applicable)}개 통과**")
        for c in bt["check"]:
            icon = "➖" if c["passed"] is None else ("✅" if c["passed"] else "❌")
            st.markdown(f"{icon} **{c['label']}** — {c['detail']}{BR}<small>{CHECK_TEXT[c['key']][1]}</small>",
                        unsafe_allow_html=True)


def metric_table(m: dict, keys: list[str]) -> None:
    rows = [{"지표": DEFINITIONS[k].label, "값": fmt_value(k, m[k]),
             "뜻 (계산식)": f"{DEFINITIONS[k].short} ({DEFINITIONS[k].formula})"} for k in keys]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch",
                 column_config={"뜻 (계산식)": st.column_config.TextColumn(width="large")})


def show_backtest(bt: dict) -> None:
    res, base, panel = bt["res"], bt["base"], bt["panel"]
    m = all_metrics(res)
    n_trials, sr_var = rlog.trial_stats()
    n_eff, rho, _ = rlog.effective_trials()
    dsr = deflated_sharpe(res.equity, max(1, int(round(n_eff))), sr_var)
    m["dsr"] = dsr
    s, e = res.period
    st.info(f"**백테스트 기간 {s.date()} → {e.date()}** ({len(res.equity):,} 거래일) · 거래 대상 {res.universe_size}종목 · "
            f"현금 수익률 {'단기국채' if res.config.cash_yield else '0%'} · 데이터: {panel.source}")
    for w in res.warnings:
        st.warning(w)
    st.caption(res.config.describe())

    st.markdown("#### 핵심 성과")
    kpi_row(m, ["total_return", "cagr", "mdd", "sharpe"])
    kpi_row(m, ["calmar", "sortino", "worst_12m", "dsr"])
    st.caption(f"DSR 계산: 시험 횟수 N = {n_trials:,} · 시험끼리의 평균 상관 ρ = {rho:.2f} → 유효 시험 수 "
               f"N_eff = ρ + (1−ρ)·N ≈ {n_eff:,.0f} (비슷한 시험을 중복으로 세지 않도록 보정)")

    st.markdown("#### 기준선과 비교")
    scorecard(res, base)
    log = st.toggle("로그 눈금", value=False, key="log_scale")
    st.plotly_chart(equity_chart(res, base, log), width="stretch")
    st.plotly_chart(drawdown_chart(res.equity, res.benchmark), width="stretch")

    st.markdown("#### 최소 통과 조건 (Gate)")
    gate_check(m)
    checklist_view(bt, dsr)

    st.markdown("#### 연도별")
    ann = annual_table(res)
    st.plotly_chart(annual_chart(ann), width="stretch")
    with st.expander("연도별 표"):
        st.dataframe(ann, hide_index=True, width="stretch", column_config={
            "연도": st.column_config.NumberColumn(format="%d"),
            **{c: st.column_config.NumberColumn(format="percent") for c in ["전략", "SPY", "차이", "전략 MDD", "평균 거래 수익"]},
            "부분 연도": st.column_config.CheckboxColumn(help="거래일이 240일 미만인 시작/끝 연도."),
        })

    st.markdown("#### 거래 단위 통계 (보조: 거래당 통계적 우위가 있는지)")
    metric_table(m, ["n_trades", "win_rate", "avg_trade", "median_trade", "profit_factor", "payoff", "t_stat",
                     "avg_hold", "trades_per_year", "exposure"])
    with st.expander(f"거래 목록 ({len(res.trades)}건) · 현재 보유 {len(res.open_positions)}종목"):
        st.dataframe(res.trades.sort_values("entry_date", ascending=False), hide_index=True, width="stretch",
                     column_config={"return": st.column_config.NumberColumn("수익률", format="percent"),
                                    "holding_days": st.column_config.NumberColumn("보유 거래일"),
                                    "exit_reason": "청산 사유"})
        if len(res.open_positions):
            st.markdown("**기간 끝 시점 보유 종목** (완료 거래 통계에는 포함되지 않음)")
            st.dataframe(res.open_positions, hide_index=True, width="stretch",
                         column_config={"unrealized_return": st.column_config.NumberColumn("평가 수익률", format="percent")})

    with st.form("save_bt", border=False):
        c1, c2 = st.columns([3, 1])
        name = c1.text_input("결과 이름", value="", placeholder="예: 55일 돌파 + 20일 저가 이탈")
        if c2.form_submit_button("💾 결과 저장", width="stretch",
                                 help="저장한 결과만 ‘최종 검증’에서 보류 구간 평가를 할 수 있습니다."):
            path = store.save_backtest(res, m, name or res.config.describe(), panel)
            st.success(f"저장했습니다: lab_results/{path.name}")


# =========================================================================== grid view

def grid_controls(cfg: StrategyConfig):
    params = cfg.tunable() + ["max_positions"]

    def axis(col, label: str, key: str, allow_none: bool, default_idx: int):
        options = ([None] if allow_none else []) + params
        sk = f"grid_{key}"
        if st.session_state.get(sk, "missing") not in options:  # the chosen block was switched off
            st.session_state.pop(sk, None)
        choice = col.selectbox(label, options, key=init(sk, options[min(default_idx, len(options) - 1)]),
                               format_func=lambda k: "없음 (1차원 격자)" if k is None else param_label(k))
        if choice is None:
            return None, None
        if choice == "max_positions":
            lo, hi, step, cur = 1, 50, 1, cfg.max_positions
        else:
            p = ALL_BLOCKS[choice].param
            lo, hi, step, cur = p.minimum, p.maximum, p.step, cfg.param_value(choice)
        a, b, c = col.columns(3)
        pre = f"g{key}_{choice}"
        start = a.number_input("시작", min_value=float(lo), max_value=float(hi), step=float(step),
                               key=init(f"{pre}_s", float(max(lo, cur - 2 * step))))
        stop = b.number_input("끝", min_value=float(lo), max_value=float(hi), step=float(step),
                              key=init(f"{pre}_e", float(min(hi, cur + 2 * step))))
        stp = c.number_input("간격", min_value=float(step) / 4, step=float(step), key=init(f"{pre}_d", float(step)))
        try:
            vals = axis_values(start, stop, stp)
        except ValueError as exc:
            col.error(str(exc))
            return choice, []
        col.caption("값: " + ", ".join(f"{v:g}" for v in vals))
        return choice, vals

    exit_params = [k for k in cfg.tunable() if k in EXIT_BLOCKS]
    y_default = 1 + params.index(exit_params[0]) if exit_params else 0  # default: signal × exit
    c1, c2 = st.columns(2, gap="large")
    x_key, xs = axis(c1, "X축 파라미터", "x", False, 0)
    y_key, ys = axis(c2, "Y축 파라미터", "y", True, y_default)

    st.markdown("**최소 통과 조건 (Gate)** — 하나라도 못 넘으면 탈락. 가중 점수는 쓰지 않습니다.")
    g1, g2, g3 = st.columns(3)
    gates = Gates(
        min_cagr_ratio=g1.number_input("수익 유지: SPY CAGR 대비 최소 비율", 0.0, 2.0, step=0.05, key=init("gate_cagr", 0.80),
                                       help=GATE_TEXT["min_cagr_ratio"][1] + "\n\n" + GATE_TEXT["min_cagr_ratio"][2]),
        max_mdd_ratio=g2.number_input("낙폭 축소: SPY MDD 대비 최대 비율", 0.1, 2.0, step=0.05, key=init("gate_mdd", 0.75),
                                      help=GATE_TEXT["max_mdd_ratio"][1] + "\n\n" + GATE_TEXT["max_mdd_ratio"][2]),
        min_trades=int(g3.number_input("최소 완료 거래 수", 0, 10000, step=10, key=init("gate_trades", 30),
                                       help=GATE_TEXT["min_trades"][1] + "\n\n" + GATE_TEXT["min_trades"][2])),
    )
    return x_key, xs, y_key, ys, gates


def heatmap(cells: pd.DataFrame, x_key: str, y_key: str | None, metric: str) -> go.Figure:
    xs = sorted(cells["x"].unique())
    ys = sorted(cells["y"].dropna().unique()) if y_key else [None]
    z, text, hover = [], [], []
    for y in ys:
        row = cells[cells["y"] == y] if y_key else cells
        zr, tr, hr = [], [], []
        for x in xs:
            r = row[row["x"] == x].iloc[0]
            v = r[metric]
            zr.append(v if pd.notna(v) else None)
            tr.append(f"{fmt_value(metric, v) if metric in DEFINITIONS else v}{'<br>✓' if r['pass'] else ''}")
            hr.append(
                f"{param_label(x_key)} = {x:g}" + (f"<br>{param_label(y_key)} = {y:g}" if y_key else "")
                + f"<br>CAGR {fmt_value('cagr', r['cagr'])} · MDD {fmt_value('mdd', r['mdd'])}"
                + f"<br>SPY 대비 수익 {r['cagr_ratio']:.2f} · 낙폭 {r['mdd_ratio']:.2f} · 거래 {int(r['n_trades'])}"
                + ("<br><b>Gate 통과</b>" if r["pass"] else f"<br>탈락: {r['fail_reasons']}")
                + (f"<br>이웃 생존율 {r['neighbor_survival']:.2f} · 영역 크기 {int(r['region_size'])}" if pd.notna(r['neighbor_survival']) else "")
            )
        z.append(zr)
        text.append(tr)
        hover.append(hr)
    better = DEFINITIONS[metric].better if metric in DEFINITIONS else "high"
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{v:g}" for v in xs], y=[f"{v:g}" for v in ys] if y_key else ["–"],
        text=text, texttemplate="%{text}", customdata=hover, hovertemplate="%{customdata}<extra></extra>",
        colorscale=SEQ_BLUE, reversescale=(better == "low"), xgap=2, ygap=2, colorbar=dict(title=""),
    ))
    fig.update_layout(height=120 + 48 * len(ys), margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title=param_label(x_key), yaxis_title=param_label(y_key) if y_key else "",
                      xaxis_type="category", yaxis_type="category")
    return fig


GRID_METRICS = ["cagr", "mdd", "calmar", "sharpe", "sortino", "worst_12m", "cagr_ratio", "mdd_ratio", "n_trades", "t_stat"]


def show_grid_result(cells: pd.DataFrame, x_key: str, y_key: str | None, summary: dict, gates: dict, key: str) -> None:
    c = st.columns(4)
    c[0].metric("전체 조합", summary["total"])
    c[1].metric("Gate 통과", summary["passed"], help="수익 유지 · 낙폭 축소 · 최소 거래 수를 모두 만족한 조합 수")
    c[2].metric("연결 영역 수", summary["regions"], help=ROBUST_TEXT["region_size"][1])
    c[3].metric("가장 큰 영역 (셀 수)", summary["largest_region"], help=ROBUST_TEXT["region_size"][1])
    st.caption(f"Gate: CAGR ≥ {gates['min_cagr_ratio']:.2f}×SPY · |MDD| ≤ {gates['max_mdd_ratio']:.2f}×|SPY MDD| · 완료 거래 ≥ {gates['min_trades']}")

    metric = st.selectbox("히트맵에 표시할 지표", GRID_METRICS, key=f"hm_metric_{key}",
                          format_func=lambda k: DEFINITIONS[k].label, help="✓ 표시는 Gate를 통과한 조합입니다.")
    st.caption(DEFINITIONS[metric].short)
    st.plotly_chart(heatmap(cells, x_key, y_key, metric), width="stretch")

    cand = cells[cells["pass"]].sort_values(["region_size", "neighbor_survival", "loyo_ratio", "calmar"], ascending=False, na_position="last")
    st.markdown("#### 통과 후보 정렬")
    st.caption("정렬 순서(사전식): ① 연결 영역 크기 → ② 이웃 생존율 → ③ LOYO 통과 비율 → ④ 칼마 비율. 앞 기준이 같을 때만 다음 기준을 봅니다.")
    if cand.empty:
        st.warning("Gate를 통과한 조합이 없습니다. 기준을 낮추기 전에, 다른 신호·청산 조합이나 유니버스를 먼저 검토하세요.")
        return
    cols = ["x"] + (["y"] if y_key else []) + ["region_size", "neighbor_survival", "loyo_ratio", "loyo_fail_years",
                                               "cagr", "mdd", "calmar", "cagr_ratio", "mdd_ratio", "n_trades", "t_stat"]
    num = st.column_config.NumberColumn
    st.dataframe(cand[cols], hide_index=True, width="stretch", column_config={
        "x": num(param_label(x_key), format="%g"),
        "y": num(param_label(y_key) if y_key else "", format="%g"),
        "region_size": num(ROBUST_TEXT["region_size"][0], help=ROBUST_TEXT["region_size"][1]),
        "neighbor_survival": num(ROBUST_TEXT["neighbor_survival"][0], format="%.2f", help=ROBUST_TEXT["neighbor_survival"][1]),
        "loyo_ratio": num(ROBUST_TEXT["loyo_ratio"][0], format="%.2f", help=ROBUST_TEXT["loyo_ratio"][1]),
        "loyo_fail_years": st.column_config.TextColumn("LOYO 실패 연도", help="그 해를 빼면 Gate를 통과하지 못하는 연도"),
        "cagr": num("CAGR", format="percent", help=tip("cagr")),
        "mdd": num("MDD", format="percent", help=tip("mdd")),
        "calmar": num("Calmar", format="%.2f", help=tip("calmar")),
        "cagr_ratio": num("SPY 대비 수익", format="%.2f", help=tip("cagr_ratio")),
        "mdd_ratio": num("SPY 대비 낙폭", format="%.2f", help=tip("mdd_ratio")),
        "n_trades": num("거래 수", format="%d", help=tip("n_trades")),
        "t_stat": num("t-통계량", format="%.2f", help=tip("t_stat")),
    })


def apply_params(values: dict[str, float]) -> None:
    """Callback: write grid-cell parameters back into the strategy builder widgets."""
    for k, v in values.items():
        if k == "max_positions":
            st.session_state["max_positions"] = int(v)
        else:
            prefix = "entry" if k in ENTRY_BLOCKS else "exit"
            st.session_state[f"{prefix}_val_{k}"] = int(v) if ALL_BLOCKS[k].param.is_int else float(v)
    st.session_state["auto_run"] = True
    st.toast("파라미터를 바꿨습니다. ‘📈 백테스트’ 탭에서 결과를 확인하세요.")


def grid_tab(panel: Panel, cfg: StrategyConfig) -> None:
    st.caption("파라미터 1–2개를 바꿔 가며 같은 전략을 반복 실행합니다. 나머지 설정은 위 ②–④ 그대로 고정됩니다. "
               "격자의 모든 칸이 시험 횟수에 기록됩니다.")
    x_key, xs, y_key, ys, gates = grid_controls(cfg)
    n = len(xs or []) * (len(ys) if y_key else 1)
    too_big = (len(xs or []) > MAX_AXIS_VALUES) or (y_key and len(ys or []) > MAX_AXIS_VALUES)
    st.caption(f"조합 {n}개 · 예상 소요 약 {max(1, round(estimate_seconds(panel, cfg, n)))}초"
               + (f" · ⚠️ 축마다 최대 {MAX_AXIS_VALUES}개" if too_big else ""))
    if st.button("▶ 격자 실행", type="primary", disabled=bool(too_big or n == 0)):
        bar = st.progress(0.0, text="실행 중…")
        collected = []
        try:
            g = run_grid(panel, cfg, x_key, xs, y_key, ys, gates,
                         progress=lambda f: bar.progress(f, text=f"실행 중… {f:.0%}"),
                         on_result=lambda c, r: collected.append(("grid", c, panel.tradable_symbols, r.equity)))
            st.session_state["grid_result"] = (g, panel.source)
            rlog.log_trials(collected)
        except ValueError as exc:
            st.error(str(exc))
        bar.empty()
    stored = st.session_state.get("grid_result")
    if stored is None:
        return
    g, source = stored
    st.caption(f"기준 전략: {g.base.describe()} · 기간 {g.base.start} → {g.base.end} · {g.seconds:.0f}초")
    show_grid_result(g.cells, g.x_key, g.y_key, g.summary(), g.gates.to_dict(), "live")
    cand = g.candidates()
    if len(cand):
        labels = [
            f"{r['순위']}위 · {param_label(g.x_key)}={r['x']:g}" + (f", {param_label(g.y_key)}={r['y']:g}" if g.y_key else "")
            for _, r in cand.head(20).iterrows()
        ]
        c1, c2 = st.columns([3, 1])
        pick = c1.selectbox("후보를 골라 단일 백테스트로 자세히 보기", range(len(labels)), format_func=lambda i: labels[i])
        r = cand.iloc[pick]
        vals = {g.x_key: r["x"], **({g.y_key: r["y"]} if g.y_key else {})}
        c2.button("이 값으로 설정 + 실행", on_click=apply_params, args=(vals,), width="stretch",
                  help="위 ②–④의 해당 파라미터를 이 값으로 바꾸고 ‘백테스트’ 탭에서 실행합니다.")
    with st.form("save_grid", border=False):
        c1, c2 = st.columns([3, 1])
        name = c1.text_input("결과 이름", value="", placeholder="예: 돌파 N × 저가 이탈 N")
        if c2.form_submit_button("💾 격자 저장", width="stretch"):
            s, e = pd.Timestamp(g.base.start), pd.Timestamp(g.base.end)
            path = store.save_grid(g, name or f"격자 {param_label(g.x_key)} × {param_label(g.y_key) if g.y_key else '-'}", source, (s, e))
            st.success(f"저장했습니다: lab_results/{path.name}")


# =========================================================================== family comparison view

FAMILY_FMT = {"pass_share": "percent", "largest_region": "%d", "beat_ma200": "percent", "median_calmar": "%.2f",
              "p25_calmar": "%.2f", "median_cagr": "percent", "median_mdd": "percent", "median_trades": "%d"}


def _fam_fmt(metric: str, v: float) -> str:
    if v is None or not np.isfinite(v):
        return "–"
    kind = FAMILY_FMT[metric]
    return f"{v:.0%}" if kind == "percent" else (f"{v:,.0f}" if kind == "%d" else f"{v:.2f}")


def family_matrix(summary: pd.DataFrame, metric: str) -> go.Figure:
    entries = [e for e in ENTRY_MENU if e in set(summary["entry"])]
    exits = [x for x in EXIT_MENU if x in set(summary["exit"])]
    z, text = [], []
    for e in entries:
        row = [summary.loc[(summary["entry"] == e) & (summary["exit"] == x), metric].iloc[0] for x in exits]
        z.append(row)
        text.append([_fam_fmt(metric, v) for v in row])
    fig = go.Figure(go.Heatmap(
        z=z, x=[EXIT_BLOCKS[x].label for x in exits], y=[ENTRY_BLOCKS[e].label for e in entries], text=text,
        texttemplate="%{text}", colorscale=SEQ_BLUE, xgap=3, ygap=3, colorbar=dict(title=""),
        hovertemplate="진입: %{y}<br>청산: %{x}<br>" + SUMMARY_TEXT[metric][0] + ": %{text}<extra></extra>",
    ))
    fig.update_layout(height=80 + 45 * len(entries), margin=dict(l=10, r=10, t=10, b=10), xaxis_title="청산 계열",
                      yaxis_title="진입 계열",
                      yaxis_autorange="reversed")
    return fig


def show_family_result(summary: pd.DataFrame, cells: pd.DataFrame, gates: dict, ma200: float, key: str) -> None:
    st.caption(f"Gate: CAGR ≥ {gates['min_cagr_ratio']:.2f}×SPY · |MDD| ≤ {gates['max_mdd_ratio']:.2f}×|SPY MDD| · "
               f"완료 거래 ≥ {gates['min_trades']} · 같은 조건의 SPY 200일선 칼마 = {ma200:.2f}")
    st.markdown("#### 계열 순위")
    st.caption("정렬(사전식): ① Gate 통과 비율 → ② 가장 큰 연결 영역 → ③ SPY 200일선 이긴 비율 → ④ 중간 칼마. "
               "앞 기준이 같을 때만 다음 기준을 봅니다. 가중 점수는 없습니다. 각 열 이름에 마우스를 올리면 정의가 보입니다.")
    num = st.column_config.NumberColumn
    st.dataframe(summary.drop(columns=["entry", "exit"]), hide_index=True, width="stretch", column_config={
        "family": st.column_config.TextColumn("계열 (진입 × 청산)", width="large"),
        **{k: num(SUMMARY_TEXT[k][0], format=FAMILY_FMT[k], help=SUMMARY_TEXT[k][1]) for k in SUMMARY_TEXT},
    })
    metric = st.selectbox("계열 지도에 표시할 값", list(SUMMARY_TEXT), key=f"fam_metric_{key}",
                          format_func=lambda k: SUMMARY_TEXT[k][0], help="행 = 진입 계열, 열 = 청산 계열")
    st.caption(SUMMARY_TEXT[metric][1])
    st.plotly_chart(family_matrix(summary, metric), width="stretch")

    st.markdown("#### 계열 자세히 보기")
    labels = {f"{r['entry']}|{r['exit']}": f"{r['순위']}위 · {r['family']}" for _, r in summary.iterrows()}
    pick = st.selectbox("계열 선택", list(labels), format_func=labels.get, key=f"fam_pick_{key}")
    e, x = pick.split("|")
    sub = cells[(cells["entry"] == e) & (cells["exit"] == x)].reset_index(drop=True)
    passed = sub[sub["pass"]]
    show_grid_result(sub, e, x, {"total": len(sub), "passed": len(passed), "regions": passed["region"].nunique(),
                                 "largest_region": int(sub["region_size"].max())}, gates, f"fam_{key}")
    if key == "live":
        cand = passed.sort_values(["region_size", "neighbor_survival", "loyo_ratio", "calmar"], ascending=False)
        pool = cand if len(cand) else sub.sort_values("calmar", ascending=False)
        opts = [(r["x"], r["y"]) for _, r in pool.head(20).iterrows()]
        c1, c2 = st.columns([3, 1])
        i = c1.selectbox("이 계열의 한 칸을 단일 백테스트로 보기" + ("" if len(cand) else " (통과 칸 없음 · 칼마 순)"),
                         range(len(opts)), key=f"fam_cell_{pick}",
                         format_func=lambda k: f"{param_label(e)}={opts[k][0]:g}, {param_label(x)}={opts[k][1]:g}")
        c2.button("이 계열·값으로 설정 + 실행", on_click=apply_family, args=(e, opts[i][0], x, opts[i][1]),
                  width="stretch", help="② 진입·③ 청산 조건을 이 계열 하나씩으로 바꾸고 ‘백테스트’ 탭에서 실행합니다.")


def apply_family(entry_key: str, entry_value: float, exit_key: str, exit_value: float) -> None:
    """Callback: replace the builder's entry/exit selection with one family cell."""
    for k in ENTRY_BLOCKS:
        st.session_state[f"entry_on_{k}"] = k == entry_key
    for k in EXIT_BLOCKS:
        st.session_state[f"exit_on_{k}"] = k == exit_key
    apply_params({entry_key: entry_value, exit_key: exit_value})


def family_tab(panel: Panel, cfg: StrategyConfig) -> None:
    n_fam = len(family_pairs())
    st.caption(f"미리 정해 둔 메뉴(진입 {len(ENTRY_MENU)}종 × 청산 {len(EXIT_MENU)}종 = {n_fam}개 계열)를 각각 6×6 격자로 돌려, "
               "**최고점이 아니라 격자 전체가 얼마나 넓게 통하는지**로 "
               "계열을 비교합니다. 위 ①(유니버스·기간)과 ④(포트폴리오·비용·현금) 설정을 쓰고, ②·③에서 고른 조건은 쓰지 않습니다. "
               f"Gate 기준은 ‘🧭 강건성 격자’ 탭의 값을 씁니다. 모든 칸({cell_count()}개)이 시험 횟수에 기록됩니다.")
    with st.expander(f"고정 메뉴 보기 ({MENU_VERSION})"):
        st.markdown("각 축의 값은 결과를 보기 전에 정했습니다. 짧은·중간·긴 기간을 대략 등비 간격으로 덮습니다. "
                    "결과를 본 뒤 값을 바꾸면 그것은 새로운 시험입니다.")
        st.dataframe(pd.DataFrame(
            [{"구분": "진입", "계열": ENTRY_BLOCKS[k].label, "조건": ENTRY_BLOCKS[k].rule,
              "격자 값": ", ".join(f"{v:g}" for v in vals)} for k, vals in ENTRY_MENU.items()]
            + [{"구분": "청산", "계열": EXIT_BLOCKS[k].label, "조건": EXIT_BLOCKS[k].rule,
                "격자 값": ", ".join(f"{v:g}" for v in vals)} for k, vals in EXIT_MENU.items()]),
            hide_index=True, width="stretch")
    est = estimate_seconds(panel, cfg, cell_count())
    st.caption(f"예상 소요 약 {max(1, round(est / 60))}분 (컴퓨터 상태에 따라 더 걸릴 수 있음) · 실행 중에는 다른 조작을 하지 마세요.")
    if st.button(f"▶ {n_fam}개 계열 비교 실행", type="primary"):
        bar = st.progress(0.0, text="계열 비교 중…")
        collected = []
        try:
            fc = run_family_comparison(panel, cfg, gates_now(),
                                       progress=lambda f: bar.progress(min(f, 1.0), text=f"계열 비교 중… {f:.0%}"),
                                       on_result=lambda c, r: collected.append(("family", c, panel.tradable_symbols, r.equity)))
            st.session_state["family_result"] = (fc, panel.source, panel.tradable_symbols)
            rlog.log_trials(collected)
        except ValueError as exc:
            st.error(str(exc))
        bar.empty()
    stored = st.session_state.get("family_result")
    if stored is None:
        return
    fc, source, universe = stored
    if fc.base.start != cfg.start or fc.base.end != cfg.end or universe != panel.tradable_symbols:
        st.warning("아래 결과는 현재 유니버스·기간과 다른 설정으로 실행한 것입니다.")
    st.caption(f"기간 {fc.base.start} → {fc.base.end} · 거래 대상 {len(universe)}종목 · {fc.seconds:.0f}초")
    show_family_result(fc.summary(), fc.all_cells(), fc.gates.to_dict(), fc.ma200_calmar, "live")
    with st.form("save_family", border=False):
        c1, c2 = st.columns([3, 1])
        name = c1.text_input("결과 이름", value="", placeholder="예: 장기 ETF 2000–2023 계열 비교")
        if c2.form_submit_button("💾 계열 비교 저장", width="stretch"):
            path = store.save_family(fc, name or f"계열 비교 {fc.base.start} – {fc.base.end}", source, universe)
            st.success(f"저장했습니다: lab_results/{path.name}")


# =========================================================================== confirmation-condition test view

VERDICT_SHORT = {"base": "기준", "improve": "✅ 개선", "mixed": "↔ 엇갈림", "same": "＝ 변화 없음", "worse": "❌ 악화"}


def show_refinement(table: pd.DataFrame, cells: pd.DataFrame, entry_key: str, exit_key: str, gates: dict,
                    ma200: float, key: str) -> None:
    st.caption(f"계열: **{family_label(entry_key, exit_key)}** · Gate: CAGR ≥ {gates['min_cagr_ratio']:.2f}×SPY · "
               f"|MDD| ≤ {gates['max_mdd_ratio']:.2f}×|SPY MDD| · 완료 거래 ≥ {gates['min_trades']} · "
               f"SPY 200일선 칼마 = {ma200:.2f}")
    improved = table.loc[table["verdict"] == "improve", "condition"].tolist()
    if improved:
        st.success("모든 기준에서 같거나 나아진 조건: " + ", ".join(improved)
                   + ". 다른 유니버스·기간에서도 개선되는지 확인한 뒤에 채택하세요.")
    else:
        st.info("모든 기준에서 같거나 나아진 조건이 없습니다. 이 데이터에서는 확인 조건 없이 기본 계열을 유지하는 것이 맞습니다.")
    num = st.column_config.NumberColumn
    shown = table.drop(columns=["addon"]).assign(verdict=table["verdict"].map(VERDICT_SHORT))
    st.dataframe(shown, hide_index=True, width="stretch", column_config={
        "condition": st.column_config.TextColumn("추가한 확인 조건 (기본값)", width="medium"),
        "verdict": st.column_config.TextColumn("판정", help="\n\n".join(f"{VERDICT_SHORT[k]}: {v}" for k, v in VERDICT_TEXT.items())),
        **{k: num(SUMMARY_TEXT[k][0], format=FAMILY_FMT[k], help=SUMMARY_TEXT[k][1]) for k in SUMMARY_TEXT},
    })
    labels = {r["addon"]: f"{VERDICT_SHORT[r['verdict']]} · {r['condition']}" for _, r in table.iterrows()}
    pick = st.selectbox("격자 자세히 보기", list(labels), format_func=labels.get, key=f"ref_pick_{key}")
    sub = cells[cells["addon"].fillna("") == pick].reset_index(drop=True)
    passed = sub[sub["pass"]]
    show_grid_result(sub, entry_key, exit_key, {"total": len(sub), "passed": len(passed), "regions": passed["region"].nunique(),
                                                "largest_region": int(sub["region_size"].max())}, gates, f"ref_{key}")
    if key == "live":
        pool = passed.sort_values(["region_size", "neighbor_survival", "loyo_ratio", "calmar"], ascending=False)
        pool = pool if len(pool) else sub.sort_values("calmar", ascending=False)
        opts = [(r["x"], r["y"]) for _, r in pool.head(20).iterrows()]
        c1, c2 = st.columns([3, 1])
        i = c1.selectbox("한 칸을 단일 백테스트로 보기", range(len(opts)), key=f"ref_cell_{pick}",
                         format_func=lambda k: f"{param_label(entry_key)}={opts[k][0]:g}, {param_label(exit_key)}={opts[k][1]:g}")
        c2.button("이 조합으로 설정 + 실행", on_click=apply_refinement, width="stretch",
                  args=(entry_key, opts[i][0], exit_key, opts[i][1], pick),
                  help="② 진입·③ 청산을 이 계열(+ 확인 조건)로 바꾸고 ‘백테스트’ 탭에서 실행합니다.")


def apply_refinement(entry_key: str, entry_value: float, exit_key: str, exit_value: float, addon: str) -> None:
    for k in ENTRY_BLOCKS:
        st.session_state[f"entry_on_{k}"] = k in (entry_key, addon)
    for k in EXIT_BLOCKS:
        st.session_state[f"exit_on_{k}"] = k == exit_key
    values = {entry_key: entry_value, exit_key: exit_value}
    if addon and ENTRY_BLOCKS[addon].param is not None:
        values[addon] = addon_value(addon)
    apply_params(values)


def refine_tab(panel: Panel, cfg: StrategyConfig) -> None:
    st.caption("고른 계열 하나에 **확인 조건을 하나씩** 더해 같은 6×6 격자를 다시 돌리고, 계열 비교의 네 기준으로 원래 계열과 비교합니다. "
               "확인 조건은 각 블록의 **기본값 하나로만** 시험합니다(새 파라미터 탐색 없음). "
               "네 기준 모두에서 같거나 낫고 하나 이상 나아야 ‘개선’입니다. 위 ①·④ 설정을 쓰고 ②·③은 쓰지 않습니다.")
    pairs = [(e, x) for e in ENTRY_MENU for x in EXIT_MENU]
    stored_fam = st.session_state.get("family_result")
    default = 0
    if stored_fam is not None:
        top = stored_fam[0].summary().iloc[0]
        default = pairs.index((top["entry"], top["exit"]))
    choice = st.selectbox("다듬을 계열", range(len(pairs)), index=default, key="ref_family",
                          format_func=lambda i: family_label(*pairs[i]),
                          help="‘전략 계열 비교’를 실행했다면 1위 계열이 기본으로 선택됩니다.")
    e, x = pairs[choice]
    n = (1 + len(addon_candidates(e))) * len(ENTRY_MENU[e]) * len(EXIT_MENU[x])
    st.caption(f"기준 + 확인 조건 {len(addon_candidates(e))}개 = 격자 {1 + len(addon_candidates(e))}개 · 백테스트 {n}회 · "
               f"예상 약 {max(1, round(estimate_seconds(panel, cfg, n) / 60))}분 · 모두 시험 횟수에 기록됩니다.")
    if st.button("▶ 확인 조건 시험 실행", type="primary"):
        bar = st.progress(0.0, text="시험 중…")
        collected = []
        try:
            ref = run_refinement(panel, cfg, e, x, gates_now(),
                                 progress=lambda f: bar.progress(min(f, 1.0), text=f"시험 중… {f:.0%}"),
                                 on_result=lambda c, r: collected.append(("refine", c, panel.tradable_symbols, r.equity)))
            st.session_state["refine_result"] = (ref, panel.source, panel.tradable_symbols)
            rlog.log_trials(collected)
        except ValueError as exc:
            st.error(str(exc))
        bar.empty()
    stored = st.session_state.get("refine_result")
    if stored is None:
        return
    ref, source, universe = stored
    if ref.base.start != cfg.start or ref.base.end != cfg.end or universe != panel.tradable_symbols:
        st.warning("아래 결과는 현재 유니버스·기간과 다른 설정으로 실행한 것입니다.")
    st.caption(f"기간 {ref.base.start} → {ref.base.end} · 거래 대상 {len(universe)}종목 · {ref.seconds:.0f}초")
    show_refinement(ref.table(), ref.all_cells(), ref.entry_key, ref.exit_key, ref.gates.to_dict(), ref.ma200_calmar, "live")
    with st.form("save_refine", border=False):
        c1, c2 = st.columns([3, 1])
        name = c1.text_input("결과 이름", value="", placeholder="예: 수익률×ATR 확인 조건 · 장기 ETF")
        if c2.form_submit_button("💾 확인 조건 시험 저장", width="stretch"):
            path = store.save_refinement(ref, name or f"확인 조건 · {family_label(ref.entry_key, ref.exit_key)}", source, universe)
            st.success(f"저장했습니다: lab_results/{path.name}")


# =========================================================================== common add-on test view

def show_addon_result(table: pd.DataFrame, cells: pd.DataFrame, label: str, gates: dict, ma200: float, key: str) -> None:
    st.caption(f"추가한 조건: **{label}** · Gate: CAGR ≥ {gates['min_cagr_ratio']:.2f}×SPY · "
               f"|MDD| ≤ {gates['max_mdd_ratio']:.2f}×|SPY MDD| · 완료 거래 ≥ {gates['min_trades']} · "
               f"SPY 200일선 칼마 = {ma200:.2f}")
    n = len(table)
    counts = {k: int((table["verdict"] == k).sum()) for k in ("improve", "mixed", "same", "worse")}
    cols = st.columns(4)
    for c, k in zip(cols, counts):
        c.metric(VERDICT_SHORT[k], f"{counts[k]} / {n}", help=VERDICT_TEXT[k])
    if counts["improve"] * 2 > n:
        st.success(f"시험한 계열의 과반({counts['improve']}/{n})이 개선됐습니다. 이 유니버스·기간에서는 **대체로 도움이 되는 조건**입니다. "
                   "다른 유니버스·기간에서도 과반이 개선돼야 일반적으로 도움이 된다고 판단합니다.")
    elif counts["worse"] * 2 > n:
        st.error(f"시험한 계열의 과반({counts['worse']}/{n})이 악화됐습니다. 이 유니버스·기간에서는 **대체로 해로운 조건**입니다.")
    else:
        st.info("개선도 악화도 과반이 아닙니다. 이 데이터만으로는 **판단 보류**입니다.")
    num = st.column_config.NumberColumn
    shown = table.drop(columns=["entry", "exit"]).assign(verdict=table["verdict"].map(VERDICT_SHORT))
    config = {"family": st.column_config.TextColumn("계열 (진입 × 청산)", width="large"),
              "verdict": st.column_config.TextColumn("판정", help="\n\n".join(f"{VERDICT_SHORT[k]}: {v}" for k, v in VERDICT_TEXT.items()))}
    for k in ORDER:
        config[f"{k}_plain"] = num(f"{SUMMARY_TEXT[k][0]} · 원래", format=FAMILY_FMT[k], help=SUMMARY_TEXT[k][1])
        config[f"{k}_added"] = num(f"{SUMMARY_TEXT[k][0]} · 추가 후", format=FAMILY_FMT[k], help=SUMMARY_TEXT[k][1])
    st.dataframe(shown, hide_index=True, width="stretch", column_config=config)

    st.markdown("#### 격자 자세히 보기")
    labels = {f"{r['entry']}|{r['exit']}": f"{VERDICT_SHORT[r['verdict']]} · {r['family']}" for _, r in table.iterrows()}
    c1, c2 = st.columns([3, 1])
    pick = c1.selectbox("계열 선택", list(labels), format_func=labels.get, key=f"addon_pick_{key}")
    variant = c2.radio("격자", ["added", "plain"], key=f"addon_variant_{key}", horizontal=True,
                       format_func={"added": "추가 후", "plain": "원래"}.get)
    e, x = pick.split("|")
    sub = cells[(cells["entry"] == e) & (cells["exit"] == x) & (cells["variant"] == variant)].reset_index(drop=True)
    passed = sub[sub["pass"]]
    show_grid_result(sub, e, x, {"total": len(sub), "passed": len(passed), "regions": passed["region"].nunique(),
                                 "largest_region": int(sub["region_size"].max())}, gates, f"addon_{key}_{variant}")


def reusable_family_result(panel: Panel, cfg: StrategyConfig):
    """The stored plain family comparison when it was run on the current universe, settings and menu."""
    stored = st.session_state.get("family_result")
    if stored is None:
        return None
    fc, _, universe = stored
    same = (replace(fc.base, entries={}, exits={}) == replace(cfg, entries={}, exits={})
            and universe == panel.tradable_symbols and fc.gates == gates_now()
            and not fc.extra_entries and not fc.extra_exits and list(fc.grids) == family_pairs())
    return fc if same else None


def addon_tab(panel: Panel, cfg: StrategyConfig) -> None:
    st.caption("조건 **하나**를 기본값으로 **모든 계열**에 똑같이 더하고, 계열마다 원래 격자와 네 기준으로 비교합니다. "
               "‘이 조건이 대체로 도움이 되는가?’를 보는 시험입니다(한 계열에 여러 조건을 시험하는 ‘확인 조건 시험’의 반대 방향). "
               "그 조건 자체로 만든 계열은 같은 블록을 두 번 넣을 수 없어 빠집니다. 위 ①·④ 설정을 쓰고 ②·③은 쓰지 않습니다.")
    choices = addon_choices()
    default = choices.index(("exit", "stop_loss_pct"))
    i = st.selectbox("모든 계열에 더할 조건", range(len(choices)), index=default, key="addon_choice",
                     format_func=lambda k: addon_label(*choices[k]),
                     help="진입 조건은 AND(필터), 청산 조건은 OR(추가 청산 규칙)로 더해집니다. 값은 블록 기본값으로 고정합니다.")
    kind, key = choices[i]
    b = (ENTRY_BLOCKS if kind == "entry" else EXIT_BLOCKS)[key]
    st.caption(f"`{b.rule}` · {b.help}")
    extra = {key: None}
    ee, xx = (extra, {}) if kind == "entry" else ({}, extra)
    plain = reusable_family_result(panel, cfg)
    n = cell_count(ee, xx) + (0 if plain is not None else cell_count())
    st.caption(f"계열 {len(family_pairs(ee, xx))}개 · 백테스트 {n}회"
               + (" (‘전략 계열 비교’ 결과를 재사용)" if plain is not None else " (원래 계열 포함)")
               + f" · 예상 약 {max(1, round(estimate_seconds(panel, cfg, n) / 60))}분 · 모두 시험 횟수에 기록됩니다.")
    if st.button("▶ 공통 조건 시험 실행", type="primary"):
        bar = st.progress(0.0, text="시험 중…")
        collected = []
        try:
            at = run_addon_test(panel, cfg, gates_now(), kind, key, plain=plain,
                                progress=lambda f: bar.progress(min(f, 1.0), text=f"시험 중… {f:.0%}"),
                                on_result=lambda c, r: collected.append(("addon", c, panel.tradable_symbols, r.equity)))
            st.session_state["addon_result"] = (at, panel.source, panel.tradable_symbols)
            rlog.log_trials(collected)
        except ValueError as exc:
            st.error(str(exc))
        bar.empty()
    stored = st.session_state.get("addon_result")
    if stored is None:
        return
    at, source, universe = stored
    base = at.plain.base
    if base.start != cfg.start or base.end != cfg.end or universe != panel.tradable_symbols:
        st.warning("아래 결과는 현재 유니버스·기간과 다른 설정으로 실행한 것입니다.")
    st.caption(f"기간 {base.start} → {base.end} · 거래 대상 {len(universe)}종목")
    show_addon_result(at.table(), at.all_cells(), at.label, at.plain.gates.to_dict(), at.plain.ma200_calmar, "live")
    with st.form("save_addon", border=False):
        c1, c2 = st.columns([3, 1])
        name = c1.text_input("결과 이름", value="", placeholder="예: 공통 손절 10% · 장기 ETF")
        if c2.form_submit_button("💾 공통 조건 시험 저장", width="stretch"):
            path = store.save_addon(at, name or f"공통 조건 · {at.label}", source, universe)
            st.success(f"저장했습니다: lab_results/{path.name}")


# =========================================================================== pages

def research_page() -> None:
    st.title("추세추종 전략 연구실")
    st.caption("전략 정의 → 백테스트 결과 → 강건성 → 진단 순서로 봅니다. 모든 지표의 정의는 이름 옆 ⓘ 또는 ‘용어 설명’ 페이지에 있습니다.")
    panel = universe_section()
    if panel is None:
        return
    st.divider()
    cfg = strategy_section()
    if cfg is None:
        return
    with st.expander("실행 규칙 자세히 보기"):
        st.markdown(RULES_KO)
    st.divider()

    tab_bt, tab_grid, tab_fam, tab_ref, tab_add = st.tabs(["📈 백테스트", "🧭 강건성 격자", "🧩 전략 계열 비교", "🧪 확인 조건 시험",
                                                          "🧷 공통 조건 시험"])
    with tab_bt:
        if st.button("▶ 백테스트 실행", type="primary") or st.session_state.pop("auto_run", False):
            try:
                with st.spinner("백테스트 중…"):
                    run_and_store(panel, cfg)
            except ValueError as exc:
                st.error(str(exc))
        bt = st.session_state.get("bt")
        if bt is not None:
            if bt["res"].config != cfg or bt["panel"].tradable_symbols != panel.tradable_symbols:
                st.warning("아래 결과는 현재 설정과 다릅니다. 다시 실행하면 갱신됩니다.")
            show_backtest(bt)
    with tab_grid:
        grid_tab(panel, cfg)
    with tab_fam:
        family_tab(panel, cfg)
    with tab_ref:
        refine_tab(panel, cfg)
    with tab_add:
        addon_tab(panel, cfg)


def holdout_page() -> None:
    st.title("🔒 최종 검증 (보류 구간)")
    hold = rlog.load_holdout()
    st.markdown(
        "연구에 한 번도 쓰지 않은 기간에서 전략을 **딱 한 번** 확인하는 곳입니다.  \n"
        "1. 전략 연구 페이지에서 전략을 다듬고 **결과를 저장**합니다 (보류 구간 이전 데이터만 사용).  \n"
        "2. 여기서 저장한 전략을 보류 구간에 **그대로** 적용합니다. 설정은 바꿀 수 없습니다.  \n"
        "3. 결과를 보고 전략을 고쳐 다시 확인하면, 보류 구간도 ‘맞춰진’ 데이터가 됩니다. 평가 기록은 모두 남습니다."
    )
    if not hold["enabled"]:
        st.warning("보류 구간이 꺼져 있습니다. ‘전략 연구’ 페이지의 ‘보류 구간 설정 바꾸기’에서 켜세요.")
    c1, c2 = st.columns(2)
    c1.metric("보류 구간 시작", hold["start"] if hold["enabled"] else "꺼짐")
    c2.metric("지금까지 평가 횟수", len(hold["evaluations"]),
              help="많이 확인할수록 보류 구간의 검증력이 떨어집니다. 후보 몇 개만 골라 한 번씩 확인하세요.")

    saved = [i for i in store.list_results() if i["kind"] == "backtest"]
    if not saved:
        st.info("저장된 백테스트가 없습니다. ‘전략 연구’에서 결과를 저장하세요.")
    else:
        idx = st.selectbox("평가할 전략 (저장된 결과)", range(len(saved)),
                           format_func=lambda i: f"{saved[i]['name']} · 연구 기간 {' → '.join(saved[i]['period'])}")
        meta = saved[idx]
        st.caption(meta["description"])
        if "data_kind" not in meta:
            st.warning("이 결과는 이전 버전에서 저장되어 유니버스 정보가 없습니다. 같은 전략을 다시 실행해 저장하세요.")
        else:
            cfg0 = StrategyConfig.from_dict(meta["config"])
            sf = cfg0.strategy_fingerprint()
            prior = [e for e in hold["evaluations"]  # older records have no strategy fingerprint: match the rule text
                     if e.get("strategy_fingerprint", sf if e["description"] == cfg0.describe() else None) == sf]
            if prior:
                st.warning(f"이 전략은 이미 보류 구간에서 {len(prior)}번 평가했습니다 (마지막 {prior[-1]['time']}). "
                           "다시 평가해도 새로운 정보는 없습니다.")
            if meta["period"][1] >= hold["start"]:
                st.warning("이 결과의 연구 기간이 현재 보류 구간과 겹칩니다. 보류 구간 설정이 바뀐 뒤 저장된 결과입니다.")
            ok = st.checkbox("평가 결과를 본 뒤 이 전략을 고쳐서 다시 평가하지 않겠습니다.", key="holdout_confirm")
            if st.button("🔒 보류 구간에서 평가", type="primary", disabled=not (ok and hold["enabled"])):
                with st.spinner("보류 구간 백테스트 중…"):
                    try:
                        panel = rebuild_panel(meta)
                        cfg = StrategyConfig.from_dict({**meta["config"], "start": hold["start"],
                                                        "end": str(panel.last_date.date())})
                        res = run_backtest(panel, cfg)
                        m = all_metrics(res)
                        base = baselines(panel, cfg, res.equity.index)
                        rlog.record_holdout_evaluation(meta["name"], cfg, m, (cfg.start, cfg.end))
                        st.session_state["holdout_view"] = (meta["path"], res, m, base)
                        st.rerun()
                    except Exception as exc:  # data rebuild / empty period
                        st.error(f"평가하지 못했습니다: {exc}")
            view = st.session_state.get("holdout_view")
            if view is not None and view[0] == meta["path"]:
                _, res, m, base = view
                s, e = res.period
                st.info(f"**보류 구간 {s.date()} → {e.date()}** ({len(res.equity):,} 거래일)")
                keys = ["cagr", "mdd", "sharpe", "calmar", "cagr_ratio", "mdd_ratio", "n_trades"]
                ins = meta["metrics"]
                st.dataframe(pd.DataFrame([{
                    "지표": DEFINITIONS[k].label, "연구 기간": fmt_value(k, ins.get(k)), "보류 구간": fmt_value(k, m.get(k)),
                } for k in keys] + [
                    {"지표": "SPY CAGR", "연구 기간": fmt_value("cagr", ins.get("spy_cagr")), "보류 구간": fmt_value("cagr", m["spy_cagr"])},
                    {"지표": "SPY MDD", "연구 기간": fmt_value("mdd", ins.get("spy_mdd")), "보류 구간": fmt_value("mdd", m["spy_mdd"])},
                ]), hide_index=True, width="stretch")
                # holdout verdict: return and drawdown gates only; the trade count of a short window says how
                # reliable the result is, not how good it is
                g = gates_now()
                failed = check_gates(m, Gates(g.min_cagr_ratio, g.max_mdd_ratio, 0))
                st.markdown(("✅ 보류 구간에서도 수익·낙폭 Gate 통과" if not failed else "❌ 보류 구간 Gate 미달: " + ", ".join(failed))
                            + f"  \n<small>완료 거래 {int(m['n_trades'])}건 · 보류 구간이 {len(res.equity) / 252:.1f}년으로 짧아 "
                              "우연의 영향이 큽니다. 통과/미달 하나로 결론 내리지 말고 연구 기간과의 차이를 보세요.</small>",
                            unsafe_allow_html=True)
                st.plotly_chart(equity_chart(res, base, False), width="stretch")

    if hold["evaluations"]:
        st.markdown("#### 평가 기록")
        st.dataframe(pd.DataFrame([{
            "시각": e["time"], "이름": e["name"], "기간": " → ".join(e["period"]),
            "CAGR": e["metrics"].get("cagr"), "MDD": e["metrics"].get("mdd"), "SPY CAGR": e["metrics"].get("spy_cagr"),
            "전략": e["description"],
        } for e in reversed(hold["evaluations"])]), hide_index=True, width="stretch", column_config={
            "CAGR": st.column_config.NumberColumn(format="percent"), "MDD": st.column_config.NumberColumn(format="percent"),
            "SPY CAGR": st.column_config.NumberColumn(format="percent"),
        })


def saved_page() -> None:
    st.title("저장된 결과")
    items = store.list_results()
    if not items:
        st.info("아직 저장된 결과가 없습니다. ‘전략 연구’에서 결과를 저장하세요.")
        return
    bts = [i for i in items if i["kind"] == "backtest"]
    grids = [i for i in items if i["kind"] == "grid"]
    fams = [i for i in items if i["kind"] == "family"]
    refs = [i for i in items if i["kind"] == "refine"]
    adds = [i for i in items if i["kind"] == "addon"]
    t1, t2, t3, t4, t5 = st.tabs([f"백테스트 ({len(bts)})", f"강건성 격자 ({len(grids)})", f"계열 비교 ({len(fams)})",
                                  f"확인 조건 시험 ({len(refs)})", f"공통 조건 시험 ({len(adds)})"])
    with t1:
        if not bts:
            st.info("저장된 백테스트가 없습니다.")
        else:
            keys = ["cagr", "mdd", "sharpe", "calmar", "cagr_ratio", "mdd_ratio", "n_trades"]
            rows = [{"선택": i < 3, "이름": b["name"], "저장 시각": b["saved_at"], "기간": " → ".join(b["period"]),
                     "전략": b["description"], **{DEFINITIONS[k].label: b["metrics"].get(k) for k in keys}}
                    for i, b in enumerate(bts)]
            df = pd.DataFrame(rows)
            edited = st.data_editor(df, hide_index=True, width="stretch", disabled=[c for c in df.columns if c != "선택"],
                                    column_config={DEFINITIONS["cagr"].label: st.column_config.NumberColumn(format="percent"),
                                                   DEFINITIONS["mdd"].label: st.column_config.NumberColumn(format="percent")})
            chosen = [bts[i] for i in np.flatnonzero(edited["선택"].to_numpy())]
            if chosen:
                fig = go.Figure()
                palette = [BLUE, ORANGE, AQUA, "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
                for i, b in enumerate(chosen[:8]):
                    eq = store.load_equity(b["path"])
                    fig.add_scatter(x=eq.index, y=eq["strategy"], name=b["name"], line=dict(color=palette[i], width=2))
                eq0 = store.load_equity(chosen[0]["path"])
                fig.add_scatter(x=eq0.index, y=eq0["SPY"], name="SPY (첫 결과 기간)", line=dict(color=GRAY, width=1.5, dash="dot"))
                fig.update_layout(height=380, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                                  title="자산 곡선 비교 (최대 8개)", legend=dict(orientation="h", y=-0.15))
                st.plotly_chart(fig, width="stretch")
                st.caption("서로 기간이 다른 결과는 시작점이 달라 직접 비교에 주의하세요.")
    with t2:
        if not grids:
            st.info("저장된 격자가 없습니다.")
        else:
            idx = st.selectbox("격자 선택", range(len(grids)),
                               format_func=lambda i: f"{grids[i]['saved_at']} · {grids[i]['name']} · {grids[i]['description']}")
            gm = grids[idx]
            st.caption(f"기간 {' → '.join(gm['period'])} · 데이터: {gm['data_source']}")
            show_grid_result(store.load_cells(gm["path"]), gm["x_key"], gm["y_key"], gm["summary"], gm["gates"], "saved")
    with t3:
        if not fams:
            st.info("저장된 계열 비교가 없습니다.")
        else:
            idx = st.selectbox("계열 비교 선택", range(len(fams)), key="saved_fam",
                               format_func=lambda i: f"{fams[i]['saved_at']} · {fams[i]['name']}")
            fm = fams[idx]
            st.caption(f"기간 {' – '.join(fm['period'])} · 데이터: {fm['data_source']} · 거래 대상 {len(fm['universe'])}종목 · "
                       f"메뉴 {fm['menu_version']}")
            show_family_result(store.load_family_summary(fm["path"]), store.load_cells(fm["path"]), fm["gates"],
                               fm["ma200_calmar"], "saved")
    with t4:
        if not refs:
            st.info("저장된 확인 조건 시험이 없습니다.")
        else:
            idx = st.selectbox("확인 조건 시험 선택", range(len(refs)), key="saved_ref",
                               format_func=lambda i: f"{refs[i]['saved_at']} · {refs[i]['name']}")
            rm = refs[idx]
            st.caption(f"기간 {' – '.join(rm['period'])} · 데이터: {rm['data_source']} · 거래 대상 {len(rm['universe'])}종목")
            show_refinement(store.load_refine_table(rm["path"]), store.load_cells(rm["path"]), rm["entry_key"],
                            rm["exit_key"], rm["gates"], rm["ma200_calmar"], "saved")
    with t5:
        if not adds:
            st.info("저장된 공통 조건 시험이 없습니다.")
        else:
            idx = st.selectbox("공통 조건 시험 선택", range(len(adds)), key="saved_addon",
                               format_func=lambda i: f"{adds[i]['saved_at']} · {adds[i]['name']}")
            am = adds[idx]
            st.caption(f"기간 {' – '.join(am['period'])} · 데이터: {am['data_source']} · 거래 대상 {len(am['universe'])}종목 · "
                       f"메뉴 {am['menu_version']}")
            show_addon_result(store.load_addon_table(am["path"]), store.load_cells(am["path"]), am["label"], am["gates"],
                              am["ma200_calmar"], "saved")


def glossary_page() -> None:
    st.title("용어 설명")
    st.markdown("화면에 나오는 모든 지표·조건의 정확한 정의입니다. 계산 코드(`lab/metrics.py`, `lab/grid.py`, `lab/blocks.py`, "
                "`lab/universe.py`)와 같은 출처에서 표시됩니다.")
    direction = {"high": "높을수록 좋음", "low": "낮을수록 좋음", "": "방향 없음"}
    groups: dict[str, list] = {}
    for d in DEFINITIONS.values():
        groups.setdefault(d.group, []).append(d)
    titles = {"포트폴리오": "포트폴리오 지표 (주 기준)", "SPY 대비": "SPY 대비 지표 (주 기준)", "거래": "거래 단위 지표 (보조)",
              "과최적화": "과최적화 통제"}
    for group, defs in groups.items():
        st.subheader(titles[group])
        for d in defs:
            st.markdown(BR.join([f"**{d.label}** · <small>{direction[d.better]}</small>", d.short, f"`{d.formula}`"]),
                        unsafe_allow_html=True)
    st.subheader("기준선 (비교 대상)")
    for k, label in BASELINE_LABELS.items():
        st.markdown(BR.join([f"**{label}**", BASELINE_HELP[k]]))
    st.subheader("강건성 판정")
    for name, rule, why in GATE_TEXT.values():
        st.markdown(BR.join([f"**Gate · {name}**", f"`{rule}`", f"<small>기본값 근거: {why}</small>"]), unsafe_allow_html=True)
    for name, rule in ROBUST_TEXT.values():
        st.markdown(BR.join([f"**{name}**", rule]))
    st.markdown(BR.join(["**격자 후보 정렬**", "연결 영역 크기 → 이웃 생존율 → LOYO 통과 비율 → 칼마 비율 순의 사전식 정렬. "
                         "앞 기준이 같을 때만 다음 기준을 봅니다. 가중 합산 점수는 쓰지 않습니다."]))
    st.markdown("**단일 전략 강건성 점검 항목**")
    for label, rule in CHECK_TEXT.values():
        st.markdown(BR.join([f"· **{label}**", rule]))
    st.subheader("전략 계열 비교")
    st.markdown(f"미리 정한 메뉴(진입 {len(ENTRY_MENU)}종 × 청산 {len(EXIT_MENU)}종)의 각 계열을 6×6 격자로 돌려, 격자 전체의 성적으로 비교합니다. "
                f"메뉴: {MENU_VERSION}. 순위는 Gate 통과 비율 → 가장 큰 연결 영역 → SPY 200일선 이긴 비율 → 중간 칼마 순의 사전식 정렬입니다.")
    for label, rule in SUMMARY_TEXT.values():
        st.markdown(BR.join([f"· **{label}**", rule]))
    st.subheader("확인 조건 시험")
    st.markdown("고른 계열에 확인 조건을 하나씩(각 블록의 기본값으로) 더해 같은 격자를 다시 돌리고, 계열 비교의 네 기준으로 원래 계열과 비교합니다.")
    for k, v in VERDICT_TEXT.items():
        st.markdown(f"· {v}")
    st.markdown("채택 원칙: 여러 유니버스·기간에서 모두 ‘개선’일 때만 조건을 더합니다. 복잡성은 성과로 증명될 때만 남깁니다.")
    st.subheader("공통 조건 시험")
    st.markdown("조건 하나를 기본값으로 모든 계열에 똑같이 더하고, 계열마다 원래 격자와 같은 네 기준으로 비교합니다(판정은 위와 같음). "
                "‘이 필터·청산 규칙이 대체로 도움이 되는가?’를 봅니다. 그 조건 자체로 만든 계열은 빠집니다.")
    st.markdown("판단 원칙: 모든 유니버스·기간에서 시험한 계열의 **과반이 개선**되면 ‘대체로 도움’, 과반이 악화되면 ‘대체로 해로움’, "
                "그 밖에는 ‘판단 보류’입니다.")
    st.subheader("보류 구간 (최종 검증)")
    st.markdown("정해 둔 날짜 이후의 데이터는 연구(백테스트·격자·강건성 점검)에 쓰지 않고 남겨 둡니다. 연구가 끝난 전략을 그 기간에 "
                "한 번 적용해 ‘처음 보는 데이터’에서도 성과가 유지되는지 봅니다. 날짜 변경과 평가는 모두 기록됩니다. "
                "기본값 2024-01-01: 스냅샷 데이터에서 연구 약 6.4년, 보류 약 2.6년이 되도록 고른 날짜입니다.")
    st.subheader("유니버스 자산군과 필터")
    for k, v in CLASS_HELP.items():
        st.markdown(BR.join([f"**{k}**", v]))
    st.markdown(BR.join(["**중복 ETF 정리**", f"일간 수익률 상관계수 ≥ {DEDUP_CORR}이면 같은 상품으로 보고 운용 규모가 큰 쪽만 남깁니다. "
                         "0.98은 같은 지수를 추종하는 ETF끼리(보통 0.99 이상)는 묶이고, 비슷하지만 다른 상품"
                         "(예: QQQ와 XLK, 약 0.95)은 남는 수준입니다."]))
    st.subheader("전략 블록")
    for k, b in ALL_BLOCKS.items():
        kind = "진입 (AND)" if k in ENTRY_BLOCKS else "청산 (OR)"
        p = b.param
        st.markdown(BR.join([
            f"**{b.label}** · <small>{kind} · {b.family}</small>", f"`{b.rule}`", b.help,
            "<small>파라미터 없음 (고정 규칙)</small>" if p is None else
            f"<small>{p.label}: 기본 {p.default:g}, 허용 범위 {p.minimum:g} – {p.maximum:g}</small>",
        ]), unsafe_allow_html=True)
    st.subheader("실행 규칙")
    st.markdown(RULES_KO)


def sidebar_status() -> None:
    n, _ = rlog.trial_stats()
    n_eff = rlog.effective_trials()[0]
    hold = rlog.load_holdout()
    with st.sidebar:
        st.caption(BR.join([
            f"🧪 지금까지 시험한 전략: **{n:,}개** (유효 ≈ {n_eff:,.0f})",
            "<small>같은 설정의 재실행은 세지 않음 · 서로 비슷한 시험은 유효 개수로 보정해 DSR에 반영</small>",
            f"🔒 보류 구간: **{hold['start'] + ' 이후' if hold['enabled'] else '꺼짐'}** · 평가 {len(hold['evaluations'])}회",
        ]), unsafe_allow_html=True)


PERSIST_PREFIXES = ("entry_", "exit_", "max_positions", "cost_bps", "min_price", "min_dv", "liq_days", "cash_yield",
                    "uni_", "yf_tickers", "start", "end", "grid_", "gx_", "gy_", "gate_", "sizing")
for _k in list(st.session_state.keys()):
    if str(_k).startswith(PERSIST_PREFIXES):
        st.session_state[_k] = st.session_state[_k]  # keeps settings when visiting another page

nav = st.navigation([
    st.Page(research_page, title="전략 연구", icon="📈", default=True),
    st.Page(holdout_page, title="최종 검증 (보류 구간)", icon="🔒"),
    st.Page(saved_page, title="저장된 결과", icon="💾"),
    st.Page(glossary_page, title="용어 설명", icon="📖"),
])
nav.run()
sidebar_status()  # after the page so the trial count includes runs made just now
