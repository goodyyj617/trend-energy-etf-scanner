"""추세추종 전략 연구실 — Streamlit app.

Run:  .venv\\Scripts\\python.exe -m streamlit run lab/app.py
"""
from __future__ import annotations

import sys
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import store  # noqa: E402
from lab.blocks import ENTRY_BLOCKS, EXIT_BLOCKS, ALL_BLOCKS  # noqa: E402
from lab.data import BENCHMARK, Panel, download_panel, load_snapshot, snapshot_available  # noqa: E402
from lab.engine import StrategyConfig, run_backtest  # noqa: E402
from lab.engine import RULES_KO  # noqa: E402
from lab.grid import GATE_TEXT, MAX_AXIS_VALUES, ROBUST_TEXT, Gates, axis_values, check_gates, run_grid  # noqa: E402
from lab.metrics import DEFINITIONS, all_metrics, annual_table, drawdown, fmt_value  # noqa: E402

st.set_page_config(page_title="추세추종 전략 연구실", page_icon="📈", layout="wide")

BLUE, ORANGE, GRAY = "#2a78d6", "#eb6834", "#8a8984"
SEQ_BLUE = [[0, "#cde2fb"], [0.5, "#5598e7"], [1, "#0d366b"]]
SECONDS_PER_BACKTEST = 0.7  # measured on the 466-ETF snapshot; used only for the time estimate
BR = "  " + chr(10)  # markdown hard line break


def tip(key: str) -> str:
    d = DEFINITIONS[key]
    return f"{d.short}\n\n계산: {d.formula}"


def param_label(key: str) -> str:
    if key == "max_positions":
        return "최대 보유 종목 수"
    b = ALL_BLOCKS[key]
    return f"{b.label} · {b.param.label}"


# =========================================================================== data

@st.cache_resource(show_spinner="ETF 스냅샷을 불러오는 중… (처음 한 번만 몇 초 걸립니다)")
def snapshot() -> Panel:
    return load_snapshot()


@st.cache_resource(show_spinner=False, max_entries=8)
def snapshot_subset(symbols: tuple[str, ...]) -> Panel:
    return snapshot().subset(list(symbols))


def data_section() -> Panel | None:
    st.subheader("① 데이터와 기간")
    source = st.radio(
        "데이터 출처", ["ETF 스냅샷 (오프라인)", "티커 직접 입력 (yfinance)"], horizontal=True,
        help="ETF 스냅샷: 저장소에 고정 저장된 466개 ETF의 2016-08-01 ~ 2026-07-30 일봉. 인터넷 없이 바로 쓸 수 있습니다.\n\n"
             "티커 직접 입력: 원하는 ETF·개별 종목을 yfinance에서 내려받아 lab_data/에 캐시합니다.",
    )
    panel: Panel | None = None
    if source.startswith("ETF"):
        if not snapshot_available():
            st.error("저장소에서 ETF 스냅샷 폴더를 찾지 못했습니다.")
            return None
        full = snapshot()
        groups = sorted(g for g in full.info["asset_group"].dropna().unique())
        c1, c2 = st.columns([3, 2])
        chosen = c1.multiselect(
            "자산군 (거래 대상 유니버스)", groups, key=init("groups", [g for g in ["US Equity", "Sector", "Industry / Theme"] if g in groups]),
            help="선택한 자산군의 ETF만 매수 후보가 됩니다. SPY는 비교 기준으로 항상 데이터에 포함됩니다.",
        )
        extra = c2.text_input("추가로 포함/제외할 티커", key=init("ticker_adjust", ""), placeholder="예: GLD, TLT, -XLU",
                              help="쉼표로 구분. 앞에 '-'를 붙이면 제외합니다.")
        syms = set(full.info.loc[full.info["asset_group"].isin(chosen), "symbol"])
        for tok in [t.strip().upper() for t in extra.split(",") if t.strip()]:
            if tok.startswith("-"):
                syms.discard(tok[1:])
            elif tok in full.symbols:
                syms.add(tok)
        if not syms:
            st.warning("거래 대상 종목이 없습니다. 자산군이나 티커를 선택하세요.")
            return None
        panel = snapshot_subset(tuple(sorted(syms)))
    else:
        tickers = st.text_area("티커 (쉼표 또는 줄바꿈으로 구분)", key=init("yf_tickers", "SPY, QQQ, IWM, EFA, EEM, GLD, TLT, XLE"),
                               help="ETF와 개별 종목 모두 가능합니다. 비교 기준 SPY는 자동으로 추가됩니다.")
        refresh = st.checkbox("캐시 무시하고 새로 받기", value=False)
        if st.button("가격 데이터 불러오기"):
            syms = [t for t in tickers.replace("\n", ",").split(",") if t.strip()]
            with st.spinner(f"{len(syms)}개 티커를 내려받는 중…"):
                try:
                    p, failed = download_panel(syms, str(date.today()), refresh=refresh)
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

    # the comparison with SPY needs SPY prices, so the selectable range starts where SPY data starts
    first = max(panel.first_date, panel.close[BENCHMARK].first_valid_index()).date()
    last = panel.last_date.date()
    default_start = max(first, date(first.year + 1, first.month, 1))
    c1, c2, c3 = st.columns([1, 1, 2])
    if st.session_state.get("_data_range") != (first, last):  # new data source/range: reset dates to its defaults
        st.session_state["_data_range"] = (first, last)
        st.session_state.pop("start", None)
        st.session_state.pop("end", None)
    start = c1.date_input("시작일", min_value=first, max_value=last, key=init("start", default_start))
    end = c2.date_input("종료일", min_value=first, max_value=last, key=init("end", last))
    c3.markdown(
        f"**거래 대상 {len(panel.tradable_symbols)}종목** · 데이터 범위 {first} ~ {last}  \n"
        f"<small>지표(이동평균 등)는 시작일 이전 데이터로 미리 계산합니다. 기본 시작일은 데이터 시작 1년 뒤입니다.</small>",
        unsafe_allow_html=True,
    )
    if start >= end:
        st.error("시작일은 종료일보다 앞이어야 합니다.")
        return None
    st.session_state["_period"] = (start, end)
    return panel


# =========================================================================== strategy builder

def init(key: str, value) -> str:
    """Set a widget's starting value once; afterwards the session state owns it."""
    if key not in st.session_state:
        st.session_state[key] = value
    return key


def block_inputs(blocks: dict, prefix: str, defaults: dict) -> dict[str, float]:
    chosen: dict[str, float] = {}
    for key, b in blocks.items():
        c1, c2, c3 = st.columns([2.2, 1.2, 3])
        on = c1.checkbox(b.label, key=init(f"{prefix}_on_{key}", key in defaults), help=b.help)
        p = b.param
        cast = int if p.is_int else float
        val = c2.number_input(p.label, min_value=cast(p.minimum), max_value=cast(p.maximum), step=cast(p.step),
                              key=init(f"{prefix}_val_{key}", cast(defaults.get(key, p.default))),
                              disabled=not on, help=p.help, label_visibility="collapsed")
        c3.caption(f"{p.label}  ·  {b.rule}")
        if on:
            chosen[key] = float(val)
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

    st.subheader("④ 포트폴리오와 비용")
    c1, c2, c3, c4 = st.columns(4)
    k = c1.number_input("최대 보유 종목 수", 1, 50, key=init("max_positions", 10),
                        help="동시에 보유할 수 있는 최대 종목 수. 새 종목에는 (총자산 ÷ 이 값)만큼, 현금이 부족하면 남은 현금만큼 배정합니다.")
    cost = c2.number_input("편도 거래 비용 (bp)", 0.0, 100.0, step=1.0, key=init("cost_bps", 10.0),
                           help="매수·매도 각각에 부과. 수수료와 슬리피지를 합친 값. 10bp = 0.10%.")
    min_price = c3.number_input("최소 주가 ($)", 0.0, 100.0, step=1.0, key=init("min_price", 5.0),
                                help="신호일 종가가 이보다 낮으면 매수하지 않습니다.")
    min_dv = c4.number_input("최소 거래대금 ($백만, 20일 평균)", 0.0, 1000.0, step=1.0, key=init("min_dv", 5.0),
                             help="신호일까지 20거래일 평균 (종가 × 거래량)이 이보다 작으면 매수하지 않습니다.")
    if not entries or not exits:
        st.warning("진입 조건과 청산 조건을 각각 하나 이상 선택하세요.")
        return None
    start, end = st.session_state["_period"]
    return StrategyConfig(entries=entries, exits=exits, start=str(start), end=str(end), max_positions=int(k),
                          cost_bps=float(cost), min_price=float(min_price), min_dollar_volume=float(min_dv) * 1e6)


# =========================================================================== backtest view

def period_banner(result, panel: Panel) -> None:
    s, e = result.period
    st.info(f"**백테스트 기간 {s.date()} → {e.date()}** ({len(result.equity):,} 거래일) · "
            f"거래 대상 {result.universe_size}종목 · 데이터: {panel.source} · 비교 기준: {BENCHMARK} (같은 날짜)")


def metric_table(m: dict, keys: list[str], spy: bool = False) -> None:
    """Metric | strategy | (SPY) | meaning — readable at any screen width."""
    rows = []
    for key in keys:
        d = DEFINITIONS[key]
        row = {"지표": d.label, "전략": fmt_value(key, m[key])}
        if spy:
            row["SPY"] = fmt_value(key, m[f"spy_{key}"]) if f"spy_{key}" in m else "–"
        row["뜻 (계산식)"] = f"{d.short} ({d.formula})"
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True,
                 column_config={"뜻 (계산식)": st.column_config.TextColumn(width="large")})


def gate_check(m: dict) -> None:
    """Show whether this single backtest meets the gates currently set in the grid tab."""
    g = Gates(st.session_state.get("gate_cagr", 0.80), st.session_state.get("gate_mdd", 0.75),
              int(st.session_state.get("gate_trades", 30)))
    failed = check_gates(m, g)
    items = [
        (GATE_TEXT["min_cagr_ratio"][0], f"CAGR {fmt_value('cagr', m['cagr'])} vs 기준 {g.min_cagr_ratio:.2f}×SPY {fmt_value('cagr', m['spy_cagr'])}"),
        (GATE_TEXT["max_mdd_ratio"][0], f"|MDD| {fmt_value('mdd', abs(m['mdd']))} vs 기준 {g.max_mdd_ratio:.2f}×|SPY MDD| {fmt_value('mdd', abs(m['spy_mdd']))}"),
        (GATE_TEXT["min_trades"][0], f"완료 거래 {int(m['n_trades'])} vs 기준 {g.min_trades}"),
    ]
    lines = [f"{'❌' if name in failed else '✅'} **{name}** — {detail}" for name, detail in items]
    st.markdown("  \n".join(lines))
    st.caption("기준값은 ‘🧭 강건성 격자’ 탭에서 바꿀 수 있습니다. 단일 결과의 통과는 강건성의 증거가 아닙니다 — 주변 파라미터도 통과하는지 격자로 확인하세요.")


def equity_chart(eq: pd.Series, bench: pd.Series, log: bool) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=eq.index, y=eq.values, name="전략", line=dict(color=BLUE, width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>전략 %{y:.3f}<extra></extra>")
    fig.add_scatter(x=bench.index, y=bench.values, name=BENCHMARK, line=dict(color=ORANGE, width=2),
                    hovertemplate="%{x|%Y-%m-%d}<br>SPY %{y:.3f}<extra></extra>")
    fig.update_layout(height=340, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      title="자산 곡선 (시작 = 1.0)", legend=dict(orientation="h", y=1.12, x=1, xanchor="right"),
                      yaxis_type="log" if log else "linear")
    return fig


def drawdown_chart(eq: pd.Series, bench: pd.Series) -> go.Figure:
    fig = go.Figure()
    for s, name, color in ((eq, "전략", BLUE), (bench, BENCHMARK, ORANGE)):
        dd = drawdown(s) * 100
        fig.add_scatter(x=dd.index, y=dd.values, name=name, line=dict(color=color, width=2),
                        hovertemplate="%{x|%Y-%m-%d}<br>" + name + " %{y:.1f}%<extra></extra>")
    fig.update_layout(height=240, margin=dict(l=10, r=10, t=30, b=10), hovermode="x unified",
                      title="낙폭 (직전 최고점 대비, %)", showlegend=False, yaxis_ticksuffix="%")
    return fig


def show_backtest(result, panel: Panel) -> None:
    m = all_metrics(result)
    period_banner(result, panel)
    for w in result.warnings:
        st.warning(w)
    st.caption(result.config.describe())

    st.markdown("#### 포트폴리오 성과 (주 기준)")
    metric_table(m, ["cagr", "mdd", "calmar", "sharpe", "volatility", "underwater_days", "total_return"], spy=True)
    st.markdown("**SPY 대비 · 최소 통과 조건(Gate)**")
    gate_check(m)
    metric_table(m, ["cagr_ratio", "mdd_ratio", "excess_cagr", "exposure"])
    log = st.toggle("로그 눈금", value=False, key="log_scale")
    st.plotly_chart(equity_chart(result.equity, result.benchmark, log), use_container_width=True)
    st.plotly_chart(drawdown_chart(result.equity, result.benchmark), use_container_width=True)

    st.markdown("#### 연도별 수익 (SPY와 같은 날짜)")
    ann = annual_table(result)
    st.dataframe(ann, hide_index=True, use_container_width=True, column_config={
        "연도": st.column_config.NumberColumn(format="%d"),
        **{c: st.column_config.NumberColumn(format="percent") for c in ["전략", "SPY", "차이", "전략 MDD", "평균 거래 수익"]},
        "완료 거래": st.column_config.NumberColumn(help="그 해에 청산된 거래 수"),
        "부분 연도": st.column_config.CheckboxColumn(help="거래일이 240일 미만인 시작/끝 연도. 다른 해와 직접 비교할 때 주의."),
    })

    st.markdown("#### 거래 단위 통계 (보조: 거래당 우위가 통계적으로 있는지)")
    metric_table(m, ["n_trades", "win_rate", "avg_trade", "median_trade", "profit_factor", "payoff", "t_stat",
                     "avg_hold", "trades_per_year"])

    with st.expander(f"거래 목록 ({len(result.trades)}건) · 현재 보유 {len(result.open_positions)}종목"):
        st.dataframe(result.trades.sort_values("entry_date", ascending=False), hide_index=True, use_container_width=True,
                     column_config={"return": st.column_config.NumberColumn("수익률", format="percent"),
                                    "holding_days": st.column_config.NumberColumn("보유 거래일"),
                                    "exit_reason": "청산 사유"})
        if len(result.open_positions):
            st.markdown("**기간 끝 시점 보유 종목** (완료 거래 통계에는 포함되지 않음)")
            st.dataframe(result.open_positions, hide_index=True, use_container_width=True,
                         column_config={"unrealized_return": st.column_config.NumberColumn("평가 수익률", format="percent")})

    with st.form("save_bt", border=False):
        c1, c2 = st.columns([3, 1])
        name = c1.text_input("결과 이름", value="", placeholder="예: 55일 돌파 + 20일 저가 이탈")
        if c2.form_submit_button("💾 결과 저장", use_container_width=True):
            path = store.save_backtest(result, m, name or result.config.describe(), panel.source)
            st.success(f"저장했습니다: lab_results/{path.name}")


# =========================================================================== grid view

def grid_controls(cfg: StrategyConfig):
    params = list(cfg.entries) + list(cfg.exits) + ["max_positions"]

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

    c1, c2 = st.columns(2, gap="large")
    x_key, xs = axis(c1, "X축 파라미터", "x", False, 0)
    y_key, ys = axis(c2, "Y축 파라미터", "y", True, 1 + params.index(next(iter(cfg.exits))))  # default: signal × exit

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


def heatmap(cells: pd.DataFrame, x_key: str, y_key: str | None, metric: str, overlay: bool = True) -> go.Figure:
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
            mark = "✓" if r["pass"] else ""
            tr.append(f"{fmt_value(metric, v) if metric in DEFINITIONS else v}{'<br>' + mark if overlay and mark else ''}")
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
        colorscale=SEQ_BLUE, reversescale=(better == "low"),
        xgap=2, ygap=2, colorbar=dict(title=""),
    ))
    fig.update_layout(height=120 + 48 * len(ys), margin=dict(l=10, r=10, t=10, b=10),
                      xaxis_title=param_label(x_key), yaxis_title=param_label(y_key) if y_key else "",
                      xaxis_type="category", yaxis_type="category")
    return fig


GRID_METRICS = ["cagr", "mdd", "calmar", "sharpe", "cagr_ratio", "mdd_ratio", "n_trades", "t_stat", "avg_trade"]


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
    st.plotly_chart(heatmap(cells, x_key, y_key, metric), use_container_width=True)

    cand = cells[cells["pass"]].sort_values(["region_size", "neighbor_survival", "loyo_ratio", "calmar"], ascending=False, na_position="last")
    st.markdown("#### 통과 후보 정렬")
    st.caption("정렬 순서(사전식): ① 연결 영역 크기 → ② 이웃 생존율 → ③ LOYO 통과 비율 → ④ 칼마 비율. 앞 기준이 같을 때만 다음 기준을 봅니다.")
    if cand.empty:
        st.warning("Gate를 통과한 조합이 없습니다. 기준을 낮추기 전에, 다른 신호·청산 조합이나 유니버스를 먼저 검토하세요.")
        return
    cols = ["x"] + (["y"] if y_key else []) + ["region_size", "neighbor_survival", "loyo_ratio", "loyo_fail_years",
                                               "cagr", "mdd", "calmar", "cagr_ratio", "mdd_ratio", "n_trades", "t_stat"]
    pct = st.column_config.NumberColumn
    st.dataframe(cand[cols], hide_index=True, use_container_width=True, column_config={
        "x": pct(param_label(x_key), format="%g"),
        "y": pct(param_label(y_key) if y_key else "", format="%g"),
        "region_size": pct(ROBUST_TEXT["region_size"][0], help=ROBUST_TEXT["region_size"][1]),
        "neighbor_survival": pct(ROBUST_TEXT["neighbor_survival"][0], format="%.2f", help=ROBUST_TEXT["neighbor_survival"][1]),
        "loyo_ratio": pct(ROBUST_TEXT["loyo_ratio"][0], format="%.2f", help=ROBUST_TEXT["loyo_ratio"][1]),
        "loyo_fail_years": st.column_config.TextColumn("LOYO 실패 연도", help="그 해를 빼면 Gate를 통과하지 못하는 연도"),
        "cagr": pct("CAGR", format="percent", help=tip("cagr")),
        "mdd": pct("MDD", format="percent", help=tip("mdd")),
        "calmar": pct("Calmar", format="%.2f", help=tip("calmar")),
        "cagr_ratio": pct("SPY 대비 수익", format="%.2f", help=tip("cagr_ratio")),
        "mdd_ratio": pct("SPY 대비 낙폭", format="%.2f", help=tip("mdd_ratio")),
        "n_trades": pct("거래 수", format="%d", help=tip("n_trades")),
        "t_stat": pct("t-통계량", format="%.2f", help=tip("t_stat")),
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


# =========================================================================== pages

def research_page() -> None:
    st.title("추세추종 전략 연구실")
    st.caption("전략 정의 → 백테스트 결과 → 강건성 → 진단 순서로 봅니다. 모든 지표의 정의는 이름 옆 ⓘ 또는 ‘용어 설명’ 페이지에 있습니다.")
    panel = data_section()
    if panel is None:
        return
    st.divider()
    cfg = strategy_section()
    if cfg is None:
        return
    with st.expander("실행 규칙 자세히 보기"):
        st.markdown(RULES_KO)
    st.divider()

    tab_bt, tab_grid = st.tabs(["📈 백테스트", "🧭 강건성 격자"])
    with tab_bt:
        run = st.button("▶ 백테스트 실행", type="primary") or st.session_state.pop("auto_run", False)
        if run:
            try:
                with st.spinner("백테스트 중…"):
                    st.session_state["bt_result"] = run_backtest(panel, cfg)
            except ValueError as exc:
                st.error(str(exc))
        res = st.session_state.get("bt_result")
        if res is not None:
            if res.config != cfg:
                st.warning("아래 결과는 현재 설정과 다릅니다. 다시 실행하면 갱신됩니다.")
            show_backtest(res, panel)

    with tab_grid:
        st.caption("파라미터 1–2개를 바꿔 가며 같은 전략을 반복 실행합니다. 나머지 설정은 위 ②–④ 그대로 고정됩니다.")
        x_key, xs, y_key, ys, gates = grid_controls(cfg)
        n = len(xs or []) * (len(ys) if y_key else 1)
        too_big = (len(xs or []) > MAX_AXIS_VALUES) or (y_key and len(ys or []) > MAX_AXIS_VALUES)
        st.caption(f"조합 {n}개 · 예상 소요 약 {max(1, round(n * SECONDS_PER_BACKTEST * len(panel.tradable_symbols) / 466))}초"
                   + (f" · ⚠️ 축마다 최대 {MAX_AXIS_VALUES}개" if too_big else ""))
        if st.button("▶ 격자 실행", type="primary", disabled=bool(too_big or n == 0)):
            bar = st.progress(0.0, text="실행 중…")
            try:
                g = run_grid(panel, cfg, x_key, xs, y_key, ys, gates, progress=lambda f: bar.progress(f, text=f"실행 중… {f:.0%}"))
                st.session_state["grid_result"] = g
            except ValueError as exc:
                st.error(str(exc))
            bar.empty()
        g = st.session_state.get("grid_result")
        if g is not None:
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
                c2.button("이 값으로 설정 + 실행", on_click=apply_params, args=(vals,), use_container_width=True,
                          help="위 ②–④의 해당 파라미터를 이 값으로 바꾸고 ‘백테스트’ 탭에서 실행합니다.")
            with st.form("save_grid", border=False):
                c1, c2 = st.columns([3, 1])
                name = c1.text_input("결과 이름", value="", placeholder="예: 돌파 N × 저가 이탈 N")
                if c2.form_submit_button("💾 격자 저장", use_container_width=True):
                    s, e = pd.Timestamp(g.base.start), pd.Timestamp(g.base.end)
                    path = store.save_grid(g, name or f"격자 {param_label(g.x_key)} × {param_label(g.y_key) if g.y_key else '-'}", panel.source, (s, e))
                    st.success(f"저장했습니다: lab_results/{path.name}")


def saved_page() -> None:
    st.title("저장된 결과")
    items = store.list_results()
    if not items:
        st.info("아직 저장된 결과가 없습니다. ‘전략 연구’에서 결과를 저장하세요.")
        return
    bts = [i for i in items if i["kind"] == "backtest"]
    grids = [i for i in items if i["kind"] == "grid"]
    t1, t2 = st.tabs([f"백테스트 ({len(bts)})", f"강건성 격자 ({len(grids)})"])
    with t1:
        if not bts:
            st.info("저장된 백테스트가 없습니다.")
        else:
            rows = [{"선택": i < 3, "이름": b["name"], "저장 시각": b["saved_at"], "기간": " → ".join(b["period"]),
                     "전략": b["description"], **{DEFINITIONS[k].label: b["metrics"].get(k) for k in
                     ["cagr", "mdd", "calmar", "cagr_ratio", "mdd_ratio", "n_trades", "t_stat"]}} for i, b in enumerate(bts)]
            df = pd.DataFrame(rows)
            edited = st.data_editor(df, hide_index=True, use_container_width=True, disabled=[c for c in df.columns if c != "선택"],
                                    column_config={DEFINITIONS["cagr"].label: st.column_config.NumberColumn(format="percent"),
                                                   DEFINITIONS["mdd"].label: st.column_config.NumberColumn(format="percent")})
            chosen = [bts[i] for i in np.flatnonzero(edited["선택"].to_numpy())]
            if chosen:
                fig = go.Figure()
                palette = [BLUE, ORANGE, "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
                for i, b in enumerate(chosen[:8]):
                    eq = store.load_equity(b["path"])
                    fig.add_scatter(x=eq.index, y=eq["strategy"], name=b["name"], line=dict(color=palette[i], width=2))
                eq0 = store.load_equity(chosen[0]["path"])
                fig.add_scatter(x=eq0.index, y=eq0["SPY"], name="SPY (첫 결과 기간)", line=dict(color=GRAY, width=1.5, dash="dot"))
                fig.update_layout(height=380, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                                  title="자산 곡선 비교 (최대 8개)", legend=dict(orientation="h", y=-0.15))
                st.plotly_chart(fig, use_container_width=True)
                st.caption("서로 기간이 다른 결과는 시작점이 달라 직접 비교에 주의하세요.")
    with t2:
        if not grids:
            st.info("저장된 격자가 없습니다.")
        else:
            idx = st.selectbox("격자 선택", range(len(grids)),
                               format_func=lambda i: f"{grids[i]['saved_at']} · {grids[i]['name']} · {grids[i]['description']}")
            gm = grids[idx]
            st.caption(f"기간 {' → '.join(gm['period'])} · 데이터: {gm['data_source']}")
            cells = store.load_cells(gm["path"])
            show_grid_result(cells, gm["x_key"], gm["y_key"], gm["summary"], gm["gates"], "saved")


def glossary_page() -> None:
    st.title("용어 설명")
    st.markdown("화면에 나오는 모든 지표·조건의 정확한 정의입니다. 계산 코드(`lab/metrics.py`, `lab/grid.py`, `lab/blocks.py`)와 같은 출처에서 표시됩니다.")
    direction = {"high": "높을수록 좋음", "low": "낮을수록 좋음", "": "방향 없음"}
    groups: dict[str, list] = {}
    for key, d in DEFINITIONS.items():
        groups.setdefault(d.group, []).append(d)
    for group, defs in groups.items():
        st.subheader({"포트폴리오": "포트폴리오 지표 (주 기준)", "SPY 대비": "SPY 대비 지표 (주 기준)", "거래": "거래 단위 지표 (보조)"}[group])
        for d in defs:
            st.markdown(BR.join([f"**{d.label}** · <small>{direction[d.better]}</small>", d.short, f"`{d.formula}`"]),
                        unsafe_allow_html=True)
    st.subheader("강건성 판정")
    for name, rule, why in GATE_TEXT.values():
        st.markdown(BR.join([f"**Gate · {name}**", f"`{rule}`", f"<small>기본값 근거: {why}</small>"]), unsafe_allow_html=True)
    for name, rule in ROBUST_TEXT.values():
        st.markdown(BR.join([f"**{name}**", rule]))
    st.markdown(BR.join(["**후보 정렬**", "연결 영역 크기 → 이웃 생존율 → LOYO 통과 비율 → 칼마 비율 순의 사전식 정렬. "
                         "앞 기준이 같을 때만 다음 기준을 봅니다. 가중 합산 점수는 쓰지 않습니다."]))
    st.subheader("전략 블록")
    for k, b in ALL_BLOCKS.items():
        kind = "진입 (AND)" if k in ENTRY_BLOCKS else "청산 (OR)"
        st.markdown(BR.join([
            f"**{b.label}** · <small>{kind}</small>", f"`{b.rule}`", b.help,
            f"<small>{b.param.label}: 기본 {b.param.default:g}, 허용 범위 {b.param.minimum:g} – {b.param.maximum:g}</small>",
        ]), unsafe_allow_html=True)
    st.subheader("실행 규칙")
    st.markdown(RULES_KO)


PERSIST_PREFIXES = ("entry_", "exit_", "max_positions", "cost_bps", "min_price", "min_dv", "groups",
                    "ticker_adjust", "yf_tickers", "start", "end", "grid_", "gx_", "gy_", "gate_")
for _k in list(st.session_state.keys()):
    if str(_k).startswith(PERSIST_PREFIXES):
        st.session_state[_k] = st.session_state[_k]  # keeps settings when visiting another page

nav = st.navigation([
    st.Page(research_page, title="전략 연구", icon="📈", default=True),
    st.Page(saved_page, title="저장된 결과", icon="💾"),
    st.Page(glossary_page, title="용어 설명", icon="📖"),
])
nav.run()
