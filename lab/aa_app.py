"""자산배분 연구실 — Streamlit app for allocations of Korean-listed ETFs (ISA account), in KRW.

Run:  .venv\\Scripts\\python.exe -m streamlit run lab/aa_app.py --server.port 8511
"""
from __future__ import annotations

import json
import sys
from dataclasses import replace
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lab import aa_notes as notes  # noqa: E402
from lab import aa_rebalance as rebal  # noqa: E402
from lab import aa_signals as sig  # noqa: E402
from lab import aa_store as store  # noqa: E402
from lab import aa_tactical as tac  # noqa: E402
from lab.aa_assets import (  # noqa: E402
    CASH, CLASS_ORDER, DEFAULT_PRODUCTS, FX_LABELS, PRESETS, PRODUCTS, PROXY_SYMBOLS, label, short,
)
from lab.aa_engine import (  # noqa: E402
    OVERLAY_LABELS, OVERLAY_PARAM, REBALANCE_LABELS, RULES_KO, AllocConfig, AllocResult, Leverage, MixConfig, Overlay,
    current_targets, mix_parameter_count, run_allocation, run_mix,
)
from lab.aa_korea import BOK_PATH, KrwModel, fetch_usdkrw_tail, krw_panel, load_bok_rate, load_usdkrw, splice  # noqa: E402
from lab.aa_score import (  # noqa: E402
    AXES, AXIS_TEXT, CRISIS_WINDOWS, RAW_COLUMNS, AllocGates, axis_raw_table, crisis_table, gate_results, gate_text,
    raw_metrics, scorecard,
)
from lab.aa_validate import (  # noqa: E402
    LAZY_PROBS, block_bootstrap, cscv, headline, holdout_split, lazy_table, leave_one_out, monthly_table, split_metrics,
    timing_summary, timing_table,
)
from lab.data import download_panel  # noqa: E402
from lab.metrics import DEFINITIONS, drawdown  # noqa: E402

st.set_page_config(page_title="자산배분 연구실", page_icon="🧺", layout="wide")

BLUE, ORANGE, GRAY, AQUA = "#2a78d6", "#eb6834", "#8a8984", "#1baf7a"
PALETTE = [BLUE, AQUA, "#eda100", "#e87ba4", "#4a3aa7", "#008300", "#e34948", "#7a5c2e", "#00a3c4", "#b25fd0"]
CLASS_COLORS = {"주식": "#2a78d6", "채권": "#1baf7a", "금·실물": "#eda100", "현금·달러": "#8a8984", "레버리지": "#e34948"}
RED_GREEN = [[0, "#e34948"], [0.5, "#f6f1e8"], [1, "#1baf7a"]]
GATES = AllocGates()
BR = "  " + chr(10)
DEFAULT_START = "2005-01-03"
EARLIEST = date(1993, 1, 29)
BENCH = "S&P500(원화)"
CUSTOM = "(직접 입력)"
PERIODS = {"all": "가능한 전체 기간", "2010": "2010년 이후", "2015": "2015년 이후", "2020": "2020년 이후",
           "listed": "실제 상장일 이후만 (대용 데이터 없이)", "custom": "직접 지정"}


def init(key: str, value):
    if key not in st.session_state:
        st.session_state[key] = value
    return key


def pct(v, digits: int = 1) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v * 100:.{digits}f}%"


def num(v, digits: int = 2) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


def won(v) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:,.0f}원"


# =========================================================================== data

@st.cache_resource(show_spinner="가격·환율 데이터를 불러오는 중… (처음 한 번은 1분 정도, 이후에는 저장된 파일 사용)")
def load_data(day: str, hedge_bps: float) -> dict:
    try:
        usd, failed = download_panel(PROXY_SYMBOLS, day)
    except Exception as exc:  # offline: fall back to whatever is cached
        from lab.data import CACHE_DIR
        cached = [p for p in sorted((CACHE_DIR / "yf").glob("*.csv.gz")) if p.stem.split(".")[0] in PROXY_SYMBOLS]
        if not cached:
            raise
        last = min(pd.read_csv(p, usecols=["date"])["date"].max() for p in cached)
        usd, failed = download_panel([p.stem.split(".")[0] for p in cached], str(last))
        failed = list(failed) + [f"(오프라인: {exc})"]
    fx = splice(load_usdkrw(), fetch_usdkrw_tail())
    panel, unavailable = krw_panel(usd, fx, load_bok_rate(), KrwModel(hedge_extra_bps=hedge_bps))
    return {"panel": panel, "failed": failed, "unavailable": unavailable, "fx_last": fx.index[-1].date(),
            "usd_last": usd.last_date.date(), "fx": fx}


def data_or_stop() -> dict:
    try:
        data = load_data(str(date.today()), float(st.session_state.get("m_hedge", 20.0)))
    except Exception as exc:
        st.error(f"가격 데이터를 불러오지 못했습니다: {exc}")
        st.stop()
    return data


def sidebar(data: dict) -> None:
    with st.sidebar:
        st.caption(f"데이터: 미국 대용 ETF ~ {data['usd_last']} · 원/달러 ~ {data['fx_last']} · 원화 금리: 한국은행 기준금리")
        if data["failed"]:
            st.caption("받지 못한 대용: " + ", ".join(map(str, data["failed"])))
        st.slider("헤지 추가 비용 (bp/년)", 0, 100, step=5, key=init("m_hedge", 20),
                  help="환헤지(H) 상품은 ‘원화 금리 − 달러 금리’를 받습니다. 그 밖에 선물환 매매 비용·베이시스로 매년 더 빠지는 몫입니다. "
                       "바꾸면 모든 계산이 다시 됩니다.")


# =========================================================================== strategy builder

def _set_weights(prefix: str, weights: dict[str, float]) -> None:
    for code in PRODUCTS:
        st.session_state[f"{prefix}_on_{code}"] = code in weights
        st.session_state[f"{prefix}_w_{code}"] = int(round(weights.get(code, 0)))


def apply_preset(prefix: str) -> None:
    key = st.session_state.get(f"{prefix}_preset", CUSTOM)
    if key in PRESETS:
        _set_weights(prefix, PRESETS[key].weights)
        st.session_state[f"{prefix}_name"] = PRESETS[key].name


def _int_split(raw: dict[str, float], total: int = 100) -> dict[str, int]:
    """Integers summing to `total`, proportional to raw (largest remainder)."""
    s = sum(raw.values())
    if s <= 0:
        raw = {k: 1.0 for k in raw}
        s = float(len(raw))
    exact = {k: v / s * total for k, v in raw.items()}
    out = {k: int(np.floor(v)) for k, v in exact.items()}
    for k in sorted(exact, key=lambda k: exact[k] - out[k], reverse=True)[: total - sum(out.values())]:
        out[k] += 1
    return out


def selected_codes(prefix: str) -> list[str]:
    return [c for c in PRODUCTS if st.session_state.get(f"{prefix}_on_{c}")]


def normalise_cb(prefix: str) -> None:
    codes = selected_codes(prefix)
    for k, v in _int_split({c: float(st.session_state.get(f"{prefix}_w_{c}", 0)) for c in codes}).items():
        st.session_state[f"{prefix}_w_{k}"] = v


def equal_cb(prefix: str) -> None:
    codes = selected_codes(prefix)
    for k, v in _int_split({c: 1.0 for c in codes}).items():
        st.session_state[f"{prefix}_w_{k}"] = v


def product_help(code: str, unavailable: dict) -> str:
    p = PRODUCTS[code]
    lines = [f"종목코드 {code} · {FX_LABELS[p.fx]} · 총보수 {p.ter:g}% · 상장 {p.listed}",
             f"상장 전 기간은 대용 데이터: {p.proxy_text}"]
    if p.note:
        lines.append(p.note)
    if code in unavailable:
        lines.append("⚠️ " + unavailable[code])
    return "\n\n".join(lines)


def product_picker(prefix: str, unavailable: dict) -> None:
    fx_short = {"unhedged": "환노출", "hedged": "환헤지", "krw": "원화", "krw_native": "원화"}
    for cls in CLASS_ORDER:
        codes = [c for c, p in PRODUCTS.items() if p.asset_class == cls]
        if cls == "레버리지":
            box = st.expander("레버리지 상품 (ISA에서 매수 가능 · 증권사 레버리지 ETF 사전교육과 기본예탁금 필요)",
                              expanded=any(st.session_state.get(f"{prefix}_on_{c}") for c in codes))
        else:
            box = st.container()
        with box:
            if cls != "레버리지":
                st.markdown(f"<span style='color:{CLASS_COLORS[cls]};font-weight:600'>■ {cls}</span>", unsafe_allow_html=True)
            cols = st.columns(3)
            for i, c in enumerate(codes):
                p = PRODUCTS[c]
                tag = fx_short[p.fx] + (f" · {p.leverage:g}배" if p.leverage != 1 else "")
                cols[i % 3].checkbox(f"{p.name} · {tag}", key=init(f"{prefix}_on_{c}", False), disabled=c in unavailable,
                                     help=product_help(c, unavailable))


def allocation_bar(weights: dict[str, float]) -> go.Figure:
    fig = go.Figure()
    total = sum(weights.values()) or 1.0
    for code, w in weights.items():
        cls = PRODUCTS[code].asset_class if code in PRODUCTS else "현금·달러"
        fig.add_bar(y=[""], x=[w / total], orientation="h", name=short(code), marker_color=CLASS_COLORS[cls],
                    marker_line=dict(color="white", width=2), text=f"{short(code)} {w / total * 100:.0f}%", textposition="inside",
                    insidetextanchor="middle", hovertemplate=f"{label(code)}: {w / total * 100:.1f}%<extra></extra>")
    fig.update_layout(barmode="stack", height=90, margin=dict(l=0, r=0, t=0, b=0), showlegend=False,
                      xaxis=dict(visible=False, range=[0, 1]), yaxis=dict(visible=False))
    return fig


def weight_sliders(prefix: str) -> dict[str, float]:
    codes = selected_codes(prefix)
    if not codes:
        st.info("위에서 상품을 하나 이상 체크하세요.")
        return {}
    cols = st.columns(2)
    for i, c in enumerate(codes):
        cols[i % 2].slider(label(c), 0, 100, step=1, key=init(f"{prefix}_w_{c}", 0), format="%d%%")
    weights = {c: float(st.session_state[f"{prefix}_w_{c}"]) for c in codes}
    total = sum(weights.values())
    b1, b2, b3 = st.columns([1.3, 1, 3])
    b1.button("합계 100%로 맞추기", on_click=normalise_cb, args=(prefix,), key=f"btn_norm_{prefix}",
              help="비율은 그대로 두고 합이 100이 되게 정수로 맞춥니다.")
    b2.button("균등 분배", on_click=equal_cb, args=(prefix,), key=f"btn_eq_{prefix}")
    positive = {c: w for c, w in weights.items() if w > 0}
    fee = sum(PRODUCTS[c].ter * w for c, w in positive.items()) / total if total > 0 else 0.0
    if abs(total - 100) < 1e-9:
        b3.success(f"합계 100% · 가중 평균 총보수 {fee:.3f}%/년")
    elif total == 0:
        b3.error("비중이 모두 0입니다.")
    else:
        b3.warning(f"합계 {total:g}% ({100 - total:+g}%p) — 실행할 때는 비율대로 100%로 맞춥니다.")
    if positive:
        st.plotly_chart(allocation_bar(positive), width="stretch", config={"displayModeBar": False})
    return positive


def period_section(prefix: str, last: date, codes: list[str]) -> tuple[str, str]:
    c1, c2, c3 = st.columns([1.6, 1, 1])
    choice = c1.selectbox("기간", list(PERIODS), key=init(f"{prefix}_period", "all"), format_func=PERIODS.get,
                          help="상장 전 기간은 미국 ETF + 환율 + 금리로 만든 대용 데이터입니다. ‘실제 상장일 이후만’은 고른 상품 중 가장 늦게 상장한 날부터 봅니다.")
    start = {"all": EARLIEST, "2010": date(2010, 1, 1), "2015": date(2015, 1, 1), "2020": date(2020, 1, 1)}.get(choice)
    if choice == "listed":
        start = max((date.fromisoformat(PRODUCTS[c].listed) for c in codes if c in PRODUCTS), default=EARLIEST)
        c2.caption(f"시작 {start} (가장 늦은 상장일)")
    end = last
    if choice == "custom":
        start = c2.date_input("시작일", key=init(f"{prefix}_start", date.fromisoformat(DEFAULT_START)), min_value=EARLIEST, max_value=last)
        end = c3.date_input("종료일", key=init(f"{prefix}_end", last), min_value=EARLIEST, max_value=last)
    return str(start), str(end)


def rules_section(prefix: str, codes: list[str]) -> tuple[dict, Overlay, Leverage]:
    c1, c2, c3 = st.columns([1.3, 1, 1])
    rebalance = c1.selectbox("리밸런싱", list(REBALANCE_LABELS), key=init(f"{prefix}_rebalance", "Q"),
                             format_func=REBALANCE_LABELS.get, help="overlay를 켜면 매월 결정합니다.")
    band = c2.number_input("밴드 (%p)", 0.0, 30.0, step=1.0, key=init(f"{prefix}_band", 0.0),
                           help="0이면 결정일마다 거래. 5면 모든 상품이 목표 ±5%p 안에 있을 때 거래를 건너뜁니다.")
    cost = c3.number_input("거래비용 (bp, 한쪽)", 0.0, 100.0, step=5.0, key=init(f"{prefix}_cost", 10.0),
                           help="매매 수수료 + 호가 차이. 10bp = 0.1%.")
    o1, o2, o3 = st.columns([2.2, 1, 1.4])
    kind = o1.selectbox("overlay (전술 규칙)", list(OVERLAY_LABELS), key=init(f"{prefix}_ov_kind", "none"), format_func=OVERLAY_LABELS.get)
    param, safe = None, CASH
    if kind != "none":
        plabel, pdefault, pmin, pmax = OVERLAY_PARAM[kind]
        param = o2.number_input(plabel, float(pmin), float(pmax), step=1.0, key=init(f"{prefix}_ov_{kind}", float(pdefault)))
        if kind != "riskparity":
            options = [CASH, "459580", "261240", "0091C0", "484790", "411060"]
            safe = o3.selectbox("안전자산 (비운 비중이 가는 곳)", options, key=init(f"{prefix}_ov_safe", CASH), format_func=label)
    with st.expander("가상 레버리지 (연구용 · 실제로는 위 레버리지 상품으로 구현)"):
        st.caption("배분 전체에 일일 리셋 배수를 겁니다. 국내에 이런 상품은 없으니 ‘레버리지를 쓰면 어떻게 되나’를 볼 때만 쓰고, "
                   "실제 운용은 레버리지 ETF를 체크해 비중으로 구성하세요.")
        l1, l2, l3 = st.columns(3)
        factor = l1.number_input("전체 노출 배수 (변동성 목표에서는 상한)", 0.1, 3.0, step=0.1, key=init(f"{prefix}_lev", 1.0))
        spread = l2.number_input("조달 스프레드 (bp/년, 원화 금리 위)", 0.0, 500.0, step=10.0, key=init(f"{prefix}_spread", 100.0))
        ter = l3.number_input("레버리지 상품 추가 보수 (bp/년)", 0.0, 300.0, step=10.0, key=init(f"{prefix}_ter", 60.0))
    return (dict(rebalance=rebalance, band=float(band), cost_bps=float(cost)),
            Overlay(kind, None if param is None else float(param), safe), Leverage(float(factor), float(spread), float(ter)))


def strategy_builder(prefix: str, data: dict) -> AllocConfig | None:
    if f"{prefix}_on_379800" not in st.session_state:  # first visit: the user's own portfolio
        st.session_state[f"{prefix}_preset"] = "aaa_2026"
        apply_preset(prefix)
    c1, c2 = st.columns([1.2, 3])
    preset = c1.selectbox("프리셋 불러오기", [CUSTOM] + list(PRESETS), key=init(f"{prefix}_preset", "aaa_2026"),
                          format_func=lambda k: k if k == CUSTOM else PRESETS[k].name, on_change=apply_preset, args=(prefix,))
    if preset in PRESETS:
        c2.caption(PRESETS[preset].note)
    st.markdown("**상품 고르기** · 상품 이름에 마우스를 올리면 종목코드·보수·상장일·대용 데이터가 보입니다.")
    product_picker(prefix, data["unavailable"])
    st.markdown("**비중**")
    weights = weight_sliders(prefix)
    if not weights:
        return None
    st.markdown("**규칙**")
    start, end = period_section(prefix, data["panel"].last_date.date(), list(weights))
    rules, overlay, lev = rules_section(prefix, list(weights))
    name = st.text_input("전략 이름", key=init(f"{prefix}_name", "내 배분"))
    if start >= end:
        st.warning("시작일이 종료일보다 앞서야 합니다.")
        return None
    return AllocConfig(name or "내 배분", weights, start, end, overlay=overlay, leverage=lev, **rules)


# =========================================================================== result views

def kpi_row(m: dict) -> None:
    cols = st.columns(8)
    items = [("CAGR", pct(m["cagr"]), f"{(m['cagr'] - m['spy_cagr']) * 100:+.1f}%p", f"{BENCH} {pct(m['spy_cagr'])}"),
             ("최대 낙폭", pct(m["mdd"]), f"{(m['mdd'] - m['spy_mdd']) * 100:+.1f}%p", f"{BENCH} {pct(m['spy_mdd'])}"),
             ("현금 초과 샤프", num(m["sharpe_excess"]), None, "mean(r − 원화 현금) / std × √252"),
             ("칼마", num(m["calmar"]), f"{m['calmar'] - m['spy_calmar']:+.2f}", f"{BENCH} {num(m['spy_calmar'])}"),
             ("변동성", pct(m.get("volatility")), None, "연율화 표준편차"),
             ("최악 12개월", pct(m["worst_12m"]), None, None),
             ("CVaR 5% (월)", pct(m["cvar_5"]), None, f"최악 달 {pct(m['worst_month'])}"),
             ("연간 회전율", pct(m["turnover"], 0), None, f"비용 드래그 {pct(m['cost_drag'], 2)}/년")]
    for c, (lab, val, delta, sub) in zip(cols, items):
        c.metric(lab, val, delta=delta, help=sub)


def gates_line(m: dict) -> None:
    g = gate_results(m)
    txt = " · ".join(("✅ " if ok else "❌ ") + k for k, ok in g.items())
    st.markdown(txt + f"  \n<small>{BENCH} 대비 · 탐색 Gate: 칼마 ≥ {BENCH} 칼마 그리고 |MDD| ≤ 0.75×|{BENCH} MDD| · "
                f"최종 Gate: {gate_text()}</small>", unsafe_allow_html=True)


def equity_chart(res: AllocResult, log: bool = True) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=res.equity.index, y=res.equity, name=res.config.name, line=dict(color=BLUE, width=2))
    fig.add_scatter(x=res.benchmark.index, y=res.benchmark, name=BENCH, line=dict(color=ORANGE, width=1.5))
    if res.sleeves:
        for i, (n, s) in enumerate(res.sleeves.items()):
            fig.add_scatter(x=s.index, y=s, name=f"슬리브 · {n}", line=dict(color=PALETTE[(i + 2) % len(PALETTE)], width=1, dash="dot"))
    fig.update_layout(height=380, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10), title="자산 곡선 (시작 = 1, 원화)",
                      yaxis_type="log" if log else "linear", legend=dict(orientation="h", y=-0.15))
    return fig


def drawdown_chart(res: AllocResult) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=res.equity.index, y=drawdown(res.equity), name=res.config.name, fill="tozeroy", line=dict(color=BLUE, width=1))
    fig.add_scatter(x=res.benchmark.index, y=drawdown(res.benchmark), name=BENCH, line=dict(color=ORANGE, width=1))
    fig.update_layout(height=260, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10), title="낙폭", yaxis_tickformat=".0%",
                      legend=dict(orientation="h", y=-0.2))
    return fig


def annual_frame(res: AllocResult) -> pd.DataFrame:
    rows = []
    eq, bm = res.equity, res.benchmark
    for year, part in eq.groupby(eq.index.year):
        prev = eq[eq.index < part.index[0]]
        prev_b = bm[bm.index < part.index[0]]
        e0 = prev.iloc[-1] if len(prev) else part.iloc[0]
        b0 = prev_b.iloc[-1] if len(prev_b) else bm.loc[part.index[0]]
        rows.append({"연도": int(year), "전략": part.iloc[-1] / e0 - 1, BENCH: bm.loc[part.index[-1]] / b0 - 1,
                     "전략 MDD": drawdown(part / e0).min(), "부분 연도": len(part) < 240})
    return pd.DataFrame(rows)


def annual_chart(ann: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=ann["연도"], y=ann["전략"], name="전략", marker_color=BLUE)
    fig.add_bar(x=ann["연도"], y=ann[BENCH], name=BENCH, marker_color=ORANGE)
    fig.update_layout(height=280, barmode="group", margin=dict(l=10, r=10, t=30, b=10), title="연도별 수익률", yaxis_tickformat=".0%",
                      legend=dict(orientation="h", y=-0.2))
    return fig


def weights_chart(res: AllocResult) -> go.Figure:
    w = res.weights.iloc[::5]  # weekly points are enough to read
    fig = go.Figure()
    for i, col in enumerate(w.columns):
        if (w[col].abs() > 1e-9).any():
            fig.add_scatter(x=w.index, y=w[col], name=short(col), stackgroup="one", line=dict(width=0.5, color=PALETTE[i % len(PALETTE)]))
    if (res.exposure != 1.0).any():
        fig.add_scatter(x=res.exposure.index, y=res.exposure, name="노출 k", line=dict(color="black", width=1.5, dash="dash"), yaxis="y2")
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="노출"))
    fig.update_layout(height=300, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10), title="비중 변화 (가상 레버리지 전)",
                      yaxis_tickformat=".0%", legend=dict(orientation="h", y=-0.2))
    return fig


def isa_table(cfg: AllocConfig, key: str) -> None:
    """What to buy in the ISA account for this strategy."""
    st.markdown("**ISA에서 이 전략을 실행할 상품**")
    amount = st.number_input("투자금 (원)", 0, 10_000_000_000, step=1_000_000, key=init(f"{key}_amount", 10_000_000),
                             help="비중대로 나눈 금액을 보여 줍니다. 실제 매수 수량은 ‘리밸런싱 계산기’에서 현재가를 넣어 계산하세요.")
    total = sum(cfg.weights.values())
    rows = []
    for code, w in cfg.weights.items():
        p = PRODUCTS.get(code)
        rows.append({"상품": p.name if p else label(code), "종목코드": code if p else "", "비중": w / total,
                     "금액": amount * w / total, "환": FX_LABELS[p.fx] if p else "원화", "배수": p.leverage if p else 1.0,
                     "총보수(%)": p.ter if p else 0.0, "상장일": p.listed if p else "", "상장 전 대용": p.proxy_text if p else "한국은행 기준금리"})
    df = pd.DataFrame(rows)
    st.dataframe(df, hide_index=True, width="stretch", column_config={
        "비중": st.column_config.NumberColumn(format="percent"), "금액": st.column_config.NumberColumn(format="localized"),
        "배수": st.column_config.NumberColumn(format="%.0fx"), "총보수(%)": st.column_config.NumberColumn(format="%.4g")})
    fee = float((df["비중"] * df["총보수(%)"]).sum())
    extra = []
    if cfg.overlay.kind != "none":
        extra.append(f"overlay({cfg.overlay.describe()})가 켜져 있어 매달 비중이 바뀝니다. 이번 달 목표는 ‘리밸런싱 계산기’에서 확인하세요.")
    if cfg.leverage.factor != 1.0:
        extra.append("가상 레버리지는 국내 상품으로 그대로 살 수 없습니다.")
    st.caption(f"가중 평균 총보수 {fee:.3f}%/년. " + " ".join(extra))


def show_result(res: AllocResult, m: dict, key: str) -> None:
    for w in res.warnings:
        st.caption("ℹ️ " + w)
    s, e = res.period
    st.caption(f"{res.config.describe()} · 기간 {s.date()} → {e.date()} ({len(res.equity):,} 거래일) · 리밸런싱 거래 {res.n_trades}회")
    kpi_row(m)
    gates_line(m)
    c1, c2 = st.columns([3, 2])
    c1.plotly_chart(equity_chart(res, st.toggle("로그 축", value=True, key=f"log_{key}")), width="stretch")
    c2.plotly_chart(drawdown_chart(res), width="stretch")
    c3, c4 = st.columns(2)
    ann = annual_frame(res)
    c3.plotly_chart(annual_chart(ann), width="stretch")
    c4.plotly_chart(weights_chart(res), width="stretch")
    isa_table(res.config, key)
    with st.expander("월별 수익률"):
        st.plotly_chart(monthly_heatmap(res.equity), width="stretch")
    with st.expander("연도별 표 · 위기 구간"):
        a, b = st.columns(2)
        a.dataframe(ann, hide_index=True, width="stretch", column_config={
            "전략": st.column_config.NumberColumn(format="percent"), BENCH: st.column_config.NumberColumn(format="percent"),
            "전략 MDD": st.column_config.NumberColumn(format="percent")})
        ct = crisis_table(res).rename(columns={"S&P500": BENCH})
        if len(ct):
            b.dataframe(ct, hide_index=True, width="stretch", column_config={
                "전략": st.column_config.NumberColumn(format="percent"), BENCH: st.column_config.NumberColumn(format="percent")})
        else:
            b.info("기간 안에 등록된 위기 구간이 없습니다.")


# =========================================================================== page 1: research

def research_page() -> None:
    data = data_or_stop()
    sidebar(data)
    panel = data["panel"]
    st.title("자산배분 연구실")
    st.caption("ISA에서 살 수 있는 **국내 상장 ETF**로 배분을 만들고 **원화 기준**으로 시험합니다. 상장 전 기간은 미국 ETF + 원/달러 환율 + 두 나라 금리로 만든 대용 데이터입니다. "
               "세금은 반영하지 않습니다. 모든 지표의 정의는 ‘용어·상품 설명’에 있습니다.")
    st.subheader("① 배분")
    cfg = strategy_builder("r", data)
    with st.expander("실행 규칙 자세히 보기"):
        st.markdown(RULES_KO)
    if cfg is None:
        return
    st.divider()
    st.subheader("② 결과")
    if st.button("▶ 백테스트 실행", type="primary"):
        try:
            with st.spinner("계산 중…"):
                res = run_allocation(panel, cfg)
                m = raw_metrics(res)
                notes.log_run(cfg.name, cfg.describe(), (str(res.period[0].date()), str(res.period[1].date())), m, config=cfg.to_dict())
                st.session_state["aa_result"] = (res, m, panel.source)
        except ValueError as exc:
            st.error(str(exc))
    stored = st.session_state.get("aa_result")
    if stored is None:
        st.caption("▶ 버튼을 누르면 결과가 나오고, 실행한 전략은 모두 ‘전략 비교’ 페이지에 쌓입니다.")
        return
    res, m, source = stored
    if res.config != cfg:
        st.warning("아래 결과는 현재 설정과 다릅니다. 다시 실행하면 갱신됩니다.")
    show_result(res, m, "live")
    with st.form("save_alloc", border=False):
        c1, c2 = st.columns([3, 1])
        name = c1.text_input("저장 이름", value=res.config.name)
        if c2.form_submit_button("💾 전략 저장"):
            path = store.save_result(res, m, name, source)
            st.success(f"저장했습니다: lab_results/{path.name}")


# =========================================================================== shared: candidates and re-runs

def usable(obj, panel) -> bool:
    cfgs = [c for c, _ in obj.sleeves] if isinstance(obj, MixConfig) else [obj]
    return all(all(s == CASH or s in panel.symbols for s in c.weights) for c in cfgs)


def config_key(obj) -> str:
    if isinstance(obj, MixConfig):
        d = obj.to_dict()
        d.pop("name")
        for s in d["sleeves"]:
            s["config"] = {k: v for k, v in s["config"].items() if k not in ("name", "start", "end")}
        return "mix:" + json.dumps(d, sort_keys=True, default=str)
    d = {k: v for k, v in obj.to_dict().items() if k not in ("name", "start", "end")}
    return json.dumps(d, sort_keys=True, default=str)


def config_from_json(text: str):
    d = json.loads(text)
    return MixConfig.from_dict(d["mix"]) if "mix" in d else AllocConfig.from_dict(d)


def run_candidate(panel, obj, start: str, end: str) -> tuple[AllocResult, int]:
    if isinstance(obj, MixConfig):
        mix = MixConfig(obj.name, tuple((replace(c, start=start, end=end), w) for c, w in obj.sleeves), obj.rebalance, obj.cost_bps)
        return run_mix(panel, mix), mix_parameter_count(mix)
    cfg = replace(obj, start=start, end=end)
    return run_allocation(panel, cfg), cfg.parameter_count()


def common_period(results: dict[str, AllocResult]) -> dict[str, AllocResult]:
    """Trim every result to the dates all of them share, restarting each curve at 1."""
    idx = None
    for r in results.values():
        idx = r.equity.index if idx is None else idx.intersection(r.equity.index)
    out = {}
    for n, r in results.items():
        if len(r.equity) == len(idx):
            out[n] = r
            continue
        eq, bm = r.equity.loc[idx], r.benchmark.loc[idx]
        out[n] = AllocResult(r.config, eq / eq.iloc[0], bm / bm.iloc[0], r.cash.loc[idx], r.weights.loc[idx], r.exposure.loc[idx],
                             r.turnover, r.n_trades, r.sleeves, r.warnings)
    return out


def log_config(obj) -> dict:
    return {"mix": obj.to_dict()} if isinstance(obj, MixConfig) else obj.to_dict()


# =========================================================================== page 2: compare everything tested

def tested_strategies(panel, include_presets: bool) -> tuple[list[dict], int]:
    """Every strategy run in this lab (run log), saved strategies and optionally presets; one row per setting."""
    entries, seen, skipped = [], set(), 0
    runs = notes.load_runs()
    for _, row in runs.iloc[::-1].iterrows():  # newest first
        text = row.get("config")
        if not isinstance(text, str) or not text.strip():
            skipped += 1
            continue
        try:
            obj = config_from_json(text)
        except (ValueError, KeyError, TypeError):
            skipped += 1
            continue
        if not usable(obj, panel):
            skipped += 1
            continue
        k = config_key(obj)
        if k in seen:
            continue
        seen.add(k)
        entries.append({"이름": str(row["name"]), "출처": "실행 기록", "시각": str(row["time"])[:16].replace("T", " "), "obj": obj})
    for meta in store.list_saved():
        try:
            obj = store.config_of(meta)
        except (KeyError, TypeError, ValueError):
            continue
        if not usable(obj, panel):
            skipped += 1
            continue
        k = config_key(obj)
        if k in seen:
            continue
        seen.add(k)
        entries.append({"이름": meta["name"], "출처": "저장", "시각": meta["saved_at"][:16].replace("T", " "), "obj": obj})
    if include_presets:
        for p in PRESETS.values():
            obj = AllocConfig(p.name, p.weights, DEFAULT_START, str(panel.last_date.date()))
            k = config_key(obj)
            if k in seen or not usable(obj, panel):
                continue
            seen.add(k)
            entries.append({"이름": p.name, "출처": "프리셋", "시각": "", "obj": obj})
    counts: dict[str, int] = {}
    for e in entries:  # same name, different setting (e.g. a preset edited before running): number them
        counts[e["이름"]] = counts.get(e["이름"], 0) + 1
        if counts[e["이름"]] > 1:
            e["이름"] = f"{e['이름']} #{counts[e['이름']]}"
    return entries, skipped


def compare_page() -> None:
    data = data_or_stop()
    sidebar(data)
    panel = data["panel"]
    st.title("전략 비교")
    st.caption("이 연구실에서 **실행한 모든 전략**(같은 설정은 한 번만)과 저장한 전략을 같은 기간으로 다시 돌려 한 표·한 그래프로 봅니다. "
               "표의 ‘선택’을 켜고 끄면 그래프가 바뀝니다. 열 이름을 누르면 정렬됩니다.")
    c1, c2, c3, c4 = st.columns([1, 1, 1.2, 1.2])
    last = panel.last_date.date()
    start = c1.date_input("시작일", key=init("cmp_start", date.fromisoformat(DEFAULT_START)), min_value=EARLIEST, max_value=last)
    end = c2.date_input("종료일", key=init("cmp_end", last), min_value=EARLIEST, max_value=last)
    include_presets = c3.toggle("프리셋도 함께", key=init("cmp_presets", True))
    same_period = c4.toggle("모두 공통 기간으로 맞추기", key=init("cmp_common", True),
                            help="켜면 모든 전략이 데이터가 겹치는 기간에서만 비교됩니다(공정한 비교). 끄면 각자 가능한 기간 전체.")
    entries, skipped = tested_strategies(panel, include_presets)
    if skipped:
        st.caption(f"예전 버전(미국 ETF 기준)이나 설정이 없는 기록 {skipped}건은 제외했습니다.")
    if not entries:
        st.info("아직 실행한 전략이 없습니다. ‘전략 연구’에서 ▶ 백테스트 실행을 누르면 여기에 쌓입니다.")
        return
    cache = st.session_state.setdefault("cmp_cache", {})
    version = (str(panel.last_date.date()), float(st.session_state.get("m_hedge", 20.0)), str(start), str(end))
    results, errors = {}, []
    with st.spinner(f"{len(entries)}개 전략 계산 중…"):
        for i, e in enumerate(entries):
            ck = (config_key(e["obj"]), version)
            if ck not in cache:
                try:
                    cache[ck] = run_candidate(panel, e["obj"], str(start), str(end))[0]
                except ValueError as exc:
                    cache[ck] = str(exc)
            r = cache[ck]
            if isinstance(r, str):
                errors.append(f"{e['이름']}: {r}")
            else:
                results[i] = r
    for err in errors[:5]:
        st.caption("⚠️ " + err)
    if not results:
        return
    if same_period:
        results = {k: v for k, v in zip(results, common_period({str(k): v for k, v in results.items()}).values())}
    metrics = {i: raw_metrics(r) for i, r in results.items()}
    rows = []
    for i, r in results.items():
        e, m = entries[i], metrics[i]
        g = gate_results(m)
        rows.append({"선택": False, "이름": e["이름"], "출처": e["출처"], "시각": e["시각"], "CAGR": m["cagr"], "MDD": m["mdd"],
                     "샤프": m["sharpe_excess"], "칼마": m["calmar"], "변동성": m.get("volatility"), "최악 12개월": m["worst_12m"],
                     "탐색 Gate": g["탐색 Gate"], "최종 Gate": g["최종 Gate"], "기간": f"{r.period[0].date()} → {r.period[1].date()}",
                     "설정": r.config.describe() if not r.sleeves else e["obj"].describe(), "_i": i})
    df = pd.DataFrame(rows)
    df["선택"] = df.index < 6  # newest six; the editor keeps the user's ticks while the list is unchanged
    first = next(iter(results.values()))
    bm = curve_bench = first.benchmark
    bmm = raw_metrics(first)
    st.caption(f"기준선 {BENCH}: CAGR {pct(bmm['spy_cagr'])} · MDD {pct(bmm['spy_mdd'])} · 칼마 {num(bmm['spy_calmar'])} "
               f"(첫 전략의 기간 기준) · 최종 Gate 경계: CAGR ≥ {pct(GATES.min_cagr_ratio * bmm['spy_cagr'])}, MDD ≥ {pct(-GATES.max_mdd_abs)}")
    edited = st.data_editor(df.drop(columns=["_i"]), hide_index=True, width="stretch", key=f"cmp_editor_{len(df)}_{version}",
                            disabled=[c for c in df.columns if c != "선택"], column_config={
                                "선택": st.column_config.CheckboxColumn("선택", help="그래프에 그릴 전략"),
                                "CAGR": st.column_config.NumberColumn(format="percent"), "MDD": st.column_config.NumberColumn(format="percent"),
                                "샤프": st.column_config.NumberColumn("현금 초과 샤프", format="%.2f"),
                                "칼마": st.column_config.NumberColumn(format="%.2f"), "변동성": st.column_config.NumberColumn(format="percent"),
                                "최악 12개월": st.column_config.NumberColumn(format="percent"),
                                "탐색 Gate": st.column_config.CheckboxColumn(), "최종 Gate": st.column_config.CheckboxColumn()})
    chosen = [int(df.loc[j, "_i"]) for j in np.flatnonzero(edited["선택"].to_numpy())]
    # risk/return map: every strategy, chosen ones coloured
    fig = go.Figure()
    others = [i for i in results if i not in chosen]
    fig.add_scatter(x=[metrics[i]["mdd"] for i in others], y=[metrics[i]["cagr"] for i in others], mode="markers", name="그 밖의 전략",
                    marker=dict(color=GRAY, size=8, opacity=0.5), text=[entries[i]["이름"] for i in others],
                    hovertemplate="%{text}<br>CAGR %{y:.1%} · MDD %{x:.1%}<extra></extra>")
    for n, i in enumerate(chosen):
        fig.add_scatter(x=[metrics[i]["mdd"]], y=[metrics[i]["cagr"]], mode="markers+text", name=entries[i]["이름"], text=[entries[i]["이름"]],
                        textposition="top center" if n % 2 == 0 else "bottom center", marker=dict(color=PALETTE[n % len(PALETTE)], size=12),
                        hovertemplate="%{text}<br>CAGR %{y:.1%} · MDD %{x:.1%}<extra></extra>")
    fig.add_scatter(x=[bmm["spy_mdd"]], y=[bmm["spy_cagr"]], mode="markers+text", name=BENCH, text=[BENCH], textposition="bottom center",
                    marker=dict(color=ORANGE, size=14, symbol="star"))
    fig.add_vline(x=-GATES.max_mdd_abs, line=dict(color=GRAY, dash="dot"), annotation_text="최종 Gate MDD 기준", annotation_position="bottom right")
    fig.add_hline(y=GATES.min_cagr_ratio * bmm["spy_cagr"], line=dict(color=GRAY, dash="dot"), annotation_text="최종 Gate CAGR 기준", annotation_position="bottom left")
    fig.update_layout(height=420, margin=dict(l=10, r=10, t=30, b=10), title="수익·낙폭 지도 (오른쪽 위일수록 좋음 · 점선 오른쪽 위 = 최종 Gate 통과 영역)",
                      xaxis=dict(title="최대 낙폭", tickformat=".0%"), yaxis=dict(title="CAGR", tickformat=".0%"), showlegend=False)
    st.plotly_chart(fig, width="stretch")
    if not chosen:
        st.info("표에서 ‘선택’을 켜면 자산 곡선과 낙폭을 겹쳐 그립니다.")
        return
    log = st.toggle("로그 축", value=True, key="cmp_log")
    eq_fig, dd_fig = go.Figure(), go.Figure()
    for n, i in enumerate(chosen[:10]):
        r = results[i]
        color = PALETTE[n % len(PALETTE)]
        eq_fig.add_scatter(x=r.equity.index, y=r.equity, name=entries[i]["이름"], line=dict(color=color, width=2))
        dd_fig.add_scatter(x=r.equity.index, y=drawdown(r.equity), name=entries[i]["이름"], line=dict(color=color, width=1.2))
    eq_fig.add_scatter(x=curve_bench.index, y=curve_bench, name=BENCH, line=dict(color=ORANGE, width=1.5, dash="dot"))
    dd_fig.add_scatter(x=bm.index, y=drawdown(bm), name=BENCH, line=dict(color=ORANGE, width=1, dash="dot"))
    eq_fig.update_layout(height=400, hovermode="x unified", yaxis_type="log" if log else "linear", margin=dict(l=10, r=10, t=30, b=10),
                         title="자산 곡선 (시작 = 1, 원화)", legend=dict(orientation="h", y=-0.15))
    dd_fig.update_layout(height=280, hovermode="x unified", yaxis_tickformat=".0%", margin=dict(l=10, r=10, t=30, b=10), title="낙폭",
                         legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(eq_fig, width="stretch")
    st.plotly_chart(dd_fig, width="stretch")
    if len(chosen) > 10:
        st.caption("그래프는 처음 10개만 그립니다.")
    names = []
    for i in chosen[:10]:  # unique labels (a preset and a run may share a name)
        n = entries[i]["이름"]
        while n in names:
            n += " "
        names.append(n)
    years = pd.concat([annual_frame(results[i]).set_index("연도")["전략"].rename(n) for i, n in zip(chosen[:10], names)], axis=1)
    if len(years.columns):
        years[BENCH] = annual_frame(results[chosen[0]]).set_index("연도")[BENCH]
        with st.expander("연도별 수익률 표"):
            st.dataframe(years.T, width="stretch", column_config={c: st.column_config.NumberColumn(str(c), format="percent") for c in years.index})


# =========================================================================== page 3: scorecard and mixes

def candidates(panel) -> dict[str, object]:
    """name -> AllocConfig | MixConfig for everything the scorecard can compare."""
    out: dict[str, object] = {}
    for p in PRESETS.values():
        out[f"프리셋 · {p.name}"] = AllocConfig(p.name, p.weights, DEFAULT_START, str(panel.last_date.date()))
    for meta in store.list_saved():
        obj = store.config_of(meta)
        if usable(obj, panel):
            out[f"저장 · {meta['name']} ({meta['saved_at'][:10]})"] = obj
    live = st.session_state.get("aa_result")
    if live is not None and usable(live[0].config, panel):
        out[f"현재 화면 · {live[0].config.name}"] = live[0].config
    return out


def score_heatmap(table: pd.DataFrame, raw: pd.DataFrame) -> go.Figure:
    cols = ["견고 (참고)"] + AXES
    z = table[cols].to_numpy(dtype=float)
    detail = axis_raw_table(raw).loc[table.index]
    hover = []
    for name in table.index:
        row = []
        for c in cols:
            if c == "견고 (참고)":
                row.append(f"{name}<br>견고 (축 점수 평균) {table.loc[name, c]:.0f}")
            else:
                col, lab, kind = RAW_COLUMNS[c]
                v = detail.loc[name, c]
                txt = pct(v) if kind == "pct" else (f"{v:.0f}" if kind == "int" else num(v))
                row.append(f"{name}<br>{c} 점수 {table.loc[name, c]:.0f}<br>{lab}: {txt}")
        hover.append(row)
    fig = go.Figure(go.Heatmap(z=z, x=cols, y=list(table.index), colorscale=RED_GREEN, zmin=0, zmax=100, text=np.round(z, 0),
                               texttemplate="%{text:.0f}", hovertext=hover, hoverinfo="text", showscale=False, xgap=2, ygap=2))
    fig.update_layout(height=80 + 34 * len(table), margin=dict(l=10, r=10, t=10, b=10), yaxis=dict(autorange="reversed"),
                      xaxis=dict(side="top"))
    return fig


def scorecard_page() -> None:
    data = data_or_stop()
    sidebar(data)
    panel = data["panel"]
    st.title("견고함 성적표 · 전략 믹스")
    st.markdown("고른 전략들을 **같은 기간**에 다시 돌려 아홉 축으로 점수를 매깁니다. 점수는 비교 집합 안의 순위(0–100)라 집합이 바뀌면 같이 바뀝니다. "
                "‘견고’는 축 점수의 평균으로 **참고용**이며, 판정은 Gate(통과/미달)로 합니다.")
    cands = candidates(panel)
    default = [k for k in cands if k.startswith("프리셋 · 내 AAA") or k.endswith("S&P500 보유") or "영구" in k or "60/40" in k]
    chosen = st.multiselect("비교할 전략", list(cands), key=init("sc_chosen", default))
    c1, c2, c3 = st.columns([1, 1, 2])
    last = panel.last_date.date()
    start = c1.date_input("시작일", key=init("sc_start", date.fromisoformat(DEFAULT_START)), min_value=EARLIEST, max_value=last)
    end = c2.date_input("종료일", key=init("sc_end", last), min_value=EARLIEST, max_value=last)
    with c3.expander("견고 가중치 (기본 동일)"):
        wcols = st.columns(len(AXES))
        axis_w = {a: wcols[i].number_input(a, 0.0, 5.0, step=0.5, key=init(f"sc_w_{a}", 1.0)) for i, a in enumerate(AXES)}

    with st.expander("믹스 만들기 (전략을 비율로 섞기)"):
        st.caption("각 슬리브는 따로 운용되고, 슬리브 사이 비율은 정해진 주기로 되돌립니다. 믹스는 비교 목록에 한 줄로 더해집니다.")
        mix_parts = st.multiselect("슬리브", list(cands), key=init("mix_parts", []), max_selections=5)
        mw = {}
        if mix_parts:
            cols = st.columns(len(mix_parts) + 2)
            for i, k in enumerate(mix_parts):
                mw[k] = cols[i].slider(f"{cands[k].name}", 0, 100, step=5, key=init(f"mix_w_{k}", int(round(100 / len(mix_parts)))), format="%d%%")
            mix_reb = cols[-2].selectbox("슬리브 재조정", ["M", "Q", "A", "none"], key=init("mix_reb", "M"), format_func=REBALANCE_LABELS.get)
            mix_name = cols[-1].text_input("믹스 이름", key=init("mix_name", "믹스 1"))
            if st.button("➕ 믹스를 비교 목록에 추가"):
                parts = tuple((cands[k], float(w)) for k, w in mw.items() if w > 0 and not isinstance(cands[k], MixConfig))
                if len(parts) < 2:
                    st.warning("슬리브를 두 개 이상 고르세요 (믹스 안의 믹스는 지원하지 않습니다).")
                else:
                    st.session_state.setdefault("sc_mixes", {})[f"믹스 · {mix_name}"] = MixConfig(mix_name, parts, mix_reb)
    mixes = st.session_state.get("sc_mixes", {})
    if mixes:
        keep = st.multiselect("추가된 믹스", list(mixes), default=list(mixes), key="sc_mix_keep")
        mixes = {k: mixes[k] for k in keep}
        st.session_state["sc_mixes"] = mixes

    if st.button("▶ 성적표 계산", type="primary"):
        todo = {k: cands[k] for k in chosen}
        todo.update(mixes)
        if len(todo) < 2:
            st.warning("두 개 이상 고르세요. 점수는 비교 집합 안의 순위입니다.")
            return
        results, params, objs, errors = {}, {}, {}, []
        with st.spinner(f"{len(todo)}개 전략 계산 중…"):
            for name, obj in todo.items():
                try:
                    res, n = run_candidate(panel, obj, str(start), str(end))
                    short_name = name.split(" · ", 1)[-1]
                    results[short_name], params[short_name], objs[short_name] = res, n, obj
                except ValueError as exc:
                    errors.append(f"{name}: {exc}")
        for e in errors:
            st.error(e)
        if len(results) >= 2:
            results = common_period(results)
            idx = next(iter(results.values())).equity.index
            table, raw = scorecard(results, axis_w, params)
            st.session_state["sc_view"] = (table, raw, results, (str(idx[0].date()), str(idx[-1].date())))
            for n, r in results.items():
                notes.log_run(n, objs[n].describe(), (str(idx[0].date()), str(idx[-1].date())), raw.loc[n].to_dict(),
                              kind="mix" if r.sleeves else "alloc", config=log_config(objs[n]))
    view = st.session_state.get("sc_view")
    if view is None:
        return
    table, raw, results, period = view
    st.caption(f"공통 기간 {period[0]} → {period[1]} · {len(table)}개 전략 · 초록 = 그 축에서 더 나음, 빨강 = 더 아쉬움. 칸에 마우스를 올리면 원래 값이 보입니다.")
    st.plotly_chart(score_heatmap(table, raw), width="stretch")
    show = table.copy()
    show.insert(0, "전략", show.index)
    st.dataframe(show[["전략", "탐색 Gate", "최종 Gate", "cagr", "mdd", "calmar", "sharpe_excess", "volatility", "turnover"]],
                 hide_index=True, width="stretch", column_config={
                     "cagr": st.column_config.NumberColumn("CAGR", format="percent"), "mdd": st.column_config.NumberColumn("MDD", format="percent"),
                     "calmar": st.column_config.NumberColumn("칼마", format="%.2f"), "sharpe_excess": st.column_config.NumberColumn("현금 초과 샤프", format="%.2f"),
                     "volatility": st.column_config.NumberColumn("변동성", format="percent"), "turnover": st.column_config.NumberColumn("연간 회전율", format="percent"),
                     "탐색 Gate": st.column_config.CheckboxColumn("탐색 Gate"), "최종 Gate": st.column_config.CheckboxColumn("최종 Gate")})
    with st.expander("축별 원래 값"):
        det = axis_raw_table(raw).loc[table.index].copy()
        det.insert(0, "전략", det.index)
        st.dataframe(det, hide_index=True, width="stretch", column_config={
            a: st.column_config.NumberColumn(format="percent" if RAW_COLUMNS[a][2] == "pct" else "%.2f") for a in AXES})
    fig = go.Figure()
    for i, (n, r) in enumerate(results.items()):
        fig.add_scatter(x=r.equity.index, y=r.equity, name=n, line=dict(color=PALETTE[i % len(PALETTE)], width=1.8))
    first = next(iter(results.values()))
    fig.add_scatter(x=first.benchmark.index, y=first.benchmark, name=BENCH, line=dict(color=GRAY, width=1.5, dash="dot"))
    fig.update_layout(height=380, hovermode="x unified", yaxis_type="log", margin=dict(l=10, r=10, t=30, b=10), title="자산 곡선 (공통 기간, 로그)",
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    mix_names = [n for n, r in results.items() if r.sleeves]
    if mix_names:
        with st.form("save_mix", border=False):
            c1, c2 = st.columns([3, 1])
            pick = c1.selectbox("저장할 믹스", mix_names)
            if c2.form_submit_button("💾 믹스 저장"):
                mix = mixes.get(f"믹스 · {pick}")
                if mix is not None:
                    path = store.save_result(results[pick], raw.loc[pick].to_dict(), pick, panel.source, mix=mix)
                    st.success(f"저장했습니다: lab_results/{path.name}")


# =========================================================================== page 4: rebalancing calculator

def rebalance_page() -> None:
    data = data_or_stop()
    sidebar(data)
    panel = data["panel"]
    st.title("리밸런싱 계산기")
    st.caption("지금 ISA에 들어 있는 금액과 이번에 넣을 돈을 적으면, 목표 전략에 맞추려면 무엇을 얼마나 사고팔지 계산합니다. "
               "입력한 보유 내역은 이 컴퓨터의 lab_results/aa_holdings.json에만 저장됩니다(Git 제외).")
    targets_src: dict[str, object] = {f"프리셋 · {p.name}": AllocConfig(p.name, p.weights, DEFAULT_START, str(panel.last_date.date()))
                                      for p in PRESETS.values()}
    for meta in store.list_saved():
        obj = store.config_of(meta)
        if isinstance(obj, AllocConfig) and usable(obj, panel):
            targets_src[f"저장 · {meta['name']} ({meta['saved_at'][:10]})"] = obj
    live = st.session_state.get("aa_result")
    if live is not None and usable(live[0].config, panel):
        targets_src[f"전략 연구 화면 · {live[0].config.name}"] = live[0].config
    keys = list(targets_src)
    if st.session_state.get("rb_target") not in keys:
        st.session_state["rb_target"] = keys[0]
    pick = st.selectbox("목표 전략", keys, key="rb_target")
    cfg = targets_src[pick]
    try:
        tw, why = current_targets(panel, cfg)
    except ValueError as exc:
        st.error(str(exc))
        return
    st.caption(f"목표: {why} · " + " · ".join(f"{short(c)} {w * 100:.1f}%" for c, w in tw.items()))

    saved = rebal.load_holdings()
    held = saved.get("holdings", {})
    if saved.get("updated"):
        st.caption(f"마지막으로 저장한 보유 내역: {saved['updated']}")
    codes = list(dict.fromkeys([c for c in tw if c != CASH] + [c for c in held if c in PRODUCTS]))
    extra = st.multiselect("목표에 없지만 들고 있는 상품 추가", [c for c in PRODUCTS if c not in codes], key=init("rb_extra", []), format_func=label)
    codes += extra
    base = pd.DataFrame({"code": codes, "상품": [label(c) for c in codes],
                         "평가금액(원)": [float(held.get(c, {}).get("amount", 0.0)) for c in codes],
                         "현재가(원)": [float(held.get(c, {}).get("price", 0.0)) for c in codes]})
    edited = st.data_editor(base, hide_index=True, width="stretch", key="rb_editor", disabled=["code", "상품"], column_config={
        "code": None, "평가금액(원)": st.column_config.NumberColumn(format="localized", min_value=0, step=10000),
        "현재가(원)": st.column_config.NumberColumn(format="localized", min_value=0, step=5,
                                                 help="한 주 가격. 넣으면 살·팔 주 수를 계산합니다(소수점 아래는 버림).")})
    c1, c2, c3 = st.columns([1.2, 1.6, 1])
    contribution = c1.number_input("이번에 넣을 돈 (원)", 0, 10_000_000_000, step=100_000, key=init("rb_contrib", 0))
    mode = c2.radio("방식", ["buy_only", "full"], key=init("rb_mode", "buy_only"), horizontal=True,
                    format_func={"buy_only": "입금액만으로 맞추기 (매도 없음)", "full": "전체 리밸런싱 (매도 포함)"}.get)
    band = c3.number_input("밴드 (%p)", 0.0, 30.0, step=1.0, key=init("rb_band", 5.0),
                           help="모든 상품이 목표 ± 밴드 안이면 팔지 않고 입금액만 배분합니다.")
    holdings = dict(zip(edited["code"], edited["평가금액(원)"].fillna(0.0).astype(float)))
    prices = dict(zip(edited["code"], edited["현재가(원)"].fillna(0.0).astype(float)))
    target_codes = {c: w for c, w in tw.items() if c != CASH}
    if tw.get(CASH):
        st.caption(f"목표의 현금 {tw[CASH] * 100:.1f}%는 CD금리 ETF(459580)나 예수금으로 두세요. 아래 표는 나머지만 맞춥니다.")
    try:
        orders = rebal.rebalance_orders(holdings, target_codes, float(contribution), mode, float(band), prices)
    except ValueError as exc:
        st.info(str(exc))
        return
    shown = orders.drop(columns=["code", "계산 방식"])
    if shown["주 수"].isna().all():
        shown = shown.drop(columns=["주 수"])
    else:  # blank (not "None") where no price was entered
        shown["주 수"] = [("" if np.isnan(v) else f"{int(v):,}주") for v in shown["주 수"]]
    st.dataframe(shown, hide_index=True, width="stretch", column_config={
        "현재 금액": st.column_config.NumberColumn(format="localized"), "목표 금액": st.column_config.NumberColumn(format="localized"),
        "매매 금액": st.column_config.NumberColumn(format="localized"), "매매 후 금액": st.column_config.NumberColumn(format="localized"),
        "현재 비중": st.column_config.NumberColumn(format="percent"), "목표 비중": st.column_config.NumberColumn(format="percent"),
        "매매 후 비중": st.column_config.NumberColumn(format="percent")})
    buy = orders.loc[orders["매매 금액"] > 0, "매매 금액"].sum()
    sell = abs(float(orders.loc[orders["매매 금액"] < 0, "매매 금액"].sum()))
    left = contribution - (buy - sell)
    st.markdown(f"매수 합계 **{won(buy)}** · 매도 합계 **{won(sell)}** · 남는 현금 **{won(left)}** · 계산 방식: {orders['계산 방식'].iloc[0]}")
    if mode == "full" and orders["계산 방식"].iloc[0] == "입금액만 배분":
        st.caption(f"모든 상품이 목표 ± {band:g}%p 안에 있어 팔지 않고 입금액만 배분했습니다. 전부 목표대로 맞추려면 밴드를 0으로 두세요.")
    if prices and any(v > 0 for v in prices.values()):
        st.caption("주 수는 현재가로 나눈 뒤 소수점 아래를 버린 값이고, 매매 금액도 그 주 수 기준입니다. 남는 현금은 그 차이입니다.")
    if st.button("💾 보유 내역 저장 (다음 분기에 불러옴)"):
        rebal.save_holdings({c: {"amount": float(holdings[c]), "price": float(prices.get(c, 0.0))} for c in holdings if holdings[c] > 0 or prices.get(c)})
        st.success("저장했습니다: lab_results/aa_holdings.json")


# =========================================================================== page 5: notes

def notes_page() -> None:
    st.title("발견 · 경고 · 캠페인")
    st.caption(f"`{notes.NOTES_PATH.relative_to(notes.NOTES_PATH.parents[3])}`에 저장되는 연구 노트입니다. 다른 연구 스레드도 같은 파일을 읽고 씁니다.")
    data = notes.load_notes()
    for key, title in notes.SECTIONS.items():
        st.subheader(title)
        st.caption(notes.SECTION_HELP[key])
        entries = data.get(key, [])
        if not entries:
            st.info("아직 항목이 없습니다.")
        for day, text in entries:
            with st.container(border=True):
                st.markdown(f"<small>{day}</small>  \n{text}", unsafe_allow_html=True)
    st.divider()
    with st.form("add_note", clear_on_submit=True):
        c1, c2 = st.columns([1, 3])
        section = c1.selectbox("구분", list(notes.SECTIONS), format_func=notes.SECTIONS.get)
        text = c2.text_area("내용 (한두 문장, 숫자와 조건을 함께)", height=80)
        if st.form_submit_button("노트 추가"):
            try:
                notes.add_note(section, text)
                st.success("추가했습니다. (Git에 올리려면 커밋이 필요합니다.)")
                st.rerun()
            except ValueError as exc:
                st.warning(str(exc))
    st.subheader("이 연구실에서 시험한 기록")
    runs = notes.load_runs()
    if runs.empty:
        st.info("아직 실행 기록이 없습니다.")
        return
    st.caption(f"{len(runs)}회 실행 · 전략 이름 기준 {runs['name'].nunique()}종 · 첫 기록 {str(runs['time'].iloc[0])[:10]} · 그래프로 비교하려면 ‘전략 비교’ 페이지")
    st.dataframe(runs.drop(columns=["config"]).iloc[::-1], hide_index=True, width="stretch", column_config={
        "time": "시각", "name": "이름", "description": "설정", "period": "기간", "kind": "종류",
        "cagr": st.column_config.NumberColumn("CAGR", format="percent"), "mdd": st.column_config.NumberColumn("MDD", format="percent"),
        "calmar": st.column_config.NumberColumn("칼마", format="%.2f"), "sharpe_excess": st.column_config.NumberColumn("현금 초과 샤프", format="%.2f"),
        "turnover": st.column_config.NumberColumn("회전율", format="percent")})


# =========================================================================== page 6: saved

def saved_page() -> None:
    st.title("저장된 전략")
    items = store.list_saved()
    if not items:
        st.info("저장된 전략이 없습니다. ‘전략 연구’나 ‘성적표·믹스’에서 저장하세요.")
        return
    rows = [{"선택": i < 4, "이름": it["name"], "저장 시각": it["saved_at"], "기간": " → ".join(it["period"]), "설정": it["description"],
             "CAGR": it["metrics"].get("cagr"), "MDD": it["metrics"].get("mdd"), "칼마": it["metrics"].get("calmar"),
             "현금 초과 샤프": it["metrics"].get("sharpe_excess"), "기준": "원화 국내 ETF" if it.get("currency") == "KRW" else "미국 ETF (예전 버전)"}
            for i, it in enumerate(items)]
    df = pd.DataFrame(rows)
    edited = st.data_editor(df, hide_index=True, width="stretch", disabled=[c for c in df.columns if c != "선택"], column_config={
        "CAGR": st.column_config.NumberColumn(format="percent"), "MDD": st.column_config.NumberColumn(format="percent"),
        "칼마": st.column_config.NumberColumn(format="%.2f"), "현금 초과 샤프": st.column_config.NumberColumn(format="%.2f")})
    chosen = [items[i] for i in np.flatnonzero(edited["선택"].to_numpy())]
    if chosen:
        fig = go.Figure()
        for i, it in enumerate(chosen[:8]):
            eq = store.load_equity(it["path"])
            fig.add_scatter(x=eq.index, y=eq["strategy"], name=it["name"], line=dict(color=PALETTE[i % len(PALETTE)], width=2))
        eq0 = store.load_equity(chosen[0]["path"])
        bcol = "benchmark" if "benchmark" in eq0.columns else "SPY"
        fig.add_scatter(x=eq0.index, y=eq0[bcol], name="기준선 (첫 결과 기간)", line=dict(color=GRAY, width=1.5, dash="dot"))
        fig.update_layout(height=380, hovermode="x unified", yaxis_type="log", margin=dict(l=10, r=10, t=30, b=10), title="자산 곡선 비교 (최대 8개)",
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
        st.caption("저장할 때의 기간 그대로입니다. 같은 기간으로 다시 돌려 보려면 ‘전략 비교’ 페이지를 쓰세요.")


# =========================================================================== page 7: glossary and products

def glossary_page() -> None:
    st.title("용어 · 상품 설명")
    st.subheader("국내 상장 ETF (ISA 매수 가능)")
    df = pd.DataFrame([{"분류": p.asset_class, "상품": p.name, "종목코드": c, "환": FX_LABELS[p.fx], "배수": p.leverage, "총보수(%)": p.ter,
                        "상장일": p.listed, "상장 전 대용": p.proxy_text, "메모": p.note} for c, p in PRODUCTS.items()])
    st.dataframe(df, hide_index=True, width="stretch", column_config={"배수": st.column_config.NumberColumn(format="%.0fx"),
                                                                      "총보수(%)": st.column_config.NumberColumn(format="%.4g")})
    st.caption("종목코드·총보수·상장일·환헤지 여부는 2026-10-24에 etfshopping.com, funetf.co.kr, 운용사 페이지에서 확인했습니다. 총보수 외 실부담비용은 더 클 수 있습니다.")
    st.subheader("원화 수익률을 만드는 방법")
    st.markdown(BR.join([
        "상장 전 기간까지 길게 보려고 각 상품을 **미국 ETF(배당 재투자, 달러)**로 대신하고, 상품이 실제로 돈을 버는 방식대로 원화 수익률로 바꿉니다.",
        "· **환노출**: (1 + 미국 ETF 수익률) × (1 + 원/달러 변화) − 1. 원화가 약해지면(환율 상승) 이득.",
        "· **환헤지(H)**: 미국 ETF 수익률 − 달러 단기금리 + 원화 단기금리 − 헤지 추가 비용(왼쪽 사이드바, 기본 20bp/년). 환율 변화 대신 두 나라 금리 차이를 받습니다.",
        "· **국내 주식(코스피 200)**: EWY(MSCI 한국, 달러)를 원화로 환산.",
        "· **CD금리 ETF**: 한국은행 기준금리. **달러선물 ETF**: 원/달러 변화 + 달러 단기금리.",
        "· **레버리지 ETF**: 하루 단위로 (기초 수익률 − 조달 통화 금리) × 배수 + 금리. 변동성 손실이 자연히 들어갑니다.",
        "· 마지막에 각 상품의 총보수(와 알려진 차이, 예: 배당 제외 지수)를 뺍니다."]))
    st.markdown(BR.join([
        "**데이터**",
        "· 원/달러: 미국 연준 H.10 정오 환율(FRED DEXKOUS, 1993년부터, 저장소에 스냅샷). 인터넷이 되면 Yahoo의 최근 값을 이어 붙입니다.",
        f"· 원화 단기금리: 한국은행 기준금리 일정표(`{BOK_PATH.name}`). 2018-11 이후는 두 공개 사이트와 대조했고, 그 전은 확인이 더 필요합니다.",
        "· 달러 단기금리: BIL(초단기 미국채 ETF), 2007년 이전은 13주 국채금리."]))
    st.markdown(BR.join([
        "**알아 둘 한계**",
        "· 실제 국내 ETF의 추적오차·괴리율, 국내 금 프리미엄, 분배금 과세 시점은 들어 있지 않습니다.",
        "· 미국 시장이 닫힌 날의 국내 가격 변화(시차)는 무시합니다. 일별 숫자보다 월·연 단위 결과를 보세요.",
        "· 세금은 빠져 있습니다. ISA는 계좌 안 손익을 합쳐 200만원(서민형 400만원)까지 비과세, 초과분 9.9%."]))
    st.subheader("성적표의 아홉 축")
    st.markdown("점수는 비교 집합 안의 **백분위 순위**(0 = 가장 아쉬움, 100 = 가장 나음, 동점은 평균 순위)입니다. 집합이 바뀌면 점수도 바뀝니다. "
                "‘견고’는 축 점수의 가중 평균으로 참고용이며, 판정은 Gate로만 합니다.")
    for a in AXES:
        st.markdown(BR.join([f"**{a}**", AXIS_TEXT[a]]))
    st.subheader("Gate")
    st.markdown(BR.join([f"**탐색 Gate** · 칼마 ≥ {BENCH} 칼마 그리고 |MDD| ≤ 0.75 × |{BENCH} MDD|", "수익 조건 없이 ‘위험 대비’로만 보는 넓은 문.",
                         f"**최종 Gate** · {gate_text()}",
                         "2026-10-25 재설정(DECISIONS A26): 이전에는 CAGR ≥ 0.80 × 그리고 |MDD| ≤ 0.75 × |벤치마크 MDD|(원화로는 약 −22%)였습니다. "
                         "사용자 목표(낙폭 −15% 선, HAA+BAA 수준)에 맞춰 MDD는 절대 기준으로, CAGR 기준은 0.7배로 낮췄습니다.",
                         "**날짜 견고(참고)** · 리밸런싱 날짜를 말일·+1·+5·+10·+15거래일로 바꿔도 최대 낙폭이 −20% 이내인지. ‘전술 전략·검증’ 페이지에서 계산합니다.",
                         f"{BENCH} = KODEX 미국S&P500(환노출)의 원화 수익률. 원화로 보면 2008년처럼 원화가 약해질 때 달러 자산이 덜 빠져, 미국 기준 SPY보다 낙폭이 작습니다."]))
    st.subheader("위기 구간 (성적표 ‘위기’ 축)")
    st.dataframe(pd.DataFrame([{"구간": k, "시작": a, "끝": b} for k, (a, b) in CRISIS_WINDOWS.items()]), hide_index=True, width="stretch")
    st.caption("미국 S&P 500의 고점→저점 날짜로 2026-10-24에 등록했고, 결과를 보고 고치지 않습니다.")
    st.subheader("기본 지표 (추세추종 연구실과 같은 계산)")
    for k in ("cagr", "mdd", "calmar", "volatility", "worst_12m", "underwater_days"):
        d = DEFINITIONS[k]
        st.markdown(BR.join([f"**{d.label}**", d.short, f"`{d.formula}`"]))
    st.markdown(BR.join(["**현금 초과 샤프**", "일간 수익률에서 원화 현금(기준금리) 수익률을 뺀 뒤 계산한 샤프. 현금 비중이 큰 배분을 과대평가하지 않으려고 씁니다.",
                         "`mean(r − 현금) / std(r − 현금) × √252`"]))
    st.subheader("overlay")
    for v in OVERLAY_LABELS.values():
        st.markdown(f"· **{v}**")
    st.subheader("실행 규칙")
    st.markdown(RULES_KO)
    st.subheader("프리셋")
    st.dataframe(pd.DataFrame([{"이름": p.name, "비중": " · ".join(f"{short(s)} {v:g}" for s, v in p.weights.items()), "설명": p.note}
                               for p in PRESETS.values()]), hide_index=True, width="stretch")


# =========================================================================== page 8: tactical strategies and validation

TACTICAL_CHOICES = {
    "HAA + BAA 공격형 50:50": (("haa", 50.0), ("baa_g4", 50.0)),
    "HAA": (("haa", 100.0),),
    "BAA 공격형 (G4)": (("baa_g4", 100.0),),
    "VAA 공격형 (G4)": (("vaa_g4", 100.0),),
}
US_CLASS_COLORS = {"미국 주식": "#e34948", "해외·신흥 주식": "#eda100", "리츠": "#7a5c2e", "원자재": "#008300", "금": "#d4b106",
                   "채권": "#2a78d6", "현금성": "#8a8984"}
OFFSET_LABELS = {0: "말일", 1: "다음 달 첫 거래일 (말일 +1)", 5: "말일 +5거래일", 10: "말일 +10거래일 (중순)", 15: "말일 +15거래일"}


@st.cache_resource(show_spinner="전술 전략 신호용 미국 ETF 가격을 불러오는 중…")
def load_signal_data(day: str):
    syms = list(tac.ALL_SYMBOLS)  # VIX comes from lab/macro_snapshot (its Yahoo series has bars on US holidays)
    try:
        usd, failed = download_panel(syms, day)
    except Exception as exc:  # offline: cached files only
        from lab.data import CACHE_DIR
        cached = [s for s in syms if (CACHE_DIR / "yf" / f"{s.replace('^', '_')}.csv.gz").exists()]
        if not cached:
            raise
        last = min(pd.read_csv(CACHE_DIR / "yf" / f"{s.replace('^', '_')}.csv.gz", usecols=["date"])["date"].max() for s in cached)
        usd, failed = download_panel(cached, str(last))
        failed = list(failed) + [f"(오프라인: {exc})"]
    return usd, failed


def tactical_spec_from_ui(last: date) -> tuple[tac.TacticalSpec, str] | None:
    c1, c2, c3 = st.columns([2, 2, 2])
    choice = c1.selectbox("전략", list(TACTICAL_CHOICES) + [CUSTOM], key=init("tac_choice", "HAA + BAA 공격형 50:50"))
    if choice == CUSTOM:
        cols = st.columns(len(tac.STRATEGIES))
        parts = []
        for col, (k, name) in zip(cols, tac.STRATEGIES.items()):
            w = col.number_input(name, 0, 100, step=10, key=init(f"tac_w_{k}", 50 if k in ("haa", "baa_g4") else 0))
            if w:
                parts.append((k, float(w)))
        if not parts:
            st.info("비중을 하나 이상 넣어 주세요.")
            return None
        parts = tuple(parts)
    else:
        parts = TACTICAL_CHOICES[choice]
    holdings = c2.radio("보유 상품", ["krw", "usd"], key=init("tac_hold", "krw"), horizontal=True,
                        format_func=lambda v: "국내 ETF · 원화 (ISA)" if v == "krw" else "미국 ETF · 달러 (논문·지인 도구 재현)")
    missing_to = c3.selectbox("국내 대체가 없는 자산(DBC 등)을 담을 곳", ["459580", "411060"], key=init("tac_missing", "459580"),
                              format_func=short, disabled=holdings == "usd")
    d1, d2, d3, d4, d5 = st.columns(5)
    start = d1.date_input("시작일", key=init("tac_start", date(2008, 1, 1)), min_value=date(2007, 3, 1), max_value=last)
    end = d2.date_input("끝", key=init("tac_end", last), min_value=date(2008, 1, 1), max_value=last)
    offset = d3.selectbox("판단·매매일", list(OFFSET_LABELS), key=init("tac_offset", 0), format_func=OFFSET_LABELS.get)
    same = d4.toggle("판단일 종가에 바로 매매 (재현용)", key=init("tac_same", False),
                     help="논문·지인 도구처럼 월말 종가로 신호를 보고 같은 종가에 산다고 가정합니다. 실제로는 불가능해서 기본은 꺼짐(다음 날 종가).")
    cost = d5.number_input("거래 비용 (bp, 한쪽)", 0.0, 100.0, step=5.0, key=init("tac_cost", 10.0))
    name = " + ".join(f"{tac.STRATEGIES[k].split(' (')[0]} {w:g}" for k, w in parts)
    spec = tac.TacticalSpec(name, parts=parts, start=str(start), end=str(end), cost_bps=float(cost), offset=int(offset),
                            same_close=bool(same), missing_to=missing_to)
    return spec, holdings


def run_tactical_ui(data: dict, usd, spec: tac.TacticalSpec, holdings: str) -> AllocResult:
    hold_panel = data["panel"] if holdings == "krw" else usd
    return tac.run_tactical(hold_panel, usd.close, spec, holdings, usd.cash)


def class_area_chart(signals: pd.DataFrame) -> go.Figure:
    cls = tac.class_table(signals)
    fig = go.Figure()
    for c in cls.columns:
        fig.add_scatter(x=cls.index, y=cls[c], name=c, stackgroup="one", line=dict(width=0.5, color=US_CLASS_COLORS.get(c, GRAY)))
    fig.update_layout(height=300, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10), title="시기별 목표 자산 구성 (매월 신호, 미국 ETF 기준 분류)",
                      yaxis_tickformat=".0%", legend=dict(orientation="h", y=-0.2))
    return fig


def monthly_heatmap(equity: pd.Series) -> go.Figure:
    t = monthly_table(equity)
    z = t.to_numpy() * 100
    text = [["" if not np.isfinite(v) else f"{v:.1f}" for v in row] for row in z]
    fig = go.Figure(go.Heatmap(z=z, x=[f"{m}월" for m in range(1, 13)] + ["연간"], y=[str(y) for y in t.index], text=text,
                               texttemplate="%{text}", colorscale=RED_GREEN, zmid=0, zmin=-8, zmax=8, showscale=False,
                               hovertemplate="%{y} %{x}: %{z:.1f}%<extra></extra>"))
    fig.update_layout(height=max(260, 22 * len(t) + 60), margin=dict(l=10, r=10, t=30, b=10), title="월별 수익률 (%, 비용 차감 후)",
                      yaxis=dict(autorange="reversed", type="category"))
    return fig


def mapping_table(signals: pd.DataFrame) -> pd.DataFrame:
    last = signals.iloc[-1]
    rows = []
    for s, w in last[last > 0].sort_values(ascending=False).items():
        code, quality, note = tac.KRW_MAP.get(s, (None, "없음", ""))
        rows.append({"미국 ETF": s, "비중": float(w), "국내 대체": label(code) if code else "없음", "일치": quality, "설명": note})
    return pd.DataFrame(rows)


def verdict_lines(timing: pd.DataFrame, split: pd.DataFrame, boot: pd.DataFrame | None, lazy: pd.DataFrame | None) -> list[str]:
    ts = timing_summary(timing)
    out = []
    ok = ts["mdd_worst"] >= -0.20
    out.append(("✅" if ok else "⚠️") + f" 날짜 견고: 판단일을 바꾼 최악 MDD {pct(ts['mdd_worst'])} (말일 {pct(ts['mdd_base'])}), "
               f"CAGR {pct(ts['cagr_min'])} ~ {pct(ts['cagr_max'])}" + ("" if ok else " → 숫자가 날짜 운에 기대고 있음"))
    if "delay" in timing and (timing["delay"] == 1).any():
        d = timing[timing["delay"] == 1].iloc[0]
        out.append(f"• 한 달 늦게 따라가면: CAGR {pct(d['cagr'])} · MDD {pct(d['mdd'])}")
    if lazy is not None:
        out.append(f"• 게으른 리밸런싱(50회): CAGR 중앙 {pct(lazy['cagr'].median())} · MDD 중앙 {pct(lazy['mdd'].median())} · MDD 하위 10% {pct(lazy['mdd'].quantile(0.1))}")
    sel, hold = split.iloc[0], split.iloc[1]
    ok = hold["sharpe"] >= 0.5 * sel["sharpe"] if np.isfinite(sel["sharpe"]) else False
    out.append(("✅" if ok else "⚠️") + f" 성역(마지막 15%): 선정 구간 샤프 {num(sel['sharpe'])} → 성역 {num(hold['sharpe'])}, "
               f"성역 CAGR {pct(hold['cagr'])} · MDD {pct(hold['mdd'])}")
    if boot is not None:
        out.append(f"• 부트스트랩 90% 구간: 샤프 {num(boot['샤프'].iloc[0])} ~ {num(boot['샤프'].iloc[2])}, MDD {pct(boot['MDD'].iloc[0])} ~ {pct(boot['MDD'].iloc[2])}")
    return out


def tactical_page() -> None:
    data = data_or_stop()
    sidebar(data)
    st.title("전술 전략 · 검증")
    st.caption("Keller 계열 전술 배분(HAA·BAA·VAA)을 **미국 ETF 가격으로 신호를 보고** 국내 ETF(원화) 또는 미국 ETF(달러)로 담아 시험합니다. "
               "아래 검증 도구(날짜 민감도·게으른 리밸런싱·성역·부트스트랩·정직 vs 균등·위기 회피 스위치)는 프리셋에도 씁니다. DECISIONS A24–A29.")
    try:
        usd, failed = load_signal_data(str(date.today()))
    except Exception as exc:
        st.error(f"미국 ETF 가격을 불러오지 못했습니다: {exc}")
        return
    if failed:
        st.caption("받지 못한 종목: " + ", ".join(map(str, failed)))
    last = min(data["panel"].last_date.date(), usd.last_date.date())
    tab1, tab2, tab3, tab4 = st.tabs(["① 전술 전략", "② 검증", "③ 정직 vs 균등", "④ 위기 회피 스위치"])

    with tab1:
        with st.expander("전략 규칙"):
            for k, v in tac.STRATEGY_TEXT.items():
                st.markdown(f"**{tac.STRATEGIES[k]}** · {v}")
            st.caption("모멘텀: 13612U = 1·3·6·12개월 수익률의 평균, 13612W = (12·1개월 + 4·3개월 + 2·6개월 + 12개월) / 4, "
                       "SMA12 = 현재가 / 최근 13개 월말가 평균 − 1. VEA·BND·BIL은 상장 전 기간을 EFA·AGG·미국 단기금리로 잇습니다.")
        got = tactical_spec_from_ui(last)
        if got is None:
            return
        spec, holdings = got
        if st.button("▶ 실행", type="primary", key="btn_tac_run"):
            try:
                with st.spinner("계산 중…"):
                    res = run_tactical_ui(data, usd, spec, holdings)
                    st.session_state["tac_result"] = (res, raw_metrics(res, spec.parameter_count()), holdings)
            except ValueError as exc:
                st.error(str(exc))
        stored = st.session_state.get("tac_result")
        if stored:
            res, m, hold = stored
            if res.spec != spec or hold != holdings:
                st.warning("아래 결과는 현재 설정과 다릅니다. 다시 실행하면 갱신됩니다.")
            s, e = res.period
            bench = BENCH if hold == "krw" else "SPY(달러)"
            st.caption(f"{res.spec.describe()} · {'국내 ETF 원화' if hold == 'krw' else '미국 ETF 달러'} · {s.date()} → {e.date()} · "
                       f"매매 {res.n_trades}회 · 월간 샤프(무위험 0, 지인 도구와 같은 방식) {num(tac.monthly_sharpe(res.equity))}")
            kpi_row(m)
            if hold == "krw":
                gates_line(m)
            else:
                st.caption(f"달러 결과는 원화 Gate와 비교하지 않습니다. 기준선은 {bench}.")
            c1, c2 = st.columns([3, 2])
            c1.plotly_chart(equity_chart(res), width="stretch")
            c2.plotly_chart(drawdown_chart(res), width="stretch")
            c3, c4 = st.columns(2)
            c3.plotly_chart(class_area_chart(res.signals.loc[s - pd.offsets.MonthEnd(1):]), width="stretch")
            c4.plotly_chart(monthly_heatmap(res.equity), width="stretch")
            st.markdown(f"**이번 달 목표({res.signals.index[-1].date()} 신호)와 국내 대체**")
            st.dataframe(mapping_table(res.signals), hide_index=True, width="stretch",
                         column_config={"비중": st.column_config.NumberColumn(format="percent")})
            cov = tac.krw_coverage(res.signals.loc[s - pd.offsets.MonthEnd(1):])
            st.caption(f"기간 평균으로 목표 비중의 {pct(cov['정확'].mean(), 0)}는 같은 지수·같은 환노출 국내 상품, {pct(cov['근사'].mean(), 0)}는 비슷한 상품"
                       f"(환헤지·지수 차이), {pct(cov['없음'].mean(), 0)}는 국내 상품이 없어 ‘{short(spec.missing_to)}’로 대체했습니다. "
                       "국내 대체의 차이는 원화 결과에 이미 들어 있습니다.")

    with tab2:
        st.markdown("결과 숫자 하나가 **운**인지 보려고, 같은 전략을 조건만 바꿔 다시 돌립니다. 지인 도구의 ‘실전 이탈 리스크’·‘진짜 성역’·‘부트스트랩’에 해당합니다.")
        targets = ["현재 전술 전략 설정"] + [p.name for p in PRESETS.values()]
        target = st.selectbox("검증할 전략", targets, key=init("val_target", targets[0]))
        c1, c2 = st.columns(2)
        do_boot = c1.toggle("부트스트랩 (1000회, 수십 초)", value=True, key=init("val_boot", True))
        do_lazy = c2.toggle("게으른 리밸런싱 (50회, 전술 전략만)", value=True, key=init("val_lazy", True))
        if st.button("▶ 전체 검증 실행", type="primary", key="btn_val_run"):
            got = None
            try:
                with st.spinner("여러 번 다시 돌리는 중…"):
                    if target == targets[0]:
                        spec, holdings = (tactical_spec_from_state(last))
                        base = run_tactical_ui(data, usd, spec, holdings)
                        timing = timing_table(lambda o, d: run_tactical_ui(data, usd, replace(spec, offset=o, delay=d), holdings))
                        lazy = None
                        if do_lazy:
                            months = [str(d.to_period("M")) for d in base.signals.index]
                            lazy = lazy_table(lambda skip: run_tactical_ui(data, usd, replace(spec, skip_months=skip), holdings), months)
                        loo = leave_one_out(list(spec.parts), lambda parts: run_tactical_ui(data, usd, replace(spec, parts=tuple(parts)), holdings)) \
                            if len(spec.parts) > 1 else None
                    else:
                        preset = next(p for p in PRESETS.values() if p.name == target)
                        cfg = AllocConfig(preset.name, preset.weights, DEFAULT_START, str(last), rebalance="Q", cost_bps=10.0)
                        base = run_allocation(data["panel"], cfg)
                        timing = timing_table(lambda o, d: run_allocation(data["panel"], replace(cfg, rebalance_offset=o)), delays=(0,))
                        lazy, loo = None, None
                    split = split_metrics(base)
                    boot = block_bootstrap(base) if do_boot else None
                    got = (target, base, timing, lazy, loo, split, boot)
            except ValueError as exc:
                st.error(str(exc))
            if got:
                st.session_state["val_result"] = got
        got = st.session_state.get("val_result")
        if got:
            target_done, base, timing, lazy, loo, split, boot = got
            st.markdown(f"#### {target_done}")
            for line_ in verdict_lines(timing, split, boot, lazy):
                st.markdown(line_)
            fmt = {"cagr": st.column_config.NumberColumn("CAGR", format="percent"), "mdd": st.column_config.NumberColumn("MDD", format="percent"),
                   "sharpe": st.column_config.NumberColumn("현금 초과 샤프", format="%.2f"), "calmar": st.column_config.NumberColumn("칼마", format="%.2f")}
            a, b = st.columns(2)
            a.markdown("**리밸런싱 날짜 · 지연**")
            a.dataframe(timing.drop(columns=["offset", "delay"]), hide_index=True, width="stretch", column_config=fmt)
            b.markdown("**선정 구간 / 성역 / 전체**")
            b.dataframe(split, width="stretch", column_config=fmt)
            if boot is not None:
                a.markdown("**블록 부트스트랩 (21거래일 블록, 1000회) 백분위**")
                a.dataframe(boot, width="stretch", column_config={"CAGR": st.column_config.NumberColumn(format="percent"),
                                                                  "MDD": st.column_config.NumberColumn(format="percent"),
                                                                  "샤프": st.column_config.NumberColumn(format="%.2f")})
            if loo is not None:
                b.markdown("**하나씩 빼 보기 (잭나이프)**")
                b.dataframe(loo.replace(tac.STRATEGIES), hide_index=True, width="stretch", column_config=fmt)
            if lazy is not None:
                st.caption(f"게으른 리밸런싱: 매매 직후 다음 매매가 1·2·3개월 뒤일 확률 {', '.join(f'{p:.0%}' for p in LAZY_PROBS)}로 50번(시드 고정). "
                           "건너뛴 달은 비중이 그대로 흘러갑니다.")

    with tab3:
        st.markdown("여러 전략을 섞을 때 과거로 맞춘 비중(**정직**)이 같은 비율(**균등**)보다 나은지 봅니다. 기간을 10토막으로 나눠 5토막에서 비중을 고르고 "
                    "나머지 5토막에서 채점하기를 252가지(10C5)로 반복합니다(CSCV). 마지막 15%(성역)는 쓰지 않습니다. "
                    "섞기는 일간 가중 평균(매일 비중 유지)으로 근사합니다.")
        options = list(TACTICAL_CHOICES)[1:] + [p.name for p in PRESETS.values()]
        chosen = st.multiselect("후보 전략 (원화)", options, key=init("cv_set", ["HAA", "BAA 공격형 (G4)", "VAA 공격형 (G4)", "안정형 40 (금·현금 분산)",
                                                                       "내 AAA (2026-06 이후)", "영구 포트폴리오", "60/40", "S&P500 보유"]))
        c1, c2 = st.columns(2)
        max_k = c1.selectbox("한 조합에 담을 전략 수 최대", [2, 3], index=1, key=init("cv_k", 3))
        step = c2.selectbox("비중 격자", [0.1, 0.2, 0.25], key=init("cv_step", 0.1), format_func=lambda v: f"{v:.0%}")
        if st.button("▶ 252분할 검증", type="primary", key="btn_cv_run") and len(chosen) >= 2:
            with st.spinner("후보 조합을 252번 채점하는 중…"):
                eqs = {}
                for n in chosen:
                    if n in TACTICAL_CHOICES:
                        sp = tac.TacticalSpec(n, parts=TACTICAL_CHOICES[n], start="2007-03-01", end=str(last))
                        eqs[n] = run_tactical_ui(data, usd, sp, "krw").equity
                    else:
                        preset = next(p for p in PRESETS.values() if p.name == n)
                        eqs[n] = run_allocation(data["panel"], AllocConfig(n, preset.weights, "2007-03-01", str(last), rebalance="Q", cost_bps=10.0)).equity
                eq = pd.DataFrame(eqs).dropna()
                cut = holdout_split(eq.index)
                rets = eq.loc[:cut].pct_change(fill_method=None).dropna()
                st.session_state["cv_result"] = (cscv(rets, data["panel"].cash, step=float(step), max_k=int(max_k)), eq.index[0], cut)
        cv = st.session_state.get("cv_result")
        if cv:
            out, a, cut = cv
            ok = out["win_rate"] > 0.6 and out["pbo"] < 0.3
            st.markdown(("✅" if ok else "⚠️") + f" 튜닝한 비중이 균등을 이긴 분할 **{out['win_rate']:.0%}** · PBO(튜닝 1등이 표본밖에서 중간 아래로 떨어질 확률) **{out['pbo']:.0%}** "
                        f"· 후보 {out['candidates']:,}개 · 구간 {a.date()} – {cut.date()}")
            st.markdown(f"표본밖 샤프 중앙값: 튜닝 **{num(out['tuned_out_median'])}** · 균등 **{num(out['equal_out_median'])}** · 천장(답을 본 최고) {num(out['ceiling_out_median'])}")
            st.caption("승률이 50% 근처이고 PBO가 50% 근처면 ‘비중 튜닝은 과거 암기’입니다. 이 경우 고른 전략들을 같은 비율로 담는 쪽이 정직한 선택입니다.")
            st.dataframe(out["top_picks"].rename("뽑힌 분할 수").rename_axis("자주 뽑힌 튜닝 비중").to_frame(), width="stretch")

    with tab4:
        st.markdown("정적 배분의 **주식 상품**을 신호가 켜진 달에만 안전 상품으로 옮깁니다(월말 판단, 다음 날 종가 매매, 나머지는 분기 리밸런싱). "
                    "‘사후 정답’은 위기 구간을 미리 안다고 가정한 상한선이라 전략이 아닙니다.")
        c1, c2, c3 = st.columns(3)
        pname = c1.selectbox("배분", [p.name for p in PRESETS.values()], key=init("sw_preset", PRESETS["stable40"].name))
        safe = c2.selectbox("회피처", ["459580", "261240", "0091C0"], key=init("sw_safe", "459580"), format_func=short)
        sigs = c3.multiselect("신호", list(sig.SIGNALS), key=init("sw_sigs", list(sig.SIGNALS)), format_func=sig.SIGNALS.get)
        if st.button("▶ 스위치 비교", type="primary", key="btn_sw_run"):
            preset = next(p for p in PRESETS.values() if p.name == pname)
            cfg = AllocConfig(preset.name, preset.weights, DEFAULT_START, str(last), rebalance="Q", cost_bps=10.0)
            equity_codes = [c for c, p in PRODUCTS.items() if p.asset_class == "주식"]
            dates = data["panel"].close.index
            rows = [("스위치 없음", run_allocation(data["panel"], cfg))]
            with st.spinner("신호별로 다시 돌리는 중…"):
                for k in sigs:
                    flags = sig.signal_series(k, usd.close, usd.cash, dates)
                    rows.append((sig.SIGNALS[k], sig.run_switch(data["panel"], cfg, flags, equity_codes, safe)))
            table = []
            for name_, r in rows:
                h = headline(r)
                row = {"신호": name_, "CAGR": h["cagr"], "MDD": h["mdd"], "샤프": h["sharpe"]}
                for w, (a, b) in sig.ORACLE_WINDOWS.items():
                    part = r.equity.loc[a:b]
                    row[f"{w} 낙폭"] = float((part / part.cummax() - 1).min()) if len(part) > 5 else np.nan
                late = r.equity.loc["2015-01-02":]
                row["2015+ CAGR"] = headline(late, r.cash)["cagr"]
                row["회피 달 비율"] = float(r.risk_off.mean()) if hasattr(r, "risk_off") else 0.0
                table.append(row)
            st.session_state["sw_result"] = (pname, pd.DataFrame(table), rows)
        got = st.session_state.get("sw_result")
        if got:
            pname_done, table, rows = got
            st.markdown(f"#### {pname_done}")
            st.dataframe(table, hide_index=True, width="stretch", column_config={
                c: st.column_config.NumberColumn(format="%.2f" if c == "샤프" else "percent") for c in table.columns if c != "신호"})
            pick = st.selectbox("그래프로 볼 신호", [n for n, _ in rows[1:]], key="sw_pick") if len(rows) > 1 else None
            if pick:
                r0, r1 = rows[0][1], dict(rows)[pick]
                fig = go.Figure()
                fig.add_scatter(x=r0.equity.index, y=r0.equity, name="스위치 없음", line=dict(color=GRAY, width=1.5))
                fig.add_scatter(x=r1.equity.index, y=r1.equity, name=pick, line=dict(color=BLUE, width=2))
                off = r1.risk_off.astype(int)
                fig.add_scatter(x=off.index, y=off * r1.equity.max(), name="회피 중", fill="tozeroy", line=dict(width=0),
                                fillcolor="rgba(227,73,72,0.12)", hoverinfo="skip")
                fig.update_layout(height=380, yaxis_type="log", hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10),
                                  title="자산 곡선과 회피 구간(붉은 배경)", legend=dict(orientation="h", y=-0.15))
                st.plotly_chart(fig, width="stretch")
            st.caption("신호 목록은 결과를 보기 전에 정했습니다(DECISIONS A29). 2022년을 잘 피한 TIP 카나리아는 2023년에 발표된 규칙이라 2022년이 표본 안입니다.")


def tactical_spec_from_state(last: date) -> tuple[tac.TacticalSpec, str]:
    """The tactical settings currently in the ① tab (session state), for the validation tab."""
    ss = st.session_state
    choice = ss.get("tac_choice", "HAA + BAA 공격형 50:50")
    if choice == CUSTOM:
        parts = tuple((k, float(ss.get(f"tac_w_{k}", 0))) for k in tac.STRATEGIES if ss.get(f"tac_w_{k}", 0))
    else:
        parts = TACTICAL_CHOICES[choice]
    holdings = ss.get("tac_hold", "krw")
    spec = tac.TacticalSpec(choice, parts=parts, start=str(ss.get("tac_start", date(2008, 1, 1))), end=str(ss.get("tac_end", last)),
                            cost_bps=float(ss.get("tac_cost", 10.0)), offset=0, same_close=bool(ss.get("tac_same", False)),
                            missing_to=ss.get("tac_missing", "459580"))
    return spec, holdings


# =========================================================================== navigation

PERSIST_PREFIXES = ("r_", "sc_", "mix_", "cmp_", "rb_", "m_", "tac_", "val_", "cv_", "sw_")
for _k in list(st.session_state.keys()):
    if str(_k).startswith(PERSIST_PREFIXES) and "_editor" not in str(_k):
        st.session_state[_k] = st.session_state[_k]

nav = st.navigation([
    st.Page(research_page, title="전략 연구", icon="🧺", default=True),
    st.Page(compare_page, title="전략 비교", icon="📊"),
    st.Page(scorecard_page, title="성적표 · 믹스", icon="🏁"),
    st.Page(tactical_page, title="전술 전략 · 검증", icon="🧭"),
    st.Page(rebalance_page, title="리밸런싱 계산기", icon="🧮"),
    st.Page(notes_page, title="발견 · 경고 · 캠페인", icon="📝"),
    st.Page(saved_page, title="저장된 전략", icon="💾"),
    st.Page(glossary_page, title="용어 · 상품 설명", icon="📖"),
])
nav.run()
