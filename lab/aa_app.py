"""자산배분 연구실 — Streamlit app (asset allocation, overlays, leverage, strategy mixes, scorecard).

Run:  .venv\\Scripts\\python.exe -m streamlit run lab/aa_app.py --server.port 8511
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

from lab import aa_notes as notes  # noqa: E402
from lab import aa_store as store  # noqa: E402
from lab.aa_assets import ASSETS, CASH, CLASS_ORDER, DEFAULT_ASSETS, PRESETS, label  # noqa: E402
from lab.aa_engine import (  # noqa: E402
    OVERLAY_LABELS, OVERLAY_PARAM, REBALANCE_LABELS, RULES_KO, AllocConfig, AllocResult, Leverage, MixConfig, Overlay,
    mix_parameter_count, run_allocation, run_mix,
)
from lab.aa_score import (  # noqa: E402
    AXES, AXIS_TEXT, CRISIS_WINDOWS, RAW_COLUMNS, AllocGates, axis_raw_table, crisis_table, gate_results, raw_metrics,
    scorecard,
)
from lab.data import download_panel  # noqa: E402
from lab.metrics import DEFINITIONS, drawdown, fmt_value  # noqa: E402

st.set_page_config(page_title="자산배분 연구실", page_icon="🧺", layout="wide")

BLUE, ORANGE, GRAY, AQUA = "#2a78d6", "#eb6834", "#8a8984", "#1baf7a"
PALETTE = [BLUE, AQUA, "#eda100", "#e87ba4", "#4a3aa7", "#008300", "#e34948", "#7a5c2e"]
RED_GREEN = [[0, "#e34948"], [0.5, "#f6f1e8"], [1, "#1baf7a"]]
BR = "  " + chr(10)
DEFAULT_START = "2005-01-03"
ALL_SYMBOLS = list(ASSETS)


def init(key: str, value):
    if key not in st.session_state:
        st.session_state[key] = value
    return key


def pct(v: float, digits: int = 1) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v * 100:.{digits}f}%"


def num(v: float, digits: int = 2) -> str:
    return "–" if v is None or not np.isfinite(v) else f"{v:.{digits}f}"


# =========================================================================== data

@st.cache_resource(show_spinner="가격 데이터를 불러오는 중… (처음 한 번은 1분 정도, 이후에는 저장된 파일 사용)")
def load_panel(day: str):
    try:
        panel, failed = download_panel(ALL_SYMBOLS, day)
    except Exception as exc:  # offline: fall back to whatever is cached
        from lab.data import CACHE_DIR
        cached = sorted((CACHE_DIR / "yf").glob("*.csv.gz"))
        if not cached:
            raise
        last = min(pd.read_csv(p, usecols=["date"])["date"].max() for p in cached)  # every file reaches this date
        panel, failed = download_panel(ALL_SYMBOLS, str(last))
        failed = list(failed) + [f"(오프라인: {exc})"]
    return panel, failed


def panel_or_stop():
    try:
        panel, failed = load_panel(str(date.today()))
    except Exception as exc:
        st.error(f"가격 데이터를 불러오지 못했습니다: {exc}")
        st.stop()
    if failed:
        st.caption("받지 못한 티커: " + ", ".join(map(str, failed)))
    return panel


# =========================================================================== strategy builder

def apply_preset(prefix: str, key: str) -> None:
    p = PRESETS[key]
    st.session_state[f"{prefix}_assets"] = [s for s in p.weights if s != CASH]
    for s in ALL_SYMBOLS + [CASH]:
        st.session_state[f"{prefix}_w_{s}"] = float(p.weights.get(s, 0.0))
    st.session_state[f"{prefix}_name"] = p.name


def weights_section(prefix: str) -> dict[str, float]:
    c1, c2 = st.columns([1.2, 3])
    preset = c1.selectbox("프리셋 불러오기", ["(직접 입력)"] + list(PRESETS), key=init(f"{prefix}_preset", "aaa_2026"),
                          format_func=lambda k: k if k == "(직접 입력)" else PRESETS[k].name)
    if preset != "(직접 입력)" and st.session_state.get(f"{prefix}_preset_applied") != preset:
        apply_preset(prefix, preset)
        st.session_state[f"{prefix}_preset_applied"] = preset
    if preset != "(직접 입력)":
        c2.caption(PRESETS[preset].note)
    assets = st.multiselect("자산 (미국 상장 ETF 대용)", ALL_SYMBOLS, key=init(f"{prefix}_assets", list(DEFAULT_ASSETS)),
                            format_func=label, help="\n\n".join(f"**{s}** {a.label} · 국내 상품: {a.korea}" for s, a in ASSETS.items()))
    if not assets:
        st.warning("자산을 하나 이상 고르세요.")
        return {}
    cols = st.columns(min(len(assets) + 1, 7))
    weights: dict[str, float] = {}
    for i, s in enumerate(assets + [CASH]):
        key = init(f"{prefix}_w_{s}", 0.0)
        weights[s] = cols[i % len(cols)].number_input(f"{s} (%)" if s != CASH else "현금 (%)", 0.0, 100.0, step=5.0, key=key,
                                                      help=label(s))
    total = sum(weights.values())
    if abs(total - 100) > 0.01:
        st.caption(f"비중 합 {total:g}% → 실행 시 100%로 정규화합니다.")
    return {s: v for s, v in weights.items() if v > 0}


def rules_section(prefix: str, last: date) -> tuple[dict, Overlay, Leverage, str, str]:
    c1, c2, c3, c4, c5 = st.columns([1.3, 1, 1, 1.2, 1.2])
    rebalance = c1.selectbox("리밸런싱", list(REBALANCE_LABELS), key=init(f"{prefix}_rebalance", "Q"),
                             format_func=REBALANCE_LABELS.get, help="overlay를 켜면 매월 결정합니다.")
    band = c2.number_input("밴드 (%p)", 0.0, 30.0, step=1.0, key=init(f"{prefix}_band", 0.0),
                           help="0이면 결정일마다 거래. 5면 모든 자산이 목표 ±5%p 안에 있을 때 거래를 건너뜁니다.")
    cost = c3.number_input("거래비용 (bp, 한쪽)", 0.0, 100.0, step=5.0, key=init(f"{prefix}_cost", 10.0))
    start = c4.date_input("시작일", key=init(f"{prefix}_start", date.fromisoformat(DEFAULT_START)), min_value=date(1993, 1, 29), max_value=last)
    end = c5.date_input("종료일", key=init(f"{prefix}_end", last), min_value=date(1993, 1, 29), max_value=last)
    o1, o2, o3 = st.columns([2.2, 1, 1.2])
    kind = o1.selectbox("overlay (전술 규칙)", list(OVERLAY_LABELS), key=init(f"{prefix}_ov_kind", "none"), format_func=OVERLAY_LABELS.get)
    param = None
    safe = CASH
    if kind != "none":
        plabel, pdefault, pmin, pmax = OVERLAY_PARAM[kind]
        param = o2.number_input(plabel, float(pmin), float(pmax), step=1.0, key=init(f"{prefix}_ov_{kind}", float(pdefault)))
        if kind != "riskparity":
            safe = o3.selectbox("안전자산 (비운 비중이 가는 곳)", [CASH] + ALL_SYMBOLS, key=init(f"{prefix}_ov_safe", CASH), format_func=label)
    with st.expander("레버리지 (기본: 없음)"):
        l1, l2, l3 = st.columns(3)
        factor = l1.number_input("전체 노출 배수 (변동성 목표에서는 상한)", 0.1, 3.0, step=0.1, key=init(f"{prefix}_lev", 1.0),
                                 help="1 = 레버리지 없음. 1.5면 매일 자산의 150%를 보유하고 50%를 빌린 것과 같음(일일 리셋).")
        spread = l2.number_input("조달 스프레드 (bp/년, 현금금리 위)", 0.0, 500.0, step=10.0, key=init(f"{prefix}_spread", 100.0))
        ter = l3.number_input("레버리지 상품 추가 보수 (bp/년)", 0.0, 300.0, step=10.0, key=init(f"{prefix}_ter", 60.0))
        lev_assets = st.multiselect("자산별 배수 (예: S&P500 2배 ETF)", ALL_SYMBOLS, key=init(f"{prefix}_lev_assets", []), format_func=label)
        per_asset = {}
        if lev_assets:
            cols = st.columns(min(len(lev_assets), 6))
            for i, s in enumerate(lev_assets):
                per_asset[s] = cols[i % len(cols)].number_input(f"{s} 배수", 0.5, 3.0, step=0.5, key=init(f"{prefix}_lm_{s}", 2.0))
    return (dict(rebalance=rebalance, band=float(band), cost_bps=float(cost)), Overlay(kind, None if param is None else float(param), safe),
            Leverage(float(factor), float(spread), float(ter), {k: float(v) for k, v in per_asset.items()}), str(start), str(end))


def strategy_builder(prefix: str, last: date) -> AllocConfig | None:
    weights = weights_section(prefix)
    if not weights:
        return None
    rules, overlay, lev, start, end = rules_section(prefix, last)
    name = st.text_input("전략 이름", key=init(f"{prefix}_name", "내 배분"))
    if start >= end:
        st.warning("시작일이 종료일보다 앞서야 합니다.")
        return None
    return AllocConfig(name or "내 배분", weights, start, end, overlay=overlay, leverage=lev, **rules)


# =========================================================================== result views

def kpi_row(m: dict) -> None:
    cols = st.columns(8)
    items = [("CAGR", pct(m["cagr"]), f"SPY {pct(m['spy_cagr'])}"),
             ("최대 낙폭", pct(m["mdd"]), f"SPY {pct(m['spy_mdd'])}"),
             ("현금 초과 샤프", num(m["sharpe_excess"]), "mean(r−현금)/std × √252"),
             ("칼마", num(m["calmar"]), f"SPY {num(m['spy_calmar'])}"),
             ("최악 12개월", pct(m["worst_12m"]), None),
             ("CVaR 5% (월)", pct(m["cvar_5"]), f"최악 달 {pct(m['worst_month'])}"),
             ("연간 회전율", pct(m["turnover"], 0), f"비용 드래그 {pct(m['cost_drag'], 2)}/년"),
             ("평균 노출", num(m["exposure_mean"]), "1 = 무레버리지 전액 투자")]
    for c, (lab, val, sub) in zip(cols, items):
        c.metric(lab, val, help=sub)


def gates_line(m: dict) -> None:
    g = gate_results(m)
    txt = []
    for k, ok in g.items():
        txt.append(("✅ " if ok else "❌ ") + k)
    st.markdown(" · ".join(txt) + "  \n<small>탐색 Gate: 칼마 ≥ SPY 칼마 그리고 |MDD| ≤ 0.75×|SPY MDD| · 최종 Gate: CAGR ≥ 0.8×SPY CAGR 그리고 |MDD| ≤ 0.75×|SPY MDD|</small>",
                unsafe_allow_html=True)


def equity_chart(res: AllocResult, log: bool = True) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=res.equity.index, y=res.equity, name=res.config.name, line=dict(color=BLUE, width=2))
    fig.add_scatter(x=res.benchmark.index, y=res.benchmark, name="SPY", line=dict(color=ORANGE, width=1.5))
    if res.sleeves:
        for i, (n, s) in enumerate(res.sleeves.items()):
            fig.add_scatter(x=s.index, y=s, name=f"슬리브 · {n}", line=dict(color=PALETTE[(i + 2) % len(PALETTE)], width=1, dash="dot"))
    fig.update_layout(height=380, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10), title="자산 곡선 (시작 = 1)",
                      yaxis_type="log" if log else "linear", legend=dict(orientation="h", y=-0.15))
    return fig


def drawdown_chart(res: AllocResult) -> go.Figure:
    fig = go.Figure()
    fig.add_scatter(x=res.equity.index, y=drawdown(res.equity), name=res.config.name, fill="tozeroy", line=dict(color=BLUE, width=1))
    fig.add_scatter(x=res.benchmark.index, y=drawdown(res.benchmark), name="SPY", line=dict(color=ORANGE, width=1))
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
        rows.append({"연도": int(year), "전략": part.iloc[-1] / e0 - 1, "SPY": bm.loc[part.index[-1]] / b0 - 1,
                     "전략 MDD": drawdown(part / e0).min(), "부분 연도": len(part) < 240})
    return pd.DataFrame(rows)


def annual_chart(ann: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    fig.add_bar(x=ann["연도"], y=ann["전략"], name="전략", marker_color=BLUE)
    fig.add_bar(x=ann["연도"], y=ann["SPY"], name="SPY", marker_color=ORANGE)
    fig.update_layout(height=280, barmode="group", margin=dict(l=10, r=10, t=30, b=10), title="연도별 수익률", yaxis_tickformat=".0%",
                      legend=dict(orientation="h", y=-0.2))
    return fig


def weights_chart(res: AllocResult) -> go.Figure:
    w = res.weights.iloc[::5]  # weekly points are enough to read
    fig = go.Figure()
    for i, col in enumerate(w.columns):
        if (w[col].abs() > 1e-9).any():
            fig.add_scatter(x=w.index, y=w[col], name=label(col), stackgroup="one", line=dict(width=0.5, color=PALETTE[i % len(PALETTE)]))
    if (res.exposure != 1.0).any():
        fig.add_scatter(x=res.exposure.index, y=res.exposure, name="노출 k", line=dict(color="black", width=1.5, dash="dash"), yaxis="y2")
        fig.update_layout(yaxis2=dict(overlaying="y", side="right", title="노출"))
    fig.update_layout(height=300, hovermode="x unified", margin=dict(l=10, r=10, t=30, b=10), title="비중 (레버리지 전)", yaxis_tickformat=".0%",
                      legend=dict(orientation="h", y=-0.2))
    return fig


def show_result(res: AllocResult, m: dict, key: str) -> None:
    for w in res.warnings:
        st.caption("⚠️ " + w)
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
    with st.expander("연도별 표 · 위기 구간"):
        a, b = st.columns(2)
        a.dataframe(ann, hide_index=True, width="stretch", column_config={
            "전략": st.column_config.NumberColumn(format="percent"), "SPY": st.column_config.NumberColumn(format="percent"),
            "전략 MDD": st.column_config.NumberColumn(format="percent")})
        ct = crisis_table(res)
        if len(ct):
            b.dataframe(ct, hide_index=True, width="stretch", column_config={
                "전략": st.column_config.NumberColumn(format="percent"), "SPY": st.column_config.NumberColumn(format="percent")})
        else:
            b.info("기간 안에 등록된 위기 구간이 없습니다.")


# =========================================================================== page 1: research

def research_page() -> None:
    st.title("자산배분 연구실")
    st.caption("배분 → overlay → 레버리지 순서로 전략을 정하고, SPY와 같은 날짜로 비교합니다. 세금·환율·헤지 비용은 반영하지 않습니다. "
               "모든 지표의 정의는 ‘용어 설명’ 페이지에 있습니다.")
    panel = panel_or_stop()
    last = panel.last_date.date()
    st.subheader("① 배분")
    cfg = strategy_builder("r", last)
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
                notes.log_run(cfg.name, cfg.describe(), (str(res.period[0].date()), str(res.period[1].date())), m)
                st.session_state["aa_result"] = (res, m, panel.source)
        except ValueError as exc:
            st.error(str(exc))
    stored = st.session_state.get("aa_result")
    if stored is None:
        return
    res, m, source = stored
    if res.config != cfg:
        st.warning("아래 결과는 현재 설정과 다릅니다. 다시 실행하면 갱신됩니다.")
    show_result(res, m, "live")
    with st.form("save_alloc", border=False):
        c1, c2 = st.columns([3, 1])
        name = c1.text_input("저장 이름", value=res.config.name)
        if c2.form_submit_button("💾 전략 저장", width="stretch"):
            path = store.save_result(res, m, name, source)
            st.success(f"저장했습니다: lab_results/{path.name} — ‘성적표·믹스’에서 비교할 수 있습니다.")


# =========================================================================== page 2: scorecard and mixes

def candidates(panel) -> dict[str, object]:
    """name -> AllocConfig | MixConfig for everything the scorecard can compare."""
    out: dict[str, object] = {}
    for key, p in PRESETS.items():
        out[f"프리셋 · {p.name}"] = AllocConfig(p.name, p.weights, DEFAULT_START, str(panel.last_date.date()))
    for meta in store.list_saved():
        out[f"저장 · {meta['name']} ({meta['saved_at'][:10]})"] = store.config_of(meta)
    live = st.session_state.get("aa_result")
    if live is not None:
        out[f"현재 화면 · {live[0].config.name}"] = live[0].config
    return out


def run_candidate(panel, obj, start: str, end: str) -> tuple[AllocResult, int]:
    from dataclasses import replace
    if isinstance(obj, MixConfig):
        mix = MixConfig(obj.name, tuple((replace(c, start=start, end=end), w) for c, w in obj.sleeves), obj.rebalance, obj.cost_bps)
        return run_mix(panel, mix), mix_parameter_count(mix)
    cfg = replace(obj, start=start, end=end)
    return run_allocation(panel, cfg), cfg.parameter_count()


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
    st.title("견고함 성적표 · 전략 믹스")
    st.markdown("고른 전략들을 **같은 기간**에 다시 돌려 아홉 축으로 점수를 매깁니다. 점수는 비교 집합 안의 순위(0–100)라 집합이 바뀌면 같이 바뀝니다. "
                "‘견고’는 축 점수의 평균으로 **참고용**이며, 판정은 Gate(통과/미달)로 합니다.")
    panel = panel_or_stop()
    cands = candidates(panel)
    default = [k for k in cands if k.startswith("프리셋 · 내 AAA") or k.endswith("SPY 보유") or "영구" in k or "60/40" in k]
    chosen = st.multiselect("비교할 전략", list(cands), key=init("sc_chosen", default))
    c1, c2, c3 = st.columns([1, 1, 2])
    start = c1.date_input("시작일", key=init("sc_start", date.fromisoformat(DEFAULT_START)), min_value=date(1993, 1, 29), max_value=panel.last_date.date())
    end = c2.date_input("종료일", key=init("sc_end", panel.last_date.date()), min_value=date(1993, 1, 29), max_value=panel.last_date.date())
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
                mw[k] = cols[i].number_input(f"{cands[k].name} (%)", 0.0, 100.0, step=5.0, key=init(f"mix_w_{k}", round(100 / len(mix_parts))))
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
        results, params, errors = {}, {}, []
        with st.spinner(f"{len(todo)}개 전략 계산 중…"):
            for name, obj in todo.items():
                try:
                    res, n = run_candidate(panel, obj, str(start), str(end))
                    short = name.split(" · ", 1)[-1]
                    results[short], params[short] = res, n
                except ValueError as exc:
                    errors.append(f"{name}: {exc}")
        for e in errors:
            st.error(e)
        if len(results) >= 2:
            # one common period so every strategy sees the same crises
            idx = None
            for r in results.values():
                idx = r.equity.index if idx is None else idx.intersection(r.equity.index)
            for n, r in results.items():
                if len(r.equity) != len(idx):
                    eq = r.equity.loc[idx]
                    bm = r.benchmark.loc[idx]
                    results[n] = AllocResult(r.config, eq / eq.iloc[0], bm / bm.iloc[0], r.cash.loc[idx], r.weights.loc[idx],
                                             r.exposure.loc[idx], r.turnover, r.n_trades, r.sleeves, r.warnings)
            table, raw = scorecard(results, axis_w, params)
            st.session_state["sc_view"] = (table, raw, results, (str(idx[0].date()), str(idx[-1].date())))
            for n, r in results.items():
                notes.log_run(n, r.config.describe(), (str(idx[0].date()), str(idx[-1].date())), raw.loc[n].to_dict(),
                              kind="mix" if r.sleeves else "alloc")
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
    fig.add_scatter(x=first.benchmark.index, y=first.benchmark, name="SPY", line=dict(color=GRAY, width=1.5, dash="dot"))
    fig.update_layout(height=380, hovermode="x unified", yaxis_type="log", margin=dict(l=10, r=10, t=30, b=10), title="자산 곡선 (공통 기간, 로그)",
                      legend=dict(orientation="h", y=-0.15))
    st.plotly_chart(fig, width="stretch")
    mix_names = [n for n, r in results.items() if r.sleeves]
    if mix_names:
        with st.form("save_mix", border=False):
            c1, c2 = st.columns([3, 1])
            pick = c1.selectbox("저장할 믹스", mix_names)
            if c2.form_submit_button("💾 믹스 저장", width="stretch"):
                mix = mixes.get(f"믹스 · {pick}")
                if mix is not None:
                    path = store.save_result(results[pick], raw.loc[pick].to_dict(), pick, panel.source, mix=mix)
                    st.success(f"저장했습니다: lab_results/{path.name}")


# =========================================================================== page 3: notes

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
        for day, text in (entries if key != "campaign" else entries):
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
    st.caption(f"{len(runs)}회 실행 · 전략 이름 기준 {runs['name'].nunique()}종 · 첫 기록 {str(runs['time'].iloc[0])[:10]}")
    st.dataframe(runs.iloc[::-1], hide_index=True, width="stretch", column_config={
        "time": "시각", "name": "이름", "description": "설정", "period": "기간", "kind": "종류",
        "cagr": st.column_config.NumberColumn("CAGR", format="percent"), "mdd": st.column_config.NumberColumn("MDD", format="percent"),
        "calmar": st.column_config.NumberColumn("칼마", format="%.2f"), "sharpe_excess": st.column_config.NumberColumn("현금 초과 샤프", format="%.2f"),
        "turnover": st.column_config.NumberColumn("회전율", format="percent")})


# =========================================================================== page 4: saved

def saved_page() -> None:
    st.title("저장된 전략")
    items = store.list_saved()
    if not items:
        st.info("저장된 전략이 없습니다. ‘전략 연구’나 ‘성적표·믹스’에서 저장하세요.")
        return
    rows = [{"선택": i < 4, "이름": it["name"], "저장 시각": it["saved_at"], "기간": " → ".join(it["period"]), "설정": it["description"],
             "CAGR": it["metrics"].get("cagr"), "MDD": it["metrics"].get("mdd"), "칼마": it["metrics"].get("calmar"),
             "현금 초과 샤프": it["metrics"].get("sharpe_excess")} for i, it in enumerate(items)]
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
        fig.add_scatter(x=eq0.index, y=eq0["SPY"], name="SPY (첫 결과 기간)", line=dict(color=GRAY, width=1.5, dash="dot"))
        fig.update_layout(height=380, hovermode="x unified", yaxis_type="log", margin=dict(l=10, r=10, t=30, b=10), title="자산 곡선 비교 (최대 8개)",
                          legend=dict(orientation="h", y=-0.15))
        st.plotly_chart(fig, width="stretch")
        st.caption("기간이 다른 결과는 시작점이 달라 직접 비교에 주의하세요. 같은 기간으로 보려면 ‘성적표·믹스’에서 다시 돌리세요.")


# =========================================================================== page 5: glossary

def glossary_page() -> None:
    st.title("용어 설명")
    st.subheader("성적표의 아홉 축")
    st.markdown("점수는 비교 집합 안의 **백분위 순위**(0 = 가장 아쉬움, 100 = 가장 나음, 동점은 평균 순위)입니다. 집합이 바뀌면 점수도 바뀝니다. "
                "‘견고’는 축 점수의 가중 평균으로 참고용이며, 판정은 Gate로만 합니다 (가중 합산 점수로 전략을 고르지 않는다는 연구실 원칙과 같음).")
    for a in AXES:
        st.markdown(BR.join([f"**{a}**", AXIS_TEXT[a]]))
    st.subheader("Gate")
    st.markdown(BR.join(["**탐색 Gate** · 칼마 ≥ SPY 칼마 그리고 |MDD| ≤ 0.75 × |SPY MDD|", "수익 조건 없이 ‘위험 대비’로만 보는 넓은 문. 추세추종 연구실의 탐색 Gate와 같음.",
                         "**최종 Gate** · CAGR ≥ 0.80 × SPY CAGR 그리고 |MDD| ≤ 0.75 × |SPY MDD|", "헌장(CHARTER.md)의 최종 기준."]))
    st.subheader("위기 구간 (성적표 ‘위기’ 축)")
    st.dataframe(pd.DataFrame([{"구간": k, "시작": a, "끝": b} for k, (a, b) in CRISIS_WINDOWS.items()]), hide_index=True, width="stretch")
    st.caption("SPY의 고점→저점 날짜로 2026-10-24에 등록했고, 결과를 보고 고치지 않습니다.")
    st.subheader("기본 지표 (추세추종 연구실과 같은 계산)")
    for k in ("cagr", "mdd", "calmar", "volatility", "worst_12m", "underwater_days"):
        d = DEFINITIONS[k]
        st.markdown(BR.join([f"**{d.label}**", d.short, f"`{d.formula}`"]))
    st.markdown(BR.join(["**현금 초과 샤프**", "일간 수익률에서 현금(단기국채) 수익률을 뺀 뒤 계산한 샤프. 현금 비중이 큰 배분을 과대평가하지 않으려고 씁니다.",
                         "`mean(r − 현금) / std(r − 현금) × √252`"]))
    st.subheader("overlay")
    for k, v in OVERLAY_LABELS.items():
        st.markdown(f"· **{v}**")
    st.subheader("실행 규칙")
    st.markdown(RULES_KO)
    st.subheader("자산 대용과 국내 상품")
    df = pd.DataFrame([{"자산군": a.asset_class, "티커": s, "뜻": a.label, "국내 상장 상품(참고)": a.korea} for s, a in ASSETS.items()])
    df["자산군"] = pd.Categorical(df["자산군"], CLASS_ORDER)
    st.dataframe(df.sort_values(["자산군", "티커"]), hide_index=True, width="stretch")
    st.caption("백테스트는 미국 상장 ETF의 달러 수익률입니다. 국내 상품의 환율·헤지 비용·보수·세금은 들어 있지 않습니다.")
    st.subheader("프리셋")
    st.dataframe(pd.DataFrame([{"이름": p.name, "비중": " · ".join(f"{s} {v:g}" for s, v in p.weights.items()), "설명": p.note} for p in PRESETS.values()]),
                 hide_index=True, width="stretch")


# =========================================================================== navigation

PERSIST_PREFIXES = ("r_", "sc_", "mix_")
for _k in list(st.session_state.keys()):
    if str(_k).startswith(PERSIST_PREFIXES):
        st.session_state[_k] = st.session_state[_k]

nav = st.navigation([
    st.Page(research_page, title="전략 연구", icon="🧺", default=True),
    st.Page(scorecard_page, title="성적표 · 믹스", icon="🏁"),
    st.Page(notes_page, title="발견 · 경고 · 캠페인", icon="📝"),
    st.Page(saved_page, title="저장된 전략", icon="💾"),
    st.Page(glossary_page, title="용어 설명", icon="📖"),
])
nav.run()
