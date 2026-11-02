"""Registered dynamic presets of A42 (post hoc, user's decision 2026-11-02; allocation DECISIONS A42 등록).

Three of the stage-5 representatives, with the settings of `docs/research/allocation/2026-11-02_a42_stage5/
candidates.csv` exactly (`tests/test_aa_dynamic.py` re-runs them against the stage-5 numbers):

  ②  현재 보유 + D1 나스닥 추세 → 주식을 CD로 (확인 10일, 쿨다운 21일, 4분할) + HAA50·BAA50 위성 50 %
  ③  (가) + K2 카나리아 → 나스닥을 안전 묶음으로 (쿨다운 21일) + 변동성 목표 8 % + 위성 50 %
  ⑤  R6 (2024-12 비중) + K3 HAA → 주식을 CD로 + 변동성 목표 10 %

`run` simulates a preset the way stage 5 did (aa_canary.DynamicSim, quarterly rebalance, 10 bp, signal read at the
close and traded at the next close, the satellite mixed daily at a fixed share); `current_targets` gives today's target
weights from today's signal state (each tranche's held state, the volatility level held now, the satellite's latest
monthly target).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import aa_canary as K
from . import aa_layers as LY
from . import aa_tactical as TAC
from .aa_assets import CASH, PRESETS
from .aa_korea import load_usdkrw
from .data import download_panel

START, END = "2000-09-01", "2024-12-31"  # A42 research window (stage 5 numbers)
SAT_SPEC = dict(parts=(("haa", 50.0), ("baa_g4", 50.0)))
R6_2024 = {"379810": 6, "379800": 5, "453630": 5, "453640": 5, "463640": 5, "KRXGOLD": 6, "459580": 7, "0046A0": 7,
           "329750": 6, "305080": 6, "0091C0": 6, "476760": 6, "484790": 6, "114260": 6, "148070": 6, "439870": 6,
           "446720": 6}  # rule R6 on 2024-12-31 (stage 3 targets.csv)
COMMON_LIMITS = (
    "사후: A42 5단계 결과를 본 뒤 사용자가 고른 대표(2026-11-02). 미리 정한 규칙으로 나온 프리셋이 아닙니다.",
    "4단계에서 동적 층이 가짜 신호 200개의 95%를 넘은 비율은 7.7%로 우연 기준 5%를 겨우 넘었습니다. 이 설정도 그 기준을 넘은 것 중 하나입니다.",
)


@dataclass(frozen=True)
class DynamicPreset:
    key: str
    name: str
    stage5_name: str  # the row in the stage-5 result files
    base: str  # the static target it sits on (stage-5 '대상')
    weights: dict[str, float]
    setting: LY.Setting
    satellite: float
    note: str
    limits: tuple[str, ...] = field(default=())
    badge: str = "사후"


DYNAMIC_PRESETS: dict[str, DynamicPreset] = {p.key: p for p in [
    DynamicPreset(
        "a42_dyn2", "A42 ② 현재 보유 + 나스닥 추세 → CD + 위성 50% (사후)",
        "F3 현재 보유 · D1 나스닥 자기 추세 · 모든 위험자산 → CD (A38형) · 확인 10일 · 쿨다운 21일 · 트랑슈 4 · 변동성 목표 없음 · 위성 50%",
        "F3 현재 보유", dict(PRESETS["aaa_2026"].weights), LY.Setting("d1", "riskcd", 10, 4, 0.0, 21), 0.5,
        "현재 보유(S&P500 30 · MSCI선진국 15 · EM선물(H) 10 · 미국30년(H) 10 · 미국10년(H) 25 · 금현물 10)의 반에, 나스닥100 원화 종가가 200일 평균 아래로 "
        "10거래일 연속 있으면 주식 상품을 모두 CD금리 ETF로 옮깁니다(신호 매매 뒤 21거래일은 다시 따르지 않음, 판단을 0·5·10·15일 늦게 읽는 4분할). "
        "나머지 반은 HAA50·BAA50 국내판(월말 결정). 2000-09 – 2024-12 대표 세트 8.84% / −10.3%, 봉인 구간(1991-03 – 2000-08) MDD −17.5%.",
        COMMON_LIMITS + ("위성(HAA50·BAA50)은 DBC 자료 때문에 2007년 이후만 돌고, 봉인 구간에서는 시험하지 못했습니다(그 전은 본체만).",
                         "봉인 구간에서 MSCI선진국·EM선물 25%는 자료가 없어 나머지에 나눴습니다.")),
    DynamicPreset(
        "a42_dyn3", "A42 ③ (가) + 카나리아 → 안전 묶음 + 변동성 목표 8% + 위성 50% (사후)",
        "F4 (가) · K2 VAA·BAA · 나스닥 → 안전 묶음 · 확인 0일 · 쿨다운 21일 · 트랑슈 1 · 변동성 목표 8% · 위성 50%",
        "F4 (가)", dict(PRESETS["a41_ga"].weights), LY.Setting("k2", "safe", 0, 1, 0.08, 21), 0.5,
        "(가)(나스닥100 18 · S&P500 3 · 헬스케어 8 · KRX 금 30 · 미국10년(H) 30 · 미국30년(H) 11)의 반에, SPY·EFA·EEM·AGG 중 하나라도 13612W 모멘텀이 음수면 "
        "나스닥100을 안전 묶음(금·국채·현금)으로 옮기고(쿨다운 21일), 월말마다 직전 60일 변동성이 연 8%를 넘으면 그만큼 CD로 줄입니다. 나머지 반은 HAA50·BAA50 국내판. "
        "2000-09 – 2024-12 대표 세트 9.07% / −8.7%, 봉인 구간 MDD −7.8%.",
        COMMON_LIMITS + ("위성(HAA50·BAA50)은 2007년 이후만 돌고, 봉인 구간에서는 시험하지 못했습니다(그 전은 본체만). CAGR 우위의 대부분은 위성 몫입니다.",)),
    DynamicPreset(
        "a42_dyn5", "A42 ⑤ R6 + HAA 카나리아 → CD + 변동성 목표 10% (사후)",
        "R6 주식 위험 몫 상한 · K3 HAA · 모든 위험자산 → CD (A38형) · 확인 0일 · 쿨다운 0일 · 트랑슈 1 · 변동성 목표 10% · 위성 0%",
        "R6 주식 위험 몫 상한", dict(R6_2024), LY.Setting("k3", "riskcd", 0, 1, 0.10, 0), 0.0,
        "규칙 R6(주식 위험 몫 ≤ 50%)의 2024-12 비중(17개 상품)에, TIP의 13612U 모멘텀이 음수면 주식 상품을 모두 CD로 옮기고, 월말 변동성 목표 연 10%. "
        "2003-11 – 2024-12 대표 세트 7.49% / −7.2%, 봉인 구간 MDD −10.3%.",
        COMMON_LIMITS + ("봉인 구간에는 TIP 자료가 없어(2000-06 –) 신호가 한 번도 켜지지 않았습니다. 봉인 구간 결과는 변동성 목표만의 결과입니다.",
                         "연구 구간이 2003-11(미국 배당 대용 DVY 시작)부터이고, 비중은 고정이라 규칙을 매년 다시 계산하지 않습니다.")),
]}


# --------------------------------------------------------------------------- inputs

def inputs(end: str) -> dict:
    """USD canary closes (chained), USD cash, USD/KRW and the tactical-strategy closes, up to `end`."""
    usd, _ = download_panel(K.CANARY_TICKERS, end=end)
    can_close, _ = K.chained(usd.close)
    tac, _ = download_panel(TAC.ALL_SYMBOLS, end=end)
    return {"can_close": can_close, "usd_cash": usd.cash, "fx": load_usdkrw(), "tac_close": tac.close, "tac_cash": tac.cash}


def _window(panel, w: dict, start: str, end: str) -> tuple[str, str]:
    first = panel.close[list(w)].apply(lambda s: s.first_valid_index()).max()
    last = min(pd.Timestamp(end), panel.close.index[-1])
    return str(max(pd.Timestamp(start), first).date()), str(last.date())


def _parts(panel, p: DynamicPreset, inp: dict, start: str, end: str):
    w = p.weights
    s, e = _window(panel, w, start, end)
    cols = sorted(set(w) | set(K.DEFENSIVE) | {LY.CD})
    sim = K.DynamicSim(panel, cols, s, e, cost_bps=10.0)
    dates = sim.dates
    base_tab = LY.action_tables(w, "none", "none", sim.columns)
    bp, _, _ = sim.run(*K.static_rows(base_tab, sim))
    base_r = np.concatenate([[0.0], bp[0, 1:] / bp[0, :-1] - 1.0])
    st = p.setting
    ki = K.Inputs(inp["can_close"], inp["usd_cash"], inp["fx"], panel, None)
    S = K.states(st.signal, ki, K.RULES[st.signal].params[0], dates).to_numpy()[None, :]
    lev = LY.vol_levels(base_r, dates, st.vol) if st.vol else None
    tab = LY.action_tables(w, st.signal, st.action, sim.columns)
    sat_res = None
    if p.satellite > 0:
        spec = TAC.TacticalSpec("HAA50+BAA50 국내판", start=START, end=e, **SAT_SPEC)
        sat_res = TAC.run_tactical(panel, inp["tac_close"], spec, "krw", inp["tac_cash"])
    return sim, dates, bp[0], S, lev, tab, sat_res


def run(panel, p: DynamicPreset, inp: dict, start: str = START, end: str = END) -> dict:
    """Equity of the preset and of its static base (both start at 1), the signal states and the exposure level."""
    sim, dates, base_path, S, lev, tab, sat_res = _parts(panel, p, inp, start, end)
    qrows = K.quarter_rows(dates)
    P = LY.paths(sim, tab, S, p.setting, qrows, lev)
    sat = sat_res.equity.reindex(dates).to_numpy(float) if sat_res is not None else None
    eq = LY.mix(P, sat, p.satellite)[0]
    return {"dates": dates, "equity": pd.Series(eq / eq[0], index=dates), "static": pd.Series(base_path / base_path[0], index=dates),
            "states": pd.Series(S[0], index=dates),
            "exposure": pd.Series(LY.K_LEVELS[lev] if lev is not None else 1.0, index=dates),
            "cagr": float(LY.cagr(eq[None, :])[0]), "mdd": float(LY.mdd(eq[None, :])[0])}


def current_targets(panel, p: DynamicPreset, inp: dict) -> tuple[dict[str, float], str]:
    """Target weights (fractions over product codes) from the signal state on the panel's last date, and a text."""
    end = str(panel.close.index[-1].date())
    sim, dates, _, S, lev, tab, sat_res = _parts(panel, p, inp, START, end)
    st = p.setting
    Sc = LY.confirm_many(S, st.confirm) if st.confirm else S
    L = len(LY.K_LEVELS) if lev is not None else 1
    tabs = LY.scaled_tables(tab, sim.columns, lev is not None)
    qrows = K.quarter_rows(dates)
    core = np.zeros(len(sim.columns))
    held = []
    for k in LY.TRANCHE_LAGS[st.tranches]:
        sc = K.schedule(LY.delay(Sc, k)[0], qrows, st.cooldown)
        h = int(sc.choice[-1])
        held.append(h)
        core += tabs[h * L + (int(lev[-1]) if lev is not None else 0)]
    core /= len(held)
    w = {c: float(x) * (1 - p.satellite) for c, x in zip(sim.columns, core) if x > 1e-9}
    sat_txt = ""
    if sat_res is not None and p.satellite > 0:
        last = sat_res.signals.iloc[-1]
        sw = TAC.map_weights(last[last > 0].to_dict(), "krw", list(panel.close.columns), TAC.TacticalSpec("x").missing_to)
        for c, x in sw.items():
            w[c] = w.get(c, 0.0) + float(x) * p.satellite
        sat_txt = f" · 위성 {p.satellite:.0%}: {sat_res.signals.index[-1].date()} 월말 신호"
    raw_today = int(S[0, -1])
    since = dates[np.flatnonzero(S[0] != raw_today)[-1] + 1] if (S[0] != raw_today).any() else dates[0]
    state = "위험(신호 켜짐)" if raw_today else "정상"
    txt = (f"{dates[-1].date()} 기준 {K.RULES[st.signal].label}: {state} ({since.date()}부터)"
           f" · 보유 상태 {'·'.join('위험' if h else '정상' for h in held)}"
           + (f" · 변동성 노출 {LY.K_LEVELS[int(lev[-1])]:.2f}" if lev is not None else "") + sat_txt)
    tot = sum(w.values())
    return {c: x / tot for c, x in w.items()}, txt


__all__ = ["DynamicPreset", "DYNAMIC_PRESETS", "inputs", "run", "current_targets", "R6_2024", "START", "END"]
