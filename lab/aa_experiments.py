"""Allocation lab: the experiment log (page '실험 기록').

User rule (2026-10-26 18:02 KST): every test must say why it was run, what it found, and show a graph, so that it can be
read again later without the memos. Each `Experiment` holds the first two as text and names its charts; the chart data
below is read from the committed result files only (no prices needed), and lab/aa_app.py draws it.

To add an experiment: commit its result files, add the reading function here if a new kind of chart is needed, then
append an `Experiment` to `EXPERIMENTS` (newest last) in the same PR. Nothing here imports streamlit.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from lab import aa_landscape as ls
from lab import aa_local as loc
from lab import aa_search_results as sr

ALLOC = sr.ROOT / "docs" / "research" / "allocation"
STAGE2_PRESETS_DIR = ALLOC / "2026-10-26_stage2_presets"  # A36: the three registered search presets
LOCAL_DIR = ALLOC / "2026-10-27_local_study"  # A37: the structure check around B2 · B3 · B4
LOCAL_MEMO = ALLOC / "2026-10-27_local_study.md"
FILTER_DIR = ALLOC / "2026-10-27_trend_filter"  # A38: a 10-month trend filter on the equities of B2 · B3 · B4
FILTER_MEMO = ALLOC / "2026-10-27_trend_filter.md"
COMP_DIR = ALLOC / "2026-10-27_composition"  # A39: is the B2–B5 composition itself sound
COMP_MEMO = ALLOC / "2026-10-27_composition.md"
CANARY_DIR = ALLOC / "2026-10-27_canary"  # A40: canary signals on B2–B5
CANARY_MEMO = ALLOC / "2026-10-27_canary.md"
A41_DIR = ALLOC / "2026-10-28_a41"  # A41: a map of the whole weight space of 13 products, then robustness and the hold-out
A41_MEMO = ALLOC / "2026-10-28_a41.md"
A42_0_DIR = ALLOC / "2026-10-29_a42_stage0"  # A42 stage 0: data, 24 price sets, hedge-cost estimate
A42_0_MEMO = ALLOC / "2026-10-29_a42_stage0.md"
A42_0_ORDER = ("(가)", "안정형 40", "B2", "B4", "현재 보유", "S&P500")
A42_1_DIR = ALLOC / "2026-10-31_a42_stage1"  # A42 stage 1: factor exposures and eight fixed scenarios
A42_1_MEMO = ALLOC / "2026-10-31_a42_stage1.md"


# the charts lab/aa_app.py knows how to draw (its EXPERIMENT_CHARTS has exactly these keys; a test checks)
CHART_KEYS = ("frontier_model", "frontier_compare", "stage2_model", "stage2_real", "stage2_presets", "neighbour_presets",
              "product_gap", "candidate_shift", "kodex200_share",
              "local_neighbours", "local_groups", "local_bootstrap", "dotcom_curves", "dotcom_named",
              "filter_mdd", "filter_curves", "filter_groups", "filter_cagr",
              "comp_loo", "comp_swap", "comp_simple", "comp_reselect", "comp_model_gap",
              "canary_scatter", "canary_verdicts", "canary_placebo", "canary_trades", "canary_currency",
              "a41_map", "a41_sobol", "a41_include", "a41_rebalance", "a41_holdout",
              "a42_0_sets", "a42_0_hedge", "a42_1_scenarios", "a42_1_check")


@dataclass(frozen=True)
class Experiment:
    key: str
    date: str
    decision: str  # DECISIONS number(s), e.g. "A31"
    title: str
    why: str
    results: tuple[str, ...]
    charts: tuple[str, ...]  # keys of CHARTS in lab/aa_app.py
    memo: Path
    where: str = ""  # where else in the app to look


EXPERIMENTS: tuple[Experiment, ...] = (
    Experiment(
        "a31", "2026-10-25", "A31", "전역 탐색 1단계: 고정 비중 25만 개",
        "안정형 40·45를 최종안으로 받아들이지 않으셨고, 규칙을 ‘MDD −15% 이내만 필수, CAGR은 눈으로 본다’로 바꾸셨습니다. "
        "그래서 살 수 있는 국내 상장 28종(+ 가상 환노출 미국채 2종) 전체에서 고정 비중 배분을 무작위로 뽑아, "
        "MDD 구간마다 CAGR을 어디까지 올릴 수 있는지와 그 재료가 무엇인지 보려고 했습니다.",
        ("249,934개 배분 중 40,992개가 MDD −15% 안이었습니다(모형 가격, 2005–2026, 분기 리밸런싱).",
         "MDD 구간별 최고 CAGR은 12.1 / 11.9 / 11.1 / 10.6 / 9.4%((−15,−14] … (−11,−10])로 등록 프리셋보다 1.2–2.7%p 높습니다.",
         "더 높은 수익의 재료는 거의 나스닥100입니다. 나스닥·섹터·레버리지·가상 상품을 빼면 경계선이 안정형 프리셋 근처까지 내려옵니다.",
         "CSCV PBO 0.87: 훈련 기간에서 ‘선 위 최고’를 고르면 표본 외에서 −15% 안에 남는 경우가 15%뿐입니다. 경계선 위 한 점을 고르는 방식은 믿기 어렵습니다."),
        ("frontier_model",), sr.MEMO_PATH, "‘전역 탐색 결과’ 페이지(기록된 결과 = 모형만)"),
    Experiment(
        "a32", "2026-10-25", "A32", "2단계: 고른 후보를 설정 하나씩 바꿔 흔들기",
        "1단계에서 고른 후보(13.2%짜리 다듬은 후보 R, 전체 최고 A, 구간별 최고 B1–B5)가 진짜 좋은 배분인지, 아니면 "
        "−15% 선에 붙여 고른 우연인지 보려고 했습니다. 리밸런싱 밴드·주기·비용·날짜·헤지 비용·시작 연도·이웃 비중을 하나씩 바꿨습니다.",
        ("R은 밴드·월간·날짜 이동·가상 상품 교체에서, B1은 밴드 3·5%p에서 −15%를 넘고, 둘 다 ±2%p 이웃의 절반 이상이 선 밖입니다. 선에 붙여 고른 몫입니다.",
         "A는 14개 시험 중 3개(밴드 3%p·연간·날짜 이동)에서 0.1–0.2%p 넘습니다.",
         "B2–B5와 안정형 40은 모든 MDD 시험에서 −15% 안에 남았습니다. ‘내 AAA’는 기준부터 선 밖(−16.25%)입니다."),
        ("stage2_model",), sr.STAGE2_MEMO, "‘전역 탐색 결과’ 페이지 ⑥"),
    Experiment(
        "a33", "2026-10-26", "A33", "실제 국내 ETF 가격과 모형 대조",
        "지금까지의 숫자가 실제 국내 가격으로 계산된 줄 아셨는데, 실제로는 전 기간이 모형(미국 ETF × 원/달러)이었습니다. "
        "그래서 국내 상장 31종의 실제 가격(네이버 종가 + 분배금)을 받아, 상품마다 모형과 얼마나 다른지, 그리고 상장 후 구간을 "
        "실제 가격으로 바꾸면 후보의 결론이 바뀌는지 확인했습니다.",
        ("상품별 월수익률 상관은 대부분 0.94–0.98로 방향은 잘 맞습니다. 연 수익 차이는 핵심 주식 상품이 ±0.5%p 안입니다.",
         "후보의 전체 기간 CAGR은 ±0.25%p, MDD는 0–0.45%p만 움직였습니다. 그런데 선에 붙어 있던 R(−15.37%)과 B1(−15.30%)은 선 밖으로 나갔습니다.",
         "KODEX 200은 2002년부터 실제 가격이 있고, 실제 총수익이 모형보다 연 3.7%p 높습니다. 원인은 아직 확인하지 않았습니다.",
         "대부분 상품의 실제 구간은 1–5년뿐이라 2008·2020 위기 낙폭은 여전히 모형 값입니다."),
        ("product_gap", "candidate_shift"), sr.REAL_MEMO, "사이드바 ‘가격 기준’, ‘용어·상품 설명’의 상품 표"),
    Experiment(
        "a35", "2026-10-26", "A34 · A35", "1·2단계를 실제 가격으로 다시 실행",
        "실제 가격 우선(상장 후 실제, 상장 전 모형)을 연구실 기본값으로 정하셨으니(A34), 기록된 탐색 결과도 같은 기준으로 "
        "맞추고 결론이 그대로인지 확인했습니다. seed·표본 수·규칙·후보는 그대로 두고 가격만 바꿨습니다.",
        ("경계선은 0–0.2%p만 올랐습니다(12.2 / 12.1 / 11.2 / 10.6 / 9.5%). −15% 안 배분은 40,992 → 41,407개, PBO는 0.87로 같습니다.",
         "같은 후보의 2단계: B2–B5와 안정형 40은 모든 MDD 시험 통과, A는 같은 세 시험에서 0.1–0.3%p 넘고, R·B1은 기준부터 선 밖입니다.",
         "실제 가격에서 새로 뽑힌 대표 후보는 KODEX 200을 자주 담습니다(CAGR 상위 500개 중 32%, 모형 16%). 위의 +3.7%p/년 차이 때문이라 원인 확인 전에는 쓰지 않습니다."),
        ("frontier_compare", "stage2_real", "kodex200_share"), sr.RERUN_MEMO, "‘전역 탐색 결과’ 페이지(기록된 결과 = 실제 가격 우선)"),
    Experiment(
        "a36", "2026-10-26", "A36", "탐색 프리셋 3개 등록 전 확인",
        "B2·B4·A를 모두 프리셋으로 등록하기로 하셨습니다. A에는 ISA에서 살 수 없는 가상 상품(미국10년 환노출) 8%가 있어 "
        "미국10년(H)에 합쳤는데, 바꾼 배분은 아직 시험한 적이 없어서 등록 전에 같은 2단계 시험(실제 가격 우선)을 돌렸습니다.",
        ("탐색 B2 10.69% / −13.23%: MDD 시험 12개 모두 통과(최악 −13.84%), ±5%p 이웃의 91%가 선 안.",
         "탐색 B4 9.69% / −11.65%: 12개 모두 통과(최악 −12.04%), ±5%p 이웃까지 100% 선 안. 세 프리셋 중 가장 튼튼합니다.",
         "탐색 A(미국10년(H)) 12.06% / −14.48%: 리밸런싱 밴드 3%p에서 −15.61%로 선을 넘고, ±5%p 이웃의 53%만 선 안입니다. 수익은 가장 높지만 가장 덜 튼튼하고 2배 상품 12%를 담습니다."),
        ("stage2_presets", "neighbour_presets"), sr.RERUN_MEMO, "모든 페이지의 프리셋 목록(‘탐색 B2’·‘탐색 B4’·‘탐색 A’), 리밸런싱 계산기"),
    Experiment(
        "a37", "2026-10-27", "A37", "B2·B3·B4의 공통 구조 검증 (닷컴 붕괴 포함)",
        "B2·B3·B4가 지금까지 가장 믿을 만한 배분이라고 정리하시고, 셋의 공통 구조(주식 41–45%를 나스닥100 중심으로, 금 12–35%, "
        "장기 미국채 대신 CD금리·중기채)를 더 믿을 수 있는지 물으셨습니다. 주변에서 더 좋은 한 점을 다시 고르면 ‘선에 붙여 고르기’"
        "(A31 PBO 0.87)를 되풀이하게 되므로, 점이 아니라 구조를 네 가지로 시험했습니다. ① 세 배분 주변을 촘촘히 흔들기, "
        "② 구조 규칙을 지킨 배분 2만 개 대 같은 상품의 무작위 배분 2만 개, 그리고 특징을 하나씩 빼 보기, ③ 기간·순위 안정성"
        "(PBO, 앞뒤 반기 순위, 1년 덩어리 부트스트랩), ④ 미국 대용 자료를 2000-09까지 늘린 닷컴 붕괴 시험입니다. 판정 규칙은 실행 전에 "
        "설계 문서에 고정했습니다. 세 배분은 모두 신호 없는 고정 비중이고, 분기 리밸런싱·편도 10bp·실제 가격 우선(상장 후 실제 국내 가격, "
        "상장 전과 2005년 이전은 미국 ETF × 원/달러 모형)입니다. "
        "B2 = 나스닥100 19 · S&P500 필수소비재 19 · 헬스케어 6 · 미국10년(H) 10 · 미국30년(H) 6 · 투자등급회사채 5 · 골드선물(H) 35. "
        "B3 = 나스닥100 33 · MSCI선진국 12 · 미국10년(H) 10 · 물가채 7 · KRX금현물 12 · CD금리 20 · 달러선물 6. "
        "B4 = 나스닥100 30 · 필수소비재 11 · 투자등급회사채 8 · KRX금현물 15 · CD금리 36.",
        ("① 통과: 비중을 합계 10%p 안에서 옮긴 이웃(예: 한 상품에서 5%p를 다른 상품으로)은 세 배분 모두 100% −15% 안입니다. 20–30%p를 옮겨도 "
         "B2 84%, B3 86%, B4 99%가 선 안입니다. 다만 같은 역할의 상품 하나를 통째로 바꾸면 선을 넘는 경우가 있습니다(B2 골드선물(H)→금현물 −17.1%, "
         "필수소비재→유틸리티 −15.1%, B3 CD금리→미국10년(H) −15.8%, B4 CD금리→달러선물 −15.5%).",
         "② 실패: 구조 규칙(주식 35–50%·그중 나스닥 40% 이상, 금 10–35%, 30년(H) 6% 이하·리츠 없음)을 지킨 2만 개 중 −15% 안은 32.6%로, "
         "같은 21개 상품의 무작위 배분 29.7%와 거의 같습니다(고정한 기준은 2배). 이 구조가 주는 것은 안전이 아니라 수익입니다(CAGR 중앙 10.9% 대 7.4%).",
         "특징 빼기: 낙폭을 막는 것은 ‘장기채 대신 현금·중기채’입니다(30년(H)로 바꾸면 선 안 40% → 9%). 방어 섹터를 S&P500으로 바꾸면 50% → 37%. "
         "나스닥을 S&P500으로 바꾸면 선 안 비율이 오히려 오르고(32.6 → 37.4%), 금을 빼면 조금 내립니다(→ 30.7%). 나스닥과 금이 하는 일은 수익(CAGR 각 +1.6, +1.7%p)입니다.",
         "③ 통과(단서 있음): 구조 안에서 훈련 기간 최고를 고르는 규칙의 PBO는 0.40(무작위 0.61, 전역 0.87), 앞뒤 반기 CAGR 순위 상관은 0.58입니다. "
         "MDD 순위 상관은 0.28로 낮고, 1년 덩어리를 섞은 가상 역사에서 −15%를 넘을 확률은 B2 18%, B3 47%, B4 11%입니다(안정형 40 9%).",
         "④ 실패, 가장 무거운 시험: 2000-09부터 돌리면 MDD가 B2 −16.4%, B3 −30.8%, B4 −25.6%입니다(최저점 2002-07-23). 같은 구조 2만 개 중 선 안은 0.4%, "
         "무작위는 24%입니다. 등록 프리셋 중 선 안에 남은 것은 안정형 35(−11.7%)·영구(−12.0%)·안정형 40(−13.7%)·골든 버터플라이(−14.8%)뿐입니다. "
         "그래서 탐색 B2·B4·A 프리셋 설명에 ‘2005년 이후에만 확인됨’을 붙였습니다.",
         "한계: 2005년 이전은 전부 모형(미국 ETF·뮤추얼펀드·금 선물 × 원/달러)이고, 2000년 3–8월은 빠져 있습니다. 결과는 표본 내 연구이며 승인이 아닙니다."),
        ("local_neighbours", "local_groups", "local_bootstrap", "dotcom_curves", "dotcom_named"), LOCAL_MEMO,
        "‘용어·상품 설명’의 프리셋 설명(탐색 B2·B4·A), ‘발견·경고·캠페인’"),    Experiment(
        "a38", "2026-10-27", "A38", "B2·B3·B4의 주식에 10개월 추세 필터 얹기",
        "A37에서 B2·B3·B4는 2005–2026에는 튼튼했지만, 2000-09부터 돌리면 닷컴 하락장에서 MDD가 −16~−31%로 −15%를 넘었습니다. "
        "그래서 다음 방향으로 ‘추세 필터를 얹어 시험’을 고르셨고, 설계 문서의 규칙대로 실행했습니다. 비중은 그대로 두고 주식 상품에만 "
        "Faber(2007)의 10개월 이동평균 필터를 겁니다. 매월 말, 주식 상품의 원화 월말 종가가 최근 10개 월말 종가 평균보다 낮으면 그 상품의 "
        "비중을 다음 한 달 동안 KODEX CD금리액티브로 옮기고, 다시 위로 올라오면 돌려놓습니다. 금·채권·CD금리·달러선물은 그대로 둡니다. "
        "필터를 쓰면 매월 목표 비중으로 맞추므로 ‘정적 매월’도 같이 봤습니다. 다음 거래일 종가 체결, 편도 10bp, 실제 가격 우선(2005년 이전은 "
        "미국 대용 × 원/달러 모형)입니다. 판정은 실행 전에 고정했습니다. 필터 10에서 두 기간 모두 MDD −15% 안이면 통과, 8·12개월에서도 "
        "안이면 견고, 구조 가족 2,000개의 50% 이상이 두 기간 모두 안이면 ‘구조 + 필터가 넓게 통함’입니다. "
        "B2 = 나스닥100 19 · 필수소비재 19 · 헬스케어 6 · 미국10년(H) 10 · 미국30년(H) 6 · 투자등급회사채 5 · 골드선물(H) 35 (필터 대상은 앞의 세 주식). "
        "B3 = 나스닥100 33 · MSCI선진국 12 · 미국10년(H) 10 · 물가채 7 · KRX금현물 12 · CD금리 20 · 달러선물 6 (나스닥·선진국). "
        "B4 = 나스닥100 30 · 필수소비재 11 · 투자등급회사채 8 · KRX금현물 15 · CD금리 36 (나스닥·필수소비재).",
        ("점 판정·견고 통과: 필터 10을 얹으면 2000-09부터 MDD가 B2 −16.4 → −13.0%, B3 −30.8 → −13.4%, B4 −25.6 → −11.4%입니다. 닷컴 구간"
         "(2000-09–2003) 낙폭은 −4.7 / −5.3 / −3.7%로 줄었습니다. 필터 8·12개월도 두 기간 모두 −15% 안이고, 결정일을 1–15거래일 옮긴 최악 MDD도 "
         "−13.1 / −14.1 / −11.5%입니다.",
         "2005년부터만 보면 MDD는 거의 그대로입니다(정적 −13.2 / −12.9 / −11.7% → 필터 −13.0 / −13.4 / −11.4%). 필터 판의 MDD는 2008이 정하고, "
         "2008에는 필터가 돕지 못했습니다(B3는 −12.4 → −13.4%로 오히려 깊어짐). 2020 낙폭은 1.5–2.1%p 얕아졌습니다.",
         "비용: CAGR이 2005–2026에 연 1.6–1.8%p(10.7 → 8.9, 10.0 → 8.2, 9.7 → 8.0%), 2016–2026에 2.1–2.5%p 줄었습니다. 2000-09–2026은 "
         "거의 같습니다(9.5 → 8.9, 7.8 → 7.7, 7.8 → 7.7%). 주식 비중의 평균 약 20%가 CD금리에 가 있었고, 21년 동안 필터가 58–95번 바뀌었습니다.",
         "구조 판정 실패: A37 구조 가족 2,000개 중 필터 10에서 두 기간 모두 −15% 안은 45.9%로 기준 50%에 못 미칩니다. 같은 상품의 무작위 배분에 "
         "필터를 얹으면 50.2%입니다. 필터를 얹고 나면 구조가 주는 것은 낙폭이 아니라 수익뿐입니다(2005– CAGR 중앙 9.2% 대 6.3%).",
         "비교: 필터 비용을 치른 뒤의 수익은 신호 없는 안정형 40(S&P500 25 · 선진국 8 · EM 6 · 30년(H) 12 · 10년(H) 13 · 금현물 20 · CD금리 16, "
         "분기)과 비슷합니다. 안정형 40은 2005– 8.7% / −11.8%, 2000-09– 8.0% / −13.7%이고, B2 필터 판은 두 기간 모두 8.9% / −13.0%입니다.",
         "점검용 S&P500 보유에 필터를 걸면 2000-09부터는 −45.4 → −30.7%지만 2005부터는 −29.6 → −30.7%로 나빠지고, 결정일을 옮기면 −39.4%까지 "
         "갑니다. 상품 하나에서는 결정일 운이 큽니다.",
         "규칙에 따라 B2·B3·B4의 ‘필터 판’을 프리셋 후보로 제안했고, 사용자가 등록하지 않기로 정했습니다(2026-10-27). 정적 프리셋(탐색 B2·B4)은 그대로 둡니다. "
         "한계: 10개월은 닷컴이 들어 있는 표본에서 알려진 값이라 이 통과는 독립 시험이 아니고, 2005년 이전은 전부 모형이며, 결과는 표본 내 연구입니다."),
        ("filter_mdd", "filter_curves", "filter_cagr", "filter_groups"), FILTER_MEMO,
        "‘전략 연구’ 페이지 overlay의 ‘추세 필터’는 모든 위험자산에 겁니다(주식만 고르는 선택은 아직 화면에 없음)"),
    Experiment(
        "a39", "2026-10-27", "A39", "B2–B5의 종목 구성은 정말 괜찮은가",
        "“가장 보고 싶은 건, 현재 B2~B5의 종목 구성이 정말로 괜찮은 것인가야. 나스닥은 다 들어있는데 나머지는 꽤 달라서 그래.” "
        "네 배분은 A31이 비슷한 기록의 수만 개 중에서 고른 것이라, 나스닥100을 뺀 나머지가 실제 선택인지 우연인지를 다섯 가지로 시험했습니다. "
        "① 상품 하나씩 빼기(그 비중을 나머지에 비례 배분), ② 나스닥 비중은 두고 나머지를 다른 배분 것으로 바꾸기(4×4), ③ 더 단순한 배분"
        "(네 배분 비중의 평균 ‘B 평균’, 나스닥100 · KRX금현물 · CD금리 세 상품 뼈대 24개), ④ 2005–2015 / 2016–2026 한 반기에서 고른 상위 100개가 "
        "B 구성을 다시 뽑는지, ⑤ 상품별 모형 가격이 상장 후 실제 가격과 맞는지입니다. 판정 규칙은 실행 전에 설계 문서에 고정했습니다. "
        "네 배분은 신호 없는 고정 비중이고, 분기 말 종가로 판단해 다음 거래일 종가에 원래 비중으로 되돌리며(밴드 없음, 편도 10bp), 실제 가격 우선"
        "(상장 후 실제 국내 가격, 상장 전 미국 대용 × 원/달러 모형)입니다. "
        "B2 = 나스닥100 19 · 필수소비재 19 · 헬스케어 6 · 미국10년(H) 10 · 미국30년(H) 6 · 투자등급회사채 5 · 골드선물(H) 35. "
        "B3 = 나스닥100 33 · MSCI선진국 12 · 미국10년(H) 10 · 물가채 7 · KRX금현물 12 · CD금리 20 · 달러선물 6. "
        "B4 = 나스닥100 30 · 필수소비재 11 · 투자등급회사채 8 · KRX금현물 15 · CD금리 36. "
        "B5 = 나스닥100 32 · KODEX 200 4 · 골드선물(H) 17 · CD금리 35 · 달러선물 12.",
        ("한 줄 답: 공통 재료는 나스닥100 · 금 · 현금성(CD금리·달러선물·중기채)이고, 그 밖의 차이는 2005–2026 기록을 거의 바꾸지 않습니다. "
         "나머지 상품은 수익과 낙폭 사이에서 어디에 앉을지 정하는 다이얼에 가깝습니다.",
         "① 나스닥100은 넷 모두에서 ‘제 몫’입니다(빼면 CAGR −1.6 / −3.6 / −3.4 / −3.8%p). ‘빼면 나아짐’은 작은 조각 셋뿐이고 폭도 작습니다"
         "(B2 헬스케어 6: CAGR −0.17%p, MDD 그대로 / B3 물가채 7, B4 회사채 8: CAGR +0.3–0.4%p, MDD 0.35–0.4%p 깊어짐). "
         "B2의 골드선물(H) 35는 2005년부터는 빼도 비슷하지만, 2000-09부터는 빼면 MDD가 −16.4 → −32.0%입니다.",
         "② 나머지를 바꾼 12개는 2005–2026에서 모두 −15% 안이지만 5개는 원래보다 CAGR이 0.5%p 넘게 낮아 ‘나머지에 의미 있음’입니다. "
         "B2의 나머지(방어 섹터·골드선물·채권)는 어떤 나스닥 비중과도 11.6–11.9% / −14.4 – −14.9%로 수익 쪽, B5의 나머지(CD·골드선물·달러)는 가장 얕고 낮은 쪽입니다.",
         "③ ‘같은 일을 한다’: B 평균(13종) 10.01% / −11.91%, 세 상품 뼈대 24개 중 8개(예: 나스닥100 30 · KRX금 30 · CD 40 = 10.00% / −12.75%, "
         "35 · 30 · 35 = 10.72% / −13.48%로 B2와 같은 점). 2000-09부터는 나스닥 25% 이상 뼈대와 B 평균(−24.2%)이 모두 −15%를 넘습니다.",
         "④ 2016–2026에서 고르면 B 구성(나스닥 중앙 20.5%, 금 32%)이 다시 뽑히지만, 2005–2015에서 고르면 나스닥 0%·방어 섹터 46.5%가 뽑힙니다. "
         "두 반기 모두 상위 2% 안인 것은 B2뿐(146위 · 279위)이고, B3–B5는 2016년 이후에만 상위입니다(2005–15에서 5,916–11,828위).",
         "⑤ 13종 중 11종이 ‘모형 경고’입니다. 채권은 실제가 모형보다 연 3.0–4.4%p 높고(겹친 기간 1–3년), 섹터 +1.3–2.0, KODEX 200 +3.7, "
         "골드선물(H) −1.4%p, KRX금현물은 월 상관 0.87입니다. 괜찮은 것은 나스닥100과 MSCI선진국뿐입니다.",
         "규칙에 따라 B 평균을 프리셋 후보로 제안했고, 사용자가 등록하지 않고 A40에도 넣지 않기로 정했습니다(2026-10-27, ‘등록 없이 기록만’). 뼈대는 정보로만 보입니다. "
         "한계: 2005년부터 상장일까지는 대부분 모형 값이고, 2000-09 시험은 2005년 이전이 전부 모형입니다. 표본 내 결과입니다."),
        ("comp_loo", "comp_swap", "comp_simple", "comp_reselect", "comp_model_gap"), COMP_MEMO,
        "뼈대나 B 평균의 비중은 ‘전략 연구’ 페이지에서 상품과 비중을 직접 골라 다시 돌려 볼 수 있습니다"),
    Experiment(
        "a40", "2026-10-27", "A40", "B2–B5에 카나리아 신호로 동적 배분 얹기 (신호 14개)",
        "“해당 종목 구성과 비율에 대해 카나리아 자산군 도입으로 동적 자산배분을 해보는 거야. … 최소 8가지 이상에 대해 테스트할 수 있도록 구성해봐.” "
        "“매매는 정기 리밸런싱에 더하지 말고, 신호가 바뀐 순간 기준으로 … 추후에 다시 1개월 재매매 금지 규칙 같은 걸 넣어서 다시 해줘.” "
        "B2–B5의 비중은 그대로 두고, 퀀트에서 알려진 위험 신호 14개가 위험을 알리면 나스닥100을 방어 주식(필수소비재·헬스케어) 또는 그 배분의 안전 묶음"
        "(채권·금·CD금리·달러선물, 비례)으로 옮겼습니다. 신호는 사지 않는 카나리아 11개(미국 ETF 달러 가격·FRED·원/달러: DAA, VAA·BAA, HAA, "
        "S&P500 200일선, 12개월 절대 모멘텀, 신용 스프레드, 장단기 금리차, VIX, 섹터 폭, 금융 불안 지수, 원/달러 추세)와 보유 상품 원화 가격 3개"
        "(나스닥100 자기 200일선, 나스닥 대 방어 주식, 주식 대 안전 묶음 12개월 수익)입니다. 매일 종가로 상태를 읽고 바뀐 다음 거래일 종가에 매매하는 "
        "‘바로 매매’와, 신호 매매 뒤 21거래일은 따르지 않는 ‘1개월 금지’ 두 판을 돌렸습니다. 분기 말 리밸런싱은 그때 들고 있는 상태의 목표로 맞춥니다. "
        "기간은 2005–2026과 닷컴을 넣은 2000-09–2026이고 편도 10bp, 실제 가격 우선입니다. 판정 1–5(두 기간 MDD −15% 안, 이웃·날짜 견고, "
        "같은 수익의 정적 배분보다 1%p 얕음, 날짜를 밀어 낸 가짜 신호 200개의 90% 이상보다 얕음)는 실행 전에 고정했습니다.",
        ("한 줄 답: ‘미국 주식 추세가 꺾이면 나스닥을 안전 묶음으로’가 B3·B4·B5에서 다섯 판정을 모두 넘었습니다. S&P500 200일선, S&P500 12개월 절대 모멘텀, "
         "VAA·BAA 네 카나리아, 나스닥100 자기 200일선입니다. B2와 매크로 신호에는 통과한 것이 없습니다.",
         "208개 조합(배분 4 × 신호·행동 26 × 판 2) 중 27개가 후보입니다. 판정별 통과는 200 / 53 / 39 / 60 / 66개입니다.",
         "후보의 MDD는 2005– −7.9 – −13.3%, 2000-09– −8.2 – −13.6%입니다(정적 B3 / B4 / B5는 2000-09부터 −30.8 / −25.6 / −26.2%). "
         "같은 수익의 정적 배분보다 2000-09 MDD가 1.4–11.1%p(중앙 5.7) 얕고, 가짜 신호의 97–100%보다 얕습니다. 나스닥을 덜 든 효과가 아니라 시점이 맞았다는 뜻입니다.",
         "대가: 2005–2026 CAGR 연 0.8–2.0%p(중앙 1.2, A38 필터 1.6–1.8보다 조금 작음), 2016–2026 중앙 1.3%p. 2000-09–2026 CAGR은 정적과 거의 같습니다(−0.6 – +0.4%p). "
         "예: B4 + 12개월 절대 모멘텀 → 안전 묶음, 1개월 금지 = 8.90% / −10.25%(2005–), 7.91% / −11.72%(2000-09–), 신호 매매 연 1.4번. "
         "비용을 치른 수익은 신호 없는 안정형 40(8.67% / −11.8%, 2000-09– 8.02% / −13.7%)과 비슷하고 MDD는 조금 얕습니다.",
         "2005년 이후의 짧은 위기에서는 효과가 작습니다(중앙: 2008 1.4, 2020 3.3, 2022 1.7%p 얕음). 주된 효과는 닷컴처럼 1–2년 이어진 하락장입니다.",
         "통과 못 한 신호: 신용 스프레드·장단기 금리차(2000-09 MDD −22 – −25%), VIX(가짜 대부분보다도 나쁨), 섹터 폭, 금융 불안 지수, 원/달러, "
         "DAA(바로 매매 연 31번), HAA. B2는 같은 수익의 정적 배분이 신호와 같거나 더 얕습니다.",
         "1개월 금지는 신호 매매를 1/3로 줄이고(VAA·BAA 24.0 → 7.7, 200일선 6.2 → 2.6, 절대 모멘텀 4.5 → 1.4번/년) 후보는 14 → 13개로 거의 그대로입니다. "
         "신호를 원화 환산 가격으로 읽으면 S&P500 200일선의 2000-09 MDD가 평균 5%p 깊어져, 달러 가격으로 읽는 편이 맞았습니다.",
         "규칙에 따라 후보 27행을 프리셋 후보로 제안했고, 사용자가 등록하지 않고 기록만 남기기로 정했습니다(2026-10-27). 정적 프리셋(탐색 B2·B4)은 그대로 둡니다. "
         "한계: 추세 규칙들은 닷컴·2008이 든 자료로 알려져 독립 시험이 아니고, 208개를 시험했고(후보는 한 생각과 그 이웃에 몰림), 2005–2026의 상당 부분이 모형 가격이며, 표본 내 결과입니다."),
        ("canary_scatter", "canary_verdicts", "canary_placebo", "canary_trades", "canary_currency"), CANARY_MEMO,
        "신호 판은 아직 연구실 화면에서 돌릴 수 없습니다(실행기 scripts/aa_canary_study.py)"),
    Experiment(
        "a41", "2026-10-28", "A41", "원칙으로 짓고 백테스트는 검증만: 13개 상품 비중 지도와 강건성",
        "A31–A40은 25만 개 배분 가운데 과거 성적이 가장 좋은 점을 골랐고, 그 고르는 규칙의 과최적화 확률(PBO)은 0.87이었습니다. 고른 B3–B5는 닷컴에서 "
        "−26 – −31%였습니다. 사용자는 아예 새롭게 접근하기로 하고, MDD −15% 목표는 유지하되 원칙으로 배분을 짓고 백테스트는 검증에만 쓰며, 2025–26년 "
        "자료는 마지막에 한 번만 보기로 정했습니다. 그래서 결과를 보기 전에 재료(국내 상장 13개 상품: 나스닥100, S&P500, 필수소비재, 헬스케어, 유틸리티, "
        "KRX 금현물 계좌, CD금리, 미국초단기채, 미국단기채, 미국10년 환노출·(H), 미국30년 환노출·(H)), 판정 넷(기준선 MDD −15% 안, ±5%p 이웃 80% 이상 안, "
        "리밸런싱 528가지 중 90% 이상 안, 10년 창 모두 안), 고르는 방식(최고점이 아니라 넓고 평탄한 영역의 중심)을 고정했습니다(PR #100). "
        "그다음 비중 공간 전체를 1,048,576점으로 고르게 훑고, 어느 상품이 결과를 좌우하는지 읽은 뒤, 후보를 엔진으로 흔들어 봤습니다. "
        "규칙은 모두 정적 배분이고, 분기 말 종가로 판단해 다음 거래일 종가에 원래 비중으로 되돌리며 편도 10bp입니다. 연구 구간은 닷컴을 포함한 "
        "2000-09 – 2024-12, 원화, 상장 후 실제 국내 가격·상장 전 모형 가격입니다(2005년 이전은 전부 모형).",
        ("한 줄 답: −15% 안에서 CAGR 상한은 약 9.1%입니다. 미리 정한 규칙으로는 전체 구간 후보 중심이 0개였고, 판정 넷을 모두 넘은 배분은 비교용 안정형 40뿐입니다"
         "(7.77% / −13.7%, 리밸런싱 528가지 모두 −15% 안, 최악 −14.8%).",
         "지도: 105만 점 중 11.2%(117,305점)가 −15% 안, 9.4%가 −17 – −15% 경계입니다. 엔진과의 차이는 4.4e-14입니다.",
         "무엇이 좌우하나: 낙폭은 나스닥100 비중이 가장 크게 좌우합니다(Sobol 총 지수 0.36). 그다음은 CD금리·미국10년(H) 0.18, 미국30년(H) 0.15, 유틸리티 0.12입니다. "
         "CD금리를 넣으면 −15% 안에 드는 배분이 12.6%p, 미국10년(H)을 넣으면 10.2%p 늘고, 유틸리티(−3.9)·미국30년 환노출(−2.3)·나스닥100(−1.6)은 줄입니다. "
         "같은 만기에서 환헤지 국채가 환노출보다 낙폭을 더 막았습니다(헤지판은 연구 구간 대부분 모형 가격).",
         "평탄한 영역 10점은 나스닥100 13–21 · KRX 금현물 25–32 · 미국10년(H) 14–40 · 미국30년(H) 0–24의 한 구조였습니다(CAGR 8.6–9.0%, MDD −13.3 – −14.3%). "
         "‘비중 10% 이상 상품 집합마다 5점 이상’ 묶음 규칙에서 조각나 중심이 생기지 않았습니다.",
         "과최적화 비용: 앞·뒤 절반에서만 고른 중심 5개는 반대쪽 절반에서 무너져 전체 구간 MDD −19.0 – −42.1%, 리밸런싱 528가지 중 −15% 안 0%였습니다. "
         "PBO는 최고점 고르기 0.91, 평탄 영역 고르기 0.63입니다.",
         "결과를 본 뒤 보완(사후): 평탄한 영역의 평균을 중심으로 두었습니다. (가) 나스닥100 18 · S&P500 3 · 헬스케어 8 · KRX 금현물 30 · 미국10년(H) 30 · "
         "미국30년(H) 11 → 8.84% / −13.71%(2005– 9.84%). 기준선·이웃(95%)·10년 창은 넘었지만 리밸런싱 528가지 중 81%만 −15% 안이라 판정 3에 미달했습니다. "
         "월 100%, 분기 92%, 반기 86%, 연 75%이고, 드물게 맞추거나 밴드를 넓히면 2008·2022에 선을 넘습니다.",
         "(나) −17% 쪽 평균 중심(나스닥100 17 · S&P500 6 · 필수소비재 2 · 헬스케어 8 · KRX 금현물 35 · 미국10년(H) 13 · 미국30년(H) 19)은 9.40% / −15.73%입니다. "
         "−17% 선으로 봐도 리밸런싱 81%라 판정 3에 미달합니다. (가)보다 CAGR 0.56%p를 더 얻고 MDD가 2%p 깊습니다.",
         "규칙형(13개 전부, 분기마다 과거 자료로 다시 계산): 동일 비중 7.23% / −18.3%, 역변동성 5.86% / −9.4%, 주식 위험 몫 50% 상한 6.83% / −13.75%.",
         "보류 구간 2025-01-02 – 2026-10-27(대상마다 한 번, 실제 가격 우선, 기록만): (가) 평균 중심 +23.0% / −11.3%, (나) +24.8% / −13.2%, 안정형 40 +21.5% / −9.1%, "
         "B2 +26.4% / −10.0%, S&P500 +25.4% / −19.2%, 규칙형 +9.9 – +11.7%. 16개 중 11개가 −15% 안이었습니다. 큰 위기가 없던 22개월이라 낙폭 검증력은 약합니다.",
         "한계: 미국10년(H)은 2025-08 상장이라 연구 구간 전부 모형 가격(금리차 + 헤지 비용 20bp)이고, 2000–2004는 전부 모형입니다. 표본 좌표는 설계의 Sobol 대신 "
         "의사 난수(PCG64)를 썼습니다(실행 전 변경 기록). 사후 중심은 결과를 본 뒤 더한 것이고, 모든 숫자는 표본 안 결과입니다. 프리셋은 등록하지 않았습니다."),
        ("a41_map", "a41_sobol", "a41_include", "a41_rebalance", "a41_holdout"), A41_MEMO,
        "‘B2–B5 시험 모음’의 기간 ‘A41 연구 구간’(B2·B4의 리밸런싱 528가지와 A41의 새 배분), PNG 그림 ①–⑧은 docs/research/allocation/2026-10-28_a41/figures/"),
    Experiment(
        "a42_0", "2026-10-29", "A42", "A42 0단계: 빠진 자료를 채우고 가격 가정 24가지로 기존 결과를 다시 재기",
        "A42는 고른 비중이 아니라 고르는 규칙을 검증하고, 모형 불확실성을 축으로 넣어 '모든 가격 가정에서 버티는가'로 판정합니다. 그 전제로, "
        "기존 결과가 자료 선택에 얼마나 기대는지 먼저 재야 했습니다. 원화 현금은 한국은행 기준금리로 잡았지만 실제 CD금리는 보통 더 높고, 환헤지 미국채는 "
        "연구 구간 전부가 모형이며 헤지 비용은 20bp 하나로 가정했고, 국고채 ETF는 한 번도 쓰지 않았습니다. 그래서 CD91·국고채 금리·새 상품(국고채 3·10·30년, "
        "미국 배당)을 받고, 가격 세트 24개(가격 {모형, 보정 모형, 실제 우선} × 헤지 비용 {0, 20, 40, 60bp} × 원화 현금 {기준금리, CD91})를 만들어, "
        "기존 배분 6개를 2000-09 – 2024-12(분기, 10bp)에서 다시 돌렸습니다. 이 단계는 무엇도 고르지 않습니다.",
        ("한 줄 답: 어느 배분이 MDD −15% 안인지는 24세트 모두에서 같습니다. (가) 평균 중심과 안정형 40은 24/24 안, B2·B4·현재 보유·S&P500은 0/24입니다.",
         "MDD는 세트에 따라 0.1–1.2%p만 움직이고, CAGR은 0.4–1.0%p 움직입니다. 헤지 비용 20bp마다 −0.06 – −0.09%p, CD91로 바꾸면 +0.10 – +0.15%p, "
         "보정 모형으로 바꾸면 S&P500 −0.86%p·현재 보유 −0.48%p입니다.",
         "대표 세트(실제 우선 · 헤지 60bp · CD91): (가) 8.81% / −13.68%, 안정형 40 7.80% / −13.68%, B2 9.11% / −16.36%, B4 7.70% / −25.31%, "
         "현재 보유 8.01% / −17.47%, S&P500 8.89% / −45.42%.",
         "환헤지 비용 추정: 실제 환헤지 미국채 ETF 267440(RISE 미국장기국채선물(H), 2017-04 –)을 같은 대용의 헤지 모형(비용 0)과 겹쳐 연 70bp(CD91 기준, "
         "기준금리로는 49bp)를 얻었고, 가장 가까운 격자값 60bp를 대표 세트에 썼습니다. 95% 구간이 약 ±300bp라 근거는 약합니다(선물과 TLT의 듀레이션 차이가 섞임).",
         "새 경로의 기본값(실제 우선 · 20bp · 기준금리)은 예전과 비트 단위로 같고, (가)가 A41과 똑같이 8.843% / −13.706%로 나옵니다.",
         "자료 흠: 달러선물 2019-03-14 이상 체결은 보간, KODEX 200의 2026-07 급변(EWY도 같이 움직임)과 2025-12-29 분배금(종가가 분배만큼 떨어짐)은 그대로 둡니다. "
         "서울 매매기준율로 바꿔도 CAGR은 0.013%p 이하로 바뀌지만, (가)의 MDD는 −13.71%에서 −12.19%로 움직였습니다.",
         "봉인 구간(1990-01 – 2000-08) 자료 15개는 값을 보지 않고 드라이브에만 저장했고 해시만 남겼습니다(금은 월간 시리즈). 5단계에서 한 번 엽니다.",
         "한계: 국고채 대용은 월평균 금리로 만든 근사라 실제 ETF와 월 상관이 0.45–0.55뿐이고, 보정 모형의 겹침은 2024-12에서 잘랐습니다(보류 구간 자료가 섞이지 않게)."),
        ("a42_0_sets", "a42_0_hedge"), A42_0_MEMO,
        "PNG 그림은 docs/research/allocation/figures/a42_stage0/"),
    Experiment(
        "a42_1", "2026-10-31", "A42", "A42 1단계: 배분의 요인 노출과 과거에 없던 충격 8개",
        "백테스트는 지나간 위기 네 번만 보여 줍니다. 그래서 배분마다 주식·성장주·미국 금리·한국 금리·환율·금 여섯 요인에 대한 노출을 숫자로 내고, "
        "실행 전에 정해 둔 충격 8개(2022형 동반 하락 + 원화 강세, 원화 초강세, 금 폭락, 금리 급등, 성장주 역전, 스태그플레이션, 닷컴형, 원화 완충 없는 2008형)를 "
        "넣어 봤습니다. 상품마다 2005–2024 월간 원화 수익(대표 가격 세트: 실제 우선 · 헤지 60bp · CD91)을 여섯 요인에 회귀해 베타를 내고, 배분 노출 = 비중 × 베타의 합, "
        "시나리오 손실 = 노출 × 충격의 합(선형)입니다. 판정은 '8개 중 −25%보다 나쁜 것이 2개 이하'이고, 무엇도 고르지 않습니다.",
        ("한 줄 답: 기존 배분 6개 중 S&P500만 시나리오 판정에 미달했고(−25%보다 나쁜 것 3개), (가)·안정형 40·B2·B4·현재 보유는 통과했습니다(0–1개).",
         "가장 아픈 충격은 S1(2022형 동반 하락 + 원화 강세)입니다: (가) −27.0%, 현재 보유 −27.3%, B2 −25.9%, B4 −25.1%, 안정형 40 −23.1%, S&P500 −39.8%.",
         "원화 노출이 큽니다: 원화가 1% 약해질 때 (가) +0.57%, B4 +0.64%, 안정형 40 +0.47%(달러 자산을 환헤지 없이 들고 있는 몫). 그래서 S2(원화 25% 강세)에서 "
         "(가) −14.2%, B4 −16.0%입니다. 금 노출은 B2 0.34, (가) 0.29로 S3(금 −40%)에서 B2 −13.7%, (가) −11.7%입니다.",
         "원화 완충: 2008 위기 구간(2007-10 – 2009-03) 실제 수익은 (가) +33.6%, 안정형 40 +9.9%, S&P500 −24.8%였지만, 원화가 움직이지 않는 S8에서는 "
         "(가) −4.2%, 안정형 40 −11.6%, S&P500 −45.0%입니다. 그때 원화 약세(+68%)가 낙폭을 20–40%p 덮었다는 뜻입니다.",
         "생성기 점검(판정 아님): 2020·2022는 선형 추정이 실제와 5%p 안에서 맞았습니다. 2008은 주가 −55%와 원화 +68%가 함께 움직여 선형 합이 크게 틀렸고"
         "(S&P500 추정 +11.6% vs 실제 −24.8%), 같은 노출을 로그 움직임에 쓰면 −25.3%로 맞습니다. 큰 충격이 겹치는 시나리오(S1·S7·S8)의 선형 손실은 그만큼 거칩니다.",
         "한계: 국고채 베타는 월평균 금리로 만든 요인이라 약하고(한국 금리 노출이 거의 0으로 나옴), 상장 전이 모형인 상품의 베타는 모형의 베타입니다. "
         "선형 생성기라 레버리지 경로 의존과 위기 때 상관 변화를 잡지 못합니다."),
        ("a42_1_scenarios", "a42_1_check"), A42_1_MEMO,
        "PNG 그림은 docs/research/allocation/figures/a42_stage1/"),
)


# --------------------------------------------------------------------------- chart data (percent units)

def frontier_table(folder: Path | str) -> pd.DataFrame:
    """The first table of frontier.md as long rows: 부분집합 (count stripped), band_hi (upper MDD edge, %), cagr (%)."""
    lines = [ln for ln in (Path(folder) / "frontier.md").read_text(encoding="utf-8").splitlines() if ln.startswith("|")]
    head = [c.strip() for c in lines[0].strip("|").split("|")]
    bands = [int(re.findall(r"-?\d+", h)[1]) for h in head[1:]]
    rows = []
    for ln in lines[2:]:
        cells = [c.strip() for c in ln.strip("|").split("|")]
        if len(cells) != len(head):
            break
        name = re.sub(r"\s*\([\d,]+\)$", "", cells[0])
        for hi, v in zip(bands, cells[1:]):
            if v not in ("–", "-", ""):
                rows.append({"부분집합": name, "band_hi": hi, "cagr": float(v)})
    return pd.DataFrame(rows)


def preset_points(folder: Path | str) -> pd.DataFrame:
    """Registered presets on the search simulator (presets.csv), CAGR / MDD in percent."""
    p = pd.read_csv(Path(folder) / "presets.csv")
    return pd.DataFrame({"이름": p["이름"], "cagr": p["cagr"] * 100, "mdd": p["mdd"] * 100})


def stage2_tests(folder: Path | str) -> pd.DataFrame:
    """Every MDD test of a stage-2 table as long rows: 후보, 시험, MDD (%), and whether it stays within −15 %."""
    s2 = sr.load_stage2(folder)
    rows = []
    for _, r in s2.iterrows():
        for col, label in sr.STAGE2_TESTS:
            if col in s2.columns and not pd.isna(r[col]):
                rows.append({"후보": r["후보"], "시험": label, "MDD": float(r[col])})
    out = pd.DataFrame(rows)
    out["선 안"] = out["MDD"] >= sr.STAGE2_PCT_LINE
    return out


def neighbour_shares(folder: Path | str) -> pd.DataFrame:
    """Per stage-2 row: the share of ±2 / ±5 %p neighbours within −15 % (percent)."""
    s2 = sr.load_stage2(folder)
    rows = [{"후보": r["후보"], "이웃": f"±{step}%p", "선 안 비율": float(r[f"이웃±{step} −15% 안"])}
            for _, r in s2.iterrows() for step in (2, 5)]
    return pd.DataFrame(rows)


def product_gap(folder: Path | str = sr.REAL_DIR) -> pd.DataFrame:
    """A33 per product: real − model CAGR (%p/yr) over the real window, its length and the monthly correlation."""
    p = sr.load_real_products(folder)
    p = p[p["kind"] == "product"].copy()
    for c in ("diff_pp", "years", "corr_monthly", "cagr_real", "cagr_model"):
        p[c] = p[c].astype(float)
    return p[["name", "start", "years", "cagr_real", "cagr_model", "diff_pp", "corr_monthly"]].sort_values("diff_pp").reset_index(drop=True)


CANDIDATE_LABELS = {"aaa_2026": "내 AAA", "stable40": "안정형 40", "stable45_gold26": "안정형 45·금 26"}


def candidate_shift(folder: Path | str = sr.REAL_DIR) -> pd.DataFrame:
    """A33 per candidate: CAGR / MDD on the model panel and on the hybrid panel (percent)."""
    c = sr.load_real_candidates(folder)
    return pd.DataFrame({"후보": [CANDIDATE_LABELS.get(x, x) for x in c["candidate"]],
                         "모형 CAGR": c["cagr_model"], "모형 MDD": c["mdd_model"],
                         "실제 CAGR": c["cagr_hybrid"], "실제 MDD": c["mdd_hybrid"]})


def holding_share(res: sr.SearchResults, code: str, top: int = 500) -> tuple[float, float]:
    """Among the `top` raw samples by CAGR: the share holding `code` and its mean weight (%)."""
    raw = np.flatnonzero((res.cands["출처"] != "다듬음").to_numpy())
    order = raw[np.argsort(-res.cands["cagr"].to_numpy()[raw], kind="stable")[:top]]
    w = res.W[order, sr.SYMBOLS.index(code)]
    return float((w > 0).mean() * 100), float(w.mean())


# --------------------------------------------------------------------------- A37 structure check (percent units)

LOCAL_GROUP_ORDER = ("구조 가족", "X1", "X2", "X3", "X4", "무작위")
DOTCOM_CURVES = ("B2", "B3", "B4", "안정형 40 (금·현금 분산)", "60/40", "S&P500 보유")


def group_label(name: str) -> str:
    if name in loc.ABLATIONS:
        return f"{name} {loc.ABLATIONS[name]}"
    return {"구조 가족": "구조 가족 (규칙 R1–R3)", "무작위": "무작위 (같은 21개 상품)"}.get(name, name)


def local_neighbour_bins(folder: Path | str = LOCAL_DIR) -> pd.DataFrame:
    """A37 ①: per centre and L1-distance bin, the share of random neighbours within −15 %, their median CAGR and worst MDD."""
    b = pd.read_csv(Path(folder) / "neighbour_bins.csv")
    b = b[~b["거리 구간"].str.contains("읽는")]
    return pd.DataFrame({"중심": b["중심"], "거리": b["거리 구간"], "수": b["수"].astype(int), "선 안": b["선안"] * 100,
                         "CAGR 중앙": b["cagr_중앙"] * 100, "MDD 최악": b["mdd_최악"] * 100}).reset_index(drop=True)


def local_groups(folder: Path | str = LOCAL_DIR) -> pd.DataFrame:
    """A37 ② and ④ per group (the structure family, each feature taken out, random): the share within −15 % from 2005 and
    from 2000-09, and the median CAGR / MDD from 2005."""
    g = pd.read_csv(Path(folder) / "groups.csv").set_index("묶음")
    d = pd.read_csv(Path(folder) / "dotcom_groups.csv").set_index("묶음")
    return pd.DataFrame([{"묶음": group_label(k), "수": int(g.at[k, "수"]), "2005~ 선 안": g.at[k, "선안"] * 100,
                          "2000-09~ 선 안": d.at[k, "2000~ 선안"] * 100, "CAGR 중앙": g.at[k, "cagr_중앙"] * 100,
                          "MDD 중앙": g.at[k, "mdd_중앙"] * 100, "닷컴 낙폭 중앙": d.at[k, "닷컴 창 낙폭 중앙"] * 100}
                         for k in LOCAL_GROUP_ORDER])


def bootstrap_mdds(folder: Path | str = LOCAL_DIR) -> pd.DataFrame:
    """A37 ③: the MDD (%) of each of the 1,000 one-year-block bootstrap histories per named allocation, as long rows."""
    long = pd.read_csv(Path(folder) / "bootstrap_mdd.csv.gz").melt(var_name="이름", value_name="MDD")
    long["MDD"] *= 100
    return long


def dotcom_drawdowns(folder: Path | str = LOCAL_DIR, names: tuple[str, ...] = DOTCOM_CURVES) -> pd.DataFrame:
    """A37 ④: drawdown from the running peak (%) from 2000-09 to 2006, long rows. The curves are kept every fifth trading
    day, so these drawdowns are slightly shallower than the daily MDD in dotcom.csv."""
    p = pd.read_csv(Path(folder) / "dotcom_paths.csv.gz", parse_dates=["date"]).set_index("date")
    p = p[[n for n in names if n in p.columns]]
    dd = (p / np.maximum(p.cummax(), 1.0) - 1) * 100
    return dd.reset_index().melt(id_vars="date", var_name="이름", value_name="낙폭")


def dotcom_named(folder: Path | str = LOCAL_DIR) -> pd.DataFrame:
    """A37 ④ per named allocation: CAGR / MDD from 2000-09 and from 2005 (%), and the date of the deepest point."""
    d = pd.read_csv(Path(folder) / "dotcom.csv")
    return pd.DataFrame({"이름": d["이름"], "2000-09~ CAGR": d["cagr"] * 100, "2000-09~ MDD": d["mdd"] * 100,
                         "2005~ CAGR": d["cagr_2005"] * 100, "2005~ MDD": d["mdd_2005"] * 100, "최저점": d["최저점"]})


# --------------------------------------------------------------------------- A38 trend filter (percent units)

FILTER_ERAS = {"2005": "2005–2026", "2000-09": "2000-09–2026"}  # the two start dates; before 2005 is the US-proxy model
FILTER_SETTINGS = ("정적 분기", "정적 월", "필터 8", "필터 10", "필터 12", "Faber 원형 10")
FILTER_STRUCTURE_SHARE = 50.0  # A38 structure verdict: ≥ 50 % of the family inside in both periods with 필터 10


def filter_named(folder: Path | str = FILTER_DIR) -> pd.DataFrame:
    """A38 per named allocation, period and setting: CAGR, MDD, CAGR from 2016, the worst MDD over the rebalance-date
    offsets, the mean share of the equity weight the filter held in CD, and the number of month ends it switched (%)."""
    n = pd.read_csv(Path(folder) / "named.csv", dtype={"기간": str})
    return pd.DataFrame({"이름": n["이름"], "기간": n["기간"], "설정": n["설정"], "CAGR": n["cagr"] * 100, "MDD": n["mdd"] * 100,
                         "2016~ CAGR": n["cagr_2016"] * 100, "날짜 최악 MDD": n["날짜 최악 MDD"] * 100,
                         "꺼진 비중": n["꺼진 비중"] * 100, "꺼진 달": n["꺼진 달"] * 100, "전환": n["전환"].astype(int)})


def filter_drawdowns(folder: Path | str = FILTER_DIR) -> pd.DataFrame:
    """A38: drawdown from the running peak (%) from 2000-09 to 2006 with 정적 분기 and 필터 10, long rows (every fifth
    trading day, so slightly shallower than the daily MDD)."""
    p = pd.read_csv(Path(folder) / "curves.csv.gz", parse_dates=["date"]).set_index("date")
    dd = (p / np.maximum(p.cummax(), 1.0) - 1) * 100
    long = dd.reset_index().melt(id_vars="date", var_name="열", value_name="낙폭")
    long[["이름", "설정"]] = long["열"].str.split(" · ", n=1, expand=True)
    return long.drop(columns="열")


def filter_groups(folder: Path | str = FILTER_DIR) -> pd.DataFrame:
    """A38 per sample group (A37's structure family, random) and setting: the share within −15 % from each start and in
    both, and the median CAGR / MDD (%)."""
    g = pd.read_csv(Path(folder) / "groups.csv")
    out = pd.DataFrame({"묶음": g["묶음"].map(group_label), "설정": g["설정"], "수": g["수"].astype(int),
                        "두 기간 모두": g["두 기간 모두 선 안"] * 100})
    for era in FILTER_ERAS:
        out[f"{era} 선 안"] = g[f"{era} 선 안"] * 100
        out[f"{era} CAGR 중앙"] = g[f"{era} cagr 중앙"] * 100
        out[f"{era} MDD 중앙"] = g[f"{era} mdd 중앙"] * 100
    return out


# --------------------------------------------------------------------------- A39 composition check (percent units)

COMP_NAMES = ("B2", "B3", "B4", "B5")
COMP_REFERENCE = "안정형 40 (금·현금 분산)"
PRODUCT_LABELS = {"200": "KODEX 200"}  # the short names of the result files that read badly alone


def comp_leave_one_out(folder: Path | str = COMP_DIR) -> pd.DataFrame:
    """A39 ① with the taken-out weight spread pro rata (the summary reading): per allocation and product, the change
    (without − with, %p) of CAGR / MDD over the whole period and each half, the 2000-09 MDD change, and the verdict."""
    lo = pd.read_csv(Path(folder) / "loo.csv")
    lo = lo[lo["방식"] == "pro"]
    out = pd.DataFrame({"배분": lo["배분"], "상품": lo["상품"].replace(PRODUCT_LABELS), "판정": lo["판정"]})
    for c in ("Δcagr", "Δmdd", "Δcagr_h1", "Δmdd_h1", "Δcagr_h2", "Δmdd_h2", "Δmdd_2000"):
        out[c] = lo[c] * 100
    return out.reset_index(drop=True)


def comp_swap(folder: Path | str = COMP_DIR) -> pd.DataFrame:
    """A39 ②: the row allocation's nasdaq weight with the column allocation's other products, CAGR / MDD (%) from 2005
    and from 2000-09; the diagonal is the allocation itself."""
    s = pd.read_csv(Path(folder) / "swap.csv")
    return pd.DataFrame({"행": s["행"], "열": s["열"], "CAGR": s["cagr"] * 100, "MDD": s["mdd"] * 100,
                         "2000-09 CAGR": s["cagr_2000"] * 100, "2000-09 MDD": s["mdd_2000"] * 100})


def comp_simple(folder: Path | str = COMP_DIR) -> pd.DataFrame:
    """A39 ③ and its references: B2–B5 and 안정형 40 as registered, B 평균 and the 24 skeletons, CAGR / MDD (%) from 2005
    and from 2000-09 with the ③ verdict (blank for the references)."""
    n = pd.read_csv(Path(folder) / "named.csv")
    ref = n[n["변형"].isin(["등록 비중 그대로", "프리셋"])]
    s = pd.read_csv(Path(folder) / "simple.csv")
    rows = pd.concat([ref.assign(판정=""), s], ignore_index=True)
    return pd.DataFrame({"묶음": rows["배분"], "이름": np.where(rows["배분"] == "뼈대", rows["변형"], rows["배분"]),
                         "CAGR": rows["cagr"] * 100, "MDD": rows["mdd"] * 100, "2000-09 CAGR": rows["cagr_2000"] * 100,
                         "2000-09 MDD": rows["mdd_2000"] * 100, "판정": rows["판정"].fillna("")})


def comp_reselect(folder: Path | str = COMP_DIR) -> pd.DataFrame:
    """A39 ④: role shares (%) of the top 100 chosen on each half, long rows (고른 기간, 역할, 비중), plus B2–B5 as
    고른 기간 = the allocation's name."""
    from lab import aa_composition as comp

    t = pd.read_csv(Path(folder) / "reselect_top.csv.gz")
    long = t.melt(id_vars="고른 기간", value_vars=list(comp.ROLES), var_name="역할", value_name="%")
    b = comp.role_shares(np.array([comp.vector(comp.CENTRES[k]) for k in COMP_NAMES]))[list(comp.ROLES)]
    b.insert(0, "고른 기간", list(COMP_NAMES))
    out = pd.concat([long, b.melt(id_vars="고른 기간", var_name="역할", value_name="%")], ignore_index=True)
    return out.rename(columns={"%": "비중"})


def comp_ranks(folder: Path | str = COMP_DIR) -> pd.DataFrame:
    """A39 ④: where B2–B5 and 안정형 40 rank by CAGR among the samples inside −15 % on each half."""
    return pd.read_csv(Path(folder) / "reselect_ranks.csv")


def comp_model_gap(folder: Path | str = COMP_DIR) -> pd.DataFrame:
    """A39 ⑤: per product of B2–B5, real − model CAGR (%p/yr) after listing, its standard error, the overlap and the flag."""
    m = pd.read_csv(Path(folder) / "model_flags.csv", dtype={"코드": str})
    m["상품"] = m["상품"].replace(PRODUCT_LABELS)
    return m[["코드", "상품", "상장", "겹친 해", "실제 CAGR", "모형 CAGR", "차이 %p", "차이 표준오차 %p", "월 상관", "판정"]]


# --------------------------------------------------------------------------- A40 canary signals (percent units)

CANARY_ERAS = {"2005": "2005–2026", "2000-09": "2000-09–2026"}
CANARY_VERSIONS = ("바로 매매", "1개월 금지")
CANARY_VERDICTS = ("1 2005 −15% 안", "2 2000-09 −15% 안", "3 이웃·날짜 견고", "4 같은 비용보다 1%p 얕음", "5 가짜 90% 이상")


def canary_rows(folder: Path | str = CANARY_DIR) -> pd.DataFrame:
    """A40 one row per allocation × signal × action × version: CAGR / MDD (%) in both periods (MDD from the signal start),
    signal trades a year, the comparisons behind verdicts 3–5, the five verdicts, the pass count and 후보 / 탈락."""
    v = pd.read_csv(Path(folder) / "verdicts.csv")
    out = v[["배분", "신호", "신호 이름", "행동", "행동 이름", "판", *CANARY_VERDICTS, "통과 수", "판정", "2000-09 늦게 시작",
             "2000-09 측정 시작"]].copy()
    for era in CANARY_ERAS:
        out[f"{era} CAGR"] = v[f"cagr_{era}"] * 100
        out[f"{era} MDD"] = v[f"mdd_meas_{era}"] * 100
        out[f"{era} 연 신호 매매"] = v[f"연 신호 매매_{era}"]
    out["날짜 이동 최악 MDD"] = v["날짜 이동 최악 MDD"] * 100
    out["같은 비용 2000-09 MDD"] = v["같은 비용 mdd_2000-09"] * 100
    out["가짜보다 얕은 비율"] = v["가짜보다 얕은 비율"] * 100
    out["21일 안 되돌림"] = v["21일 안 되돌림_2005"]
    return out


def canary_static(folder: Path | str = CANARY_DIR) -> pd.DataFrame:
    """A40 comparators: static B2–B5 and 안정형 40 (quarterly) and A38's equity 10-month filter, CAGR / MDD (%) per period."""
    c = pd.read_csv(Path(folder) / "comparators.csv", dtype={"기간": str})
    return pd.DataFrame({"종류": c["종류"], "기간": c["기간"], "배분": c["배분"], "CAGR": c["cagr"] * 100, "MDD": c["mdd_meas"] * 100})


def canary_placebo(folder: Path | str = CANARY_DIR) -> pd.DataFrame:
    """A40: per period, allocation, signal and action (바로 매매), the real MDD against the 200 placebos' 5 % / median / 95 %
    (%), and the share of placebos deeper than the real signal (%)."""
    p = pd.read_csv(Path(folder) / "placebo_summary.csv", dtype={"기간": str})
    names = pd.read_csv(Path(folder) / "rules.csv").drop_duplicates("신호").set_index("신호")["이름"]
    out = pd.DataFrame({"기간": p["기간"], "배분": p["배분"], "신호": p["신호"], "신호 이름": p["신호"].map(names), "행동": p["행동"]})
    for c in ("가짜 5%", "가짜 중앙", "가짜 95%", "진짜 MDD", "가짜보다 얕은 비율"):
        out[c] = p[c] * 100
    return out


def canary_currency(folder: Path | str = CANARY_DIR) -> pd.DataFrame:
    """A40 currency check: K1 · K3 · K4 read on US-dollar prices (as run) and on KRW-converted prices, centre setting,
    바로 매매: CAGR / MDD (%) of both per period, allocation and action."""
    r = pd.read_csv(Path(folder) / "runs.csv.gz", dtype={"기간": str})
    key = ["기간", "배분", "신호", "행동", "판", "설정", "비용", "날짜 이동"]
    krw = r[r["통화"] == "krw"]
    m = krw.merge(r[r["통화"] == "usd"], on=key, suffixes=("_krw", "_usd"))
    return pd.DataFrame({"기간": m["기간"], "배분": m["배분"], "신호": m["신호"], "행동": m["행동"],
                         "달러 CAGR": m["cagr_meas_usd"] * 100, "달러 MDD": m["mdd_meas_usd"] * 100,
                         "원화 CAGR": m["cagr_meas_krw"] * 100, "원화 MDD": m["mdd_meas_krw"] * 100})


# --------------------------------------------------------------------------- A41 (percent units, like the rest)

A41_ZONES = ("안", "경계", "밖")  # MDD ≥ −15 %, −17 – −15 %, below −17 % (the map keeps a 20,000-point sample of 밖)
A41_MAP_SHOWN = 6000  # points per zone drawn on the page, evenly thinned from the committed table (236,348 rows)
A41_RULES = {"equal": "규칙형 동일 비중", "invvol": "규칙형 역변동성 (현금성 30%)", "riskcap": "규칙형 주식 위험 몫 50% 상한"}
A41_KINDS = ("사후 중심", "비교 배분", "규칙형", "절반 구간 중심", "S&P500")  # drawing / sorting order


def a41_names(text: str) -> str:
    """'379810:18;KRXGOLD:30' -> '나스닥100 18 · KRX금현물 30' (A41 short names; other codes by their search-universe name)."""
    parts = []
    for part in str(text).split(";"):
        code, v = part.split(":")
        parts.append(f"{ls.SHORT.get(code) or sr.name_of(code)} {int(float(v))}")
    return " · ".join(parts)


def a41_label(name: str) -> str:
    """Runner target name -> page name: '중심 전체 (가) 평균·보완 1' -> '평균 중심 (가) · 사후 보완',
    '중심 앞 절반 (나) 1' -> '앞 절반에서 고른 중심 (나) 1'; comparators and rule weights unchanged."""
    m = re.fullmatch(r"중심 전체 \((.)\) 평균·보완 1", name)
    if m:
        return f"평균 중심 ({m.group(1)}) · 사후 보완"
    m = re.fullmatch(r"중심 (앞|뒤) 절반 \((.)\) (\d+)", name)
    if m:
        return f"{m.group(1)} 절반에서 고른 중심 ({m.group(2)}) {m.group(3)}"
    m = re.fullmatch(r"규칙형 (equal|invvol|riskcap)(?: \(13개\))?", name)
    return A41_RULES[m.group(1)] if m else name


A41_SHORT_RULES = {"규칙형 동일 비중": "규칙형 동일", "규칙형 역변동성 (현금성 30%)": "규칙형 역변동성", "규칙형 주식 위험 몫 50% 상한": "규칙형 위험 몫 상한"}


def a41_short(label: str) -> str:
    """Page name -> chart label: '평균 중심 (가) · 사후 보완' -> '(가) 평균 · 사후', '앞 절반에서 고른 중심 (가) 1' -> '앞 절반 (가) 1'."""
    m = re.fullmatch(r"평균 중심 \((.)\) · 사후 보완", label)
    if m:
        return f"({m.group(1)}) 평균 · 사후"
    label = label.replace("절반에서 고른 중심", "절반")
    return A41_SHORT_RULES.get(label, label.replace("S&P500 보유", "S&P500"))


def a41_kind(label: str) -> str:
    if "사후 보완" in label:
        return "사후 중심"
    if "절반에서 고른" in label:
        return "절반 구간 중심"
    if label.startswith("규칙형"):
        return "규칙형"
    return "S&P500" if label.startswith("S&P500") else "비교 배분"


def a41_map(folder: Path | str = A41_DIR, per_zone: int = A41_MAP_SHOWN) -> pd.DataFrame:
    """A41 ① the weight map, every ``len(zone) // per_zone``-th point of each zone: 구역, 비중 (names), CAGR, MDD,
    순위 값 (CAGR from every quarter-end start, 10th percentile), all in %."""
    m = pd.read_csv(Path(folder) / "map.csv.gz", usecols=["비중", "cagr", "mdd", "rank", "zone"])
    parts = []
    for z in A41_ZONES:
        g = m[m["zone"] == z]
        parts.append(g.iloc[::max(1, len(g) // per_zone)])
    out = pd.concat(parts, ignore_index=True)
    return pd.DataFrame({"구역": out["zone"], "비중": out["비중"].map(a41_names), "CAGR": out["cagr"] * 100, "MDD": out["mdd"] * 100,
                         "순위 값": out["rank"] * 100})


def a41_marks(folder: Path | str = A41_DIR) -> pd.DataFrame:
    """A41 named points on the research window (2000-09 – 2024-12): the comparators (engine), the centres (half windows
    and the full window's mean added after the results) and the three rule-based weights on all 13 products."""
    folder = Path(folder)
    rows = []
    for r in pd.read_csv(folder / "named.csv").to_dict("records"):
        label = r["이름"]
        rows.append({"이름": label, "종류": a41_kind(label), "비중": a41_names(r["비중"]), "CAGR": r["cagr"] * 100, "MDD": r["mdd"] * 100,
                     "순위 값": r["rank"] * 100})
    for r in pd.read_csv(folder / "centres.csv").to_dict("records"):
        label = a41_label(f"중심 {r['고른 구간']} {r['묶음']} {r['번호']}")
        rows.append({"이름": label, "종류": a41_kind(label), "비중": a41_names(r["비중"]), "CAGR": r["cagr"] * 100, "MDD": r["mdd"] * 100,
                     "순위 값": r["rank"] * 100})
    rl = pd.read_csv(folder / "rules.csv")
    for r in rl[rl["상품 집합"] == "13개 전부"].to_dict("records"):
        rows.append({"이름": A41_RULES[r["규칙"]], "종류": "규칙형", "비중": "13개 전부, 분기마다 과거 자료로 다시 계산",
                     "CAGR": r["cagr"] * 100, "MDD": r["mdd"] * 100, "순위 값": r["rank"] * 100})
    return pd.DataFrame(rows)


def a41_sobol(folder: Path | str = A41_DIR) -> pd.DataFrame:
    """A41 ② Sobol indices per product for MDD and CAGR (fractions of the output variance; ST = total, S1 = first order,
    with the 95 % bootstrap range)."""
    s = pd.read_csv(Path(folder) / "sobol.csv", dtype={"상품": str})
    s = s[s["출력"].isin(["mdd", "cagr"])]
    return s.assign(출력=s["출력"].map({"mdd": "MDD", "cagr": "CAGR"}))[["출력", "상품", "이름", "S1", "ST", "ST_lo", "ST_hi"]]


def a41_include(folder: Path | str = A41_DIR) -> pd.DataFrame:
    """A41 ② putting a product in: over the 4,095 pairs of product sets that differ by that product only, the median change
    of the share of points inside −15 % (%p) and of the median CAGR (%p)."""
    i = pd.read_csv(Path(folder) / "include_exclude.csv")
    return pd.DataFrame({"상품": i["상품"], "안 비율 변화": i["넣으면 안 비율 변화 중앙"] * 100, "CAGR 변화": i["넣으면 CAGR 중앙 변화"] * 100,
                         "넣었을 때 안 비율": i["넣었을 때 안 비율 중앙"] * 100, "뺐을 때 안 비율": i["뺐을 때 안 비율 중앙"] * 100})


A41_FREQ = {"M": "월", "Q": "분기", "S": "반기", "A": "연"}


def a41_rebalance(folder: Path | str = A41_DIR) -> pd.DataFrame:
    """A41 ④ the 528 rebalancing versions of every target (engine): 이름, 종류, 주기, 변형, CAGR, MDD (%)."""
    r = pd.read_csv(Path(folder) / "robustness.csv.gz")
    r = r[r["시험"] == "리밸런싱"].dropna(subset=["mdd"])
    label = r["대상"].map(a41_label)
    return pd.DataFrame({"이름": label, "종류": label.map(a41_kind), "주기": r["변형"].str[0].map(A41_FREQ), "변형": r["변형"],
                         "CAGR": r["cagr"] * 100, "MDD": r["mdd"] * 100}).reset_index(drop=True)


def a41_verdicts(folder: Path | str = A41_DIR) -> pd.DataFrame:
    """A41 ④ verdicts per target (the runner's table) with page names; shares and MDDs in %."""
    v = pd.read_csv(Path(folder) / "verdicts.csv")
    out = v.assign(이름=v["대상"].map(a41_label))
    out["종류"] = out["이름"].map(a41_kind)
    for c in out.columns:
        if c.endswith("비율") or "MDD" in c or c in ("cagr", "mdd") or "CAGR" in c or c.startswith("순위값"):
            if pd.api.types.is_numeric_dtype(out[c]):
                out[c] = out[c] * 100
    return out


def a42_0_sets(folder: Path | str = A42_0_DIR) -> pd.DataFrame:
    """A42 stage 0: per allocation, CAGR and MDD (%) range over the 24 price sets and the representative set's values."""
    s = pd.read_csv(Path(folder) / "sets.csv")
    g = s.groupby("배분")
    rep = s[s["대표"]].set_index("배분")
    out = pd.DataFrame({"CAGR 최저": g["cagr"].min() * 100, "CAGR 최고": g["cagr"].max() * 100, "MDD 최악": g["mdd"].min() * 100,
                        "MDD 최선": g["mdd"].max() * 100, "−15% 안 세트": g["mdd"].apply(lambda x: int((x >= -0.15).sum())),
                        "대표 CAGR": rep["cagr"] * 100, "대표 MDD": rep["mdd"] * 100})
    return out.reindex([n for n in A42_0_ORDER if n in out.index]).reset_index(names="배분")


def a42_0_hedge(folder: Path | str = A42_0_DIR) -> pd.DataFrame:
    """A42 0-2: hedge-cost estimates (bp a year) with 95 % block-bootstrap intervals, per product and KRW cash."""
    h = pd.read_csv(Path(folder) / "hedge.csv", dtype={"상품": str})
    return h[["상품", "이름", "원화 현금", "겹침 시작", "겹침 일", "h 추정 (bp)", "95% 하한 (bp)", "95% 상한 (bp)"]].copy()


def a42_1_scenarios(folder: Path | str = A42_1_DIR) -> pd.DataFrame:
    """A42 stage 1: linear scenario losses (%) per allocation and scenario, long form, with the scenario names."""
    from .aa_exposure import SCENARIOS
    s = pd.read_csv(Path(folder) / "scenarios.csv", index_col=0)
    long = s[list(SCENARIOS)].mul(100).reset_index(names="배분").melt(id_vars="배분", var_name="시나리오", value_name="손실")
    long["이름"] = long["시나리오"].map(lambda k: f"{k} {SCENARIOS[k][0]}")
    long["판정"] = long["배분"].map(s["판정"])
    return long


def a42_1_check(folder: Path | str = A42_1_DIR) -> pd.DataFrame:
    """A42 stage 1 generator check: actual crisis-window return vs the linear and log estimates (%)."""
    c = pd.read_csv(Path(folder) / "crisis_check.csv", dtype={"위기": str})
    return pd.DataFrame({"배분": c["배분"], "위기": c["위기"], "실제": c["실제"] * 100, "선형 추정": c["선형 추정"] * 100,
                         "로그 추정": c["로그 추정 (참고)"] * 100})


def a41_holdout(folder: Path | str = A41_DIR) -> pd.DataFrame:
    """A41 hold-out (2025-01-02 – 2026-10-27, looked at once): cumulative return, CAGR and MDD (%) on real-first prices and
    on the model alone, per target."""
    h = pd.read_csv(Path(folder) / "holdout.csv")
    label = h["대상"].map(a41_label)
    out = pd.DataFrame({"이름": label, "종류": label.map(a41_kind), "누적 수익": h["실제 누적 수익"] * 100, "CAGR": h["실제 CAGR"] * 100,
                        "MDD": h["실제 MDD"] * 100, "모형 누적 수익": h["모형 누적 수익"] * 100, "모형 MDD": h["모형 MDD"] * 100,
                        "시작": h["실제 시작"], "끝": h["실제 끝"]})
    if "실행" in h:
        out["실행"] = h["실행"].fillna("첫 실행")
    return out
