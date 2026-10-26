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

from lab import aa_local as loc
from lab import aa_search_results as sr

ALLOC = sr.ROOT / "docs" / "research" / "allocation"
STAGE2_PRESETS_DIR = ALLOC / "2026-10-26_stage2_presets"  # A36: the three registered search presets
LOCAL_DIR = ALLOC / "2026-10-27_local_study"  # A37: the structure check around B2 · B3 · B4
LOCAL_MEMO = ALLOC / "2026-10-27_local_study.md"
FILTER_DIR = ALLOC / "2026-10-27_trend_filter"  # A38: a 10-month trend filter on the equities of B2 · B3 · B4
FILTER_MEMO = ALLOC / "2026-10-27_trend_filter.md"


# the charts lab/aa_app.py knows how to draw (its EXPERIMENT_CHARTS has exactly these keys; a test checks)
CHART_KEYS = ("frontier_model", "frontier_compare", "stage2_model", "stage2_real", "stage2_presets", "neighbour_presets",
              "product_gap", "candidate_shift", "kodex200_share",
              "local_neighbours", "local_groups", "local_bootstrap", "dotcom_curves", "dotcom_named",
              "filter_mdd", "filter_curves", "filter_groups", "filter_cagr")


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
         "규칙에 따라 B2·B3·B4의 ‘필터 판’을 프리셋 후보로 제안합니다. 등록은 사용자가 정하고, 정적 프리셋(탐색 B2·B4)은 그대로 둡니다. "
         "한계: 10개월은 닷컴이 들어 있는 표본에서 알려진 값이라 이 통과는 독립 시험이 아니고, 2005년 이전은 전부 모형이며, 결과는 표본 내 연구입니다."),
        ("filter_mdd", "filter_curves", "filter_cagr", "filter_groups"), FILTER_MEMO,
        "‘전략 연구’ 페이지 overlay의 ‘추세 필터’는 모든 위험자산에 겁니다(주식만 고르는 선택은 아직 화면에 없음)"),
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
