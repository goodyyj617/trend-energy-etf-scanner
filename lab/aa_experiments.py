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

from lab import aa_search_results as sr

ALLOC = sr.ROOT / "docs" / "research" / "allocation"
STAGE2_PRESETS_DIR = ALLOC / "2026-10-26_stage2_presets"  # A36: the three registered search presets


# the charts lab/aa_app.py knows how to draw (its EXPERIMENT_CHARTS has exactly these keys; a test checks)
CHART_KEYS = ("frontier_model", "frontier_compare", "stage2_model", "stage2_real", "stage2_presets", "neighbour_presets",
              "product_gap", "candidate_shift", "kodex200_share")


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
