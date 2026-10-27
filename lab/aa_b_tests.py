"""Every committed result that re-ran the static allocations B2–B5 (or allocations next to them), gathered as
CAGR / MDD points for the page 'B2–B5 시험 모음' (user request, 2026-10-27: "지금까지 B2–B5에 관련해서 테스트한 건 따로
다시 정리해서 한 눈에 MDD와 CAGR을 볼 수 있도록").

Nothing here runs a backtest: it only reads the result files of the studies that tested B2–B5
(A32 stage 2 re-run on real prices = A35, A37 local study, A38 trend filter, A39 composition, A40 canary signals).
Values are fractions (0.107 = 10.7 %).
A study that adds B2–B5 results later registers a reader in ``FAMILIES`` and ``points``.
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from lab import aa_search_results as sr

ALLOC = sr.ROOT / "docs" / "research" / "allocation"
STAGE2_DIR = ALLOC / "2026-10-26_stage2_real"  # A32 stage 2 on real prices (A35)
LOCAL_DIR = ALLOC / "2026-10-27_local_study"  # A37
FILTER_DIR = ALLOC / "2026-10-27_trend_filter"  # A38
COMP_DIR = ALLOC / "2026-10-27_composition"  # A39
CANARY_DIR = ALLOC / "2026-10-27_canary"  # A40
SEARCH_DIR = sr.RESULTS_REAL_DIR  # A31 stage 1 on real prices (A35), the grey background

NAMES = ("B2", "B3", "B4", "B5")
REFERENCE = "안정형 40"
OTHERS = ("B 평균", "뼈대")  # A39 ③: allocations built from B2–B5 rather than B2–B5 themselves (drawn grey)
# stage-2 row label of each B (the band the search picked it from)
STAGE2_LABELS = {"B2": "B 레버·가상 없음 (-14,-13]", "B3": "B 레버·가상 없음 (-13,-12]",
                 "B4": "B 레버·가상 없음 (-12,-11]", "B5": "B 레버·가상 없음 (-11,-10]"}
STAGE2_REFERENCE = "비교: 안정형 40 (금·현금 분산)"

PERIODS = {"2005": "2005-01-03 – 2026-10-16 (상장 후 실제 가격, 상장 전 모형)",
           "2000-09": "2000-09-01 – 2026-10-16 (닷컴 포함, 2005년 이전은 전부 모형)"}

# the registered weights, whole percents (the same numbers as the presets / aa_local.CENTRES / stage2.csv)
WEIGHTS = {
    "B2": "나스닥100 19 · 필수소비재 19 · 헬스케어 6 · 미국10년(H) 10 · 30년(H) 6 · 투자등급회사채 5 · 골드선물(H) 35",
    "B3": "나스닥100 33 · MSCI선진국 12 · 미국10년(H) 10 · 물가채 7 · KRX금 12 · CD금리 20 · 달러선물 6",
    "B4": "나스닥100 30 · 필수소비재 11 · 투자등급회사채 8 · KRX금 15 · CD금리 36",
    "B5": "나스닥100 32 · KODEX 200 4 · 골드선물(H) 17 · CD금리 35 · 달러선물 12",
}
RULES = "정적 배분, 분기 말 종가로 판단해 다음 거래일 종가에 원래 비중으로 되돌림, 밴드 없음, 편도 10 bp"


@dataclass(frozen=True)
class Family:
    label: str
    what: str  # what was changed, in the user's words
    source: str  # where it is recorded
    periods: tuple[str, ...]


FAMILIES: dict[str, Family] = {
    "centre": Family("등록 비중 그대로", RULES, "A35 2단계 실제 가격 재실행 · A37 ④ · A38", ("2005", "2000-09")),
    "stage2": Family("2단계 설정 바꾸기",
                     "같은 비중으로 리밸런싱 밴드 0·3·5 %p, 월간·연간 리밸런싱, 편도 비용 20 bp, 헤지 추가 비용 40 bp(기본 20)",
                     "A32 → A35 (2026-10-26_stage2_real/stage2.csv), 2000-09의 월간은 A38", ("2005", "2000-09")),
    "start": Family("2단계 시작 연도 바꾸기", "같은 비중을 2008·2010·2015년부터 돌림(기간이 짧아 MDD·CAGR을 그대로 비교하면 안 됨)",
                    "A32 → A35 (stage2.csv)", ("2005",)),
    "nb2": Family("±2 %p 이웃", "한 상품에서 2 %p를 떼어 다른 상품에 준 배분 전부", "A32 → A35 (2026-10-26_stage2_real/neighbours.csv.gz)",
                  ("2005",)),
    "nb5": Family("±5 %p 이웃", "한 상품에서 5 %p를 떼어 다른 상품에 준 배분 전부", "A32 → A35 (neighbours.csv.gz)", ("2005",)),
    "local": Family("넓은 이웃 (합계 2–38 %p)",
                    "비중을 여러 상품에서 함께 옮기거나 새 상품 하나를 들인 무작위 이웃 5,000개씩. 거리 = 비중 차이 절댓값의 합",
                    "A37 ① (2026-10-27_local_study/neighbours.csv.gz)", ("2005",)),
    "swap": Family("같은 역할 상품 바꾸기", "한 상품의 비중 전부를 같은 역할의 다른 상품으로(금선물↔금현물, CD↔달러선물·10년(H), 섹터끼리, 나스닥→S&P500 등)",
                   "A37 ① (2026-10-27_local_study/substitutions.csv)", ("2005",)),
    "filter": Family("추세 필터 얹기", "비중은 그대로, 월말에 주식 상품의 원화 종가가 N개월 평균 아래면 그 달은 그 비중을 CD금리로"
                     "(필터 8·10·12 = 주식 상품만, Faber 원형 10 = 금·채권을 포함한 모든 위험자산)", "A38 (2026-10-27_trend_filter/named.csv), B5의 필터 10은 A40 비교용 (2026-10-27_canary/comparators.csv)", ("2005", "2000-09")),
    "loo": Family("상품 하나씩 빼기", "한 상품을 빼고 그 비중을 나머지 상품에 비례해 나누거나(비례 배분) 전부 CD금리로. 점 설명 끝의 판정은 "
                  "2005–2015·2016–2026 두 반기로 본 것(제 몫 = 빼면 두 반기 모두 MDD 0.5 %p 이상 깊어지거나 CAGR 0.3 %p 이상 낮아짐, "
                  "빼면 나아짐 = 어느 반기도 그만큼 나빠지지 않고 하나 이상 좋아짐, 나머지는 기간 따라 다름)",
                  "A39 ① (2026-10-27_composition/loo.csv)", ("2005", "2000-09")),
    "swaprest": Family("나머지 맞바꾸기", "나스닥100 비중은 그 배분 그대로 두고, 나머지 상품을 다른 B의 나머지 상품으로 바꿈"
                       "(다른 B의 나머지 비율대로 100 − 나스닥을 채움)", "A39 ② (2026-10-27_composition/swap.csv)", ("2005", "2000-09")),
    "simple": Family("B 평균 · 세 상품 뼈대", "B2–B5 비중의 평균(13개 상품)과 나스닥100 20–35 × KRX금 10–35 × 나머지 CD금리 뼈대 24개. "
                     "B2–B5가 아니라 새로 만든 배분이라 회색으로 그림. 판정 ‘같은 일을 한다’ = 2005–2026 MDD −15% 안이고 CAGR이 B2–B5 중 "
                     "가장 낮은 것 이상", "A39 ③ (2026-10-27_composition/simple.csv)", ("2005", "2000-09")),
    "canary_pass": Family("카나리아 신호 · 판정 통과", "아래 ‘카나리아 신호 · 탈락’과 같은 시험 중 판정 1–5(2005와 2000-09 MDD −15% 안, "
                          "이웃 설정·판단일 이동에도 견고, 같은 비용의 정적 배분보다 2000-09 MDD 1 %p 이상 얕음, 가짜 신호 200개 중 90% 이상보다 "
                          "얕음)를 모두 통과한 조합", "A40 (2026-10-27_canary/verdicts.csv)", ("2005", "2000-09")),
    "canary_fail": Family("카나리아 신호 · 탈락", "비중은 그대로 두고 신호 14개 중 하나를 매일 종가로 읽어, 위험 신호가 켜지면 다음 거래일 종가에 "
                          "나스닥100을 방어 주식(필수소비재·헬스케어) 또는 안전 묶음(채권·금·CD·달러)으로 옮기고 꺼지면 되돌림(D2·D3는 10 %p "
                          "기울이기). ‘바로 매매’와 ‘1개월 금지’(신호 매매 뒤 21거래일은 신호를 따르지 않음) 두 판. 분기 말 리밸런싱은 그대로",
                          "A40 (2026-10-27_canary/verdicts.csv)", ("2005", "2000-09")),
}

STAGE2_VARIANTS = (("엔진 밴드0", "stage2", "엔진 재계산 (밴드 0)"), ("엔진 밴드3", "stage2", "밴드 3 %p"), ("엔진 밴드5", "stage2", "밴드 5 %p"),
                   ("M", "stage2", "월간 리밸런싱"), ("A", "stage2", "연간 리밸런싱"), ("20bp", "stage2", "편도 비용 20 bp"),
                   ("헤지40", "stage2", "헤지 추가 비용 40 bp"),
                   ("2008~", "start", "2008년부터"), ("2010~", "start", "2010년부터"), ("2015~", "start", "2015년부터"))
FILTER_SETTINGS = ("필터 8", "필터 10", "필터 12", "Faber 원형 10")

COLUMNS = ["배분", "묶음", "변형", "기간", "cagr", "mdd", "거리", "비중"]


def names(text: str) -> str:
    """'379810:24;453630:19' -> '미국나스닥100 24 · 미국S&P500필수소비재 19' (the stored code text of the result files)."""
    return " · ".join(f"{sr.name_of(c)} {v}" for c, v in sr.parse_codes(text).items())


def _frame(rows: list[dict]) -> pd.DataFrame:
    out = pd.DataFrame(rows, columns=COLUMNS)
    out["거리"] = pd.to_numeric(out["거리"], errors="coerce")
    return out


def stage2_points() -> pd.DataFrame:
    """The four B rows of the stage-2 table (real prices): the centre and each changed setting."""
    s2 = pd.read_csv(STAGE2_DIR / "stage2.csv").set_index("후보")
    rows = []
    for name, label in {**STAGE2_LABELS, REFERENCE: STAGE2_REFERENCE}.items():
        r = s2.loc[label]
        rows.append(dict(배분=name, 묶음="centre", 변형="등록 비중 그대로", 기간="2005", cagr=r["CAGR"] / 100, mdd=r["MDD"] / 100,
                         비중=WEIGHTS.get(name, "")))
        if name == REFERENCE:
            continue
        for col, fam, text in STAGE2_VARIANTS:
            c, m = r.get(f"{col} CAGR"), r.get(f"{col} MDD")
            if pd.notna(c) and pd.notna(m):
                rows.append(dict(배분=name, 묶음=fam, 변형=text, 기간="2005", cagr=c / 100, mdd=m / 100, 비중=WEIGHTS[name]))
    return _frame(rows)


def stage2_extra() -> pd.DataFrame:
    """Stage-2 numbers that are not a CAGR/MDD pair: worst MDD over the decision-date shifts and the neighbour counts."""
    s2 = pd.read_csv(STAGE2_DIR / "stage2.csv").set_index("후보")
    cols = {"날짜 최악 MDD": "판단일 이동 최악 MDD", "이웃±2 −15% 안": "±2 %p 이웃 −15% 안 (%)", "이웃±5 −15% 안": "±5 %p 이웃 −15% 안 (%)",
            "이웃±5 최악 MDD": "±5 %p 이웃 최악 MDD"}
    out = s2.loc[[STAGE2_LABELS[n] for n in NAMES], list(cols)].rename(columns=cols)
    out.index = list(NAMES)
    return out


def neighbour_points() -> pd.DataFrame:
    nb = pd.read_csv(STAGE2_DIR / "neighbours.csv.gz")
    of = {v: k for k, v in STAGE2_LABELS.items()}
    nb = nb[nb["후보"].isin(of)]
    return _frame([dict(배분=of[r.후보], 묶음=f"nb{int(r.step)}", 변형=f"±{int(r.step)} %p 이웃", 기간="2005", cagr=r.cagr, mdd=r.mdd,
                        거리=2 * int(r.step), 비중=names(r.w)) for r in nb.itertuples()])


def local_points() -> pd.DataFrame:
    nb = pd.read_csv(LOCAL_DIR / "neighbours.csv.gz", low_memory=False)
    nb = nb[nb["종류"] == "이웃"]
    return _frame([dict(배분=r.중심, 묶음="local", 변형=f"넓은 이웃 · 거리 {int(r.거리)} %p", 기간="2005", cagr=r.cagr, mdd=r.mdd,
                        거리=int(r.거리), 비중=names(r.비중)) for r in nb.itertuples()])


def swap_points() -> pd.DataFrame:
    s = pd.read_csv(LOCAL_DIR / "substitutions.csv")
    s = s[s["종류"] == "바꾸기"]
    return _frame([dict(배분=r.중심, 묶음="swap", 변형=str(r.바꾸기), 기간="2005", cagr=r.cagr, mdd=r.mdd) for r in s.itertuples()])


def filter_points() -> pd.DataFrame:
    """A38: the static centre on the extended panel (2000-09) and every filter setting in both periods."""
    n = pd.read_csv(FILTER_DIR / "named.csv")
    n = n[n["이름"].isin(list(NAMES) + [REFERENCE])]
    rows = []
    for r in n.itertuples():
        if r.설정 == "정적 분기":
            if r.기간 == "2000-09":  # 2005 centres come from the stage-2 table
                rows.append(dict(배분=r.이름, 묶음="centre", 변형="등록 비중 그대로", 기간="2000-09", cagr=r.cagr, mdd=r.mdd,
                                 비중=WEIGHTS.get(r.이름, "")))
            continue
        if r.이름 == REFERENCE:
            continue
        if r.설정 == "정적 월":  # 2005 has the stage-2 monthly run; the extended panel only this one
            if r.기간 == "2000-09":
                rows.append(dict(배분=r.이름, 묶음="stage2", 변형="월간 리밸런싱 (A38)", 기간=r.기간, cagr=r.cagr, mdd=r.mdd,
                                 비중=WEIGHTS.get(r.이름, "")))
        elif r.설정 in FILTER_SETTINGS:
            rows.append(dict(배분=r.이름, 묶음="filter", 변형=r.설정, 기간=r.기간, cagr=r.cagr, mdd=r.mdd, 비중=WEIGHTS.get(r.이름, "")))
    return _frame(rows)


def b5_filter_points() -> pd.DataFrame:
    """A38 ran the trend filter on B2–B4 only; A40 ran its 'equity 10-month' setting on B5 as a comparator."""
    c = pd.read_csv(CANARY_DIR / "comparators.csv")
    c = c[(c["종류"] == "A38 주식 10개월선") & (c["배분"] == "B5")]
    return _frame([dict(배분="B5", 묶음="filter", 변형="필터 10 (A40 비교용)", 기간=r.기간, cagr=r.cagr, mdd=r.mdd, 비중=WEIGHTS["B5"])
                   for r in c.itertuples()])


def _periods(row) -> list[tuple[str, float, float]]:
    """A39 rows hold both periods side by side: (cagr, mdd) on 2005 and (cagr_2000, mdd_2000) on 2000-09."""
    return [("2005", row.cagr, row.mdd), ("2000-09", row.cagr_2000, row.mdd_2000)]


def composition_points() -> pd.DataFrame:
    """A39: the 2000-09 centres (B5 had none before), ① leave one out, ② swap the rest, ③ B 평균 and the skeletons."""
    rows = []
    named = pd.read_csv(COMP_DIR / "named.csv")
    for r in named[named["시험"] == "중심"].itertuples():
        rows.append(dict(배분=r.배분, 묶음="centre", 변형="등록 비중 그대로", 기간="2000-09", cagr=r.cagr_2000, mdd=r.mdd_2000,
                         비중=WEIGHTS.get(r.배분, "")))
    for r in pd.read_csv(COMP_DIR / "loo.csv", dtype={"뺀상품": str}).itertuples():
        for period, c, m in _periods(r):
            rows.append(dict(배분=r.배분, 묶음="loo", 변형=f"{r.변형} · 판정: {r.판정}", 기간=period, cagr=c, mdd=m))
    for r in pd.read_csv(COMP_DIR / "swap.csv").itertuples():
        if r.행 == r.열:
            continue
        for period, c, m in _periods(r):
            rows.append(dict(배분=r.행, 묶음="swaprest", 변형=f"나스닥100은 {r.행} 그대로, 나머지를 {r.열}의 것으로", 기간=period, cagr=c,
                             mdd=m, 비중=names(r.비중)))
    for r in pd.read_csv(COMP_DIR / "simple.csv").itertuples():
        for period, c, m in _periods(r):
            rows.append(dict(배분=r.배분, 묶음="simple", 변형=f"{r.변형} · 판정: {r.판정}", 기간=period, cagr=c, mdd=m,
                             비중=names(r.비중)))
    return _frame(rows)


def canary_points() -> pd.DataFrame:
    """A40: every signal × action × version on its registered B, both periods. The verdicts measured MDD from the day
    the signal's data starts (``mdd_meas``); the point is the whole period like every other point, so a late start is
    spelled out in the label."""
    v = pd.read_csv(CANARY_DIR / "verdicts.csv")
    rows = []
    for r in v.to_dict("records"):
        ok = r["판정"] == "후보"
        verdict = "판정 1–5 통과" if ok else f"탈락 (판정 {int(r['통과 수'])}/5 통과)"
        for period in ("2005", "2000-09"):
            text = f"{r['신호 이름']} · {r['행동 이름']} · {r['판']} · 연 신호 매매 {r[f'연 신호 매매_{period}']:.1f}회"
            meas = r[f"mdd_meas_{period}"]
            if abs(meas - r[f"mdd_{period}"]) > 1e-12:
                start = r.get(f"{period} 측정 시작", "")
                text += f" · 신호 자료가 늦게 시작해 판정은 {start}부터 잰 MDD {100 * meas:.1f}%".replace(" -", " −")
            rows.append(dict(배분=r["배분"], 묶음="canary_pass" if ok else "canary_fail", 변형=f"{text} · {verdict}", 기간=period,
                             cagr=r[f"cagr_{period}"], mdd=r[f"mdd_{period}"], 비중=WEIGHTS[r["배분"]]))
    return _frame(rows)


def points() -> pd.DataFrame:
    """All points, both periods. ``묶음`` is a key of FAMILIES; ``배분`` is B2–B5, the reference 안정형 40, or one of
    OTHERS (A39 ③). A centre recorded by two studies (the 2000-09 runs of A38 and A39 agree) is kept once."""
    parts = [stage2_points(), neighbour_points(), local_points(), swap_points(), filter_points(), b5_filter_points(),
             composition_points(), canary_points()]
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
    centre_dup = (out["묶음"] == "centre") & out.duplicated(["배분", "묶음", "기간"])
    out = out[~centre_dup].reset_index(drop=True)
    out["−15% 안"] = out["mdd"] >= sr.MDD_LINE
    return out


def background() -> pd.DataFrame:
    """The A31 stage-1 allocations on real prices (2005 period only): within the line, plus the sample outside it."""
    p = pd.read_csv(SEARCH_DIR / "passing.csv.gz", usecols=["cagr", "mdd"])
    c = pd.read_csv(SEARCH_DIR / "cloud.csv.gz", usecols=["cagr", "mdd"])
    return pd.concat([p, c], ignore_index=True)


def summary(pts: pd.DataFrame, period: str = "2005") -> pd.DataFrame:
    """One row per B × family: how many points, the share inside −15 %, the worst MDD, the CAGR range."""
    p = pts[(pts["기간"] == period) & pts["배분"].isin(NAMES)]
    rows = []
    for (name, fam), g in p.groupby(["배분", "묶음"], sort=False):
        rows.append({"배분": name, "시험": FAMILIES[fam].label, "점 수": len(g), "−15% 안 (%)": 100 * g["−15% 안"].mean(),
                     "최악 MDD (%)": 100 * g["mdd"].min(), "CAGR 최저 (%)": 100 * g["cagr"].min(),
                     "CAGR 중앙 (%)": 100 * g["cagr"].median(), "CAGR 최고 (%)": 100 * g["cagr"].max(), "_order": list(FAMILIES).index(fam)})
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(["배분", "_order"]).drop(columns="_order").reset_index(drop=True)


def centre(pts: pd.DataFrame, name: str, period: str) -> pd.Series | None:
    c = pts[(pts["배분"] == name) & (pts["묶음"] == "centre") & (pts["기간"] == period)]
    return c.iloc[0] if len(c) else None


__all__ = ["NAMES", "REFERENCE", "OTHERS", "PERIODS", "WEIGHTS", "RULES", "FAMILIES", "points", "stage2_extra", "background", "summary", "centre"]
