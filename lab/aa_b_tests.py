"""Every committed result that re-ran the static allocations B2–B5 (or allocations next to them), gathered as
CAGR / MDD points for the page 'B2–B5 시험 모음' (user request, 2026-10-27: "지금까지 B2–B5에 관련해서 테스트한 건 따로
다시 정리해서 한 눈에 MDD와 CAGR을 볼 수 있도록").

Nothing here runs a backtest: it only reads the result files of the studies that tested B2–B5
(A32 stage 2 re-run on real prices = A35, A37 local study, A38 trend filter). Values are fractions (0.107 = 10.7 %).
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
SEARCH_DIR = sr.RESULTS_REAL_DIR  # A31 stage 1 on real prices (A35), the grey background

NAMES = ("B2", "B3", "B4", "B5")
REFERENCE = "안정형 40"
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
                     "(필터 8·10·12 = 주식 상품만, Faber 원형 10 = 금·채권을 포함한 모든 위험자산)", "A38 (2026-10-27_trend_filter/named.csv)", ("2005", "2000-09")),
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


def points() -> pd.DataFrame:
    """All points, both periods. ``묶음`` is a key of FAMILIES; ``배분`` is B2–B5 or the reference 안정형 40."""
    parts = [stage2_points(), neighbour_points(), local_points(), swap_points(), filter_points()]
    out = pd.concat([p for p in parts if len(p)], ignore_index=True)
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


__all__ = ["NAMES", "REFERENCE", "PERIODS", "WEIGHTS", "RULES", "FAMILIES", "points", "stage2_extra", "background", "summary", "centre"]
