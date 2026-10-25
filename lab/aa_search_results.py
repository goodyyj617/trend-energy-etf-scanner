"""Allocation lab: the global static search (DECISIONS A31) — its universe, and its saved results for the app.

The A31 universe (30 products: 28 Korean-listed ETFs and two synthetic unhedged US treasuries) lives here so that
`scripts/aa_global_search.py` (the run) and the app page '전역 탐색 결과' (the viewer) share one definition.

Result files (docs/research/allocation/2026-10-25_global_search/):
    passing.csv.gz        every sampled allocation within the MDD line (weights as "code:pct;…")
    cloud.csv.gz          a random sample of the allocations outside the line (CAGR and MDD only, for the scatter)
    top300_refined.csv    the greedy-refined list (출처 = 다듬음; all walked to the line, not a candidate list)
    presets.csv           the registered presets on the same simulator
    representatives.csv   hand-picked representatives with sensitivity checks
    summary.json          run parameters and campaign-level dials

Nothing here imports streamlit; the app adds the screen.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from lab.aa_assets import PRODUCTS, Product, _p, short
from lab.aa_engine import AllocConfig
from lab.aa_score import CRISIS_WINDOWS

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "docs" / "research" / "allocation" / "2026-10-25_global_search"
MEMO_PATH = ROOT / "docs" / "research" / "allocation" / "2026-10-25_global_search.md"

START, END = "2005-01-03", "2026-10-16"
REBALANCE, COST_BPS = "Q", 10.0
MDD_LINE = -0.15  # the only mandatory cut (user rule, 2026-10-25)
TIMING_LINE = -0.20  # A30
ROLL_MIN_SHARE = 0.80
WINDOWS = {"2008": CRISIS_WINDOWS["금융위기 2007–09"], "2011": CRISIS_WINDOWS["유럽 재정위기 2011"],
           "2018": CRISIS_WINDOWS["2018년 4분기"], "2020": CRISIS_WINDOWS["코로나 2020"], "2022": CRISIS_WINDOWS["금리 급등 2022"]}

# Two synthetic products (A31): unhedged plain US treasuries. Real products to verify before any preset:
# 10년 → TIGER 미국채10년선물 305080 (2018-08) / KODEX 미국채10년선물 308620 (2018-10); 30년 → ACE 미국30년국채액티브 453850 (2023-03).
SYNTHETIC: dict[str, Product] = {
    "SYN_IEF_U": _p("SYN_IEF_U", "미국10년국채 환노출(가상)", "채권", [("IEF", 1)], "unhedged", 0.15, "2018-08-30",
                    note="가상 상품: IEF × 원/달러. 실제 상품(TIGER 305080 / KODEX 308620)은 확인 전."),
    "SYN_TLT_U": _p("SYN_TLT_U", "미국30년국채 환노출(가상)", "채권", [("TLT", 1)], "unhedged", 0.15, "2023-03-14",
                    note="가상 상품: TLT × 원/달러. 실제 상품(ACE 453850)은 확인 전."),
}
SHORT_NAMES = {"SYN_IEF_U": "미국10년 환노출*", "SYN_TLT_U": "미국30년 환노출*"}
CLASSES: dict[str, list[str]] = {
    "미국 주식": ["379800", "449180", "379810", "245340", "463640", "453630", "453640", "218420"],
    "그 밖 주식": ["251350", "291890", "069500", "280930"],
    "환헤지 채권": ["0091C0", "484790"],
    "환노출 채권": ["468370", "468630", "SYN_IEF_U", "SYN_TLT_U"],
    "금·실물": ["411060", "132030", "352560", "182480"],
    "현금·달러": ["459580", "261240"],
    "레버리지": ["225040", "409820", "418660", "452250", "225130", "122630"],
}
EQUITY_CLASSES = ("미국 주식", "그 밖 주식")
LEV_CLASS = "레버리지"
SYMBOLS = [s for c in CLASSES.values() for s in c]
CLASS_OF = {s: c for c, syms in CLASSES.items() for s in syms}
ALL_PRODUCTS = {**PRODUCTS, **SYNTHETIC}


def name_of(code: str) -> str:
    return SHORT_NAMES.get(code, short(code))


_CODE_OF = {name_of(s): s for s in SYMBOLS}
_POS = {s: i for i, s in enumerate(SYMBOLS)}

# product groups a buyer may want to leave out (the subsets of frontier.md)
EXCLUDE_GROUPS: dict[str, list[str]] = {
    "나스닥100": [s for s in SYMBOLS if "나스닥" in name_of(s)],
    "방어·경기 섹터": [s for s in SYMBOLS if any(t in name_of(s) for t in ("유틸리티", "필수소비재", "헬스케어", "에너지"))],
    "레버리지 상품": list(CLASSES[LEV_CLASS]),
    "가상 상품": list(SYNTHETIC),
}
GROUP_HELP = {
    "나스닥100": "KODEX 미국나스닥100과 나스닥100 2배 상품. 탐색에서 MDD 대비 수익을 끌어올린 재료가 거의 이것이었습니다(메모 §3).",
    "방어·경기 섹터": "S&P500 유틸리티·필수소비재·헬스케어·에너지. 대용 XLU·XLP·XLV·XLE는 1998년부터 있습니다.",
    "레버리지 상품": "2배 상품 6종(합계 20% 이하로만 뽑음). ISA에서 사려면 레버리지 ETF 사전교육과 기본예탁금이 필요합니다.",
    "가상 상품": "환노출 미국채 10년·30년(가상, IEF·TLT × 원/달러). 실제 상품(TIGER 305080 / KODEX 308620 / ACE 453850)은 확인 전입니다.",
}

FLAG_TEXT = {
    "반기": "2005–2014 또는 2015–2026 한쪽만 떼어 봤을 때 MDD가 −15% 밖",
    "위기": "등록된 위기 구간(2008·2011·2018·2020·2022) 중 하나의 낙폭이 −15% 밖",
    "보류": "마지막 15% 기간(보류 구간)만 떼어 봤을 때 MDD가 −15% 밖",
    "창": "5년 롤링 창 중 MDD −15% 안에 든 비율이 80% 미만 (상위 2,000개만 계산)",
    "이웃": "두 상품 사이에서 5%p를 옮긴 이웃 배분 중 하나라도 −15% 밖 (상위 300개만 계산)",
    "날짜": "리밸런싱 날짜를 0·1·5·10·15거래일 미룬 최악 MDD가 −20% 밖 (상위 300개만 계산)",
}

APP_COLS = ["cagr", "mdd", "sharpe_excess", "calmar", "turnover", "cagr_h1", "mdd_h1", "cagr_h2", "mdd_h2",
            "dd_2008", "dd_2011", "dd_2018", "dd_2020", "dd_2022", "cagr_2008", "mdd_2008", "cagr_2010", "mdd_2010",
            "cagr_holdout", "mdd_holdout", "cagr_sel", "mdd_sel", "roll_share", "roll_med_cagr", "nb_worst_mdd",
            "timing_worst", "pbo_oos_within"]


# --------------------------------------------------------------------------- weights as text

def describe(w: np.ndarray) -> str:
    """Display text, e.g. '미국나스닥100 30 · ACE KRX금현물 25' (whole percents over SYMBOLS)."""
    return " · ".join(f"{name_of(s)} {int(v)}" for s, v in zip(SYMBOLS, w) if v > 0)


def code_text(w: np.ndarray) -> str:
    """Compact storage text, e.g. '379810:30;411060:25'."""
    return ";".join(f"{s}:{int(v)}" for s, v in zip(SYMBOLS, w) if v > 0)


def parse_codes(text: str) -> dict[str, int]:
    out = {}
    for part in str(text).split(";"):
        code, v = part.split(":")
        if code not in _POS:
            raise ValueError(f"탐색 유니버스에 없는 상품코드: {code}")
        out[code] = int(v)
    return out


def parse_names(text: str) -> dict[str, int]:
    out = {}
    for part in str(text).split(" · "):
        name, v = part.rsplit(" ", 1)
        if name not in _CODE_OF:
            raise ValueError(f"모르는 상품 이름: {name}")
        out[_CODE_OF[name]] = int(v)
    return out


def to_vector(weights: dict[str, int]) -> np.ndarray:
    w = np.zeros(len(SYMBOLS), dtype=int)
    for code, v in weights.items():
        w[_POS[code]] = v
    return w


def matrix(dicts) -> np.ndarray:
    return np.vstack([to_vector(d) for d in dicts]) if len(dicts) else np.zeros((0, len(SYMBOLS)), dtype=int)


def weights_of(w: np.ndarray) -> dict[str, int]:
    return {s: int(v) for s, v in zip(SYMBOLS, w) if v > 0}


# --------------------------------------------------------------------------- derived columns

def class_shares(W: np.ndarray) -> pd.DataFrame:
    out = {}
    for c, syms in CLASSES.items():
        out[c] = W[:, [_POS[s] for s in syms]].sum(axis=1)
    df = pd.DataFrame(out)
    df["주식"] = df["미국 주식"] + df["그 밖 주식"]
    df["채권"] = df["환헤지 채권"] + df["환노출 채권"]
    unhedged = [i for i, s in enumerate(SYMBOLS) if ALL_PRODUCTS[s].fx == "unhedged"]
    df["환노출"] = W[:, unhedged].sum(axis=1)
    df["상품 수"] = (W > 0).sum(axis=1)
    return df


def flags(df: pd.DataFrame) -> pd.Series:
    """⚠ flags from the information columns (never used to drop a candidate). Missing values raise no flag."""
    out = []
    for _, r in df.iterrows():
        f = []
        if r.get("mdd_h1", 0) < MDD_LINE or r.get("mdd_h2", 0) < MDD_LINE:
            f.append("반기")
        if any(r.get(f"dd_{k}", 0) < MDD_LINE for k in WINDOWS):
            f.append("위기")
        if r.get("mdd_holdout", 0) < MDD_LINE:
            f.append("보류")
        if r.get("roll_share", 1) < ROLL_MIN_SHARE:
            f.append("창")
        if r.get("nb_worst_mdd", 0) < MDD_LINE:
            f.append("이웃")
        if r.get("timing_worst", 0) < TIMING_LINE:
            f.append("날짜")
        out.append(" ".join(f))
    return pd.Series(out, index=df.index, dtype=object)


# --------------------------------------------------------------------------- files

def app_tables(full: pd.DataFrame, cloud_n: int = 12_000, seed: int = 2026) -> tuple[pd.DataFrame, pd.DataFrame]:
    """From the full candidate table (one row per sampled allocation, weights in '비중' display text):
    every row within the MDD line with APP_COLS and compact weights, and a random sample of the rest."""
    inside = full["mdd"] >= MDD_LINE
    passing = full.loc[inside, [c for c in APP_COLS if c in full.columns]].round(4)
    passing["출처"] = full.loc[inside, "출처"]
    passing["w"] = [code_text(to_vector(parse_names(t))) for t in full.loc[inside, "비중"]]
    rest = full.loc[~inside, ["cagr", "mdd"]]
    cloud = rest.sample(min(cloud_n, len(rest)), random_state=seed).round(4) if len(rest) else rest
    return passing, cloud


def _with_derived(df: pd.DataFrame, W: np.ndarray) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    shares = class_shares(W)
    for c in shares.columns:
        df[c] = shares[c].to_numpy()
    df["비중"] = [describe(w) for w in W]
    df["⚠"] = flags(df)
    return df


@dataclass
class SearchResults:
    cands: pd.DataFrame  # passing raw samples + refined (출처 다듬음)
    W: np.ndarray  # whole-percent weights over SYMBOLS, one row per cands row
    presets: pd.DataFrame
    preset_W: np.ndarray
    cloud: pd.DataFrame
    reps: pd.DataFrame
    summary: dict = field(default_factory=dict)


def load_results(folder: Path | str = RESULTS_DIR) -> SearchResults:
    folder = Path(folder)
    passing = pd.read_csv(folder / "passing.csv.gz", dtype={"w": str, "출처": str})
    W_pass = matrix([parse_codes(t) for t in passing["w"]])
    passing = passing.drop(columns=["w"])
    refined = pd.read_csv(folder / "top300_refined.csv")
    W_ref = matrix([parse_names(t) for t in refined["비중"]])
    keep = [c for c in APP_COLS + ["seed_rank", "refine_rounds", "refine_gain"] if c in refined.columns]
    refined = refined[keep].assign(출처="다듬음")
    cands = pd.concat([passing, refined], ignore_index=True)
    W = np.vstack([W_pass, W_ref])
    presets = pd.read_csv(folder / "presets.csv")
    preset_W = matrix([parse_names(t) for t in presets["비중"]])
    presets = _with_derived(presets[["이름"] + [c for c in APP_COLS if c in presets.columns]], preset_W)
    reps = pd.read_csv(folder / "representatives.csv")
    cloud = pd.read_csv(folder / "cloud.csv.gz")
    summary = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
    return SearchResults(_with_derived(cands, W), W, presets, preset_W, cloud, reps, summary)


# --------------------------------------------------------------------------- filters and frontier

def exclude_mask(W: np.ndarray, codes) -> np.ndarray:
    """True for rows that hold none of `codes`."""
    idx = [_POS[c] for c in codes if c in _POS]
    if not idx:
        return np.ones(len(W), dtype=bool)
    return W[:, idx].sum(axis=1) == 0


def band_of(mdd: float) -> int:
    """Upper edge (in %) of the 1 %p MDD band (lower, upper] that holds `mdd`, as in frontier.md."""
    return int(math.ceil(round(mdd * 100, 9)))


def band_label(hi: int) -> str:
    return f"({hi - 1}, {hi}]"


def band_frontier(df: pd.DataFrame, lo: float = MDD_LINE) -> pd.DataFrame:
    """Per 1 %p MDD band from the line upward: number of rows, best CAGR and the row label of the best."""
    cols = ["구간", "상한", "후보 수", "최고 CAGR", "그 후보 MDD", "행"]
    if df.empty:
        return pd.DataFrame(columns=cols)
    floor = band_of(lo) + 1  # the line itself belongs to the first band
    bands = np.maximum([band_of(v) for v in df["mdd"]], floor)
    rows = []
    for hi in sorted(set(bands)):
        part = df[bands == hi]
        best = part["cagr"].idxmax()
        rows.append({"구간": band_label(hi), "상한": hi, "후보 수": len(part), "최고 CAGR": float(part.loc[best, "cagr"]),
                     "그 후보 MDD": float(part.loc[best, "mdd"]), "행": best})
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- hand-off to the engine and the research page

def buyable(weights: dict) -> bool:
    return all(c in PRODUCTS for c in weights)


def candidate_config(name: str, weights: dict, start: str = START, end: str = END) -> AllocConfig:
    """The engine setting the search used for every candidate (quarterly, 10 bp, no band/overlay/leverage)."""
    return AllocConfig(name, {k: float(v) for k, v in weights.items()}, start, end, rebalance=REBALANCE, band=0.0,
                       cost_bps=COST_BPS)


def research_state(prefix: str, weights: dict, name: str, end: str, start: str = START) -> dict:
    """Session-state values that load this candidate into the strategy builder of the research page."""
    if not buyable(weights):
        raise ValueError("가상 상품이 들어 있어 전략 연구 페이지로 보낼 수 없습니다(국내 상품으로 확인 전).")
    state = {}
    for code in PRODUCTS:
        state[f"{prefix}_on_{code}"] = code in weights
        state[f"{prefix}_w_{code}"] = int(weights.get(code, 0))
    state.update({f"{prefix}_name": name, f"{prefix}_period": "custom", f"{prefix}_start": date.fromisoformat(start),
                  f"{prefix}_end": date.fromisoformat(end), f"{prefix}_rebalance": REBALANCE, f"{prefix}_band": 0.0,
                  f"{prefix}_cost": COST_BPS, f"{prefix}_ov_kind": "none", f"{prefix}_lev": 1.0})
    return state
