"""Stage T: the trading-cost assumption (DECISIONS 100).

    python -m lab.cost_study --stage repro|measure|accounts|report --panel DIR --stage-s DIR --out DIR
        [--crsp DIR] [--workers N] [--feature-workers N]

Every account so far paid a flat 20 bp one way (40 bp in the candidate conditions) and the SPY200
benchmark paid nothing on its switches. This runner measures an effective spread per year x cap-rank
bucket x raw-price bucket with the EDGE estimator (lab/edge_spread.py, Ardia, Guidotti & Kroencke 2024)
on the panel's daily OHLC, and replays Stage S's raw-basis grids (lab/rebase_study.py's job lists and
verdict function, unchanged) under four cost models:
  T0  flat 20 bp (40 bp for the stage rule's candidate cells), benchmarks as before   (repro)
  TZ  zero cost for the accounts and every benchmark                                (bound)
  T1  the measured cell cost per fill, SPY200 paying SPY's cell on each switch, the
      top-N EW paying member cells on membership changes and the members' mean cell
      cost on regime switches, the universe EW costless as before                   (primary)
  T2  2 x T1                                                                        (sensitivity)
Inputs are Stage S's raw-basis signal tables, base table, eligible mask and regime states (--stage-s,
the folder lab_crsp/stage_s_raw); the job functions draw the same random orders, placebo shifts and
pseudo restrictions. Only the cost a fill pays changes: pick_study.replay_risk / calendar_weighted get
a per-fill lookup (CellCost) in place of the number the job functions pass.

Stages (each skips finished files; later stages refuse to run unless repro.json passed):
  repro     the fill-cell code of every day x symbol (cap rank and raw close of t-1); T0 through the
            lookup path (constant tables) for bundles A-D, the 40 bp cells and E = Stage S's raw result
            tables to 1e-9; T0's verdicts = Stage S's.
  measure   EDGE per stock x quarter, the cell cost table, checks (a) / (b), cross-checks.
  accounts  TZ, T1, T2 (T1 / T2 only if (a) and (b) pass): bundles A-E and the cost the accounts paid.
  report    verdicts 3-6, the predictions, aggregates and the six figures.
Only dates 1993-02-01 - 2023-12-31 enter a metric (EDGE: 1993-01 - 2023-12); nothing later is computed
into a metric or printed.

실행 전에 정한 세부 (설계가 비워 둔 것, 계좌 단계 전에 커밋):
  - EDGE 입력: 종목 × 달력 분기마다 그 분기 패널 거래일 전부의 수정 시 · 고 · 저 · 종가(float32 → float64).
    네 값 중 하나라도 없는 날은 네 값을 모두 빈 날(NaN)로 바꿔 넣고, 네 값이 모두 있는 날이 40일 이상인 종목 ·
    분기만 씁니다. 상장폐지 봉(시 = 고 = 저 = 종)도 패널에 있는 그대로 둡니다. SPY는 넣지 않습니다.
    sign=True의 s로 q = sign(s) · s².
  - 종목 · 분기의 칸: 시총 구간은 네 값이 모두 있는 날들의 시총 순위(panel cap_rank) 중앙값(순위 없는 날 =
    3000 밖), 주가 구간은 같은 날들의 실제 종가(raw_close) 중앙값(NaN 제외). 실제 종가가 하나도 없으면 주가
    칸에는 들어가지 않고 연도 × 시총 구간(주가 합침)에만 들어갑니다.
  - 구간 경계: 시총 순위 ≤500 / ≤1000 / ≤2000 / ≤3000 / 그 밖, 실제 주가 <5 / <10 / <20 / <50 / 그 이상.
  - 칸 값: S = √max(q 평균, 0), 편도 = S/2. 종목 · 분기 100개 미만이면 연도 × 시총 구간, 그것도 100개 미만이면
    시대 × 시총 구간 값. 하한 1bp, 상한 500bp는 마지막에 한 번 적용.
  - 점검 (가) · (나): 연도 × 시총 구간(주가 합침)의 편도 S/2(하한 · 상한 전, 대체 없음)로 판정. 시대 값 = 그 시대
    연도 값들의 단순 평균. 부등호는 모두 엄격(같으면 실패). '3000 밖'은 판정에 넣지 않습니다.
  - 체결 칸: 창의 t일 체결은 t의 연도(1993 이전 · 2023 이후는 1993 · 2023 표), t−1일의 시총 순위(없으면 3000 밖),
    t−1일까지의 마지막 실제 종가(앞 값 채움; 없으면 연도 × 시총 구간 값)로 정합니다. 기준선의 종가 체결(구성 변화,
    국면 전환)은 그 수익이 들어가는 날 t의 같은 칸을 씁니다.
  - SPY(200일선 기준선)는 시총 순위가 없으므로 시총 1–500 구간, 주가는 SPY의 실제 종가 구간으로 둡니다.
  - T2 = T1 최종 비용(하한 · 상한 적용 뒤) × 2. 상한을 다시 적용하지 않습니다.
  - 동일비중 기준선(T1 · T2): 유니버스 전체는 비용 없음. 동일비중 × 국면은 전환일에 그날 보유 종목(t−1 매수 가능
    + 그날 수익 있음)의 칸 비용 평균. 범위별 상위 N은 들어오고 나가는 종목이 자기 칸 비용(비중 1/n), 그 × 국면의
    전환은 그날 범위 보유 종목의 칸 비용 평균.
  - T0는 숫자를 그대로 넘기지 않고 모든 칸이 20bp(후보 칸은 40bp)인 표로 같은 조회 경로를 지나갑니다.
    40bp 칸은 T0에서만 단계 규칙대로 고른 작업을 돌리고, T 모형에서는 따로 돌리지 않습니다.
  - 후보(판정 6): T1 결과를 20bp 자리에, T2 결과를 40bp 자리에 넣어 regime_study / large_study의 후보 판정을
    그대로 씁니다. 40bp 칸이 20bp 기준선과 견줘졌던 것처럼 T2 칸은 T1 기준선과 견줍니다.
  - 오차 크기(판정 3): 시스템 값(regime_study.systems / large_study.systems: 매수 15개 중앙값)의 T1 − T0 차이,
    국면 없는 8개와 A1 96개(설정 4 × 매도 2 × A1 국면 4 × 범위 3) 각각의 중앙값, 세 기간.
  - 따로 보는 숫자(판정 5): A1 96개 중 같은 모형의 SPY 200일선 + 국채 맞춘 수익보다 엄격하게 큰 수.
  - K1–K5: rebase_study.judge 그대로(같은 모형의 기준선). 결론마다 TZ · T1 · T2 모두 유지 = '비용에 견고',
    T1 유지 · TZ나 T2에서 바뀜 = '유지, 비용 수준에 민감', T1에서 바뀜 = '바뀜'.
  - 계좌가 실제로 낸 비용: Q 격자(U3000 · U1000 · U500 × 국면 없음 + A1 넷, 설정 4, 매도 2, 매수 15 + E0의
    무작위 순서 전부)의 체결마다 (체결 금액, 비용)을 모아 Σ금액 × 비용 / Σ금액(편도 bp). 한 계좌의 범위는 그
    계좌가 살 수 있는 줄의 신호일 시총 구간 중 가장 넓은 것으로 정합니다(작업마다 범위 셋이 같은 수인지 확인).
    제한 없는 계좌는 비중을 금액으로 씁니다. 시대 1993–2000 / 2001–2007 / 2008–2023과 예상 비교용 2010–2023.
  - 교차 확인 ②: 호가 단위 1993-01 – 1997-06 1/8달러, 1997-07 – 2000-12 1/16달러(기억에 의존, 확인하지 못함).
    종목 · 분기마다 (단위/2) ÷ 실제 종가 중앙값을 칸별로 평균해 칸 비용과 나란히 적습니다.
  - 교차 확인 ①: 원자료에 종가 매수 · 매도 호가 열이 없음(BIDLO · ASKHI는 일중 저가 · 고가, 거래 없는 날에만
    호가)이라 하지 않습니다. ③: 자료 내려받기 승인 대기라 하지 않습니다.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import account_study as ns, basis_study as bs, crsp, entry_study as es, figures, goal_study as gs
from . import large_study as ls, pair_study as pj, pick_study as ph, rebase_study as rb, regime_study as rs
from .crsp_study import load_panel_with_cash
from .data import BENCHMARK
from .edge_spread import edge_rows

# ---- registered (DECISIONS 100): fixed before any result
MODELS = ("T0", "TZ", "T1", "T2")
T_MODELS = ("TZ", "T1", "T2")
CAP_EDGES = (500.0, 1000.0, 2000.0, 3000.0)  # rank buckets 0..4 (4 = beyond 3000)
CAP_LABELS = ("1–500", "501–1000", "1001–2000", "2001–3000", "3000 밖")
PRICE_EDGES = (5.0, 10.0, 20.0, 50.0)  # raw-price buckets 0..4
PRICE_LABELS = ("5달러 미만", "5–10", "10–20", "20–50", "50 이상")
N_CAP, N_PRICE = 5, 5
CODE_POOLED = N_CAP * N_PRICE  # codes 25..29: price unknown -> the year x cap value
N_CODES = CODE_POOLED + N_CAP
YEARS = list(range(1993, 2024))
ERAS = {"1993-2000": (1993, 2000), "2001-2007": (2001, 2007), "2008-2023": (2008, 2023)}
PAID_ERAS = dict(ERAS, **{"2010-2023": (2010, 2023)})
MIN_DAYS = 40
MIN_CELL = 100
FLOOR_BP, CAP_BP = 1.0, 500.0
LARGE_PP = 0.005  # verdict 3: any |median T1 - T0| >= 0.5 pp
ERR_PERIODS = ("1993-2023", "1993-2007", "2008-2023")
TZ_GAP_COUNT = 10  # verdict 5: >= 10 of 96 above SPY200 in 2008-2023 under TZ
EDGE_START, EDGE_END = "1993-01-01", "2023-12-31"
TICK_CHANGE = pd.Timestamp("1997-07-01")  # 1/8 -> 1/16 (from memory, unverified)
TRIALS = rb.TRIALS  # 74,174, unchanged
_W: dict = {}


def log(m: str) -> None:
    rb.log(m)


# --------------------------------------------------------------------------- the per-fill lookup

def cap_bucket(rank: np.ndarray) -> np.ndarray:
    r = np.where(np.isfinite(rank), rank, np.inf)
    return np.searchsorted(np.asarray(CAP_EDGES), r, side="left").astype(np.uint8)


def price_bucket(price: np.ndarray) -> np.ndarray:
    """0..4, or 255 when the price is unknown."""
    p = np.asarray(price, dtype=float)
    b = np.searchsorted(np.asarray(PRICE_EDGES), np.where(np.isfinite(p), p, 0.0), side="right").astype(np.uint8)
    return np.where(np.isfinite(p) & (p > 0), b, 255).astype(np.uint8)


def fill_codes(cap_rank: np.ndarray, raw_close: np.ndarray, spy_col: int | None) -> np.ndarray:
    """Cell code of a fill on day t (rows = the panel's days, columns = its symbols): cap bucket of t-1's
    rank and price bucket of the last raw close up to t-1; price unknown -> CODE_POOLED + cap bucket.
    Row 0 has no t-1 (beyond 3000, price unknown). SPY: cap bucket 0."""
    rc = pd.DataFrame(raw_close).ffill().to_numpy(dtype=float)
    cb = cap_bucket(cap_rank.astype(float))
    if spy_col is not None:
        cb[:, spy_col] = 0
    pb = price_bucket(rc)
    code = np.where(pb == 255, CODE_POOLED + cb, cb * N_PRICE + pb).astype(np.uint8)
    out = np.empty_like(code)
    out[0] = CODE_POOLED + N_CAP - 1
    out[1:] = code[:-1]
    return out


def year_index(dates: pd.DatetimeIndex) -> np.ndarray:
    return (np.clip(dates.year.to_numpy(), YEARS[0], YEARS[-1]) - YEARS[0]).astype(np.int16)


class CellCost:
    """One-way cost of a fill on day t (panel row t = window row t - w0) in symbol column j:
    tab[year of t, code[t, j]] (a fraction). at / take read window coordinates (the accounts' arrays)."""

    def __init__(self, code: np.ndarray, yi: np.ndarray, tab: np.ndarray, w0: int = 0):
        self.code, self.yi, self.tab, self.w0 = code, yi, np.asarray(tab, dtype=float), int(w0)

    def at(self, i: int, j: int) -> float:
        t = i + self.w0
        return float(self.tab[self.yi[t], self.code[t, j]])

    def take(self, i, j) -> np.ndarray:
        t = np.asarray(i) + self.w0
        return self.tab[self.yi[t], self.code[t, np.asarray(j)]]

    def dense(self, rows: int | None = None) -> np.ndarray:
        """Panel rows x symbols (the first `rows` panel rows)."""
        n = self.code.shape[0] if rows is None else rows
        out = np.empty(self.code[:n].shape, dtype=float)
        for a in range(0, n, 512):
            out[a:a + 512] = self.tab[self.yi[a:a + 512, None], self.code[a:a + 512]]
        return out

    def column(self, j: int) -> np.ndarray:
        return self.tab[self.yi, self.code[:, j]]


def model_tab(model: str, cost_dir: Path, cost: float = pj.COST_BPS / 10_000) -> np.ndarray:
    """The (years x codes) table of a model; T0 = every cell `cost` (the job's 20 or 40 bp)."""
    if model == "T0":
        return np.full((len(YEARS), N_CODES), cost)
    if model == "TZ":
        return np.zeros((len(YEARS), N_CODES))
    t1 = np.load(cost_dir / "tab_T1.npy")
    return t1 if model == "T1" else 2.0 * t1


def load_codes(cost_dir: Path, mmap: bool = True) -> tuple[np.ndarray, np.ndarray]:
    return np.load(cost_dir / "codes.npy", mmap_mode="r" if mmap else None), np.load(cost_dir / "years.npy")


# --------------------------------------------------------------------------- workers

def _c_init(panel_dir: str, eligible: str, dirs: dict, cfg: dict, model: str, cost_dir: str):
    rb._w_init(panel_dir, eligible, dirs, cfg)
    code, yi = load_codes(Path(cost_dir))
    w0 = ls._W["w0"]
    tabs = {c: model_tab(model, Path(cost_dir), c) for c in ((pj.COST_BPS / 10_000, pj.CHECK_COST_BPS / 10_000) if model == "T0"
                                                              else (pj.COST_BPS / 10_000,))}
    _W.update(model=model, code=code, w0=w0, look={c: CellCost(code, yi, t, w0) for c, t in tabs.items()}, paid=None, slots=None)
    orig_rr, orig_cw = ph.replay_risk, ph.calendar_weighted
    _W.update(orig_rr=orig_rr, orig_cw=orig_cw)
    ph.replay_risk = _replay_risk
    ph.calendar_weighted = _calendar_weighted
    if model != "T0":  # the window shares of every job are judged against this model's benchmarks
        px, wdates, cash = ls._W["px"], ls._W["wdates"], ls._W["cash"]
        spy = spy_cost(model, Path(cost_dir), px, w0, len(wdates))
        gs._W["bench"] = gs.benchmarks_table(px, wdates, w0, cash, spy)[1]


def _lookup(cost: float) -> CellCost:
    if cost not in _W["look"]:
        raise ValueError(f"모형 {_W['model']}에 없는 비용 {cost}")
    return _W["look"][cost]


def _band_of(ei: np.ndarray, sj: np.ndarray) -> str:
    """The widest signal-day cap bucket among the rows an account may buy (code of the entry day = t-1 rank)."""
    if not len(ei):
        return "empty"
    c = np.asarray(_W["code"][np.asarray(ei) + _W["w0"], np.asarray(sj)], dtype=int)
    cb = int(np.where(c >= CODE_POOLED, c - CODE_POOLED, c // N_PRICE).max())
    return "U500" if cb == 0 else "U1000" if cb == 1 else "U3000"


def _record(days: np.ndarray, value: np.ndarray, cost: np.ndarray, band: str, slots) -> None:
    yrs = YEARS[0] + np.asarray(_W["look_yi"][np.asarray(days) + _W["w0"]], dtype=int)
    df = pd.DataFrame({"year": yrs, "value": value, "cv": value * cost})
    g = df.groupby("year").agg(value=("value", "sum"), cv=("cv", "sum"), fills=("value", "size"))
    for y, r in g.iterrows():
        k = (slots, band, int(y))
        a = _W["paid"].setdefault(k, [0.0, 0.0, 0])
        a[0] += r["value"]
        a[1] += r["cv"]
        a[2] += int(r["fills"])


def _replay_risk(si, ei, sj, sale, closed, w, key, O, LC, cash_r, cost, **kw):
    look = _lookup(cost)
    if _W["paid"] is None:
        return _W["orig_rr"](si, ei, sj, sale, closed, w, key, O, LC, cash_r, look, **kw)
    fills: list = []
    eq, tr = _W["orig_rr"](si, ei, sj, sale, closed, w, key, O, LC, cash_r, look, fills=fills, **kw)
    if fills:
        f = np.asarray(fills, dtype=float)
        _record(f[:, 0].astype(int), f[:, 2], f[:, 3], _band_of(ei, sj), kw.get("slots", ph.SLOTS))
    return eq, tr


def _calendar_weighted(ei, sale, closed, sj, w, O, C, cash, cost):
    look = _lookup(cost)
    out = _W["orig_cw"](ei, sale, closed, sj, w, O, C, cash, look)
    if _W["paid"] is not None and len(ei):
        ex = closed & (sale < len(cash))
        days = np.concatenate([ei, sale[ex]])
        val = np.concatenate([w, w[ex]])
        _record(days, val, look.take(days, np.concatenate([sj, sj[ex]])), "unlimited", "none")
    return out


def _c_job(job):
    kind, j = job
    mode = j[-1] if kind in ("P", "Q", "O") else "grid"
    _W["paid"] = {} if _W["model"] != "T0" and ((kind == "Q" and mode == "grid") or kind == "U") else None
    _W["look_yi"] = next(iter(_W["look"].values())).yi
    job_, parts, sec = rb._w_job(job)
    if _W["paid"] is not None:
        st = j[2] if kind == "Q" else None
        rows = [{"setting": ns.setting_name(st) if st else "제한 없음", "band": b, "year": y, "value": v[0], "cost_value": v[1], "fills": v[2],
                 "slots_seen": s} for (s, b, y), v in _W["paid"].items()]
        parts["paid"] = pd.DataFrame(rows).assign(gen=j[0] if kind == "Q" else ("E0" if j[0].startswith("R0") else j[0]),
                                                  exit=j[1], kind=kind)
    _W["paid"] = None
    return job_, parts, sec


def run_pool(jobs: list, mdir: Path, a, eligible: str, dirs: dict, cfg: dict, model: str, workers: int) -> None:
    todo = [j for j in jobs if not rb.done(mdir, j)]
    log(f"[{model}] 계좌 작업 {len(todo)}개 남음 (전체 {len(jobs)})")
    if not todo:
        return
    size = pd.read_parquet(cfg["base"], columns=["src"]).groupby("src").size()
    todo.sort(key=lambda j: -rb.job_cost(j, size))
    t0 = time.time()
    with rb.use_basis("raw"), ProcessPoolExecutor(max_workers=workers, initializer=_c_init,
                                                  initargs=(a.panel, eligible, dirs, cfg, model, str(cost_dir(a)))) as ex:
        futs = [ex.submit(_c_job, j) for j in todo]
        for i, f in enumerate(as_completed(futs)):
            job, parts, sec = f.result()
            sub, name = rb.job_file(job)
            d = mdir / sub / "jobs"
            d.mkdir(parents=True, exist_ok=True)
            for k, frame in parts.items():
                if k in ("trades", "curve"):
                    if len(frame):
                        frame.to_parquet(d / f"{name}_{k}.parquet")
                elif k != "acc":
                    frame.to_csv(d / f"{name}_{k}.csv", index=False)
            parts["acc"].to_csv(d / f"{name}_acc.csv", index=False)  # last: marks the job done
            if (i + 1) % 20 == 0 or i + 1 == len(todo):
                el = time.time() - t0
                log(f"  [{model} {i + 1}/{len(todo)}] {name} {sec:.0f}s · 경과 {el / 60:.0f}분 · 남은 예상 {(len(todo) - i - 1) * el / (i + 1) / 60:.0f}분")


# --------------------------------------------------------------------------- paths

def cost_dir(a) -> Path:
    return Path(a.out) / "cost"


def s_paths(a) -> dict:
    s = Path(a.stage_s)
    raw = s / "raw"
    return {"root": s, "raw": raw, "eligible": str(raw / "eligible_daily.parquet"), "dirs": rb.table_dirs(raw / "tables"),
            "cfg": rb.cfg_of(raw, s / "regimes.parquet"), "regimes": s / "regimes.parquet"}


def model_dir(a, model: str) -> Path:
    return Path(a.out) / model


# --------------------------------------------------------------------------- benchmarks (bundle E)

def spy_cost(model: str, cdir: Path, px, w0: int, n: int):
    """SPY200's switching cost per window day (None = no cost, as before)."""
    if model == "T0":
        return None
    if model == "TZ":
        return 0.0
    code, yi = load_codes(cdir)
    j = int(px.cols.get_loc(BENCHMARK))
    return CellCost(code, yi, model_tab(model, cdir)).column(j)[w0:w0 + n]


def held_mean(cost: np.ndarray, member: np.ndarray, close: pd.DataFrame | None = None) -> np.ndarray:
    """Per panel row t: the mean cost of row t over the names held on t (members at t-1, with a return
    on t when close is given); 0 with none."""
    held = np.vstack([np.zeros((1, member.shape[1]), dtype=bool), member[:-1]])
    if close is not None:
        r = (close / close.ffill().shift(1) - 1.0).to_numpy(dtype=float)
        held = held & np.isfinite(r)
    n = held.sum(axis=1)
    return np.where(n > 0, np.where(held, np.nan_to_num(cost), 0.0).sum(axis=1) / np.maximum(n, 1), 0.0)


def benchmarks(a, model: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """rebase_study.benchmarks with this model's costs: Stage P's table (SPY mix, SPY200, EW, EW x each
    regime) and Stage Q's (the first three + per band top-N EW and that under each A1 regime)."""
    S = s_paths(a)
    states = pd.read_parquet(S["regimes"])
    with rb.use_basis("raw"):
        pj._init(a.panel, S["eligible"], S["dirs"], S["cfg"])
        px, wdates, w0, cash = pj._W["px"], pj._W["wdates"], pj._W["w0"], pj._W["cash"]
        n = len(wdates)
        cdir = cost_dir(a)
        btab, _ = gs.benchmarks_table(px, wdates, w0, cash, spy_cost(model, cdir, px, w0, n))
        sp = gs._setup(cash, wdates)
        ew = gs.bench_curves(px, wdates, w0, cash)["ew"]
        panel = load_panel_with_cash(Path(a.panel), "raw")
        close = bs.cut(panel.close)
        cap = bs.cut(panel.extra["cap_rank"]).reindex(columns=close.columns).to_numpy(dtype=float)
        del panel
        elig = pd.read_parquet(S["eligible"]).reindex(index=close.index, columns=close.columns, fill_value=False).to_numpy(dtype=bool)
        if model in ("T1", "T2"):
            code, yi = load_codes(cdir)
            assert code.shape[1] == close.shape[1]
            dense = CellCost(code, yi, model_tab(model, cdir)).dense(len(close))
            sw_univ = held_mean(dense, elig, close)[w0:w0 + n]
        else:
            dense = None
            sw_univ = pj.COST_BPS / 10_000 if model == "T0" else 0.0
        st_w = {r: states[r].reindex(wdates).fillna(False).to_numpy(dtype=bool) for r in rs.REGIMES}
        EQ = np.column_stack([rs.ew_regime(ew, cash, st_w[r], sw_univ) for r in rs.REGIMES])
        R = EQ[1:] / EQ[:-1] - 1.0
        gm = gs.goal_metrics(R, np.repeat(sp["c"][:, None], R.shape[1], axis=1), sp["years"], sp["wins"])
        ewr = [{"benchmark": f"ew_{r}", "label": f"동일비중 × {r} {rs.REGIMES[r]} (A2) + 국채",
                **{k: float(v[i]) for k, v in gm.items() if k != "win_cagr"}, **es.period_stats(pd.Series(EQ[:, i], index=wdates))}
               for i, r in enumerate(rs.REGIMES)]
        btab_p = pd.concat([btab, pd.DataFrame(ewr)], ignore_index=True)
        btab_b = band_benchmarks(close, cap, elig, wdates, w0, cash, states, model, dense)
        btab_q = pd.concat([btab, btab_b], ignore_index=True)
    return btab_p, btab_q


def band_benchmarks(close: pd.DataFrame, cap: np.ndarray, e: np.ndarray, wdates, w0: int, cash: np.ndarray, states: pd.DataFrame,
                    model: str, dense: np.ndarray | None) -> pd.DataFrame:
    """large_study.band_benchmarks with this model's costs (T0: the same 20 bp; TZ: none; T1 / T2: the
    member cells)."""
    n = len(wdates)
    st_w = {r: states[r].reindex(wdates).fillna(False).to_numpy(dtype=bool) for r in ls.REGIMES}
    names, curves = [], []
    for b, limit in ls.BANDS.items():
        member = e if not np.isfinite(limit) else e & (cap <= limit)
        if dense is None:
            c = pj.COST_BPS / 10_000 if model == "T0" else 0.0
            full, sw = ls.top_n_ew(close, member, c), c
        else:
            full = ls.top_n_ew(close, member, dense)
            sw = held_mean(dense, member, close)[w0:w0 + n]
        ew = full[w0:w0 + n] / full[w0]
        names.append((b, "none"))
        curves.append(ew)
        for r in ls.REGIMES:
            names.append((b, r))
            curves.append(rs.ew_regime(ew, cash, st_w[r], sw))
    sp = gs._setup(cash, wdates)
    EQ = np.column_stack(curves)
    R = EQ[1:] / EQ[:-1] - 1.0
    gm = gs.goal_metrics(R, np.repeat(sp["c"][:, None], R.shape[1], axis=1), sp["years"], sp["wins"])
    rows = []
    for i, (b, r) in enumerate(names):
        lab = f"{ls.BAND_LABELS[b]} 동일비중" + ("" if r == "none" else f" × {r} {rs.REGIMES[r]}") + " + 국채"
        rows.append({"benchmark": f"ew_{b}" + ("" if r == "none" else f"_{r}"), "label": lab, "band": b, "regime": r,
                     **{k: float(v[i]) for k, v in gm.items() if k != "win_cagr"}, **ls.ts_period_stats(EQ[:, i], wdates)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- stage: repro

def build_codes(a) -> None:
    cdir = cost_dir(a)
    if (cdir / "codes.npy").exists():
        return
    cdir.mkdir(parents=True, exist_ok=True)
    panel = crsp.load_panel(Path(a.panel), "raw")
    cols = panel.close.columns
    cap = panel.extra["cap_rank"].reindex(index=panel.close.index, columns=cols).to_numpy(dtype=float)
    raw = panel.extra["raw_close"].reindex(index=panel.close.index, columns=cols).to_numpy(dtype=float)
    spy = int(cols.get_loc(BENCHMARK)) if BENCHMARK in cols else None
    code = fill_codes(cap, raw, spy)
    np.save(cdir / "codes.npy", code)
    np.save(cdir / "years.npy", year_index(panel.close.index))
    info = {"rows": int(code.shape[0]), "cols": int(code.shape[1]), "spy_col": spy,
            "spy_raw_close_days": int(np.isfinite(raw[:, spy]).sum()) if spy is not None else 0,
            "first": str(panel.close.index[0].date()), "last": str(panel.close.index[-1].date())}
    (cdir / "codes.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    log(f"체결 칸 코드: {info}")


def all_jobs(t0: bool) -> list:
    return rb.p_jobs(rb.GENS) + rb.q_jobs(rb.GENS) + rb.o_jobs(rb.GENS) + rb.u_jobs(rb.GENS)


def summarize_model(mdir: Path) -> dict:
    out = {"gP": rs.summarize(rb.read_jobs(mdir / "p" / "jobs")), "gQ": ls.summarize(rb.read_jobs(mdir / "q" / "jobs")),
           "gO": gs.summarize(rb.read_jobs(mdir / "o" / "jobs")), "unl": rb.read_jobs(mdir / "unl" / "jobs")}
    out["gP"].to_csv(mdir / "p" / "goal.csv", index=False)
    out["gQ"].to_csv(mdir / "q" / "goal.csv", index=False)
    out["gO"].to_csv(mdir / "o" / "goal.csv", index=False)
    out["unl"].to_csv(mdir / "unlimited.csv", index=False)
    return out


def run_model(a, model: str, workers: int) -> None:
    S = s_paths(a)
    mdir = model_dir(a, model)
    run_pool(all_jobs(model == "T0"), mdir, a, S["eligible"], S["dirs"], S["cfg"], model, workers)
    if not (mdir / "benchmarks_q.csv").exists():
        bp_, bq_ = benchmarks(a, model)
        bp_.to_csv(mdir / "benchmarks_p.csv", index=False)
        bq_.to_csv(mdir / "benchmarks_q.csv", index=False)
    g = summarize_model(mdir)
    if model == "T0":
        bench, _, _ = rb.bench_from(pd.read_csv(mdir / "benchmarks_p.csv"))
        j40 = rb.p_jobs40(g["gP"], bench, rb.GENS) + rb.q_jobs40(g["gQ"], bench, rb.GENS)
        run_pool(j40, mdir, a, S["eligible"], S["dirs"], S["cfg"], model, workers)
        rs.summarize(rb.read_jobs(mdir / "p" / "jobs", bp=True)).to_csv(mdir / "p" / "goal40.csv", index=False)
        ls.summarize(rb.read_jobs(mdir / "q" / "jobs", bp=True)).to_csv(mdir / "q" / "goal40.csv", index=False)


def stage_repro(a) -> dict:
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    build_codes(a)
    run_model(a, "T0", a.workers)
    S = s_paths(a)
    raw, mdir = S["raw"], model_dir(a, "T0")
    pairs = {"p_goal": ("p/goal.csv", rb.P_KEYS), "q_goal": ("q/goal.csv", rb.Q_KEYS), "p_40bp": ("p/goal40.csv", rb.P_KEYS),
             "q_40bp": ("q/goal40.csv", rb.Q_KEYS), "o_goal": ("o/goal.csv", ["setting", "gen", "exit", "kind", "cost", "mode"]),
             "unlimited": ("unlimited.csv", ["src", "exit"]), "benchmarks_p": ("benchmarks_p.csv", ["benchmark"]),
             "benchmarks_q": ("benchmarks_q.csv", ["benchmark"])}
    res: dict = {}
    for k, (f, keys) in pairs.items():
        c = rb.compare(pd.read_csv(mdir / f), pd.read_csv(raw / f), keys)
        c.to_csv(out / f"repro_{k}.csv.gz", index=False)
        res[k] = rb.summary_of(c)
        log(f"  T0 {k}: {res[k]}")
    # T0's verdicts = Stage S's raw verdicts
    J = rb.judge(load_model(a, "T0"))
    vs = Path(a.stage_s) / "agg" / "verdict.json"
    if vs.exists():
        stored = json.loads(vs.read_text(encoding="utf-8"))
        res["labels_T0"] = rb.k_labels(J["K"])
        res["labels_stage_s"] = stored["labels"]
        res["k_equal"] = {k: _k_core(J["K"][k]) == _k_core(stored["K"]["raw"][k]) for k in J["K"]}
    res["pass"] = bool(all(res[k]["pass"] for k in pairs) and res.get("labels_T0") == res.get("labels_stage_s")
                       and all(res.get("k_equal", {}).values()))
    (out / "repro.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"재현 판정: {'통과' if res['pass'] else '실패 — 뒤 단계는 돌지 않습니다'}")
    return res


def _k_core(k: dict) -> dict:
    """The counts of a K entry (floats rounded to 1e-9) for comparison."""
    out = {}
    for kk, v in k.items():
        if isinstance(v, float):
            out[kk] = round(v, 9)
        elif isinstance(v, dict):
            out[kk] = json.loads(json.dumps(v, default=float))
            out[kk] = {a: (round(b, 9) if isinstance(b, float) else b) for a, b in out[kk].items()}
        else:
            out[kk] = v
    return out


def need_repro(a) -> dict:
    f = Path(a.out) / "repro.json"
    rp = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    if not rp.get("pass"):
        raise SystemExit("재현 점검을 통과하지 않아 이 단계는 돌지 않습니다 (repro.json).")
    return rp


# --------------------------------------------------------------------------- stage: measure

def quarter_edges(panel, spy: str | None) -> pd.DataFrame:
    """EDGE (sign=True) per stock x calendar quarter with >= MIN_DAYS complete days, plus its median cap
    rank and raw close over those days."""
    idx = panel.close.index
    sel = (idx >= pd.Timestamp(EDGE_START)) & (idx <= pd.Timestamp(EDGE_END))
    cols = panel.close.columns
    keep_cols = np.array([c != spy for c in cols])
    q = idx.to_period("Q")
    rows = []
    cap_all = panel.extra["cap_rank"].reindex(index=idx, columns=cols)
    raw_all = panel.extra["raw_close"].reindex(index=idx, columns=cols)
    for per in pd.unique(q[sel]):
        r = np.flatnonzero(sel & (q == per))
        arr = [getattr(panel, f).iloc[r].to_numpy(dtype=float).T[keep_cols] for f in ("open", "high", "low", "close")]
        ok = np.isfinite(arr[0]) & np.isfinite(arr[1]) & np.isfinite(arr[2]) & np.isfinite(arr[3])
        nd = ok.sum(axis=1)
        use = nd >= MIN_DAYS
        if not use.any():
            continue
        o, h, l_, c = (np.where(ok[use], x[use], np.nan) for x in arr)
        s = edge_rows(o, h, l_, c, sign=True)
        rank = cap_all.iloc[r].to_numpy(dtype=float).T[keep_cols][use]
        rank = np.where(ok[use], np.where(np.isfinite(rank), rank, np.inf), np.nan)
        rc = raw_all.iloc[r].to_numpy(dtype=float).T[keep_cols][use]
        rc = np.where(ok[use], rc, np.nan)
        with np.errstate(all="ignore"):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                rmed = np.nanmedian(rank, axis=1)
                pmed = np.nanmedian(rc, axis=1)
        tick = (1 / 8 if per.start_time < TICK_CHANGE else 1 / 16) if per.year <= 2000 else np.nan
        rows.append(pd.DataFrame({"symbol": np.asarray(cols)[keep_cols][use], "year": per.year, "quarter": per.quarter, "days": nd[use],
                                  "s": s, "rank_med": rmed, "price_med": pmed, "tick": tick}))
        log(f"  {per}: 종목 {int(use.sum()):,}")
    out = pd.concat(rows, ignore_index=True)
    out["q"] = np.sign(out["s"]) * out["s"] ** 2
    out["cap_b"] = cap_bucket(out["rank_med"].to_numpy())
    out["price_b"] = price_bucket(out["price_med"].to_numpy())
    return out


def era_of(year) -> np.ndarray:
    y = np.asarray(year)
    return np.select([y <= 2000, y <= 2007], ["1993-2000", "2001-2007"], "2008-2023")


def pooled(sq: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    g = sq[np.isfinite(sq["q"])].groupby(keys)["q"].agg(n="size", mean_q="mean").reset_index()
    g["half_bp"] = np.sqrt(np.maximum(g["mean_q"], 0.0)) / 2 * 1e4
    return g


def cell_table(sq: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """(cells: year x cap x price with fallbacks and the final one-way bp, year x cap pooled, era x cap)."""
    s = sq.assign(era=era_of(sq["year"]))
    cell = pooled(s[s["price_b"] != 255], ["year", "cap_b", "price_b"]).set_index(["year", "cap_b", "price_b"])
    yc = pooled(s, ["year", "cap_b"]).set_index(["year", "cap_b"])
    ec = pooled(s, ["era", "cap_b"]).set_index(["era", "cap_b"])
    rows = []
    for y in YEARS:
        for c in range(N_CAP):
            for p in list(range(N_PRICE)) + [255]:
                k = (y, c, p)
                src, n, half = None, 0, np.nan
                if p != 255 and k in cell.index and cell.loc[k, "n"] >= MIN_CELL:
                    src, n, half = "칸", int(cell.loc[k, "n"]), float(cell.loc[k, "half_bp"])
                elif (y, c) in yc.index and yc.loc[(y, c), "n"] >= MIN_CELL:
                    src, n, half = "연도 × 시총", int(yc.loc[(y, c), "n"]), float(yc.loc[(y, c), "half_bp"])
                else:
                    e = str(era_of([y])[0])
                    if (e, c) in ec.index and ec.loc[(e, c), "n"] >= MIN_CELL:
                        src, n, half = "시대 × 시총", int(ec.loc[(e, c), "n"]), float(ec.loc[(e, c), "half_bp"])
                if src is None:
                    raise SystemExit(f"칸 {k}: 대체 값도 종목 · 분기 {MIN_CELL}개 미만")
                own = int(cell.loc[k, "n"]) if p != 255 and k in cell.index else 0
                rows.append({"year": y, "cap_b": c, "price_b": p, "cap": CAP_LABELS[c], "price": PRICE_LABELS[p] if p != 255 else "주가 없음",
                             "n_own": own, "source": src, "n_used": n, "half_bp_raw": half,
                             "cost_bp": float(np.clip(half, FLOOR_BP, CAP_BP))})
    return pd.DataFrame(rows), yc.reset_index(), ec.reset_index()


def checks(yc: pd.DataFrame) -> dict:
    """(a) and (b) on the year x cap pooled half-spreads (bp, no floor / cap / fallback)."""
    t = yc.pivot(index="year", columns="cap_b", values="half_bp")
    t = t.reindex(index=YEARS)
    a1 = {int(y): bool(t.loc[y, 0] < t.loc[y, 3]) for y in YEARS}
    era_mean = {e: {c: float(t.loc[a:b, c].mean()) for c in range(4)} for e, (a, b) in ERAS.items()}
    a2 = {e: bool(all(m[c] < m[c + 1] for c in range(3))) for e, m in era_mean.items()}
    b_ = {c: bool(era_mean["1993-2000"][c] > era_mean["2008-2023"][c]) for c in range(4)}
    res = {"a_years_1_500_below_2001_3000": a1, "a_years_fail": [y for y, v in a1.items() if not v], "a_era_order": a2,
           "era_mean_bp": {e: {CAP_LABELS[c]: v for c, v in m.items()} for e, m in era_mean.items()},
           "b_1993_2000_above_2008_2023": {CAP_LABELS[c]: v for c, v in b_.items()}}
    res["a_pass"] = bool(all(a1.values()) and all(a2.values()))
    res["b_pass"] = bool(all(b_.values()))
    res["pass"] = res["a_pass"] and res["b_pass"]
    return res


def stage_measure(a) -> dict:
    need_repro(a)
    cdir = cost_dir(a)
    f_sq = cdir / "stock_quarters.parquet"
    if not f_sq.exists():
        panel = crsp.load_panel(Path(a.panel), "raw")
        sq = quarter_edges(panel, BENCHMARK if BENCHMARK in panel.close.columns else None)
        del panel
        sq.to_parquet(f_sq)
    sq = pd.read_parquet(f_sq)
    cells, yc, ec = cell_table(sq)
    cells.to_csv(cdir / "cells.csv", index=False)
    yc.to_csv(cdir / "year_cap.csv", index=False)
    ec.to_csv(cdir / "era_cap.csv", index=False)
    tab = np.zeros((len(YEARS), N_CODES))
    for _, r in cells.iterrows():
        code = CODE_POOLED + r["cap_b"] if r["price_b"] == 255 else r["cap_b"] * N_PRICE + r["price_b"]
        tab[int(r["year"]) - YEARS[0], int(code)] = r["cost_bp"] / 1e4
    np.save(cdir / "tab_T1.npy", tab)
    ck = checks(yc)
    # cross-check 2: half tick / price before 2001
    pre = sq[(sq["year"] <= 2000) & np.isfinite(sq["price_med"]) & (sq["price_med"] > 0)]
    tick = pre.assign(tick_bp=pre["tick"] / 2 / pre["price_med"] * 1e4).groupby(["year", "cap_b", "price_b"])["tick_bp"].agg(["size", "mean"])
    tick = tick.reset_index().rename(columns={"size": "n", "mean": "half_tick_bp"})
    tick = tick.merge(cells[["year", "cap_b", "price_b", "half_bp_raw", "cost_bp", "source"]], on=["year", "cap_b", "price_b"], how="left")
    tick.to_csv(cdir / "check_tick.csv", index=False)
    schema = raw_schema(Path(a.crsp))
    res = {"stock_quarters": int(len(sq)), "finite": int(np.isfinite(sq["q"]).sum()), "negative_s": int((sq["s"] < 0).sum()),
           "checks": ck, "raw_columns": schema, "bid_ask_columns": [c for c in schema if c.upper() in ("BID", "ASK")],
           "cross_check_1": "하지 않음 — 원자료에 종가 매수 · 매도 호가(BID · ASK) 열이 없음" if not any(c.upper() in ("BID", "ASK") for c in schema) else "있음",
           "cross_check_3": "하지 않음 (자료 내려받기 승인 대기)",
           "cells_by_source": cells.groupby("source").size().to_dict()}
    (cdir / "measure.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    log(f"측정 점검 (가) {'통과' if ck['a_pass'] else '실패'} · (나) {'통과' if ck['b_pass'] else '실패'}")
    log(json.dumps(ck["era_mean_bp"], ensure_ascii=False))
    return res


def raw_schema(src: Path) -> list[str]:
    """Column names of the first CRSP source file (schema only, no rows read)."""
    import pyarrow.parquet as pq
    files = sorted(src.glob("CRSP_*.parquet"))
    return [f.name for f in pq.ParquetFile(files[0]).schema_arrow] if files else []


# --------------------------------------------------------------------------- stage: accounts

def stage_accounts(a) -> None:
    need_repro(a)
    m = json.loads((cost_dir(a) / "measure.json").read_text(encoding="utf-8"))
    models = list(T_MODELS) if m["checks"]["pass"] else ["TZ"]
    if not m["checks"]["pass"]:
        log("측정 점검 실패: T1 · T2 계좌는 돌리지 않고 TZ만 돌립니다.")
    for model in models:
        run_model(a, model, a.workers)


# --------------------------------------------------------------------------- verdicts

def load_model(a, model: str) -> dict:
    r = model_dir(a, model)
    d = {"gP": pd.read_csv(r / "p" / "goal.csv"), "accP": rb.read_jobs(r / "p" / "jobs", prefix="E0__"), "btabP": pd.read_csv(r / "benchmarks_p.csv"),
         "gQ": pd.read_csv(r / "q" / "goal.csv"), "accQ": rb.read_jobs(r / "q" / "jobs", prefix="E0__"), "btabQ": pd.read_csv(r / "benchmarks_q.csv"),
         "gO": pd.read_csv(r / "o" / "goal.csv"), "unl": pd.read_csv(r / "unlimited.csv")}
    empty = pd.DataFrame(columns=["setting", "gen", "exit", "filter", "band"])
    if model == "T0":
        d["gP40"], d["gQ40"] = pd.read_csv(r / "p" / "goal40.csv"), pd.read_csv(r / "q" / "goal40.csv")
    elif model == "T1" and (model_dir(a, "T2") / "q" / "goal.csv").exists():  # T2 in the 40 bp place
        t2 = model_dir(a, "T2")
        g2p, g2q = pd.read_csv(t2 / "p" / "goal.csv"), pd.read_csv(t2 / "q" / "goal.csv")
        d["gP40"], d["gQ40"] = g2p[g2p["mode"] == "grid"], g2q[g2q["mode"] == "grid"]
    else:
        d["gP40"], d["gQ40"] = empty, empty
    return d


def combine(labels: dict) -> dict:
    out = {}
    for k in labels["T1"]:
        if labels["T1"][k] == "바뀜":
            out[k] = "바뀜"
        elif all(labels[m][k] == "유지" for m in T_MODELS if m in labels):
            out[k] = "비용에 견고"
        else:
            out[k] = "유지, 비용 수준에 민감"
    return out


def error_size(J: dict) -> tuple[pd.DataFrame, dict]:
    """Verdict 3: T1 - T0 system values, medians over the 8 no-gate and the 96 A1 systems."""
    rows = []
    for grp, key, sel, keys in (("국면 없음 8개", "sysP", lambda s: s["filter"] == "none", ["setting", "exit", "filter"]),
                                ("A1 96개", "sysQ", lambda s: s["filter"].isin(rb.A1), ["setting", "exit", "filter", "band"])):
        a0 = J["T0"][key][sel(J["T0"][key])].set_index(keys)
        a1 = J["T1"][key][sel(J["T1"][key])].set_index(keys)
        idx = a0.index.intersection(a1.index)
        for p in ERR_PERIODS:
            d = a1.loc[idx, f"dd_cagr_{p}"] - a0.loc[idx, f"dd_cagr_{p}"]
            rows.append({"group": grp, "period": p, "systems": len(idx), "median_pp": float(d.median()) * 100,
                         "min_pp": float(d.min()) * 100, "max_pp": float(d.max()) * 100})
    t = pd.DataFrame(rows)
    big = bool((t["median_pp"].abs() >= LARGE_PP * 100).any())
    return t, {"label": "큼" if big else "작음", "max_abs_median_pp": float(t["median_pp"].abs().max())}


def above_ma200(J: dict) -> pd.DataFrame:
    rows = []
    for m in J:
        s = J[m]["sysQ"]
        a1 = s[s["filter"].isin(rb.A1)]
        b = J[m]["bench"]["ma200"]
        rows.append({"model": m, "systems": len(a1), **{f"above_{p}": int((a1[f"dd_cagr_{p}"] > b[f"dd_cagr_{p}"]).sum()) for p in ERR_PERIODS},
                     **{f"ma200_{p}": float(b[f"dd_cagr_{p}"]) for p in ERR_PERIODS},
                     **{f"median_{p}": float(a1[f"dd_cagr_{p}"].median()) for p in ERR_PERIODS},
                     "best_2008-2023": float(a1["dd_cagr_2008-2023"].max())})
    return pd.DataFrame(rows)


def paid_table(a) -> pd.DataFrame:
    rows = []
    for m in ("T1", "T2"):
        for sub in ("q", "unl"):
            d = model_dir(a, m) / sub / "jobs"
            files = sorted(d.glob("*_paid.csv")) if d.exists() else []
            for f in files:
                rows.append(pd.read_csv(f).assign(model=m))
    if not rows:
        return pd.DataFrame()
    p = pd.concat(rows, ignore_index=True)
    out = []
    for (m, st, b), g in p.groupby(["model", "setting", "band"]):
        for e, (y0, y1) in PAID_ERAS.items():
            h = g[(g["year"] >= y0) & (g["year"] <= y1)]
            out.append({"model": m, "setting": st, "band": b, "era": e, "fills": int(h["fills"].sum()),
                        "paid_bp": float(h["cost_value"].sum() / h["value"].sum() * 1e4) if h["value"].sum() > 0 else np.nan})
    return pd.DataFrame(out)


def stage_report(a) -> dict:
    rp = need_repro(a)
    out = Path(a.out)
    agg = out / "agg"
    (agg / "figures").mkdir(parents=True, exist_ok=True)
    meas = json.loads((cost_dir(a) / "measure.json").read_text(encoding="utf-8"))
    models = [m for m in MODELS if (model_dir(a, m) / "unlimited.csv").exists()]
    J = {m: rb.judge(load_model(a, m)) for m in models}
    labels = {m: rb.k_labels(J[m]["K"]) for m in models}
    verdict = {"repro": rp, "measure": meas, "models": models, "K": {m: J[m]["K"] for m in models}, "labels": labels,
               "trials": {"cumulative": TRIALS, "new": 0}}
    if "T1" in J:
        verdict["combined"] = combine(labels)
        et, ev = error_size(J)
        et.to_csv(agg / "error_size.csv", index=False)
        verdict["error_size"] = ev
    else:
        why = "측정 점검 (가) · (나)를 통과하지 못해 T1 · T2 계좌를 돌리지 않음 (설계 판정 2)"
        verdict["error_size"] = {"label": "판정하지 않음", "why": why}
        verdict["combined"] = {k: "판정하지 않음 (T1 없음)" for k in labels.get("T0", {})}
    am = above_ma200(J)
    am.to_csv(agg / "above_ma200.csv", index=False)
    verdict["above_ma200"] = am.to_dict("records")
    if "TZ" in J:
        tz = am.set_index("model").loc["TZ", "above_2008-2023"]
        verdict["tz_2008_gap"] = {"above": int(tz), "depends_on_cost": bool(tz >= TZ_GAP_COUNT)}
    verdict["k5"] = {m: J[m]["K"]["K5"]["median_d2008"] for m in models}
    pt = paid_table(a)
    if len(pt):
        pt.to_csv(agg / "paid.csv", index=False)
    # candidates (T1 with T2 in the 40 bp place)
    if "T1" in J:
        for scope in ("P", "Q"):
            for kind, frame in J["T1"]["candidates"][scope].items():
                if len(frame):
                    frame.to_csv(agg / f"cand_{kind}_{scope.lower()}_T1.csv", index=False)
        verdict["candidates_T1"] = {"p": J["T1"]["reported"]["p_candidates"], "q": J["T1"]["reported"]["q_candidates"]}
    for name, f in (("sysP", "systems_p"), ("sysQ", "systems_q"), ("cntP", "helps_p"), ("cntQ", "helps_q"), ("k1", "k1"), ("k5_rows", "k5_rows")):
        pd.concat([J[m][name].assign(model=m) for m in models], ignore_index=True).to_csv(agg / f"{f}.csv", index=False)
    pd.concat([pd.read_csv(model_dir(a, m) / "benchmarks_q.csv").assign(model=m) for m in models] +
              [pd.read_csv(model_dir(a, m) / "benchmarks_p.csv").assign(model=m) for m in models], ignore_index=True).drop_duplicates(
        ["benchmark", "model"]).to_csv(agg / "benchmarks.csv", index=False)
    cells = []
    for m in models:
        for scope, key in (("P", "cellsP"), ("Q", "cellsQ")):
            c = J[m][key]
            cells.append(c[c["cost"] == pj.COST_BPS].assign(model=m, scope=scope))
    pd.concat(cells, ignore_index=True).to_csv(agg / "cells.csv.gz", index=False)
    for f in ("cells.csv", "year_cap.csv", "era_cap.csv", "check_tick.csv", "measure.json", "codes.json"):
        if (cost_dir(a) / f).exists():
            (agg / f"cost_{f}").write_bytes((cost_dir(a) / f).read_bytes())
    for f in ("repro.json",):
        (agg / f).write_bytes((out / f).read_bytes())
    for f in out.glob("repro_*.csv.gz"):
        (agg / f.name).write_bytes(f.read_bytes())
    verdict["predictions"] = predictions(a, J, pt)
    tk = tick_summary(a)
    tk.to_csv(agg / "check_tick_summary.csv", index=False)
    verdict["tick_check"] = {"cells": int(len(tk)), "edge_below_half_tick": int((tk["edge_half_bp"] < tk["half_tick_bp"]).sum())}
    (agg / "verdict.json").write_text(json.dumps(verdict, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    if figures.available():
        write_figures(a, agg, J, pt)
    log("판정: " + json.dumps({"labels": labels, "combined": verdict.get("combined"), "error": verdict.get("error_size")}, ensure_ascii=False))
    return verdict


def tick_summary(a) -> pd.DataFrame:
    """Cross-check 2 (reported): per tick era (1/8: 1993-01 - 1997-06 by quarter start, 1/16: to 2000-12) x
    cap x price bucket, the stock-quarter mean of half tick / price against the cell's EDGE half-spread
    (cells with their own >= 100 stock-quarters, quarter-count weighted over years)."""
    t = pd.read_csv(cost_dir(a) / "check_tick.csv")
    t = t[(t["source"] == "칸") & (t["price_b"] != 255)]
    t = t.assign(tick=np.where(t["year"] <= 1996, "1/8 (1993–1996)", np.where(t["year"] >= 1998, "1/16 (1998–2000)", "1997 (둘 섞임)")))
    rows = []
    for (tk, c, pb), g in t.groupby(["tick", "cap_b", "price_b"]):
        w = g["n"].to_numpy(float)
        rows.append({"tick": tk, "cap": CAP_LABELS[int(c)], "price": PRICE_LABELS[int(pb)], "stock_quarters": int(w.sum()),
                     "half_tick_bp": float(np.average(g["half_tick_bp"], weights=w)), "edge_half_bp": float(np.average(g["half_bp_raw"], weights=w)),
                     "years_edge_below_tick": int((g["half_bp_raw"] < g["half_tick_bp"]).sum()), "years": len(g)})
    return pd.DataFrame(rows)


def predictions(a, J: dict, pt: pd.DataFrame) -> list[dict]:
    """The design's 'predictions written down first' against the results."""
    cells = pd.read_csv(cost_dir(a) / "cells.csv")
    yc = pd.read_csv(cost_dir(a) / "year_cap.csv")
    t = yc.pivot(index="year", columns="cap_b", values="half_bp")
    out = []

    def add(what, expect, lo, hi, val):
        out.append({"예상": what, "범위": expect, "결과": val, "맞음": bool(lo <= val <= hi) if val == val else None})
    add("편도 1993–2000, 1–500 (bp)", "10–30", 10, 30, float(t.loc[1993:2000, 0].mean()))
    add("편도 1993–2000, 2001–3000 (bp)", "60–150", 60, 150, float(t.loc[1993:2000, 3].mean()))
    add("편도 2010–2023, 1–500 (bp)", "≤5", 0, 5, float(t.loc[2010:2023, 0].mean()))
    add("편도 2010–2023, 2001–3000 (bp)", "10–30", 10, 30, float(t.loc[2010:2023, 3].mean()))
    if len(pt):
        q = pt[(pt["model"] == "T1") & (pt["setting"] != "제한 없음")]
        for e, rng_, lo, hi in (("1993-2000", "30–70", 30, 70), ("2010-2023", "5–15", 5, 15)):
            v = q[q["era"] == e]["paid_bp"]
            add(f"계좌가 낸 비용 {e} (bp, 설정 × 범위 중앙값)", rng_, lo, hi, float(v.median()) if len(v) else np.nan)
    if "T1" in J:
        et, _ = error_size(J)
        for p, rng_, lo, hi in (("1993-2007", "−0.2 – −0.8", -0.8, -0.2), ("2008-2023", "0 – +0.3", 0.0, 0.3)):
            for grp in ("국면 없음 8개", "A1 96개"):
                v = float(et[(et["group"] == grp) & (et["period"] == p)]["median_pp"].iloc[0])
                add(f"T1 − T0 {p}, {grp} (%p)", rng_, lo, hi, v)
        lab = combine({m: rb.k_labels(J[m]["K"]) for m in J})
        out.append({"예상": "K1–K5 모두 유지 (T1)", "범위": "유지", "결과": json.dumps(lab, ensure_ascii=False),
                    "맞음": bool(all(v != "바뀜" for v in lab.values()))})
    if "TZ" in J:
        am = above_ma200(J).set_index("model")
        add("TZ에서 2008–2023 SPY200 + 국채를 넘는 A1 시스템 (96개 중)", "≤3", 0, 3, float(am.loc["TZ", "above_2008-2023"]))
    return out


# --------------------------------------------------------------------------- figures

MODEL_COLOR = {"T0": "#888888", "TZ": "#2f855a", "T1": "#2b6cb0", "T2": "#9b2c2c"}
MODEL_LABEL = {"T0": "T0 편도 20bp", "TZ": "TZ 비용 0", "T1": "T1 측정 비용", "T2": "T2 측정 × 2"}


def fig_years(agg: Path, a) -> Path:
    """(1) Yearly one-way cost per cap bucket (price pooled), with the 20 bp line."""
    plt = figures._plt()
    yc = pd.read_csv(cost_dir(a) / "year_cap.csv")
    fig, ax = plt.subplots(figsize=(11, 5.2))
    for c in range(N_CAP):
        s = yc[yc["cap_b"] == c].sort_values("year")
        col = (figures.COLORS + ["#555555"])[c]
        v = np.maximum(s["half_bp"].to_numpy(), FLOOR_BP)  # q mean <= 0 -> drawn at the 1 bp floor (hollow)
        ax.plot(s["year"], v, marker="o", ms=3, lw=1.5, color=col, label=f"시총 {CAP_LABELS[c]}")
        z = s["half_bp"].to_numpy() < FLOOR_BP
        ax.scatter(s["year"].to_numpy()[z], v[z], s=40, facecolors="white", edgecolors=col, zorder=3)
    ax.axhline(20, color="#222", ls="--", lw=1, label="지금까지의 가정 (편도 20bp)")
    ax.axvline(2001, color="#bbb", lw=0.8)
    ax.set_yscale("log")
    ax.set_ylabel("편도 비용 = EDGE 스프레드 / 2 (bp, 로그)")
    ax.set_title("연도별 편도 비용 추정 (종목 · 분기 EDGE, 주가 구간 합침; 2001 = 소수점 호가 전환; 속 빈 점 = q 평균 ≤ 0이라 하한 1bp)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8, frameon=False)
    return figures._save(fig, agg / "figures" / "cost_years.png")


def fig_cells(agg: Path, a) -> Path:
    """(2) Cap x price cell costs, three eras (mean over the era's years of the final cell cost)."""
    plt = figures._plt()
    cells = pd.read_csv(cost_dir(a) / "cells.csv")
    cells = cells[cells["price_b"] != 255].assign(era=lambda d: era_of(d["year"]))
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    for ax, e in zip(axes, ERAS):
        t = cells[cells["era"] == e].pivot_table(index="cap_b", columns="price_b", values="cost_bp", aggfunc="mean")
        ax.imshow(np.log10(t.to_numpy()), cmap="viridis_r", vmin=0, vmax=np.log10(CAP_BP), aspect="auto")
        for i in range(t.shape[0]):
            for j in range(t.shape[1]):
                ax.text(j, i, f"{t.iloc[i, j]:.0f}", ha="center", va="center", fontsize=8, color="white" if t.iloc[i, j] > 40 else "black")
        ax.set_xticks(range(N_PRICE), PRICE_LABELS, fontsize=8)
        ax.set_yticks(range(N_CAP), CAP_LABELS, fontsize=8)
        ax.set_title(f"{e} (편도 bp, 연도 평균)", fontsize=10)
        ax.set_xlabel("실제 주가 (달러)")
    axes[0].set_ylabel("시총 순위")
    fig.suptitle("시총 × 주가 칸의 측정 편도 비용 (bp, 색은 로그; 하한 1bp · 상한 500bp, 100개 미만 칸은 대체 값) — 점검 (가) 실패 시 계좌에는 쓰지 않음", fontsize=10.5)
    return figures._save(fig, agg / "figures" / "cost_cells.png")


def fig_scatter(agg: Path, J: dict, other: str = "T1") -> Path:
    """(3) Every cell's T0 vs T1 (TZ when T1 was not run) drawdown-matched return, three periods."""
    plt = figures._plt()
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.4))
    kp = ["setting", "gen", "exit", "filter"]
    for ax, p in zip(axes, ERR_PERIODS):
        allx, ally = [], []
        for scope, key, keys, col in (("P", "gP", kp, figures.COLORS[0]), ("Q", "gQ", kp + ["band"], figures.COLORS[2])):
            def sel(g):
                g = g[(g["gen"] != "E0") & (g["cost"] == pj.COST_BPS) & (g["mode"] == "grid")]
                return (g[g["band"].isin(ls.NARROW)] if scope == "Q" else g).set_index(keys)
            x0, x1 = sel(J["T0"][key]), sel(J[other][key])
            idx = x0.index.intersection(x1.index)
            ax.scatter(x0.loc[idx, f"dd_cagr_{p}"] * 100, x1.loc[idx, f"dd_cagr_{p}"] * 100, s=5, alpha=0.45, color=col,
                       label=("U3000 · 국면 11가지" if scope == "P" else "U1000 · U500 · 국면 5가지") + f" ({len(idx)}칸)")
            allx.append(x0.loc[idx, f"dd_cagr_{p}"].to_numpy())
            ally.append(x1.loc[idx, f"dd_cagr_{p}"].to_numpy())
        lo, hi = min(ax.get_xlim()[0], ax.get_ylim()[0]), max(ax.get_xlim()[1], ax.get_ylim()[1])
        ax.plot([lo, hi], [lo, hi], color="#222", lw=0.8)
        ax.axhline(J[other]["bench"]["ma200"][f"dd_cagr_{p}"] * 100, color="#8e44ad", ls="--", lw=1, label=f"SPY 200일선 + 국채 ({other})")
        d = np.median(np.concatenate(ally) - np.concatenate(allx)) * 100
        ax.set_title(f"{p} · 차이 중앙값 {d:+.2f}%p", fontsize=10)
        ax.set_xlabel("T0 편도 20bp (%)")
        ax.set_ylabel(f"{MODEL_LABEL[other]} (%)")
        ax.grid(alpha=0.3)
    axes[0].legend(fontsize=7.5, frameon=False, loc="upper left")
    fig.suptitle(f"칸마다 T0 vs {other} 낙폭 15% 맞춤 수익 (매수 15개 × 설정 4 × 매도 2)", fontsize=11)
    return figures._save(fig, agg / "figures" / f"scatter_t0_{other.lower()}.png")


def fig_gates(agg: Path, J: dict) -> Path:
    """(4) Per gate the median over the 8 systems under the four models."""
    plt = figures._plt()
    fs = list(rs.FILTERS)
    ms_ = list(J)
    fig, axes = plt.subplots(1, 3, figsize=(19, 5.3))
    xx = np.arange(len(fs))
    wdt = 0.8 / len(ms_)
    for ax, p in zip(axes, ERR_PERIODS):
        for k, m in enumerate(ms_):
            s = J[m]["sysP"]
            v = [float(s.loc[s["filter"] == f, f"dd_cagr_{p}"].median()) * 100 for f in fs]
            ax.bar(xx + (k - (len(ms_) - 1) / 2) * wdt, v, width=wdt, color=MODEL_COLOR[m], label=MODEL_LABEL[m])
            ax.axhline(J[m]["bench"]["ma200"][f"dd_cagr_{p}"] * 100, color=MODEL_COLOR[m], ls="--", lw=1)
        ax.set_xticks(xx, ["없음" if f == "none" else f for f in fs], rotation=60, fontsize=8)
        ax.set_title(f"{p} 낙폭 15% 맞춤 수익 (%) · 점선 = 같은 모형의 SPY 200일선 + 국채", fontsize=9.5)
        ax.grid(alpha=0.3, axis="y")
    axes[0].legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle("국면별 시스템 중앙값 (8개 시스템, U3000) — 비용 모형별", fontsize=11)
    return figures._save(fig, agg / "figures" / "gates_models.png")


def fig_paid(agg: Path, pt: pd.DataFrame) -> Path | None:
    """(5) Cost the accounts paid (T1): setting x band x era."""
    if not len(pt):
        return None
    plt = figures._plt()
    q = pt[pt["model"] == "T1"]
    eras = list(PAID_ERAS)
    groups = [(s, b) for s in [ns.setting_name(x) for x in rs.SETTINGS] for b in ls.BANDS] + [("제한 없음", "unlimited")]
    fig, ax = plt.subplots(figsize=(15, 5.4))
    xx = np.arange(len(groups))
    wdt = 0.8 / len(eras)
    for k, e in enumerate(eras):
        v = []
        for s, b in groups:
            r = q[(q["setting"] == s) & (q["band"] == b) & (q["era"] == e)]
            v.append(float(r["paid_bp"].iloc[0]) if len(r) else np.nan)
        ax.bar(xx + (k - 1.5) * wdt, v, width=wdt, color=(figures.COLORS + ["#555555"])[k], label=e)
    ax.axhline(20, color="#222", ls="--", lw=1, label="편도 20bp")
    ax.set_xticks(xx, [f"{s}\n{b}" for s, b in groups], fontsize=7.5)
    ax.set_ylabel("체결 금액 가중 편도 비용 (bp)")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("계좌가 실제로 낸 비용 (T1; 국면 없음 + A1 넷, 매수 15개 + E0, 매도 2 합산)", fontsize=10.5)
    return figures._save(fig, agg / "figures" / "paid.png")


def fig_k(agg: Path, J: dict) -> Path:
    """(6) K1-K5 values under the models that were run (each panel with its 'changed' line)."""
    plt = figures._plt()
    ms_ = list(J)
    k = len(ms_)
    wdt = 0.8 / k
    off = [(i - (k - 1) / 2) * wdt for i in range(k)]
    fig, axes = plt.subplots(1, 5, figsize=(23, 5), gridspec_kw={"width_ratios": [1, 1.1, 2.2, 1.1, 1]})
    # K1
    ax = axes[0]
    for i, m in enumerate(ms_):
        K1 = J[m]["K"]["K1"]
        v = [K1["above_random"], K1["a_pass"], K1["b_pass"], K1["seeds_above_random"]]
        ax.bar(np.arange(4) + off[i], v, width=wdt, color=MODEL_COLOR[m], label=MODEL_LABEL[m])
        for x, y in zip(np.arange(4) + off[i], v):
            ax.text(x, y + 0.1, str(y), ha="center", fontsize=8)
    ax.axhline(rb.K1_CHANGED, color="#c0392b", ls="--", lw=1)
    ax.set_xticks(range(4), ["나음\n(바뀜 ≥ 2)", "(가) 통과", "(나) 통과", "E0 씨앗\n나음"], fontsize=8)
    ax.set_ylim(0, 15)
    ax.set_title("K1 매수 15개 vs 무작위 매수", fontsize=9.5)
    # K2 / K4: best system vs SPY200 + T-bill per period
    for ax, kk, key, sel, t in ((axes[1], "K2", "sysP", lambda s: s["filter"] == "none", "K2 국면 없음 8개"),
                                (axes[3], "K4", "sysQ", lambda s: s["filter"].isin(ls.FILTERS), "K4 국면 · 범위 120개")):
        for i, m in enumerate(ms_):
            s = J[m][key][sel(J[m][key])]
            v = [float(s[f"dd_cagr_{p}"].max()) * 100 for p in ERR_PERIODS]
            ax.bar(np.arange(3) + off[i], v, width=wdt, color=MODEL_COLOR[m], label=MODEL_LABEL[m])
            ax.scatter(np.arange(3) + off[i], [J[m]["bench"]["ma200"][f"dd_cagr_{p}"] * 100 for p in ERR_PERIODS], marker="_", s=150, color="#8e44ad")
        ax.set_xticks(range(3), ERR_PERIODS, fontsize=8)
        ax.set_title(f"{t}: 기간별 최고 시스템 (%)\n보라 막대 = 같은 모형의 SPY 200일선 + 국채; 목표 통과 "
                     + " · ".join(f"{m} {J[m]['K'][kk]['passing']}" for m in ms_), fontsize=9)
        ax.grid(alpha=0.3, axis="y")
    # K3: helps for every gate
    ax = axes[2]
    fs = [f for f in rs.FILTERS if f != "none"]
    for i, m in enumerate(ms_):
        c = J[m]["K"]["K3"]["counts"]
        ax.bar(np.arange(len(fs)) + off[i], [c[f]["helps"] for f in fs], width=wdt, color=MODEL_COLOR[m], label=MODEL_LABEL[m])
        ax.scatter(np.arange(len(fs)) + off[i], [c[f]["q95"] for f in fs], marker="_", s=90, color="#222")
    ax.set_xticks(range(len(fs)), fs, fontsize=8)
    ax.set_ylim(0, 8)
    ax.set_title("K3 국면 필터의 '도움' 수 (8개 시스템 중; 검은 막대 = 우연 95% 분위)\nA1(R1 · R3 · R4 · R5)은 분위 위, A2는 분위 이하여야 '유지'", fontsize=9)
    ax.legend(fontsize=8, frameon=False, loc="upper right")
    ax.grid(alpha=0.3, axis="y")
    # K5
    ax = axes[4]
    for i, m in enumerate(ms_):
        v = [J[m]["K"]["K5"]["median_d2008"][b] * 100 for b in ls.NARROW]
        ax.bar(np.arange(2) + off[i], v, width=wdt, color=MODEL_COLOR[m], label=MODEL_LABEL[m])
    ax.axhline(rb.K5_PP * 100, color="#c0392b", ls="--", lw=1)
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(range(2), [f"{b} − U3000" for b in ls.NARROW], fontsize=8)
    ax.set_ylim(-0.2, 0.6)
    ax.set_title("K5 2008–2023 변화 중앙값 (%p, 40개 시스템)\n붉은 점선 = 바뀜 경계 +0.5%p", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    lab = {m: rb.k_labels(J[m]["K"]) for m in ms_}
    fig.suptitle("큰 결론 다섯 가지의 값 — " + " / ".join(f"{MODEL_LABEL[m]}: " + " ".join(f"{kk} {v}" for kk, v in lab[m].items()) for m in ms_), fontsize=10.5)
    return figures._save(fig, agg / "figures" / "k_models.png")


def write_figures(a, agg: Path, J: dict, pt: pd.DataFrame) -> list:
    paths = [fig_years(agg, a), fig_cells(agg, a)]
    if "T1" in J or "TZ" in J:
        paths.append(fig_scatter(agg, J, "T1" if "T1" in J else "TZ"))
    paths += [fig_gates(agg, J), fig_paid(agg, pt), fig_k(agg, J)]
    return [p for p in paths if p is not None]


# --------------------------------------------------------------------------- main

def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, choices=["repro", "measure", "accounts", "report"])
    ap.add_argument("--panel", required=True)
    ap.add_argument("--stage-s", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--crsp", default=r"H:\내 드라이브\CRSP", help="CRSP source folder (measure: only its parquet schema is read)")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--feature-workers", type=int, default=6)
    a = ap.parse_args(argv)
    {"repro": stage_repro, "measure": stage_measure, "accounts": stage_accounts, "report": stage_report}[a.stage](a)


if __name__ == "__main__":
    main(sys.argv[1:])
