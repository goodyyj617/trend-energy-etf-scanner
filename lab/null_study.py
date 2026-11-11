"""Stage U: the chance rate of the candidate procedure (DECISIONS 101).

    python -m lab.null_study --stage repro|tables|accounts|report --panel DIR --stage-s DIR --tmp DIR --out DIR
        [--seeds 1001-2000] [--workers N] [--table-workers N] [--feature-workers N] [--docs DIR --docs-figures DIR]

Decisions 95 / 96 call a buy rule x regime gate (x band) a candidate when, in the low-volatility-first
accounts (P1, 10 / 20 / 50 slots), both exits pass the per-cell test at 20 and 40 bp and a neighbouring
setting (and the other narrowed band) does too. On the raw universe two cells passed (E7 x R3 A1 x N20
x U1000 / U500), found by scanning 15 entries. This runner measures how often the same procedure makes
a candidate when the entry carries no information: 1,000 random-entry rules (blocks.g_random, 1% of
stock-days, seeds 1001-2000, one seed = one rule) replace the entry, everything else is Stage S's.

Inputs (--stage-s = a copy of Drive lab_crsp/stage_s_raw): raw eligible mask, regime states, raw base
table (for the 15 real entries), raw signal tables (15 entries and E0 seeds 1-5), benchmark tables,
goal tables (E0 rows for the entry-specific label, the real entries for repro), the stored candidate
and chance tables. Per seed the X1 / H1 signal tables are built in --tmp (never kept), the base rows
(M1 / P1 percentiles, cap rank, dollar volume) are attached, and the 120 cells the candidate conditions
read are replayed with regime_study.run_job / large_study.run_job unchanged; per-seed account rows go to
--out/units/U<seed>.parquet (checkpoint: a finished seed is skipped).

Stages (each skips finished work; tables / accounts / report refuse to run unless repro.json passed):
  repro     feature matrices; (a) seeds 1-5 rebuilt through the new table path = the stored raw E0
            tables row for row, their base rows = Stage S's, their 120 cells (20 bp) = Stage S's E0 rows
            per seed to 1e-9; (b) the 15 real entries through this runner's unit path = Stage S's goal
            rows (20 and 40 bp, 1e-9) and, through the candidate path, Stage S's cell flags and
            candidate table (the 2 candidate cells).
  tables    X1 / H1 signal tables per seed (--seeds), in --tmp.
  accounts  per seed: base rows, 120 cells at 20 bp, the 40 bp cells the registered rule asks for.
  report    the verdict, the reported tables, the expectations, five figures, aggregates.
Only dates 1993-02-01 - 2023-12-31 enter a metric (the job functions cut there); nothing later is
computed into a metric or printed. No DSR / PSR. Trials +0 (random entries are a null; 74,174).

실행 전에 정한 세부 (설계가 비워 둔 것, 계좌 단계 전에 커밋):
  - 씨앗 하나 = 매수 규칙 하나: g_random(값 = 씨앗)을 basis_study.table_menu('R0__X1' · 'R0__H1')로 Stage R의
    E0 표와 같은 매도(X1 25% 트레일링, H1 ATR×3 손절 + 25% 트레일링)에 돌립니다. 씨앗마다 표 파일을 따로
    만들고 그 안의 칸 번호는 0입니다(읽을 때 E0_SEEDS = [씨앗]). 소스 이름 'R0s<씨앗>'(E0와 같은 이름 규칙이라
    P1의 동점 처리 난수식이 같음), 매수 이름(gen) 'U<씨앗>'.
  - 기준표 열: account_study.build_base의 열 그대로(K2 · K8 · K13 · K11 백분위, cap_rank, dv20). 같은 함수
    (account_study.point_values → select_study.feature_values)를 패널의 모든 날 × 종목 칸에 한 번 불러 행렬로
    저장하고, 씨앗마다 그 신호의 (si, sj)에서 읽습니다. 행과 순서는 build_base(매도 X1 · H1, rebase_study의
    감싸기)와 같습니다. 재현 (가)에서 씨앗 1–5의 기준표가 Stage S 기준표와 같아야 합니다.
  - 칸: P 경로 = regime_study.run_job(gen, 매도, P1 설정, 국면 10개(R1–R5 × A1 · A2), 20bp, [소스], 'grid'),
    Q 경로 = large_study.run_job(gen, 매도, P1 설정, 필터 5개(없음 + R1 · R3 · R4 · R5 A1), [U1000, U500], 20bp,
    [소스], 'grid'). 씨앗마다 12개 작업, 120칸. P0 설정 · 가짜 국면 · 가짜 범위 · U3000의 '없음'은 돌리지 않습니다.
  - 40bp: 그 씨앗의 20bp 칸에 rebase_study.p_jobs40 · q_jobs40(= 단계들의 규칙)을 그대로 적용합니다.
  - 후보: regime_study.candidates(씨앗들의 20bp 행 + Stage S 실제 가격 E0 행, P 기준선, 40bp 통과)와
    large_study.candidates(같은 방식, Q 기준선). 기준선은 Stage S raw의 benchmarks_p / benchmarks_q. 두 경로
    중 하나에서라도 candidate가 참인 줄이 있으면 '후보 씨앗'. '매수 규칙 고유'는 entry_specific.
  - 주 판정: k = 후보 씨앗 수, n = 1,000. p̂ = k / n, P̂ = 1 − (1 − p̂)^15. 한쪽 95% Clopper–Pearson 한계는 이항
    누적확률을 이분법으로 풀어 구하고(하한: P(X ≥ k) = 5%, 상한: P(X ≤ k) = 5%; k = 0이면 하한 0, k = n이면
    상한 1) 같은 식에 넣습니다. 경계 10%: 하한 ≥ 10% → '우연으로 설명됨', 상한 < 10% → '우연으로 보기 어려움',
    그 밖 '가르지 못함'. 씨앗이 1,000개가 다 끝나지 않으면 판정하지 않습니다.
  - 세계: 씨앗 1,000개에서 서로 다른 15개를 뽑는 것을 10,000번(난수 [101, 1]). 세계마다 후보 규칙 수(후보 씨앗
    수)와 후보 칸 수(두 경로의 candidate 줄 수)를 셉니다. 관측값은 실제 15개(재현 (나)의 경로로 센 것).
  - 후보 여유(연속 척도): 칸 여유 = 세 기간(1993–2023, 두 반쪽) × 기준선 둘(SPY + 국채, SPY 200일선 + 국채)의
    (맞춘 수익 − 기준선) 중 최솟값, 20bp 칸만. 40bp는 규칙대로 20bp 통과 칸에만 돌리므로 모든 계열에 있지 않아
    넣지 않고, 창 승률 · 거래 수도 넣지 않습니다(여유가 양수여도 후보가 아닐 수 있음). 계열 여유: 결정 95 =
    min(그 설정의 두 매도 칸, 이웃 설정 중 더 나은 쪽의 두 매도 칸), 결정 96 = min(그 칸의 두 매도, 이웃 설정 중
    나은 쪽, 다른 좁힌 범위의 두 매도). 규칙 여유 = 두 경로의 모든 계열 여유 중 최댓값. maxT p = (1 + 세계 최댓값
    ≥ 실제 15개 최댓값인 세계 수) / 10,001.
  - E7 백분위: E7의 R3 A1 × N10 · N20 · N50 × U1000 · U500 × X1 · H1 칸(20bp) 세 기간 맞춘 수익이 같은 칸의 씨앗
    1,000개 중 몇 번째인지 = 100 × (작은 씨앗 비율 + 같은 씨앗 비율 / 2). N20이 후보 칸, N10 · N50이 이웃 칸.
  - 칸별 통과율: 칸마다 씨앗 1,000개 중 goal_study.cell_pass(20bp)를 채운 비율. Stage S 실제 가격 chance 표의
    P1 칸(P: 설정 3 × 매도 2 × 국면 10, Q: 설정 3 × 매도 2 × 필터 5 × 범위 2)마다 이 비율을 p0로 다시 넣어
    q95 = select_study.q95(15, p0), '우연을 넘음' = 통과 매수 수 > q95를 다시 정하고 Stage S 표시와 견줍니다.
    P0 설정 칸은 돌리지 않았으므로 다루지 않습니다.
  - 미리 적은 예상의 판정: p̂ 0.5–3%, P̂ 7–37%, 판정이 '가르지 못함' 또는 '설명됨', 후보 씨앗의 절반 넘게 결정
    96 경로에서 후보가 나옴, 후보 줄의 절반 넘게 R3 A1 또는 R5 A1, E7 후보 칸 12값(칸 4 × 세 기간) 백분위의
    중앙값이 90–99.
  - 시간 확인: 씨앗 20개(1001–1020)로 표와 계좌 시간을 재서 밤 속도 기준 전체가 20시간을 넘을 것 같으면 멈춥니다.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

from . import account_study as ns, basis_study as bs, exclude_study as ms, exit_study as xs, figures, goal_study as gs
from . import large_study as ls, pair_study as pj, rebase_study as rb, regime_study as rs, select_study as ks, stage_c_study as sc

# ---- registered (DECISIONS 101): fixed before any result
SEEDS = list(range(1001, 2001))
REPRO_SEEDS = [1, 2, 3, 4, 5]
EXITS = ["X1", "H1"]
P1 = list(rs.P1_SETTINGS)  # N10 / N20 / N50 x W2 x P1
P1_NAMES = [ns.setting_name(s) for s in P1]
P_FILTERS = [f for f in rs.FILTERS if f != "none"]  # R1-R5 x A1 / A2 (decision 95's gates)
Q_FILTERS = list(ls.FILTERS)  # none + R1 / R3 / R4 / R5 A1 (decision 96)
Q_BANDS = list(ls.NARROW)  # U1000, U500
CELLS_PER_SEED = 120
WORLD, N_WORLDS = 15, 10_000
BOUNDARY, ALPHA = 0.10, 0.05
FEATURE_KEYS = list(ms.FEATURES) + ["cap_rank", "dv20"]  # account_study.build_base's columns
TABLE_YI = {"X1": 0, "H1": xs.H_KI}
E7_CELL = ("E7", "R3A1")
E7_SETTINGS = P1_NAMES
CAND_SETTING = "N20_W2_P1"
EXPECT = {"p_hat": (0.005, 0.03), "P_hat": (0.07, 0.37), "labels": ("가르지 못함", "우연으로 설명됨"), "e7_pct": (90.0, 99.0)}
LABELS = ("우연으로 설명됨", "우연으로 보기 어려움", "가르지 못함")
TRIALS = rb.TRIALS  # 74,174, unchanged
TOL = rb.TOL
BUDGET_HOURS = 20.0
SEED = 101
_W: dict = {}


def log(m: str) -> None:
    rb.log(m)


def _jd(o):
    """JSON for numpy scalars (bool stays bool)."""
    return o.item() if hasattr(o, "item") else str(o)


def gen_of(seed: int) -> str:
    return f"U{seed}"


def src_of(seed: int) -> str:
    return f"R0s{seed}"


def parse_seeds(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part.strip():
            out.append(int(part))
    return out


# --------------------------------------------------------------------------- paths

def s_paths(a) -> dict:
    s = Path(a.stage_s)
    raw = s / "raw"
    return {"root": s, "raw": raw, "eligible": str(raw / "eligible_daily.parquet"), "base": str(raw / "base.parquet"),
            "dirs": rb.table_dirs(raw / "tables"), "regimes": str(s / "regimes.parquet"),
            "bench_p": str(raw / "benchmarks_p.csv"), "bench_q": str(raw / "benchmarks_q.csv"), "agg": s / "agg"}


def feat_dir(a) -> Path:
    return Path(a.tmp) / "features"


def seed_root(a, seed: int, repro: bool = False) -> Path:
    return Path(a.tmp) / ("repro_tables" if repro else "tables") / f"s{seed}"


def unit_file(d: Path, gen: str) -> Path:
    return d / f"{gen}.parquet"


def units_dir(a) -> Path:
    return Path(a.out) / "units"


def repro_dir(a) -> Path:
    return Path(a.out) / "repro"


# --------------------------------------------------------------------------- feature matrices

def _f_job(job):
    """account_study.point_values (the base table's function) on every day x symbol cell of the panel,
    saved as a days x symbols float32 matrix."""
    key, out = job
    t0 = time.time()
    panel = ks._F["panel"]
    R, C = panel.close.shape
    si = np.repeat(np.arange(R, dtype=np.int64), C)
    sj = np.tile(np.arange(C, dtype=np.int64), R)
    _, vals, _ = ns.point_values((key, si, sj))
    del si, sj
    arr = np.ascontiguousarray(np.asarray(vals, dtype=np.float32).reshape(R, C))
    f = Path(out) / f"{key}.npy"
    tmp = Path(out) / f"{key}.tmp.npy"
    np.save(tmp, arr)
    os.replace(tmp, f)
    meta = {"rows": R, "cols": C, "first": str(panel.close.index[0].date()), "last": str(panel.close.index[-1].date()),
            "cols_crc": zlib.crc32("|".join(map(str, panel.close.columns)).encode())}
    return key, meta, time.time() - t0


def build_features(a) -> dict:
    fd = feat_dir(a)
    mf = fd / "meta.json"
    if mf.exists() and all((fd / f"{k}.npy").exists() for k in FEATURE_KEYS):
        return json.loads(mf.read_text(encoding="utf-8"))
    fd.mkdir(parents=True, exist_ok=True)
    S = s_paths(a)
    todo = [k for k in FEATURE_KEYS if not (fd / f"{k}.npy").exists()]
    log(f"특징 행렬 {len(todo)}개 (모든 날 × 종목)")
    metas = {}
    with rb.use_basis("raw"), ProcessPoolExecutor(max_workers=min(a.feature_workers, len(todo)), initializer=ks._f_init,
                                                  initargs=(a.panel, S["eligible"])) as ex:
        for key, meta, sec in ex.map(_f_job, [(k, str(fd)) for k in todo], chunksize=1):
            metas[key] = meta
            log(f"  {key}: {sec:.0f}s")
    meta = next(iter(metas.values())) if metas else {}
    if len({json.dumps(m, sort_keys=True) for m in metas.values()}) > 1:
        raise SystemExit(f"특징 행렬의 모양이 다릅니다: {metas}")
    mf.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta


# --------------------------------------------------------------------------- signal tables (one seed = one rule)

def _t_init(panel_dir: str, elig_path: str):
    bs._t_init(panel_dir, elig_path, "raw")


def _t_seed(job):
    """Seed `seed`'s X1 and H1 tables (Stage R's E0 menu, the seed's cell at xi 0) written under root."""
    seed, root = job
    t0 = time.time()
    rows = {}
    for x in EXITS:
        name = f"R0__{x}"
        f = bs.table_path(Path(root), name)
        if f.exists():
            rows[x] = -1
            continue
        e, x_ = bs.table_menu(name, seed)
        _, sig = sc._run_signals(("full", e, x_))
        sig = sig.assign(xi=0, yi=TABLE_YI[x]).astype({"return": "float32", "holding_days": "int32", "xi": "int64", "yi": "int64"})
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_suffix(".tmp")
        sig[["xi", "yi"] + sc.SIGNAL_KEEP].to_parquet(tmp, compression="zstd")
        os.replace(tmp, f)
        rows[x] = len(sig)
    cache = sc._W["cache"]  # free this seed's entry frame, masks and exit memos; keep the window arrays
    w = next((k[1] for k in cache._memo if k[0] == "days" and len(k) > 1), None)
    cache.drop_rule_data(w)
    return seed, rows, time.time() - t0


def tables_done(root: Path) -> bool:
    return all(bs.table_path(root, f"R0__{x}").exists() for x in EXITS)


def build_tables(a, seeds: list[int], repro: bool = False) -> dict:
    jobs = [(s, str(seed_root(a, s, repro))) for s in seeds if not tables_done(seed_root(a, s, repro))]
    if repro is False:
        jobs = [j for j in jobs if not unit_file(units_dir(a), gen_of(j[0])).exists()]
    info = {"seeds": len(jobs), "sec": 0.0, "per_seed": []}
    if not jobs:
        return info
    S = s_paths(a)
    log(f"신호표: 씨앗 {len(jobs)}개 × 매도 2 (작업 프로세스 {min(a.table_workers, len(jobs))})")
    t0 = time.time()
    with rb.use_basis("raw"), ProcessPoolExecutor(max_workers=min(a.table_workers, len(jobs)), initializer=_t_init,
                                                  initargs=(a.panel, S["eligible"])) as ex:
        for i, (seed, rows, sec) in enumerate(ex.map(_t_seed, jobs, chunksize=1)):
            info["per_seed"].append({"seed": seed, "sec": sec, **{f"rows_{x}": rows[x] for x in EXITS}})
            el = time.time() - t0
            if (i + 1) % 6 == 0 or i + 1 == len(jobs) or i < 6:
                log(f"  [{i + 1}/{len(jobs)}] 씨앗 {seed}: X1 {rows['X1']:,} · H1 {rows['H1']:,}줄 · {sec:.0f}s · 경과 {el / 60:.1f}분 · "
                    f"남은 예상 {(len(jobs) - i - 1) * el / (i + 1) / 60:.0f}분")
    info["sec"] = time.time() - t0
    return info


# --------------------------------------------------------------------------- one unit (a seed or a real entry): 120 cells + 40 bp

def p_jobs20(gen: str, srcs: list[str]) -> list:
    return [(gen, x, st, list(P_FILTERS), pj.COST_BPS, list(srcs), "grid") for x in EXITS for st in P1]


def q_jobs20(gen: str, srcs: list[str]) -> list:
    bands = [(b, ls.BANDS[b]) for b in Q_BANDS]
    return [(gen, x, st, list(Q_FILTERS), bands, pj.COST_BPS, list(srcs), "grid") for x in EXITS for st in P1]


def _a_init(panel_dir: str, eligible: str, dirs: dict, cfg: dict, bench_p: str, bench_q: str, fdir: str | None):
    rb._w_init(panel_dir, eligible, dirs, cfg)
    _W.update(dirs0=dirs, seeds0=list(cfg["seeds"]), base0=ms._W["base"],
              benchP=rb.bench_from(pd.read_csv(bench_p))[0], benchQ=rb.bench_from(pd.read_csv(bench_q))[0])
    _W["feat"] = {}
    if fdir:
        px = pj._W["px"]
        meta = json.loads((Path(fdir) / "meta.json").read_text(encoding="utf-8"))
        if (meta["rows"], meta["cols"]) != (len(px.dates), len(px.cols)) or meta["cols_crc"] != zlib.crc32("|".join(map(str, px.cols)).encode()):
            raise RuntimeError("특징 행렬이 패널과 맞지 않습니다")
        _W["feat"] = {k: np.load(Path(fdir) / f"{k}.npy", mmap_mode="r") for k in FEATURE_KEYS}


def use_unit(dirs: dict, seeds: list[int], base: pd.DataFrame) -> None:
    """Point the stages' table readers at one unit's tables and base rows."""
    pj.E0_SEEDS = list(seeds)
    pj._W["dirs"] = dirs
    xs._W["dirs"] = dirs
    ms._W["base"] = base
    ms._W["prep"].clear()
    pj._W["tables"].clear()


def unit_base(srcs: list[str]) -> pd.DataFrame:
    """account_study.build_base's rows for the unit's sources (exits X1 / H1) with the feature
    matrices read at each signal's (si, sj)."""
    cols = pj._W["px"].cols
    parts = []
    for s in srcs:
        for x in EXITS:
            t = pj.rule_table(s, x)
            parts.append(pd.DataFrame({"src": s, "si": t["si"].to_numpy(), "sj": cols.get_indexer(t["symbol"].astype(str))}))
        pj._W["tables"].clear()
    base = pd.concat(parts, ignore_index=True).drop_duplicates(["src", "si", "sj"])
    base = base.sort_values(["src", "si", "sj"], kind="stable").reset_index(drop=True)
    pts = base[["si", "sj"]].drop_duplicates().reset_index(drop=True)
    si, sj = pts["si"].to_numpy(), pts["sj"].to_numpy()
    for k in FEATURE_KEYS:
        pts[k] = np.asarray(_W["feat"][k][si, sj], dtype=np.float32)
    return base.merge(pts, on=["si", "sj"], how="left")


def run_unit(unit):
    """unit = (gen, srcs, tables root or None (= Stage S's tables and base), seeds, keep base): the 120
    cells at 20 bp and the stage rule's 40 bp cells; account rows with 'path' P / Q."""
    gen, srcs, root, seeds, keep = unit
    t0 = time.time()
    base = None
    if root is None:
        use_unit(_W["dirs0"], _W["seeds0"], _W["base0"])
    else:
        use_unit(rb.table_dirs(Path(root)), seeds, _W["base0"])
        base = unit_base(srcs)
        use_unit(rb.table_dirs(Path(root)), seeds, base)
    acc = {"P": [rs.run_job(j)[1] for j in p_jobs20(gen, srcs)], "Q": [ls.run_job(j)[1] for j in q_jobs20(gen, srcs)]}
    aP, aQ = pd.concat(acc["P"], ignore_index=True), pd.concat(acc["Q"], ignore_index=True)
    j40 = rb.p_jobs40(rs.summarize(aP), _W["benchP"], {gen: list(srcs)}) + rb.q_jobs40(ls.summarize(aQ), _W["benchQ"], {gen: list(srcs)})
    for kind, job in j40:
        acc[kind].append((rs.run_job if kind == "P" else ls.run_job)(job)[1])
    out = pd.concat([pd.concat(acc["P"], ignore_index=True).assign(path="P"), pd.concat(acc["Q"], ignore_index=True).assign(path="Q")],
                    ignore_index=True)
    info = {"base_rows": len(base) if base is not None else -1, "jobs40": len(j40), "unit_sec": time.time() - t0}
    use_unit(_W["dirs0"], _W["seeds0"], _W["base0"])
    return gen, out.assign(**info), (base if keep else None), info


def run_units(a, units: list, d: Path, workers: int, fdir: str | None, base_path: str, keep_dir: Path | None = None) -> float:
    """Units through a worker pool; each finished unit is written to d/<gen>.parquet (its checkpoint)."""
    todo = [u for u in units if not unit_file(d, u[0]).exists()]
    log(f"계좌: 단위 {len(todo)}개 남음 (전체 {len(units)}, 작업 프로세스 {min(workers, max(len(todo), 1))})")
    if not todo:
        return 0.0
    d.mkdir(parents=True, exist_ok=True)
    S = s_paths(a)
    cfg = {"bins": pj.BINS_J3, "seeds": pj.E0_SEEDS, "base": base_path, "regimes": S["regimes"]}
    t0 = time.time()
    with rb.use_basis("raw"), ProcessPoolExecutor(max_workers=min(workers, len(todo)), initializer=_a_init,
                                                  initargs=(a.panel, S["eligible"], S["dirs"], cfg, S["bench_p"], S["bench_q"], fdir)) as ex:
        futs = [ex.submit(run_unit, u) for u in todo]
        for i, f in enumerate(as_completed(futs)):
            gen, acc, base, info = f.result()
            tmp = d / f"{gen}.tmp.parquet"
            acc.to_parquet(tmp)
            os.replace(tmp, unit_file(d, gen))
            if base is not None and keep_dir is not None:
                keep_dir.mkdir(parents=True, exist_ok=True)
                base.to_parquet(keep_dir / f"base_{gen}.parquet")
            el = time.time() - t0
            if (i + 1) % 10 == 0 or i + 1 == len(todo) or i < 3:
                log(f"  [{i + 1}/{len(todo)}] {gen}: {info['unit_sec']:.0f}s (40bp 작업 {info['jobs40']}) · 경과 {el / 60:.1f}분 · "
                    f"남은 예상 {(len(todo) - i - 1) * el / (i + 1) / 60:.0f}분")
    return time.time() - t0


def empty_base(a) -> str:
    f = Path(a.tmp) / "base_empty.parquet"
    if not f.exists():
        pd.read_parquet(s_paths(a)["base"]).head(0).to_parquet(f)
    return str(f)


# --------------------------------------------------------------------------- reading units back

UNIT_COLS = ("path", "base_rows", "jobs40", "unit_sec")
Q_ONLY = ("band", "beta", "cap_fill", "positions_mean", "cash_share")


def split_acc(acc: pd.DataFrame) -> dict:
    """{P20, Q20, P40, Q40} account rows of a frame of unit rows (P rows without the band column)."""
    out = {}
    for path in ("P", "Q"):
        a_ = acc[acc["path"] == path].drop(columns=[c for c in UNIT_COLS if c in acc])
        if path == "P":  # columns only the Q path writes (all empty on P rows)
            a_ = a_.drop(columns=[c for c in a_.columns if c in Q_ONLY and a_[c].isna().all()])
        for cost, tag in ((pj.COST_BPS, "20"), (pj.CHECK_COST_BPS, "40")):
            out[f"{path}{tag}"] = a_[a_["cost"] == cost].reset_index(drop=True)
    return out


def read_units(d: Path, gens: list[str]) -> dict:
    parts = [pd.read_parquet(unit_file(d, g)) for g in gens]
    return split_acc(pd.concat(parts, ignore_index=True))


def stage_s_tables(a) -> dict:
    """Stage S's raw-basis tables this runner reads: goal rows (E0 rows feed the entry-specific label),
    40 bp rows, benchmarks, stored cell flags, candidate and chance tables."""
    S = s_paths(a)
    raw, agg = S["raw"], S["agg"]
    gP, gQ = pd.read_csv(raw / "p" / "goal.csv"), pd.read_csv(raw / "q" / "goal.csv")
    sel = lambda g: g[(g["gen"] == "E0") & (g["mode"] == "grid") & (g["cost"] == pj.COST_BPS)]  # noqa: E731
    cells = pd.read_csv(agg / "cells.csv.gz")
    return {"gP": gP, "gQ": gQ, "gP40": pd.read_csv(raw / "p" / "goal40.csv"), "gQ40": pd.read_csv(raw / "q" / "goal40.csv"),
            "e0P": sel(gP).reset_index(drop=True), "e0Q": sel(gQ).reset_index(drop=True),
            "benchP": rb.bench_from(pd.read_csv(S["bench_p"]))[0], "benchQ": rb.bench_from(pd.read_csv(S["bench_q"]))[0],
            "cells": cells[cells["basis"] == "raw"].reset_index(drop=True),
            "candP": pd.read_csv(agg / "candidates_p.csv").query("basis == 'raw'").reset_index(drop=True),
            "candQ": pd.read_csv(agg / "candidates_q.csv").query("basis == 'raw'").reset_index(drop=True),
            "chP": pd.read_csv(agg / "chance_p.csv").query("basis == 'raw'").reset_index(drop=True),
            "chQ": pd.read_csv(agg / "chance_q.csv").query("basis == 'raw'").reset_index(drop=True)}


# --------------------------------------------------------------------------- the candidate path

def judge_units(acc: dict, S: dict) -> dict:
    """regime_study.candidates and large_study.candidates, unchanged, on the units' cells with Stage S's
    E0 rows and benchmarks; the 40 bp flags from the units' 40 bp cells."""
    gP = pd.concat([rs.summarize(acc["P20"]), S["e0P"]], ignore_index=True)
    gQ = pd.concat([ls.summarize(acc["Q20"]), S["e0Q"]], ignore_index=True)
    pass40P = {(r["setting"], r["gen"], r["exit"], r["filter"]): gs.cell_pass(r, S["benchP"])
               for _, r in (rs.summarize(acc["P40"]).iterrows() if len(acc["P40"]) else iter(()))}
    pass40Q = {(r["setting"], r["gen"], r["exit"], r["filter"], r["band"]): gs.cell_pass(r, S["benchQ"])
               for _, r in (ls.summarize(acc["Q40"]).iterrows() if len(acc["Q40"]) else iter(()))}
    cellsP, candP = rs.candidates(gP, S["benchP"], pass40P)
    cellsQ, candQ = ls.candidates(gQ, S["benchQ"], pass40Q)
    keep = lambda c: c[(c["gen"] != "E0") & c["setting"].isin(P1_NAMES)].reset_index(drop=True)  # noqa: E731
    return {"cellsP": keep(cellsP), "candP": candP[candP["gen"] != "E0"].reset_index(drop=True),
            "cellsQ": keep(cellsQ), "candQ": candQ[candQ["gen"] != "E0"].reset_index(drop=True)}


def cell_margin(c: pd.DataFrame, bench: dict) -> np.ndarray:
    """min over the judged periods and the gated benchmarks of (dd_cagr - benchmark)."""
    m = np.full(len(c), np.inf)
    for p in gs.JUDGED:
        for b in gs.GATED:
            m = np.minimum(m, c[f"dd_cagr_{p}"].to_numpy(dtype=float) - bench[b][f"dd_cagr_{p}"])
    return m


def margins(cellsP: pd.DataFrame, cellsQ: pd.DataFrame, benchP: dict, benchQ: dict) -> pd.DataFrame:
    """Per unit the candidate margin (see the module docstring) and the family where it is largest."""
    def own(c, bench, keys):
        c = c[c["cost"] == pj.COST_BPS]
        m = pd.Series(cell_margin(c, bench), index=c.index)
        g = m.groupby([c[k] for k in keys]).agg(["min", "size"])
        if (g["size"] != len(EXITS)).any():
            raise ValueError("칸이 매도 둘을 다 갖지 않습니다")
        return g["min"].to_dict()

    oP = own(cellsP, benchP, ["gen", "setting", "filter"])
    oQ = own(cellsQ, benchQ, ["gen", "setting", "filter", "band"])
    other = {"U1000": "U500", "U500": "U1000"}
    best: dict = {}

    def put(gen, v, fam):
        if gen not in best or v > best[gen][0]:
            best[gen] = (v, fam)

    for (gen, st, f), v in oP.items():
        i = P1_NAMES.index(st)
        nb = max((oP.get((gen, P1_NAMES[j], f), -np.inf) for j in (i - 1, i + 1) if 0 <= j < len(P1_NAMES)), default=-np.inf)
        put(gen, min(v, nb), f"95 {st} {f}")
    for (gen, st, f, b), v in oQ.items():
        i = P1_NAMES.index(st)
        nb = max((oQ.get((gen, P1_NAMES[j], f, b), -np.inf) for j in (i - 1, i + 1) if 0 <= j < len(P1_NAMES)), default=-np.inf)
        put(gen, min(v, nb, oQ.get((gen, st, f, other[b]), -np.inf)), f"96 {st} {f} {b}")
    return pd.DataFrame([{"gen": g, "margin": v, "margin_family": fam} for g, (v, fam) in best.items()])


def unit_table(J: dict, benchP: dict, benchQ: dict) -> pd.DataFrame:
    """Per unit: passing cells (of 120, 20 bp), candidate rows per route, entry-specific rows, margin."""
    cP, cQ = J["cellsP"], J["cellsQ"]
    cP20, cQ20 = cP[cP["cost"] == pj.COST_BPS], cQ[cQ["cost"] == pj.COST_BPS]
    gens = sorted(set(cP20["gen"]) | set(cQ20["gen"]))
    t = pd.DataFrame(index=pd.Index(gens, name="gen"))
    t["cells"] = cP20.groupby("gen").size().reindex(gens, fill_value=0) + cQ20.groupby("gen").size().reindex(gens, fill_value=0)
    t["pass_cells_p"] = cP20.groupby("gen")["cell_pass"].sum().reindex(gens, fill_value=0).astype(int)
    t["pass_cells_q"] = cQ20.groupby("gen")["cell_pass"].sum().reindex(gens, fill_value=0).astype(int)
    t["pass_cells"] = t["pass_cells_p"] + t["pass_cells_q"]
    for route, cand in (("95", J["candP"]), ("96", J["candQ"])):
        c = cand[cand["candidate"]] if len(cand) else cand
        t[f"cand_{route}"] = c.groupby("gen").size().reindex(gens, fill_value=0).astype(int) if len(c) else 0
        e = c[c["entry_specific"]] if len(c) else c
        t[f"es_{route}"] = e.groupby("gen").size().reindex(gens, fill_value=0).astype(int) if len(e) else 0
    t["cand_cells"] = t["cand_95"] + t["cand_96"]
    t["candidate"] = t["cand_cells"] > 0
    t["entry_specific"] = (t["es_95"] + t["es_96"]) > 0
    mg = margins(cP, cQ, benchP, benchQ).set_index("gen")
    t = t.join(mg)
    return t.reset_index()


def candidate_rows(J: dict) -> pd.DataFrame:
    parts = []
    for route, key in (("95", "candP"), ("96", "candQ")):
        c = J[key]
        if len(c):
            parts.append(c[c["candidate"]].assign(route=route))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=["gen", "route"])


# --------------------------------------------------------------------------- the primary verdict

def binom_cdf(k: int, n: int, p: float) -> float:
    if k < 0:
        return 0.0
    if k >= n or p <= 0.0:
        return 1.0
    if p >= 1.0:
        return 0.0
    lp, lq = math.log(p), math.log1p(-p)
    t = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) + i * lp + (n - i) * lq for i in range(k + 1)]
    m = max(t)
    return min(1.0, math.exp(m) * sum(math.exp(x - m) for x in t))


def _bisect(ok, lo: float, hi: float) -> float:
    for _ in range(200):
        mid = (lo + hi) / 2
        if ok(mid):
            hi = mid
        else:
            lo = mid
    return (lo + hi) / 2


def cp_upper(k: int, n: int, alpha: float = ALPHA) -> float:
    """One-sided Clopper-Pearson upper limit: the p with P(X <= k) = alpha."""
    if k >= n:
        return 1.0
    return _bisect(lambda p: binom_cdf(k, n, p) <= alpha, k / n, 1.0)


def cp_lower(k: int, n: int, alpha: float = ALPHA) -> float:
    """One-sided Clopper-Pearson lower limit: the p with P(X >= k) = alpha."""
    if k <= 0:
        return 0.0
    return _bisect(lambda p: 1.0 - binom_cdf(k - 1, n, p) >= alpha, 0.0, k / n)


def world_p(p: float, m: int = WORLD) -> float:
    return 1.0 - (1.0 - p) ** m


def verdict(k: int, n: int) -> dict:
    p = k / n
    lo, hi = cp_lower(k, n), cp_upper(k, n)
    P_lo, P_hi = world_p(lo), world_p(hi)
    label = LABELS[0] if P_lo >= BOUNDARY else LABELS[1] if P_hi < BOUNDARY else LABELS[2]
    return {"k": int(k), "n": int(n), "p_hat": p, "p_lower": lo, "p_upper": hi, "P_hat": world_p(p), "P_lower": P_lo, "P_upper": P_hi,
            "label": label}


def draw_worlds(n_seeds: int, n_worlds: int = N_WORLDS, size: int = WORLD) -> np.ndarray:
    rng = np.random.default_rng([SEED, 1])
    return np.array([rng.choice(n_seeds, size, replace=False) for _ in range(n_worlds)])


def maxt_p(world_max: np.ndarray, observed: float) -> float:
    return float((1 + int((world_max >= observed).sum())) / (len(world_max) + 1))


def pct_rank(values: np.ndarray, x: float) -> float:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    return float(100.0 * ((v < x).mean() + 0.5 * (v == x).mean())) if len(v) else np.nan


# --------------------------------------------------------------------------- stages

def need_repro(a) -> dict:
    f = repro_dir(a) / "repro.json"
    rp = json.loads(f.read_text(encoding="utf-8")) if f.exists() else {}
    if not rp.get("pass"):
        raise SystemExit("재현 점검을 통과하지 않아 이 단계는 돌지 않습니다 (repro/repro.json).")
    return rp


def flags_compare(mine: pd.DataFrame, stored: pd.DataFrame, keys: list[str], cols: list[str]) -> dict:
    m = mine.set_index(keys)[cols]
    s = stored.set_index(keys)[cols]
    common = s.index.intersection(m.index)
    diff = (m.loc[common].astype(bool).to_numpy() != s.loc[common].astype(bool).to_numpy())
    return {"rows": len(common), "missing": int(len(s.index.difference(m.index))), "extra": int(len(m.index.difference(s.index))),
            "differ": int(diff.any(axis=1).sum()), "pass": bool(len(common) and not diff.any() and len(common) == len(s) == len(m))}


def stage_repro(a) -> dict:
    rd = repro_dir(a)
    rd.mkdir(parents=True, exist_ok=True)
    S = s_paths(a)
    T = stage_s_tables(a)
    res: dict = {}
    build_features(a)
    # (a) tables of seeds 1-5 through the new path
    log("재현 (가): 씨앗 1–5의 신호표를 새 경로로 → 저장된 실제 가격 E0 표")
    t_info = build_tables(a, REPRO_SEEDS, repro=True)
    rows = []
    for k, s in enumerate(REPRO_SEEDS):
        for x in EXITS:
            name = f"R0__{x}"
            mine = bs.read_cell(bs.table_path(seed_root(a, s, True), name), 0, TABLE_YI[x])
            theirs = bs.read_cell(bs.table_path(Path(S["raw"]) / "tables", name), k, TABLE_YI[x])
            rows.append({"seed": s, "exit": x, **bs.same_rows(mine, theirs)})
    tb = pd.DataFrame(rows)
    tb.to_csv(rd / "repro_tables.csv", index=False)
    res["a_tables"] = tb.to_dict("records")
    res["a_tables_pass"] = bool(tb["equal"].all())
    log("\n" + tb.to_string(index=False))
    # units: seeds 1-5 (own tables, base from the feature matrices) and the 15 real entries (Stage S's)
    units = [(gen_of(s), [src_of(s)], str(seed_root(a, s, True)), [s], True) for s in REPRO_SEEDS]
    units += [(e, [e], None, None, False) for e in rb.ENTRIES]
    run_units(a, units, rd / "units", a.workers, str(feat_dir(a)), S["base"], keep_dir=Path(a.tmp) / "repro_base")
    # (a) base rows
    sb = pd.read_parquet(S["base"])
    brows = []
    for s in REPRO_SEEDS:
        mine = pd.read_parquet(Path(a.tmp) / "repro_base" / f"base_{gen_of(s)}.parquet").reset_index(drop=True)
        theirs = sb[sb["src"] == src_of(s)].reset_index(drop=True)
        brows.append({"seed": s, "rows_new": len(mine), "rows_stored": len(theirs), "equal": bool(mine.equals(theirs))})
    bt = pd.DataFrame(brows)
    bt.to_csv(rd / "repro_base.csv", index=False)
    res["a_base"] = bt.to_dict("records")
    log("\n" + bt.to_string(index=False))
    # (a) the seeds' cells against Stage S's E0 rows per seed
    seeds_acc = read_units(rd / "units", [gen_of(s) for s in REPRO_SEEDS])
    jP = pd.concat([pd.read_csv(f) for f in sorted((Path(S["raw"]) / "p" / "jobs").glob("E0__*P1__grid__all_acc.csv"))], ignore_index=True)
    jQ = pd.concat([pd.read_csv(f) for f in sorted((Path(S["raw"]) / "q" / "jobs").glob("E0__*P1__grid__all_acc.csv"))], ignore_index=True)
    jP = jP[jP["filter"].isin(P_FILTERS)]
    jQ = jQ[jQ["filter"].isin(Q_FILTERS) & jQ["band"].isin(Q_BANDS)]
    kP = ["setting", "exit", "filter", "src", "cost", "mode"]
    for name, mine, theirs, keys in (("a_acc_p", seeds_acc["P20"], jP, kP), ("a_acc_q", seeds_acc["Q20"], jQ, kP + ["band"])):
        c = rb.compare(mine.drop(columns=["gen"]), theirs.drop(columns=["gen"]), keys)
        c.to_csv(rd / f"repro_{name}.csv.gz", index=False)
        res[name] = rb.summary_of(c)
        log(f"  {name}: {res[name]}")
    # (b) the 15 real entries: values, then the candidate path
    real = read_units(rd / "units", rb.ENTRIES)
    mP = rs.summarize(real["P20"])
    mQ = ls.summarize(real["Q20"])
    sP = T["gP"][T["gP"]["gen"].isin(rb.ENTRIES) & T["gP"]["setting"].isin(P1_NAMES) & T["gP"]["filter"].isin(P_FILTERS)
                 & (T["gP"]["mode"] == "grid") & (T["gP"]["cost"] == pj.COST_BPS)]
    sQ = T["gQ"][T["gQ"]["gen"].isin(rb.ENTRIES) & T["gQ"]["setting"].isin(P1_NAMES) & T["gQ"]["filter"].isin(Q_FILTERS)
                 & T["gQ"]["band"].isin(Q_BANDS) & (T["gQ"]["mode"] == "grid") & (T["gQ"]["cost"] == pj.COST_BPS)]
    s40P = T["gP40"][T["gP40"]["gen"].isin(rb.ENTRIES)]
    s40Q = T["gQ40"][T["gQ40"]["gen"].isin(rb.ENTRIES)]
    checks = {"b_goal_p": (mP, sP, rb.P_KEYS), "b_goal_q": (mQ, sQ, rb.Q_KEYS),
              "b_40bp_p": (rs.summarize(real["P40"]), s40P, rb.P_KEYS), "b_40bp_q": (ls.summarize(real["Q40"]), s40Q, rb.Q_KEYS)}
    for name, (mine, theirs, keys) in checks.items():
        c = rb.compare(mine, theirs, keys)
        c.to_csv(rd / f"repro_{name}.csv.gz", index=False)
        res[name] = rb.summary_of(c)
        log(f"  {name}: {res[name]}")
    J = judge_units(real, T)
    cells = T["cells"]
    sc_P = cells[(cells["scope"] == "P") & cells["gen"].isin(rb.ENTRIES) & cells["setting"].isin(P1_NAMES) & (cells["cost"] == pj.COST_BPS)]
    sc_Q = cells[(cells["scope"] == "Q") & cells["gen"].isin(rb.ENTRIES) & cells["setting"].isin(P1_NAMES) & (cells["cost"] == pj.COST_BPS)]
    flag_cols = ["cell_pass", "beats_e0", "pass_40bp"]
    res["b_flags_p"] = flags_compare(J["cellsP"], sc_P, ["setting", "gen", "exit", "filter"], flag_cols)
    res["b_flags_q"] = flags_compare(J["cellsQ"], sc_Q, ["setting", "gen", "exit", "filter", "band"], flag_cols)
    cand_cols = ["n20", "n40", "n_e0", "ok", "candidate", "entry_specific"]
    stP = T["candP"][T["candP"]["gen"].isin(rb.ENTRIES)]
    stQ = T["candQ"][T["candQ"]["gen"].isin(rb.ENTRIES)]
    kp, kq = ["setting", "gen", "filter"], ["setting", "gen", "filter", "band"]
    res["b_cand_p"] = flags_compare(J["candP"], stP, kp, cand_cols + ["neighbour"])
    res["b_cand_q"] = flags_compare(J["candQ"], stQ, kq, cand_cols + ["neighbour_n", "neighbour_band"])
    counts = ["n20", "n40", "n_e0"]
    res["b_cand_counts_p"] = bool(np.array_equal(J["candP"].set_index(kp)[counts].sort_index().to_numpy(dtype=np.int64),
                                                 stP.set_index(kp)[counts].sort_index().to_numpy(dtype=np.int64)))
    res["b_cand_counts_q"] = bool(np.array_equal(J["candQ"].set_index(kq)[counts].sort_index().to_numpy(dtype=np.int64),
                                                 stQ.set_index(kq)[counts].sort_index().to_numpy(dtype=np.int64)))
    cr = candidate_rows(J)
    res["b_candidates"] = cr[["route", "setting", "gen", "filter"] + (["band"] if "band" in cr else []) + ["entry_specific"]].to_dict("records")
    want = {("96", "N20_W2_P1", "E7", "R3A1", "U1000"), ("96", "N20_W2_P1", "E7", "R3A1", "U500")}
    got = {(r["route"], r["setting"], r["gen"], r["filter"], r.get("band")) for r in res["b_candidates"]}
    res["b_candidates_pass"] = got == want
    log(f"  후보 경로: 칸 표시 P {res['b_flags_p']} · Q {res['b_flags_q']} · 후보표 P {res['b_cand_p']} · Q {res['b_cand_q']} · 후보 {sorted(got)}")
    res["tables_timing"] = t_info
    res["pass"] = bool(res["a_tables_pass"] and bt["equal"].all() and res["a_acc_p"]["pass"] and res["a_acc_q"]["pass"]
                       and all(res[k]["pass"] for k in checks) and res["b_flags_p"]["pass"] and res["b_flags_q"]["pass"]
                       and res["b_cand_p"]["pass"] and res["b_cand_q"]["pass"] and res["b_cand_counts_p"] and res["b_cand_counts_q"]
                       and res["b_candidates_pass"])
    (rd / "repro.json").write_text(json.dumps(res, ensure_ascii=False, indent=1, default=_jd), encoding="utf-8")
    log(f"재현 판정: {'통과' if res['pass'] else '실패 — 뒤 단계는 돌지 않습니다'}")
    return res


def stage_tables(a) -> None:
    need_repro(a)
    build_features(a)
    seeds = parse_seeds(a.seeds)
    info = build_tables(a, seeds)
    f = Path(a.out) / f"tables_timing_{seeds[0]}-{seeds[-1]}.json"
    if info["seeds"]:
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text(json.dumps(info, indent=1), encoding="utf-8")
    missing = [s for s in seeds if not tables_done(seed_root(a, s)) and not unit_file(units_dir(a), gen_of(s)).exists()]
    if missing:
        raise SystemExit(f"신호표가 없는 씨앗: {missing[:10]} …")
    (Path(a.out) / "tables_done.json").write_text(json.dumps({"seeds": f"{seeds[0]}-{seeds[-1]}", "time": time.strftime("%Y-%m-%d %H:%M:%S")}),
                                                  encoding="utf-8")


def stage_accounts(a) -> None:
    need_repro(a)
    seeds = parse_seeds(a.seeds)
    missing = [s for s in seeds if not unit_file(units_dir(a), gen_of(s)).exists() and not tables_done(seed_root(a, s))]
    if missing:
        raise SystemExit(f"신호표가 없는 씨앗: {missing[:10]} …")
    units = [(gen_of(s), [src_of(s)], str(seed_root(a, s)), [s], False) for s in seeds]
    sec = run_units(a, units, units_dir(a), a.workers, str(feat_dir(a)), empty_base(a))
    done = [s for s in seeds if unit_file(units_dir(a), gen_of(s)).exists()]
    out = {"seeds": f"{seeds[0]}-{seeds[-1]}", "done": len(done), "of": len(seeds), "sec": sec, "time": time.strftime("%Y-%m-%d %H:%M:%S")}
    (Path(a.out) / f"accounts_done_{seeds[0]}-{seeds[-1]}.json").write_text(json.dumps(out), encoding="utf-8")
    log(f"계좌 끝: {out}")


# --------------------------------------------------------------------------- report

def chance_flags(J: dict, T: dict) -> pd.DataFrame:
    """Stage S's raw chance rows (P1 cells) with p0 from the 1,000 seeds' pass rate of the same cell."""
    rows = []
    for scope, cells, ch, keys in (("P", J["cellsP"], T["chP"], ["setting", "exit", "filter"]),
                                   ("Q", J["cellsQ"], T["chQ"], ["setting", "exit", "filter", "band"])):
        c = cells[cells["cost"] == pj.COST_BPS]
        rate = c.groupby(keys)["cell_pass"].agg(["mean", "size"])
        s = ch[ch["setting"].isin(P1_NAMES)]
        if scope == "Q":
            s = s[s["band"].isin(Q_BANDS)]
        else:
            s = s[s["filter"].isin(P_FILTERS)]
        for _, r in s.iterrows():
            k = tuple(r[x] for x in keys)
            if k not in rate.index:
                continue
            p0 = float(rate.loc[k, "mean"])
            q = ks.q95(int(r["entries"]), p0)
            rows.append({"scope": scope, **{x: r[x] for x in keys}, "entries": int(r["entries"]), "passing": int(r["passing"]),
                         "passing_list": r["passing_list"] if isinstance(r["passing_list"], str) else "",
                         "seeds5_passing": int(r["seeds_passing"]), "p0_5": float(r["p0"]), "q95_5": int(r["q95"]),
                         "beyond_5": bool(r["beyond_chance"]), "seeds": int(rate.loc[k, "size"]), "p0_1000": p0, "q95_1000": int(q),
                         "beyond_1000": bool(r["passing"] > q)})
    out = pd.DataFrame(rows)
    out["changed"] = out["beyond_5"] != out["beyond_1000"]
    return out


def cell_rates(J: dict) -> pd.DataFrame:
    parts = []
    for scope, cells, keys in (("P", J["cellsP"], ["setting", "exit", "filter"]), ("Q", J["cellsQ"], ["setting", "exit", "filter", "band"])):
        c = cells[cells["cost"] == pj.COST_BPS]
        g = c.groupby(keys).agg(seeds=("cell_pass", "size"), pass_rate=("cell_pass", "mean"),
                                **{f"median_dd_cagr_{p}": (f"dd_cagr_{p}", "median") for p in gs.JUDGED}).reset_index()
        parts.append(g.assign(scope=scope))
    return pd.concat(parts, ignore_index=True)


def e7_percentiles(Jn: dict, Jr: dict) -> pd.DataFrame:
    rows = []
    cn, cr = Jn["cellsQ"], Jr["cellsQ"]
    cn, cr = cn[cn["cost"] == pj.COST_BPS], cr[cr["cost"] == pj.COST_BPS]
    for st in E7_SETTINGS:
        for b in Q_BANDS:
            for x in EXITS:
                sel = lambda c: c[(c["setting"] == st) & (c["band"] == b) & (c["exit"] == x) & (c["filter"] == E7_CELL[1])]  # noqa: E731
                n_, e_ = sel(cn), sel(cr[cr["gen"] == E7_CELL[0]])
                if not len(e_):
                    continue
                e_ = e_.iloc[0]
                for p in gs.JUDGED:
                    v = n_[f"dd_cagr_{p}"].to_numpy(dtype=float)
                    rows.append({"setting": st, "band": b, "exit": x, "period": p, "candidate_cell": st == CAND_SETTING, "e7": float(e_[f"dd_cagr_{p}"]),
                                 "e7_cell_pass": bool(e_["cell_pass"]), "null_median": float(np.median(v)), "null_q90": float(np.quantile(v, 0.9)),
                                 "null_q99": float(np.quantile(v, 0.99)), "percentile": pct_rank(v, float(e_[f"dd_cagr_{p}"])), "seeds": len(v)})
    return pd.DataFrame(rows)


def stage_report(a) -> dict:
    rp = need_repro(a)
    T = stage_s_tables(a)
    agg = Path(a.out) / "agg"
    (agg / "figures").mkdir(parents=True, exist_ok=True)
    seeds = parse_seeds(a.seeds)
    gens = [gen_of(s) for s in seeds]
    missing = [g for g in gens if not unit_file(units_dir(a), g).exists()]
    if missing:
        raise SystemExit(f"끝나지 않은 씨앗 {len(missing)}개 — 판정하지 않습니다 ({missing[:5]} …)")
    log(f"씨앗 {len(gens)}개 읽기")
    acc = read_units(units_dir(a), gens)
    real = read_units(repro_dir(a) / "units", rb.ENTRIES)
    Jn = judge_units(acc, T)
    Jr = judge_units(real, T)
    un = unit_table(Jn, T["benchP"], T["benchQ"])
    ur = unit_table(Jr, T["benchP"], T["benchQ"])
    if len(un) != len(gens) or not (un["cells"] == CELLS_PER_SEED).all():
        raise SystemExit("씨앗마다 120칸이 아닙니다")
    un["seed"] = un["gen"].str[1:].astype(int)
    un = un.sort_values("seed").reset_index(drop=True)
    k = int(un["candidate"].sum())
    V = verdict(k, len(un))
    cr_n, cr_r = candidate_rows(Jn), candidate_rows(Jr)
    routes = {"95": int((un["cand_95"] > 0).sum()), "96": int((un["cand_96"] > 0).sum()),
              "both": int(((un["cand_95"] > 0) & (un["cand_96"] > 0)).sum()), "entry_specific": int(un["entry_specific"].sum())}
    route_v = {r: verdict(routes[r], len(un)) for r in ("95", "96", "entry_specific")}
    # worlds
    W = draw_worlds(len(un))
    cand = un["candidate"].to_numpy()
    ccells = un["cand_cells"].to_numpy()
    marg = un["margin"].to_numpy(dtype=float)
    w_entries, w_cells, w_max = cand[W].sum(axis=1), ccells[W].sum(axis=1), marg[W].max(axis=1)
    obs = {"entries": int(ur["candidate"].sum()), "cells": int(ur["cand_cells"].sum()), "max_margin": float(ur["margin"].max()),
           "max_margin_entry": str(ur.loc[ur["margin"].idxmax(), "gen"])}
    worlds = {"n": N_WORLDS, "size": WORLD, "share_ge1_entry": float((w_entries >= 1).mean()),
              "share_ge_obs_entries": float((w_entries >= obs["entries"]).mean()), "share_ge_obs_cells": float((w_cells >= obs["cells"]).mean()),
              "entries_dist": {int(v): int(c) for v, c in zip(*np.unique(w_entries, return_counts=True))},
              "cells_dist": {int(v): int(c) for v, c in zip(*np.unique(w_cells, return_counts=True))},
              "max_margin_quantiles": {q: float(np.quantile(w_max, q)) for q in (0.05, 0.5, 0.95)},
              "maxt_p": maxt_p(w_max, obs["max_margin"])}
    # reported tables
    flags = chance_flags(Jn, T)
    rates = cell_rates(Jn)
    e7 = e7_percentiles(Jn, Jr)
    gate_share = float(cr_n["filter"].isin(["R3A1", "R5A1"]).mean()) if len(cr_n) else np.nan
    route96_share = float((un.loc[un["candidate"], "cand_96"] > 0).mean()) if k else np.nan
    e7_med = float(e7.loc[e7["candidate_cell"], "percentile"].median()) if len(e7) else np.nan
    expect = pd.DataFrame([
        {"item": "p̂ (후보 씨앗 비율)", "expected": "0.5–3%", "actual": f"{V['p_hat']:.2%}", "met": EXPECT["p_hat"][0] <= V["p_hat"] <= EXPECT["p_hat"][1]},
        {"item": "P̂ (15개 세계에 후보 하나 이상)", "expected": "7–37%", "actual": f"{V['P_hat']:.1%}", "met": EXPECT["P_hat"][0] <= V["P_hat"] <= EXPECT["P_hat"][1]},
        {"item": "판정", "expected": "가르지 못함 또는 우연으로 설명됨", "actual": V["label"], "met": V["label"] in EXPECT["labels"]},
        {"item": "후보 씨앗이 주로 결정 96 경로", "expected": "절반 넘게", "actual": "후보 없음" if not k else f"{route96_share:.0%}",
         "met": bool(k and route96_share > 0.5)},
        {"item": "후보 줄의 국면이 주로 R3 · R5 A1", "expected": "절반 넘게", "actual": "후보 없음" if not len(cr_n) else f"{gate_share:.0%}",
         "met": bool(len(cr_n) and gate_share > 0.5)},
        {"item": "E7 후보 칸의 백분위", "expected": "90–99 (12값 중앙값)", "actual": f"{e7_med:.1f}",
         "met": bool(EXPECT["e7_pct"][0] <= e7_med <= EXPECT["e7_pct"][1])},
    ])
    # write aggregates
    un.to_csv(agg / "units.csv", index=False)
    ur.to_csv(agg / "real_entries.csv", index=False)
    cr_n.to_csv(agg / "candidates_null.csv", index=False)
    cr_r.to_csv(agg / "candidates_real.csv", index=False)
    flags.to_csv(agg / "chance_flags.csv", index=False)
    rates.to_csv(agg / "cell_rates.csv", index=False)
    e7.to_csv(agg / "e7_percentiles.csv", index=False)
    expect.to_csv(agg / "expectations.csv", index=False)
    pd.DataFrame({"entries": w_entries, "cells": w_cells, "max_margin": w_max}).describe().to_csv(agg / "worlds_summary.csv")
    big = Path(a.out) / "agg_big"
    big.mkdir(parents=True, exist_ok=True)
    for name, c in (("cells_p", Jn["cellsP"]), ("cells_q", Jn["cellsQ"])):
        c.to_csv(big / f"{name}.csv.gz", index=False)
    timing = {}
    for f in sorted(Path(a.out).glob("*_done*.json")) + sorted(Path(a.out).glob("tables_timing_*.json")):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
            timing[f.name] = {kk: vv for kk, vv in d.items() if kk != "per_seed"}
        except Exception:
            pass
    out = {"verdict": V, "routes": routes, "route_verdicts": route_v, "observed": obs, "worlds": worlds,
           "candidate_rows_null": {"rows": len(cr_n), "by_filter": cr_n["filter"].value_counts().to_dict() if len(cr_n) else {},
                                   "by_setting": cr_n["setting"].value_counts().to_dict() if len(cr_n) else {},
                                   "by_band": cr_n["band"].value_counts().to_dict() if len(cr_n) and "band" in cr_n else {}},
           "pass_cells": {"null_mean": float(un["pass_cells"].mean()), "null_share_any": float((un["pass_cells"] > 0).mean()),
                          "null_cell_rate": float(un["pass_cells"].sum() / (CELLS_PER_SEED * len(un))),
                          "real": ur.set_index("gen")["pass_cells"].to_dict(), "real_mean": float(ur["pass_cells"].mean())},
           "margins": {"null_quantiles": {q: float(np.quantile(marg, q)) for q in (0.05, 0.5, 0.95, 0.99)},
                       "null_share_positive": float((marg > 0).mean()), "real": ur.set_index("gen")["margin"].to_dict(),
                       "real_rank_of_max": pct_rank(marg, obs["max_margin"])},
           "e7": {"candidate_cells_median_pct": e7_med, "rows": e7.to_dict("records")},
           "chance_flags": {"cells": len(flags), "changed": int(flags["changed"].sum()), "beyond_5": int(flags["beyond_5"].sum()),
                            "beyond_1000": int(flags["beyond_1000"].sum()),
                            "changed_rows": flags[flags["changed"]].to_dict("records")},
           "expectations": expect.to_dict("records"), "repro_pass": rp["pass"], "timing": timing, "trials": {"cumulative": TRIALS, "new": 0}}
    (agg / "verdict.json").write_text(json.dumps(out, ensure_ascii=False, indent=1, default=_jd), encoding="utf-8")
    shutil.copy2(repro_dir(a) / "repro.json", agg / "repro.json")
    for f in ("repro_tables.csv", "repro_base.csv"):
        shutil.copy2(repro_dir(a) / f, agg / f)
    if figures.available():
        write_figures(agg, un, ur, W, w_entries, w_max, obs, rates, e7, V, T)
    if a.docs:
        dd = Path(a.docs)
        dd.mkdir(parents=True, exist_ok=True)
        for f in agg.glob("*"):
            if f.is_file():
                shutil.copy2(f, dd / f.name)
    if a.docs_figures:
        fd = Path(a.docs_figures)
        fd.mkdir(parents=True, exist_ok=True)
        for f in (agg / "figures").glob("*.png"):
            shutil.copy2(f, fd / f.name)
    log(f"판정: {V['label']} · 후보 씨앗 {k}/{len(un)} · p̂ {V['p_hat']:.2%} ({V['p_lower']:.2%}–{V['p_upper']:.2%}) · "
        f"P̂ {V['P_hat']:.1%} ({V['P_lower']:.1%}–{V['P_upper']:.1%}) · 경로 {routes} · maxT p {worlds['maxt_p']:.3f}")
    return out


# --------------------------------------------------------------------------- figures

def fig_pass_cells(agg: Path, un: pd.DataFrame, ur: pd.DataFrame) -> Path:
    """(1) Passing cells (of 120, 20 bp) per random rule vs the 15 real entries."""
    plt = figures._plt()
    fig, ax = plt.subplots(figsize=(12, 5))
    v = un["pass_cells"].to_numpy()
    edges = np.arange(-0.5, max(int(v.max()), int(ur["pass_cells"].max())) + 1.5, 1.0)
    ax.hist(v, bins=edges, color="#bbbbbb", label=f"무작위 매수 규칙 1,000개 (평균 {v.mean():.1f}칸, 하나라도 통과 {np.mean(v > 0):.0%})")
    ymax = ax.get_ylim()[1]
    for i, r in ur.sort_values("pass_cells").reset_index(drop=True).iterrows():
        ax.plot([r["pass_cells"]], [ymax * (0.95 - 0.045 * i)], marker="v", color=figures.COLORS[1] if r["candidate"] else figures.COLORS[0], ms=6)
        ax.text(r["pass_cells"] + 0.25, ymax * (0.95 - 0.045 * i), r["gen"], fontsize=7, va="center")
    ax.plot([], [], marker="v", ls="", color=figures.COLORS[0], label="실제 매수 15개")
    ax.plot([], [], marker="v", ls="", color=figures.COLORS[1], label="그중 후보를 만든 매수")
    ax.set_xlabel("칸 기준(맞춘 수익이 두 기준선을 세 기간 모두 넘음 + 창 승률 + 거래 수)을 채운 칸 수 (120칸 중, 편도 20bp)")
    ax.set_ylabel("무작위 규칙 수")
    ax.legend(fontsize=8, frameon=False, loc="center right")
    ax.grid(alpha=0.3, axis="y")
    ax.set_title("규칙마다 통과한 칸 수: 무작위 매수 1,000개 vs 실제 매수 15개", fontsize=10.5)
    return figures._save(fig, agg / "figures" / "pass_cells.png")


def fig_margins(agg: Path, un: pd.DataFrame, ur: pd.DataFrame, w_max: np.ndarray, obs: dict, p_maxt: float) -> Path:
    """(2) Candidate margin per random rule vs the 15 real entries, and the 15-rule worlds' maxima."""
    plt = figures._plt()
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2))
    ax = axes[0]
    m = un["margin"].to_numpy(dtype=float) * 100
    ax.hist(m, bins=40, color="#bbbbbb", label=f"무작위 매수 규칙 1,000개 (양수 {np.mean(m > 0):.1%})")
    for _, r in ur.iterrows():
        ax.axvline(r["margin"] * 100, color=figures.COLORS[1] if r["candidate"] else figures.COLORS[0], lw=1, alpha=0.8)
    ax.plot([], [], color=figures.COLORS[0], label="실제 매수 15개")
    ax.plot([], [], color=figures.COLORS[1], label="후보를 만든 매수")
    ax.axvline(0, color="#333", lw=0.8, ls="--")
    ax.set_xlabel("후보 여유 (%p): 후보에 필요한 칸들의 (맞춘 수익 − 기준선) 최솟값, 가장 나은 계열 (20bp)")
    ax.set_ylabel("규칙 수")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    ax = axes[1]
    ax.hist(w_max * 100, bins=40, color="#bbbbbb", label=f"무작위 15개 세계 {len(w_max):,}개의 최댓값")
    ax.axvline(obs["max_margin"] * 100, color=figures.COLORS[1], lw=2, label=f"실제 15개의 최댓값 ({obs['max_margin_entry']}) · maxT p = {p_maxt:.3f}")
    ax.axvline(0, color="#333", lw=0.8, ls="--")
    ax.set_xlabel("15개 중 가장 큰 후보 여유 (%p)")
    ax.legend(fontsize=8, frameon=False)
    ax.grid(alpha=0.3)
    fig.suptitle("후보 여유(연속 척도): 무작위 매수 규칙과 실제 매수", fontsize=11)
    return figures._save(fig, agg / "figures" / "margins.png")


def fig_worlds(agg: Path, w_entries: np.ndarray, obs: dict, V: dict) -> Path:
    """(3) Candidate rules per world of 15 random rules vs the observed 1."""
    plt = figures._plt()
    fig, ax = plt.subplots(figsize=(10, 4.8))
    vals, cnt = np.unique(w_entries, return_counts=True)
    ax.bar(vals, cnt / len(w_entries) * 100, color="#999999", width=0.6)
    for v_, c_ in zip(vals, cnt):
        ax.text(v_, c_ / len(w_entries) * 100 + 0.8, f"{c_ / len(w_entries):.1%}", ha="center", fontsize=8)
    ax.axvline(obs["entries"], color=figures.COLORS[1], lw=2, ls="--", label=f"관측: 실제 15개 중 후보 규칙 {obs['entries']}개")
    ax.set_xticks(range(0, int(max(vals.max(), obs["entries"])) + 1))
    ax.set_xlabel("15개 세계에서 후보를 만든 규칙 수")
    ax.set_ylabel("세계 비율 (%)")
    ax.legend(fontsize=8.5, frameon=False)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(f"무작위 매수 15개 세계 {len(w_entries):,}개: 후보 하나 이상 {np.mean(w_entries >= 1):.1%} "
                 f"(식으로 P̂ = {V['P_hat']:.1%}, 95% 구간 {V['P_lower']:.1%}–{V['P_upper']:.1%}) — 판정 '{V['label']}'", fontsize=10)
    return figures._save(fig, agg / "figures" / "worlds.png")


def fig_rates(agg: Path, rates: pd.DataFrame) -> Path:
    """(4) Per-cell pass rate over the 1,000 random rules: setting x gate (x band), per exit."""
    plt = figures._plt()
    fig, axes = plt.subplots(1, 2, figsize=(17, 5.6))
    for ax, x in zip(axes, EXITS):
        rows_lab, cols_lab = [], []
        for st in P1_NAMES:
            cols_lab.append(st.split("_")[0])
        rows = [("U3000", f) for f in P_FILTERS] + [(b, f) for b in Q_BANDS for f in Q_FILTERS]
        mat = np.full((len(rows), len(P1_NAMES)), np.nan)
        for i, (b, f) in enumerate(rows):
            for j, st in enumerate(P1_NAMES):
                if b == "U3000":
                    s = rates[(rates["scope"] == "P") & (rates["setting"] == st) & (rates["exit"] == x) & (rates["filter"] == f)]
                else:
                    s = rates[(rates["scope"] == "Q") & (rates["setting"] == st) & (rates["exit"] == x) & (rates["filter"] == f) & (rates["band"] == b)]
                if len(s):
                    mat[i, j] = float(s["pass_rate"].iloc[0]) * 100
            rows_lab.append(f"{b} · {'국면 없음' if f == 'none' else f}")
        im = ax.imshow(mat, cmap="viridis", aspect="auto", vmin=0, vmax=max(5.0, float(np.nanmax(mat))))
        for i in range(mat.shape[0]):
            for j in range(mat.shape[1]):
                if np.isfinite(mat[i, j]):
                    ax.text(j, i, f"{mat[i, j]:.1f}", ha="center", va="center", fontsize=7, color="white" if mat[i, j] < np.nanmax(mat) * 0.6 else "black")
        ax.set_xticks(range(len(P1_NAMES)), [f"{c} · W2 · P1" for c in cols_lab], fontsize=8)
        ax.set_yticks(range(len(rows)), rows_lab, fontsize=7)
        ax.set_title(f"매도 {x}", fontsize=10)
        fig.colorbar(im, ax=ax, shrink=0.8, label="통과율 (%)")
    fig.suptitle("칸별 통과율: 무작위 매수 규칙 1,000개 중 칸 기준을 채운 비율 (편도 20bp)", fontsize=11)
    return figures._save(fig, agg / "figures" / "cell_rates.png")


def fig_e7(agg: Path, e7: pd.DataFrame, T: dict) -> Path:
    """(5) E7's candidate cells (N20 x R3 A1 x U1000 / U500 x X1 / H1): 2008-2023 matched return of the
    1,000 random rules at the same cell vs E7."""
    plt = figures._plt()
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.4), sharey=True)
    cn = pd.read_csv(Path(agg).parent / "agg_big" / "cells_q.csv.gz")
    cn = cn[cn["cost"] == pj.COST_BPS]
    p = "2008-2023"
    k = 0
    for b in Q_BANDS:
        for x in EXITS:
            ax = axes[k]
            k += 1
            v = cn[(cn["setting"] == CAND_SETTING) & (cn["band"] == b) & (cn["exit"] == x) & (cn["filter"] == E7_CELL[1])][f"dd_cagr_{p}"].to_numpy() * 100
            r = e7[(e7["setting"] == CAND_SETTING) & (e7["band"] == b) & (e7["exit"] == x) & (e7["period"] == p)]
            ax.hist(v, bins=30, color="#bbbbbb", label="무작위 매수 1,000개")
            if len(r):
                ax.axvline(float(r["e7"].iloc[0]) * 100, color=figures.COLORS[1], lw=2, label=f"E7 ({float(r['percentile'].iloc[0]):.1f} 백분위)")
            ax.axvline(T["benchQ"]["ma200"][f"dd_cagr_{p}"] * 100, color="#8e44ad", ls="--", lw=1.2, label="SPY 200일선 + 국채")
            ax.set_title(f"N20 × R3 A1 × {b} × {x}", fontsize=9.5)
            ax.set_xlabel(f"{p} 낙폭 15% 맞춤 수익 (%)")
            ax.legend(fontsize=7, frameon=False)
            ax.grid(alpha=0.3)
    axes[0].set_ylabel("무작위 규칙 수")
    fig.suptitle("E7 후보 칸의 2008–2023 맞춘 수익: 같은 칸의 무작위 매수 1,000개 분포와 E7 (15개를 훑은 뒤 고른 칸 — 서술)", fontsize=10.5)
    return figures._save(fig, agg / "figures" / "e7_cells.png")


def write_figures(agg: Path, un, ur, W, w_entries, w_max, obs, rates, e7, V, T) -> list[Path]:
    p = maxt_p(w_max, obs["max_margin"])
    return [fig_pass_cells(agg, un, ur), fig_margins(agg, un, ur, w_max, obs, p), fig_worlds(agg, w_entries, obs, V),
            fig_rates(agg, rates), fig_e7(agg, e7, T)]


# --------------------------------------------------------------------------- main

def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--stage", required=True, help="repro | tables | accounts | report, or several joined by commas (run in order)")
    ap.add_argument("--panel", required=True)
    ap.add_argument("--stage-s", required=True, help="드라이브 lab_crsp/stage_s_raw의 사본")
    ap.add_argument("--tmp", required=True, help="신호표 · 특징 행렬을 두는 임시 폴더")
    ap.add_argument("--out", required=True, help="결과 폴더 (드라이브 lab_crsp/stage_u_null)")
    ap.add_argument("--seeds", default=f"{SEEDS[0]}-{SEEDS[-1]}")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--table-workers", type=int, default=6)
    ap.add_argument("--feature-workers", type=int, default=6)
    ap.add_argument("--docs", default=None, help="report: 집계를 복사할 저장소 폴더")
    ap.add_argument("--docs-figures", default=None, help="report: 그림을 복사할 저장소 폴더")
    a = ap.parse_args(argv)
    run = {"repro": stage_repro, "tables": stage_tables, "accounts": stage_accounts, "report": stage_report}
    stages = a.stage.split(",")
    if any(s not in run for s in stages):
        ap.error(f"--stage: {a.stage}")
    for s in stages:
        run[s](a)
    if "accounts" in stages:
        (Path(a.out) / f"RUN_DONE_{a.seeds}").write_text(time.strftime("%Y-%m-%d %H:%M:%S") + f" {a.stage} {a.seeds}\n", encoding="utf-8")


if __name__ == "__main__":
    main(sys.argv[1:])
