"""A41: principle-first map of 13 products, backtests used only to check (allocation DECISIONS A41).

    .venv/Scripts/python.exe scripts/aa_redesign_study.py --quick --out %TEMP%\\a41_quick           # pipeline check
    .venv/Scripts/python.exe scripts/aa_redesign_study.py --stage 1 --out docs/research/allocation/2026-10-28_a41
    ... --stage 2, --stage 3, --stage 4 (each reads the previous stage's work files)

Design fixed before results: docs/research/allocation/2026-10-28_a41_design.md (lab/aa_landscape.py holds the pieces).

  stage 1  map:        2^20 sampled allocations, metrics on 2000-09 – 2024-12 (KRW, real prices after listing)
  stage 2  landscape:  Scheffé surrogate, Sobol indices (real simulations), exchange rates, leave-one-out, sets
  stage 3  choose:     ±5 %p neighbours in rank order, flat regions (가)(나), centres, half cross-check, PBO,
                       rule-based comparators
  stage 4  robustness: engine runs for every centre / rule / comparator: ±5 / ±10 neighbours, leave-one-out,
                       528 rebalancing versions, 184 monthly starts, 172 ten-year windows

Big work files go to <out>/_work (not committed); the tables, figures and summary.json are the record. The hold-out
(2025–) is not read here; it is a separate step after the user has seen the candidates.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab import aa_landscape as LS  # noqa: E402
from lab import aa_local as loc  # noqa: E402
from lab import aa_real  # noqa: E402
from lab import aa_search as srch  # noqa: E402
from lab.aa_assets import CASH, PRESETS, PROXY_SYMBOLS, PRODUCTS  # noqa: E402
from lab.aa_engine import AllocConfig, run_allocation  # noqa: E402
from lab.aa_korea import krw_panel, load_bok_rate, load_usdkrw  # noqa: E402
from lab.aa_score import CRISIS_WINDOWS  # noqa: E402
from lab.data import download_panel  # noqa: E402

DATA_END = "2026-10-27"
COST_BPS = 10.0
FULL = dict(n=2 ** 20, chunk=2048, saltelli=2 ** 14, exch_points=3000, loo_points=50_000, pbo_points=20_000,
            out_sample=20_000, nb_max=200_000)
QUICK = dict(n=2 ** 12, chunk=512, saltelli=2 ** 8, exch_points=40, loo_points=400, pbo_points=600,
             out_sample=500, nb_max=400)
WINDOWS = {"dotcom": LS.DOTCOM, "2008": CRISIS_WINDOWS["금융위기 2007–09"], "2011": CRISIS_WINDOWS["유럽 재정위기 2011"],
           "2018": CRISIS_WINDOWS["2018년 4분기"], "2020": CRISIS_WINDOWS["코로나 2020"], "2022": CRISIS_WINDOWS["금리 급등 2022"]}
COMPARE = {"B2": "gs_b2", "B4": "gs_b4", "안정형 40": "stable40"}
SKELETONS = {"뼈대 30·30·40": {"379810": 30, "KRXGOLD": 30, "459580": 40},
             "뼈대 35·30·35": {"379810": 35, "KRXGOLD": 30, "459580": 35}}
SPX = {"S&P500 보유": {"379800": 100}}


# --------------------------------------------------------------------------- data

def load_panel():
    """KRW panel 2000-09 – , proxies carried back (A37 extend_history), real prices after listing with A41 chains."""
    usd, failed = download_panel(sorted(set(PROXY_SYMBOLS) | {"IEF", "TLT", "SHY"}), end=DATA_END)
    subs, failed_s = download_panel(sorted({c for v in loc.SUBSTITUTES.values() for c in v}), end=DATA_END)
    if failed or failed_s:
        print("받지 못한 종목:", failed + failed_s)
    ext, table = loc.extend_history(usd, subs, LS.START)
    krw, unavailable = krw_panel(ext, load_usdkrw(), load_bok_rate(), products=PRODUCTS)
    need = set(LS.UNIVERSE) | {c for k in COMPARE.values() for c in PRESETS[k].weights if c != CASH}
    missing = sorted(need & set(unavailable))
    if missing:
        raise SystemExit(f"대용 데이터가 없는 상품: {missing}")
    krw, info, msg = aa_real.apply_source(krw, "real", chains=aa_real.A41_CHAINS)
    if info is None:
        raise SystemExit(msg)
    print(msg)
    first = krw.close[sorted(need)].apply(lambda s: s.first_valid_index())
    late = first[first > pd.Timestamp(LS.START)]
    if len(late):
        raise SystemExit(f"{LS.START} 이후에 데이터가 시작되는 상품: {late.to_dict()}")
    return krw, table, info


def real_share(info: dict, dates: pd.DatetimeIndex) -> pd.DataFrame:
    """Per universe product: share of research-window days on real prices (any link of its chain)."""
    rows = []
    for c in LS.UNIVERSE:
        chain = info.get(c, {}).get("chain", [])
        t0 = min((t for _, t in chain), default=None)
        share = float((dates >= t0).mean()) if t0 is not None else 0.0
        rows.append({"코드": c, "상품": LS.SHORT[c], "역할": LS.ROLE[c], "실제 가격 시작": None if t0 is None else str(t0.date()),
                     "이음": " ← ".join(code for code, _ in reversed(chain)) or "모형만", "연구 구간 중 실제 가격 비율": round(share, 3)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- simulation in parallel

_SIM = None
_LAY = None


def _init(sim, lay):
    global _SIM, _LAY
    _SIM, _LAY = sim, lay


def _chunk(W: np.ndarray) -> pd.DataFrame:
    P, turn = _SIM.run(LS.to_fraction(W))
    return LS.path_metrics(P, turn, _LAY)


def evaluate(sim, lay, W: np.ndarray, chunk: int, workers: int, label: str) -> pd.DataFrame:
    parts = [W[a:a + chunk] for a in range(0, len(W), chunk)]
    out, t0 = [], time.time()
    if workers <= 1 or len(parts) == 1:
        _init(sim, lay)
        for k, p in enumerate(parts):
            out.append(_chunk(p))
            if k % 20 == 0:
                print(f"  {label}: {min((k + 1) * chunk, len(W)):,}/{len(W):,} ({time.time() - t0:.0f}s)", flush=True)
    else:
        with ProcessPoolExecutor(workers, initializer=_init, initargs=(sim, lay)) as ex:
            for k, df in enumerate(ex.map(_chunk, parts)):
                out.append(df)
                if k % 50 == 0:
                    print(f"  {label}: {min((k + 1) * chunk, len(W)):,}/{len(W):,} ({time.time() - t0:.0f}s)", flush=True)
    print(f"  {label}: {len(W):,} 끝 ({time.time() - t0:.0f}s)", flush=True)
    return pd.concat(out, ignore_index=True)


def engine_check(krw, sim, W: np.ndarray) -> float:
    P, turn = sim.run(LS.to_fraction(W))
    worst = 0.0
    for k, w in enumerate(W):
        wt = {LS.UNIVERSE[i]: float(x) for i, x in enumerate(w) if x > 0}
        res = run_allocation(krw, AllocConfig(f"c{k}", wt, LS.START, LS.END, rebalance="Q", cost_bps=COST_BPS))
        worst = max(worst, float(np.max(np.abs(P[k] / res.equity.to_numpy() - 1.0))),
                    abs(turn[k] - res.turnover) / max(res.turnover, 1e-12))
    print(f"엔진 대조: {len(W)}개, 최대 상대 오차 {worst:.2e}")
    if worst > 1e-6:
        raise SystemExit("시뮬레이터가 엔진과 다릅니다. 멈춥니다.")
    return worst


def wstr(w) -> str:
    return ";".join(f"{LS.UNIVERSE[i]}:{int(x)}" for i, x in enumerate(w) if x > 0)


def wparse(s: str) -> np.ndarray:
    v = np.zeros(LS.D, np.int64)
    for part in str(s).split(";"):
        if part:
            c, x = part.split(":")
            v[LS.COL[c]] = int(x)
    return v


def named_rows(krw, lay) -> pd.DataFrame:
    """Comparators through the engine on the same window: B2, B4, 안정형 40, two skeletons, S&P500."""
    named = {**{k: PRESETS[v].weights for k, v in COMPARE.items()}, **SKELETONS, **SPX}
    rows = []
    for name, w in named.items():
        res = run_allocation(krw, AllocConfig(name, dict(w), LS.START, LS.END, rebalance="Q", cost_bps=COST_BPS))
        if len(res.equity) != len(lay.dates):
            raise SystemExit(f"{name}: 기간이 다릅니다")
        m = LS.path_metrics(res.equity.to_numpy()[None, :], np.array([res.turnover]), lay).iloc[0].to_dict()
        rows.append({"이름": name, "비중": ";".join(f"{c}:{v:g}" for c, v in w.items()), **m})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- stages

def stage1(P, out: Path, work: Path, workers: int) -> dict:
    krw, table, info = load_panel()
    table.to_csv(out / "proxies.csv", index=False)
    sim = srch.StaticSim(krw, LS.UNIVERSE, LS.START, LS.END, rebalance="Q", cost_bps=COST_BPS)
    lay = LS.layout(sim.dates, WINDOWS)
    real_share(info, sim.dates).to_csv(out / "real_share.csv", index=False)
    U, W = LS.sample(P["n"])
    check = np.vstack([W[:6], LS.weights_from_u(LS.rqmc(4, 2 * LS.D, LS.SEED + 7))])
    err = engine_check(krw, sim, check)
    M = evaluate(sim, lay, W, P["chunk"], workers, "① 지도")
    M["zone"] = LS.zone(M["mdd"])
    np.save(work / "W.npy", W)
    np.save(work / "U.npy", U.astype(np.float32))
    M.to_parquet(work / "metrics.parquet") if _has_parquet() else M.to_pickle(work / "metrics.pkl")
    named = named_rows(krw, lay)
    named["zone"] = LS.zone(named["mdd"])
    named.to_csv(out / "named.csv", index=False)
    # committed map: every 안 / 경계 point and a fixed sample of 밖
    keep = np.flatnonzero(M["zone"].to_numpy() != "밖")
    outside = np.flatnonzero(M["zone"].to_numpy() == "밖")
    rng = np.random.default_rng(LS.SEED)
    keep = np.sort(np.concatenate([keep, rng.choice(outside, size=min(P["out_sample"], len(outside)), replace=False)]))
    mp = M.iloc[keep].copy()
    mp.insert(0, "점", keep)
    mp.insert(1, "비중", [wstr(W[i]) for i in keep])
    float_cols = [c for c in mp.columns if mp[c].dtype.kind == "f"]
    mp[float_cols] = mp[float_cols].round(5)
    mp.to_csv(out / "map.csv.gz", index=False)
    counts = M["zone"].value_counts().to_dict()
    return {"engine_check_max_rel": err, "n": int(len(W)), "zones": {k: int(v) for k, v in counts.items()},
            "dates": [str(sim.dates[0].date()), str(sim.dates[-1].date())], "quarter_starts": int(len(lay.starts)),
            "ten_year_windows": int(len(lay.win_starts))}


def _has_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except ImportError:
        return False


# The work files are written by stage 1 of this script on this computer; the pickle fallback (no pyarrow) only
# ever reads that file back.
def load_work(work: Path) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    W = np.load(work / "W.npy")
    U = np.load(work / "U.npy")
    M = pd.read_parquet(work / "metrics.parquet") if (work / "metrics.parquet").exists() else pd.read_pickle(work / "metrics.pkl")
    return W, U, M


def _sim(krw):
    sim = srch.StaticSim(krw, LS.UNIVERSE, LS.START, LS.END, rebalance="Q", cost_bps=COST_BPS)
    return sim, LS.layout(sim.dates, WINDOWS)


OUTPUTS = ["cagr", "mdd", "rank", "dd_dotcom"]


def stage2(P, out: Path, work: Path, workers: int) -> dict:
    W, U, M = load_work(work)
    krw, _, _ = load_panel()
    sim, lay = _sim(krw)
    res: dict = {}
    # surrogate (gradient map only)
    sur = []
    grads = []
    inside = np.flatnonzero(M["zone"].to_numpy() == "안")
    for y in ("cagr", "mdd", "rank"):
        fit = LS.scheffe_fit(W, M[y].to_numpy())
        sur.append({"출력": y, "R2_적합": fit["r2_fit"], "R2_검증": fit["r2_test"]})
        if len(inside):
            g = LS.scheffe_gradient(fit["beta"], W[inside[:20000]]).mean(axis=0)
            grads.append(pd.Series(g, index=LS.UNIVERSE, name=y))
    pd.DataFrame(sur).to_csv(out / "surrogate_r2.csv", index=False)
    if grads:
        gd = pd.concat(grads, axis=1).mul(100.0)  # percent points per 1 %p
        gd.insert(0, "상품", [LS.SHORT[c] for c in gd.index])
        gd.to_csv(out / "surrogate_gradient.csv", index_label="코드")
    res["surrogate"] = sur
    # Sobol indices on real simulations
    A, B, mixed = LS.saltelli(P["saltelli"])
    allW = np.vstack([LS.weights_from_u(X) for X in (A, B, *mixed)])
    fm = evaluate(sim, lay, allW, P["chunk"], workers, "② Sobol")
    N = len(A)
    tabs = []
    for y in OUTPUTS:
        f = fm[y].to_numpy()
        t = LS.sobol_indices(f[:N], f[N:2 * N], [f[(2 + i) * N:(3 + i) * N] for i in range(LS.D)])
        t.insert(0, "출력", y)
        t.insert(2, "이름", [LS.SHORT[c] for c in t["상품"]])
        tabs.append(t)
    pd.concat(tabs).to_csv(out / "sobol.csv", index=False)
    # exchange rates around 3,000 points inside the line
    rng = np.random.default_rng(LS.SEED + 2)
    pts = rng.choice(inside, size=min(P["exch_points"], len(inside)), replace=False) if len(inside) else np.array([], int)
    exch_rows = []
    for step in (1, 5):
        Ws, meta = [], []
        for p in pts:
            V, pairs = LS.exchanges(W[p], step)
            Ws.append(V)
            meta += [(p, i, j) for i, j in pairs]
        if not meta:
            continue
        mv = evaluate(sim, lay, np.vstack(Ws), P["chunk"], workers, f"② 교환 {step}%p")
        meta = np.array(meta)
        base = M.iloc[meta[:, 0]]
        d = pd.DataFrame({"from": meta[:, 1], "to": meta[:, 2], "dcagr": mv["cagr"].to_numpy() - base["cagr"].to_numpy(),
                          "dmdd": mv["mdd"].to_numpy() - base["mdd"].to_numpy(), "drank": mv["rank"].to_numpy() - base["rank"].to_numpy(),
                          "out": (mv["mdd"].to_numpy() < LS.LINE)})
        g = d.groupby(["from", "to"]).agg(점=("dcagr", "size"), CAGR변화_중앙=("dcagr", "median"), MDD변화_중앙=("dmdd", "median"),
                                         순위값변화_중앙=("drank", "median"), 선밖으로=("out", "mean")).reset_index()
        g.insert(0, "단계", f"{step}%p")
        g["줄이는 상품"] = [LS.SHORT[LS.UNIVERSE[i]] for i in g["from"]]
        g["늘리는 상품"] = [LS.SHORT[LS.UNIVERSE[j]] for j in g["to"]]
        exch_rows.append(g.drop(columns=["from", "to"]))
    if exch_rows:
        pd.concat(exch_rows).to_csv(out / "exchange.csv", index=False)
    # leave one out
    pts = rng.choice(len(W), size=min(P["loo_points"], len(W)), replace=False)
    Ws, meta = [], []
    for p in pts:
        for i, v in LS.drop_one(W[p], "pro"):
            Ws.append(v)
            meta.append((p, i))
    lo = evaluate(sim, lay, np.vstack(Ws), P["chunk"], workers, "② 하나 빼기")
    meta = np.array(meta)
    base = M.iloc[meta[:, 0]]
    z0, z1 = base["zone"].to_numpy(), LS.zone(lo["mdd"])
    d = pd.DataFrame({"i": meta[:, 1], "dcagr": lo["cagr"].to_numpy() - base["cagr"].to_numpy(),
                      "dmdd": lo["mdd"].to_numpy() - base["mdd"].to_numpy(), "in_to_out": (z0 == "안") & (z1 != "안"),
                      "out_to_in": (z0 != "안") & (z1 == "안"), "was_in": z0 == "안"})
    g = d.groupby("i").agg(점=("dcagr", "size"), CAGR변화_중앙=("dcagr", "median"), CAGR변화_하위10=("dcagr", lambda s: s.quantile(0.1)),
                           MDD변화_중앙=("dmdd", "median"), MDD변화_하위10=("dmdd", lambda s: s.quantile(0.1)),
                           안에서_밖으로=("in_to_out", "sum"), 밖에서_안으로=("out_to_in", "sum"), 원래_안=("was_in", "sum")).reset_index()
    g.insert(0, "상품", [LS.SHORT[LS.UNIVERSE[i]] for i in g["i"]])
    g["뺐을 때 안→밖 비율"] = g["안에서_밖으로"] / g["원래_안"].clip(lower=1)
    g.drop(columns=["i"]).to_csv(out / "leave_one_out.csv", index=False)
    # product sets
    key = (W > 0).astype(np.int64) @ (1 << np.arange(LS.D))
    sets = pd.DataFrame({"key": key, "zone": M["zone"], "cagr": M["cagr"], "rank": M["rank"]})
    gs = sets.groupby("key").agg(점=("cagr", "size"), 안=("zone", lambda s: (s == "안").mean()), 경계=("zone", lambda s: (s == "경계").mean()),
                                 CAGR_중앙=("cagr", "median"), CAGR_상위10=("cagr", lambda s: s.quantile(0.9)), 순위값_최고=("rank", "max")).reset_index()
    gs.insert(1, "상품 집합", [" + ".join(LS.SHORT[LS.UNIVERSE[i]] for i in range(LS.D) if k >> i & 1) for k in gs["key"]])
    gs.insert(2, "상품 수", [bin(int(k)).count("1") for k in gs["key"]])
    gs.drop(columns=["key"]).sort_values("안", ascending=False).to_csv(out / "sets.csv.gz", index=False)
    # include / exclude by product (pairs of sets that differ only in that product)
    inc_rows = []
    gk = gs.set_index("key")
    for i in range(LS.D):
        bit = 1 << i
        with_i = gk[(gk.index & bit) > 0]
        pair = with_i.index - bit
        both = [k for k in pair if k in gk.index and k > 0]
        if not both:
            continue
        a = gk.loc[[k + bit for k in both]]
        b = gk.loc[both]
        inc_rows.append({"상품": LS.SHORT[LS.UNIVERSE[i]], "짝 수": len(both), "넣었을 때 안 비율 중앙": float(a["안"].median()),
                         "뺐을 때 안 비율 중앙": float(b["안"].median()),
                         "넣으면 안 비율 변화 중앙": float(np.median(a["안"].to_numpy() - b["안"].to_numpy())),
                         "넣으면 CAGR 중앙 변화": float(np.median(a["CAGR_중앙"].to_numpy() - b["CAGR_중앙"].to_numpy()))})
    pd.DataFrame(inc_rows).to_csv(out / "include_exclude.csv", index=False)
    return res


def stage3(P, out: Path, work: Path, workers: int) -> dict:
    W, U, M = load_work(work)
    krw, _, _ = load_panel()
    sim, lay = _sim(krw)
    nb_share = pd.Series(np.nan, index=M.index)
    nb_worst = pd.Series(np.nan, index=M.index)

    def neighbour_pass(cols: dict, near: bool) -> pd.Series:
        """Compute ±5 %p neighbours in rank order (by `rank` col) until the threshold is safe."""
        lo, hi = (LS.NEAR, LS.LINE) if near else (LS.LINE, 1.0)
        win_line = LS.NEAR if near else LS.LINE
        line = LS.NEAR if near else LS.LINE
        mdd, win, rk = M[cols["mdd"]], M[cols["win"]], M[cols["rank"]]
        elig = M.index[(mdd >= lo) & (mdd < hi) & (win >= win_line)]
        order = elig[np.argsort(-rk.loc[elig].to_numpy())]
        share = pd.Series(np.nan, index=M.index)
        best = None
        done = 0
        batch = 200
        for a in range(0, min(len(order), P["nb_max"]), batch):
            idx = order[a:a + batch]
            Ws, owner = [], []
            for p in idx:
                V = LS.neighbours(W[p], 5)
                Ws.append(V)
                owner += [p] * len(V)
            mv = evaluate(sim, lay, np.vstack(Ws), P["chunk"], workers, f"③ 이웃 {cols['tag']}{'(나)' if near else '(가)'}") \
                if cols["tag"] == "전체" else _half_eval(sim, lay, np.vstack(Ws), P, workers, cols)
            ok = (mv[cols["mdd"]].to_numpy() >= line)
            s = pd.Series(ok).groupby(np.array(owner)).mean()
            share.loc[s.index] = s.to_numpy()
            done += len(idx)
            passed = share.loc[order[:done]].dropna()
            passed = passed[passed >= LS.NB_SHARE]
            if len(passed):
                best = rk.loc[passed.index].max() if best is None else max(best, rk.loc[passed.index].max())
            if best is not None and rk.loc[idx[-1]] < best - LS.FLAT_PP:
                break
        return share

    tags = {"전체": {"mdd": "mdd", "win": "win_worst", "rank": "rank", "tag": "전체"},
            "앞 절반": {"mdd": "mdd_h1", "win": "win_worst_h1", "rank": "rank_h1", "tag": "앞 절반"},
            "뒤 절반": {"mdd": "mdd_h2", "win": "win_worst_h2", "rank": "rank_h2", "tag": "뒤 절반"}}
    cents, regions = [], {}
    for tag, cols in tags.items():
        for near in (False, True):
            share = neighbour_pass(cols, near)
            df = M.rename(columns={cols["mdd"]: "_m", cols["win"]: "_w", cols["rank"]: "_r"})[["_m", "_w", "_r"]]
            df.columns = ["mdd", "win_worst", "rank"]
            reg = LS.flat_region(df, share, near=near)
            label = f"{tag} {'(나)' if near else '(가)'}"
            if near and len(reg) and tag in regions and len(regions[tag]) and reg["rank"].max() <= regions[tag]["rank"].max():
                reg = reg.iloc[0:0]  # (나) only when it beats (가)
            regions[tag if not near else tag + "_near"] = reg
            for k, c in enumerate(LS.centres(W, reg)):
                cents.append({"고른 구간": tag, "묶음": "(나)" if near else "(가)", "번호": k + 1, "영역 점 수": int(len(reg)),
                              "주요 상품": " + ".join(LS.SHORT[x] for x in c["묶음"].split("+")), "묶음 점 수": c["점 수"], "비중": wstr(c["비중"])})
            if tag == "전체":
                nb_share.update(share)
    C = pd.DataFrame(cents, columns=["고른 구간", "묶음", "번호", "영역 점 수", "주요 상품", "묶음 점 수", "비중"])
    # re-measure centres on every window
    if len(C):
        Wc = np.vstack([wparse(s) for s in C["비중"]])
        mc = evaluate(sim, lay, Wc, P["chunk"], 1, "③ 중심")
        C = pd.concat([C.reset_index(drop=True), mc.reset_index(drop=True)], axis=1)
        C["zone"] = LS.zone(C["mdd"])
    C.to_csv(out / "centres.csv", index=False)
    # PBO (R1 = A31 rule, R2 = flat-region mean, nearest point)
    rng = np.random.default_rng(LS.SEED)
    pick = rng.choice(len(W), size=min(P["pbo_points"], len(W)), replace=False)
    paths, _ = sim.run(LS.to_fraction(W[pick]))
    R = srch.monthly_returns(paths, sim.dates)
    pbo1 = {k: v for k, v in srch.pbo_selection(R, mdd_line=LS.LINE, blocks=10).items() if k != "picked"}
    pbo2 = _pbo_r2(R, W[pick])
    # rule-based comparators
    rules = []
    sets = {"13개 전부": LS.UNIVERSE}
    for _, r in C.iterrows():
        codes = [c for c in LS.UNIVERSE if wparse(r["비중"])[LS.COL[c]] > 0]
        sets[f"{r['고른 구간']} {r['묶음']} {r['번호']}"] = codes
    for sname, codes in sets.items():
        for rule in ("equal", "invvol", "riskcap"):
            eq, turn = LS.run_rule(krw, rule, codes, LS.START, LS.END, COST_BPS)
            m = LS.path_metrics(eq.to_numpy()[None, :], np.array([turn]), lay).iloc[0].to_dict()
            rules.append({"규칙": rule, "상품 집합": sname, "상품": ";".join(codes), **m})
    RL = pd.DataFrame(rules)
    RL["zone"] = LS.zone(RL["mdd"])
    RL.to_csv(out / "rules.csv", index=False)
    nb_share.to_frame("nb5_share").dropna().to_csv(work / "nb_share.csv")
    return {"centres": int(len(C)), "regions": {k: int(len(v)) for k, v in regions.items()},
            "pbo_r1": pbo1.get("pbo"), "pbo_r2": pbo2.get("pbo"), "pbo_r1_detail": pbo1, "pbo_r2_detail": pbo2}


def _half_eval(sim, lay, W, P, workers, cols):
    return evaluate(sim, lay, W, P["chunk"], workers, f"③ 이웃 {cols['tag']}")


def _pbo_r2(R: np.ndarray, W: np.ndarray, blocks: int = 10) -> dict:
    """CSCV with rule R2: in the training half, the points within −15 % whose CAGR is within 0.3 %p of the best;
    pick the point nearest (L1) to their mean weights; PBO = share of splits where it ranks below the median OOS."""
    from itertools import combinations
    T = R.shape[1]
    edges = np.linspace(0, T, blocks + 1).astype(int)
    parts = [np.arange(edges[k], edges[k + 1]) for k in range(blocks)]
    lam, picks_mdd = [], []
    for tr in combinations(range(blocks), blocks // 2):
        te = [k for k in range(blocks) if k not in tr]
        a, b = np.concatenate([parts[k] for k in tr]), np.concatenate([parts[k] for k in te])

        def stats(cols):
            g = np.cumprod(1 + R[:, cols], axis=1)
            cagr = g[:, -1] ** (12 / len(cols)) - 1
            mdd = (g / np.maximum.accumulate(g, axis=1) - 1).min(axis=1)
            return cagr, mdd
        ca, ma = stats(a)
        cb, mb = stats(b)
        ok = np.flatnonzero(ma >= LS.LINE)
        if len(ok) == 0:
            continue
        flat = ok[ca[ok] >= ca[ok].max() - LS.FLAT_PP]
        centre = W[flat].mean(axis=0)
        p = int(np.argmin(np.abs(W - centre).sum(axis=1)))
        okb = np.flatnonzero(mb >= LS.LINE)
        rank = (cb[okb] < cb[p]).mean() if len(okb) else 0.0
        lam.append(rank)
        picks_mdd.append(mb[p])
    lam = np.array(lam)
    return {"pbo": float((lam < 0.5).mean()) if len(lam) else None, "splits": int(len(lam)),
            "pick_oos_mdd_median": float(np.median(picks_mdd)) if picks_mdd else None,
            "pick_oos_within_share": float((np.array(picks_mdd) >= LS.LINE).mean()) if picks_mdd else None}


# --------------------------------------------------------------------------- stage 4 (engine)

REBAL = ([("M", 0)] + [("Q", p) for p in range(3)] + [("S", p) for p in range(6)] + [("A", p) for p in range(12)])
OFFSETS = (0, 5, 10, 15, 20)
BANDS = (0.0, 3.0, 5.0, 10.0)


def _metrics_of(eq: pd.Series, turnover: float) -> dict:
    v = eq.to_numpy()
    T = len(v)
    peak = np.maximum.accumulate(v)
    return {"cagr": (v[-1] / v[0]) ** (252 / max(T - 1, 1)) - 1, "mdd": float((v / peak - 1).min()), "turnover": turnover}


def _robust_one(args) -> list[dict]:
    krw, name, w, kind = args
    out = []

    def run(cfg_kw, tag, variant):
        base = dict(start=LS.START, end=LS.END, rebalance="Q", cost_bps=COST_BPS)
        base.update(cfg_kw)
        try:
            r = run_allocation(krw, AllocConfig(name, dict(w), **base))
        except ValueError as exc:
            out.append({"대상": name, "시험": tag, "변형": variant, "오류": str(exc)})
            return
        out.append({"대상": name, "시험": tag, "변형": variant, **_metrics_of(r.equity, r.turnover),
                    "시작": str(r.equity.index[0].date()), "끝": str(r.equity.index[-1].date())})

    run({}, "기준", "분기·10bp")
    vec = np.array([w.get(c, 0) for c in LS.UNIVERSE], dtype=np.int64)
    if kind == "grid" and vec.sum() == 100:
        for step in (5, 10):
            for v in LS.neighbours(vec, step):
                wt = {LS.UNIVERSE[i]: float(x) for i, x in enumerate(v) if x > 0}
                try:
                    r = run_allocation(krw, AllocConfig(name, wt, LS.START, LS.END, rebalance="Q", cost_bps=COST_BPS))
                    out.append({"대상": name, "시험": f"이웃 ±{step}", "변형": wstr(v), **_metrics_of(r.equity, r.turnover)})
                except ValueError as exc:
                    out.append({"대상": name, "시험": f"이웃 ±{step}", "변형": wstr(v), "오류": str(exc)})
        for mode in ("pro", "cd"):
            for i, v in LS.drop_one(vec, mode):
                wt = {LS.UNIVERSE[k]: float(x) for k, x in enumerate(v) if x > 0}
                r = run_allocation(krw, AllocConfig(name, wt, LS.START, LS.END, rebalance="Q", cost_bps=COST_BPS))
                out.append({"대상": name, "시험": f"하나 빼기 ({'비례' if mode == 'pro' else 'CD로'})", "변형": LS.SHORT[LS.UNIVERSE[i]],
                            **_metrics_of(r.equity, r.turnover)})
    for freq, ph in REBAL:
        for off in OFFSETS:
            for band in BANDS:
                kw = {"rebalance": freq, "rebalance_phase": ph, "rebalance_offset": off, "band": band}  # 22 × 5 × 4 = 440
                run(kw, "리밸런싱", f"{freq}{ph}·+{off}일·밴드{band:g}·10bp")
                if off == 0:
                    run({**kw, "cost_bps": 20.0}, "리밸런싱", f"{freq}{ph}·+{off}일·밴드{band:g}·20bp")
    months = pd.date_range("2000-09-01", "2015-12-01", freq="MS")
    for m in months:
        run({"start": str(m.date())}, "시작 시점", str(m.date()))
    for m in pd.date_range("2000-09-01", "2014-12-01", freq="MS"):
        end = (m + pd.DateOffset(years=10) - pd.Timedelta(days=1)).date()
        run({"start": str(m.date()), "end": str(end)}, "10년 창", str(m.date()))
    return out


def stage4(P, out: Path, work: Path, workers: int) -> dict:
    krw, _, _ = load_panel()
    C = pd.read_csv(out / "centres.csv") if (out / "centres.csv").exists() else pd.DataFrame(columns=["비중"])
    RL = pd.read_csv(out / "rules.csv")
    jobs = []
    for _, r in C.iterrows():
        v = wparse(r["비중"])
        jobs.append((krw, f"중심 {r['고른 구간']} {r['묶음']} {r['번호']}", {LS.UNIVERSE[i]: float(x) for i, x in enumerate(v) if x > 0}, "grid"))
    named = {**{k: PRESETS[v].weights for k, v in COMPARE.items()}, **SKELETONS, **SPX}
    for k, w in named.items():
        jobs.append((krw, k, dict(w), "named"))
    if P is QUICK:
        jobs = jobs[:3]
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max(1, workers)) as ex:
        for k, res in enumerate(ex.map(_robust_one, jobs)):
            rows += res
            print(f"  ④ {k + 1}/{len(jobs)} ({time.time() - t0:.0f}s)", flush=True)
    R = pd.DataFrame(rows)
    R.to_csv(out / "robustness.csv.gz", index=False)
    verdicts = []
    for name, g in R.groupby("대상", sort=False):
        base = g[g["시험"] == "기준"]
        nb5 = g[g["시험"] == "이웃 ±5"]
        reb = g[g["시험"] == "리밸런싱"]
        st = g[g["시험"] == "시작 시점"]
        win = g[g["시험"] == "10년 창"]
        b_mdd = float(base["mdd"].iloc[0]) if len(base) else np.nan
        row = {"대상": name, "cagr": float(base["cagr"].iloc[0]) if len(base) else np.nan, "mdd": b_mdd,
               "1 기준선": bool(b_mdd >= LS.LINE),
               "2 이웃 ±5 안 비율": float((nb5["mdd"] >= LS.LINE).mean()) if len(nb5) else np.nan,
               "3 리밸런싱 안 비율": float((reb["mdd"] >= LS.LINE).mean()) if len(reb) else np.nan,
               "리밸런싱 최악 MDD": float(reb["mdd"].min()) if len(reb) else np.nan,
               "4 10년 창 모두 안": bool((win["mdd"] >= LS.LINE).all()) if len(win) else False,
               "10년 창 최악 MDD": float(win["mdd"].min()) if len(win) else np.nan,
               "순위값 (매달 시작 CAGR 하위 10%)": float(st["cagr"].quantile(0.1)) if len(st) else np.nan,
               "시작 시점 CAGR 중앙": float(st["cagr"].median()) if len(st) else np.nan,
               "10년 CAGR 폭": float(win["cagr"].max() - win["cagr"].min()) if len(win) else np.nan}
        row["2 통과"] = bool(row["2 이웃 ±5 안 비율"] >= LS.NB_SHARE) if len(nb5) else None
        row["3 통과"] = bool(row["3 리밸런싱 안 비율"] >= 0.90) if len(reb) else None
        row["모두 통과"] = bool(row["1 기준선"] and row["2 통과"] and row["3 통과"] and row["4 10년 창 모두 안"])
        verdicts.append(row)
    V = pd.DataFrame(verdicts)
    V.to_csv(out / "verdicts.csv", index=False)
    return {"targets": int(len(V)), "all_pass": int(V["모두 통과"].sum())}


def _jsonable(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (pd.Timestamp,)):
        return str(o.date())
    return str(o)


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", default="all", choices=("1", "2", "3", "4", "all"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--workers", type=int, default=max(1, min(12, (os.cpu_count() or 2) - 1)))
    args = ap.parse_args(argv)
    P = QUICK if args.quick else FULL
    out = Path(args.out)
    work = out / "_work"
    work.mkdir(parents=True, exist_ok=True)
    summ_path = out / "summary.json"
    summary = json.loads(summ_path.read_text(encoding="utf-8")) if summ_path.exists() else {}
    summary["params"] = {"quick": args.quick, **{k: v for k, v in P.items()}, "start": LS.START, "end": LS.END,
                         "seed": LS.SEED, "cost_bps": COST_BPS, "universe": LS.UNIVERSE}
    stages = ["1", "2", "3", "4"] if args.stage == "all" else [args.stage]
    for s in stages:
        t0 = time.time()
        print(f"=== {s}단계", flush=True)
        res = {"1": stage1, "2": stage2, "3": stage3, "4": stage4}[s](P, out, work, args.workers)
        res["seconds"] = round(time.time() - t0, 1)
        summary[f"stage{s}"] = res
        summ_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=_jsonable), encoding="utf-8")
        print(json.dumps(res, ensure_ascii=False, default=_jsonable)[:2000], flush=True)
    print(f"끝 → {out}")


if __name__ == "__main__":
    main(sys.argv[1:])
