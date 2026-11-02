"""A42 stage 5: final verdict, the sealed window, the trials ledger and the deflated Sharpe (DECISIONS A42 §5).

    .venv/Scripts/python.exe scripts/aa_a42_stage5.py --stage final --out docs/research/allocation/2026-11-02_a42_stage5
    .venv/Scripts/python.exe scripts/aa_a42_stage5.py --stage sealed --unseal-a42 --out ...     # once per target
    .venv/Scripts/python.exe scripts/aa_a42_stage5.py --stage ledger --out ...
    .venv/Scripts/python.exe scripts/aa_a42_stage5.py --stage figures --out ...
    add --quick for a pipeline check (few candidates, 3 price sets, no sealed data)

final   every candidate = static targets (R4, R5, R6, F2 안정형 40, F4 (가), F3 현재 보유), the fixed allocations F1 60/40 and
        F5 S&P500, and the 113 stage-4 passes (dynamic layer + satellite). Verdicts fixed in the design: V1 representative
        set MDD ≥ −15 % (−17 – −15 % listed as '경계'), V2 MDD ≥ −17 % in all 24 price sets, V3 ≥ 90 % of the declared
        policy's variants (quarter phase 0–2 × decision offset 0/5/10/15/20 days × 10 bp, plus 20 bp) inside −15 %,
        V4 the stage-1 scenario verdict on the target weights, V5 for rule weights (R4–R6) the stage-2 walk-forward
        verdict. Passers are sorted by the 10th percentile of CAGR over monthly starts (to 2015-12, representative set).
        No threshold is lowered when nothing passes. The volatility target without a signal (R4 / R6) is shown for
        reference outside the verdict.
sealed  the candidates that passed, F1–F5 and the reference rows, each run ONCE on 1991-03 – 2000-08 (the sealed series
        from Drive, hash-checked; KRW cash starts 1991-03) and joined to the research window. Verdict MDD ≥ −20 %. Every
        opening is appended to <out>/holdout.json before the run, and a target already in it is never run again.
        Products without sealed data are listed and their weight goes pro rata to the rest during the sealed part.
ledger  docs/research/allocation/trials_ledger.csv (A31–A42 trial counts) and the deflated Sharpe of the candidates.
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

from lab import aa_canary as K  # noqa: E402
from lab import aa_exposure as E  # noqa: E402
from lab import aa_layers as LY  # noqa: E402
from lab import aa_pricesets as PS  # noqa: E402
from lab import aa_sealed as SEAL  # noqa: E402
from lab import aa_tactical as TAC  # noqa: E402
from lab.aa_assets import CASH, PRODUCTS  # noqa: E402
from lab.aa_engine import TRADING_DAYS, decision_rows  # noqa: E402
from lab.aa_korea import KTB_DURATION, KrwModel, krw_cash_returns, ktb_returns, load_usdkrw, product_returns  # noqa: E402
from lab.aa_landscape import LAST_START  # noqa: E402
from lab.data import download_panel  # noqa: E402
from lab.metrics import deflated_sharpe  # noqa: E402

START, END = PS.START, "2024-12-31"
REP = PS.PriceSet("real", 60.0, "cd91")
ALLOC = ROOT / "docs" / "research" / "allocation"
STAGE3 = ALLOC / "2026-11-01_a42_stage3" / "targets.csv"
STAGE4 = ALLOC / "2026-11-02_a42_stage4" / "passed.csv"
WALK = ALLOC / "2026-10-31_a42_stage2" / "walkforward_table.csv"
BETAS = ALLOC / "2026-10-31_a42_stage1" / "betas.csv"
LEDGER = ALLOC / "trials_ledger.csv"
FIG_DIR = ALLOC / "figures" / "a42_stage5"
STATIC = ("R4 동일 비중", "R5 역변동성", "R6 주식 위험 몫 상한", "F2 안정형 40", "F4 (가)", "F3 현재 보유", "F1 60/40", "F5 S&P500")
FIXED = ("F1 60/40", "F2 안정형 40", "F3 현재 보유", "F4 (가)", "F5 S&P500")
RULE_OF = {"R4 동일 비중": "R4", "R5 역변동성": "R5", "R6 주식 위험 몫 상한": "R6"}
LINE, NEAR, SEALED_LINE = -0.15, -0.17, -0.20
POLICY = [(ph, off, 10.0) for ph in range(3) for off in (0, 5, 10, 15, 20)] + [(0, 0, 20.0)]
SEALED_FROM = "1991-03-04"  # first trading day with a KRW cash rate (CD91 1991-03) in the sealed files


# --------------------------------------------------------------------------- candidates

def targets() -> dict[str, dict[str, float]]:
    t = pd.read_csv(STAGE3)
    return {n: {c: float(x) for c, x in (p.split(":") for p in s.split(";"))} for n, s in zip(t["대상"], t["비중"])}


def candidates(quick: bool = False) -> pd.DataFrame:
    rows = [{"이름": n, "대상": n, "종류": "고정 배분" if n in FIXED else "정적 대상", **LY.Setting().__dict__, "위성": 0.0}
            for n in STATIC]
    for n in ("R4 동일 비중", "R6 주식 위험 몫 상한"):
        rows.append({"이름": f"{n} + 변동성 목표 10% (참고)", "대상": n, "종류": "참고 (판정 밖)",
                     **LY.Setting(vol=0.10).__dict__, "위성": 0.0})
    p = pd.read_csv(STAGE4)
    for _, r in p.iterrows():
        st = LY.Setting(r["signal"], r["action"], int(r["confirm"]), int(r["tranches"]), float(r["vol"]), int(r["cooldown"]))
        rows.append({"이름": f"{r['대상']} · {st.label()} · 위성 {r['위성']:.0%}", "대상": r["대상"], "종류": "동적 (4단계 통과)",
                     **st.__dict__, "위성": float(r["위성"])})
    c = pd.DataFrame(rows)
    if quick:
        c = pd.concat([c[c["종류"] != "동적 (4단계 통과)"].head(4), c[c["종류"] == "동적 (4단계 통과)"].head(3)])
    return c.reset_index(drop=True)


def setting_of(row) -> LY.Setting:
    return LY.Setting(row["signal"], row["action"], int(row["confirm"]), int(row["tranches"]), float(row["vol"]), int(row["cooldown"]))


# --------------------------------------------------------------------------- one price set: every candidate

def qrows_of(dates, phase=0, offset=0):
    rows = decision_rows(dates, "Q", phase) + int(offset)
    return rows[(rows >= 0) & (rows < len(dates) - 1)]


def path_name(name: str) -> str:
    import hashlib
    return "path_" + hashlib.md5(name.encode("utf-8")).hexdigest()[:16] + ".npy"


def rank_value(P: np.ndarray, dates: pd.DatetimeIndex) -> float:
    """10th percentile of CAGR from every month-end start up to LAST_START to the end."""
    rows = decision_rows(dates, "M")
    rows = rows[dates[rows] <= pd.Timestamp(LAST_START)]
    T = len(dates) - 1
    c = (P[-1] / P[rows]) ** (TRADING_DAYS / np.maximum(T - rows, 1)) - 1.0
    return float(np.quantile(c, 0.10))


def set_job(args):
    """All candidates on one price set. full = also the policy variants and the rank value (representative set)."""
    label, cands, full, inp_parts = args
    ps = next(p for p in PS.all_sets() if p.label == label)
    panel = PS.build(ps.prices, ps.hedge_bp, ps.cash, quirks=True)[0]
    can_close, usd_cash, fx, tac_close, tac_cash = inp_parts
    inp = K.Inputs(can_close, usd_cash, fx, panel, None)
    sat_eq = TAC.run_tactical(panel, tac_close, TAC.TacticalSpec("HAA50+BAA50 국내판", start=START, end=END), "krw", tac_cash).equity
    T = targets()
    ctx = {}
    out = []
    for _, c in cands.iterrows():
        w = T[c["대상"]]
        key = c["대상"]
        if key not in ctx:
            first = panel.close[list(w)].apply(lambda s: s.first_valid_index()).max()
            start = str(max(pd.Timestamp(START), first).date())
            cols = sorted(set(w) | set(K.DEFENSIVE) | {LY.CD})
            sims = {cost: K.DynamicSim(panel, cols, start, END, cost_bps=cost) for cost in (10.0, 20.0)}
            dates = sims[10.0].dates
            base_tab = LY.action_tables(w, "none", "none", sims[10.0].columns)
            bp, _, _ = sims[10.0].run(*K.static_rows(base_tab, sims[10.0]))
            base_r = np.concatenate([[0.0], bp[0, 1:] / bp[0, :-1] - 1.0])
            ctx[key] = {"sims": sims, "dates": dates, "base_r": base_r, "sat": sat_eq.reindex(dates).to_numpy(float),
                        "states": {}}
        x = ctx[key]
        st = setting_of(c)
        if st.signal != "none" and st.signal not in x["states"]:
            x["states"][st.signal] = K.states(st.signal, inp, K.RULES[st.signal].params[0], x["dates"]).to_numpy()
        S = x["states"][st.signal][None, :] if st.signal != "none" else np.zeros((1, len(x["dates"])), dtype=int)
        lev = LY.vol_levels(x["base_r"], x["dates"], st.vol) if st.vol else None

        def run(phase=0, offset=0, cost=10.0):
            sim = x["sims"][cost]
            tab = LY.action_tables(w, st.signal, st.action, sim.columns)
            P = LY.paths(sim, tab, S, st, qrows_of(x["dates"], phase, offset), lev)
            return LY.mix(P, x["sat"], c["위성"])[0]
        P = run()
        row = {"세트": label, "이름": c["이름"], "cagr": float(LY.cagr(P)[0]), "mdd": float(LY.mdd(P)[0]),
               "시작": str(x["dates"][0].date())}
        if full:
            var = [float(LY.mdd(run(ph, off, cost))[0]) for ph, off, cost in POLICY]
            row.update({"정책 변형 안 비율": float(np.mean(np.asarray(var) >= LINE)), "정책 변형 최악 MDD": float(min(var)),
                        "순위값 (매달 시작 CAGR 하위 10%)": rank_value(P, x["dates"])})
            np.save(Path(os.environ["A42S5_WORK"]) / path_name(c["이름"]), P)
        out.append(row)
    return pd.DataFrame(out)


def canary_inputs():
    usd, _ = download_panel(K.CANARY_TICKERS, end=PS.DATA_END)
    can_close, _ = K.chained(usd.close)
    tac, _ = download_panel(TAC.ALL_SYMBOLS, end=PS.DATA_END)
    return can_close, usd.cash, load_usdkrw(), tac.close, tac.cash


def stage_final(out: Path, quick: bool, workers: int) -> dict:
    C = candidates(quick)
    C.to_csv(out / "candidates.csv", index=False)
    work = out / "_work"
    work.mkdir(exist_ok=True)
    os.environ["A42S5_WORK"] = str(work)
    parts = canary_inputs()
    sets = [REP.label] + [p.label for p in PS.all_sets() if p != REP]
    if quick:
        sets = sets[:3]
    jobs = []
    for lb in sets:
        f = work / f"set_{lb}.csv"
        if not f.exists():
            jobs.append((lb, C, lb == REP.label, parts))
    t0 = time.time()
    with ProcessPoolExecutor(min(workers, len(jobs) or 1)) as ex:
        for k, (job, res) in enumerate(zip(jobs, ex.map(set_job, jobs))):
            res.to_csv(work / f"set_{job[0]}.csv", index=False)
            print(f"  세트 {k + 1}/{len(jobs)} {job[0]} ({time.time() - t0:.0f}s)", flush=True)
    R = pd.concat([pd.read_csv(work / f"set_{lb}.csv") for lb in sets], ignore_index=True)
    R.to_csv(out / "runs.csv.gz", index=False)
    V = final_verdicts(C, R)
    V.to_csv(out / "final.csv", index=False)
    return {"candidates": int(len(C)), "pass": int((V["최종"] == "통과").sum()), "border": int((V["최종"] == "경계").sum()),
            "sets": len(sets), "seconds": round(time.time() - t0, 1)}


def scenario_pass(w: dict) -> tuple[bool, int]:
    beta = pd.read_csv(BETAS, dtype={"상품": str}).set_index("상품")
    losses = E.scenario_losses(E.exposure({k: v / 100.0 for k, v in w.items()}, beta))
    return bool(E.scenario_verdict(losses)), int((losses < -0.25).sum())


def final_verdicts(C: pd.DataFrame, R: pd.DataFrame) -> pd.DataFrame:
    T = targets()
    wf = pd.read_csv(WALK)
    rule_ok = {r: bool(wf.loc[wf["규칙"] == r, "규칙 통과 (24세트 ≥ −17%)"].iloc[0]) for r in ("R4", "R5", "R6")}
    rep = R[R["세트"] == REP.label].set_index("이름")
    worst = R.groupby("이름")["mdd"].min()
    n_sets = R.groupby("이름")["세트"].nunique()
    rows = []
    for _, c in C.iterrows():
        n = c["이름"]
        r = rep.loc[n]
        sc_ok, sc_bad = scenario_pass(T[c["대상"]])
        v = {"이름": n, "대상": c["대상"], "종류": c["종류"], "cagr": r["cagr"], "mdd": r["mdd"], "시작": r["시작"],
             "순위값 (매달 시작 CAGR 하위 10%)": r["순위값 (매달 시작 CAGR 하위 10%)"],
             "V1 대표 MDD ≥ −15%": bool(r["mdd"] >= LINE), "V1 경계 (−17 – −15%)": bool(NEAR <= r["mdd"] < LINE),
             "24세트 최악 MDD": float(worst[n]), "세트 수": int(n_sets[n]), "V2 24세트 ≥ −17%": bool(worst[n] >= NEAR),
             "정책 변형 안 비율": r["정책 변형 안 비율"], "정책 변형 최악 MDD": r["정책 변형 최악 MDD"],
             "V3 정책 변형 90%": bool(r["정책 변형 안 비율"] >= 0.90),
             "시나리오 −25% 넘는 수": sc_bad, "V4 시나리오": sc_ok,
             "V5 walk-forward": rule_ok.get(RULE_OF.get(c["대상"], ""), True), "규칙 비중": c["대상"] in RULE_OF}
        allv = v["V2 24세트 ≥ −17%"] and v["V3 정책 변형 90%"] and v["V4 시나리오"] and v["V5 walk-forward"]
        v["최종"] = ("통과" if v["V1 대표 MDD ≥ −15%"] and allv else "경계" if v["V1 경계 (−17 – −15%)"] and allv else "탈락")
        if c["종류"].startswith("참고"):
            v["최종"] = "참고 (판정 밖): " + v["최종"]
        rows.append(v)
    V = pd.DataFrame(rows)
    order = V["최종"].map(lambda x: 0 if x == "통과" else 1 if x == "경계" else 3 if x.startswith("참고") else 2)
    return V.assign(_o=order).sort_values(["_o", "순위값 (매달 시작 CAGR 하위 10%)"], ascending=[True, False])         .drop(columns="_o").reset_index(drop=True)


# --------------------------------------------------------------------------- the sealed window

SEALED_PROXY = {"SPY": "yf_VFINX.csv", "QQQ": "yf__NDX.csv", "XLP": "yf_FDFAX.csv", "XLV": "yf_FSPHX.csv",
                "XLU": "yf_FSUTX.csv", "IEF": "yf_VFITX.csv", "TLT": "yf_VUSTX.csv", "GLD": "gold_monthly.csv"}
NDX_DIV = 0.005  # ^NDX is a price index: 0.5 % a year added (design 0-5)
IEF_DURATION = 7.5  # DGS10 total-return approximation before VFITX (1991-10)


_SEALED_CACHE: dict = {}


def sealed_inputs():
    """Daily USD proxy returns, USD cash, USD/KRW and Korean rates for the sealed window (hash-checked reads)."""
    if "inputs" in _SEALED_CACHE:
        return _SEALED_CACHE["inputs"]
    rd = lambda f: SEAL.read(f, unseal=True)  # noqa: E731
    dex = rd("fred_DEXKOUS.csv")
    dex = pd.Series(pd.to_numeric(dex["DEXKOUS"], errors="coerce").to_numpy(), index=pd.to_datetime(dex["date"])).dropna()
    idx = dex.index[(dex.index >= pd.Timestamp(SEALED_FROM)) & (dex.index <= pd.Timestamp(SEAL.SEAL_END))]
    rets = {}
    for sym, f in SEALED_PROXY.items():
        d = rd(f)
        if f == "gold_monthly.csv":
            s = pd.Series(pd.to_numeric(d["Price"], errors="coerce").to_numpy(),
                          index=pd.to_datetime(d["Date"]) + pd.offsets.MonthEnd(0))
        else:
            s = pd.Series(pd.to_numeric(d["close"], errors="coerce").to_numpy(), index=pd.to_datetime(d["date"]))
        line = s.reindex(s.index.union(idx)).ffill().reindex(idx)
        r = line.pct_change(fill_method=None)
        if sym == "QQQ":
            r = r + NDX_DIV / TRADING_DAYS
        rets[sym] = r
    dgs = rd("fred_DGS10.csv")
    y = pd.Series(pd.to_numeric(dgs["DGS10"], errors="coerce").to_numpy(), index=pd.to_datetime(dgs["date"])).ffill() / 100
    y = y.reindex(y.index.union(idx)).ffill().reindex(idx)
    approx = y.shift(1) / TRADING_DAYS - IEF_DURATION * (y - y.shift(1))
    rets["IEF"] = rets["IEF"].where(rets["IEF"].notna(), approx)
    usd_rets = pd.DataFrame(rets, index=idx)
    tb = rd("fred_DTB3.csv")
    tb = pd.Series(pd.to_numeric(tb["DTB3"], errors="coerce").to_numpy(), index=pd.to_datetime(tb["date"])).ffill() / 100
    usd_cash = tb.reindex(tb.index.union(idx)).ffill().reindex(idx) / TRADING_DAYS
    fx_ret = dex.reindex(idx).pct_change(fill_method=None).fillna(0.0)
    kr = rd("kr_rates_monthly.csv")
    kr["month"] = pd.to_datetime(kr["month"])
    kr = kr.set_index("month")
    kr["ktb30"] = np.nan
    _SEALED_CACHE["inputs"] = (idx, usd_rets, usd_cash, fx_ret, kr)
    return _SEALED_CACHE["inputs"]


def sealed_product_returns(codes: list[str], hedge_bp: float):
    idx, usd_rets, usd_cash, fx_ret, kr = sealed_inputs()
    krw_cash = krw_cash_returns(idx, kr["cd91"].dropna())
    full = kr[[c for c in ("ktb3", "ktb10", "ktb20", "ktb30") if kr[c].notna().any()]]
    ktb = pd.DataFrame(np.nan, index=idx, columns=list(KTB_DURATION))
    if "ktb10" in full:  # the 10- and 30-year yields do not exist in the sealed files: those products are missing
        ktb = ktb_returns(idx, kr)
    elif "ktb3" in full:
        k3 = ktb_returns(idx, kr.assign(ktb10=kr["ktb3"]))  # only the 3-year column is used below
        ktb["KTB3"] = k3["KTB3"]
    out, missing = {}, {}
    for c in codes:
        p = PRODUCTS[c]
        need = [s for s, _ in p.proxy if s not in usd_rets.columns and s not in ("KRW_CASH", "USD_CASH") and s not in KTB_DURATION]
        ktb_need = [s for s, _ in p.proxy if s in KTB_DURATION and ktb[s].iloc[1:].isna().any()]
        if need or ktb_need:
            missing[c] = ", ".join(need + ktb_need)
            continue
        r = product_returns(p, usd_rets, fx_ret, usd_cash, krw_cash, KrwModel(hedge_extra_bps=hedge_bp), ktb)
        out[c] = r.fillna(0.0)
    return idx, pd.DataFrame(out, index=idx), krw_cash, missing


def sealed_panel(cols: list[str]):
    """A small Panel over the sealed window for the product columns that have data, cash = KRW cash."""
    from lab.data import Panel
    idx, rets, cash, missing = sealed_product_returns(cols, REP.hedge_bp)
    close = 100 * (1 + rets).cumprod()
    p = Panel(open=close, high=close, low=close, close=close, volume=close * 0, tradable=list(close.columns), cash=cash)
    return p, missing


def stage_sealed(out: Path, quick: bool, unseal: bool) -> dict:
    if not unseal:
        raise SystemExit("봉인 구간은 --unseal-a42가 있어야 엽니다.")
    V = pd.read_csv(out / "final.csv")
    C = pd.read_csv(out / "candidates.csv")
    pick = V[(V["최종"] == "통과") | V["이름"].isin(FIXED) | V["종류"].str.startswith("참고")]["이름"].tolist()
    log_path = out / "holdout.json"
    log = json.loads(log_path.read_text(encoding="utf-8")) if log_path.exists() else []
    done = {e["이름"] for e in log}
    res_path = out / "sealed.csv"
    old = pd.read_csv(res_path) if res_path.exists() else pd.DataFrame()
    T = targets()
    parts = canary_inputs()
    rep_panel = PS.build(REP.prices, REP.hedge_bp, REP.cash, quirks=True)[0]
    sat = TAC.run_tactical(rep_panel, parts[3], TAC.TacticalSpec("HAA50+BAA50 국내판", start=START, end=END), "krw", parts[4]).equity
    rows = []
    for n in pick:
        if n in done:
            print(f"  이미 열었음, 다시 돌리지 않음: {n}")
            continue
        c = C[C["이름"] == n].iloc[0]
        w = T[c["대상"]]
        st = setting_of(c)
        log.append({"이름": n, "열린 시각": pd.Timestamp.now().isoformat(timespec="seconds"), "구간": f"{SEALED_FROM} – {SEAL.SEAL_END}",
                    "가격 세트": REP.label, "설정": st.label(), "위성": float(c["위성"])})
        log_path.write_text(json.dumps(log, ensure_ascii=False, indent=2), encoding="utf-8")  # before the run: one opening
        cols = sorted(set(w) | set(K.DEFENSIVE) | {LY.CD})
        sp, missing = sealed_panel(sorted(set(cols) | {K.NASDAQ}))  # nasdaq always: the D1 signal reads it
        have = [x for x in cols if x in sp.close.columns]
        miss_w = {k: v for k, v in w.items() if k not in have}
        w_s = {k: v for k, v in w.items() if k in have}
        sim = K.DynamicSim(sp, have, str(sp.close.index[0].date()), SEAL.SEAL_END)
        tab = LY.action_tables(w_s, st.signal if all(x in have for x in K.DEFENSIVE) else "none", st.action, sim.columns) \
            if w_s else None
        inp = K.Inputs(parts[0], parts[1], parts[2], sp, None)
        S = (K.states(st.signal, inp, K.RULES[st.signal].params[0], sim.dates).to_numpy()[None, :]
             if st.signal != "none" else np.zeros((1, sim.n_days), dtype=int))
        base_tab = LY.action_tables(w_s, "none", "none", sim.columns)
        bp, _, _ = sim.run(*K.static_rows(base_tab, sim))
        base_r = np.concatenate([[0.0], bp[0, 1:] / bp[0, :-1] - 1.0])
        lev = LY.vol_levels(base_r, sim.dates, st.vol) if st.vol else None
        Ps = LY.paths(sim, tab, S, st, qrows_of(sim.dates), lev)[0]
        # research window, as in the final stage (representative set)
        first = rep_panel.close[list(w)].apply(lambda s: s.first_valid_index()).max()
        rsim = K.DynamicSim(rep_panel, sorted(set(w) | set(K.DEFENSIVE) | {LY.CD}), str(max(pd.Timestamp(START), first).date()), END)
        rinp = K.Inputs(parts[0], parts[1], parts[2], rep_panel, None)
        rS = (K.states(st.signal, rinp, K.RULES[st.signal].params[0], rsim.dates).to_numpy()[None, :]
              if st.signal != "none" else np.zeros((1, rsim.n_days), dtype=int))
        rb, _, _ = rsim.run(*K.static_rows(LY.action_tables(w, "none", "none", rsim.columns), rsim))
        rlev = LY.vol_levels(np.concatenate([[0.0], rb[0, 1:] / rb[0, :-1] - 1.0]), rsim.dates, st.vol) if st.vol else None
        Pr = LY.mix(LY.paths(rsim, LY.action_tables(w, st.signal, st.action, rsim.columns), rS, st, qrows_of(rsim.dates), rlev),
                    sat.reindex(rsim.dates).to_numpy(float), float(c["위성"]))[0]
        gap = rsim.dates[0] > pd.Timestamp(START) + pd.Timedelta(days=7)
        joined = np.concatenate([Ps, Ps[-1] * Pr[1:] / Pr[0]]) if not gap else Ps
        row = {"이름": n, "대상": c["대상"], "종류": c["종류"], "봉인 구간 시작": str(sim.dates[0].date()),
               "봉인 구간 CAGR": float(LY.cagr(Ps)[0]), "봉인 구간 MDD": float(LY.mdd(Ps)[0]),
               "이어 붙인 MDD": float(LY.mdd(joined)[0]) if not gap else np.nan,
               "이어 붙인 CAGR": float(LY.cagr(joined)[0]) if not gap else np.nan,
               "자료 없는 상품": "; ".join(f"{k} {v:g}% ({missing.get(k, '자료 없음')})" for k, v in miss_w.items()),
               "자료 없는 비중": float(sum(miss_w.values())),
               "신호 자료": ("없음" if st.signal == "none" else f"{int((S[0] > 0).sum())}일 위험 신호"),
               "연구 구간이 늦게 시작": bool(gap)}
        m = row["이어 붙인 MDD"] if not gap else row["봉인 구간 MDD"]
        row["봉인 판정 (MDD ≥ −20%)"] = "통과" if m >= SEALED_LINE else "미달"
        if c["종류"].startswith("참고"):
            row["봉인 판정 (MDD ≥ −20%)"] = "참고: " + row["봉인 판정 (MDD ≥ −20%)"]
        rows.append(row)
        pd.concat([old, pd.DataFrame(rows)], ignore_index=True).to_csv(res_path, index=False)  # saved after each opening
        print(f"  열었음: {n}", flush=True)
    S_ = pd.concat([old, pd.DataFrame(rows)], ignore_index=True)
    S_.to_csv(res_path, index=False)
    return {"opened": len(rows), "total": int(len(S_)), "pass": int((S_["봉인 판정 (MDD ≥ −20%)"] == "통과").sum())}


# --------------------------------------------------------------------------- ledger and DSR

def stage_ledger(out: Path) -> dict:
    L = pd.read_csv(LEDGER)
    n_total = int(L["시험 수"].sum())
    V = pd.read_csv(out / "final.csv")
    work = out / "_work"
    paths = {n: np.load(work / path_name(n)) for n in V["이름"] if (work / path_name(n)).exists()}
    srs = []
    for n, P in paths.items():
        r = np.diff(P) / P[:-1]
        srs.append(r.mean() / r.std())
    sr_var = float(np.var(srs)) if len(srs) > 1 else 0.0
    rows = []
    for _, v in V.iterrows():
        if v["이름"] not in paths:
            continue
        dates = pd.bdate_range("2000-01-01", periods=len(paths[v["이름"]]))  # only the count of days matters
        eq = pd.Series(paths[v["이름"]], index=dates)
        rows.append({"이름": v["이름"], "최종": v["최종"], "DSR (N = 장부 합계)": deflated_sharpe(eq, n_total, sr_var),
                     "DSR (N = A42만)": deflated_sharpe(eq, int(L.loc[L["연구"] == "A42", "시험 수"].sum()), sr_var),
                     "PSR (N = 1)": deflated_sharpe(eq, 1, 0.0)})
    D = pd.DataFrame(rows)
    D.to_csv(out / "dsr.csv", index=False)
    return {"ledger_total": n_total, "sr_var_daily": sr_var, "rows": len(D)}


TRACKED = ("F4 (가)", "F3 현재 보유", "F2 안정형 40", "R6 주식 위험 몫 상한")  # proposed representatives' weights + F2, F3


def stage_track(out: Path) -> dict:
    """track.csv for lab/aa_track.py: weights and the 5–95 % quantiles of monthly returns of the static weights
    (quarterly, 10 bp) in the research window, representative set. Dynamic layers and satellites are not tracked."""
    T = targets()
    panel = PS.build(REP.prices, REP.hedge_bp, REP.cash, quirks=True)[0]
    rows = []
    for n in TRACKED:
        w = T[n]
        first = panel.close[list(w)].apply(lambda s: s.first_valid_index()).max()
        sim = K.DynamicSim(panel, sorted(w), str(max(pd.Timestamp(START), first).date()), END)
        P, _, _ = sim.run(*K.static_rows(LY.action_tables(w, "none", "none", sim.columns), sim))
        m = pd.Series(P[0], index=sim.dates).resample("ME").last().pct_change().dropna()
        rows.append({"이름": n, "비중": ";".join(f"{c}:{int(x)}" for c, x in w.items()),
                     "예상 5%": float(m.quantile(0.05)), "예상 95%": float(m.quantile(0.95)), "기준 월 수": int(len(m))})
    t = pd.DataFrame(rows)
    t.to_csv(out / "track.csv", index=False)
    return {"tracked": list(TRACKED)}


def figures(out: Path, dest: Path = FIG_DIR) -> list[str]:
    from lab import figures as figs
    if not figs.available():
        return []
    plt = figs._plt()
    dest.mkdir(parents=True, exist_ok=True)
    made = []
    m = pd.read_csv(out / "final_with_sealed.csv")
    kind_color = {"정적 대상": "#4C78A8", "고정 배분": "#E45756", "동적 (4단계 통과)": "#F58518", "참고 (판정 밖)": "#72B7B2"}
    fig, ax = plt.subplots(figsize=(10, 6))
    for k, col in kind_color.items():
        g = m[m["종류"] == k]
        ok = g["최종"].astype(str).str.contains("통과")
        ax.scatter(g.loc[ok, "mdd"] * 100, g.loc[ok, "순위값 (매달 시작 CAGR 하위 10%)"] * 100, s=22, color=col, label=f"{k} (최종 통과)")
        ax.scatter(g.loc[~ok, "mdd"] * 100, g.loc[~ok, "순위값 (매달 시작 CAGR 하위 10%)"] * 100, s=40, marker="x", color=col,
                   label=f"{k} (최종 탈락)" if (~ok).any() else None)
    for _, r in m[~m["종류"].str.startswith("동적")].iterrows():
        ax.annotate(r["이름"].replace(" (참고)", "*"), (r["mdd"] * 100, r["순위값 (매달 시작 CAGR 하위 10%)"] * 100), fontsize=7,
                    xytext=(3, 3), textcoords="offset points")
    ax.axvline(-15, color="gray", lw=0.8, ls="--")
    ax.set_xlabel("대표 세트 MDD (%), 2000-09 – 2024-12")
    ax.set_ylabel("순위값: 매달 시작 CAGR 하위 10% (%)")
    ax.set_title("① 최종 판정 (V1–V5): 후보 123개의 MDD와 순위값\n* = 신호 없는 변동성 목표 10% (판정 밖 참고)")
    ax.legend(fontsize=7)
    made.append(str(figs._save(fig, dest / "1_final.png")))
    fig, ax = plt.subplots(figsize=(10, 6))
    g = m[m["봉인 구간 MDD"].notna()]
    for k, col in kind_color.items():
        h = g[g["종류"] == k]
        ax.scatter(h["mdd"] * 100, h["봉인 구간 MDD"] * 100, s=24, color=col, label=k)
    for _, r in g[~g["종류"].str.startswith("동적")].iterrows():
        ax.annotate(r["이름"].replace(" (참고)", "*"), (r["mdd"] * 100, r["봉인 구간 MDD"] * 100), fontsize=7, xytext=(3, 3),
                    textcoords="offset points")
    ax.axhline(-20, color="black", lw=0.8, ls="--")
    ax.plot([-50, 0], [-50, 0], color="gray", lw=0.6)
    ax.set_xlabel("연구 구간 MDD (%), 대표 세트")
    ax.set_ylabel("봉인 구간 MDD (%), 1991-03 – 2000-08")
    ax.set_title("② 봉인 구간 ① 개봉 (대상마다 한 번): 연구 구간 vs 봉인 구간 MDD\n점선 −20% = 봉인 판정. 자료 없는 상품은 봉인 구간에서 나머지에 비례 배분")
    ax.legend(fontsize=7)
    made.append(str(figs._save(fig, dest / "2_sealed.png")))
    L = pd.read_csv(LEDGER)
    tot = L.groupby("연구", sort=False)["시험 수"].sum()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.bar(tot.index, tot.to_numpy(), color="#4C78A8")
    ax.set_yscale("log")
    for i, v in enumerate(tot.to_numpy()):
        ax.text(i, v * 1.15, f"{int(v):,}", ha="center", fontsize=7)
    ax.set_ylabel("시험 수 (로그)")
    ax.set_title(f"③ 다중 시험 장부: A31–A42 합계 {int(tot.sum()):,}")
    made.append(str(figs._save(fig, dest / "3_ledger.png")))
    return made


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--stage", required=True, choices=("final", "sealed", "ledger", "track", "figures"))
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--unseal-a42", action="store_true", dest="unseal")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sp = out / "summary.json"
    summ = json.loads(sp.read_text(encoding="utf-8")) if sp.exists() else {}
    if a.stage == "final":
        summ["final"] = stage_final(out, a.quick, a.workers)
    elif a.stage == "sealed":
        summ["sealed"] = stage_sealed(out, a.quick, a.unseal)
    elif a.stage == "ledger":
        summ["ledger"] = stage_ledger(out)
    elif a.stage == "track":
        summ["track"] = stage_track(out)
    else:
        print(figures(out))
        return
    sp.write_text(json.dumps(summ, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(json.dumps(summ, ensure_ascii=False, default=str)[:2000])


if __name__ == "__main__":
    main(sys.argv[1:])
