"""A44: decision-date, cost and mixing test of the HAA50+BAA50 satellite inside ② and ③ (allocation DECISIONS A44).

    .venv/Scripts/python.exe scripts/aa_a44_satellite.py --quick --out %TEMP%\\a44_quick
    .venv/Scripts/python.exe scripts/aa_a44_satellite.py --out docs/research/allocation/2026-11-11_a44_satellite
    ... --figures-only

Cores: (가) static, ③'s core ((가) + K2 → safe bundle, cooldown 21, vol target 8 %), 현재 보유 static, ②'s core (현재 보유
+ D1 → CD, confirm 10, cooldown 21, 4 tranches), exactly as registered in `lab/aa_dynamic.py`. Each is combined with the
satellite at 25 / 50 %, the satellite decided 0 / 1 / 5 / 10 / 15 trading days after month end or split in four
tranches, at 10 / 20 bp, mixed daily for free (stage 5) or restored monthly / quarterly at 10 bp. Three price sets
(representative + the two stage-4 extremes); the verdicts read the representative set.
Design fixed before the run: docs/research/allocation/2026-11-11_a44_design.md. A few minutes, one process.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab import aa_canary as K  # noqa: E402
from lab import aa_dynamic as D  # noqa: E402
from lab import aa_layers as LY  # noqa: E402
from lab import aa_satellite as SA  # noqa: E402

FIG_DIR = ROOT / "docs" / "research" / "allocation" / "figures" / "a44_satellite"
STAGE5 = ROOT / "docs" / "research" / "allocation" / "2026-11-02_a42_stage5" / "runs.csv.gz"
SETS = {"대표": ("real", 60.0, "cd91"), "극단 1": ("model", 60.0, "base"), "극단 2": ("real", 0.0, "cd91")}  # stage 4
CORES = {  # core name -> (registered preset, dynamic layer on?)
    "(가) 정적": ("a42_dyn3", False),
    "③ 본체: (가) + K2 → 안전 묶음 + 변동성 목표 8%": ("a42_dyn3", True),
    "현재 보유 정적": ("a42_dyn2", False),
    "② 본체: 현재 보유 + D1 → CD (4분할)": ("a42_dyn2", True),
}
COMBOS = {  # combination name -> core name
    "(가) + 위성 (신호 없음)": "(가) 정적",
    "③ = (가) + K2 + 변동성 8% + 위성": "③ 본체: (가) + K2 → 안전 묶음 + 변동성 목표 8%",
    "현재 보유 + 위성 (신호 없음)": "현재 보유 정적",
    "② = 현재 보유 + D1 + 위성": "② 본체: 현재 보유 + D1 → CD (4분할)",
}
REGISTERED = {"③ = (가) + K2 + 변동성 8% + 위성": "a42_dyn3", "② = 현재 보유 + D1 + 위성": "a42_dyn2"}
SAT_ALONE = "위성 단독 (HAA50·BAA50 국내판)"
REPRO_TOL = 1e-6  # as tests/test_aa_dynamic.py: later price downloads revise past closes by a hair


def core_paths(panel, inp: dict) -> tuple[pd.DatetimeIndex, dict[str, np.ndarray]]:
    out, dates0 = {}, None
    for name, (key, dynamic) in CORES.items():
        p = D.DYNAMIC_PRESETS[key]
        sim, dates, base, S, lev, tab, _ = D._parts(panel, p, inp, D.START, D.END)
        if dates0 is None:
            dates0 = dates
        if not dates.equals(dates0):
            raise RuntimeError(f"{name}: 날짜가 다른 본체와 다릅니다 ({dates[0].date()} – {dates[-1].date()})")
        path = LY.paths(sim, tab, S, p.setting, K.quarter_rows(dates), lev)[0] if dynamic else base
        out[name] = np.asarray(path, float) / path[0]
    return dates0, out


def run_set(sname: str, panel, inp: dict) -> pd.DataFrame:
    dates, cores = core_paths(panel, inp)
    end = str(dates[-1].date())
    rows = []

    def add(combo, core, share, off, cost, mix, path, windows=tuple(SA.WINDOWS)):
        for w in windows:
            c, m = SA.window_metrics(path, dates, w)
            rows.append({"세트": sname, "조합": combo, "본체": core, "구간": w, "위성 몫": share, "오프셋": off,
                         "위성 비용 bp": cost, "섞기": mix, "cagr": c, "mdd": m})

    for cname, path in cores.items():
        add(cname, cname, 0.0, "—", np.nan, "—", path)
    for cost in SA.COSTS:
        sats = {str(o): SA.satellite_equity(panel, inp["tac_close"], inp["tac_cash"], o, cost, D.START, end) for o in SA.OFFSETS}
        sats[SA.TRANCHE] = SA.tranche_equity([sats[str(o)] for o in SA.TRANCHE_OFFSETS])
        for off, eq in sats.items():
            sat = eq.reindex(dates).to_numpy(float)
            alone = np.where(np.isfinite(sat), sat, np.nan)
            live = np.isfinite(alone)
            first = int(np.flatnonzero(live)[0])
            alone = np.concatenate([np.full(first, alone[first]), alone[first:]])  # flat before it starts
            add(SAT_ALONE, "—", 1.0, off, cost, "—", alone, ("위성 구간",))
            for combo, cname in COMBOS.items():
                for share in SA.SHARES:
                    for mix in SA.MIXES:
                        add(combo, cname, share, off, cost, mix, SA.combine(cores[cname], sat, share, mix, dates))
        print(f"  {sname}: 위성 비용 {cost:g}bp 끝", flush=True)
    return pd.DataFrame(rows)


def reproduction(R: pd.DataFrame) -> dict:
    """Offset 0 · 10 bp · daily mix · 50 % on the representative set must be the stage-5 number."""
    s5 = pd.read_csv(STAGE5)
    s5 = s5[s5["세트"] == "real·h60·cd91"].set_index("이름")
    out = {}
    for combo, key in REGISTERED.items():
        r = R[(R["세트"] == "대표") & (R["조합"] == combo) & (R["구간"] == "전체") & (R["위성 몫"] == 0.5) & (R["오프셋"] == "0")
              & (R["위성 비용 bp"] == 10.0) & (R["섞기"] == "daily")].iloc[0]
        want = s5.loc[D.DYNAMIC_PRESETS[key].stage5_name]
        out[combo] = {"cagr": float(r["cagr"]), "mdd": float(r["mdd"]), "stage5_cagr": float(want["cagr"]),
                      "stage5_mdd": float(want["mdd"]),
                      "같음": bool(abs(r["cagr"] - want["cagr"]) < REPRO_TOL and abs(r["mdd"] - want["mdd"]) < REPRO_TOL)}
    return out


def analyse(out: Path, R: pd.DataFrame, meta: dict) -> dict:
    rep = R[R["세트"] == "대표"]
    V = pd.concat([SA.verdict_table(rep, w) for w in SA.WINDOWS], ignore_index=True)
    V.to_csv(out / "verdicts.csv", index=False)
    X = []  # the same verdict table on the extreme sets, for information
    for s in sorted(set(R["세트"]) - {"대표"}):
        v = SA.verdict_table(R[R["세트"] == s], "전체")
        v.insert(0, "세트", s)
        X.append(v)
    if X:
        pd.concat(X, ignore_index=True).to_csv(out / "verdicts_extreme_sets.csv", index=False)
    alone = rep[rep["조합"] == SAT_ALONE]
    full = V[V["구간"] == "전체"].set_index("조합")
    summ = {"meta": meta, "repro": reproduction(R),
            "judgements": {c: {k: (v.item() if hasattr(v, "item") else v) for k, v in full.loc[c].items()
                               if k in ("W1 날짜 견고", "W2 구역", "W3 위성 기여", "날짜 민감", "종합", "MDD 최악 (오프셋)",
                                        "MDD 최선 (오프셋)", "4분할 MDD", "4분할 CAGR", "4분할 구역")} for c in full.index},
            "satellite_alone_mdd_10bp": {str(o): float(alone[(alone["오프셋"] == str(o)) & (alone["위성 비용 bp"] == 10.0)]["mdd"].iloc[0])
                                         for o in [*SA.OFFSETS, SA.TRANCHE]}}
    (out / "summary.json").write_text(json.dumps(summ, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    return summ


def figures(out: Path, dest: Path = FIG_DIR) -> list[str]:
    from lab import figures as figs
    if not figs.available():
        return []
    plt = figs._plt()
    R = pd.read_csv(out / "runs.csv", keep_default_na=False, na_values=[""])
    R = R[R["세트"] == "대표"]
    offs = [*[str(o) for o in SA.OFFSETS], SA.TRANCHE]
    labels = ["월말", "+1일", "+5일", "+10일", "+15일", "4분할"]
    j = R[(R["구간"] == "전체") & (R["위성 몫"] == 0.5) & (R["섞기"] == SA.JUDGED["mix"])]
    made = []
    x = np.arange(len(offs))
    for fname, col, ylabel, title in (
            ("1_mdd_by_offset.png", "mdd", "MDD (%)", "① 위성 결정일을 옮기면 조합의 MDD가 얼마나 달라지나"),
            ("2_cagr_by_offset.png", "cagr", "CAGR (%)", "② 위성 결정일·교체 비용별 CAGR와 위성 없는 같은 본체")):
        fig, axes = plt.subplots(1, len(COMBOS), figsize=(4.3 * len(COMBOS), 4.8), sharey=True)
        for ax, (combo, cname) in zip(axes, COMBOS.items()):
            for k, (cost, color) in enumerate(zip(SA.COSTS, ("#4C78A8", "#F58518"))):
                g = j[(j["조합"] == combo) & (j["위성 비용 bp"] == cost)].set_index("오프셋").reindex(offs)
                ax.bar(x + (k - 0.5) * 0.38, g[col] * 100, 0.38, color=color, label=f"위성 비용 {cost:g}bp")
            base = R[(R["조합"] == cname) & (R["구간"] == "전체") & (R["위성 몫"] == 0.0)][col].iloc[0] * 100
            ax.axhline(base, color="#333333", ls="--", lw=1, label="위성 없는 같은 본체")
            if col == "mdd":
                for y, c in ((-15, "#54A24B"), (-17, "#E0A800"), (-20, "#D62728")):
                    ax.axhline(y, color=c, lw=0.8, ls=":")
            ax.set_xticks(x, labels, fontsize=8)
            ax.set_title(combo, fontsize=9)
        axes[0].set_ylabel(ylabel)
        axes[0].legend(fontsize=7)
        fig.suptitle(f"{title} (2000-09 – 2024-12, 대표 세트, 위성 50%, 분기 말 복원 10bp)", fontsize=10)
        made.append(str(figs._save(fig, dest / fname)))
    # ③ the satellite alone, and what the mixing convention costs
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.6))
    al = R[(R["조합"] == SAT_ALONE) & (R["구간"] == "위성 구간")]
    for k, (cost, color) in enumerate(zip(SA.COSTS, ("#4C78A8", "#F58518"))):
        g = al[al["위성 비용 bp"] == cost].set_index("오프셋").reindex(offs)
        a1.bar(x + (k - 0.5) * 0.38, g["mdd"] * 100, 0.38, color=color, label=f"{cost:g}bp")
    for y, c in ((-15, "#54A24B"), (-20, "#D62728")):
        a1.axhline(y, color=c, lw=0.8, ls=":")
    a1.set_xticks(x, labels)
    a1.set_ylabel("MDD (%)")
    a1.set_title("위성 단독 MDD (2007-04 – 2024-12)", fontsize=10)
    a1.legend(fontsize=8)
    m = R[(R["구간"] == "전체") & (R["위성 몫"] == 0.5) & (R["오프셋"] == "0") & (R["위성 비용 bp"] == 10.0)]
    xm = np.arange(len(COMBOS))
    for k, (mix, color) in enumerate(zip(SA.MIXES, ("#BBBBBB", "#4C78A8", "#72B7B2"))):
        g = m[m["섞기"] == mix].set_index("조합").reindex(list(COMBOS))
        a2.bar(xm + (k - 1) * 0.27, g["cagr"] * 100, 0.27, color=color, label=SA.MIX_KO[mix])
    a2.set_xticks(xm, [c.split(" = ")[0] if " = " in c else c for c in COMBOS], fontsize=7, rotation=12)
    a2.set_ylabel("CAGR (%)")
    a2.set_title("본체와 위성을 섞는 방식별 CAGR (월말 결정, 10bp)", fontsize=10)
    a2.legend(fontsize=7)
    lo = float(np.nanmin(m["cagr"])) * 100
    a2.set_ylim(max(0.0, lo - 1.0), None)
    fig.suptitle("③ 위성 단독의 날짜 민감도와 섞는 방식의 비용 (대표 세트)", fontsize=10)
    made.append(str(figs._save(fig, dest / "3_satellite_alone_and_mixing.png")))
    return made


def main(argv):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--quick", action="store_true", help="representative price set only")
    ap.add_argument("--figures-only", action="store_true")
    args = ap.parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    dest = FIG_DIR if not args.quick else out / "figures"
    if args.figures_only:
        print(figures(out, dest))
        return
    from lab import aa_pricesets as PS
    t0 = time.time()
    inp = D.inputs(PS.DATA_END)
    names = ["대표"] if args.quick else list(SETS)
    frames = []
    for s in names:
        pr, hb, cash = SETS[s]
        panel = PS.build(pr, hb, cash, quirks=True)[0]
        frames.append(run_set(s, panel, inp))
        print(f"  {s} 끝 ({time.time() - t0:.0f}s)", flush=True)
    R = pd.concat(frames, ignore_index=True)
    R.to_csv(out / "runs.csv", index=False)
    summ = analyse(out, R, {"quick": args.quick, "sets": names, "seconds": round(time.time() - t0, 1)})
    print(json.dumps({k: summ[k] for k in ("repro", "judgements", "satellite_alone_mdd_10bp")}, ensure_ascii=False, default=str))
    if not all(v["같음"] for v in summ["repro"].values()):
        print("경고: 등록 조건(월말 · 10bp · 매일 고정 비율)이 5단계 숫자와 다릅니다. 결과를 쓰기 전에 원인을 확인하세요.", flush=True)
    print(figures(out, dest))


if __name__ == "__main__":
    main(sys.argv[1:])
