"""A42 registered dynamic presets (lab/aa_dynamic.py): definitions match the stage-5 files, and re-running them gives
the stage-5 numbers (needs the local price files; skipped where they are missing, e.g. in CI)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lab import aa_dynamic as D

ROOT = Path(__file__).resolve().parents[1]
S5 = ROOT / "docs" / "research" / "allocation" / "2026-11-02_a42_stage5"
S3 = ROOT / "docs" / "research" / "allocation" / "2026-11-01_a42_stage3" / "targets.csv"


def test_definitions_match_stage5_candidates():
    c = pd.read_csv(S5 / "candidates.csv").set_index("이름")
    t = pd.read_csv(S3).set_index("대상")["비중"]
    for p in D.DYNAMIC_PRESETS.values():
        row = c.loc[p.stage5_name]
        st = p.setting
        assert (row["signal"], row["action"]) == (st.signal, st.action)
        assert (int(row["confirm"]), int(row["tranches"]), int(row["cooldown"])) == (st.confirm, st.tranches, st.cooldown)
        assert np.isclose(float(row["vol"]), st.vol) and np.isclose(float(row["위성"]), p.satellite)
        assert row["대상"] == p.base
        want = {k: float(v) for k, v in (x.split(":") for x in t[p.base].split(";"))}
        assert {k: float(v) for k, v in p.weights.items()} == want
        assert p.badge == "사후" and any("7.7%" in x for x in p.limits)


def _data():
    from lab import aa_real
    if aa_real.load_files() is None:
        pytest.skip("실제 가격 파일이 없습니다 (lab_data/kr)")
    try:
        from lab import aa_pricesets as PS
        panel = PS.build("real", 60.0, "cd91", quirks=True)[0]
        inp = D.inputs(PS.DATA_END)
    except Exception as exc:  # offline without caches
        pytest.skip(f"가격 자료를 만들지 못했습니다: {exc}")
    return panel, inp


TOL = 1e-6  # with the 2026-10-27 price files the re-run equals stage 5 to 1e-17; the lab refreshes lab_data/kr every
# 20 hours and later downloads can revise a past distribution or close by a hair (2026-11-02 refresh: 3e-8 in CAGR)


def test_rerun_equals_stage5():
    panel, inp = _data()
    runs = pd.read_csv(S5 / "runs.csv.gz")
    rep = runs[runs["세트"] == "real·h60·cd91"].set_index("이름")
    for p in D.DYNAMIC_PRESETS.values():
        r = D.run(panel, p, inp)
        assert abs(r["cagr"] - rep.loc[p.stage5_name, "cagr"]) < TOL
        assert abs(r["mdd"] - rep.loc[p.stage5_name, "mdd"]) < TOL
        tw, why = D.current_targets(panel, p, inp)
        assert np.isclose(sum(tw.values()), 1.0) and min(tw.values()) >= 0 and "기준" in why
