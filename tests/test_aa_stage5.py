"""A42 stage 5: sealed-window guard, the monthly tracker and the committed result files. No network."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from lab import aa_sealed, aa_track

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "research" / "allocation" / "2026-11-02_a42_stage5"


def test_sealed_window_needs_unseal():
    with pytest.raises(aa_sealed.SealedError):
        aa_sealed.guard("1995-01-02")
    with pytest.raises(aa_sealed.SealedError):
        aa_sealed.read("fred_DGS10.csv")
    aa_sealed.guard("2000-09-01")  # the research window is open


def test_month_returns_use_the_last_close_before_and_inside():
    idx = pd.to_datetime(["2026-08-28", "2026-08-31", "2026-09-15", "2026-09-30", "2026-10-01"])
    s = pd.Series([100.0, 110.0, 105.0, 121.0, 130.0], index=idx)
    r = aa_track.month_returns({"A": s, "B": s[:2]}, pd.Period("2026-09", "M"))
    assert np.isclose(r["A"], 121.0 / 110.0 - 1) and "B" not in r


def test_track_file_and_workflow():
    t = aa_track.load_track()
    assert len(t) >= 3 and (t["예상 5%"] < 0).all() and (t["예상 95%"] > 0).all()
    assert all(abs(sum(w.values()) - 100) < 1e-9 for w in t["weights"])
    wf = (ROOT / ".github" / "workflows" / "aa_a42_track.yml").read_text(encoding="utf-8")
    assert "python -m lab.aa_track" in wf and "issues: write" in wf and "contents: read" in wf


def test_sealed_openings_are_logged_once():
    log = json.loads((OUT / "holdout.json").read_text(encoding="utf-8"))
    names = [e["이름"] for e in log]
    assert len(names) == len(set(names))
    s = pd.read_csv(OUT / "sealed.csv")
    assert set(s["이름"]) == set(names)


def test_final_and_ledger_files():
    f = pd.read_csv(OUT / "final.csv")
    assert {"통과", "탈락"} <= set(f["최종"])
    for c in ("V1 대표 MDD ≥ −15%", "V2 24세트 ≥ −17%", "V3 정책 변형 90%", "V4 시나리오", "V5 walk-forward"):
        assert c in f
    L = pd.read_csv(ROOT / "docs" / "research" / "allocation" / "trials_ledger.csv")
    assert {"A31", "A41", "A42"} <= set(L["연구"]) and (L["시험 수"] > 0).all()


def test_stage5_chart_data():
    from lab import aa_experiments as ex
    m = ex.a42_5_table()
    assert len(m) == 123 and m["봉인 구간 MDD"].notna().sum() == 123


def test_summary_numbers_match_stage5():
    from lab import aa_summary as S
    m = pd.read_csv(OUT / "final_with_sealed.csv").set_index("이름")
    for p in S.PICKS:
        r = m.loc[p.stage5_name]
        assert round(r["cagr"] * 100, 2) == p.cagr and round(r["mdd"] * 100, 1) == p.mdd
        assert round(r["봉인 구간 MDD"] * 100, 1) == p.sealed_mdd
        # a pick is a stage-5 candidate that passed both verdicts, or a reference row that passed them (A43 reading, 2026-11-11)
        assert str(r["최종"]).endswith("통과") and str(r["봉인 판정 (MDD ≥ −20%)"]).endswith("통과")
        assert not str(r["최종"]).endswith("탈락")
    assert (S.ALLOC / "SUMMARY.md").exists() and len(S.scatter_data()) == len(m)


def test_summary_after_tax_numbers_match_a45():
    from lab import aa_summary as S
    t = pd.read_csv(S.A45).set_index("대상")
    for p in S.PICKS:
        r = t.loc[p.a45_name]
        assert round(r["세후 CAGR (TWR)"] * 100, 2) == p.after_cagr and round(r["세후 MDD (TWR)"] * 100, 1) == p.after_mdd
        assert r["X1 세후 구역"] == p.after_zone
