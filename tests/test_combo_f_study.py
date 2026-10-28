"""Stage F (lab/combo_f_study.py, DECISIONS 84) on small synthetic data."""
from dataclasses import replace

import numpy as np
import pandas as pd

from lab import combo_f_study as cf, research_log as rlog, screen_e_study as se, stage_c_study as sc
from tests.test_screen_e_study import Serial
from tests.test_stage_c import _study_panel


def test_menu_is_registered():
    assert cf.SINGLES == ["x_weak_industry", "x_overextended", "x_ivol", "x_volatility", "s_efficiency"]
    f1 = ["x_weak_industry", "x_overextended"]
    assert cf.COMBOS == {"F1": f1, "F2": f1 + ["x_ivol"], "F3": f1 + ["s_efficiency"],
                         "F4": f1 + ["x_ivol", "s_efficiency"], "F5": f1 + ["x_ivol", "x_volatility"]}
    assert all(s in se.EXCLUDE or s in se.SELECT for s in cf.SINGLES)
    assert se.EXCLUDE["x_weak_industry"] == 30 and se.EXCLUDE["x_overextended"] == 10
    assert se.EXCLUDE["x_ivol"] == 20 and se.EXCLUDE["x_volatility"] == 20
    assert cf.names() == ["none"] + cf.SINGLES + ["F1", "F2", "F3", "F4", "F5"]


def _cells(screen, sample, values, col):
    return pd.DataFrame({"sample": sample, "entry": "E1", "exit": "X1", "xi": range(len(values)), "yi": 0,
                         "screen": screen, col: values, "cagr": values, "mdd": -0.3, "dd_excess": 0.0, "signal_share": 0.5})


def test_best_single_and_stacking_rule(monkeypatch):
    monkeypatch.setattr(cf, "COMBOS", {"F1": ["a", "b"]})
    monkeypatch.setattr(cf, "SAMPLES", {"s1": None, "s2": None})
    scr = pd.DataFrame({"screen": ["a", "a", "b", "b"], "sample": ["s1", "s2"] * 2,
                        "median_d_expectancy": [0.01, 0.03, 0.025, 0.0], "median_d_cagr": [0.0, 0.0, 0.01, 0.01]})
    assert cf.best_single(scr, ["a", "b"], "expectancy") == "a"  # mean 0.02 vs 0.0125
    assert cf.best_single(scr, ["a", "b"], "cagr") == "b"
    v = np.arange(10) / 100
    c = pd.concat([_cells("a", s, v, "expectancy") for s in ("s1", "s2")]
                  + [_cells("b", s, v, "expectancy") for s in ("s1", "s2")]
                  # combo better on 7/10 cells in s1, on 5/10 in s2 -> does not stack
                  + [_cells("F1", "s1", v + np.r_[[0.01] * 7, [-0.01] * 3], "expectancy"),
                     _cells("F1", "s2", v + np.r_[[0.01] * 5, [-0.01] * 5], "expectancy")], ignore_index=True)
    st = cf.stacking(c, scr)
    u = st[st["view"] == "unlimited"].set_index("sample")
    assert u.loc["s1", "share_better"] == 0.7 and u.loc["s2", "share_better"] == 0.5
    assert u["best_single"].eq("a").all() and not u["stacks"].any()
    c.loc[(c["screen"] == "F1") & (c["sample"] == "s2") & (c["xi"] >= 5), "expectancy"] += 0.02  # now 10/10 in s2
    assert cf.stacking(c, scr).query("view == 'unlimited'")["stacks"].all()


def test_pipeline_combo_masks_and_summary(tmp_path, monkeypatch):
    monkeypatch.setattr(se, "_W", {})
    p, dates = _study_panel()
    monkeypatch.setattr(sc, "SAMPLES", {"a": (str(dates[260].date()), str(dates[-1].date()), "A")})
    monkeypatch.setattr(se, "SAMPLES", sc.SAMPLES)
    monkeypatch.setattr(cf, "SAMPLES", sc.SAMPLES)
    monkeypatch.setattr(sc, "ENTRIES", {"E0": sc.ENTRIES["E0"], "E1": sc.ENTRIES["E1"]})
    monkeypatch.setattr(sc, "EXITS", {"X1": sc.EXITS["X1"]})
    monkeypatch.setattr(sc, "HYGIENE", {"h_price": 5})
    monkeypatch.setattr(sc, "SLOTS", 2)
    monkeypatch.setattr(cf, "SINGLES", ["x_volatility", "s_efficiency"])
    monkeypatch.setattr(cf, "COMBOS", {"F1": ["x_volatility", "s_efficiency"]})
    monkeypatch.setattr(rlog, "RESULTS_DIR", tmp_path / "res")
    elig = sc.daily_eligible(p)
    sc._setup(p, elig)
    sc.run_signal_tables(Serial(), tmp_path / "sig", lambda m: None)
    pe = replace(p, eligible=elig)
    out = tmp_path / "out"
    cf._activate()
    cf.precompute_masks(pe, elig, out, lambda m: None)
    v, e, f = (pd.read_parquet(se.mask_file(out, n)) for n in ("x_volatility", "s_efficiency", "F1"))
    assert f.equals(v & e) and f.to_numpy().sum() < e.to_numpy().sum()
    # the efficiency screen is ranked among all eligible names, as in Stage E (not after the other screen)
    pd.testing.assert_frame_equal(e, se.screen_mask(pe, elig, "s_efficiency"))
    se._setup(pe, str(tmp_path / "sig"), str(out))
    cf.run_pairs(Serial(), out, pe, lambda m: None)
    rows = pd.read_csv(out / "a" / "E1__X1.csv")
    assert set(rows["screen"]) == {"none", "x_volatility", "s_efficiency", "F1"} and len(rows) == 9 * 4
    trials = pd.read_csv(tmp_path / "res" / "trials.csv")
    assert len(trials) == 9 and trials["kind"].eq("screen_F1").all()  # only the combos are new trials
    # a Stage E run of the same screens is reproduced exactly
    se._W.clear()
    monkeypatch.setattr(se, "EXCLUDE", {"x_volatility": 20})
    monkeypatch.setattr(se, "SELECT", ["s_efficiency"])
    se.precompute_masks(pe, elig, tmp_path / "e", lambda m: None)
    se._setup(pe, str(tmp_path / "sig"), str(tmp_path / "e"))
    se.run_pairs(Serial(), tmp_path / "e", pe, lambda m: None)
    se.summarize(tmp_path / "e")
    res = cf.summarize(out, None, tmp_path / "e")
    rep = res["reproduction"]
    assert rep["rows_compared"] == rep["rows_expected"] == 27
    assert all(rep[k] == 0 for k in rep if k.startswith(("max_abs", "nan_")))
    st = pd.read_csv(out / "stacking.csv")
    assert len(st) == 2 and set(st["best_single"]) <= {"x_volatility", "s_efficiency"}
    assert {"helps_unlimited", "stacks", "candidates", "best_single"} <= set(res)
