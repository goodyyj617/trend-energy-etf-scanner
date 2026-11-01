"""A42 stage 1: factor betas, exposures, scenarios (lab/aa_exposure.py) on synthetic data, and the recorded chart data."""
import numpy as np
import pandas as pd

from lab import aa_experiments as ex
from lab import aa_exposure as E


def synthetic(n=120, seed=0):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2005-01-31", periods=n, freq="ME")
    F = pd.DataFrame({"EQ": rng.normal(0.007, 0.04, n), "GROWTH": rng.normal(0, 0.02, n), "UST": rng.normal(0, 20, n),
                      "KTB": rng.normal(0, 15, n), "FX": rng.normal(0, 0.03, n), "GOLD": rng.normal(0.005, 0.045, n)}, index=idx)
    true = {"A": [1.0, 1.0, 0, 0, 1.0, 0], "B": [0, 0, -0.0008, 0, 0, 0], "C": [0, 0, 0, 0, 1.0, 1.0]}
    R = pd.DataFrame({k: F[list(E.FACTORS)].to_numpy() @ np.array(v) + rng.normal(0, 0.001, n) for k, v in true.items()}, index=idx)
    return F, R, true


def test_betas_recover_the_true_loadings():
    F, R, true = synthetic()
    b = E.betas(R, F, window=("2005-01-01", "2014-12-31"))
    for k, v in true.items():
        assert np.allclose(b.loc[k, list(E.FACTORS)].to_numpy(float), v, atol=0.01), k
        assert b.loc[k, "개월"] == 120
    assert abs(b.loc["B", "UST"] + 0.0008) < 2e-5  # rate beta per bp (duration ≈ 8)
    short = E.betas(R.iloc[:30], F)
    assert short[list(E.FACTORS)].isna().all().all()  # fewer than 36 months: no betas


def test_exposure_and_scenario_arithmetic():
    F, R, _ = synthetic()
    b = E.betas(R, F, window=("2005-01-01", "2014-12-31"))
    e = E.exposure({"A": 50, "C": 50}, b)
    assert np.isclose(e["EQ"], 0.5, atol=0.01) and np.isclose(e["FX"], 1.0, atol=0.01) and np.isclose(e["GOLD"], 0.5, atol=0.01)
    v = E.shock_vector(E.SCENARIOS["S2"][1])
    assert v["FX"] == -0.25 and v["UST"] == 0.0
    assert np.isclose(E.shock_vector(E.SCENARIOS["S4"][1])["UST"], 300.0)  # bp stay in bp
    loss = E.scenario_losses(e)
    assert np.isclose(loss["S2"], -0.25, atol=0.005) and np.isclose(loss["S3"], 0.5 * -0.40, atol=0.005)
    assert E.scenario_verdict(pd.Series([-0.3, -0.26, -0.1] + [0] * 5)) and not E.scenario_verdict(pd.Series([-0.3] * 3 + [0] * 5))
    assert set(E.SCENARIOS) == {f"S{k}" for k in range(1, 9)}


def test_a42_stage1_chart_data():
    s = ex.a42_1_scenarios()
    assert len(s) == 6 * 8 and set(s["시나리오"]) == set(E.SCENARIOS)
    verdict = s.drop_duplicates("배분").set_index("배분")["판정"]
    assert not verdict["S&P500"] and verdict[["(가)", "안정형 40", "B2", "B4", "현재 보유"]].all()
    s1 = s[s["시나리오"] == "S1"].set_index("배분")["손실"]
    assert round(s1["(가)"], 1) == -27.0 and round(s1["S&P500"], 1) == -39.8
    c = ex.a42_1_check()
    sp = c[(c["배분"] == "S&P500") & (c["위기"] == "2008")].iloc[0]
    assert round(sp["실제"], 1) == -24.8 and abs(sp["로그 추정"] - sp["실제"]) < abs(sp["선형 추정"] - sp["실제"])
