"""A41 landscape pieces (lab/aa_landscape.py): sampling, moves, metrics, surrogate, Sobol indices, centres, rules."""
import numpy as np
import pandas as pd

from lab import aa_landscape as LS


def test_rqmc_is_in_the_unit_cube_and_reproducible():
    a, b = LS.rqmc(1000, 26, seed=1), LS.rqmc(1000, 26, seed=1)
    assert a.shape == (1000, 26) and (a >= 0).all() and (a < 1).all() and np.array_equal(a, b)
    assert np.allclose(a.mean(axis=0), 0.5, atol=0.03)  # evenly spread in every dimension
    assert not np.array_equal(a, LS.rqmc(1000, 26, seed=2))


def test_weights_are_whole_percents_with_a_floor_and_every_set_appears():
    U, W = LS.sample(4096)
    assert W.shape == (4096, LS.D) and (W.sum(axis=1) == 100).all()
    held = W > 0
    assert (W[held] >= LS.MIN_PCT).all() and held.any(axis=1).all()
    assert abs(held.mean() - 0.5) < 0.03  # inclusion probability 0.5 per product
    # inclusion follows the coordinate
    assert np.array_equal(held[np.any(U[:, :LS.D] < 0.5, axis=1)], (U[:, :LS.D] < 0.5)[np.any(U[:, :LS.D] < 0.5, axis=1)])


def test_neighbours_exchanges_and_drop_one():
    w = np.zeros(LS.D, np.int64)
    w[[0, 5, 6]] = [30, 30, 40]
    nb = LS.neighbours(w, 5)
    assert (nb.sum(axis=1) == 100).all() and not (nb == w).all(axis=1).any()
    assert len(nb) == 3 * 2 + (LS.D - 3)  # ±5 on the three held, +5 % on each missing one
    assert any(v[0] == 35 for v in nb) and any(v[0] == 25 for v in nb) and any(v[1] == 5 for v in nb)
    V, pairs = LS.exchanges(w, 5)
    assert len(V) == 3 * (LS.D - 1) and (V.sum(axis=1) == 100).all()
    i, j = pairs[0]
    assert V[0][i] == w[i] - 5 and V[0][j] == w[j] + 5
    pro = dict(LS.drop_one(w, "pro"))
    assert pro[0][0] == 0 and pro[0].sum() == 100 and pro[0][5] == 43 and pro[0][6] == 57
    cd = dict(LS.drop_one(w, "cd"))
    assert 6 not in cd and cd[0][6] == 70


def test_path_metrics_start_values_match_a_direct_calculation():
    dates = pd.bdate_range("2000-09-01", "2024-12-31")
    rng = np.random.default_rng(0)
    P = np.cumprod(1 + rng.normal(3e-4, 0.01, (3, len(dates))), axis=1)
    lay = LS.layout(dates, {"dotcom": LS.DOTCOM})
    M = LS.path_metrics(P, np.zeros(3), lay)
    s = lay.starts
    T = len(dates)
    direct = (P[:, [-1]] / P[:, s]) ** (252 / (T - 1 - s)) - 1
    assert np.allclose(M["rank"], np.percentile(direct, 10, axis=1))
    assert dates[s[0]] <= pd.Timestamp("2000-09-30") and dates[s[-1]] <= pd.Timestamp("2015-12-31")
    a, b = lay.win_starts[0], lay.win_ends[0]
    sub = P[:, a:b + 1]
    assert np.isclose(M["win_worst"][0], min(((P[k, a2:b2 + 1] / np.maximum.accumulate(P[k, a2:b2 + 1])) - 1).min()
                                             for k, (a2, b2) in [(0, ab) for ab in zip(lay.win_starts, lay.win_ends)]))
    assert sub.shape[1] > 2400 and list(LS.zone(np.array([-0.1, -0.16, -0.2]))) == ["안", "경계", "밖"]


def test_scheffe_fits_a_quadratic_mixture_exactly_and_its_gradient():
    _, W = LS.sample(3000)
    X = W / 100.0
    y = 0.1 * X[:, 0] - 0.05 * X[:, 3] + 0.3 * X[:, 0] * X[:, 5]
    fit = LS.scheffe_fit(W, y)
    assert fit["r2_test"] > 0.999999
    g = LS.scheffe_gradient(fit["beta"], W[:5])
    assert np.allclose(g[:, 0], (0.1 + 0.3 * X[:5, 5]) / 100, atol=1e-8)


def test_sobol_indices_find_the_only_product_that_matters():
    A, B, mixed = LS.saltelli(2048)
    f = lambda U: LS.weights_from_u(U)[:, 0].astype(float)  # noqa: E731 - output = weight of product 0
    t = LS.sobol_indices(f(A), f(B), [f(M) for M in mixed], boot=20)
    assert t.loc[0, "ST"] > 0.3 and t.loc[0, "ST"] == t["ST"].max()
    assert (t.loc[1:, "S1"].abs() < 0.05).all()


def test_flat_region_and_centres():
    rng = np.random.default_rng(1)
    _, W = LS.sample(2000)
    df = pd.DataFrame({"mdd": rng.uniform(-0.2, -0.05, 2000), "win_worst": -0.1, "rank": rng.uniform(0.05, 0.08, 2000)})
    share = pd.Series(0.9, index=df.index)
    reg = LS.flat_region(df, share)
    assert len(reg) and (reg["mdd"] >= LS.LINE).all() and reg["rank"].min() >= reg["rank"].max() - LS.FLAT_PP
    near = LS.flat_region(df, share, near=True)
    assert ((near["mdd"] < LS.LINE) & (near["mdd"] >= LS.NEAR)).all()
    assert LS.flat_region(df, share * 0.5).empty  # neighbour share below 80 %
    W2 = np.zeros((10, LS.D), np.int64)
    W2[:, 0], W2[:, 5], W2[:, 6] = 40, 30, 30
    W2[5:, 6], W2[5:, 7] = 25, 5
    cs = LS.centres(W2, pd.DataFrame({"rank": np.ones(10)}, index=np.arange(10)))
    assert len(cs) == 1 and cs[0]["점 수"] == 10 and cs[0]["비중"].sum() == 100 and list(cs[0]["비중"][[0, 5, 6, 7]]) == [40, 30, 28, 2]


def test_rule_targets_use_only_past_data():
    idx = pd.bdate_range("2001-01-01", periods=600)
    rng = np.random.default_rng(2)
    rets = pd.DataFrame({"379810": rng.normal(0, 0.02, 600), "459580": rng.normal(1e-4, 1e-5, 600),
                         "KRXGOLD": rng.normal(0, 0.01, 600)}, index=idx)
    t = LS.rule_targets(rets, "invvol", list(rets.columns), np.array([10, 400]))
    assert np.isclose(sum(t[10].values()), 1) and np.allclose(list(t[10].values()), 1 / 3)  # too little history: equal
    w = t[400]
    assert np.isclose(w["459580"], 0.30) and w["KRXGOLD"] > w["379810"]
    changed = rets.copy()
    changed.iloc[401:] *= 10
    assert LS.rule_targets(changed, "invvol", list(rets.columns), np.array([400]))[400] == w  # the future does not matter
    rc = LS.rule_targets(rets, "riskcap", list(rets.columns), np.array([400]))[400]
    assert rc["379810"] < 1 / 3 and np.isclose(sum(rc.values()), 1)
