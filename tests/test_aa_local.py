"""Allocation lab: the local structure study around B2 · B3 · B4 (DECISIONS A37)."""
import numpy as np
import pytest

from lab import aa_local as loc
from lab import aa_search_results as sr
from lab.aa_assets import PRESETS


def test_universe_is_the_b_subset_without_kodex200():
    assert len(loc.UNIVERSE) == 21
    assert loc.KODEX200 not in loc.UNIVERSE
    assert not set(loc.UNIVERSE) & set(sr.SYNTHETIC)
    assert not set(loc.UNIVERSE) & set(sr.CLASSES[sr.LEV_CLASS])


def test_centres_match_the_registered_presets_and_satisfy_the_rules():
    assert loc.CENTRES["B2"] == PRESETS["gs_b2"].weights
    assert loc.CENTRES["B4"] == PRESETS["gs_b4"].weights
    W = np.vstack([loc.vector(w) for w in loc.CENTRES.values()])
    assert (W.sum(axis=1) == 100).all()
    assert loc.rules(W)["all"].all()


def test_rules_catch_each_broken_feature():
    base = loc.vector(loc.CENTRES["B4"])  # nasdaq 30, staples 11, corp 8, gold 15, CD 36
    no_nasdaq = loc.ablate(base, "X1")
    no_gold = loc.ablate(base, "X2")
    long_bonds = loc.ablate(base, "X3")
    r = loc.rules(np.vstack([no_nasdaq, no_gold, long_bonds]))
    assert not r["R1"].iloc[0] and r["R2"].iloc[0]
    assert not r["R2"].iloc[1] and r["R1"].iloc[1]
    assert not r["R3"].iloc[2]


def test_family_samples_obey_the_rules_and_are_distinct():
    W = loc.sample_family(400, seed=3)
    assert W.shape == (400, len(sr.SYMBOLS))
    assert (W.sum(axis=1) == 100).all()
    assert ((W == 0) | (W >= loc.MIN_PCT)).all()
    assert ((W > 0).sum(axis=1) <= loc.MAX_PRODUCTS).all()
    assert loc.rules(W)["all"].all()
    assert len({w.tobytes() for w in W}) == len(W)
    # the family is broad: nasdaq alone, mixed equity, both gold products, with and without long bonds
    col = loc.COL
    assert (W[:, col[loc.LONG]] == 0).any() and (W[:, col[loc.LONG]] > 0).any()
    assert ((W[:, col["411060"]] > 0) & (W[:, col["132030"]] > 0)).any()
    assert np.array_equal(loc.sample_family(50, seed=3), W[:50])


def test_control_uses_only_universe_products():
    W = loc.sample_control(40, 3, seed=1)
    assert (W.sum(axis=1) == 100).all()
    outside = [loc.COL[s] for s in sr.SYMBOLS if s not in loc.UNIVERSE]
    assert (W[:, outside] == 0).all()


@pytest.mark.parametrize("kind", ["X1", "X2", "X3", "X4"])
def test_ablations_keep_the_total_and_move_only_their_feature(kind):
    b2 = loc.vector(loc.CENTRES["B2"])
    v = loc.ablate(b2, kind)
    assert v is not None and v.sum() == 100 and (v >= 0).all()
    col = loc.COL
    if kind == "X1":
        assert v[col[loc.NASDAQ]] == 0 and v[col[loc.SP500]] == 19
    if kind == "X2":
        assert v[[col[s] for s in loc.GOLD]].sum() == 0
        assert v[col[loc.TEN]] + v[col[loc.LONG]] + v[col[loc.CORP]] == 10 + 6 + 5 + 35
        assert v[col[loc.TEN]] == 27  # 35 spread over 10 / 6 / 5 → 16.7 / 10.0 / 8.3
    if kind == "X3":
        assert v[col[loc.TEN]] == 0 and v[col[loc.LONG]] == 16
    if kind == "X4":
        assert v[[col[s] for s in loc.SECTORS]].sum() == 0 and v[col[loc.SP500]] == 25
    untouched = {"X1": [loc.NASDAQ, loc.SP500], "X2": [*loc.GOLD, loc.TEN, loc.LONG, loc.CORP],
                 "X3": [loc.TEN, loc.CD, loc.LONG], "X4": [*loc.SECTORS, loc.SP500]}[kind]
    others = [col[s] for s in sr.SYMBOLS if s not in untouched]
    assert np.array_equal(v[others], b2[others])


def test_ablation_is_none_when_the_feature_is_missing():
    b3 = loc.vector(loc.CENTRES["B3"])  # no defensive sector
    assert loc.ablate(b3, "X4") is None
    with pytest.raises(ValueError):
        loc.ablate(b3, "X9")


def test_perturb_gives_distinct_neighbours_over_the_distance_range():
    c = loc.vector(loc.CENTRES["B3"])
    W = loc.perturb(c, 600, seed=5)
    assert (W.sum(axis=1) == 100).all()
    assert ((W == 0) | (W >= loc.MIN_PCT)).all()
    d = loc.l1(W, c)
    assert (d > 0).all()
    assert d.min() <= 6 and d.max() >= 25
    assert len({w.tobytes() for w in W}) == len(W)
    outside = [loc.COL[s] for s in sr.SYMBOLS if s not in loc.UNIVERSE]
    assert (W[:, outside] == 0).all()
    bins = loc.dist_bin(d)
    assert {"0–5", "5–10", "10–20", "20–30"} <= set(bins)


def test_substitutions_swap_whole_weights_between_siblings():
    c = loc.vector(loc.CENTRES["B4"])
    subs = dict(loc.substitutions(c))
    v = subs[f"{sr.name_of('411060')} → {sr.name_of('132030')}"]
    assert v[loc.COL["132030"]] == 15 and v[loc.COL["411060"]] == 0
    assert all(w.sum() == 100 for w in subs.values())
    assert 5 <= len(subs) <= 20


def test_dist_bins():
    assert list(loc.dist_bin(np.array([2, 5, 6, 10, 11, 30, 31]))) == ["0–5", "0–5", "5–10", "5–10", "10–20",
                                                                       "20–30", "30+"]


def test_block_bootstrap_mdd_is_reproducible_and_keeps_a_flat_path_flat():
    rng = np.random.default_rng(0)
    T = 1000
    flat = np.cumprod(np.full(T, 1.0001))
    noisy = np.cumprod(1 + rng.normal(0, 0.01, T))
    P = np.vstack([flat, noisy])
    a = loc.block_bootstrap_mdd(P, n=20, block=50, seed=4)
    b = loc.block_bootstrap_mdd(P, n=20, block=50, seed=4)
    assert a.shape == (2, 20)
    np.testing.assert_array_equal(a, b)
    assert np.allclose(a[0], 0.0)
    assert (a[1] < 0).all()
    # one block as long as the history: a rotation of the same returns
    full = loc.block_bootstrap_mdd(P[1:], n=5, block=T, seed=1)
    assert full.shape == (1, 5)


# --------------------------------------------------------------------------- the script, end to end on made-up prices

def _synthetic_krw(start="2004-12-01"):
    import pandas as pd

    from lab.data import Panel

    dates = pd.bdate_range(start, "2026-10-20")
    rng = np.random.default_rng(7)
    vol = {s: 0.004 + 0.012 * (sr.CLASS_OF[s] in sr.EQUITY_CLASSES) for s in sr.SYMBOLS}
    close = pd.DataFrame({s: 100 * np.cumprod(1 + rng.normal(0.0002, vol[s], len(dates))) for s in sr.SYMBOLS}, index=dates)
    panel = Panel(open=close, high=close, low=close, close=close, volume=close * 0 + 1, tradable=list(sr.SYMBOLS))
    panel.cash = pd.Series(0.02 / 252, index=dates)
    panel.benchmark = "379800"  # as krw_panel sets it
    return panel


def test_local_study_script_runs_on_a_quick_setting(tmp_path, monkeypatch):
    import json
    import sys
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import aa_local_study as S

    krw = _synthetic_krw()
    monkeypatch.setattr(S.G, "load", lambda prices="model": krw)
    S.main(["--quick", "--prices", "model", "--out", str(tmp_path)])
    for f in ("neighbours.csv.gz", "neighbour_bins.csv", "substitutions.csv", "structure.csv.gz", "groups.csv",
              "ablations.csv", "halves.csv", "bootstrap.csv", "bootstrap_mdd.csv.gz", "crisis.csv", "summary.json"):
        assert (tmp_path / f).exists(), f
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["engine_check_max_rel"] < 1e-6
    groups = pd.read_csv(tmp_path / "groups.csv")
    assert list(groups["묶음"][:6]) == ["구조 가족", "X1", "X2", "X3", "X4", "무작위"]
    assert groups.loc[0, "수"] == S.QUICK["n_family"]
    nb = pd.read_csv(tmp_path / "neighbours.csv.gz")
    assert (nb["종류"] == "중심").sum() == 3 and (nb["종류"] == "이웃").sum() == 3 * S.QUICK["n_nb"]
    boot = pd.read_csv(tmp_path / "bootstrap.csv")
    assert list(boot["이름"][:3]) == ["B2", "B3", "B4"]
    assert boot["선 넘을 확률"].between(0, 1).all()


# --------------------------------------------------------------------------- ④ dot-com extension

def _usd_panel(cols: dict, dates):
    import pandas as pd

    from lab.data import Panel

    close = pd.DataFrame(cols, index=dates)
    return Panel(open=close, high=close, low=close, close=close, volume=close * 0 + 1, tradable=list(cols))


def test_extend_history_splices_the_best_tracking_substitute():
    import pandas as pd

    dates = pd.bdate_range("1999-01-04", "2008-12-31")
    rng = np.random.default_rng(3)
    base = rng.normal(0.0002, 0.005, len(dates))
    good = 50 * np.cumprod(1 + base + rng.normal(0, 0.0005, len(dates)))
    bad = 80 * np.cumprod(1 + rng.normal(0.0002, 0.005, len(dates)))
    ief = 100 * np.cumprod(1 + base)
    first = dates.get_loc(pd.Timestamp("2002-07-30"))
    ief[:first] = np.nan
    usd = _usd_panel({"IEF": ief, "SPY": 100 * np.cumprod(1 + rng.normal(0, 0.01, len(dates)))}, dates)
    subs = _usd_panel({"BAD": bad, "GOOD": good}, dates)
    ext, table = loc.extend_history(usd, subs, "2000-09-01", candidates={"IEF": ("BAD", "GOOD"), "SPY": ("BAD",)})
    row = table.set_index("대용").loc["IEF"]
    assert row["이어 붙인 상품"] == "GOOD" and row["월 상관"] > 0.9
    assert table.set_index("대용").loc["SPY", "비고"] == "시작 전부터 있음"
    s = ext.close["IEF"]
    assert s.first_valid_index() == dates[0]
    np.testing.assert_allclose(s.iloc[first:].to_numpy(), ief[first:])  # the proxy's own days are untouched
    r_ext = s.iloc[1:first + 1].pct_change(fill_method=None).dropna().to_numpy()
    r_sub = pd.Series(good, index=dates).iloc[1:first + 1].pct_change(fill_method=None).dropna().to_numpy()
    np.testing.assert_allclose(r_ext, r_sub, rtol=1e-9)  # before the splice it moves like the substitute
    assert usd.close["IEF"].isna().sum() == first  # the input panel is not changed


def test_extend_history_reports_a_missing_substitute():
    import pandas as pd

    dates = pd.bdate_range("1999-01-04", "2004-12-31")
    gld = np.full(len(dates), 40.0)
    gld[:dates.get_loc(pd.Timestamp("2004-11-18"))] = np.nan
    usd = _usd_panel({"GLD": gld}, dates)
    subs = _usd_panel({"OTHER": np.ones(len(dates))}, dates)
    ext, table = loc.extend_history(usd, subs, "2000-09-01", candidates={"GLD": ("GC=F",)})
    assert table.loc[0, "이어 붙인 상품"] == "" and "GC=F" in table.loc[0, "비고"]
    assert ext.close["GLD"].first_valid_index() == pd.Timestamp("2004-11-18")


def test_dotcom_part_runs_on_made_up_prices(tmp_path):
    import json
    import sys
    from pathlib import Path

    import pandas as pd

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
    import aa_local_study as S

    krw = _synthetic_krw(start="2000-06-01")
    S.dotcom_study(S.QUICK, "model", tmp_path, loader=lambda prices: (krw, pd.DataFrame({"대용": ["IEF"]})))
    named = pd.read_csv(tmp_path / "dotcom.csv")
    assert list(named["이름"][:3]) == ["B2", "B3", "B4"] and len(named) == 3 + len(S.DOTCOM_COMPARE)
    assert (named["mdd"] <= named["mdd_2005"] + 1e-12).all()  # the longer history can only add drawdown
    groups = pd.read_csv(tmp_path / "dotcom_groups.csv")
    assert list(groups["묶음"]) == ["구조 가족", "X1", "X2", "X3", "X4", "무작위"]
    curves = pd.read_csv(tmp_path / "dotcom_paths.csv.gz", parse_dates=["date"])
    assert curves["date"].min() >= pd.Timestamp("2000-09-01") and curves["date"].max() <= pd.Timestamp("2006-12-31")
    summary = json.loads((tmp_path / "dotcom_summary.json").read_text(encoding="utf-8"))
    assert summary["engine_check_max_rel"] < 1e-6
