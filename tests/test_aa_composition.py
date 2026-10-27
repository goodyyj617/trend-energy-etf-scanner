"""Allocation lab: is the B2–B5 composition itself good? (DECISIONS A39, lab/aa_composition.py)."""
import numpy as np
import pandas as pd
import pytest

from lab import aa_composition as C
from lab import aa_search_results as sr
from lab.aa_assets import PRESETS


def test_centres_are_the_registered_and_stage2_allocations():
    assert C.CENTRES["B2"] == PRESETS["gs_b2"].weights and C.CENTRES["B4"] == PRESETS["gs_b4"].weights
    s2 = pd.read_csv(sr.ROOT / "docs/research/allocation/2026-10-26_stage2_real/stage2.csv").set_index("후보")
    assert sr.parse_names(s2.loc["B 레버·가상 없음 (-11,-10]", "비중"]) == C.CENTRES["B5"]
    for w in C.CENTRES.values():
        assert sum(w.values()) == 100 and w[C.NASDAQ] > 0


def test_whole_keeps_small_parts_and_breaks_ties_by_the_larger_weight():
    w = C.whole(np.array([28.5, 4.5, 0.5, 66.5]))  # four remainders of .5, two to hand out
    assert w.tolist() == [29, 4, 0, 67] and w.sum() == 100
    assert C.whole(np.array([1.0, 1.0, 1.0])).tolist() == [34, 33, 33]
    with pytest.raises(ValueError):
        C.whole(np.zeros(3))


def test_b_average_is_the_text_in_the_design():
    text = ("미국나스닥100 29 · 미국S&P500필수소비재 8 · 미국S&P500헬스케어 1 · MSCI선진국 3 · 200 1 · 미국10년국채액티브(H) 5 · "
            "미국30년국채액티브(H) 1 · iShares미국인플레이션국채액티브 2 · iShares미국투자등급회사채액티브 3 · ACE KRX금현물 7 · "
            "골드선물(H) 13 · CD금리액티브(합성) 23 · 미국달러선물 4")
    assert sr.describe(C.b_average()) == text


@pytest.mark.parametrize("name", C.NAMES)
def test_leave_one_out_removes_one_product_and_keeps_the_total(name):
    w = C.vector(C.CENTRES[name])
    pro = C.leave_one_out(w, "pro")
    assert [code for code, _ in pro] == [s for s in sr.SYMBOLS if s in C.CENTRES[name]]
    for code, v in pro:
        assert v.sum() == 100 and v[C.COL[code]] == 0
        assert set(np.flatnonzero(v)) <= set(np.flatnonzero(w))
    for code, v in C.leave_one_out(w, "cd"):
        assert code != C.CD and v.sum() == 100 and v[C.COL[code]] == 0
        assert v[C.COL[C.CD]] == w[C.COL[C.CD]] + w[C.COL[code]]  # whole percents in, whole percents out
        others = [i for i in np.flatnonzero(w) if sr.SYMBOLS[i] not in (code, C.CD)]
        assert (v[others] == w[others]).all()
    with pytest.raises(ValueError):
        C.leave_one_out(w, "x")


def test_swap_keeps_the_row_nasdaq_and_the_column_products():
    cells = C.swap_matrix()
    assert len(cells) == 16
    for r, c, v in cells:
        assert v.sum() == 100 and v[C.COL[C.NASDAQ]] == C.CENTRES[r][C.NASDAQ]
        held = {sr.SYMBOLS[i] for i in np.flatnonzero(v)} - {C.NASDAQ}
        assert held <= set(C.CENTRES[c]) - {C.NASDAQ}
        if r == c:
            assert sr.weights_of(v) == C.CENTRES[r]
    v = dict(((r, c), v) for r, c, v in cells)[("B2", "B4")]
    assert sr.weights_of(v) == {C.NASDAQ: 19, "453630": 13, "468630": 9, "411060": 17, C.CD: 42}


def test_swap_verdict():
    rows = [(r, c, 0.10, -0.12) for r in C.NAMES for c in C.NAMES]
    t = pd.DataFrame(rows, columns=["행", "열", "cagr", "mdd"])
    assert C.swap_verdict(t)["판정"] == "바꿔 써도 됨"
    t.loc[(t["행"] == "B2") & (t["열"] == "B5"), "cagr"] = 0.094
    v = C.swap_verdict(t)
    assert v["판정"] == "나머지에 의미 있음" and v["0.5%p 넘게 낮음"] == 1 and v["선 밖"] == 0
    t.loc[(t["행"] == "B2") & (t["열"] == "B2"), "mdd"] = -0.20  # the diagonal is never judged
    assert C.swap_verdict(t)["선 밖"] == 0


@pytest.mark.parametrize("dc, dm, label", [
    ((-0.004, -0.003), (0.0, 0.0), "제 몫"),
    ((0.0, -0.004), (-0.006, 0.0), "제 몫"),
    ((0.001, -0.001), (0.002, -0.004), "있으나 마나"),
    ((0.004, 0.0), (0.0, 0.0), "빼면 나아짐"),
    ((-0.004, 0.004), (0.0, 0.0), "기간 따라 다름"),
    ((-0.004, 0.0), (0.0, 0.0), "기간 따라 다름"),
])
def test_loo_verdict(dc, dm, label):
    assert C.loo_verdict(dc, dm) == label


def test_skeleton_grid_and_simple_verdict():
    grid = C.skeleton_grid()
    assert len(grid) == 24 and all(v.sum() == 100 and (v > 0).sum() == 3 for _, v in grid)
    assert grid[0][0] == "나스닥100 20 · KRX금 10 · CD금리 70"
    assert C.simple_verdict(0.093, -0.14, [0.0928, 0.11]) == "같은 일을 한다"
    assert C.simple_verdict(0.092, -0.14, [0.0928, 0.11]) == "못 미침"
    assert C.simple_verdict(0.12, -0.151, [0.0928]) == "못 미침"


def test_subset_b_and_reselect():
    W = np.zeros((3, len(sr.SYMBOLS)), dtype=int)
    W[0, C.COL[C.NASDAQ]] = 100
    W[1, C.COL["409820"]] = 100  # leveraged nasdaq
    W[2, C.COL["SYN_IEF_U"]] = 100
    assert C.subset_b(W).tolist() == [True, False, False]
    df = pd.DataFrame({"cagr_h1": [0.10, 0.20, 0.15, 0.30], "mdd_h1": [-0.10, -0.14, -0.149, -0.16],
                       "cagr_h2": [0.0] * 4, "mdd_h2": [0.0] * 4})
    assert C.reselect(df, "h1", "h2", top=2).tolist() == [1, 2]  # the 30 % one is outside the line
    with pytest.raises(KeyError):
        C.reselect(df, "h1", "h3")


def test_role_shares_cover_every_weight_and_the_reselect_verdict():
    W = np.vstack([C.vector(v) for v in C.CENTRES.values()])
    shares = C.role_shares(W)
    assert (shares.sum(axis=1) == 100).all()
    assert shares["나스닥"].tolist() == [19, 33, 30, 32] and shares["금"].tolist() == [35, 12, 15, 17]
    assert shares.loc[0, "방어 주식"] == 25 and shares.loc[3, "현금·달러"] == 47
    assert C.reselect_verdict(shares) == "다시 뽑힘"
    assert C.reselect_verdict(shares.assign(금=5)) == "다른 구성이 뽑힘"


def test_model_flags_read_the_a33_table():
    p = pd.read_csv(sr.REAL_DIR / "products.csv")
    f = C.model_flags(p, C.products_of()).set_index("코드")
    assert len(f) == 13 and set(f.index) == {s for v in C.CENTRES.values() for s in v}
    assert f.loc[C.NASDAQ, "판정"] == "괜찮음"  # +0.5 %p a year, monthly correlation 0.98
    assert f.loc["453630", "판정"] == "모형 경고"  # 필수소비재 +2.0 %p a year
    assert f.loc["411060", "판정"] == "모형 경고"  # KRX gold: correlation 0.87
    short = C.model_flags(p.assign(years=0.5), [C.NASDAQ]).iloc[0]
    assert short["판정"] == "비교 불가"
    assert C.model_flags(p, ["SYN_IEF_U"]).iloc[0]["판정"] == "자료 없음"  # synthetic rows are not products


# --------------------------------------------------------------------------- the script on made-up prices

def test_composition_script_runs_on_made_up_prices(tmp_path, monkeypatch):
    import json
    import sys

    sys.path.insert(0, str(sr.ROOT / "scripts"))
    import aa_composition_study as S
    from test_aa_local import _synthetic_krw

    krw, krwx = _synthetic_krw(), _synthetic_krw(start="2000-06-01")
    monkeypatch.setattr(S.G, "load", lambda prices="model": krw)
    monkeypatch.setattr(S.L, "load_extended", lambda prices: (krwx, pd.DataFrame({"대용": ["IEF"]})))
    S.main(["--quick", "--prices", "model", "--out", str(tmp_path)])
    for f in ("model_flags.csv", "named.csv", "loo.csv", "swap.csv", "simple.csv", "reselect_top.csv.gz",
              "reselect_ranks.csv", "summary.json", "dotcom_proxies.csv"):
        assert (tmp_path / f).exists(), f
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert summary["engine_check_max_rel"] < 1e-6 and summary["engine_check_max_rel_2000"] < 1e-6
    named = pd.read_csv(tmp_path / "named.csv")
    assert list(named["배분"][:5]) == [*C.NAMES, PRESETS["stable40"].name]
    assert named["mdd_2000"].notna().all() and named["dd_dotcom_2000"].notna().all()
    loo = pd.read_csv(tmp_path / "loo.csv")
    n_held = sum(len(v) for v in C.CENTRES.values())
    assert (loo["방식"] == "pro").sum() == n_held and (loo["방식"] == "cd").sum() == n_held - 3  # B3–B5 hold CD금리
    assert set(loo["판정"]) <= {"제 몫", "있으나 마나", "빼면 나아짐", "기간 따라 다름"}
    sw = pd.read_csv(tmp_path / "swap.csv")
    assert len(sw) == 16
    simple = pd.read_csv(tmp_path / "simple.csv")
    assert (simple["배분"] == "뼈대").sum() == 24 and (simple["배분"] == "B 평균").sum() == 1
    top = pd.read_csv(tmp_path / "reselect_top.csv.gz")
    assert set(top["고른 기간"]) == {"2005–2015에서 고름", "2016–2026에서 고름"}
    ranks = pd.read_csv(tmp_path / "reselect_ranks.csv")
    assert list(ranks["배분"]) == [*C.NAMES, PRESETS["stable40"].name] and (ranks["h1 순위"] >= 1).all()
