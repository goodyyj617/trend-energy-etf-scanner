"""Allocation lab: reading the global search results (DECISIONS A31) for the app page '전역 탐색 결과'."""
import json

import numpy as np
import pandas as pd
import pytest

from lab import aa_search_results as sr
from lab.aa_assets import PRODUCTS


def weights_row(**pct) -> np.ndarray:
    w = np.zeros(len(sr.SYMBOLS), dtype=int)
    for code, v in pct.items():
        w[sr.SYMBOLS.index(code)] = v
    return w


# --------------------------------------------------------------------------- universe and weight text

def test_universe_is_the_recorded_a31_universe():
    summary = json.loads((sr.RESULTS_DIR / "summary.json").read_text(encoding="utf-8"))
    assert sr.SYMBOLS == summary["params"]["universe"] and len(sr.SYMBOLS) == 30
    assert set(sr.SYNTHETIC) == {"SYN_IEF_U", "SYN_TLT_U"}
    assert all(s in PRODUCTS for s in sr.SYMBOLS if s not in sr.SYNTHETIC)
    assert len({sr.name_of(s) for s in sr.SYMBOLS}) == 30  # names are unique, so the display text can be read back


def test_weight_text_round_trips_through_codes_and_names():
    w = weights_row(**{"379810": 30, "411060": 25, "SYN_IEF_U": 15, "0091C0": 30})
    expected = {"379810": 30, "0091C0": 30, "SYN_IEF_U": 15, "411060": 25}
    assert sr.parse_codes(sr.code_text(w)) == expected
    assert sr.parse_names(sr.describe(w)) == expected
    assert "미국10년 환노출*" in sr.describe(w)


def test_parse_names_rejects_an_unknown_product():
    with pytest.raises(ValueError, match="모르는 상품"):
        sr.parse_names("없는상품 50 · ACE KRX금현물 50")


def test_class_shares_and_flags():
    W = np.vstack([weights_row(**{"379800": 60, "0091C0": 30, "SYN_IEF_U": 10}),
                   weights_row(**{"409820": 20, "411060": 80})])
    shares = sr.class_shares(W)
    assert shares.loc[0, "주식"] == 60 and shares.loc[0, "채권"] == 40 and shares.loc[0, "상품 수"] == 3
    assert shares.loc[0, "환노출"] == 70  # 379800 and the synthetic treasury are unhedged
    assert shares.loc[1, "레버리지"] == 20
    df = pd.DataFrame({"mdd_h1": [-0.10, -0.20], "mdd_h2": [-0.1, -0.1], "dd_2008": [-0.1, -0.1], "dd_2020": [-0.16, -0.1],
                       "mdd_holdout": [-0.1, -0.1], "roll_share": [np.nan, 0.5], "nb_worst_mdd": [np.nan, -0.1],
                       "timing_worst": [-0.25, np.nan]})
    assert sr.flags(df).tolist() == ["위기 날짜", "반기 창"]  # missing information is not a flag


# --------------------------------------------------------------------------- app tables

def full_table() -> pd.DataFrame:
    rng = np.random.default_rng(1)
    rows = []
    for i in range(60):
        w = weights_row(**{"379800": 50, "411060": 30, "459580": 20}) if i % 2 else weights_row(**{"379810": 70, "0091C0": 30})
        rows.append({"cagr": rng.uniform(0.03, 0.12), "mdd": -0.05 - 0.004 * i, "sharpe_excess": 0.5, "calmar": 0.6,
                     "turnover": 0.1, "출처": "계층" if i % 3 else "희소", "비중": sr.describe(w), "roll_share": np.nan})
    return pd.DataFrame(rows)


def test_app_tables_keep_every_passing_row_and_a_sample_of_the_rest():
    full = full_table()
    passing, cloud = sr.app_tables(full, cloud_n=10, seed=3)
    assert len(passing) == int((full["mdd"] >= sr.MDD_LINE).sum()) and (passing["mdd"] >= sr.MDD_LINE).all()
    assert "w" in passing.columns and "비중" not in passing.columns
    assert sr.parse_codes(passing["w"].iloc[0]) == sr.parse_names(full.loc[passing.index[0], "비중"])
    assert len(cloud) == 10 and list(cloud.columns) == ["cagr", "mdd"] and (cloud["mdd"] < sr.MDD_LINE).all()


def test_load_results_on_a_small_folder(tmp_path):
    full = full_table()
    passing, cloud = sr.app_tables(full, cloud_n=5, seed=3)
    passing.to_csv(tmp_path / "passing.csv.gz", index=False)
    cloud.to_csv(tmp_path / "cloud.csv.gz", index=False)
    refined = full.iloc[:3].assign(출처="다듬음", seed_rank=[1, 2, 3])
    refined.to_csv(tmp_path / "top300_refined.csv", index=False)
    presets = full.iloc[:2].assign(이름=["가", "나"], 출처="프리셋")
    presets.to_csv(tmp_path / "presets.csv", index=False)
    pd.DataFrame({"후보": ["A"], "비중": [full.loc[0, "비중"]]}).to_csv(tmp_path / "representatives.csv", index=False)
    (tmp_path / "summary.json").write_text(json.dumps({"n_candidates": 60, "n_pass": len(passing)}), encoding="utf-8")
    res = sr.load_results(tmp_path)
    assert len(res.cands) == len(passing) + 3 and set(res.cands["출처"]) <= {"계층", "희소", "다듬음"}
    assert res.W.shape == (len(res.cands), 30) and np.all(res.W.sum(axis=1) == 100)
    assert res.cands["비중"].iloc[0] == sr.describe(res.W[0])
    assert {"주식", "채권", "레버리지", "상품 수", "⚠"} <= set(res.cands.columns)
    assert res.presets["이름"].tolist() == ["가", "나"] and res.preset_W.shape == (2, 30)
    assert (res.presets["출처"] == "프리셋").all()  # the detail view reads 출처 for presets too
    assert len(res.cloud) == 5 and res.summary["n_pass"] == len(passing)


# --------------------------------------------------------------------------- filters and frontier

def test_exclude_mask_and_band_frontier():
    W = np.vstack([weights_row(**{"379810": 50, "411060": 50}), weights_row(**{"379800": 50, "411060": 50}),
                   weights_row(**{"379800": 40, "411060": 60})])
    df = pd.DataFrame({"cagr": [0.12, 0.09, 0.08], "mdd": [-0.142, -0.147, -0.121]})
    keep = sr.exclude_mask(W, ["379810"])
    assert keep.tolist() == [False, True, True]
    fr = sr.band_frontier(df)
    assert fr["구간"].tolist() == ["(-15, -14]", "(-13, -12]"] and fr["후보 수"].tolist() == [2, 1]
    assert fr["최고 CAGR"].tolist() == [0.12, 0.08] and fr["행"].tolist() == [0, 2]
    fr2 = sr.band_frontier(df[keep])
    assert fr2["최고 CAGR"].tolist() == [0.09, 0.08] and fr2["행"].tolist() == [1, 2]
    assert sr.band_of(-0.15) == -15 and sr.band_of(-0.1400001) == -14 and sr.band_of(-0.14) == -14


def test_groups_cover_the_subsets_of_the_recorded_frontier():
    g = sr.EXCLUDE_GROUPS
    assert set(g["레버리지 상품"]) == set(sr.CLASSES["레버리지"]) and set(g["가상 상품"]) == set(sr.SYNTHETIC)
    assert {"379810"} <= set(g["나스닥100"]) and len(g["방어·경기 섹터"]) == 4


def test_frontier_of_the_committed_results_matches_the_recorded_table():
    res = sr.load_results()
    raw = (res.cands["출처"] != "다듬음").to_numpy()
    assert int(raw.sum()) == res.summary["n_pass"] == 40_992
    all_best = sr.band_frontier(res.cands[raw])["최고 CAGR"].round(3).tolist()[:5]
    assert all_best == [0.121, 0.119, 0.111, 0.106, 0.094]  # frontier.md, row 전체
    absent = [c for k in ("레버리지 상품", "가상 상품", "방어·경기 섹터", "나스닥100") for c in sr.EXCLUDE_GROUPS[k]]
    keep = raw & sr.exclude_mask(res.W, absent)
    best = sr.band_frontier(res.cands[keep])["최고 CAGR"].round(3).tolist()[:5]
    assert best == [0.098, 0.093, 0.091, 0.088, 0.080]  # frontier.md, row … 나스닥100도 없음
    assert len(res.presets) == 13 and np.all(res.preset_W.sum(axis=1) == 100)


# --------------------------------------------------------------------------- hand-off to the engine and the research page

def test_candidate_config_and_research_handoff():
    w = {"379810": 30, "411060": 70}
    cfg = sr.candidate_config("후보 1", w)
    assert cfg.weights == {"379810": 30.0, "411060": 70.0} and cfg.start == sr.START and cfg.end == sr.END
    assert cfg.rebalance == "Q" and cfg.cost_bps == 10.0 and cfg.band == 0.0
    assert sr.buyable(w) and not sr.buyable({"SYN_IEF_U": 40, "411060": 60})
    state = sr.research_state("r", w, "후보 1", end="2026-10-23")
    assert state["r_on_379810"] is True and state["r_w_379810"] == 30 and state["r_on_379800"] is False
    assert state["r_period"] == "custom" and str(state["r_start"]) == sr.START and str(state["r_end"]) == "2026-10-23"
    assert state["r_rebalance"] == "Q" and state["r_cost"] == 10.0 and state["r_ov_kind"] == "none" and state["r_lev"] == 1.0
    with pytest.raises(ValueError, match="가상"):
        sr.research_state("r", {"SYN_IEF_U": 40, "411060": 60}, "x", end="2026-10-23")


# --------------------------------------------------------------------------- stage 2 (A32)

def test_stage2_scorecard_counts_the_tests_within_the_line(tmp_path):
    df = pd.DataFrame([
        {"후보": "x", "CAGR": 10.0, "MDD": -14.0, "엔진 밴드3 MDD": -15.2, "M MDD": -14.5, "날짜 최악 MDD": -16.0,
         "이웃±2 −15% 안": 50.0, "이웃±5 −15% 안": 20.0, "가상→(H) MDD": np.nan, "상품 수": 3, "레버리지 %": 0, "비중": "미국S&P500 100"},
        {"후보": "y", "CAGR": 9.0, "MDD": -12.0, "엔진 밴드3 MDD": -12.5, "M MDD": -12.1, "날짜 최악 MDD": -12.3,
         "이웃±2 −15% 안": 100.0, "이웃±5 −15% 안": 95.0, "가상→(H) MDD": -13.0, "상품 수": 4, "레버리지 %": 0, "비중": "ACE KRX금현물 100"},
    ])
    card = sr.stage2_scorecard(df)
    assert list(card["후보"]) == ["x", "y"]
    # x: 밴드3 and 날짜 break the line among its 4 MDD tests (기준, 밴드3, 월간, 날짜); (H) is NaN so not a test
    assert card.loc[0, "선 안 시험"] == "2 / 4" and "밴드 3" in card.loc[0, "선을 넘긴 시험"] and "날짜" in card.loc[0, "선을 넘긴 시험"]
    assert card.loc[1, "선 안 시험"] == "5 / 5" and card.loc[1, "선을 넘긴 시험"] == "—"
    assert card.loc[0, "최악 MDD"] == -16.0 and card.loc[1, "최악 MDD"] == -13.0
    assert card.loc[0, "이웃 ±2%p 선 안"] == 50.0 and card.loc[1, "이웃 ±5%p 선 안"] == 95.0  # shares are shown, not counted


def test_load_stage2_reads_the_committed_table_and_names_its_weights():
    s2 = sr.load_stage2()
    assert s2 is not None and len(s2) >= 7
    assert {"R 다듬음 최고 (−15,−14]", "A 전체 최고"} <= set(s2["후보"])
    for t in s2["비중"]:
        sr.parse_names(t)  # every row's weights resolve to universe products
    card = sr.stage2_scorecard(s2)
    r = card.set_index("후보")
    assert r.loc["R 다듬음 최고 (−15,−14]", "최악 MDD"] < -15 and r.loc["B 레버·가상 없음 (-12,-11]", "선을 넘긴 시험"] == "—"
    assert sr.load_stage2(sr.ROOT / "no_such_folder") is None


def test_stage2_picks_point_at_search_rows_or_presets():
    res = sr.load_results()
    s2 = sr.load_stage2()
    picks = sr.stage2_picks(s2, res)
    by = dict(zip(s2["후보"], picks))
    kind, i = by["R 다듬음 최고 (−15,−14]"]  # the refined best is a 다듬음 row of the results
    assert kind == "c" and res.cands.at[i, "출처"] == "다듬음"
    kind, i = by["A 전체 최고"]
    assert kind == "c" and sr.describe(res.W[i]) == s2.set_index("후보").at["A 전체 최고", "비중"]
    kind, i = by["비교: 안정형 40 (금·현금 분산)"]  # comparators are registered presets
    assert kind == "p" and res.presets.at[i, "이름"] == "안정형 40 (금·현금 분산)"
    unknown = pd.DataFrame([{"후보": "비교: 없는 프리셋", "비중": "미국S&P500 100"}, {"후보": "x", "비중": "없는상품 100"}])
    assert sr.stage2_picks(unknown, res) == [("c", -1), ("c", -1)]
