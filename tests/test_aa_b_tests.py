"""The 'B2–B5 시험 모음' page data (lab/aa_b_tests.py): it only gathers committed results, so the numbers must match
the tables they come from."""
import re

import pandas as pd
import pytest

from lab import aa_b_tests as bt
from lab import aa_local as loc
from lab import aa_search_results as sr
from lab.aa_assets import PRESETS


@pytest.fixture(scope="module")
def pts():
    return bt.points()


def test_centres_match_the_stage2_table_and_the_dotcom_runs(pts):
    expect_2005 = {"B2": (10.69, -13.23), "B3": (9.99, -12.87), "B4": (9.69, -11.65), "B5": (9.28, -10.91), "안정형 40": (8.67, -11.83)}
    for name, (c, m) in expect_2005.items():
        r = bt.centre(pts, name, "2005")
        assert r["cagr"] == pytest.approx(c / 100) and r["mdd"] == pytest.approx(m / 100)
    expect_dotcom = {"B2": -16.4, "B3": -30.8, "B4": -25.6}  # A37 ④
    for name, m in expect_dotcom.items():
        assert round(100 * bt.centre(pts, name, "2000-09")["mdd"], 1) == m
    assert bt.centre(pts, "B5", "2000-09") is None  # not run on the extended panel yet


def test_neighbour_counts_and_shares_match_stage2(pts):
    s2 = pd.read_csv(bt.STAGE2_DIR / "stage2.csv").set_index("후보")
    for name, label in bt.STAGE2_LABELS.items():
        for step in (2, 5):
            part = pts[(pts["배분"] == name) & (pts["묶음"] == f"nb{step}")]
            assert len(part) == s2.loc[label, f"이웃±{step} 수"]
            assert 100 * part["−15% 안"].mean() == pytest.approx(s2.loc[label, f"이웃±{step} −15% 안"], abs=0.01)
            assert 100 * part["mdd"].min() == pytest.approx(s2.loc[label, f"이웃±{step} 최악 MDD"], abs=0.01)


def test_families_periods_and_sources(pts):
    assert set(pts["묶음"]) <= set(bt.FAMILIES)
    assert set(pts["기간"]) <= set(bt.PERIODS)
    for fam, f in bt.FAMILIES.items():
        got = set(pts.loc[pts["묶음"] == fam, "기간"])
        assert got == set(f.periods), fam
    local = pts[pts["묶음"] == "local"].groupby("배분").size()
    assert local.to_dict() == {"B2": 5000, "B3": 5000, "B4": 5000}
    # A38: the equity-only 10-month filter keeps B2–B4 inside −15 % from 2000-09
    f10 = pts[(pts["묶음"] == "filter") & (pts["변형"] == "필터 10") & (pts["기간"] == "2000-09")].set_index("배분")["mdd"]
    assert (f10 >= sr.MDD_LINE).all() and set(f10.index) == {"B2", "B3", "B4"}
    assert pts["비중"].fillna("").str.contains(":").sum() == 0  # code text turned into product names


def test_summary_rows(pts):
    s = bt.summary(pts, "2005")
    assert set(s["배분"]) == set(bt.NAMES)
    row = s[(s["배분"] == "B4") & (s["시험"] == bt.FAMILIES["nb5"].label)].iloc[0]
    assert row["점 수"] == 145 and row["−15% 안 (%)"] == pytest.approx(100.0)
    assert bt.summary(pts, "2000-09")["배분"].nunique() == 3


def _numbers(text: str) -> list[int]:
    return sorted(int(v) for v in re.findall(r" (\d+)(?= ·|$)", text))


def test_weights_text_matches_the_registered_allocations():
    assert _numbers(bt.WEIGHTS["B2"]) == sorted(PRESETS["gs_b2"].weights.values())
    assert _numbers(bt.WEIGHTS["B4"]) == sorted(PRESETS["gs_b4"].weights.values())
    assert _numbers(bt.WEIGHTS["B3"]) == sorted(loc.CENTRES["B3"].values())
    s2 = pd.read_csv(bt.STAGE2_DIR / "stage2.csv").set_index("후보")
    assert _numbers(bt.WEIGHTS["B5"]) == _numbers(s2.loc[bt.STAGE2_LABELS["B5"], "비중"])


def test_page_is_wired():
    app = (sr.ROOT / "lab" / "aa_app.py").read_text(encoding="utf-8")
    link = (sr.ROOT / "scripts" / "aa_link.py").read_text(encoding="utf-8")
    assert "def btests_page" in app and "st.Page(btests_page" in app and '"bt_"' in app
    assert '"btests": "btests_page"' in link
