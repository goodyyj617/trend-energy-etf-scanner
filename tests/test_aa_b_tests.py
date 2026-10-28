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
    expect_dotcom = {"B2": -16.4, "B3": -30.8, "B4": -25.6, "B5": -26.2}  # A37 ④, B5 from A39
    for name, m in expect_dotcom.items():
        assert round(100 * bt.centre(pts, name, "2000-09")["mdd"], 1) == m
    one = pts[pts["묶음"] == "centre"].groupby(["배분", "기간"]).size()
    assert (one == 1).all()  # A38 and A39 both ran the 2000-09 centres; the page keeps one


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
    assert bt.summary(pts, "2000-09")["배분"].nunique() == 4


def test_composition_and_canary_points_match_their_tables(pts):
    for fam, n in {"loo": 45, "swaprest": 12, "simple": 25, "canary_pass": 27, "canary_fail": 181}.items():
        for period in ("2005", "2000-09"):
            assert ((pts["묶음"] == fam) & (pts["기간"] == period)).sum() == n, (fam, period)
    passing = pts[(pts["묶음"] == "canary_pass") & (pts["기간"] == "2005")].groupby("배분").size()
    assert passing.to_dict() == {"B3": 7, "B4": 8, "B5": 12}  # A40: no B2 row passes
    assert set(pts.loc[pts["묶음"] == "simple", "배분"]) == {"B 평균", "뼈대"}

    def one(fam, name, start):
        part = pts[(pts["묶음"] == fam) & (pts["배분"] == name) & pts["변형"].str.startswith(start)].set_index("기간")
        return {k: (round(100 * r["cagr"], 2), round(100 * r["mdd"], 2)) for k, r in part.iterrows()}

    # numbers quoted in the A39 / A40 memos
    assert one("canary_pass", "B4", "K5 절대 모멘텀 · 나스닥 → 안전 묶음 · 1개월 금지") == {"2005": (8.9, -10.25), "2000-09": (7.91, -11.72)}
    assert one("simple", "뼈대", "나스닥100 30 · KRX금 30 · CD금리 40")["2005"] == (10.0, -12.75)
    assert one("simple", "B 평균", "네 배분")["2005"] == (10.01, -11.91)
    late = pts[pts["변형"].str.contains("늦게 시작", regex=False)]
    assert len(late) and (late["기간"] == "2000-09").all() and late["변형"].str.contains(r"\d{4}-\d{2}-\d{2}부터 잰 MDD −").all()


def test_a41_points_match_the_a41_tables(pts):
    a = pts[pts["기간"] == "a41"]
    expect = {"B2": (9.10, -16.40), "B4": (7.57, -25.57), "안정형 40": (7.77, -13.73)}  # A41 memo, engine, 2000-09 – 2024-12
    for name, (c, m) in expect.items():
        r = bt.centre(pts, name, "a41")
        assert (round(100 * r["cagr"], 2), round(100 * r["mdd"], 2)) == (c, m), name
    rebal = a[a["묶음"] == "a41_rebal"]
    assert rebal.groupby("배분").size().to_dict() == {"B2": 528, "B4": 528}
    v = pd.read_csv(bt.A41_DIR / "verdicts.csv").set_index("대상")
    for name in ("B2", "B4"):
        assert rebal[rebal["배분"] == name]["−15% 안"].mean() == pytest.approx(v.at[name, "3 리밸런싱 안 비율"])
    new = a[a["묶음"] == "a41_new"].set_index("변형")
    assert set(new["배분"]) == {"A41"} and len(new) >= 9  # 5 half-window centres, 3 rule weights, the mean centre(s)
    r = new.loc["평균 중심 (가) · 사후 보완"]
    assert (round(100 * r["cagr"], 2), round(100 * r["mdd"], 2)) == (8.84, -13.71)
    assert r["비중"] == "나스닥100 18 · S&P500 3 · 헬스케어 8 · KRX금현물 30 · 미국10년(H) 30 · 미국30년(H) 11"


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
