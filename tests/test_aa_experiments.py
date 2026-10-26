"""Allocation lab: the experiment log (lab/aa_experiments.py, page '실험 기록'). User rule 2026-10-26: why, key results, graphs."""
import re

from lab import aa_experiments as ex
from lab import aa_search_results as sr


def test_every_experiment_has_why_results_a_chart_and_a_memo():
    keys = [e.key for e in ex.EXPERIMENTS]
    assert len(keys) == len(set(keys)) and keys == sorted(keys)  # oldest first, the page shows newest first
    for e in ex.EXPERIMENTS:
        assert e.why and e.results and e.charts, e.key
        assert set(e.charts) <= set(ex.CHART_KEYS), e.key
        assert e.memo.exists(), e.memo


def test_the_app_draws_exactly_the_registered_chart_keys():
    src = (sr.ROOT / "lab" / "aa_app.py").read_text(encoding="utf-8")
    block = src[src.index("EXPERIMENT_CHARTS = {"):]
    block = block[:block.index("\n}\n")]
    assert set(re.findall(r'^\s+"(\w+)":', block, flags=re.M)) == set(ex.CHART_KEYS)
    assert "experiments_page" in src and '"experiments": "experiments_page"' in (sr.ROOT / "scripts" / "aa_link.py").read_text(encoding="utf-8")


def test_frontier_table_reads_the_recorded_frontiers():
    model, real = ex.frontier_table(sr.RESULTS_DIR), ex.frontier_table(sr.RESULTS_REAL_DIR)
    top = lambda t, name: t[t["부분집합"] == name].sort_values("band_hi")["cagr"].tolist()[:5]  # noqa: E731
    assert top(model, "전체") == [12.1, 11.9, 11.1, 10.6, 9.4]
    assert top(real, "전체") == [12.2, 12.1, 11.2, 10.6, 9.5]
    assert top(real, "… 나스닥100도 없음") == [10.3, 10.1, 9.3, 8.7, 8.1]
    assert len(ex.preset_points(sr.RESULTS_DIR)) == 13


def test_stage2_and_neighbour_data_of_the_presets_check():
    t = ex.stage2_tests(ex.STAGE2_PRESETS_DIR)
    out = t[~t["선 안"]]
    assert list(out["후보"]) == ["탐색 A (가상→미국10년(H))"] and list(out["시험"]) == ["밴드 3%p"]
    assert t.groupby("후보").size().tolist() == [12, 12, 12]
    nb = ex.neighbour_shares(ex.STAGE2_PRESETS_DIR).set_index(["후보", "이웃"])["선 안 비율"]
    assert nb[("탐색 B4", "±5%p")] == 100.0 and round(nb[("탐색 A (가상→미국10년(H))", "±5%p")]) == 53


def test_real_price_chart_data():
    g = ex.product_gap()
    assert len(g) == 28 and g["diff_pp"].is_monotonic_increasing
    assert round(g.set_index("name").at["KODEX 200", "diff_pp"], 1) == 3.7
    c = ex.candidate_shift().set_index("후보")
    assert c.at["R", "실제 MDD"] < -15 < c.at["R", "모형 MDD"] and "내 AAA" in c.index
    share_model, _ = ex.holding_share(sr.load_results(sr.RESULTS_DIR), "069500")
    share_real, _ = ex.holding_share(sr.load_results(sr.RESULTS_REAL_DIR), "069500")
    assert round(share_model) == 16 and round(share_real) == 32
