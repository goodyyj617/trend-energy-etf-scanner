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


def test_local_study_chart_data():
    b = ex.local_neighbour_bins()
    assert set(b["중심"]) == {"B2", "B3", "B4"} and len(b) == 15
    assert (b[b["거리"].isin(["0–5", "5–10"])]["선 안"] == 100).all()
    g = ex.local_groups().set_index("묶음")
    fam, rnd = g.loc[ex.group_label("구조 가족")], g.loc[ex.group_label("무작위")]
    assert round(fam["2005~ 선 안"], 1) == 32.6 and round(rnd["2005~ 선 안"], 1) == 29.7
    assert round(fam["2000-09~ 선 안"], 1) == 0.4 and round(rnd["2000-09~ 선 안"]) == 24
    assert list(g.index) == [ex.group_label(k) for k in ex.LOCAL_GROUP_ORDER]
    boot = ex.bootstrap_mdds()
    p_out = (boot["MDD"] < sr.STAGE2_PCT_LINE).groupby(boot["이름"]).mean()
    assert round(p_out["B2"], 3) == 0.178 and round(p_out["B4"], 3) == 0.107 and boot.groupby("이름").size().eq(1000).all()
    n = ex.dotcom_named().set_index("이름")
    assert round(n.at["B2", "2000-09~ MDD"], 1) == -16.4 and round(n.at["B4", "2000-09~ MDD"], 1) == -25.6
    assert set(n.index[n["2000-09~ MDD"] >= sr.STAGE2_PCT_LINE]) == {"안정형 35 (금·현금 분산)", "안정형 40 (금·현금 분산)", "영구 포트폴리오", "골든 버터플라이"}
    dd = ex.dotcom_drawdowns()
    assert set(dd["이름"]) == set(ex.DOTCOM_CURVES)
    worst = dd.groupby("이름")["낙폭"].min()
    assert (worst >= n.loc[worst.index, "2000-09~ MDD"] - 1e-9).all()  # curves kept every 5th day are never deeper than the daily MDD


def test_search_presets_say_they_are_confirmed_only_from_2005():
    from lab.aa_assets import PRESETS
    for key in ("gs_b2", "gs_b4", "gs_a"):
        assert "2005년 이후에만 확인됨" in PRESETS[key].note, key


def test_trend_filter_chart_data():
    n = ex.filter_named().set_index(["이름", "기간", "설정"])
    assert len(n) == 5 * 2 * len(ex.FILTER_SETTINGS) and set(n.index.get_level_values("설정")) == set(ex.FILTER_SETTINGS)
    for name, mdd05, mdd00 in (("B2", -13.2, -16.4), ("B3", -12.9, -30.8), ("B4", -11.7, -25.6)):  # A37's static values
        assert round(n.at[(name, "2005", "정적 분기"), "MDD"], 1) == mdd05 and round(n.at[(name, "2000-09", "정적 분기"), "MDD"], 1) == mdd00
        for setting in ("필터 8", "필터 10", "필터 12"):  # the pre-registered point and robustness reading
            assert n.at[(name, "2005", setting), "MDD"] >= sr.STAGE2_PCT_LINE and n.at[(name, "2000-09", setting), "MDD"] >= sr.STAGE2_PCT_LINE
    assert (n.xs("정적 분기", level="설정")["꺼진 비중"] == 0).all() and (n["날짜 최악 MDD"] <= n["MDD"] + 1e-9).all()
    assert round(n.at[("B2", "2005", "필터 10"), "CAGR"], 2) == 8.89 and round(n.at[("B2", "2005", "정적 분기"), "CAGR"], 2) == 10.69
    g = ex.filter_groups().set_index(["묶음", "설정"])
    fam, rnd = ex.group_label("구조 가족"), ex.group_label("무작위")
    assert round(g.at[(fam, "필터 10"), "두 기간 모두"], 2) == 45.85 < ex.FILTER_STRUCTURE_SHARE
    assert round(g.at[(rnd, "필터 10"), "두 기간 모두"], 2) == 50.25 and round(g.at[(fam, "정적 분기"), "2000-09 선 안"], 1) == 0.3
    assert (g["수"] == 2000).all()
    dd = ex.filter_drawdowns()
    assert set(dd["설정"]) == {"정적 분기", "필터 10"} and dd["date"].max() <= __import__("pandas").Timestamp("2006-12-31")
    worst = dd.groupby(["이름", "설정"])["낙폭"].min()
    assert worst[("B3", "정적 분기")] < -25 < -10 < worst[("B3", "필터 10")]
    v = __import__("json").loads((ex.FILTER_DIR / "summary.json").read_text(encoding="utf-8"))["verdict"]
    assert v["제안할 필터 판 후보"] == ["B2", "B3", "B4"] and v["구조 판정 (≥ 50%)"] is False


def test_composition_chart_data():
    lo = ex.comp_leave_one_out()
    assert len(lo) == 24 and (lo[lo["상품"] == "미국나스닥100"]["판정"] == "제 몫").all()
    assert set(lo.loc[lo["판정"] == "빼면 나아짐", "상품"]) == {"미국S&P500헬스케어", "iShares미국인플레이션국채액티브", "iShares미국투자등급회사채액티브"}
    s = ex.comp_swap()
    assert len(s) == 16 and (s["MDD"] >= sr.STAGE2_PCT_LINE).all() and (s["2000-09 MDD"] < sr.STAGE2_PCT_LINE).sum() == 11 + 4
    simple = ex.comp_simple().set_index("이름")
    assert round(simple.at["B 평균", "CAGR"], 2) == 10.01 and round(simple.at["B 평균", "MDD"], 2) == -11.91
    assert (simple["판정"] == "같은 일을 한다").sum() == 9 and round(simple.at["B2", "MDD"], 2) == -13.23  # B 평균 + 8 skeletons
    r = ex.comp_reselect()
    assert set(r["고른 기간"]) == {"2005–2015에서 고름", "2016–2026에서 고름", *ex.COMP_NAMES}
    med = r[r["고른 기간"] == "2016–2026에서 고름"].groupby("역할")["비중"].median()
    assert med["나스닥"] == 20.5 and med["금"] == 32
    b = r[r["고른 기간"].isin(ex.COMP_NAMES)].groupby("고른 기간")["비중"].sum()
    assert (b == 100).all()
    m = ex.comp_model_gap()
    assert len(m) == 13 and set(m.loc[m["판정"] == "괜찮음", "상품"]) == {"미국나스닥100", "MSCI선진국"}


def test_canary_chart_data():
    d = ex.canary_rows()
    assert len(d) == 208 and (d["판정"] == "후보").sum() == 27
    cand = d[d["판정"] == "후보"]
    assert set(cand["배분"]) == {"B3", "B4", "B5"} and cand[list(ex.CANARY_VERDICTS)].all().all()
    assert (cand["2005 MDD"] >= sr.STAGE2_PCT_LINE).all() and (cand["2000-09 MDD"] >= sr.STAGE2_PCT_LINE).all()
    assert (cand["가짜보다 얕은 비율"] >= 90).all() and (cand["2000-09 MDD"] >= cand["같은 비용 2000-09 MDD"] + 1 - 1e-9).all()
    assert set(cand.loc[cand["행동"] == "safe", "신호"]) == {"k2", "k4", "k5", "d1"}
    st = ex.canary_static().set_index(["종류", "기간", "배분"])["MDD"]
    for name, mdd00 in (("B2", -16.4), ("B3", -30.8), ("B4", -25.6), ("B5", -26.2)):  # A37 / A39 static values
        assert round(st[("정적 분기", "2000-09", name)], 1) == mdd00
    p = ex.canary_placebo()
    assert len(p) == 208 and (p["가짜 5%"] <= p["가짜 중앙"]).all() and (p["가짜 중앙"] <= p["가짜 95%"]).all()
    c = ex.canary_currency()
    assert len(c) == 48 and set(c["신호"]) == {"k1", "k3", "k4"}
    k4 = c[(c["신호"] == "k4") & (c["기간"] == "2000-09")]
    assert (k4["원화 MDD"] < k4["달러 MDD"]).all()
    trades = d.groupby(["신호", "판"])["2005 연 신호 매매"].mean().unstack()
    assert (trades["1개월 금지"] < trades["바로 매매"]).all()


def test_a41_chart_data():
    m = ex.a41_map()
    assert set(m["구역"]) == set(ex.A41_ZONES) and m.groupby("구역").size().between(5000, 7000).all()
    assert (m.loc[m["구역"] == "안", "MDD"] >= sr.STAGE2_PCT_LINE).all() and (m.loc[m["구역"] == "밖", "MDD"] < -17).all()
    k = ex.a41_marks().set_index("이름")
    assert (round(k.at["안정형 40", "CAGR"], 2), round(k.at["안정형 40", "MDD"], 1)) == (7.77, -13.7)
    assert round(k.at["평균 중심 (가) · 사후 보완", "MDD"], 2) == -13.71 and k.at["평균 중심 (가) · 사후 보완", "종류"] == "사후 중심"
    assert (k["종류"] == "절반 구간 중심").sum() == 5 and (k["종류"] == "규칙형").sum() == 3
    s = ex.a41_sobol()
    mdd = s[s["출력"] == "MDD"].set_index("이름")["ST"]
    assert mdd.idxmax() == "나스닥100" and round(mdd.max(), 2) == 0.36
    inc = ex.a41_include().set_index("상품")["안 비율 변화"]
    assert round(inc["CD금리"], 1) == 12.6 and round(inc["미국10년(H)"], 1) == 10.2 and inc.idxmin() == "유틸리티"
    r = ex.a41_rebalance()
    assert r.groupby("이름").size().eq(528).all() and set(r["주기"]) == {"월", "분기", "반기", "연"}
    post = r[r["이름"] == "평균 중심 (가) · 사후 보완"]
    share = (post["MDD"] >= sr.STAGE2_PCT_LINE).groupby(post["주기"]).mean()
    assert share["월"] == 1.0 and share["연"] < share["반기"] < share["분기"] < 1.0
    assert (r.loc[r["이름"] == "안정형 40", "MDD"] >= sr.STAGE2_PCT_LINE).all()
    v = ex.a41_verdicts().set_index("이름")
    assert v.at["평균 중심 (가) · 사후 보완", "모두 통과"] == False  # noqa: E712 - verdict 3 (rebalancing ≥ 90 %) missed
    assert round(v.at["평균 중심 (가) · 사후 보완", "3 리밸런싱 안 비율"], 2) == 81.25
    h = ex.a41_holdout().set_index("이름")
    assert set(h["시작"]) == {"2025-01-02"} and set(h["끝"]) == {"2026-10-27"}
    assert (round(h.at["평균 중심 (가) · 사후 보완", "누적 수익"], 1), round(h.at["평균 중심 (가) · 사후 보완", "MDD"], 1)) == (23.0, -11.3)
    assert (round(h.at["S&P500 보유", "누적 수익"], 1), round(h.at["S&P500 보유", "MDD"], 1)) == (25.4, -19.2)
    assert len(h) == 16 and (h["실행"] != "첫 실행").sum() == 4  # (나) and the three rule weights, added the same day
    assert (round(h.at["평균 중심 (나) · 사후 보완", "누적 수익"], 1), round(h.at["평균 중심 (나) · 사후 보완", "MDD"], 1)) == (24.8, -13.2)
    assert (h["종류"] == "규칙형").sum() == 3 and (h.loc[h["종류"] == "규칙형", "누적 수익"] < h.at["안정형 40", "누적 수익"]).all()
    na = v.loc["평균 중심 (나) · 사후 보완"]  # design §5: the (나) group is also read against −17 %
    assert round(na["mdd"], 2) == -15.73 and na["−17% 1 기준선"] == True and na["−17% 모두 통과"] == False  # noqa: E712
    assert round(na["−17% 3 리밸런싱 안 비율"], 1) == 80.9


def test_a41_preset_is_the_post_hoc_centre():
    """The registered A41 preset is the (가) post-hoc centre row, labelled post hoc with its rebalancing condition."""
    import pandas as pd

    from lab.aa_assets import PRESETS
    c = pd.read_csv(ex.A41_DIR / "centres.csv", encoding="utf-8").set_index("묶음")
    p = PRESETS["a41_ga"]
    text = c.at["(가) 평균·보완", "비중"]  # "code:percent;..."
    assert p.weights == {k: int(v) for k, v in (part.split(":") for part in text.split(";"))}
    assert "사후" in p.name and "월·분기" in p.name and "모형 가격" in p.note


def test_a42_stage0_chart_data():
    t = ex.a42_0_sets().set_index("배분")
    assert list(t.index) == list(ex.A42_0_ORDER)
    assert t.at["(가)", "−15% 안 세트"] == 24 and t.at["안정형 40", "−15% 안 세트"] == 24
    assert (t.loc[["B2", "B4", "현재 보유", "S&P500"], "−15% 안 세트"] == 0).all()
    assert (round(t.at["(가)", "대표 CAGR"], 2), round(t.at["(가)", "대표 MDD"], 2)) == (8.81, -13.68)
    assert ((t["MDD 최선"] - t["MDD 최악"]) <= 1.25).all() and (t["CAGR 최저"] <= t["대표 CAGR"]).all()
    h = ex.a42_0_hedge()
    pick = h[(h["상품"] == "267440") & (h["원화 현금"] == "cd91")].iloc[0]
    assert round(pick["h 추정 (bp)"]) == 70 and pick["95% 하한 (bp)"] < 0 < pick["95% 상한 (bp)"]

