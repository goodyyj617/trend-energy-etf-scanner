"""Allocation lab A38: B2 · B3 · B4 with a 10-month trend filter on their equities (scripts/aa_trend_filter_study.py)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import aa_trend_filter_study as T  # noqa: E402
from test_aa_local import _synthetic_krw  # noqa: E402

from lab import aa_local as loc  # noqa: E402


def test_the_filter_reaches_only_the_equities_an_allocation_holds():
    b4 = T.config(loc.CENTRES["B4"], "필터 10", "2005-01-03")
    assert b4.overlay.kind == "sma" and b4.overlay.param == 10 and b4.overlay.safe == T.SAFE
    assert set(b4.overlay.only) == {"379810", "453630"} and b4.rebalance == "M"
    faber = T.config(loc.CENTRES["B4"], "Faber 원형 10", "2005-01-03")
    assert faber.overlay.only == ()  # every risky product
    no_equity = T.config({"411060": 50, "459580": 50}, "필터 10", "2005-01-03")
    assert no_equity.overlay.kind == "none" and no_equity.rebalance == "M"
    assert T.config(loc.CENTRES["B2"], "정적 분기", "2005-01-03").rebalance == "Q"
    assert T.NAMED["B2"] == loc.CENTRES["B2"] and set(T.ERAS) == {"2005", "2000-09"}


def test_filter_activity_follows_the_engine_decisions():
    panel = _synthetic_krw("2003-01-02")
    close = panel.close.copy()
    close["379810"] = 100 * np.cumprod(np.full(len(close), 1 - 0.001))  # nasdaq only falls: always below its average
    panel.close = close
    cfg = T.config(loc.CENTRES["B4"], "필터 10", "2005-01-03")
    res = T.run_allocation(panel, cfg)
    act = T.filter_activity(panel, cfg, res.equity.index)
    nasdaq_share = 30 / (30 + 11)
    assert act["꺼진 비중"] >= nasdaq_share - 1e-9 and act["꺼진 달"] == pytest.approx(1.0)
    assert res.weights["379810"].iloc[-1] == pytest.approx(0.0)
    assert T.filter_activity(panel, T.config(loc.CENTRES["B4"], "정적 월", "2005-01-03"), res.equity.index)["전환"] == 0


def test_study_runs_on_made_up_prices(tmp_path, monkeypatch):
    panels = {"2005": _synthetic_krw("2003-01-02"), "2000-09": _synthetic_krw("1999-06-01")}
    monkeypatch.setattr(T, "LOADERS", {era: (lambda prices, p=p: p) for era, p in panels.items()})
    monkeypatch.setattr(T, "QUICK", dict(n_sample=4, offsets=(0, 5), chunk=2))  # keeps CI short
    T.main(["--quick", "--workers", "1", "--prices", "model", "--out", str(tmp_path)])
    named = pd.read_csv(tmp_path / "named.csv")
    assert len(named) == len(T.NAMED) * len(T.SETTINGS) * len(T.ERAS)
    assert {"cagr", "mdd", "cagr_2016", "dd_2008", "날짜 최악 MDD", "꺼진 비중", "전환"} <= set(named.columns)
    assert named.loc[named["기간"] == "2000-09", "dd_dotcom"].notna().all()
    static = named[named["설정"] == "정적 분기"]
    assert (static["꺼진 비중"] == 0).all() and (named["날짜 최악 MDD"] <= named["mdd"] + 1e-12).all()
    groups = pd.read_csv(tmp_path / "groups.csv")
    assert set(groups["묶음"]) == {"구조 가족", "무작위"} and set(groups["설정"]) == set(T.SAMPLE_SETTINGS)
    assert (groups["수"] == T.QUICK["n_sample"]).all()
    samples = pd.read_csv(tmp_path / "samples.csv.gz")
    assert len(samples) == 2 * T.QUICK["n_sample"] * len(T.SAMPLE_SETTINGS) * len(T.ERAS)
    curves = pd.read_csv(tmp_path / "curves.csv.gz", parse_dates=["date"])
    assert curves["date"].max() <= pd.Timestamp(T.CURVE_END) and "B2 · 필터 10" in curves.columns
    warm = pd.read_csv(tmp_path / "warmup.csv")
    assert warm["2000-09에 필터 작동"].all()
    summary = json.loads((tmp_path / "summary.json").read_text(encoding="utf-8"))
    assert set(summary["verdict"]["점 판정 (필터 10, 두 기간 −15% 안)"]) == {"B2", "B3", "B4"}


def test_sample_rows_are_the_a37_members():
    rows = T.sample_rows(5)
    assert np.array_equal(rows["구조 가족"], loc.sample_family(20_000, seed=T.FAMILY_SEED)[:5])
    assert loc.rules(rows["구조 가족"])["all"].all()
