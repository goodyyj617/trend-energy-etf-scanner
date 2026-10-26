"""Asset-allocation lab, real Korean ETF prices (lab/aa_real.py, DECISIONS A33): total return, calendar, hybrid panel."""
import numpy as np
import pandas as pd

from lab import aa_real
from lab.aa_real import SPLICE_LAG, apply_source, hybrid_panel, splice_date, splice_table, to_us_grid, total_return_index
from lab.data import panel_from_long


def model_panel(cols: dict[str, np.ndarray], start="2024-01-01"):
    dates = pd.bdate_range(start, periods=len(next(iter(cols.values()))))
    rows = [{"date": d, "symbol": s, "open": v, "high": v, "low": v, "close": v, "volume": 0}
            for s, c in cols.items() for d, v in zip(dates, c)]
    p = panel_from_long(pd.DataFrame(rows))
    p.benchmark = next(iter(cols))
    p.fx = pd.Series(1300.0, index=p.close.index)
    return p, dates


def test_total_return_adds_distributions_on_the_ex_date_or_the_next_close():
    idx = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05", "2024-01-08"])
    close = pd.Series([100.0, 100.0, 99.0, 99.0], index=idx)
    divs = pd.Series([1.0], index=pd.to_datetime(["2024-01-04"]))  # not a Naver day: goes to 01-05
    tr = total_return_index(close, divs)
    assert tr.iloc[0] == 1.0 and tr.iloc[1] == 1.0
    assert np.isclose(tr.iloc[2], 1.0)  # 99 + 1 over 100
    assert np.isclose(tr.iloc[3], 1.0)
    assert np.isclose(total_return_index(close).iloc[-1], 0.99)


def test_us_dates_take_the_next_seoul_close_and_kospi_the_same_day():
    seoul = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05"])
    tr = pd.Series([1.0, 2.0, 3.0, 4.0], index=seoul)
    us = pd.to_datetime(["2024-01-02", "2024-01-03", "2024-01-05"])
    nxt, used = to_us_grid(tr, us, same_day=False)
    assert list(nxt) == [2.0, 3.0, 3.0]  # 01-05 has no later Seoul day: carried
    assert used.iloc[0] == pd.Timestamp("2024-01-03")
    same, _ = to_us_grid(tr, us, same_day=True)
    assert list(same) == [1.0, 2.0, 4.0]


def test_splice_date_waits_splice_lag_seoul_days():
    seoul = pd.bdate_range("2024-03-01", periods=20)
    tr = pd.Series(np.linspace(1, 2, 20), index=seoul)
    us = pd.bdate_range("2024-01-01", periods=80)
    _, used = to_us_grid(tr, us, same_day=False)
    t0 = splice_date(tr, used)
    assert used[t0] >= seoul[SPLICE_LAG]
    assert used[us[us.get_loc(t0) - 1]] < seoul[SPLICE_LAG]


def test_hybrid_keeps_the_model_before_the_splice_and_follows_real_returns_after():
    n = 60
    model = 100 * 1.001 ** np.arange(n)
    p, dates = model_panel({"379800": model, "SYN_IEF_U": model.copy(), "411060": model.copy()})
    seoul = dates[20:] + pd.Timedelta(days=1)  # listed later; closes the day after each US date
    real = pd.Series(1.0 + 0.01 * np.arange(len(seoul)), index=seoul)
    hyb, info = hybrid_panel(p, {"379800": real, "305080": real * 3})
    t0 = info["379800"]["splice"]
    before = p.close.index < t0
    assert np.allclose(hyb.close.loc[before, "379800"], p.close.loc[before, "379800"])
    after = hyb.close.loc[~before, "379800"]
    vals, _ = to_us_grid(real, p.close.index, same_day=False)
    assert np.allclose(after / after.iloc[0], vals[~before] / vals[t0])
    assert info["SYN_IEF_U"]["real"] == "305080" and info["SYN_IEF_U"]["splice"] == t0  # synthetic -> real product
    assert info["411060"]["splice"] is None and hyb.close["411060"].equals(p.close["411060"])  # no real data: model
    assert hyb.open is hyb.close and hyb.benchmark == p.benchmark and hyb.fx is p.fx
    assert p.close["379800"].equals(pd.Series(model, index=p.close.index, name="379800"))  # input untouched
    table = splice_table(info, end=p.close.index[-1])
    assert list(table["코드"])[-1] == "411060" and table.iloc[-1]["그 전"] == "전 기간 모형"


def test_apply_source_falls_back_to_the_model_without_files(tmp_path):
    p, _ = model_panel({"379800": np.ones(30)})
    out, info, msg = apply_source(p, "real", tmp_path)
    assert out is p and info is None and "파일이 없어" in msg
    out, info, msg = apply_source(p, "model", tmp_path)
    assert out is p and info is None


def test_apply_source_reads_the_naver_and_yahoo_files(tmp_path):
    n = 40
    p, dates = model_panel({"379800": 100 * 1.001 ** np.arange(n)})
    seoul = dates[5:] + pd.Timedelta(days=1)
    pd.DataFrame({"code": "379800", "date": seoul.strftime("%Y-%m-%d"), "open": 1, "high": 1, "low": 1,
                  "close": np.linspace(10000, 11000, len(seoul)), "volume": 1}).to_csv(tmp_path / "naver.csv.gz", index=False)
    pd.DataFrame({"code": ["379800"], "date": [seoul[10].strftime("%Y-%m-%d")], "close": [1.0], "dividends": [50.0]}).to_csv(
        tmp_path / "yahoo.csv.gz", index=False)
    out, info, msg = apply_source(p, "real", tmp_path)
    assert info["379800"]["splice"] is not None and "1/1종" in msg
    assert out.close["379800"].iloc[-1] != p.close["379800"].iloc[-1]


def test_real_codes_cover_every_product_and_the_synthetic_stand_ins():
    from lab.aa_assets import PRODUCTS
    assert set(PRODUCTS) <= set(aa_real.REAL_CODES)
    assert set(aa_real.SYN_REAL.values()) <= set(aa_real.REAL_CODES)


def test_hybrid_continues_with_model_returns_after_the_real_file_ends():
    n = 60
    model = 100 * 1.002 ** np.arange(n)
    p, dates = model_panel({"379800": model})
    seoul = dates[10:40] + pd.Timedelta(days=1)  # the real file stops 20 US days before the panel does
    real = pd.Series(1.0 + 0.001 * np.arange(len(seoul)), index=seoul)
    hyb, info = hybrid_panel(p, {"379800": real})
    tail = hyb.close["379800"].loc[dates[40]:]
    assert np.allclose(tail.pct_change().dropna(), 0.002)  # model returns, not a flat carried price
    assert info["379800"]["real_last"] == seoul[-1]


def test_ensure_files_keeps_fresh_files_and_reports_a_failed_download(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(aa_real, "fetch", lambda folder: calls.append(folder) or (_ for _ in ()).throw(RuntimeError("offline")))
    assert "offline" in aa_real.ensure_files(tmp_path)  # missing: tries, fails, says so
    (tmp_path / "naver.csv.gz").write_text("code,date,close\n")
    assert aa_real.ensure_files(tmp_path) is None and len(calls) == 1  # fresh: no download
    assert "offline" in aa_real.ensure_files(tmp_path, max_age_hours=0)  # stale: tries again


def test_partial_download_does_not_replace_the_files(tmp_path, monkeypatch):
    monkeypatch.setattr(aa_real, "_fetch_naver", lambda code, end: None)
    monkeypatch.setattr(aa_real, "_fetch_yahoo", lambda code: None)
    import pytest
    with pytest.raises(RuntimeError):
        aa_real.fetch(["379800", "379810"], tmp_path)
    assert not (tmp_path / "naver.csv.gz").exists()


def test_the_default_price_source_is_real_after_the_users_choice():
    assert aa_real.DEFAULT_SOURCE == "real"
