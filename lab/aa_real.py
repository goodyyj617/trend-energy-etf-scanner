"""Real Korean ETF prices for the allocation lab (DECISIONS A33).

The lab's KRW panel (lab/aa_korea.py) is a model: US proxy ETF × USD/KRW (+ cash rates, hedge cost, fees). This
module swaps in the real price of each Korean-listed product after its listing and keeps the model before it
(a "hybrid" panel):

- Real total return per code: r_t = (close_t + distribution_t) / close_{t-1} − 1, with Naver closes (unadjusted,
  full history) and Yahoo distributions (ex-date, KRW). A distribution whose ex-date is missing on Naver goes to the
  next Naver day. Yahoo closes are not used: they look distribution-adjusted.
- Calendar: the model is on US dates. A Korean ETF with a US underlying closes in Seoul after the US close of the
  previous US day, so US date t takes the real close of the first Seoul trading day strictly after t. KOSPI 200
  products (069500, 122630) and the CD-rate ETF (459580) take the last Seoul close on or before t.
- Splice: each product switches to real prices from the first US date mapped to its 6th Seoul trading day
  (5 days after the first real close, past the listing-day noise), scaled to meet the model there.
- The two synthetic unhedged treasuries of A31 have no product of their own; they take the real prices of
  TIGER 미국채10년선물 (305080) and ACE 미국30년국채액티브 (453850).

Price files live in lab_data/kr/ (gitignored, like lab_data/yf): naver.csv.gz (code,date,open,high,low,close,
volume) and yahoo.csv.gz (code,date,…,close,…,dividends). `python -m lab.aa_real fetch` downloads them (needs
Naver and Yahoo; the cloud container cannot reach either). Without the files every function falls back to the
model and says so.
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .aa_assets import PRODUCTS
from .data import CACHE_DIR, Panel

REAL_DIR = CACHE_DIR / "kr"
SAME_DAY = frozenset({"069500", "122630", "459580"})  # KOSPI (model from EWY) and KRW cash: same calendar date
SPLICE_LAG = 5  # Seoul trading days after the first real close
SYN_REAL = {"SYN_IEF_U": "305080", "SYN_TLT_U": "453850"}  # synthetic product -> real product standing in for it
EXTRA_CODES = {"305080": "TIGER 미국채10년선물", "308620": "KODEX 미국채10년선물", "453850": "ACE 미국30년국채액티브"}
REAL_CODES = sorted(set(PRODUCTS) | set(EXTRA_CODES))
SOURCES = {"real": "실제 가격 우선 (상장 후 실제 · 상장 전 모형)", "model": "모형만 (미국 대용 × 원/달러)"}
DEFAULT_SOURCE = "model"  # the user decides whether the hybrid becomes the default (decision card, 2026-10-26)


# --------------------------------------------------------------------------- real series

def load_files(folder: Path | str = REAL_DIR) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    """(naver, yahoo) tables, or None when the files are not there."""
    folder = Path(folder)
    nav, yah = folder / "naver.csv.gz", folder / "yahoo.csv.gz"
    if not nav.exists():
        return None
    naver = pd.read_csv(nav, dtype={"code": str}, parse_dates=["date"])
    yahoo = pd.read_csv(yah, dtype={"code": str}, parse_dates=["date"]) if yah.exists() else pd.DataFrame(
        columns=["code", "date", "close", "dividends"])
    return naver, yahoo


def total_return_index(close: pd.Series, dividends: pd.Series | None = None) -> pd.Series:
    """Seoul-calendar total-return index (1.0 on the first close) from closes and ex-date distributions."""
    close = close.astype(float).sort_index()
    close = close[close > 0]
    div = pd.Series(0.0, index=close.index)
    if dividends is not None:
        for d, v in dividends[dividends > 0].items():
            pos = close.index.searchsorted(d, side="left")
            if pos < len(close):
                div.iloc[pos] += float(v)
    r = (close + div) / close.shift(1) - 1.0
    return (1.0 + r.fillna(0.0)).cumprod()


def real_series(naver: pd.DataFrame, yahoo: pd.DataFrame) -> dict[str, pd.Series]:
    """code -> Seoul-calendar total-return index."""
    out = {}
    for code, n in naver.groupby("code"):
        y = yahoo[yahoo["code"] == code]
        divs = y.set_index("date")["dividends"].astype(float) if "dividends" in y and len(y) else None
        tr = total_return_index(n.set_index("date")["close"], divs)
        if len(tr) > SPLICE_LAG + 1:
            out[str(code)] = tr
    return out


def to_us_grid(tr: pd.Series, us_index: pd.DatetimeIndex, same_day: bool) -> tuple[pd.Series, pd.Series]:
    """Real value on each US date and the Seoul date it came from (see the module docstring). Forward-filled."""
    seoul = tr.index
    pos = seoul.searchsorted(us_index, side="right")
    if same_day:
        pos = pos - 1
        ok = pos >= 0
    else:
        ok = pos < len(seoul)
    vals = np.full(len(us_index), np.nan)
    used = pd.Series(pd.NaT, index=us_index, dtype="datetime64[ns]")
    vals[ok] = tr.to_numpy()[pos[ok]]
    used[ok] = seoul[pos[ok]]
    return pd.Series(vals, index=us_index).ffill(), used.ffill()


def splice_date(tr: pd.Series, used: pd.Series) -> pd.Timestamp | None:
    """First US date whose Seoul source date is at least SPLICE_LAG trading days after the first real close."""
    s5 = tr.index[min(SPLICE_LAG, len(tr) - 1)]
    ok = used.notna() & (used >= s5)
    return used.index[ok.to_numpy()][0] if ok.any() else None


# --------------------------------------------------------------------------- hybrid panel

def hybrid_panel(panel: Panel, real: dict[str, pd.Series], syn_real: dict[str, str] | None = None
                 ) -> tuple[Panel, dict[str, dict]]:
    """Copy of a KRW model panel with every column real from its splice date on.

    Returns (panel, info) where info maps each panel column to {"real": code, "splice": Timestamp | None,
    "real_last": Timestamp | None}; columns without real data map to splice None."""
    syn_real = SYN_REAL if syn_real is None else syn_real
    us = panel.close.index
    close = panel.close.copy()
    info: dict[str, dict] = {}
    for col in panel.close.columns:
        rc = syn_real.get(col, col)
        tr = real.get(rc)
        if tr is None:
            info[col] = {"real": None, "splice": None, "real_last": None}
            continue
        vals, used = to_us_grid(tr, us, rc in SAME_DAY)
        t0 = splice_date(tr, used)
        model0 = panel.close.at[t0, col] if t0 is not None else np.nan
        if t0 is None or not np.isfinite(model0):
            info[col] = {"real": rc, "splice": None, "real_last": tr.index[-1]}
            continue
        after = us >= t0
        close.loc[after, col] = model0 * vals[after] / vals.at[t0]
        info[col] = {"real": rc, "splice": t0, "real_last": tr.index[-1]}
    out = copy.copy(panel)
    out.open = out.high = out.low = out.close = close
    out.source = panel.source + " · 상장 후 실제 국내 가격(네이버 종가 + Yahoo 분배금)"
    for attr in ("benchmark", "fx"):
        if hasattr(panel, attr):
            setattr(out, attr, getattr(panel, attr))
    return out, info


def splice_table(info: dict[str, dict], names: dict[str, str] | None = None, end: pd.Timestamp | None = None) -> pd.DataFrame:
    """One row per column: which real product, from which date the price is real, and how many years that is."""
    rows = []
    for col, d in info.items():
        t0 = d["splice"]
        years = (end - t0).days / 365.25 if (t0 is not None and end is not None) else np.nan
        rows.append({"코드": col, "상품": (names or {}).get(col, col), "실제 가격 상품": d["real"] or "—",
                     "실제 가격 시작": t0.date() if t0 is not None else None,
                     "실제 구간(년)": round(years, 1) if np.isfinite(years) else np.nan,
                     "그 전": "모형" if t0 is not None else "전 기간 모형"})
    return pd.DataFrame(rows).sort_values("실제 가격 시작", na_position="last", kind="stable").reset_index(drop=True)


def apply_source(panel: Panel, source: str, folder: Path | str = REAL_DIR) -> tuple[Panel, dict[str, dict] | None, str]:
    """The panel for a price source ("real" or "model"), the splice info (None for the model) and a status line."""
    if source != "real":
        return panel, None, "모형만: 모든 상품이 전 기간 미국 대용 × 원/달러입니다."
    files = load_files(folder)
    if files is None:
        return panel, None, (f"실제 가격 파일이 없어 모형으로 계산했습니다 ({Path(folder) / 'naver.csv.gz'}). "
                             "`python -m lab.aa_real fetch`로 받으세요.")
    hyb, info = hybrid_panel(panel, real_series(*files))
    n = sum(1 for d in info.values() if d["splice"] is not None)
    last = max((d["real_last"] for d in info.values() if d["real_last"] is not None), default=None)
    return hyb, info, f"실제 가격: {n}/{len(info)}종, 상장 5거래일 뒤부터 (실제 가격 ~ {last.date() if last is not None else '–'})"


# --------------------------------------------------------------------------- download (user's computer)

UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"


def _fetch_yahoo(code: str) -> pd.DataFrame | None:
    import yfinance as yf
    for suffix in (".KS", ".KQ"):
        try:
            h = yf.Ticker(f"{code}{suffix}").history(period="max", interval="1d", auto_adjust=False, actions=True)
        except Exception:  # noqa: BLE001 - try the other market, then give up on this code
            h = None
        time.sleep(0.5)
        if h is not None and not h.empty:
            idx = pd.DatetimeIndex(h.index)
            idx = idx.tz_localize(None) if idx.tz is not None else idx
            return pd.DataFrame({"code": code, "date": idx.strftime("%Y-%m-%d"), "close": h["Close"].to_numpy(),
                                 "dividends": h["Dividends"].to_numpy() if "Dividends" in h else 0.0})
    return None


def _fetch_naver(code: str, end: str) -> pd.DataFrame | None:
    import ast

    import requests
    url = (f"https://api.finance.naver.com/siseJson.naver?symbol={code}&requestType=1&startTime=20000101"
           f"&endTime={end}&timeframe=day")
    try:
        r = requests.get(url, headers={"User-Agent": UA}, timeout=30)
        r.raise_for_status()
        rows = ast.literal_eval("".join(r.text.split()))
    except Exception:  # noqa: BLE001
        return None
    finally:
        time.sleep(0.5)
    if not rows or len(rows) < 2:
        return None
    data = rows[1:]
    return pd.DataFrame({"code": code, "date": [pd.to_datetime(str(x[0]), format="%Y%m%d").strftime("%Y-%m-%d") for x in data],
                         "open": [x[1] for x in data], "high": [x[2] for x in data], "low": [x[3] for x in data],
                         "close": [x[4] for x in data], "volume": [x[5] for x in data]})


def fetch(codes: list[str] = REAL_CODES, folder: Path | str = REAL_DIR) -> pd.DataFrame:
    """Download Naver closes and Yahoo distributions for `codes` into `folder`. Returns a per-code summary."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    end = (pd.Timestamp.today() + pd.Timedelta(days=7)).strftime("%Y%m%d")
    nav, yah, summary = [], [], []
    for code in codes:
        n, y = _fetch_naver(code, end), _fetch_yahoo(code)
        if n is not None:
            nav.append(n)
        if y is not None:
            yah.append(y)
        summary.append({"code": code, "naver_rows": 0 if n is None else len(n), "naver_first": None if n is None else n["date"].min(),
                        "naver_last": None if n is None else n["date"].max(), "yahoo_rows": 0 if y is None else len(y),
                        "distributions": 0 if y is None else int((y["dividends"] > 0).sum())})
        print(code, summary[-1], flush=True)
    if not nav:
        raise RuntimeError("네이버에서 가격을 하나도 받지 못했습니다.")
    pd.concat(nav, ignore_index=True).to_csv(folder / "naver.csv.gz", index=False)
    if yah:
        pd.concat(yah, ignore_index=True).to_csv(folder / "yahoo.csv.gz", index=False)
    s = pd.DataFrame(summary)
    s.to_csv(folder / "summary.csv", index=False)
    return s


if __name__ == "__main__":
    if sys.argv[1:2] == ["fetch"]:
        print(fetch())
    else:
        raise SystemExit("usage: python -m lab.aa_real fetch")
