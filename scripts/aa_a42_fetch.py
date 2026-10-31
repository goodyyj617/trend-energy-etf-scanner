"""A42 stage 0 data (allocation DECISIONS A42 §0): Korean rates, sealed 1990-01 – 2000-08 series, new product prices.

    .venv/Scripts/python.exe scripts/aa_a42_fetch.py rates      # ECOS monthly CD91 / KTB → lab/aa_kr_rates.csv (2000-09 –)
    .venv/Scripts/python.exe scripts/aa_a42_fetch.py sealed     # 1990-01 – 2000-08 → Drive lab_exchange/a42_sealed/ + hashes
    .venv/Scripts/python.exe scripts/aa_a42_fetch.py gold       # monthly gold 1990-01 – 2000-08 → sealed folder (datahub gold-prices)
    .venv/Scripts/python.exe scripts/aa_a42_fetch.py prices     # Naver/Yahoo for the A42 products (lab_data/kr, not committed)

Sealing (design §0-5): everything dated before 2000-09-01 goes only to the sealed folder on Drive. This script never
prints values from it; it writes the files and their SHA-256 into
docs/research/allocation/2026-10-29_a42_stage0/sealed_hashes.csv. lab/aa_sealed.py refuses to read the folder unless
the caller passes unseal=True (stage 5, `--unseal-a42`).

ECOS is read with the public 'sample' key (10 rows per call), monthly table 721Y001; the daily table would need
thousands of calls. FRED and yfinance need no key.
"""
from __future__ import annotations

import hashlib
import io
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lab import aa_sealed  # noqa: E402

RATES_PATH = ROOT / "lab" / "aa_kr_rates.csv"
HASHES = ROOT / "docs" / "research" / "allocation" / "2026-10-29_a42_stage0" / "sealed_hashes.csv"
ECOS = "https://ecos.bok.or.kr/api/StatisticSearch/sample/json/kr/{a}/{b}/721Y001/M/{start}/{end}/{item}"
# column -> ECOS 721Y001 item (monthly averages, % a year)
ECOS_ITEMS = {"cd91": "2010000", "ktb3": "5020000", "ktb5": "5040000", "ktb10": "5050000", "ktb20": "5060000", "ktb30": "5070000"}
FRED = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={sid}"
SEALED_FRED = ["DGS10", "DGS30", "DTB3", "DEXKOUS", "KORCPIALLMINMEI"]
SEALED_YF = ["VFINX", "^NDX", "VFITX", "VUSTX", "FDFAX", "FSPHX", "FSUTX", "VFISX"]
UA = {"User-Agent": "curl/8.0"}  # FRED stalls on browser user agents


def _get(url: str, **kw):
    import requests
    for k in range(4):
        try:
            r = requests.get(url, headers=UA, timeout=120, **kw)
            r.raise_for_status()
            return r
        except Exception:  # noqa: BLE001 - retry with backoff, then give up
            time.sleep(2 ** (k + 1))
    raise RuntimeError(f"받지 못함: {url}")


def ecos_monthly(item: str, start: str = "199001", end: str = "202612") -> pd.Series:
    """One ECOS 721Y001 monthly series, paged 10 rows at a time (sample key)."""
    rows, a = [], 1
    while True:
        js = _get(ECOS.format(a=a, b=a + 9, start=start, end=end, item=item)).json()
        block = js.get("StatisticSearch", {})
        part = block.get("row", [])
        rows += part
        total = int(block.get("list_total_count", 0) or 0)
        a += 10
        if not part or a > total:
            break
        time.sleep(0.15)
    s = pd.Series({pd.Period(r["TIME"], "M").to_timestamp(): float(r["DATA_VALUE"]) for r in rows}, dtype=float)
    return s.sort_index()


def rates() -> None:
    cols = {k: ecos_monthly(v) for k, v in ECOS_ITEMS.items()}
    df = pd.DataFrame(cols)
    df.index.name = "month"
    research = df[df.index >= pd.Timestamp(aa_sealed.SEAL_END) + pd.Timedelta(days=1) - pd.offsets.MonthBegin(1)]
    research = research[research.index >= pd.Timestamp("2000-09-01")]
    research.to_csv(RATES_PATH, float_format="%.4f")
    sealed = df[df.index < pd.Timestamp("2000-09-01")]
    _write_sealed("kr_rates_monthly.csv", sealed.to_csv(float_format="%.4f"))
    print(f"aa_kr_rates.csv: {len(research)}개월, 열 {list(research.columns)}; 봉인 쪽 {len(sealed)}개월 (값은 보지 않음)")


def _write_sealed(name: str, text: str) -> None:
    folder = aa_sealed.SEALED_DIR
    folder.mkdir(parents=True, exist_ok=True)
    data = text.encode("utf-8")
    (folder / name).write_bytes(data)
    HASHES.parent.mkdir(parents=True, exist_ok=True)
    old = pd.read_csv(HASHES) if HASHES.exists() else pd.DataFrame(columns=["file", "sha256", "bytes", "written"])
    old = old[old["file"] != name]
    row = {"file": name, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "written": pd.Timestamp.now().isoformat(timespec="seconds")}
    pd.concat([old, pd.DataFrame([row])], ignore_index=True).to_csv(HASHES, index=False)


def sealed() -> None:
    """1990-01-02 – 2000-08-31 only (and nothing after): FRED and yfinance series, written unread to Drive."""
    lo, hi = pd.Timestamp(aa_sealed.SEAL_START), pd.Timestamp(aa_sealed.SEAL_END)
    report = []
    for sid in SEALED_FRED:
        df = pd.read_csv(io.StringIO(_get(FRED.format(sid=sid)).text))
        df.columns = ["date", sid]
        d = pd.to_datetime(df["date"])
        part = df[(d >= lo) & (d <= hi)]
        _write_sealed(f"fred_{sid}.csv", part.to_csv(index=False))
        report.append((sid, len(part)))
    import yfinance as yf
    for t in SEALED_YF:
        h = yf.Ticker(t).history(start=str(lo.date()), end=str((hi + pd.Timedelta(days=1)).date()), auto_adjust=True)
        if h is None or h.empty:
            report.append((t, 0))
            continue
        idx = pd.DatetimeIndex(h.index)
        idx = idx.tz_localize(None) if idx.tz is not None else idx
        part = pd.DataFrame({"date": idx.strftime("%Y-%m-%d"), "close": h["Close"].to_numpy()})
        part = part[(pd.to_datetime(part["date"]) >= lo) & (pd.to_datetime(part["date"]) <= hi)]
        _write_sealed(f"yf_{t.replace('^', '_')}.csv", part.to_csv(index=False))
        report.append((t, len(part)))
        time.sleep(0.5)
    print("봉인 시리즈 (행 수만):", report)
    print("금: 일간 출처는 없음(FRED LBMA 시리즈 삭제됨). 월간은 `gold` 명령으로 따로 받는다.")


FX_ECOS_PATH = ROOT / "lab" / "aa_usdkrw_ecos.csv.gz"
ECOS_DAILY = "https://ecos.bok.or.kr/api/StatisticSearch/sample/json/kr/{a}/{b}/731Y001/D/{start}/{end}/0000001"


def fx() -> None:
    """A42 0-8: daily Seoul 매매기준율 (ECOS 731Y001, KRW per USD), 2000-09-01 – 2026-10-27, 10 rows per call."""
    rows, a = [], 1
    while True:
        js = _get(ECOS_DAILY.format(a=a, b=a + 9, start="20000901", end="20261027")).json()
        block = js.get("StatisticSearch", {})
        part = block.get("row", [])
        rows += part
        total = int(block.get("list_total_count", 0) or 0)
        a += 10
        if not part or a > total:
            break
        time.sleep(0.1)
    df = pd.DataFrame({"date": [pd.to_datetime(r["TIME"]).strftime("%Y-%m-%d") for r in rows],
                       "krw_per_usd": [float(r["DATA_VALUE"]) for r in rows]})
    df.to_csv(FX_ECOS_PATH, index=False)
    print(f"매매기준율 {len(df)}일 → {FX_ECOS_PATH.name}")


GOLD_MONTHLY = "https://raw.githubusercontent.com/datasets/gold-prices/main/data/monthly.csv"  # datahub.io gold-prices (USD/oz, monthly)


def sealed_gold() -> None:
    """Monthly gold price (USD per troy ounce, datahub 'gold-prices') for 1990-01 – 2000-08 only, written unread."""
    df = pd.read_csv(io.StringIO(_get(GOLD_MONTHLY).text))
    d = pd.to_datetime(df["Date"].astype(str) + "-01")
    part = df[(d >= pd.Timestamp("1990-01-01")) & (d <= pd.Timestamp(aa_sealed.SEAL_END))]
    _write_sealed("gold_monthly.csv", part.to_csv(index=False))
    print(f"봉인 금 (월간): {len(part)}개월 (값은 보지 않음)")


def prices() -> None:
    from lab import aa_real
    s = aa_real.fetch()
    print(s.to_string())


if __name__ == "__main__":
    cmd = sys.argv[1:2]
    {"rates": rates, "sealed": sealed, "gold": sealed_gold, "prices": prices, "fx": fx}.get(cmd[0] if cmd else "", lambda: (_ for _ in ()).throw(SystemExit(__doc__)))()
