"""CRSP daily stock file -> lab Panel with a point-in-time universe (DECISIONS 70).

Input: the user's CRSP export, one parquet (or csv) per 5-year block, columns as in the CRSP
daily stock file (PERMNO, date, SHRCD, EXCHCD, TICKER, PRC, VOL, RET, SHROUT, CFACPR, CFACSHR,
OPENPRC, BIDLO, ASKHI, DLSTCD, DLRET). The data is licensed: it is read where the user keeps
it and the built panel is written next to it or under lab_data/, never into the repository.

Universe: common stocks (SHRCD 10, 11) on NYSE, AMEX and Nasdaq (EXCHCD 1, 2, 3). At each
year-end the TOP_N by market value (|PRC| x SHROUT) are the entry candidates of the following
year ("eligible"). A stock that drops out may still be held until its exit rule fires.

Prices: total-return adjusted like yfinance's Adj Close. The chain prod(1 + RET) is anchored so
that the last price equals |PRC| / CFACPR; open, high and low are scaled by the same factor.
Volume is VOL x CFACPR so that close x volume stays the day's dollar volume. Days without a
trade (PRC < 0, a bid/ask midpoint) get high = low = close.

Delisting: on the trading day after a stock's last CRSP row with a delisting code, one
synthetic bar is added at last close x (1 + DLRET), so the engine's forced "delist" exit sells
at that price. A missing DLRET is -30% for performance delistings (codes 500, 520-584,
Shumway 1997) and 0 otherwise. The last two bars are never entry candidates.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from .data import BENCHMARK, FIELDS, Panel

SPY_PERMNO = "84398"
COMMON_SHRCD = {"10", "11"}
EXCHANGES = {"1", "2", "3"}
COLUMNS = ["PERMNO", "date", "SHRCD", "EXCHCD", "TICKER", "PRC", "VOL", "RET", "SHROUT", "CFACPR",
           "OPENPRC", "BIDLO", "ASKHI", "DLSTCD", "DLRET"]
TOP_N = 1000


def source_files(src: Path) -> list[Path]:
    files = sorted(src.glob("CRSP_*.parquet"))
    if not files:
        files = sorted(src.glob("CRSP_*.csv"))
    if not files:
        raise FileNotFoundError(f"CRSP_*.parquet 또는 CRSP_*.csv 파일이 없습니다: {src}")
    return files


def read_block(path: Path, permnos: set[str] | None = None, columns: list[str] = COLUMNS) -> pd.DataFrame:
    """One 5-year file, common stocks on the three exchanges plus SPY, optionally only `permnos`."""
    if path.suffix == ".parquet":
        import pyarrow.parquet as pq

        df = pq.read_table(path, columns=columns).to_pandas()
    else:
        df = pd.read_csv(path, usecols=columns, dtype={"PERMNO": str, "SHRCD": str, "EXCHCD": str, "TICKER": str},
                         na_values=["", "A", "B", "C", "P", "S", "T"], low_memory=False)
    for c in ("PERMNO", "SHRCD", "EXCHCD"):
        df[c] = df[c].astype(str).str.replace(r"\.0$", "", regex=True)
    keep = (df["SHRCD"].isin(COMMON_SHRCD) & df["EXCHCD"].isin(EXCHANGES)) | (df["PERMNO"] == SPY_PERMNO)
    if permnos is not None:
        keep &= df["PERMNO"].isin(permnos)
    df = df.loc[keep, columns].copy()
    df["date"] = pd.to_datetime(df["date"])
    for c in ("PRC", "VOL", "RET", "SHROUT", "CFACPR", "OPENPRC", "BIDLO", "ASKHI", "DLSTCD", "DLRET"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
    return df


def year_end_caps(files: list[Path]) -> pd.DataFrame:
    """Market value (|PRC| x SHROUT, $ thousands) of every common stock at each year-end."""
    parts = []
    for f in files:
        df = read_block(f, columns=["PERMNO", "date", "SHRCD", "EXCHCD", "PRC", "SHROUT"])
        df = df[df["PERMNO"] != SPY_PERMNO]
        df["year"] = df["date"].dt.year
        last = df.sort_values("date").groupby(["PERMNO", "year"], sort=False).tail(1)
        last = last[last["date"].dt.month == 12]  # must still be listed in December
        last["cap"] = last["PRC"].abs() * last["SHROUT"]
        parts.append(last[["PERMNO", "year", "cap"]])
    return pd.concat(parts, ignore_index=True).dropna()


def membership(caps: pd.DataFrame, top_n: int) -> dict[int, list[str]]:
    """{year: PERMNOs eligible to be bought during that year} = top_n of the previous year-end."""
    out = {}
    for year, g in caps.groupby("year"):
        out[int(year) + 1] = g.nlargest(top_n, "cap")["PERMNO"].tolist()
    return out


def adjust(df: pd.DataFrame) -> pd.DataFrame:
    """Total-return adjusted OHLCV per PERMNO (long form) plus the delisting bar."""
    df = df.sort_values(["PERMNO", "date"]).reset_index(drop=True)
    price = df["PRC"].abs()
    split_close = price / df["CFACPR"]
    ret = df["RET"].fillna(0.0)
    chain = (1.0 + ret).groupby(df["PERMNO"]).cumprod()
    last_chain = chain.groupby(df["PERMNO"]).transform("last")
    last_split = split_close.groupby(df["PERMNO"]).transform("last")
    adj_close = chain / last_chain * last_split
    factor = adj_close / split_close
    traded = df["PRC"] > 0
    out = pd.DataFrame({
        "symbol": df["PERMNO"], "date": df["date"],
        "close": adj_close,
        "open": (df["OPENPRC"] / df["CFACPR"] * factor).where(traded),
        "high": (df["ASKHI"] / df["CFACPR"] * factor).where(traded, adj_close),
        "low": (df["BIDLO"] / df["CFACPR"] * factor).where(traded, adj_close),
        "volume": (df["VOL"] * df["CFACPR"]).fillna(0.0),
    })
    out["open"] = out["open"].fillna(out["close"].where(~traded))  # untraded day: open = midpoint
    # delisting bar
    last_rows = df.groupby("PERMNO").tail(1)
    delisted = last_rows[last_rows["DLSTCD"].notna() & (last_rows["DLSTCD"] >= 200)]
    if len(delisted):
        code = delisted["DLSTCD"]
        perf = (code == 500) | ((code >= 520) & (code <= 584))
        dlret = delisted["DLRET"].where(delisted["DLRET"].notna(), np.where(perf, -0.30, 0.0))
        px = out.loc[delisted.index, "close"].to_numpy() * (1.0 + dlret.to_numpy())
        bars = pd.DataFrame({"symbol": delisted["PERMNO"].to_numpy(), "date": pd.NaT, "close": px, "open": px,
                             "high": px, "low": px, "volume": 0.0, "_after": delisted["date"].to_numpy()})
        out = pd.concat([out, bars], ignore_index=True)
    return out


def build_panel(src: Path, top_n: int = TOP_N, years: tuple[int, int] | None = None,
                files: list[Path] | None = None, log=print) -> tuple[Panel, pd.DataFrame, dict]:
    """Build the point-in-time panel. Returns (panel without cash, info, meta)."""
    files = files or source_files(src)
    log(f"[1/3] 연말 시가총액 계산: 파일 {len(files)}개")
    caps = year_end_caps(files)
    if years:
        caps = caps[(caps["year"] >= years[0] - 1) & (caps["year"] <= years[1])]
    members = membership(caps, top_n)
    wanted = {p for ps in members.values() for p in ps} | {SPY_PERMNO}
    log(f"[2/3] 가격 읽기: 누적 {len(wanted) - 1}개 종목 (연도별 상위 {top_n})")
    long = pd.concat([read_block(f, wanted) for f in files], ignore_index=True)
    if years:
        long = long[(long["date"].dt.year >= years[0]) & (long["date"].dt.year <= years[1])]
    tickers = long.sort_values("date").groupby("PERMNO")["TICKER"].last()
    dl = long.sort_values("date").groupby("PERMNO").tail(1).set_index("PERMNO")["DLSTCD"]
    dl = dl.where(dl >= 200)  # 100 = still listed
    adj = adjust(long)
    calendar = pd.DatetimeIndex(sorted(adj["date"].dropna().unique()))
    bars = adj["date"].isna()
    if bars.any():  # place each delisting bar on the next calendar day
        pos = np.searchsorted(calendar.values, adj.loc[bars, "_after"].to_numpy(), side="right")
        ok = pos < len(calendar)
        adj.loc[bars[bars].index[ok], "date"] = calendar[pos[ok]]
        adj = adj[adj["date"].notna()]
    adj = adj.drop(columns="_after", errors="ignore")
    log("[3/3] 표 만들기")
    tables = {f: adj.pivot(index="date", columns="symbol", values=f).sort_index().astype("float32") for f in FIELDS}
    symbols = list(tables["close"].columns)
    close = tables["close"]
    eligible = pd.DataFrame(False, index=close.index, columns=symbols)
    yrs = close.index.year
    for year, ps in members.items():
        cols = [p for p in ps if p in eligible.columns]
        eligible.loc[yrs == year, cols] = True
    eligible &= close.notna()
    # a symbol whose series ends before the data does: never buy on its last two bars (the
    # delisting bar and the day before it)
    has = close.notna()
    n_after = has.iloc[::-1].cumsum().iloc[::-1]  # bars remaining incl. today
    ends_early = has.iloc[::-1].idxmax() < close.index[-1]
    eligible &= (n_after > 2) | ~ends_early
    eligible[SPY_PERMNO] = False
    info = pd.DataFrame({"symbol": symbols, "name": [tickers.get(s, "") for s in symbols], "asset_group": "주식",
                         "delist_code": [dl.get(s, np.nan) for s in symbols]})
    meta = {"top_n": top_n, "years": sorted(members), "files": [f.name for f in files], "symbols": len(symbols) - 1,
            "first": str(close.index[0].date()), "last": str(close.index[-1].date())}
    panel = Panel(**tables, info=info, source=f"CRSP 시점별 시가총액 상위 {top_n} (보통주)", tradable=[s for s in symbols if s != SPY_PERMNO],
                  kind="crsp", eligible=eligible, delist_exit=True)
    panel = rename_benchmark(panel)
    return panel, info, meta


def rename_benchmark(panel: Panel) -> Panel:
    """The engine looks for the column named SPY."""
    for f in FIELDS:
        getattr(panel, f).rename(columns={SPY_PERMNO: BENCHMARK}, inplace=True)
    if panel.eligible is not None:
        panel.eligible.rename(columns={SPY_PERMNO: BENCHMARK}, inplace=True)
    panel.info.loc[panel.info["symbol"] == SPY_PERMNO, "symbol"] = BENCHMARK
    return panel


def save_panel(panel: Panel, info: pd.DataFrame, meta: dict, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    for f in FIELDS:
        getattr(panel, f).to_parquet(out / f"{f}.parquet")
    panel.eligible.to_parquet(out / "eligible.parquet")
    info.to_csv(out / "info.csv", index=False, encoding="utf-8")
    (out / "meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")


def load_panel(out: Path) -> Panel:
    tables = {f: pd.read_parquet(out / f"{f}.parquet") for f in FIELDS}
    eligible = pd.read_parquet(out / "eligible.parquet")
    info = pd.read_csv(out / "info.csv", dtype={"symbol": str, "name": str})
    meta = json.loads((out / "meta.json").read_text(encoding="utf-8"))
    symbols = list(tables["close"].columns)
    return Panel(**tables, info=info, source=f"CRSP 시점별 시가총액 상위 {meta['top_n']} (보통주)",
                 tradable=[s for s in symbols if s != BENCHMARK], kind="crsp", eligible=eligible, delist_exit=True)


def main(argv: list[str]) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, help="CRSP_*.parquet 파일이 있는 폴더")
    ap.add_argument("--out", required=True, help="패널을 저장할 폴더 (저장소 밖)")
    ap.add_argument("--top", type=int, default=TOP_N)
    ap.add_argument("--years", type=int, nargs=2, metavar=("FIRST", "LAST"), default=None)
    a = ap.parse_args(argv)
    panel, info, meta = build_panel(Path(a.src), a.top, tuple(a.years) if a.years else None)
    save_panel(panel, info, meta, Path(a.out))
    print(f"저장: {a.out} · 종목 {meta['symbols']} · {meta['first']} – {meta['last']}")


if __name__ == "__main__":
    main(sys.argv[1:])
