"""Price data for the research lab.

Sources:
- the frozen ETF snapshot already in the repository (offline, 2016-08 .. 2026-07);
- yfinance downloads (ETFs or stocks), cached per symbol in lab_data/yf/.

All prices are split/dividend adjusted OHLC with raw volume (yfinance auto_adjust), so
returns include distributions.

Every panel also carries a daily cash return series (what uninvested cash earns):
the T-bill ETF BIL's total return, and before BIL existed (2007-05) the 13-week
Treasury bill rate (^IRX) / 252.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SNAPSHOT_DIR = ROOT / "docs" / "research" / "trend_v2" / "phase_a2"
CACHE_DIR = ROOT / "lab_data"
FIELDS = ("open", "high", "low", "close", "volume")
BENCHMARK = "SPY"
CASH_ETF = "BIL"
TBILL_RATE = "^IRX"  # 13-week T-bill yield, percent per year

# ETFs listed by 2004 (most by 2001), so tests include the 2000-02 and 2008 bear markets.
# One or two liquid representatives per asset class; chosen by history length, not performance.
LONG_HISTORY_ETFS: dict[str, list[str]] = {
    "미국 주식": ["SPY", "QQQ", "DIA", "MDY", "IWM", "IWD", "IWF"],
    "섹터·테마 주식": ["XLB", "XLE", "XLF", "XLI", "XLK", "XLP", "XLU", "XLV", "XLY", "IBB", "SMH"],
    "해외 주식": ["EFA", "EEM", "EWJ", "EWG", "EWU", "EWZ", "EWY", "EWT", "EWC", "EWA", "EWH", "FXI", "ILF"],
    "채권": ["TLT", "IEF", "LQD", "TIP", "AGG"],
    "원자재": ["GLD"],
    "부동산": ["IYR"],
}


@dataclass
class Panel:
    """Wide price tables: one row per trading date, one column per symbol."""

    open: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    close: pd.DataFrame
    volume: pd.DataFrame
    info: pd.DataFrame = field(default_factory=pd.DataFrame)  # symbol, name, asset_group
    source: str = ""
    tradable: list[str] | None = None  # None = every symbol; the benchmark may be data-only
    cash: pd.Series | None = None  # daily return of uninvested cash, indexed like close
    kind: str = ""  # "snapshot" | "yfinance": how to rebuild this panel later

    @property
    def tradable_symbols(self) -> list[str]:
        return self.symbols if self.tradable is None else [s for s in self.symbols if s in set(self.tradable)]

    @property
    def symbols(self) -> list[str]:
        return list(self.close.columns)

    @property
    def first_date(self) -> pd.Timestamp:
        return self.close.index[0]

    @property
    def last_date(self) -> pd.Timestamp:
        return self.close.index[-1]

    def subset(self, symbols: list[str]) -> "Panel":
        """Tradable universe = symbols; the benchmark is always kept as data."""
        wanted = set(symbols)
        keep = [s for s in self.close.columns if s in wanted or s == BENCHMARK]
        info = self.info[self.info["symbol"].isin(keep)] if not self.info.empty else self.info
        return Panel(
            *(getattr(self, f)[keep] for f in FIELDS), info=info.reset_index(drop=True), source=self.source,
            tradable=[s for s in keep if s in wanted], cash=self.cash, kind=self.kind,
        )


def panel_from_long(df: pd.DataFrame, info: pd.DataFrame | None = None, source: str = "") -> Panel:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["date", "symbol"], keep="last")
    tables = {f: df.pivot(index="date", columns="symbol", values=f).sort_index() for f in FIELDS}
    if info is None:
        info = pd.DataFrame({"symbol": tables["close"].columns, "name": "", "asset_group": ""})
    return Panel(**tables, info=info.reset_index(drop=True), source=source)


def cash_from_closes(index: pd.DatetimeIndex, bil_close: pd.Series | None, irx_close: pd.Series | None) -> pd.Series:
    """Daily cash return: BIL total return where BIL exists, otherwise T-bill rate / 252."""
    out = pd.Series(0.0, index=index)
    if irx_close is not None and len(irx_close):
        out = (irx_close.reindex(index).ffill().fillna(0.0) / 100.0 / 252.0).rename(None)
    if bil_close is not None and len(bil_close):
        bil = bil_close.reindex(index).ffill()
        r = bil.pct_change()
        has = bil.shift(1).notna() & r.notna()
        out = out.where(~has, r)
    return out.fillna(0.0)


# --------------------------------------------------------------------------- snapshot

def snapshot_available() -> bool:
    return (SNAPSHOT_DIR / "prices").is_dir()


def load_snapshot() -> Panel:
    """Load the frozen ETF snapshot. A pickle cache in lab_data/ makes later loads fast."""
    cache = CACHE_DIR / "snapshot_panel_v2.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    shards = sorted((SNAPSHOT_DIR / "prices").glob("*.csv.gz"))
    long = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    universe = pd.read_csv(SNAPSHOT_DIR / "universe_snapshot.csv", usecols=["symbol", "name", "asset_group", "aum"])
    present = set(long["symbol"])
    info = universe[universe["symbol"].isin(present)].drop_duplicates("symbol")
    panel = panel_from_long(long, info=info, source="ETF 스냅샷 (2016-08-01 – 2026-07-30)")
    panel.kind = "snapshot"
    panel.cash = cash_from_closes(panel.close.index, panel.close.get(CASH_ETF), None)
    CACHE_DIR.mkdir(exist_ok=True)
    pd.to_pickle(panel, cache)
    return panel


# --------------------------------------------------------------------------- yfinance

def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / "yf" / f"{symbol.replace('^', '_')}.csv.gz"


def _download_one(sym: str, want_end: pd.Timestamp, refresh: bool) -> pd.DataFrame | None:
    import yfinance as yf

    from src.prices import filter_completed_daily_bars, yahoo_symbol

    path = _cache_path(sym)
    if path.exists() and not refresh:
        cached = pd.read_csv(path, parse_dates=["date"])
        # files hold the full history from the last download, so only freshness matters
        if cached["date"].max() >= min(want_end, pd.Timestamp.today().normalize() - pd.Timedelta(days=4)):
            return cached
    raw = yf.download(yahoo_symbol(sym), period="max", auto_adjust=True, actions=False,
                      progress=False, threads=False, multi_level_index=False)
    if raw is None or raw.empty:
        return None
    raw = raw.reset_index().rename(columns=str.lower)
    raw["symbol"] = sym
    raw = raw[["date", "symbol", *FIELDS]].dropna(subset=["close"])
    raw, _ = filter_completed_daily_bars(raw)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw.to_csv(path, index=False)
    raw["date"] = pd.to_datetime(raw["date"])
    return raw


def download_panel(symbols: list[str], end: str, refresh: bool = False, source: str = "yfinance 직접 다운로드") -> tuple[Panel, list[str]]:
    """Download (or reuse cached) daily bars for symbols. Returns (panel, failed_symbols).

    The benchmark is always added as data; BIL and the T-bill rate are fetched for the
    cash series only (they are tradable only if requested explicitly).
    """
    requested = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    want_end = pd.Timestamp(end)
    frames, failed = [], []
    for sym in requested + ([BENCHMARK] if BENCHMARK not in requested else []):
        df = _download_one(sym, want_end, refresh)
        if df is None:
            failed.append(sym)
        else:
            frames.append(df)
    if not frames:
        raise ValueError("가격 데이터를 하나도 받지 못했습니다.")
    panel = panel_from_long(pd.concat(frames, ignore_index=True), source=source)
    panel.tradable = [s for s in requested if s in panel.symbols]
    panel.kind = "yfinance"
    bil = _download_one(CASH_ETF, want_end, refresh)
    irx = _download_one(TBILL_RATE, want_end, refresh)
    panel.cash = cash_from_closes(
        panel.close.index,
        bil.set_index("date")["close"] if bil is not None else None,
        irx.set_index("date")["close"] if irx is not None else None,
    )
    return panel, failed
