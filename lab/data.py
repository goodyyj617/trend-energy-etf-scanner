"""Price data for the research lab.

Two sources:
- the frozen ETF snapshot already in the repository (offline, 2016-08 .. 2026-07);
- yfinance downloads for user-entered tickers (ETFs or stocks), cached locally.

All prices are split/dividend adjusted OHLC with raw volume (yfinance auto_adjust).
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
            tradable=[s for s in keep if s in wanted],
        )


def panel_from_long(df: pd.DataFrame, info: pd.DataFrame | None = None, source: str = "") -> Panel:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates(["date", "symbol"], keep="last")
    tables = {f: df.pivot(index="date", columns="symbol", values=f).sort_index() for f in FIELDS}
    if info is None:
        info = pd.DataFrame({"symbol": tables["close"].columns, "name": "", "asset_group": ""})
    return Panel(**tables, info=info.reset_index(drop=True), source=source)


# --------------------------------------------------------------------------- snapshot

def snapshot_available() -> bool:
    return (SNAPSHOT_DIR / "prices").is_dir()


def load_snapshot() -> Panel:
    """Load the frozen ETF snapshot. A pickle cache in lab_data/ makes later loads fast."""
    cache = CACHE_DIR / "snapshot_panel.pkl"
    if cache.exists():
        return pd.read_pickle(cache)
    shards = sorted((SNAPSHOT_DIR / "prices").glob("*.csv.gz"))
    long = pd.concat([pd.read_csv(p) for p in shards], ignore_index=True)
    universe = pd.read_csv(SNAPSHOT_DIR / "universe_snapshot.csv", usecols=["symbol", "name", "asset_group", "aum"])
    present = set(long["symbol"])
    info = universe[universe["symbol"].isin(present)].drop_duplicates("symbol")
    panel = panel_from_long(long, info=info, source="ETF 스냅샷 (Phase A2, 2016-08-01 ~ 2026-07-30)")
    CACHE_DIR.mkdir(exist_ok=True)
    pd.to_pickle(panel, cache)
    return panel


# --------------------------------------------------------------------------- yfinance

def _cache_path(symbol: str) -> Path:
    return CACHE_DIR / "yf" / f"{symbol}.csv.gz"


def download_panel(symbols: list[str], end: str, refresh: bool = False) -> tuple[Panel, list[str]]:
    """Download (or reuse cached) daily bars for symbols. Returns (panel, failed_symbols).

    The cache stores each symbol's full history; a cached file is reused while its last bar
    is recent (within 4 days of today, or reaching `end`), otherwise the symbol is downloaded again.
    """
    import yfinance as yf

    from src.prices import filter_completed_daily_bars, yahoo_symbol

    requested = list(dict.fromkeys(s.strip().upper() for s in symbols if s.strip()))
    symbols = requested + ([BENCHMARK] if BENCHMARK not in requested else [])
    want_end = pd.Timestamp(end)
    frames, failed = [], []
    (CACHE_DIR / "yf").mkdir(parents=True, exist_ok=True)
    for sym in symbols:
        path = _cache_path(sym)
        cached = None
        if path.exists() and not refresh:
            cached = pd.read_csv(path, parse_dates=["date"])
            # files hold the full history from the last download, so only freshness matters
            fresh = cached["date"].max() >= min(want_end, pd.Timestamp.today().normalize() - pd.Timedelta(days=4))
            if not fresh:
                cached = None
        if cached is None:
            raw = yf.download(yahoo_symbol(sym), period="max", auto_adjust=True, actions=False,
                              progress=False, threads=False, multi_level_index=False)
            if raw is None or raw.empty:
                failed.append(sym)
                continue
            raw = raw.reset_index().rename(columns=str.lower)
            raw["symbol"] = sym
            raw = raw[["date", "symbol", *FIELDS]].dropna(subset=["close"])
            raw, _ = filter_completed_daily_bars(raw)
            raw.to_csv(path, index=False)
            cached = raw
            cached["date"] = pd.to_datetime(cached["date"])
        frames.append(cached)
    if not frames:
        raise ValueError("가격 데이터를 하나도 받지 못했습니다.")
    long = pd.concat(frames, ignore_index=True)
    panel = panel_from_long(long, source="yfinance 직접 다운로드")
    panel.tradable = [s for s in requested if s in panel.symbols]
    return panel, failed
