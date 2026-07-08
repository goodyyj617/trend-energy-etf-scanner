from __future__ import annotations

import io
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import requests
import yaml
import yfinance as yf


ROOT = Path(__file__).resolve().parents[1]

CONFIG_DIR = ROOT / "config"
AUM_PATH = CONFIG_DIR / "aum.csv"
UNIVERSE_YML_PATH = CONFIG_DIR / "universe.yml"
MANUAL_OVERRIDES_PATH = CONFIG_DIR / "manual_overrides.csv"

NASDAQ_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_LISTED_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"


DEFAULT_AUM_COLUMNS = [
    "symbol",
    "yf_symbol",
    "name",
    "aum",
    "aum_date",
    "source",
    "status",
    "last_checked",
    "error",
]


OBVIOUS_EXCLUDE_PATTERNS = [
    # leveraged / inverse
    r"\b2X\b",
    r"\b3X\b",
    r"\b-2X\b",
    r"\b-3X\b",
    r"\bULTRA\b",
    r"\bULTRAPRO\b",
    r"\bLEVERAGED\b",
    r"\bDAILY\b.*\bBULL\b",
    r"\bDAILY\b.*\bBEAR\b",
    r"\bINVERSE\b",

    # volatility products
    r"\bVIX\b",
    r"\bVOLATILITY\b",

    # single-stock ETF wording
    r"\bSINGLE[- ]STOCK\b",

    # option income / covered call / target outcome
    r"\bCOVERED CALL\b",
    r"\bBUYWRITE\b",
    r"\bOPTION INCOME\b",
    r"\bBUFFER\b",
    r"\bTARGET OUTCOME\b",
    r"\bDEFINED OUTCOME\b",
    r"\bFLEXIBLE EXCHANGE\b",
]


def utc_today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_universe_config() -> Dict:
    if not UNIVERSE_YML_PATH.exists():
        return {}

    with open(UNIVERSE_YML_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    return data


def get_auto_aum_config() -> Dict:
    cfg = load_universe_config()
    auto = cfg.get("auto_aum", {}) or {}

    return {
        "enabled": bool(auto.get("enabled", True)),

        # 0 means no count limit.
        # This is what you asked for: try all missing symbols when possible.
        "max_new_per_run": int(auto.get("max_new_per_run", 0)),

        # Stop gracefully before GitHub Actions kills the whole job.
        # Partial cache will still be saved.
        "time_budget_minutes": float(auto.get("time_budget_minutes", 75)),

        # Small pause reduces request bursts.
        "sleep_seconds": float(auto.get("sleep_seconds", 0.15)),

        # Existing successful AUM rows are refreshed only after this many days.
        # 0 means refresh every run.
        "refresh_existing_days": int(auto.get("refresh_existing_days", 180)),

        # If true, applies broad name-based exclusions before AUM lookup.
        "skip_obvious_exclusions": bool(auto.get("skip_obvious_exclusions", True)),
    }


def read_pipe_file(url: str) -> pd.DataFrame:
    headers = {
        "User-Agent": "Mozilla/5.0 trend-energy-etf-scanner"
    }
    resp = requests.get(url, headers=headers, timeout=30)
    resp.raise_for_status()

    lines = []
    for line in resp.text.splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith("File Creation Time"):
            continue
        lines.append(line)

    if not lines:
        return pd.DataFrame()

    return pd.read_csv(io.StringIO("\n".join(lines)), sep="|", dtype=str)


def normalize_symbol_for_yfinance(symbol: str) -> str:
    # Yahoo Finance commonly uses "-" where exchanges may use "."
    return str(symbol).strip().upper().replace(".", "-")


def normalize_symbol(symbol: str) -> str:
    return str(symbol).strip().upper()


def fetch_nasdaq_etf_candidates() -> pd.DataFrame:
    frames = []

    # NASDAQ-listed securities
    nasdaq = read_pipe_file(NASDAQ_LISTED_URL)
    if not nasdaq.empty:
        df = nasdaq.copy()
        df["symbol"] = df["Symbol"].map(normalize_symbol)
        df["name"] = df["Security Name"].fillna("")
        df["etf"] = df["ETF"].fillna("")
        df["test_issue"] = df["Test Issue"].fillna("")
        frames.append(df[["symbol", "name", "etf", "test_issue"]])

    # NYSE / NYSE Arca / Cboe / other listed securities
    other = read_pipe_file(OTHER_LISTED_URL)
    if not other.empty:
        df = other.copy()
        df["symbol"] = df["ACT Symbol"].map(normalize_symbol)
        df["name"] = df["Security Name"].fillna("")
        df["etf"] = df["ETF"].fillna("")
        df["test_issue"] = df["Test Issue"].fillna("")
        frames.append(df[["symbol", "name", "etf", "test_issue"]])

    if not frames:
        return pd.DataFrame(columns=["symbol", "yf_symbol", "name"])

    out = pd.concat(frames, ignore_index=True)

    out = out[
        (out["etf"].str.upper() == "Y")
        & (out["test_issue"].str.upper() == "N")
    ].copy()

    out["symbol"] = out["symbol"].map(normalize_symbol)
    out["yf_symbol"] = out["symbol"].map(normalize_symbol_for_yfinance)
    out["name"] = out["name"].fillna("").astype(str)

    out = out.drop_duplicates(subset=["symbol"], keep="first")
    out = out.sort_values("symbol").reset_index(drop=True)

    return out[["symbol", "yf_symbol", "name"]]


def is_obvious_exclusion(symbol: str, name: str) -> bool:
    text = f"{symbol} {name}".upper()

    for pat in OBVIOUS_EXCLUDE_PATTERNS:
        if re.search(pat, text):
            return True

    # Important: do NOT exclude every occurrence of "SHORT".
    # "Short-Term Treasury" and "Ultra Short Bond" can be normal duration/liquidity terms.
    if re.search(r"\bSHORT\b", text):
        safe_short_income_terms = [
            "SHORT TERM",
            "SHORT-TERM",
            "ULTRA SHORT BOND",
            "ULTRASHORT BOND",
            "SHORT DURATION",
            "SHORT TREASURY",
            "SHORT GOVERNMENT",
            "SHORT CORPORATE",
            "SHORT MUNICIPAL",
            "SHORT MUNI",
        ]

        if not any(term in text for term in safe_short_income_terms):
            # This catches names such as "Short S&P 500" but avoids short-duration bond ETFs.
            return True

    return False


def load_manual_overrides() -> pd.DataFrame:
    if not MANUAL_OVERRIDES_PATH.exists():
        return pd.DataFrame()

    try:
        df = pd.read_csv(MANUAL_OVERRIDES_PATH, dtype=str)
    except Exception:
        return pd.DataFrame()

    if "symbol" not in df.columns:
        return pd.DataFrame()

    df["symbol"] = df["symbol"].map(normalize_symbol)
    return df


def apply_manual_override_filter(candidates: pd.DataFrame) -> pd.DataFrame:
    overrides = load_manual_overrides()

    if overrides.empty:
        return candidates

    out = candidates.copy()

    if "force_exclude" in overrides.columns:
        ex = overrides[
            overrides["force_exclude"].fillna("").astype(str).str.lower().isin(["1", "true", "yes", "y"])
        ]["symbol"].tolist()
        if ex:
            out = out[~out["symbol"].isin(ex)].copy()

    # force_include is intentionally not used here to add non-Nasdaq candidates.
    # It should be handled in universe.py if needed.
    return out


def load_aum_cache() -> pd.DataFrame:
    if not AUM_PATH.exists():
        return pd.DataFrame(columns=DEFAULT_AUM_COLUMNS)

    df = pd.read_csv(AUM_PATH, dtype=str)

    for col in DEFAULT_AUM_COLUMNS:
        if col not in df.columns:
            df[col] = ""

    df["symbol"] = df["symbol"].map(normalize_symbol)
    df["yf_symbol"] = df["yf_symbol"].where(
        df["yf_symbol"].fillna("").astype(str).str.len() > 0,
        df["symbol"].map(normalize_symbol_for_yfinance),
    )

    df["aum"] = pd.to_numeric(df["aum"], errors="coerce")

    return df[DEFAULT_AUM_COLUMNS].drop_duplicates(subset=["symbol"], keep="last")


def save_aum_cache(df: pd.DataFrame) -> None:
    out = df.copy()

    for col in DEFAULT_AUM_COLUMNS:
        if col not in out.columns:
            out[col] = ""

    out["symbol"] = out["symbol"].map(normalize_symbol)
    out["yf_symbol"] = out["yf_symbol"].where(
        out["yf_symbol"].fillna("").astype(str).str.len() > 0,
        out["symbol"].map(normalize_symbol_for_yfinance),
    )
    out["aum"] = pd.to_numeric(out["aum"], errors="coerce")

    # Successful rows first, largest AUM first.
    out["_has_aum"] = out["aum"].notna()
    out = out.sort_values(
        by=["_has_aum", "aum", "symbol"],
        ascending=[False, False, True],
    ).drop(columns=["_has_aum"])

    AUM_PATH.parent.mkdir(parents=True, exist_ok=True)
    out[DEFAULT_AUM_COLUMNS].to_csv(AUM_PATH, index=False)


def checked_recently(row: pd.Series, refresh_existing_days: int) -> bool:
    if refresh_existing_days <= 0:
        return False

    last_checked = str(row.get("last_checked", "") or "")[:10]
    if not last_checked:
        return False

    try:
        last = datetime.strptime(last_checked, "%Y-%m-%d").date()
        today = datetime.strptime(utc_today(), "%Y-%m-%d").date()
    except Exception:
        return False

    return (today - last).days < refresh_existing_days


def fetch_aum_from_yfinance(yf_symbol: str) -> Tuple[Optional[float], str, str]:
    """
    Returns:
        aum, display_name, error
    """
    try:
        ticker = yf.Ticker(yf_symbol)
        info = ticker.info or {}

        aum = None
        for key in ["totalAssets", "netAssets"]:
            value = info.get(key)
            if value is not None:
                try:
                    value_float = float(value)
                    if value_float > 0:
                        aum = value_float
                        break
                except Exception:
                    pass

        display_name = (
            info.get("longName")
            or info.get("shortName")
            or info.get("quoteType")
            or ""
        )

        if aum is None:
            return None, str(display_name or ""), "missing_totalAssets"

        return aum, str(display_name or ""), ""

    except Exception as e:
        return None, "", str(e)[:300]


def build_worklist(candidates: pd.DataFrame, cache: pd.DataFrame, cfg: Dict) -> pd.DataFrame:
    candidates = candidates.copy()

    if cfg["skip_obvious_exclusions"]:
        mask = candidates.apply(
            lambda r: is_obvious_exclusion(r["symbol"], r["name"]),
            axis=1,
        )
        candidates = candidates[~mask].copy()

    candidates = apply_manual_override_filter(candidates)

    cache_map = cache.set_index("symbol", drop=False)

    rows = []
    for _, c in candidates.iterrows():
        symbol = c["symbol"]

        if symbol not in cache_map.index:
            rows.append(c)
            continue

        existing = cache_map.loc[symbol]

        # If duplicate index somehow appears, take last row.
        if isinstance(existing, pd.DataFrame):
            existing = existing.iloc[-1]

        has_aum = pd.notna(existing.get("aum")) and float(existing.get("aum")) > 0
        if not has_aum:
            rows.append(c)
            continue

        if not checked_recently(existing, cfg["refresh_existing_days"]):
            rows.append(c)

    if not rows:
        return pd.DataFrame(columns=["symbol", "yf_symbol", "name"])

    work = pd.DataFrame(rows).drop_duplicates(subset=["symbol"], keep="first")
    work = work.sort_values("symbol").reset_index(drop=True)

    max_new = cfg["max_new_per_run"]
    if max_new > 0:
        work = work.head(max_new).copy()

    return work


def upsert_cache_row(
    cache: pd.DataFrame,
    symbol: str,
    yf_symbol: str,
    name: str,
    aum: Optional[float],
    status: str,
    error: str = "",
) -> pd.DataFrame:
    now = utc_now_iso()

    row = {
        "symbol": normalize_symbol(symbol),
        "yf_symbol": yf_symbol,
        "name": name or "",
        "aum": aum if aum is not None else "",
        "aum_date": utc_today() if aum is not None else "",
        "source": "yfinance",
        "status": status,
        "last_checked": now,
        "error": error or "",
    }

    cache = cache[cache["symbol"] != row["symbol"]].copy()
    cache = pd.concat([cache, pd.DataFrame([row])], ignore_index=True)

    return cache


def update_aum_csv() -> pd.DataFrame:
    cfg = get_auto_aum_config()

    if not cfg["enabled"]:
        print("[AUM] auto_aum.enabled=false. Skipping AUM update.")
        return load_aum_cache()

    print("[AUM] Fetching ETF candidates from Nasdaq Trader symbol directories...")
    candidates = fetch_nasdaq_etf_candidates()
    print(f"[AUM] ETF candidates from Nasdaq Trader: {len(candidates)}")

    cache = load_aum_cache()
    print(f"[AUM] Existing AUM cache rows: {len(cache)}")

    work = build_worklist(candidates, cache, cfg)
    print(f"[AUM] Symbols to check this run: {len(work)}")

    if work.empty:
        save_aum_cache(cache)
        return cache

    start = time.monotonic()
    time_budget_seconds = cfg["time_budget_minutes"] * 60.0

    success = 0
    failed = 0

    for i, row in work.iterrows():
        elapsed = time.monotonic() - start
        if elapsed >= time_budget_seconds:
            print("[AUM] Time budget reached. Saving partial cache and stopping gracefully.")
            break

        symbol = row["symbol"]
        yf_symbol = row["yf_symbol"]
        fallback_name = row["name"]

        print(f"[AUM] {i + 1}/{len(work)} checking {symbol}...")

        aum, fetched_name, error = fetch_aum_from_yfinance(yf_symbol)

        final_name = fetched_name if fetched_name else fallback_name

        if aum is not None:
            cache = upsert_cache_row(
                cache=cache,
                symbol=symbol,
                yf_symbol=yf_symbol,
                name=final_name,
                aum=aum,
                status="ok",
                error="",
            )
            success += 1
        else:
            cache = upsert_cache_row(
                cache=cache,
                symbol=symbol,
                yf_symbol=yf_symbol,
                name=final_name,
                aum=None,
                status="failed",
                error=error,
            )
            failed += 1

        # Save every 25 rows so partial progress survives most interruptions.
        if (success + failed) % 25 == 0:
            save_aum_cache(cache)

        sleep_seconds = cfg["sleep_seconds"]
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    save_aum_cache(cache)

    ok_count = pd.to_numeric(cache["aum"], errors="coerce").notna().sum()
    print(f"[AUM] Done. success_this_run={success}, failed_this_run={failed}, total_with_aum={ok_count}")

    return cache


if __name__ == "__main__":
    update_aum_csv()
