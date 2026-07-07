from __future__ import annotations

from pathlib import Path

import pandas as pd
import yaml

from .data_sources import fetch_nasdaq_trader_etf_list, load_aum_csv, load_manual_overrides


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def apply_exclusion_flags(etfs: pd.DataFrame, exclusions_yml: str | Path) -> pd.DataFrame:
    rules = _load_yaml(exclusions_yml).get("exclude_keywords", {})
    out = etfs.copy()
    name_upper = out["name"].fillna("").str.upper()

    exclude_cols = []
    for group, keywords in rules.items():
        col = f"exclude_{group}"
        exclude_cols.append(col)
        mask = pd.Series(False, index=out.index)
        for kw in keywords:
            mask = mask | name_upper.str.contains(str(kw).upper(), regex=False, na=False)
        out[col] = mask

    out["exclude_by_name"] = out[exclude_cols].any(axis=1) if exclude_cols else False
    reasons = []
    for _, row in out.iterrows():
        groups = [c.replace("exclude_", "") for c in exclude_cols if bool(row[c])]
        reasons.append(",".join(groups))
    out["exclude_reason"] = reasons
    return out


def build_base_universe(
    aum_csv: str | Path = "config/aum.csv",
    exclusions_yml: str | Path = "config/exclusions.yml",
    overrides_csv: str | Path = "config/manual_overrides.csv",
    universe_yml: str | Path = "config/universe.yml",
) -> pd.DataFrame:
    cfg = _load_yaml(universe_yml)["universe"]
    etfs = fetch_nasdaq_trader_etf_list()
    etfs = apply_exclusion_flags(etfs, exclusions_yml)

    aum = load_aum_csv(aum_csv)
    df = etfs.merge(aum, on="symbol", how="inner")
    df["aum_rank"] = df["aum"].rank(ascending=False, method="first")
    df["aum_eligible"] = df["aum_rank"] <= int(cfg["aum_top_n"])

    overrides = load_manual_overrides(overrides_csv)
    if overrides:
        df["manual_action"] = df["symbol"].map(overrides).fillna("")
    else:
        df["manual_action"] = ""

    df["excluded_product_type"] = df["exclude_by_name"]
    df.loc[df["manual_action"].eq("include"), "excluded_product_type"] = False
    df.loc[df["manual_action"].eq("exclude"), "excluded_product_type"] = True

    df["base_universe_eligible"] = df["aum_eligible"] & ~df["excluded_product_type"]
    df = df.sort_values(["base_universe_eligible", "aum_rank"], ascending=[False, True]).reset_index(drop=True)
    return df
