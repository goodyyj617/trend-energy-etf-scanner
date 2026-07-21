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


def apply_configured_name_exclusions(etfs: pd.DataFrame, keyword_groups: dict) -> pd.DataFrame:
    out = etfs.copy()
    names = out["name"].fillna("").astype(str)
    reason_category = pd.Series("", index=out.index, dtype="object")
    matched_keyword = pd.Series("", index=out.index, dtype="object")

    for category, keywords in (keyword_groups or {}).items():
        for keyword in sorted(keywords or [], key=lambda value: len(str(value)), reverse=True):
            available = reason_category.eq("")
            matches = names.str.contains(str(keyword), case=False, regex=False, na=False)
            selected = available & matches
            reason_category.loc[selected] = str(category)
            matched_keyword.loc[selected] = str(keyword)

    out["exclude_income_oriented"] = reason_category.ne("")
    out["income_exclusion_reason_category"] = reason_category
    out["income_exclusion_matched_keyword"] = matched_keyword
    return out


def build_income_exclusion_review(universe: pd.DataFrame) -> pd.DataFrame:
    columns = ["symbol", "name", "matched_keyword", "reason_category", "asset_group"]
    if universe.empty or "exclude_income_oriented" not in universe:
        return pd.DataFrame(columns=columns)

    mask = universe["exclude_income_oriented"].fillna(False)
    if "excluded_product_type" in universe:
        mask &= universe["excluded_product_type"].fillna(False)

    review = universe.loc[mask].copy()
    review["matched_keyword"] = review["income_exclusion_matched_keyword"]
    review["reason_category"] = review["income_exclusion_reason_category"]
    if "asset_group" not in review:
        review["asset_group"] = "unknown"
    return review[columns].sort_values(["reason_category", "symbol"]).reset_index(drop=True)


def build_base_universe(
    aum_csv: str | Path = "config/aum.csv",
    exclusions_yml: str | Path = "config/exclusions.yml",
    overrides_csv: str | Path = "config/manual_overrides.csv",
    universe_yml: str | Path = "config/universe.yml",
) -> pd.DataFrame:
    cfg = _load_yaml(universe_yml)["universe"]
    etfs = fetch_nasdaq_trader_etf_list()
    etfs = apply_exclusion_flags(etfs, exclusions_yml)
    etfs = apply_configured_name_exclusions(etfs, cfg.get("exclude_name_keywords", {}))

    aum = load_aum_csv(aum_csv)
    df = etfs.merge(aum, on="symbol", how="inner")
    df["aum_rank"] = df["aum"].rank(ascending=False, method="first")
    df["aum_eligible"] = df["aum_rank"] <= int(cfg["aum_top_n"])

    overrides = load_manual_overrides(overrides_csv)
    if overrides:
        df["manual_action"] = df["symbol"].map(overrides).fillna("")
    else:
        df["manual_action"] = ""

    df["excluded_product_type"] = df["exclude_by_name"] | df["exclude_income_oriented"]
    df.loc[df["manual_action"].eq("include"), "excluded_product_type"] = False
    df.loc[df["manual_action"].eq("exclude"), "excluded_product_type"] = True

    df["base_universe_eligible"] = df["aum_eligible"] & ~df["excluded_product_type"]
    df = df.sort_values(["base_universe_eligible", "aum_rank"], ascending=[False, True]).reset_index(drop=True)
    return df
