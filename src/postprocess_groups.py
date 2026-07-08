from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "docs" / "data"


def norm_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).upper()


def has_any(text: str, patterns: List[str]) -> bool:
    return any(re.search(pat, text) for pat in patterns)


def classify_group(symbol: Any, name: Any) -> str:
    s = norm_text(symbol).strip()
    n = norm_text(name).strip()
    text = f" {s} {n} "

    # 1. Crypto
    if has_any(text, [
        r"\bBITCOIN\b", r"\bBTC\b", r"\bETHEREUM\b", r"\bETH\b",
        r"\bCRYPTO\b", r"\bBLOCKCHAIN\b", r"\bDIGITAL ASSET",
    ]):
        return "Crypto"

    # 2. Commodity
    if has_any(text, [
        r"\bGOLD\b", r"\bSILVER\b", r"\bCOPPER\b", r"\bPLATINUM\b",
        r"\bPALLADIUM\b", r"\bURANIUM\b", r"\bOIL\b", r"\bNATURAL GAS\b",
        r"\bGASOLINE\b", r"\bENERGY FUTURES\b", r"\bCOMMODITY\b",
        r"\bAGRICULTURE\b", r"\bCORN\b", r"\bWHEAT\b", r"\bSOYBEAN\b",
    ]):
        return "Commodity"

    # 3. Bond / Fixed Income
    if has_any(text, [
        r"\bBOND\b", r"\bFIXED INCOME\b", r"\bTREASURY\b", r"\bT-BILL\b",
        r"\bTBILL\b", r"\bMUNICIPAL\b", r"\bMUNI\b", r"\bCORPORATE\b",
        r"\bHIGH YIELD\b", r"\bINVESTMENT GRADE\b", r"\bMORTGAGE\b",
        r"\bMBS\b", r"\bLOAN\b", r"\bFLOATING RATE\b", r"\bDURATION\b",
        r"\bAGGREGATE\b", r"\bINCOME ETF\b",
    ]):
        return "Bond"

    # 4. REIT / Infrastructure
    if has_any(text, [
        r"\bREIT\b", r"\bREAL ESTATE\b", r"\bPROPERTY\b",
        r"\bINFRASTRUCTURE\b", r"\bMLP\b", r"\bMIDSTREAM\b",
    ]):
        return "REIT / Infra"

    # 5. Country / Region
    if has_any(text, [
        r"\bTAIWAN\b", r"\bKOREA\b", r"\bJAPAN\b", r"\bCHINA\b",
        r"\bINDIA\b", r"\bBRAZIL\b", r"\bMEXICO\b", r"\bCANADA\b",
        r"\bAUSTRALIA\b", r"\bGERMANY\b", r"\bFRANCE\b", r"\bITALY\b",
        r"\bSPAIN\b", r"\bUNITED KINGDOM\b", r"\bUK\b", r"\bEUROPE\b",
        r"\bEUROZONE\b", r"\bASIA\b", r"\bPACIFIC\b", r"\bLATIN AMERICA\b",
        r"\bEMERGING MARKETS\b", r"\bDEVELOPED MARKETS\b",
        r"\bFRONTIER\b", r"\bEAFE\b", r"\bEX-US\b", r"\bEX US\b",
        r"\bMSCI [A-Z ]+ ETF\b",
    ]):
        return "Country / Region"

    # 6. Industry / Theme
    if has_any(text, [
        r"\bSEMICONDUCTOR\b", r"\bCYBERSECURITY\b", r"\bBIOTECH\b",
        r"\bBIOTECHNOLOGY\b", r"\bGENOMICS\b", r"\bROBOTICS\b",
        r"\bARTIFICIAL INTELLIGENCE\b", r"\b A\.I\. \b", r"\b AI \b",
        r"\bQUANTUM\b", r"\bCLOUD\b", r"\bSOFTWARE\b", r"\bINTERNET\b",
        r"\bFINTECH\b", r"\bE-COMMERCE\b", r"\bCLEAN ENERGY\b",
        r"\bSOLAR\b", r"\bWIND\b", r"\bNUCLEAR\b", r"\bLITHIUM\b",
        r"\bBATTERY\b", r"\bAEROSPACE\b", r"\bDEFENSE\b",
        r"\bHEALTHCARE INNOVATION\b", r"\bCANNABIS\b",
        r"\bWATER\b", r"\bAGTECH\b", r"\bMETAVERSE\b",
    ]):
        return "Industry / Theme"

    # 7. Sector
    if has_any(text, [
        r"\bTECHNOLOGY\b", r"\bINFORMATION TECHNOLOGY\b",
        r"\bCOMMUNICATION SERVICES\b", r"\bCONSUMER DISCRETIONARY\b",
        r"\bCONSUMER STAPLES\b", r"\bFINANCIALS\b", r"\bHEALTH CARE\b",
        r"\bHEALTHCARE\b", r"\bINDUSTRIALS\b", r"\bMATERIALS\b",
        r"\bENERGY SELECT\b", r"\bUTILITIES\b", r"\bSECTOR\b",
    ]):
        return "Sector"

    # 8. Factor / Style
    if has_any(text, [
        r"\bVALUE\b", r"\bGROWTH\b", r"\bMOMENTUM\b", r"\bQUALITY\b",
        r"\bLOW VOLATILITY\b", r"\bMINIMUM VOLATILITY\b", r"\bMIN VOL\b",
        r"\bDIVIDEND\b", r"\bYIELD\b", r"\bEQUAL WEIGHT\b",
        r"\bFREE CASH FLOW\b", r"\bMULTIFACTOR\b", r"\bFACTOR\b",
        r"\bSIZE\b", r"\bSMALL CAP\b", r"\bMID CAP\b", r"\bLARGE CAP\b",
    ]):
        return "Factor / Style"

    # 9. Broad US equity
    if has_any(text, [
        r"\bS&P 500\b", r"\bSP500\b", r"\bTOTAL STOCK\b",
        r"\bTOTAL MARKET\b", r"\bRUSSELL 1000\b", r"\bRUSSELL 2000\b",
        r"\bRUSSELL 3000\b", r"\bNASDAQ-100\b", r"\bNASDAQ 100\b",
        r"\bDOW JONES\b", r"\bU\.S\. EQUITY\b", r"\bUS EQUITY\b",
        r"\bUSA\b",
    ]):
        return "US Equity"

    # 10. Global broad equity
    if has_any(text, [
        r"\bGLOBAL\b", r"\bWORLD\b", r"\bINTERNATIONAL\b",
        r"\bACWI\b", r"\bALL COUNTRY\b",
    ]):
        return "Global Equity"

    return "Other"


def get_value_case_insensitive(row: Dict[str, Any], names: List[str]) -> Any:
    lower_map = {str(k).lower(): k for k in row.keys()}
    for name in names:
        key = lower_map.get(name.lower())
        if key is not None:
            return row.get(key)
    return None


def set_value_case_insensitive(row: Dict[str, Any], target: str, value: Any) -> None:
    lower_map = {str(k).lower(): k for k in row.keys()}
    key = lower_map.get(target.lower(), target)
    row[key] = value


def find_json_records(obj: Any) -> Optional[List[Dict[str, Any]]]:
    if isinstance(obj, list) and all(isinstance(x, dict) for x in obj):
        return obj

    if isinstance(obj, dict):
        for key in ["rows", "data", "records"]:
            value = obj.get(key)
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                return value

        for value in obj.values():
            if isinstance(value, list) and all(isinstance(x, dict) for x in value):
                if any("symbol" in {str(k).lower() for k in x.keys()} for x in value):
                    return value

    return None


def postprocess_json(path: Path) -> None:
    if not path.exists():
        print(f"[GROUP] skip missing {path}")
        return

    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)

    records = find_json_records(obj)
    if records is None:
        print(f"[GROUP] no records found in {path}")
        return

    changed = 0
    for row in records:
        symbol = get_value_case_insensitive(row, ["symbol", "ticker"])
        name = get_value_case_insensitive(row, ["name", "security_name", "fund_name"])
        group = classify_group(symbol, name)
       
        set_value_case_insensitive(row, "group", group)
        set_value_case_insensitive(row, "asset_group", group)
        changed += 1

    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

    print(f"[GROUP] updated {changed} rows in {path}")


def find_col(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lower_map = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower_map:
            return lower_map[name.lower()]
    return None


def postprocess_csv(path: Path) -> None:
    if not path.exists():
        print(f"[GROUP] skip missing {path}")
        return

    df = pd.read_csv(path)

    symbol_col = find_col(df, ["symbol", "ticker"])
    name_col = find_col(df, ["name", "security_name", "fund_name"])

    if symbol_col is None:
        print(f"[GROUP] no symbol column in {path}")
        return

    if name_col is None:
        df["_tmp_name"] = ""
        name_col = "_tmp_name"

    groups = [
        classify_group(symbol, name)
        for symbol, name in zip(df[symbol_col], df[name_col])
    ]

    group_col = find_col(df, ["group"])
    asset_group_col = find_col(df, ["asset_group"])

    if group_col is None:
        group_col = "group"

    if asset_group_col is None:
        asset_group_col = "asset_group"

    df[group_col] = groups
    df[asset_group_col] = groups

    if "_tmp_name" in df.columns:
        df = df.drop(columns=["_tmp_name"])

    df.to_csv(path, index=False)
    print(f"[GROUP] updated {len(df)} rows in {path}")


def main() -> None:
    postprocess_json(DATA_DIR / "latest.json")
    postprocess_csv(DATA_DIR / "latest.csv")
    postprocess_csv(DATA_DIR / "universe_current.csv")

    history_dir = DATA_DIR / "history"
    if history_dir.exists():
        history_files = sorted(history_dir.glob("*.csv"))
        if history_files:
            postprocess_csv(history_files[-1])


if __name__ == "__main__":
    main()
