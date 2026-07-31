"""Loader and schema validation for the Korean-first terminology source."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


TERMINOLOGY_SCHEMA_VERSION = "korean_terminology_v1"
REQUIRED_ENTRY_FIELDS = frozenset(
    {
        "korean_term",
        "english_term",
        "abbreviation",
        "formula_text",
        "variable_definitions",
        "worked_numerical_example",
        "interpretation",
        "unit",
        "annualization_convention",
        "assumptions",
        "limitations",
        "misleading_cases",
        "applicable_decision_modes",
    }
)


def validate_terminology_source(source: Mapping[str, Any]) -> tuple[str, ...]:
    errors: list[str] = []
    if source.get("schema_version") != TERMINOLOGY_SCHEMA_VERSION:
        errors.append("invalid_schema_version")
    entries = source.get("entries")
    if not isinstance(entries, Mapping):
        return tuple(errors + ["entries_must_be_mapping"])
    for key, entry in entries.items():
        if not isinstance(entry, Mapping):
            errors.append(f"{key}:entry_must_be_mapping")
            continue
        missing = REQUIRED_ENTRY_FIELDS - set(entry)
        if missing:
            errors.append(f"{key}:missing:{','.join(sorted(missing))}")
        for collection_field in (
            "variable_definitions",
            "assumptions",
            "limitations",
            "misleading_cases",
            "applicable_decision_modes",
        ):
            if collection_field in entry and not entry[collection_field]:
                errors.append(f"{key}:{collection_field}_must_not_be_empty")
    return tuple(errors)


def load_terminology_source(path: str | Path) -> Mapping[str, Any]:
    source = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = validate_terminology_source(source)
    if errors:
        raise ValueError("invalid terminology source: " + "; ".join(errors))
    return source
