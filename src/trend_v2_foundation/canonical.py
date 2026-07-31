"""Deterministic serialization helpers for Trend Strategy v2 contracts."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields, is_dataclass
from datetime import date, datetime, timezone
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping


def deep_freeze(value: Any) -> Any:
    """Return an immutable copy suitable for frozen contract fields."""

    if isinstance(value, Mapping):
        return MappingProxyType({str(key): deep_freeze(item) for key, item in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(item) for item in value)
    if isinstance(value, (set, frozenset)):
        return tuple(sorted((deep_freeze(item) for item in value), key=canonical_json))
    return value


def canonical_data(value: Any) -> Any:
    """Normalize supported values to a deterministic JSON-compatible tree."""

    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            field.name: canonical_data(getattr(value, field.name))
            for field in fields(value)
            if not field.name.startswith("_")
        }
    if isinstance(value, Mapping):
        return {
            str(key): canonical_data(value[key])
            for key in sorted(value, key=lambda item: str(item))
        }
    if isinstance(value, (list, tuple)):
        return [canonical_data(item) for item in value]
    if isinstance(value, (set, frozenset)):
        normalized = [canonical_data(item) for item in value]
        return sorted(normalized, key=canonical_json)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            raise ValueError("datetimes must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("canonical contracts cannot contain NaN or infinity")
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"unsupported canonical value: {type(value).__name__}")


def canonical_json(value: Any) -> str:
    return json.dumps(
        canonical_data(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def canonical_bytes(value: Any) -> bytes:
    return canonical_json(value).encode("utf-8")


def content_hash(value: Any) -> str:
    payload = value if isinstance(value, bytes) else canonical_bytes(value)
    return hashlib.sha256(payload).hexdigest()


def deterministic_id(prefix: str, value: Any) -> str:
    return f"{prefix}_{content_hash(value)}"
