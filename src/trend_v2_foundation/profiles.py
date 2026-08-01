"""Load immutable evaluation profile examples from configuration."""

from __future__ import annotations

import json
from pathlib import Path

from .contracts import EvaluationProfile


def load_evaluation_profile(path: str | Path) -> EvaluationProfile:
    return EvaluationProfile.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))


def load_evaluation_profiles(directory: str | Path) -> dict[str, EvaluationProfile]:
    profiles = {}
    for path in sorted(Path(directory).glob("*.json")):
        profile = load_evaluation_profile(path)
        if profile.name in profiles:
            raise ValueError(f"duplicate evaluation profile name: {profile.name}")
        profiles[profile.name] = profile
    return profiles
