"""Public contracts for reusable Trend Strategy v2 tooling.

Keep package import lightweight: launcher bootstrap commands must not import
numeric runtime dependencies until application services are actually started.
"""

from __future__ import annotations

import importlib

_MODULES = (
    "canonical", "contracts", "result_store", "local_operability", "execution",
    "foundation_6", "robustness", "workflow", "evaluation", "decision_report", "artifact_schemas",
    "behavior", "calculation", "integration", "metrics", "profiles", "profile_studio", "terminology",
    "registry", "api", "web", "construction", "execution_service", "engine_adapter",
)


def __getattr__(name: str):
    for module_name in _MODULES:
        module = importlib.import_module(f"{__name__}.{module_name}")
        if hasattr(module, name):
            value = getattr(module, name)
            globals()[name] = value
            return value
    raise AttributeError(name)


def __dir__():
    names = set(globals())
    for module_name in _MODULES:
        names.update(dir(importlib.import_module(f"{__name__}.{module_name}")))
    return sorted(names)
