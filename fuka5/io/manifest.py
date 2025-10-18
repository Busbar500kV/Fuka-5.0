from __future__ import annotations
from typing import Any, Dict

REQUIRED_KEYS = [
    "run_id",
    "configs",
    "summaries",
    "cadence",
    "versions",
    "seeds",
]

def build_manifest(
    *,
    run_id: str,
    configs: Dict[str, Any],
    summaries: Dict[str, Any],
    cadence: Dict[str, Any],
    versions: Dict[str, Any],
    seeds: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Construct a manifest dict. Minimal schema that the UI/tools can introspect.
    """
    m = {
        "run_id": run_id,
        "configs": configs,
        "summaries": summaries,
        "cadence": cadence,
        "versions": versions,
        "seeds": seeds,
    }
    return m

def validate_manifest(m: Dict[str, Any]) -> None:
    for k in REQUIRED_KEYS:
        if k not in m:
            raise ValueError(f"manifest missing required key: {k}")