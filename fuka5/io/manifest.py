"""
fuka5.io.manifest
-----------------
Run manifest schema and integrity helpers.

The manifest is written once at run start and contains:
- run_id
- timestamps, seeds
- config paths (world/sources/training/gcp) and snapshots
- world/grid summary, graph summary
- write cadences

We also provide a light integrity check on required keys.
"""

from __future__ import annotations
from typing import Dict, Any
import time


REQUIRED_KEYS = [
    "run_id",
    "created_utc",
    "configs",
    "summaries",
    "cadence",
    "versions",
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
    Create a manifest dict that can be serialized to JSON.
    - configs: { "world_path": str, "sources_path": str, "training_path": str, "gcp_path": str,
                 "world_cfg": dict, "sources_cfg": dict, "training_cfg": dict, "gcp_cfg": dict }
    - summaries: { "world": {...}, "graph": {...}, "counts": {...} }
    - cadence: { "edges_flush_every": int, "metrics_flush_every": int, "volume_every": int }
    - versions: { "fuka5": "5.0.0" }
    - seeds: { "rng_seed": int }
    """
    return {
        "run_id": run_id,
        "created_utc": int(time.time()),
        "configs": configs,
        "summaries": summaries,
        "cadence": cadence,
        "versions": versions,
        "seeds": seeds,
    }

def check_manifest(m: Dict[str, Any]) -> None:
    """Raise AssertionError if any required key is missing."""
    for k in REQUIRED_KEYS:
        assert k in m, f"manifest missing required key: {k}"