from __future__ import annotations
from pathlib import Path
from typing import Any


def storage_path(cfg: dict[str, Any], run_id: str, *parts: str) -> Path:
    """
    Return the local filesystem path for a run artifact.

    Always local:
      <runs_dir>/<runs_prefix>/<run_id>/<parts...>
    """
    runs_dir = cfg.get("runs_dir") or "/home/busbar/fuka-runs"
    runs_prefix = cfg.get("runs_prefix") or "runs"
    return Path(runs_dir) / runs_prefix / run_id / Path(*parts)