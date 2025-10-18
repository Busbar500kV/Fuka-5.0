# Local-first replacement for the old GCS helper.
# We do NOT touch the UI; we make the backend default to local when F5_STORAGE=local
# or when cloud credentials are unavailable.

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import os


# ----------------------------
# Small env/config helpers
# ----------------------------

def env_get(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(key, default)

def _storage_mode(cfg: Dict[str, Any]) -> str:
    # If caller provides storage, honor it; otherwise fall back to env; default to local
    return str(cfg.get("storage") or env_get("F5_STORAGE", "local")).lower()

def _runs_root(cfg: Dict[str, Any]) -> Path:
    base = env_get("F5_LOCAL_RUNS_DIR") or str(cfg.get("runs_dir") or "/home/busbar/fuka-runs")
    prefix = env_get("F5_RUNS_PREFIX") or str(cfg.get("runs_prefix") or "runs")
    root = Path(base) / prefix
    root.mkdir(parents=True, exist_ok=True)
    return root


# ----------------------------
# Public API used by the UI
# ----------------------------

def get_client():
    """
    Backward-compat stub. The old code expected a GCS client here.
    We only return a real client if explicitly using 'gcs' mode and creds exist.
    """
    mode = _storage_mode({})
    if mode == "gcs":
        # If someone ever re-enables GCS explicitly, raise a clean error explaining how to proceed.
        raise RuntimeError(
            "GCS mode requested but this build is local-first. "
            "Set F5_STORAGE=local (recommended)."
        )
    return None  # local mode: no client needed


def list_runs(cfg: Dict[str, Any]) -> List[str]:
    """
    Return a list of run_ids visible to the UI.

    In local mode, this lists directories under:
        <runs_dir>/<runs_prefix>/
    and returns their names as strings.
    """
    # Local-first
    if _storage_mode(cfg) != "gcs":
        root = _runs_root(cfg)
        # Only include directories that look like FUKA_5_0_* (but keep it permissive)
        runs = [p.name for p in root.iterdir() if p.is_dir()]
        runs.sort(reverse=True)
        return runs

    # If someone truly sets storage=gcs, fail fast with a clear message.
    raise RuntimeError(
        "GCS listing is disabled in this local-first build. "
        "Set F5_STORAGE=local to browse /home/busbar/fuka-runs/runs."
    )


# ----------------------------
# Upload helpers (no-ops for local)
# These remain here so older call-sites import cleanly.
# ----------------------------

def upload_bytes(cfg: Dict[str, Any], payload: bytes, dest_uri: str, content_type: Optional[str] = None) -> None:
    """
    No-op in local mode. Present for compatibility if any code path calls it.
    """
    mode = _storage_mode(cfg)
    if mode == "gcs":
        raise RuntimeError("upload_bytes: GCS disabled in this build (use local storage).")

def upload_json(cfg: Dict[str, Any], obj: Any, dest_uri: str) -> None:
    """
    No-op in local mode. Present for compatibility if any code path calls it.
    """
    mode = _storage_mode(cfg)
    if mode == "gcs":
        raise RuntimeError("upload_json: GCS disabled in this build (use local storage).")