# Local-first replacement for the old GCS helper.
# Keeps the same public symbols expected by the UI, but everything runs locally.

from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import os, json


# ----------------------------
# Env + Config helpers
# ----------------------------

def env_get(key: str, default: Optional[str] = None) -> Optional[str]:
    return os.environ.get(key, default)

def _storage_mode(cfg: Dict[str, Any]) -> str:
    return str(cfg.get("storage") or env_get("F5_STORAGE", "local")).lower()

def _runs_root(cfg: Dict[str, Any]) -> Path:
    base = env_get("F5_LOCAL_RUNS_DIR") or str(cfg.get("runs_dir") or "/home/busbar/fuka-runs")
    prefix = env_get("F5_RUNS_PREFIX") or str(cfg.get("runs_prefix") or "runs")
    root = Path(base) / prefix
    root.mkdir(parents=True, exist_ok=True)
    return root


# ----------------------------
# Public API used by UI
# ----------------------------

def get_client():
    """Dummy client for compatibility; real GCS disabled."""
    return None


def load_gcp_config(gcp_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Return a local config so the UI can initialize seamlessly.
    Accepts an optional path argument for compatibility with the old UI.
    """
    return {
        "storage": "local",
        "runs_dir": env_get("F5_LOCAL_RUNS_DIR", "/home/busbar/fuka-runs"),
        "runs_prefix": env_get("F5_RUNS_PREFIX", "runs"),
    }


def _is_valid_run_dir(p: Path) -> bool:
    """A directory is a 'run' if it has a manifest.json or at least one volume NPZ."""
    if not p.is_dir():
        return False
    if (p / "manifest.json").exists():
        return True
    if (p / "volumes").exists() and any((p / "volumes").glob("epoch_*.npz")):
        return True
    return False


def list_runs(cfg: Dict[str, Any]) -> List[str]:
    """
    List local runs: only directories that look like runs.
    Sorted by most-recent mtime descending.
    """
    root = _runs_root(cfg)
    items = []
    for child in root.iterdir():
        if _is_valid_run_dir(child):
            try:
                mt = child.stat().st_mtime
            except Exception:
                mt = 0.0
            items.append((mt, child.name))
    items.sort(reverse=True)  # newest first
    return [name for _, name in items]


def list_blobs(cfg: Dict[str, Any], run_id: str) -> List[str]:
    """Return local files inside a given run folder (relative paths)."""
    root = _runs_root(cfg) / run_id
    if not root.exists():
        return []
    rels: List[str] = []
    for p in root.rglob("*"):
        if p.is_file():
            try:
                rels.append(str(p.relative_to(root)))
            except Exception:
                pass
    rels.sort()
    return rels


def download_blob_to_file(cfg: Dict[str, Any], src: str, dest: str) -> None:
    """Local copy helper for compatibility."""
    src_path = Path(src)
    dest_path = Path(dest)
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    if src_path.exists():
        dest_path.write_bytes(src_path.read_bytes())


def gcs_path(cfg: Dict[str, Any], run_id: str, *parts: str) -> str:
    """In local mode, just return the full local path."""
    return str(_runs_root(cfg) / run_id / Path(*parts))


# ----------------------------
# Upload helpers (no-ops)
# ----------------------------

def upload_bytes(cfg: Dict[str, Any], payload: bytes, dest_uri: str, content_type: Optional[str] = None) -> None:
    return None

def upload_json(cfg: Dict[str, Any], obj: Any, dest_uri: str) -> None:
    path = Path(dest_uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)