from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from fuka5.io.storage_facade import storage_path
from fuka5.io.compat_shim import smart_path

# ---------------------------
# Low-level helpers
# ---------------------------

def _is_gs(p: str | Path) -> bool:
    return isinstance(p, str) and str(p).startswith("gs://")

def _ensure_parent_local(p: str | Path) -> None:
    Path(p).parent.mkdir(parents=True, exist_ok=True)

def _write_bytes_local(p: str | Path, data: bytes) -> None:
    _ensure_parent_local(p)
    with open(p, "wb") as f:
        f.write(data)

def _write_json_local(p: str | Path, obj: Any) -> None:
    _ensure_parent_local(p)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)

def put_bytes(cfg: Dict[str, Any], dest_path: str | Path, payload: bytes, content_type: Optional[str] = None) -> None:
    """
    Upload payload to either GCS (when dest_path is gs://...) or local filesystem.
    """
    if _is_gs(dest_path):
        # Lazy import to avoid hard dep when running local-only
        from fuka5.io.gcs import upload_bytes as _gcs_upload_bytes
        _gcs_upload_bytes(cfg, payload, str(dest_path), content_type=content_type)
    else:
        _write_bytes_local(dest_path, payload)

def put_json(cfg: Dict[str, Any], obj: Any, dest_path: str | Path) -> None:
    if _is_gs(dest_path):
        from fuka5.io.gcs import upload_json as _gcs_upload_json
        _gcs_upload_json(cfg, obj, str(dest_path))
    else:
        _write_json_local(dest_path, obj)

# ---------------------------
# High-level writers used by sim_cli
# ---------------------------

def save_volume_npz(gcp_cfg: Dict[str, Any], run_id: str, local_dir: str, *, epoch: int, **arrays: np.ndarray) -> Path:
    """
    Write a compressed NPZ for the UI under volumes/epoch_####.npz.
    Writes to local_dir and mirrors to storage backend.
    """
    # Serialize to bytes once
    bio = io.BytesIO()
    np.savez_compressed(bio, **arrays)
    payload = bio.getvalue()

    rel_name = f"epoch_{epoch:04d}.npz"
    # Local path
    local_path = Path(local_dir) / "volumes" / rel_name
    _write_bytes_local(local_path, payload)

    # Backend path
    dest = storage_path(gcp_cfg, run_id, "volumes", rel_name)
    # If backend is local, storage_path returns a Path; otherwise a gs:// URI
    put_bytes(gcp_cfg, dest, payload, content_type="application/octet-stream")
    return local_path

class ShardWriter:
    """
    Buffered Parquet shard writer.

    kind: "edges" or "metrics"
    Writes into:
      runs/<run_id>/shards/edges-####.parquet
      runs/<run_id>/metrics/metrics-####.parquet
    """
    def __init__(self, gcp_cfg: Dict[str, Any], run_id: str, *, kind: str, local_dir: str, flush_every: int = 5) -> None:
        assert kind in ("edges", "metrics")
        self.cfg = gcp_cfg
        self.run_id = run_id
        self.kind = kind
        self.local_dir = Path(local_dir)
        self.flush_every = max(1, int(flush_every))
        self.rows: List[Dict[str, Any]] = []
        self.shard_idx: int = 0

        # Ensure folders exist locally
        sub = "shards" if self.kind == "edges" else "metrics"
        (self.local_dir / sub).mkdir(parents=True, exist_ok=True)

    def add(self, row_or_rows: Dict[str, Any] | List[Dict[str, Any]]) -> None:
        if isinstance(row_or_rows, list):
            self.rows.extend(row_or_rows)
        else:
            self.rows.append(row_or_rows)
        self.maybe_flush()

    def maybe_flush(self, *, force: bool = False) -> Optional[Path]:
        if not self.rows:
            return None
        if not force and len(self.rows) < self.flush_every:
            return None

        df = pd.DataFrame(self.rows)
        self.rows.clear()

        # Determine names
        sub = "shards" if self.kind == "edges" else "metrics"
        fname = f"{self.kind}-{self.shard_idx:04d}.parquet"
        self.shard_idx += 1

        # Write local parquet
        local_path = self.local_dir / sub / fname
        local_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(local_path, index=False)

        # Mirror to backend
        dest = storage_path(self.cfg, self.run_id, sub, fname)
        with open(local_path, "rb") as f:
            payload = f.read()
        put_bytes(self.cfg, dest, payload, content_type="application/octet-stream")
        return local_path

def write_manifest(gcp_cfg: Dict[str, Any], run_id: str, manifest: Dict[str, Any]) -> Path:
    """
    Persist manifest.json at the run root.
    """
    dest = storage_path(gcp_cfg, run_id, "manifest.json")
    put_json(gcp_cfg, manifest, dest)
    # Also ensure a local copy exists when storage is gs:// and we're on local
    lp = smart_path(str(dest))
    if not _is_gs(dest):
        # Already local
        return Path(lp)
    # For gs://, smart_path maps it to a local mirror if F5_STORAGE=local
    try:
        _write_json_local(lp, manifest)
    except Exception:
        pass
    return Path(lp)

def write_checkpoint(gcp_cfg: Dict[str, Any], run_id: str, local_dir: str, name: str, payload: Dict[str, Any]) -> Path:
    """
    Write checkpoints/<name>.json locally and mirror to backend.
    """
    fname = f"{name}.json"
    local_path = Path(local_dir) / "checkpoints" / fname
    _write_json_local(local_path, payload)

    dest = storage_path(gcp_cfg, run_id, "checkpoints", fname)
    put_json(gcp_cfg, payload, dest)
    return local_path