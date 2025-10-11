from __future__ import annotations


from fuka5.io.storage_facade import storage_path

from pathlib import Path as _Path
import json as _json

def _is_gs(p):
    return isinstance(p, str) and p.startswith("gs://")

def _ensure_parent_local(p):
    _Path(p).parent.mkdir(parents=True, exist_ok=True)

def _write_bytes_local(p, data: bytes):
    _ensure_parent_local(p)
    _Path(p).write_bytes(data)

def _write_json_local(p, obj):
    _ensure_parent_local(p)
    _Path(p).write_text(_json.dumps(obj, indent=2))
"""
fuka5.io.writers
----------------
Writers for Parquet shards (edges, metrics, state) and NPZ volumes (rho/eps/masks),
plus checkpoint helpers. Targets Google Cloud Storage via gcs.py.

Design:
- Accumulate rows per epoch in-memory; flush every N epochs to local tmp,
  then upload to GCS under runs/<RUN_ID>/shards/.
- Volumes (rho/eps/masks) saved as small NPZ files under runs/<RUN_ID>/volumes/.
- Checkpoints (configs, seeds, progress) under runs/<RUN_ID>/checkpoints/.

Dependencies: pandas, pyarrow.
"""

from typing import List, Dict, Any, Optional
import os
import json
import uuid
import numpy as np
import pandas as pd

from .gcs import gcs_path, upload_file, upload_json


class ShardWriter:
    """
    Buffered Parquet writer for a given table kind ('edges' | 'metrics' | 'state').
    """
    def __init__(
        self,
        gcp_cfg: Dict[str, Any],
        run_id: str,
        kind: str,
        local_dir: str,
        flush_every: int = 5,
    ):
        assert kind in ("edges", "metrics", "state")
        self.gcp_cfg = gcp_cfg
        self.run_id = run_id
        self.kind = kind
        self.local_dir = os.path.join(local_dir, "shards")
        self.flush_every = int(flush_every)
        os.makedirs(self.local_dir, exist_ok=True)
        self._rows: List[Dict[str, Any]] = []
        self._shard_idx = 0

    def add(self, row: Dict[str, Any]) -> None:
        self._rows.append(row)

    def extend(self, rows: List[Dict[str, Any]]) -> None:
        self._rows.extend(rows)

    def maybe_flush(self, force: bool = False) -> None:
        if not self._rows:
            return
        if (len(self._rows) >= self.flush_every) or force:
            df = pd.DataFrame(self._rows)
            shard_name = f"{self.kind}_{self._shard_idx:03d}.parquet"
            local_path = os.path.join(self.local_dir, shard_name)
            df.to_parquet(local_path, index=False)
            # upload
            dest = storage_path(self.gcp_cfg, self.run_id, "shards", shard_name)
            upload_file(self.gcp_cfg, local_path, dest, content_type="application/octet-stream")
            # reset buffer
            self._rows.clear()
            self._shard_idx += 1


def save_volume_npz(gcp_cfg: Dict[str, Any], run_id: str, local_dir: str, epoch: int, **arrays) -> None:
    """
    Save small volume arrays (rho, eps, outer_mask, core_mask, etc.) to NPZ and upload.
    """
    vol_dir = os.path.join(local_dir, "volumes")
    os.makedirs(vol_dir, exist_ok=True)
    fname = f"ep{epoch:03d}.npz"
    local_path = os.path.join(vol_dir, fname)
    np.savez_compressed(local_path, **arrays)
    dest = storage_path(gcp_cfg, run_id, "volumes", fname)
    upload_file(gcp_cfg, local_path, dest, content_type="application/octet-stream")


def write_manifest(gcp_cfg: Dict[str, Any], run_id: str, manifest: Dict[str, Any]) -> None:
    """
    Write manifest.json under the run folder.
    """
    dest = storage_path(gcp_cfg, run_id, "manifest.json")
    put_json(gcp_cfg, manifest, dest)


def write_checkpoint(gcp_cfg: Dict[str, Any], run_id: str, local_dir: str, name: str, payload: Dict[str, Any]) -> None:
    """
    Write a JSON checkpoint under checkpoints/.
    """
    ck_dir = os.path.join(local_dir, "checkpoints")
    os.makedirs(ck_dir, exist_ok=True)
    fname = f"{name}.json"
    local_path = os.path.join(ck_dir, fname)
    with open(local_path, "w", encoding="utf-8") as f:
        json.dump(payload, f)
    dest = storage_path(gcp_cfg, run_id, "checkpoints", fname)
    upload_file(gcp_cfg, local_path, dest, content_type="application/json")


def put_bytes(cfg, dest_path, payload: bytes, content_type: str | None = None):
    if _is_gs(dest_path):
        from fuka5.io.gcs import upload_bytes as _gcs_upload_bytes
        return _gcs_upload_bytes(cfg, payload, dest_path, content_type=content_type)
    else:
        _write_bytes_local(dest_path, payload)

def put_json(cfg, obj, dest_path):
    if _is_gs(dest_path):
        from fuka5.io.gcs import upload_json as _gcs_upload_json
        return _gcs_upload_json(cfg, obj, dest_path)
    else:
        _write_json_local(dest_path, obj)
