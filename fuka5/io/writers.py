from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from fuka5.io.storage_facade import storage_path
from fuka5.io.compat_shim import smart_path


# ---------------------------
# Low-level helpers
# ---------------------------

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


# ---------------------------
# High-level writers used by sim_cli
# ---------------------------

def save_volume_npz(gcp_cfg: Dict[str, Any], run_id: str, local_dir: str, *, epoch: int, **arrays: np.ndarray) -> Path:
    """
    Write compressed NPZ for the UI under volumes/.
    We write BOTH naming schemes so older/newer UIs can find snapshots:
      - epoch_0000.npz   (4-digit)
      - ep000.npz        (3-digit, legacy in UI match)
    """
    # Serialize once
    bio = io.BytesIO()
    np.savez_compressed(bio, **arrays)
    payload = bio.getvalue()

    # Local paths
    vol_dir = Path(local_dir) / "volumes"
    name_a = f"epoch_{epoch:04d}.npz"
    name_b = f"ep{epoch:03d}.npz"
    local_a = vol_dir / name_a
    local_b = vol_dir / name_b

    _write_bytes_local(local_a, payload)
    _write_bytes_local(local_b, payload)

    # Backend paths (local-only facade resolves to filesystem paths)
    dest_a = storage_path(gcp_cfg, run_id, "volumes", name_a)
    dest_b = storage_path(gcp_cfg, run_id, "volumes", name_b)
    _write_bytes_local(dest_a, payload)
    _write_bytes_local(dest_b, payload)

    return local_a


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

        sub = "shards" if self.kind == "edges" else "metrics"
        fname = f"{self.kind}-{self.shard_idx:04d}.parquet"
        self.shard_idx += 1

        local_path = self.local_dir / sub / fname
        local_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(local_path, index=False)

        # Mirror through storage facade (local-only path)
        dest = storage_path(self.cfg, self.run_id, sub, fname)
        with open(local_path, "rb") as f:
            payload = f.read()
        _write_bytes_local(dest, payload)
        return local_path


def write_manifest(gcp_cfg: Dict[str, Any], run_id: str, manifest: Dict[str, Any]) -> Path:
    """
    Persist manifest.json at the run root.
    """
    dest = storage_path(gcp_cfg, run_id, "manifest.json")
    _write_json_local(dest, manifest)
    return Path(smart_path(dest))


def write_checkpoint(gcp_cfg: Dict[str, Any], run_id: str, local_dir: str, name: str, payload: Dict[str, Any]) -> Path:
    """
    Write checkpoints/<name>.json locally and via facade path.
    """
    fname = f"{name}.json"
    local_path = Path(local_dir) / "checkpoints" / fname
    _write_json_local(local_path, payload)

    dest = storage_path(gcp_cfg, run_id, "checkpoints", fname)
    _write_json_local(dest, payload)
    return local_path