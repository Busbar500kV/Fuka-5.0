
"""
fuka5.io.gcs
------------
Dual-mode storage adapter.

- If F5_STORAGE != "local": use Google Cloud Storage via ADC (original behavior).
- If F5_STORAGE == "local": write/read from local filesystem under:
      <runs_dir>/<runs_prefix>/<RUN_ID>/...

This lets the same code run on your busbar box without GCP credentials.
"""

from __future__ import annotations
from typing import Tuple, List, Optional, Dict, Any
import io
import json
import os
import shutil
from pathlib import Path

# Optional import; only used in GCS mode
try:
    from google.cloud import storage  # type: ignore
except Exception:  # pragma: no cover
    storage = None  # lazy-checked when needed

from .. import load_json_with_env, env_get


# ---------------------------
# Mode helpers
# ---------------------------

def _is_local_storage() -> bool:
    return os.getenv("F5_STORAGE", "gcs").lower() == "local"

def _local_base(gcp_cfg: Dict[str, Any]) -> Path:
    """
    Returns <runs_dir>/<runs_prefix> as a Path, expanding ~ and env.
    Expected fields in local.default.json:
        {"runs_dir": "/home/<user>/fuka-runs", "runs_prefix": "runs"}
    """
    runs_dir = gcp_cfg.get("runs_dir") or os.getenv("F5_LOCAL_RUNS_DIR", f"/home/{os.getenv('USER','busbar')}/fuka-runs")
    runs_dir = os.path.expanduser(os.path.expandvars(str(runs_dir)))
    prefix   = gcp_cfg.get("runs_prefix") or os.getenv("F5_RUNS_PREFIX", "runs")
    return Path(runs_dir).expanduser() / str(prefix).strip("/")


# ---------------------------
# Client & config
# ---------------------------

def get_client():
    """
    Return a google-cloud-storage Client using ADC (GCS mode only).
    """
    if _is_local_storage():
        return None  # never used in local mode
    if storage is None:
        raise RuntimeError("google-cloud-storage not available but required for GCS mode.")
    return storage.Client(project=env_get("F5_GCP_PROJECT_ID"))

def load_gcp_config(path: str) -> Dict[str, Any]:
    """Load config file and expand env placeholders."""
    return load_json_with_env(path)

def bucket_and_prefix(gcp_cfg: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract (bucket_name, runs_prefix) for GCS mode.
    In local mode this is unused, but we keep it for compatibility.
    """
    bucket_uri = gcp_cfg.get("bucket", "")
    if not bucket_uri.startswith("gs://"):
        # tolerate missing bucket in local mode
        if _is_local_storage():
            return ("local", gcp_cfg.get("runs_prefix", "runs").strip("/"))
        raise AssertionError("bucket must start with gs://")
    bucket_name = bucket_uri[len("gs://"):]
    prefix = gcp_cfg.get("runs_prefix", "runs").strip("/")
    return bucket_name, prefix


# ---------------------------
# Path helpers
# ---------------------------

def gcs_path(gcp_cfg: Dict[str, Any], run_id: str, *parts: str) -> str:
    """
    Build destination path for artifacts.
      - Local mode: filesystem path <runs_dir>/<runs_prefix>/<run_id>/<parts...>
      - GCS  mode:  'gs://<bucket>/<runs_prefix>/<run_id>/<parts...>'
    """
    if _is_local_storage():
        base = _local_base(gcp_cfg) / run_id
        for q in parts:
            base = base / q
        return str(base)
    bucket_name, prefix = bucket_and_prefix(gcp_cfg)
    tail = "/".join([prefix, run_id] + list(parts)).strip("/")
    return f"gs://{bucket_name}/{tail}"


# ---------------------------
# Uploads
# ---------------------------

def upload_bytes(gcp_cfg: Dict[str, Any], data: bytes, dest_path: str, content_type: Optional[str] = None) -> None:
    """
    Upload raw bytes either to local path (local mode) or to GCS (gs://).
    """
    if _is_local_storage() or not dest_path.startswith("gs://"):
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            f.write(data)
        return

    client = get_client()
    bucket_name, _ = bucket_and_prefix(gcp_cfg)
    name = dest_path.split("/", 3)[3]
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(name)
    blob.upload_from_file(io.BytesIO(data), size=len(data), content_type=content_type)

def upload_json(gcp_cfg: Dict[str, Any], obj: Dict[str, Any], dest_path: str) -> None:
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    upload_bytes(gcp_cfg, payload, dest_path, content_type="application/json")

def upload_file(gcp_cfg: Dict[str, Any], local_path: str, dest_path: str, content_type: Optional[str] = None) -> None:
    if _is_local_storage() or not dest_path.startswith("gs://"):
        dest = Path(dest_path)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local_path, dest)
        return

    client = get_client()
    bucket_name, _ = bucket_and_prefix(gcp_cfg)
    name = dest_path.split("/", 3)[3]
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(name)
    blob.upload_from_filename(local_path, content_type=content_type)


# ---------------------------
# Listings
# ---------------------------

def list_blobs(gcp_cfg: Dict[str, Any], prefix_path: str) -> List[str]:
    """
    List object names under a prefix.
      - Local mode: returns relative file paths under the given directory.
      - GCS  mode: returns blob names (no bucket).
    """
    if _is_local_storage() or not prefix_path.startswith("gs://"):
        base = Path(prefix_path)
        if base.is_file():
            return [base.name]
        out: List[str] = []
        for p in base.rglob("*"):
            if p.is_file():
                out.append(str(p.relative_to(base)))
        return sorted(out)

    client = get_client()
    bucket_name, _ = bucket_and_prefix(gcp_cfg)
    name_prefix = prefix_path.split("/", 3)[3].rstrip("/") + "/"
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket, prefix=name_prefix)
    return [b.name for b in blobs]

def list_runs(gcp_cfg: Dict[str, Any]) -> List[str]:
    """
    Enumerate subfolders under runs_prefix (top-level run IDs).
    """
    if _is_local_storage():
        base = _local_base(gcp_cfg)
        if not base.exists():
            return []
        runs = [p.name for p in base.iterdir() if p.is_dir()]
        return sorted(runs)

    client = get_client()
    bucket_name, prefix = bucket_and_prefix(gcp_cfg)
    bucket = client.bucket(bucket_name)
    name_prefix = f"{prefix}/".strip("/")
    runs = set()
    for blob in client.list_blobs(bucket, prefix=name_prefix, delimiter=None):
        parts = blob.name.split("/")
        if len(parts) >= 2 and parts[0] == prefix and parts[1]:
            runs.add(parts[1])
    return sorted(runs)


# ---------------------------
# Downloads
# ---------------------------

def download_blob_to_file(gcp_cfg: Dict[str, Any], src_path: str, local_path: str) -> None:
    if _is_local_storage() or not src_path.startswith("gs://"):
        src = Path(src_path)
        Path(local_path).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, local_path)
        return

    client = get_client()
    bucket_name, _ = bucket_and_prefix(gcp_cfg)
    name = src_path.split("/", 3)[3]
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)
