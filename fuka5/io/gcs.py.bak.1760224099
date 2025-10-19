"""
fuka5.io.gcs
------------
Thin wrapper over google-cloud-storage using Application Default Credentials (ADC).

Capabilities
* get_client()            -> GCS client (ADC)
* bucket_and_prefix()     -> (bucket_name, runs_prefix) from env-expanded config
* gcs_path(run_id, ...)   -> "gs://bucket/prefix/run_id/..."
* upload_bytes(...), upload_file(...), upload_json(...)
* list_runs()             -> enumerate run folder names under runs_prefix
* list_blobs(prefix)      -> list blob names under a prefix
* download_blob_to_file(...)
"""

from __future__ import annotations
from typing import Tuple, List, Optional, Dict, Any
import io
import json
import os

from google.cloud import storage

from .. import load_json_with_env, env_get


# ---------------------------
# Client & config
# ---------------------------

def get_client() -> storage.Client:
    """
    Return a google-cloud-storage Client using ADC.
    Ensure VM has a service account with Storage access or run:
       gcloud auth application-default login
    """
    return storage.Client(project=env_get("F5_GCP_PROJECT_ID"))

def load_gcp_config(path: str) -> Dict[str, Any]:
    """Load gcp.default.json and expand env placeholders."""
    return load_json_with_env(path)

def bucket_and_prefix(gcp_cfg: Dict[str, Any]) -> Tuple[str, str]:
    """
    Extract bucket and runs_prefix; bucket must be 'gs://name'.
    Returns (bucket_name, runs_prefix)
    """
    bucket_uri = gcp_cfg["bucket"]
    assert bucket_uri.startswith("gs://"), "bucket must start with gs://"
    bucket_name = bucket_uri[len("gs://"):]
    prefix = gcp_cfg.get("runs_prefix", "runs").strip("/")

    return bucket_name, prefix


# ---------------------------
# Path helpers
# ---------------------------

def gcs_path(gcp_cfg: Dict[str, Any], run_id: str, *parts: str) -> str:
    """Return full gs:// path under runs_prefix/run_id/ plus parts."""
    bucket_name, prefix = bucket_and_prefix(gcp_cfg)
    tail = "/".join([prefix, run_id] + list(parts)).strip("/")
    return f"gs://{bucket_name}/{tail}"


# ---------------------------
# Uploads
# ---------------------------

def upload_bytes(gcp_cfg: Dict[str, Any], data: bytes, dest_path: str, content_type: Optional[str] = None) -> None:
    """
    Upload raw bytes to gs://... path.
    """
    client = get_client()
    bucket_name, _ = bucket_and_prefix(gcp_cfg)
    assert dest_path.startswith("gs://"), "dest_path must be gs://"
    # parse blob name
    name = dest_path.split("/", 3)[3]
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(name)
    blob.upload_from_file(io.BytesIO(data), size=len(data), content_type=content_type)

def upload_json(gcp_cfg: Dict[str, Any], obj: Dict[str, Any], dest_path: str) -> None:
    payload = json.dumps(obj, ensure_ascii=False).encode("utf-8")
    upload_bytes(gcp_cfg, payload, dest_path, content_type="application/json")

def upload_file(gcp_cfg: Dict[str, Any], local_path: str, dest_path: str, content_type: Optional[str] = None) -> None:
    client = get_client()
    bucket_name, _ = bucket_and_prefix(gcp_cfg)
    assert dest_path.startswith("gs://")
    name = dest_path.split("/", 3)[3]
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(name)
    blob.upload_from_filename(local_path, content_type=content_type)


# ---------------------------
# Listings
# ---------------------------

def list_blobs(gcp_cfg: Dict[str, Any], prefix_path: str) -> List[str]:
    """
    List blob names under a given gs://<bucket>/<prefix>.
    Returns the object names (no bucket).
    """
    client = get_client()
    bucket_name, _ = bucket_and_prefix(gcp_cfg)
    assert prefix_path.startswith("gs://")
    # name after bucket
    name_prefix = prefix_path.split("/", 3)[3].rstrip("/") + "/"
    bucket = client.bucket(bucket_name)
    blobs = client.list_blobs(bucket, prefix=name_prefix)
    return [b.name for b in blobs]

def list_runs(gcp_cfg: Dict[str, Any]) -> List[str]:
    """
    Enumerate subfolders under runs_prefix (top-level run IDs).
    """
    client = get_client()
    bucket_name, prefix = bucket_and_prefix(gcp_cfg)
    bucket = client.bucket(bucket_name)
    name_prefix = f"{prefix}/".strip("/")
    runs = set()
    for blob in client.list_blobs(bucket, prefix=name_prefix, delimiter=None):
        # Expect keys like: runs/<RUN_ID>/manifest.json
        parts = blob.name.split("/")
        if len(parts) >= 2 and parts[0] == prefix and parts[1]:
            runs.add(parts[1])
    return sorted(runs)


# ---------------------------
# Downloads
# ---------------------------

def download_blob_to_file(gcp_cfg: Dict[str, Any], src_path: str, local_path: str) -> None:
    client = get_client()
    bucket_name, _ = bucket_and_prefix(gcp_cfg)
    assert src_path.startswith("gs://")
    name = src_path.split("/", 3)[3]
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(name)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    blob.download_to_filename(local_path)