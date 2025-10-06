from __future__ import annotations
import os, tempfile, json
from pathlib import Path
from typing import Any, Dict, List

BACKEND = os.getenv("F5_STORAGE", "gcs").lower()
if BACKEND == "local":
    from . import localfs as _storage
    _load_config = _storage.load_local_config
    _list_runs   = _storage.list_runs
    _list_blobs  = _storage.list_blobs
    _dl          = _storage.download_blob_to_file
    def storage_path(cfg: Dict[str, Any], run_id: str, *parts: str) -> Path:
        return _storage.path_for(cfg, run_id, *parts)
else:
    from .gcs import (
        load_gcp_config as _load_config,
        list_runs as _list_runs,
        list_blobs as _list_blobs,
        download_blob_to_file as _dl,
        gcs_path as storage_path,
    )

def load_config(cfg_path: str) -> Dict[str, Any]: return _load_config(cfg_path)
def list_runs(cfg: Dict[str, Any]) -> List[str]:  return _list_runs(cfg)
def list_blobs(cfg: Dict[str, Any], prefix_rel: str) -> List[str]: return _list_blobs(cfg, prefix_rel)

def download_to_tmp(cfg: Dict[str, Any], rel_path: str, suffix: str) -> str:
    tf = tempfile.NamedTemporaryFile(suffix=suffix, delete=False); tf.close()
    _dl(cfg, rel_path, tf.name); return tf.name

def read_parquet(cfg: Dict[str, Any], rel_path: str):
    import pandas as pd
    return pd.read_parquet(download_to_tmp(cfg, rel_path, ".parquet"))

def read_npz(cfg: Dict[str, Any], rel_path: str):
    import numpy as np
    return np.load(download_to_tmp(cfg, rel_path, ".npz"))

def read_json(cfg: Dict[str, Any], rel_path: str) -> Any:
    p = download_to_tmp(cfg, rel_path, ".json")
    with open(p, "r", encoding="utf-8") as f: return json.load(f)
