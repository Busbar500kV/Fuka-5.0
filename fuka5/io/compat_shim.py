from __future__ import annotations
import os
from pathlib import Path

def _tail_after_bucket(gs_uri: str) -> str:
    assert gs_uri.startswith("gs://")
    return "/".join(gs_uri.split("/", 3)[3:])

def _localize(gs_uri: str) -> str:
    runs_dir = os.getenv("F5_LOCAL_RUNS_DIR")
    if not runs_dir:
        raise RuntimeError("F5_LOCAL_RUNS_DIR not set but F5_STORAGE=local")
    tail = _tail_after_bucket(gs_uri)
    return str(Path(runs_dir).expanduser() / tail)

def init_backend_shims():
    if os.getenv("F5_STORAGE", "gcs").lower() != "local":
        return
    try:
        import pandas as pd
        _rp = pd.read_parquet
        def _patched_read_parquet(path, *a, **kw):
            if isinstance(path, str) and path.startswith("gs://"):
                path = _localize(path)
            return _rp(path, *a, **kw)
        pd.read_parquet = _patched_read_parquet
    except Exception: pass
    try:
        import numpy as np
        _nl = np.load
        def _patched_np_load(path, *a, **kw):
            if isinstance(path, str) and path.startswith("gs://"):
                path = _localize(path)
            return _nl(path, *a, **kw)
        np.load = _patched_np_load
    except Exception: pass

def smart_path(p: str) -> str:
    if os.getenv("F5_STORAGE", "gcs").lower() == "local" and isinstance(p, str) and p.startswith("gs://"):
        return _localize(p)
    return p
