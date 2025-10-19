from __future__ import annotations
import json, os, shutil
from pathlib import Path
from typing import Any, Dict, List, Tuple

def _expand_env_in_str(s: str) -> str:
    return os.path.expandvars(os.path.expanduser(s))
def _expand_env_in_dict(d: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k, v in d.items():
        out[k] = _expand_env_in_str(v) if isinstance(v, str) else v
    return out

def load_local_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f: cfg = json.load(f)
    return _expand_env_in_dict(cfg)

def bucket_and_prefix(cfg: Dict[str, Any]) -> Tuple[Path, str]:
    runs_dir = Path(cfg["runs_dir"]).expanduser()
    prefix   = cfg.get("runs_prefix", "runs").strip("/")
    return runs_dir, prefix

def path_for(cfg: Dict[str, Any], run_id: str, *parts: str) -> Path:
    base, prefix = bucket_and_prefix(cfg)
    rel = "/".join(p.strip("/") for p in parts) if parts else ""
    p = base / prefix / run_id
    return p if not rel else (p / rel)

def list_runs(cfg: Dict[str, Any]) -> List[str]:
    base, prefix = bucket_and_prefix(cfg)
    root = base / prefix
    if not root.exists(): return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])

def list_blobs(cfg: Dict[str, Any], prefix_rel: str) -> List[str]:
    base, _ = bucket_and_prefix(cfg); root = base / prefix_rel; out: List[str] = []
    if root.exists():
        for p in root.rglob("*"):
            if p.is_file(): out.append(str(p.relative_to(base)))
    return out

def download_blob_to_file(cfg: Dict[str, Any], rel_path: str, dst_path: str) -> None:
    base, _ = bucket_and_prefix(cfg)
    src = base / rel_path
    Path(dst_path).parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst_path)

def upload_bytes(cfg: Dict[str, Any], rel_path: str, data: bytes) -> None:
    base, _ = bucket_and_prefix(cfg)
    dst = base / rel_path; dst.parent.mkdir(parents=True, exist_ok=True)
    with open(dst, "wb") as f: f.write(data)

def upload_file(cfg: Dict[str, Any], src_path: str, rel_path: str) -> None:
    base, _ = bucket_and_prefix(cfg)
    dst = base / rel_path; dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_path, dst)

def upload_json(cfg: Dict[str, Any], rel_path: str, obj: Any) -> None:
    import json as _json
    upload_bytes(cfg, rel_path, _json.dumps(obj).encode("utf-8"))
