from __future__ import annotations
from pathlib import Path
from typing import List, Dict, Any
import json

def list_local_volumes(run_dir: Path) -> List[str]:
    """Return all local NPZ volume files for a given run."""
    vols = sorted(run_dir.glob("volumes/epoch_*.npz"))
    return [str(v.name) for v in vols]

def load_local_manifest(run_dir: Path) -> Dict[str, Any]:
    """Read the manifest.json if present."""
    mfile = run_dir / "manifest.json"
    if not mfile.exists():
        return {}
    with open(mfile, "r", encoding="utf-8") as f:
        return json.load(f)

def find_local_run(root: str, run_id: str) -> Path:
    """Return Path to the given run id under the local root."""
    base = Path(root) / run_id
    if not base.exists():
        raise FileNotFoundError(f"Run directory not found: {base}")
    return base