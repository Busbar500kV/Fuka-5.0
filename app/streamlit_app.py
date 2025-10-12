from __future__ import annotations
import os, json
from pathlib import Path
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import streamlit as st

# ---- page config first (Streamlit rule) ----
st.set_page_config(page_title="Fuka 5.0", layout="wide")

# ---- helpers ----
def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)
    # env expansion like ${HOME}
    def _exp(v):
        if isinstance(v, str):
            return os.path.expandvars(v)
        return v
    return {k: _exp(v) for k, v in cfg.items()}

def runs_root(cfg: Dict[str, Any]) -> Path:
    return Path(cfg["runs_dir"]).expanduser() / cfg["runs_prefix"].strip("/")

def list_runs(cfg: Dict[str, Any]) -> List[str]:
    root = runs_root(cfg)
    if not root.exists():
        return []
    return sorted([p.name for p in root.iterdir() if p.is_dir()])

def vol_dir(cfg: Dict[str, Any], run_id: str) -> Path:
    return runs_root(cfg) / run_id / "volumes"

def epochs_from_local(cfg: Dict[str, Any], run_id: str) -> List[int]:
    vdir = vol_dir(cfg, run_id)
    eps: List[int] = []
    if vdir.is_dir():
        for p in vdir.glob("epoch_*.npz"):
            try:
                eps.append(int(p.stem.split("_")[1]))
            except Exception:
                pass
    return sorted(eps)

def load_npz_local(cfg: Dict[str, Any], run_id: str, epoch: int):
    p = vol_dir(cfg, run_id) / f"epoch_{epoch:04d}.npz"
    with np.load(p) as z:
        return {k: z[k] for k in z.files}, str(p)

# ---- sidebar / inputs ----
CFG_PATH = os.environ.get("F5_UI_CONFIG", "configs/local.default.json")
gcp_cfg = load_config(CFG_PATH)

st.sidebar.header("Select run / epoch (local)")
runs = list_runs(gcp_cfg)
if not runs:
    st.sidebar.warning("No runs found. Start a sim first.")
    st.stop()

run_id = st.sidebar.selectbox("Run", runs, index=len(runs)-1)
vdir = vol_dir(gcp_cfg, run_id)
epochs = epochs_from_local(gcp_cfg, run_id)

st.sidebar.code(str(vdir), language="bash")
st.sidebar.write(f"Epochs found: {epochs if epochs else '[]'}")

if not epochs:
    st.warning("Selected run has no volumes yet. Wait for the sim to write epoch_####.npz")
    st.stop()

epoch = st.sidebar.selectbox("Epoch", epochs, index=len(epochs)-1)

# ---- main view ----
st.title("Fuka 5.0 — Space–Time Capacitor Substrate")
data, path_str = load_npz_local(gcp_cfg, run_id, epoch)
st.success(f"Loaded {path_str}")
info = pd.DataFrame(
    [{"key": k, "shape": tuple(np.array(v).shape), "dtype": str(np.array(v).dtype)} for k, v in data.items()]
)
st.subheader("NPZ contents")
st.dataframe(info, use_container_width=True)

# (Optional) put simple stats so the page shows something even on phone
if "rho" in data:
    rho = np.array(data["rho"], dtype=float)
    st.write(f"rho min/max: {float(rho.min()):.3g} / {float(rho.max()):.3g} | mean {float(rho.mean()):.3g}")
