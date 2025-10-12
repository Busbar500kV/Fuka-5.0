from __future__ import annotations
import os, json
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -------------------
# Hard-set local dirs
# -------------------
RUNS_BASE = Path("/home/busbar/fuka-runs/runs")
assert RUNS_BASE.exists(), f"{RUNS_BASE} does not exist!"

st.set_page_config(page_title="Fuka 5.0", layout="wide")

def list_runs():
    return sorted([p.name for p in RUNS_BASE.iterdir() if p.is_dir()])

def list_epochs(run_id):
    vdir = RUNS_BASE / run_id / "volumes"
    eps = []
    if vdir.is_dir():
        for p in vdir.glob("epoch_*.npz"):
            try: eps.append(int(p.stem.split("_")[1]))
            except: pass
    return sorted(eps), vdir

def load_npz(run_id, epoch):
    path = RUNS_BASE / run_id / "volumes" / f"epoch_{epoch:04d}.npz"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}, path

# ---- sidebar ----
st.sidebar.header("Select run / epoch (local)")
runs = list_runs()
if not runs:
    st.sidebar.error(f"No runs found in {RUNS_BASE}.")
    st.stop()

run_id = st.sidebar.selectbox("Run", runs, index=len(runs)-1)
epochs, vdir = list_epochs(run_id)
st.sidebar.code(str(vdir))
st.sidebar.write(f"Epochs found: {epochs if epochs else '[]'}")

if not epochs:
    st.warning("No epoch_####.npz yet—let the sim write a few.")
    st.stop()

epoch = st.sidebar.selectbox("Epoch", epochs, index=len(epochs)-1)
iso_pct = st.sidebar.slider("Isosurface percentile (rho)", 50, 99, 75)
opacity = st.sidebar.slider("Isosurface opacity", 1, 100, 25) / 100.0

# ---- main ----
st.title("Fuka 5.0 — Space–Time Capacitor Substrate")
data, path = load_npz(run_id, epoch)
rho = np.array(data["rho"])
st.success(f"Loaded {path}")

# ---- 3D figure ----
x, y, z = np.arange(rho.shape[0]), np.arange(rho.shape[1]), np.arange(rho.shape[2])
iso = np.nanpercentile(rho, iso_pct)
fig = go.Figure(go.Isosurface(
    x=np.repeat(x, len(y)*len(z)),
    y=np.tile(np.repeat(y, len(z)), len(x)),
    z=np.tile(z, len(x)*len(y)),
    value=rho.ravel(order="F"),
    isomin=iso, isomax=rho.max(),
    caps=dict(x_show=False, y_show=False, z_show=False),
    opacity=opacity,
    showscale=False,
))
fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=0, b=0), height=700)
st.plotly_chart(fig, use_container_width=True)

st.subheader("NPZ contents")
st.dataframe(pd.DataFrame(
    [{"key": k, "shape": list(v.shape), "dtype": str(v.dtype)} for k, v in data.items()]
))
