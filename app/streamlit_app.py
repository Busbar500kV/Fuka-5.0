from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import pyarrow.parquet as pq  # for edge shards

# -------------------
# Local run directory
# -------------------
RUNS_BASE = Path("/home/busbar/fuka-runs/runs")
assert RUNS_BASE.exists(), f"{RUNS_BASE} does not exist!"

st.set_page_config(page_title="Fuka 5.0", layout="wide")

# -------------------
# Helpers
# -------------------
def _list_runs() -> list[str]:
    return sorted([p.name for p in RUNS_BASE.iterdir() if p.is_dir()])

def _list_epochs(run_id: str) -> Tuple[list[int], Path]:
    vdir = RUNS_BASE / run_id / "volumes"
    eps: list[int] = []
    if vdir.is_dir():
        for p in vdir.glob("epoch_*.npz"):
            try:
                eps.append(int(p.stem.split("_")[1]))
            except Exception:
                pass
    return sorted(eps), vdir

@st.cache_data(show_spinner=False, ttl=5)
def _load_npz(run_id: str, epoch: int) -> Tuple[Dict[str, np.ndarray], Path]:
    path = RUNS_BASE / run_id / "volumes" / f"epoch_{epoch:04d}.npz"
    with np.load(path) as z:
        data = {k: z[k] for k in z.files}
    return data, path

@st.cache_data(show_spinner=True, ttl=30)
def _load_edges_for_epoch(run_id: str, epoch: int, limit: int = 5000) -> pd.DataFrame:
    shards_dir = RUNS_BASE / run_id / "shards"
    if not shards_dir.exists():
        return pd.DataFrame()
    dfs = []
    for f in sorted(shards_dir.glob("edges-*.parquet")):
        try:
            tbl = pq.read_table(f)
            df = tbl.to_pandas()
        except Exception:
            continue
        dfe = df[df["epoch"] == int(epoch)]
        if not dfe.empty:
            dfs.append(dfe)
    if not dfs:
        return pd.DataFrame()
    df_all = pd.concat(dfs, ignore_index=True)
    if len(df_all) > limit:
        df_all = df_all.sample(limit, random_state=0).reset_index(drop=True)
    return df_all

def _isosurface_figure(rho: np.ndarray, iso_pct: int, opacity: float) -> go.Figure:
    x = np.arange(rho.shape[0])
    y = np.arange(rho.shape[1])
    z = np.arange(rho.shape[2])
    iso = np.nanpercentile(rho, iso_pct)

    fig = go.Figure(go.Isosurface(
        x=np.repeat(x, len(y) * len(z)),
        y=np.tile(np.repeat(y, len(z)), len(x)),
        z=np.tile(z, len(x) * len(y)),
        value=rho.ravel(order="F"),
        isomin=iso,
        isomax=float(np.nanmax(rho)),
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=opacity,
        showscale=False,
    ))
    fig.update_layout(scene=dict(aspectmode="data"),
                      margin=dict(l=0, r=0, t=0, b=0),
                      height=700)
    return fig

def _mask_points(mask: np.ndarray, stride: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    zz, yy, xx = np.where(mask)
    if zz.size == 0:
        return np.array([]), np.array([]), np.array([])
    sel = (np.arange(zz.size) % max(1, int(stride))) == 0
    return xx[sel], yy[sel], zz[sel]

# -------------------
# Sidebar: selection
# -------------------
st.sidebar.header("Select run / epoch (local)")

runs = _list_runs()
if not runs:
    st.sidebar.error(f"No runs found in {RUNS_BASE}.")
    st.stop()

run_id = st.sidebar.selectbox("Run", runs, index=len(runs) - 1)
epochs, vdir = _list_epochs(run_id)
st.sidebar.code(str(vdir))
st.sidebar.write(f"Epochs found: {epochs if epochs else '[]'}")

if not epochs:
    st.warning("No epoch_####.npz yet—let the sim write a few.")
    st.stop()

# keep a stable index across reruns
if "epoch_idx" not in st.session_state:
    st.session_state.epoch_idx = len(epochs) - 1

c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("⏪ Prev"):
        st.session_state.epoch_idx = max(0, st.session_state.epoch_idx - 1)
        st.rerun()
with c2:
    if st.button("⏩ Next"):
        st.session_state.epoch_idx = min(len(epochs) - 1, st.session_state.epoch_idx + 1)
        st.rerun()

epoch = st.sidebar.selectbox("Epoch", epochs,
                             index=st.session_state.epoch_idx,
                             key="epoch_select")
# sync session index with dropdown choice
try:
    st.session_state.epoch_idx = epochs.index(epoch)
except Exception:
    pass

iso_pct = st.sidebar.slider("Isosurface percentile (rho)", 50, 99, 75)
opacity = st.sidebar.slider("Isosurface opacity", 1, 100, 25) / 100.0

# -------------------
# Main content
# -------------------
st.title("Fuka 5.0 — Space–Time Capacitor Substrate")
data, path = _load_npz(run_id, epoch)
rho = np.array(data["rho"])
st.success(f"Loaded {path}")

fig = _isosurface_figure(rho, iso_pct=iso_pct, opacity=opacity)

# ----- Nodes overlay -----
with st.sidebar.expander("Nodes overlay", expanded=False):
    show_core   = st.checkbox("Show core nodes", value=True,  key="nodes_show_core")
    show_outer  = st.checkbox("Show outer nodes", value=True, key="nodes_show_outer")
    node_stride = st.slider("Node stride (display sampling)", 1, 12, 3, key="nodes_stride")
    node_size   = st.slider("Marker size", 1, 10, 3, key="nodes_size")
    fallback_rho_points = st.checkbox("Fallback: show rho points if masks missing",
                                      value=True, key="nodes_fallback_rho")
    rho_quantile = st.slider("rho threshold quantile", 0.90, 0.999, 0.97,
                             key="nodes_rho_quantile")

core  = data.get("core_mask")
outer = data.get("outer_mask")

if show_core and core is not None:
    xx, yy, zz = _mask_points(core, node_stride)
    if xx.size:
        fig.add_trace(go.Scatter3d(
            x=xx, y=yy, z=zz, mode="markers", name="core",
            marker=dict(size=node_size, color="yellow"), opacity=1.0))

if show_outer and outer is not None:
    xx, yy, zz = _mask_points(outer, node_stride)
    if xx.size:
        fig.add_trace(go.Scatter3d(
            x=xx, y=yy, z=zz, mode="markers", name="outer",
            marker=dict(size=node_size, color="magenta"), opacity=0.9))

if (core is None and outer is None) and fallback_rho_points:
    thr = float(np.quantile(rho[np.isfinite(rho)], float(rho_quantile)))
    zz, yy, xx = np.where(rho >= thr)
    if zz.size:
        sel = (np.arange(zz.size) % max(1, int(node_stride))) == 0
        xx, yy, zz = xx[sel], yy[sel], zz[sel]
        fig.add_trace(go.Scatter3d(
            x=xx, y=yy, z=zz, mode="markers",
            name=f"rho≥q{rho_quantile:.3f}",
            marker=dict(size=node_size, color="cyan"), opacity=0.9))

# ----- Edges overlay -----
with st.sidebar.expander("Edges overlay", expanded=False):
    show_edges   = st.checkbox("Show edges", value=True)
    color_key    = st.selectbox("Color by", ["weight", "energy", "dist", "phase"], index=0)
    max_edges    = st.slider("Max edges", 1000, 20000, 5000, step=1000)
    edge_opacity = st.slider("Edge opacity (%)", 10, 100, 95) / 100

edges = pd.DataFrame()
if show_edges:
    edges = _load_edges_for_epoch(run_id, epoch, limit=max_edges)

    # stats panel
    with st.expander("Edge stats", expanded=False):
        if edges.empty:
            st.info("No edges for this epoch.")
        else:
            st.write(f"Rows: **{len(edges):,}**")
            st.dataframe(
                edges[["epoch", "src", "dst", "dist", "weight", "energy", "phase"]]
                .describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95])
            )

    if not edges.empty:
        # build 3D polyline batch (x,y,z with NaN separators)
        # coordinates are derived from flat indices using Fortran order (z,y,x)
        sh = rho.shape
        def unravel_fortran(idx: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
            z = idx % sh[0]
            y = (idx // sh[0]) % sh[1]
            x = idx // (sh[0] * sh[1])
            return x.astype(float), y.astype(float), z.astype(float)

        xu, yu, zu = unravel_fortran(edges["src"].to_numpy(np.int64))
        xv, yv, zv = unravel_fortran(edges["dst"].to_numpy(np.int64))

        X = np.vstack([xu, xv, np.full_like(xu, np.nan)]).T.reshape(-1)
        Y = np.vstack([yu, yv, np.full_like(yu, np.nan)]).T.reshape(-1)
        Z = np.vstack([zu, zv, np.full_like(zu, np.nan)]).T.reshape(-1)

        # color by selected metric with robust 5–95% clamp
        C = edges[color_key].to_numpy(dtype=float)
        cmin, cmax = np.percentile(C, 5), np.percentile(C, 95)
        if cmax <= cmin:
            cmax = cmin + 1e-9
        Cn = (np.clip(C, cmin, cmax) - cmin) / (cmax - cmin)
        Cline = np.repeat(Cn, 3)
        Cline[2::3] = np.nan

        fig.add_trace(go.Scatter3d(
            x=X, y=Y, z=Z, mode="lines",
            line=dict(width=2, color=Cline, colorscale="Viridis"),
            opacity=edge_opacity, name=f"Edges ({len(edges)})",
            hoverinfo="skip"
        ))

# ----- Render -----
st.plotly_chart(fig, use_container_width=True)

# ----- NPZ contents quick table -----
st.subheader("NPZ contents")
st.dataframe(pd.DataFrame(
    [{"key": k, "shape": list(v.shape), "dtype": str(v.dtype)} for k, v in data.items()]
))