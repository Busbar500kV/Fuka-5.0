from __future__ import annotations
import streamlit as st
st.set_page_config(page_title='Fuka 5.0', layout='wide')
# ensure repo root is on sys.path when running from app/ subdir
from pathlib import Path
import sys, os
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
# enable gs:// -> local runtime compatibility when F5_STORAGE=local
from fuka5.io.compat_shim import init_backend_shims
init_backend_shims()
from fuka5.io.storage_facade import (
    load_config,
    list_runs,
    list_blobs,
    storage_path,
    read_parquet,
    read_npz,
    read_json,
    download_to_tmp,
)
CFG_PATH = "configs/local.default.json" if os.getenv("F5_STORAGE","gcs").lower()=="local" else "configs/gcp.default.json"
gcp_path = CFG_PATH
"""
Streamlit UI for Fuka 5.0
-------------------------
Browse GCS runs, pick an epoch, and visualize:
  - 3D charge-density rho (isosurface) with outer/core masks
  - Substrate edges overlayed (filters: band, gate mode, C/B percentiles)
  - Metrics time series and histograms
Assumptions:
- GCS credentials via ADC on the VM (or locally).
"""
import os
import tempfile
import json
from typing import Dict, Any, List, Tuple
import numpy as np
import pandas as pd

import plotly.graph_objects as go
from fuka5 import env_get
# Components (shipped separately)
try:
    from app.components.plot3d import make_volume_fig, add_edges_to_fig
    from app.components.panels import metrics_panel, filters_panel
except Exception:
    # Minimal fallbacks if components not yet shipped
    def make_volume_fig(volumes: Dict[str, np.ndarray], iso_level: float = 0.4) -> go.Figure:
        rho = volumes.get("rho")
        if rho is None:
            return go.Figure()
        nx, ny, nz = rho.shape
        fig = go.Figure(data=[
            go.Volume(
                value=rho.transpose(2,1,0),  # z,y,x for plotly
                opacity=0.1,
                surface_count=10,
                showscale=True,
            )
        ])
        fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0,r=0,t=0,b=0))
        return fig
    def add_edges_to_fig(fig: go.Figure, edges_df: pd.DataFrame, color_key: str = "C"):
        if edges_df.empty:
            return fig
        for _, r in edges_df.iterrows():
            fig.add_trace(go.Scatter3d(
                x=[r["z_u"], r["z_v"]], y=[r["y_u"], r["y_v"]], z=[r["x_u"], r["x_v"]],
                mode="lines", line=dict(width=2),
                name="edge", hovertext=f'{r["edge_id"]} {color_key}={r.get(color_key,""):.3g}',
                hoverinfo="text"
            ))
        return fig
    def metrics_panel(metrics_df: pd.DataFrame):
        st.subheader("Metrics (raw)")
        st.dataframe(metrics_df)
    def filters_panel():
        return {
            "band": st.selectbox("Band", ["low", "high"]),
            "gate_mode": st.selectbox("Gate mode", ["mix", "1", "1p"]),
            "min_C_pct": st.slider("Min C percentile", 0.0, 100.0, 20.0, 1.0),
            "min_B_pct": st.slider("Min B percentile", 0.0, 100.0, 20.0, 1.0),
            "color_key": st.selectbox("Edge color", ["C", "B", "A", "E"]),
        }
# ---------------------------
# Caching helpers
# ---------------------------
@st.cache_data(show_spinner=False)
def _load_gcp(gcp_path: str) -> Dict[str, Any]:
    return load_config(gcp_path)
@st.cache_data(show_spinner=True, ttl=60)
def _list_runs_cached(gcp_cfg: Dict[str, Any]) -> List[str]:
    return list_runs(gcp_cfg)
@st.cache_data(show_spinner=False)
def _list_run_blobs(gcp_cfg: Dict[str, Any], run_id: str) -> Dict[str, List[str]]:
    base = storage_path(gcp_cfg, run_id)
    blobs = list_blobs(gcp_cfg, base)
    vols = [b for b in blobs if b.startswith(f"{gcp_cfg['runs_prefix'].strip('/')}/{run_id}/volumes/")]
    shards = [b for b in blobs if b.startswith(f"{gcp_cfg['runs_prefix'].strip('/')}/{run_id}/shards/")]
    return {"volumes": vols, "shards": shards}
def _download_to_tmp(gcp_cfg: Dict[str, Any], blob_gs_path: str) -> str:
    tmpdir = tempfile.gettempdir()
    fname = blob_gs_path.split("/")[-1]
    local_path = os.path.join(tmpdir, "fuka5_cache", fname)
    os.makedirs(os.path.dirname(local_path), exist_ok=True)
    download_blob_to_file(gcp_cfg, blob_gs_path, local_path)
    return local_path
@st.cache_data(show_spinner=False)
def _load_volume_npz(gcp_cfg: Dict[str, Any], gs_path: str) -> Dict[str, np.ndarray]:
    local = _download_to_tmp(gcp_cfg, gs_path)
    with np.load(local) as z:
        return {k: z[k] for k in z.files}
@st.cache_data(show_spinner=False)
def _load_parquet(gcp_cfg: Dict[str, Any], gs_path: str) -> pd.DataFrame:
    local = _download_to_tmp(gcp_cfg, gs_path)
    return pd.read_parquet(local)
# ---------------------------
# UI
# ---------------------------
st.title("Fuka 5.0 — Space–Time Capacitor Substrate")
# GCP config
default_gcp_path = os.environ.get("F5_GCP_CONFIG", "configs/gcp.default.json")
gcp_cfg = _load_gcp(gcp_path)
st.sidebar.markdown(f"**Project:** `{gcp_cfg.get('project_id','')}`")
st.sidebar.markdown(f"**Bucket:** `{gcp_cfg.get('bucket','')}`")
# Runs
runs = _list_runs_cached(gcp_cfg)
if not runs:
    st.warning("No runs found in the bucket. Start a run with scripts/run_sim.sh")
    st.stop()
run_id = st.sidebar.selectbox("Run", runs, index=max(0, len(runs)-1))
st.sidebar.write(f"Selected run: `{run_id}`")
# List blobs for run, extract epochs from volumes
blobs = _list_run_blobs(gcp_cfg, run_id)
vol_blobs = sorted(blobs["volumes"])
if not vol_blobs:
    st.warning("No volume snapshots found for this run yet.")
    st.stop()
def _epoch_from_volname(name: str) -> int:
    # runs/<run>/volumes/epXXX.npz
    base = os.path.basename(name)
    try:
        return int(base.split("ep")[1].split(".")[0])
    except Exception:
        return 0
epochs = sorted([_epoch_from_volname(b) for b in vol_blobs])
epoch = st.sidebar.slider("Epoch", min_value=int(min(epochs)), max_value=int(max(epochs)), value=int(max(epochs)), step=1)
# Find matching volume npz
vol_gs = None
for b in vol_blobs:
    if f"ep{epoch:03d}.npz" in b:
        rel_path = b  # relative path under runs/<RUN_ID>/...
        break
if vol_gs is None:
    st.error("Selected epoch volume not found.")
    st.stop()
# Load volume & render
volumes = _load_volume_npz(gcp_cfg, vol_gs)
# Load shards (edges + metrics). We’ll load all and filter client-side (they’re small-ish).
edge_shards = [b for b in blobs["shards"] if "edges_" in b]
metric_shards = [b for b in blobs["shards"] if "metrics_" in b]
edges_df = pd.DataFrame()
for sh in sorted(edge_shards):
    df = read_parquet(gcp_cfg, sh)
    edges_df = pd.concat([edges_df, df], ignore_index=True)
metrics_df = pd.DataFrame()
for sh in sorted(metric_shards):
    dfm = read_parquet(gcp_cfg, sh)
    metrics_df = pd.concat([metrics_df, dfm], ignore_index=True)
# Filter by epoch
edges_ep = edges_df[edges_df["epoch"] == epoch].copy()
if edges_ep.empty:
    st.warning("No edge records for this epoch (yet). Try another epoch.")
    st.stop()
# Panels: metrics overview
with st.container():
    left, right = st.columns([2,1], gap="large")
    with left:
        st.subheader("3D World & Substrate")
        fig = make_volume_fig(volumes, iso_level=0.4)
        # Filters
        with st.expander("Edge overlay filters", expanded=True):
            filt = filters_panel()
        band = filt["band"]; mode = filt["gate_mode"]; color_key = filt["color_key"]
        # Thresholds
        c_thr = np.percentile(edges_ep["C"], filt["min_C_pct"]) if len(edges_ep) else 0.0
        b_thr = np.percentile(edges_ep["B"], filt["min_B_pct"]) if len(edges_ep) else 0.0
        # Gate column name
        gate_col = f"g_{band}_{'mix' if mode=='mix' else ('1' if mode=='1' else '1p')}"
        # Select edges meeting thresholds and with gate emphasis
        show = edges_ep[(edges_ep["C"] >= c_thr) & (edges_ep["B"] >= b_thr)]
        # Sort by gate weight descending to show more relevant edges
        if gate_col in show:
            show = show.sort_values(gate_col, ascending=False).head(200)
        fig = add_edges_to_fig(fig, show, color_key=color_key)
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})
    with right:
        st.subheader("Run & Epoch")
        st.json({"run_id": run_id, "epoch": int(epoch)})
        st.subheader("Quick metrics")
        try:
            last_metrics = metrics_df[metrics_df["epoch"] <= epoch].tail(50)
        except Exception:
            last_metrics = metrics_df
        metrics_panel(last_metrics)
# Per-edge table (top-K)
st.subheader("Top edges")
top_key = st.selectbox("Sort by", ["B","C","A","E","P_low_s1","P_high_s1","P_low_s1p","P_high_s1p"], index=0)
topk = edges_ep.sort_values(top_key, ascending=False).head(50)
st.dataframe(topk, use_container_width=True)
st.caption("Tip: Adjust volume snapshot cadence and downsampling in training/world configs for smoother UI.")
