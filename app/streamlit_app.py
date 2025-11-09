from __future__ import annotations
from pathlib import Path
import json
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

# -------------------
# Local runs root
# -------------------
RUNS_BASE = Path("/home/busbar/fuka-runs/runs")
assert RUNS_BASE.exists(), f"{RUNS_BASE} does not exist!"
st.set_page_config(page_title="Fuka 5.0 — Space–Time Capacitor Substrate", layout="wide")


# -------------------
# Helpers
# -------------------
def list_runs() -> list[str]:
    return sorted([p.name for p in RUNS_BASE.iterdir() if p.is_dir()])

def list_epochs(run_id: str):
    vdir = RUNS_BASE / run_id / "volumes"
    eps = []
    if vdir.is_dir():
        for p in vdir.glob("epoch_*.npz"):
            try:
                eps.append(int(p.stem.split("_")[1]))
            except Exception:
                pass
    return sorted(eps), vdir

def load_npz(run_id: str, epoch: int):
    path = RUNS_BASE / run_id / "volumes" / f"epoch_{epoch:04d}.npz"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}, path

def load_manifest(run_id: str) -> dict:
    mf = RUNS_BASE / run_id / "manifest.json"
    if mf.exists():
        try:
            return json.loads(mf.read_text())
        except Exception:
            pass
    return {}

def shards_dir(run_id: str) -> Path:
    return RUNS_BASE / run_id / "shards"

def edges_file_for_epoch(run_id: str, ep: int) -> Path | None:
    sd = shards_dir(run_id)
    if not sd.exists():
        return None
    p4 = sd / f"edges-{ep:04d}.parquet"
    if p4.exists():
        return p4
    p3 = sd / f"edges-{ep:03d}.parquet"
    if p3.exists():
        return p3
    return None

@st.cache_data(show_spinner=True, ttl=30)
def load_edges_epoch(run_id: str, ep: int, max_rows: int = 8000) -> pd.DataFrame:
    # Prefer exact file match, fall back to scanning all shards
    cand = edges_file_for_epoch(run_id, ep)
    dfs: list[pd.DataFrame] = []
    if cand is not None:
        try:
            df = pd.read_parquet(cand, engine="pyarrow")
            if "epoch" in df.columns:
                df = df[df["epoch"] == int(ep)]
            if not df.empty:
                dfs.append(df)
        except Exception:
            pass
    if not dfs:
        sd = shards_dir(run_id)
        if sd.exists():
            for f in sorted(sd.glob("edges-*.parquet")):
                try:
                    dfx = pd.read_parquet(f, engine="pyarrow")
                    if "epoch" in dfx.columns:
                        dfx = dfx[dfx["epoch"] == int(ep)]
                    if not dfx.empty:
                        dfs.append(dfx)
                except Exception:
                    continue
    if not dfs:
        return pd.DataFrame(columns=["epoch","src","dst","dist","weight","energy","phase"])
    return pd.concat(dfs, ignore_index=True).head(max_rows)

def build_stride_node_coords(shape: tuple[int,int,int], stride: int) -> np.ndarray:
    """
    Return array of shape (N,3) with rows [z,y,x] for the strided lattice used by the graph.
    Ordering matches np.meshgrid(indexing='ij') with C-order ravel.
    """
    zs = np.arange(0, shape[0], stride, dtype=int)
    ys = np.arange(0, shape[1], stride, dtype=int)
    xs = np.arange(0, shape[2], stride, dtype=int)
    Z, Y, X = np.meshgrid(zs, ys, xs, indexing="ij")
    coords = np.vstack([Z.ravel(order="C"), Y.ravel(order="C"), X.ravel(order="C")]).T
    return coords


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

# epoch selection with Prev/Next that keeps state in sync
if "epoch_idx" not in st.session_state:
    st.session_state.epoch_idx = len(epochs)-1
c1, c2 = st.sidebar.columns(2)
with c1:
    if st.button("⏪ Prev"):
        st.session_state.epoch_idx = max(0, st.session_state.epoch_idx-1)
        st.rerun()
with c2:
    if st.button("⏩ Next"):
        st.session_state.epoch_idx = min(len(epochs)-1, st.session_state.epoch_idx+1)
        st.rerun()
epoch = st.sidebar.selectbox("Epoch", epochs, index=st.session_state.epoch_idx, key="epoch_select")
try:
    st.session_state.epoch_idx = epochs.index(epoch)
except Exception:
    pass

iso_pct = st.sidebar.slider("Isosurface percentile (rho)", 50, 99, 75)
opacity = st.sidebar.slider("Isosurface opacity", 1, 100, 25) / 100.0

# ---- main ----
st.title("Fuka 5.0 — Space–Time Capacitor Substrate")
data, path = load_npz(run_id, epoch)
rho = np.array(data["rho"])
st.success(f"Loaded {path}")

# ---- 3D base isosurface ----
x, y, z = np.arange(rho.shape[2]), np.arange(rho.shape[1]), np.arange(rho.shape[0])  # note: order
iso = np.nanpercentile(rho, iso_pct)
fig = go.Figure(go.Isosurface(
    # Plotly expects x,y,z aligned with value's ravel order; we feed C-order raveled rho[z,y,x]
    x=np.repeat(x, len(y)*len(z)),
    y=np.tile(np.repeat(y, len(z)), len(x)),
    z=np.tile(z, len(x)*len(y)),
    value=rho.transpose(2,1,0).ravel(order="C"),  # transpose so C-order aligns to x,y,z
    isomin=iso, isomax=float(np.nanmax(rho)),
    caps=dict(x_show=False, y_show=False, z_show=False),
    opacity=opacity,
    showscale=False,
))
fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=0, b=0), height=700)

# ====== Nodes overlay ======
with st.sidebar.expander("Nodes overlay", expanded=False):
    show_core   = st.checkbox("Show core nodes", value=True,  key="nodes_show_core")
    show_outer  = st.checkbox("Show outer nodes", value=True, key="nodes_show_outer")
    node_stride = st.slider("Node stride (display sampling)", 1, 12, 3, key="nodes_stride")
    node_size   = st.slider("Marker size", 1, 10, 4, key="nodes_size")
    fallback_rho_points = st.checkbox("Fallback: show rho points if masks missing",
                                      value=True, key="nodes_fallback_rho")
    rho_quantile = st.slider("rho threshold quantile", 0.90, 0.999, 0.97,
                             key="nodes_rho_quantile")

core = data.get("core_mask")
outer = data.get("outer_mask")

def _mask_points(mask: np.ndarray, stride: int):
    # mask is in (z,y,x)
    zc, yc, xc = np.where(mask)
    if zc.size == 0:
        return np.array([]), np.array([]), np.array([])
    sel = (np.arange(zc.size) % max(1, int(stride))) == 0
    return xc[sel], yc[sel], zc[sel]  # return as x,y,z

_core_ct = _outer_ct = _rho_ct = 0
try:
    if show_core and (core is not None):
        _x,_y,_z = _mask_points(core, node_stride); _core_ct = _x.size
        if _core_ct:
            fig.add_trace(go.Scatter3d(
                x=_x, y=_y, z=_z, mode="markers", name="core",
                marker=dict(size=node_size, color="yellow"), opacity=1.0))

    if show_outer and (outer is not None):
        _x,_y,_z = _mask_points(outer, node_stride); _outer_ct = _x.size
        if _outer_ct:
            fig.add_trace(go.Scatter3d(
                x=_x, y=_y, z=_z, mode="markers", name="outer",
                marker=dict(size=node_size, color="magenta"), opacity=0.9))

    if fallback_rho_points and (_core_ct + _outer_ct == 0):
        thr = float(np.quantile(rho[np.isfinite(rho)], float(rho_quantile)))
        zz, yy, xx = np.where(rho >= thr)
        if zz.size:
            sel = (np.arange(zz.size) % max(1, int(node_stride))) == 0
            xx, yy, zz = xx[sel], yy[sel], zz[sel]
            _rho_ct = int(xx.size)
            fig.add_trace(go.Scatter3d(
                x=xx, y=yy, z=zz, mode="markers", name=f"rho≥q{rho_quantile:.3f}",
                marker=dict(size=node_size, color="cyan"), opacity=0.9))
except Exception as _e:
    st.caption(f"[nodes overlay] {type(_e).__name__}: {_e}")

# ====== Edge overlay ======
with st.sidebar.expander('Edges overlay', expanded=False):
    show_edges   = st.checkbox('Show edges', value=True)
    color_key    = st.selectbox('Color by', ['weight','energy','dist','phase'], index=0)
    max_edges    = st.slider('Max edges', 1000, 20000, 5000, step=1000)
    edge_opacity = st.slider('Edge opacity (%)', 10, 100, 50)/100

edges_drawn = 0
if show_edges:
    # Load edges for the selected epoch
    edges = load_edges_epoch(run_id, int(epoch), max_rows=max_edges)
    if not edges.empty and {"src","dst"}.issubset(edges.columns):
        # Build stride-lattice coordinates from manifest
        mf = load_manifest(run_id)
        cfg = mf.get("configs", {})
        shape = tuple(int(v) for v in cfg.get("grid_shape", [64,64,64]))
        stride = int(cfg.get("graph_stride", 6))
        coords_zyx = build_stride_node_coords(shape, stride)  # (N,3) [z,y,x]

        # Guard against bad IDs
        N = coords_zyx.shape[0]
        eid_src = np.clip(edges["src"].to_numpy(np.int64), 0, N-1)
        eid_dst = np.clip(edges["dst"].to_numpy(np.int64), 0, N-1)

        # Map to x,y,z (swap order from z,y,x)
        xyz_u = coords_zyx[eid_src][:, [2,1,0]]  # (n,3)
        xyz_v = coords_zyx[eid_dst][:, [2,1,0]]

        # Build NaN-separated line segments for Plotly
        X = np.column_stack([xyz_u[:,0], xyz_v[:,0], np.full_like(eid_src, np.nan, dtype=float)]).ravel()
        Y = np.column_stack([xyz_u[:,1], xyz_v[:,1], np.full_like(eid_src, np.nan, dtype=float)]).ravel()
        Z = np.column_stack([xyz_u[:,2], xyz_v[:,2], np.full_like(eid_src, np.nan, dtype=float)]).ravel()

        # Color selection & normalization
        if color_key not in edges.columns:
            color_key = "weight"
        C = edges[color_key].to_numpy(dtype=float)
        cmin, cmax = np.percentile(C,5), np.percentile(C,95)
        if not np.isfinite(cmin): cmin = float(np.nanmin(C))
        if not np.isfinite(cmax): cmax = float(np.nanmax(C))
        if cmax <= cmin: cmax = cmin + 1e-9
        Cn = (np.clip(C, cmin, cmax) - cmin) / (cmax - cmin)
        Cline = np.repeat(Cn, 3); Cline[2::3] = np.nan

        fig.add_trace(go.Scatter3d(
            x=X, y=Y, z=Z, mode='lines',
            line=dict(width=2, color=Cline, colorscale='Viridis'),
            opacity=edge_opacity, name=f'Edges ({len(edges)})'
        ))
        edges_drawn = len(edges)
    else:
        st.info("No edges for this epoch (or missing src/dst columns).")

st.plotly_chart(fig, use_container_width=True)

# Small debug footer
st.caption(f"Run: {run_id} • Epoch: {epoch} • Nodes shown: core={_core_ct} outer={_outer_ct} rhoPts={_rho_ct} • Edges drawn: {edges_drawn}")