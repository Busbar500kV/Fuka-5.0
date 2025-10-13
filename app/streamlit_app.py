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












# ====== Nodes overlay
with st.sidebar.expander("Nodes overlay", expanded=False):
    show_core   = st.checkbox("Show core nodes", value=True,  key="nodes_show_core")
    show_outer  = st.checkbox("Show outer nodes", value=False, key="nodes_show_outer")
    node_stride = st.slider("Node stride", 1, 12, 3, key="nodes_stride")
    node_size   = st.slider("Marker size", 1, 10, 4, key="nodes_size")
    fallback_rho_points = st.checkbox("Fallback: show rho points if masks missing",
                                      value=True, key="nodes_fallback_rho")
    rho_quantile = st.slider("rho threshold quantile", 0.90, 0.999, 0.97,
                             key="nodes_rho_quantile")

# --- DEBUG: show the variables we rely on
_dbg_run_id  = locals().get("run_id", None)
_dbg_epoch   = locals().get("epoch", None)
_dbg_gcp_cfg = locals().get("gcp_cfg", None)
# st.caption(f"[DBG] run_id={_dbg_run_id} epoch={_dbg_epoch} gcp_cfg={_dbg_gcp_cfg is not None}")

from pathlib import Path as _Path
_runs_dir  = _Path((_dbg_gcp_cfg or {}).get("runs_dir", "/home/busbar/fuka-runs")).expanduser()
_runs_pref = (_dbg_gcp_cfg or {}).get("runs_prefix", "runs").strip("/")

_epoch_path = None
if _dbg_run_id is not None and _dbg_epoch is not None:
    _epoch_path = _runs_dir / f"{_runs_pref}/{_dbg_run_id}/volumes/epoch_{int(_dbg_epoch):04d}.npz"
# st.caption(f"[DBG] epoch_path={_epoch_path} exists={( _epoch_path and _epoch_path.exists() )}")

_core_ct = _outer_ct = _rho_ct = 0
try:
    import numpy as np
    if _epoch_path and _epoch_path.exists():
        npz = np.load(_epoch_path, allow_pickle=False)
#         st.caption(f"[DBG] npz keys: {list(npz.keys())}")

        core  = npz.get("core_mask")
        outer = npz.get("outer_mask")

        def _mask_points(mask, stride):
            z, y, x = np.where(mask)
            if z.size == 0: return np.array([]), np.array([]), np.array([])
            sel = (np.arange(z.size) % max(1,int(stride))) == 0
            return x[sel], y[sel], z[sel]

        if show_core and core is not None:
            _x,_y,_z = _mask_points(core, node_stride); _core_ct = _x.size
            if _core_ct:
                fig.add_trace(go.Scatter3d(
                    x=_x, y=_y, z=_z, mode="markers", name="core",
                    marker=dict(size=node_size, color="yellow"), opacity=1.0))

        if show_outer and outer is not None:
            _x,_y,_z = _mask_points(outer, node_stride); _outer_ct = _x.size
            if _outer_ct:
                fig.add_trace(go.Scatter3d(
                    x=_x, y=_y, z=_z, mode="markers", name="outer",
                    marker=dict(size=node_size, color="magenta"), opacity=0.9))

        if fallback_rho_points and (_core_ct + _outer_ct == 0):
            rho = npz.get("rho")
            if rho is not None:
                thr = float(np.quantile(rho[np.isfinite(rho)], float(rho_quantile)))
                zz, yy, xx = np.where(rho >= thr)
                if zz.size:
                    sel = (np.arange(zz.size) % max(1,int(node_stride))) == 0
                    xx, yy, zz = xx[sel], yy[sel], zz[sel]
                    _rho_ct = int(xx.size)
                    fig.add_trace(go.Scatter3d(
                        x=xx, y=yy, z=zz, mode="markers", name=f"rho≥q{rho_quantile:.3f}",
                        marker=dict(size=node_size, color="cyan"), opacity=0.9))
except Exception as _e:
    st.caption(f"[nodes overlay] {type(_e).__name__}: {_e}")

# st.caption(f"Nodes overlay: core={_core_ct} outer={_outer_ct} rhoPts={_rho_ct}")
# ====== /Nodes overlay ======
st.plotly_chart(fig, use_container_width=True)
# ====== Edge overlay (auto-inserted) ======
import pandas as pd, numpy as np
from pathlib import Path
from plotly import graph_objects as go

_run_dir = Path('/home/busbar/fuka-runs/runs')/run_id
_shards_dir = _run_dir/'shards'

@st.cache_data(show_spinner=True, ttl=30)
def _load_edges_epoch(ep, max_rows=8000):
    import glob
    import pyarrow.parquet as pq  # noqa: F401
    dfs=[]
    for f in glob.glob(str(_shards_dir/'edges_*.parquet')):
        try:
            df = pd.read_parquet(f, engine='pyarrow')
        except Exception:
            continue
        dfe = df[df['epoch'] == int(ep)]
        if not dfe.empty:
            dfs.append(dfe)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs).head(max_rows)

with st.sidebar.expander('Edges overlay', expanded=False):
    show_edges   = st.checkbox('Show edges', value=True)
    color_key    = st.selectbox('Color by', ['C','B','A','E'], index=0)
    max_edges    = st.slider('Max edges', 1000, 20000, 5000, step=1000)
    edge_opacity = st.slider('Edge opacity (%)', 10, 100, 50)/100

if show_edges:
    edges = _load_edges_epoch(epoch, max_rows=max_edges)
    if not edges.empty:
        X = np.vstack([edges.x_u, edges.x_v, np.full_like(edges.x_u, np.nan)]).T.reshape(-1)
        Y = np.vstack([edges.y_u, edges.y_v, np.full_like(edges.y_u, np.nan)]).T.reshape(-1)
        Z = np.vstack([edges.z_u, edges.z_v, np.full_like(edges.z_u, np.nan)]).T.reshape(-1)
        C = edges[color_key].to_numpy(dtype=float)
        cmin, cmax = np.percentile(C,5), np.percentile(C,95)
        if cmax <= cmin: cmax = cmin + 1e-9
        Cn = (np.clip(C, cmin, cmax) - cmin) / (cmax - cmin)
        Cline = np.repeat(Cn, 3); Cline[2::3] = np.nan
        fig.add_trace(go.Scatter3d(
            x=X, y=Y, z=Z, mode='lines',
            line=dict(width=2, color=Cline, colorscale='Viridis'),
            opacity=edge_opacity, name=f'Edges ({len(edges)})'
        ))
        st.plotly_chart(fig, use_container_width=True)
# ====== End overlay ======


st.subheader("NPZ contents")
st.dataframe(pd.DataFrame(
    [{"key": k, "shape": list(v.shape), "dtype": str(v.dtype)} for k, v in data.items()]
))
