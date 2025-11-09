from __future__ import annotations
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

# -------------------
# Helpers
# -------------------
def list_runs():
    return sorted([p.name for p in RUNS_BASE.iterdir() if p.is_dir()])

def list_epochs(run_id):
    vdir = RUNS_BASE / run_id / "volumes"
    eps = []
    if vdir.is_dir():
        for p in vdir.glob("epoch_*.npz"):
            try:
                eps.append(int(p.stem.split("_")[1]))
            except:
                pass
    return sorted(eps), vdir

def load_npz(run_id, epoch):
    path = RUNS_BASE / run_id / "volumes" / f"epoch_{epoch:04d}.npz"
    with np.load(path) as z:
        return {k: z[k] for k in z.files}, path

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
    """Load edges for the given epoch. Columns present: ['epoch','src','dst','dist','weight','energy','phase']"""
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
        # fallback: scan & filter
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

# ---- 3D figure (isosurface) ----
x, y, z = np.arange(rho.shape[0]), np.arange(rho.shape[1]), np.arange(rho.shape[2])
iso = np.nanpercentile(rho, iso_pct)
fig = go.Figure(
    go.Isosurface(
        x=np.repeat(x, len(y) * len(z)),
        y=np.tile(np.repeat(y, len(z)), len(x)),
        z=np.tile(z, len(x) * len(y)),
        value=rho.ravel(order="F"),
        isomin=iso,
        isomax=rho.max(),
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=opacity,
        showscale=False,
    )
)
fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, t=0, b=0), height=700)

# ====== Nodes overlay ======
with st.sidebar.expander("Nodes overlay", expanded=False):
    show_core = st.checkbox("Show core nodes", value=True, key="nodes_show_core")
    show_outer = st.checkbox("Show outer nodes", value=True, key="nodes_show_outer")
    node_stride = st.slider("Node stride", 1, 12, 3, key="nodes_stride")
    node_size = st.slider("Marker size", 1, 10, 4, key="nodes_size")
    fallback_rho_points = st.checkbox(
        "Fallback: show rho points if masks missing", value=True, key="nodes_fallback_rho"
    )
    rho_quantile = st.slider(
        "rho threshold quantile", 0.90, 0.999, 0.97, key="nodes_rho_quantile"
    )

_dbg_run_id = locals().get("run_id", None)
_dbg_epoch = locals().get("epoch", None)

_runs_dir = Path("/home/busbar/fuka-runs").expanduser()
_runs_pref = "runs"

_epoch_path = None
if _dbg_run_id is not None and _dbg_epoch is not None:
    _epoch_path = _runs_dir / f"{_runs_pref}/{_dbg_run_id}/volumes/epoch_{int(_dbg_epoch):04d}.npz"

_core_ct = _outer_ct = _rho_ct = 0
try:
    if _epoch_path and _epoch_path.exists():
        npz = np.load(_epoch_path, allow_pickle=False)

        core = npz.get("core_mask")
        outer = npz.get("outer_mask")

        def _mask_points(mask, stride):
            z, y, x = np.where(mask)
            if z.size == 0:
                return np.array([]), np.array([]), np.array([])
            sel = (np.arange(z.size) % max(1, int(stride))) == 0
            return x[sel], y[sel], z[sel]

        if show_core and core is not None:
            _x, _y, _z = _mask_points(core, node_stride)
            _core_ct = _x.size
            if _core_ct:
                fig.add_trace(
                    go.Scatter3d(
                        x=_x,
                        y=_y,
                        z=_z,
                        mode="markers",
                        name="core",
                        marker=dict(size=node_size, color="yellow"),
                        opacity=1.0,
                    )
                )

        if show_outer and outer is not None:
            _x, _y, _z = _mask_points(outer, node_stride)
            _outer_ct = _x.size
            if _outer_ct:
                fig.add_trace(
                    go.Scatter3d(
                        x=_x,
                        y=_y,
                        z=_z,
                        mode="markers",
                        name="outer",
                        marker=dict(size=node_size, color="magenta"),
                        opacity=0.9,
                    )
                )

        if fallback_rho_points and (_core_ct + _outer_ct == 0):
            rho2 = npz.get("rho")
            if rho2 is not None:
                thr = float(np.quantile(rho2[np.isfinite(rho2)], float(rho_quantile)))
                zz, yy, xx = np.where(rho2 >= thr)
                if zz.size:
                    sel = (np.arange(zz.size) % max(1, int(node_stride))) == 0
                    xx, yy, zz = xx[sel], yy[sel], zz[sel]
                    _rho_ct = int(xx.size)
                    fig.add_trace(
                        go.Scatter3d(
                            x=xx,
                            y=yy,
                            z=zz,
                            mode="markers",
                            name=f"rho≥q{rho_quantile:.3f}",
                            marker=dict(size=node_size, color="cyan"),
                            opacity=0.9,
                        )
                    )
except Exception as _e:
    st.caption(f"[nodes overlay] {type(_e).__name__}: {_e}")

# ====== Edge overlay (indices → 3D coords) ======
from pathlib import Path as _P

with st.sidebar.expander("Edges overlay", expanded=False):
    show_edges   = st.checkbox("Show edges", value=True)
    color_key    = st.selectbox("Color by", ["weight","energy","dist","phase"], index=0)
    max_edges    = st.slider("Max edges", 1000, 20000, 5000, step=1000)
    edge_opacity = st.slider("Edge opacity (%)", 10, 100, 50) / 100

if show_edges:
    try:
        edges = load_edges_epoch(run_id, int(epoch), max_rows=max_edges)
        if not edges.empty:
            # Map flat node indices to (z,y,x) using rho.shape
            # Assumption: src/dst are C-order flattened indices.
            ZYZ = rho.shape  # (X,Y,Z) in our earlier isosurface convention uses ravel(order="F") for values
            # For coordinates, we'll use standard unravel_index (C-order). If orientation looks mirrored,
            # we can swap axes later — the goal is to render edges correctly in 3D.
            zi_u, yi_u, xi_u = np.unravel_index(edges["src"].to_numpy(dtype=np.int64), ZYZ, order="C")
            zi_v, yi_v, xi_v = np.unravel_index(edges["dst"].to_numpy(dtype=np.int64), ZYZ, order="C")

            # Build polyline arrays with NaN separators
            X = np.vstack([xi_u, xi_v, np.full_like(xi_u, np.nan)]).T.reshape(-1)
            Y = np.vstack([yi_u, yi_v, np.full_like(yi_u, np.nan)]).T.reshape(-1)
            Z = np.vstack([zi_u, zi_v, np.full_like(zi_u, np.nan)]).T.reshape(-1)

            # Color mapping
            if color_key in edges.columns:
                C = edges[color_key].to_numpy(dtype=float)
                cmin, cmax = np.percentile(C, 5), np.percentile(C, 95)
                if cmax <= cmin:
                    cmax = cmin + 1e-9
                Cn = (np.clip(C, cmin, cmax) - cmin) / (cmax - cmin)
                Cline = np.repeat(Cn, 3)
                Cline[2::3] = np.nan
                line_kwargs = dict(width=2, color=Cline, colorscale="Viridis")
            else:
                line_kwargs = dict(width=2)

            fig.add_trace(
                go.Scatter3d(
                    x=X, y=Y, z=Z, mode="lines",
                    line=line_kwargs,
                    opacity=edge_opacity,
                    name=f"Edges ({len(edges):,})",
                )
            )
        else:
            st.info(f"No edges found for epoch {epoch}.")
    except Exception as e:
        st.warning(f"Failed to render edges for epoch {epoch}: {e}")

st.plotly_chart(fig, use_container_width=True)

# ====== NPZ contents ======
st.subheader("NPZ contents")
st.dataframe(
    pd.DataFrame(
        [{"key": k, "shape": list(v.shape), "dtype": str(v.dtype)} for k, v in data.items()]
    )
)