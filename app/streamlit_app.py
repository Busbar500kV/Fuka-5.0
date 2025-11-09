from __future__ import annotations

import os
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------
# Local runs root (your current layout)
# ---------------------------------------------------------
RUNS_BASE = Path("/home/busbar/fuka-runs/runs")
assert RUNS_BASE.exists(), f"{RUNS_BASE} does not exist!"

st.set_page_config(page_title="Fuka 5.0", layout="wide")


# ---------------------------------------------------------
# Helpers: runs / epochs / NPZ
# ---------------------------------------------------------
def list_runs() -> List[str]:
    return sorted([p.name for p in RUNS_BASE.iterdir() if p.is_dir()])


def list_epochs(run_id: str) -> Tuple[List[int], Path]:
    vdir = RUNS_BASE / run_id / "volumes"
    eps: List[int] = []
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


# ---------------------------------------------------------
# Sidebar: selection + epoch nav
# ---------------------------------------------------------
st.sidebar.header("Select run / epoch (local)")

runs = list_runs()
if not runs:
    st.sidebar.error(f"No runs found in {RUNS_BASE}.")
    st.stop()

run_id = st.sidebar.selectbox("Run", runs, index=len(runs) - 1)
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

epoch = st.sidebar.selectbox(
    "Epoch", epochs, index=st.session_state.epoch_idx, key="epoch_select"
)
# keep idx in sync if user chooses via dropdown
try:
    st.session_state.epoch_idx = epochs.index(epoch)
except Exception:
    pass

iso_pct = st.sidebar.slider("Isosurface percentile (rho)", 50, 99, 75)
opacity = st.sidebar.slider("Isosurface opacity", 1, 100, 25) / 100.0

# ---------------------------------------------------------
# Main: load arrays
# ---------------------------------------------------------
st.title("Fuka 5.0 — Space–Time Capacitor Substrate")

data, path = load_npz(run_id, epoch)
rho = np.array(data["rho"])  # (nx, ny, nz)
st.success(f"Loaded {path}")

nx, ny, nz = rho.shape

# ---------------------------------------------------------
# Isosurface figure (rho)
# ---------------------------------------------------------
x = np.arange(nx)
y = np.arange(ny)
z = np.arange(nz)

iso = np.nanpercentile(rho, iso_pct)
fig = go.Figure(
    go.Isosurface(
        x=np.repeat(x, ny * nz),
        y=np.tile(np.repeat(y, nz), nx),
        z=np.tile(z, nx * ny),
        value=rho.ravel(order="F"),
        isomin=iso,
        isomax=rho.max(),
        caps=dict(x_show=False, y_show=False, z_show=False),
        opacity=opacity,
        showscale=False,
    )
)
fig.update_layout(
    scene=dict(aspectmode="data"),
    margin=dict(l=0, r=0, t=0, b=0),
    height=760,
)

# ---------------------------------------------------------
# Nodes overlay (core / outer masks, or rho fallback)
# ---------------------------------------------------------
with st.sidebar.expander("Nodes overlay", expanded=False):
    show_core = st.checkbox("Show core nodes", value=True, key="nodes_show_core")
    show_outer = st.checkbox("Show outer nodes", value=True, key="nodes_show_outer")
    node_stride = st.slider("Node stride (display sampling)", 1, 12, 3, key="nodes_stride")
    node_size = st.slider("Marker size", 1, 10, 4, key="nodes_size")
    fallback_rho_points = st.checkbox(
        "Fallback: show rho points if masks missing", value=True, key="nodes_fallback_rho"
    )
    rho_quantile = st.slider("rho threshold quantile", 0.90, 0.999, 0.97, key="nodes_rho_quantile")

core = data.get("core_mask", None)
outer = data.get("outer_mask", None)


def _mask_points(mask: np.ndarray, stride: int):
    zz, yy, xx = np.where(mask)
    if zz.size == 0:
        return np.array([]), np.array([]), np.array([])
    sel = (np.arange(zz.size) % max(1, int(stride))) == 0
    return xx[sel], yy[sel], zz[sel]


try:
    if show_core and core is not None:
        _x, _y, _z = _mask_points(core, node_stride)
        if _x.size:
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
        if _x.size:
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
    if fallback_rho_points and (core is None and outer is None):
        thr = float(np.quantile(rho[np.isfinite(rho)], float(rho_quantile)))
        zz, yy, xx = np.where(rho >= thr)
        if zz.size:
            sel = (np.arange(zz.size) % max(1, int(node_stride))) == 0
            xx, yy, zz = xx[sel], yy[sel], zz[sel]
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

# ---------------------------------------------------------
# Edges overlay (from shards/*.parquet)
# ---------------------------------------------------------
with st.sidebar.expander("Edges overlay", expanded=False):
    show_edges = st.checkbox("Show edges", value=True)
    color_key = st.selectbox("Color by", ["dist", "weight", "energy", "phase"], index=0)
    max_edges = st.slider("Max edges", 1000, 20000, 5000, step=1000)
    edge_opacity = st.slider("Edge opacity (%)", 10, 100, 95) / 100


@st.cache_data(show_spinner=True, ttl=30)
def _load_edges_epoch(run: str, ep: int, limit: int) -> pd.DataFrame:
    import glob

    shards_dir = RUNS_BASE / run / "shards"
    if not shards_dir.exists():
        return pd.DataFrame()

    dfs: List[pd.DataFrame] = []
    for f in glob.glob(str(shards_dir / "edges-*.parquet")):
        try:
            df = pd.read_parquet(f, engine="pyarrow")
        except Exception:
            continue
        dfe = df[df["epoch"] == int(ep)]
        if not dfe.empty:
            dfs.append(dfe)

    if not dfs:
        return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    if limit is not None:
        out = out.head(int(limit))
    return out


def _unravel_adaptive(idx: np.ndarray, shape: Tuple[int, int, int]) -> Tuple[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Try several plausible index→(x,y,z) mappings and pick the one
    that yields the widest spatial spread (variance-based score).
    """
    nx, ny, nz = shape

    def unravel_c(i):
        x = i % nx
        y = (i // nx) % ny
        z = i // (nx * ny)
        return x.astype(float), y.astype(float), z.astype(float)

    def unravel_f(i):
        z = i % nz
        y = (i // nz) % ny
        x = i // (nz * ny)
        return x.astype(float), y.astype(float), z.astype(float)

    def score(x, y, z):
        vx = max(np.var(x), 1e-9)
        vy = max(np.var(y), 1e-9)
        vz = max(np.var(z), 1e-9)

        def stuck_ratio(a, hi):
            return (np.mean(a == 0) + np.mean(a == hi)) / 2.0

        pen = 1.0 - 0.5 * (
            stuck_ratio(x, nx - 1) + stuck_ratio(y, ny - 1) + stuck_ratio(z, nz - 1)
        )
        return (vx * vy * vz) * max(pen, 1e-6)

    c_xyz = unravel_c(idx)
    f_xyz = unravel_f(idx)

    candidates = [
        ("C(x,y,z)", c_xyz),
        ("C(y,x,z)", (c_xyz[1], c_xyz[0], c_xyz[2])),
        ("C(x,z,y)", (c_xyz[0], c_xyz[2], c_xyz[1])),
        ("C(z,y,x)", (c_xyz[2], c_xyz[1], c_xyz[0])),
        ("F(x,y,z)", f_xyz),
        ("F(y,x,z)", (f_xyz[1], f_xyz[0], f_xyz[2])),
        ("F(x,z,y)", (f_xyz[0], f_xyz[2], f_xyz[1])),
        ("F(z,y,x)", (f_xyz[2], f_xyz[1], f_xyz[0])),
    ]

    best_name, best_xyz, best_score = None, None, -1.0
    for name, (x, y, z) in candidates:
        s = score(x, y, z)
        if s > best_score:
            best_name, best_xyz, best_score = name, (x, y, z), s
    return best_name, best_xyz


if show_edges:
    edges = _load_edges_epoch(run_id, epoch, max_edges)
    if edges.empty:
        st.info("No edges for this epoch.")
    else:
        missing = [c for c in ["src", "dst"] if c not in edges.columns]
        if missing:
            st.warning(f"Edges present for epoch {epoch}, but missing columns: {missing}")
        else:
            # Map src/dst → (x,y,z) adaptively
            src = edges["src"].to_numpy(np.int64)
            dst = edges["dst"].to_numpy(np.int64)

            name_u, (xu, yu, zu) = _unravel_adaptive(src, rho.shape)
            name_v, (xv, yv, zv) = _unravel_adaptive(dst, rho.shape)

            # Build line segments with NaN separators
            X = np.vstack([xu, xv, np.full_like(xu, np.nan)]).T.reshape(-1)
            Y = np.vstack([yu, yv, np.full_like(yu, np.nan)]).T.reshape(-1)
            Z = np.vstack([zu, zv, np.full_like(zu, np.nan)]).T.reshape(-1)

            # Color selection and normalization
            if color_key not in edges.columns:
                edges[color_key] = 0.0
            C = edges[color_key].to_numpy(dtype=float)
            cmin, cmax = np.percentile(C, 5), np.percentile(C, 95)
            if cmax <= cmin:
                cmax = cmin + 1e-9
            Cn = (np.clip(C, cmin, cmax) - cmin) / (cmax - cmax + 1e-12 if cmax == cmin else (cmax - cmin))
            Cline = np.repeat(Cn, 3)
            Cline[2::3] = np.nan  # keep gaps color-agnostic

            fig.add_trace(
                go.Scatter3d(
                    x=X,
                    y=Y,
                    z=Z,
                    mode="lines",
                    line=dict(width=2, color=Cline, colorscale="Viridis"),
                    opacity=edge_opacity,
                    name=f"Edges ({len(edges)})",
                )
            )

            with st.expander("Edge index decode (debug)", expanded=False):
                st.write(f"src mapping: **{name_u}**, dst mapping: **{name_v}**")

# ---------------------------------------------------------
# Final render
# ---------------------------------------------------------
st.plotly_chart(fig, use_container_width=True)

# Small table of arrays in NPZ
st.subheader("NPZ contents")
st.dataframe(
    pd.DataFrame(
        [{"key": k, "shape": list(v.shape), "dtype": str(v.dtype)} for k, v in data.items()]
    )
)