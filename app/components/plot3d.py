"""
Plotly 3D helpers for Fuka 5.0 Streamlit UI.

make_volume_fig(volumes, iso_level) -> go.Figure
add_edges_to_fig(fig, edges_df, color_key) -> go.Figure
"""

from __future__ import annotations
from typing import Dict
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def make_volume_fig(volumes: Dict[str, np.ndarray], iso_level: float = 0.4) -> go.Figure:
    """
    Render 3D rho isosurface with optional masks.
    volumes keys: "rho", "outer_mask", "core_mask".
    """
    rho = volumes.get("rho")
    outer = volumes.get("outer_mask")
    core = volumes.get("core_mask")

    fig = go.Figure()
    if rho is None:
        return fig

    # plotly expects z,y,x indexing; transpose accordingly
    rho_plot = rho.transpose(2, 1, 0)

    # Use volume rendering with isosurface-like opacity bands
    fig.add_trace(go.Volume(
        value=rho_plot,
        opacity=0.08,
        surface_count=12,
        showscale=True,
        coloraxis=None,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name="rho"
    ))

    # Optionally overlay outer/core masks as semi-transparent cubes of points
    def _mask_points(mask: np.ndarray, step: int = 3):
        idx = np.argwhere(mask)
        if idx.size == 0:
            return None
        # thin points for perf
        idx = idx[::step]
        # swap xyz -> z,y,x for plotly
        return idx[:, [2,1,0]]

    if outer is not None:
        pts = _mask_points(outer)
        if pts is not None:
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=1),
                name="outer"
            ))

    if core is not None:
        pts = _mask_points(core, step=2)
        if pts is not None:
            fig.add_trace(go.Scatter3d(
                x=pts[:, 0], y=pts[:, 1], z=pts[:, 2],
                mode="markers",
                marker=dict(size=2),
                name="core"
            ))

    fig.update_layout(
        scene=dict(aspectmode="data"),
        margin=dict(l=0, r=0, t=0, b=0),
        showlegend=True,
    )
    return fig


def add_edges_to_fig(fig: go.Figure, edges_df: pd.DataFrame, color_key: str = "C") -> go.Figure:
    """
    Overlay edges as 3D line segments; color intensity by color_key if present.
    Expects columns x_u,y_u,z_u, x_v,y_v,z_v (world indexing).
    """
    if edges_df.empty:
        return fig

    # Normalize color values to [0,1] for alpha/width scaling
    vals = edges_df[color_key].to_numpy() if color_key in edges_df else np.ones(len(edges_df))
    vmin, vmax = float(np.min(vals)), float(np.max(vals))
    span = vmax - vmin if vmax > vmin else 1.0
    norm = (vals - vmin) / span

    # Build one Scatter3d per, keeping reasonable cap for performance
    max_lines = min(300, len(edges_df))
    for i in range(max_lines):
        r = edges_df.iloc[i]
        lw = 1.0 + 3.0 * norm[i]
        fig.add_trace(go.Scatter3d(
            x=[r["z_u"], r["z_v"]],
            y=[r["y_u"], r["y_v"]],
            z=[r["x_u"], r["x_v"]],
            mode="lines",
            line=dict(width=lw),
            name=str(r["edge_id"]),
            hovertext=f'{r["edge_id"]} {color_key}={r.get(color_key,0):.3g}',
            hoverinfo="text",
            showlegend=False
        ))
    return fig