"""
Streamlit panels for Fuka 5.0 UI:
- metrics_panel(metrics_df): time-series & small stats
- filters_panel(): returns dict of edge overlay filters
"""

from __future__ import annotations
from typing import Dict, Any
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


def _line(fig: go.Figure, x, y, name: str):
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name))


def metrics_panel(metrics_df: pd.DataFrame):
    """Render time-series charts for core metrics."""
    if metrics_df is None or metrics_df.empty:
        st.info("No metrics yet.")
        return

    st.markdown("### Metrics")

    cols = st.columns(2)

    with cols[0]:
        fig = go.Figure()
        _line(fig, metrics_df["epoch"], metrics_df.get("sumC", 0), "sumC")
        _line(fig, metrics_df["epoch"], metrics_df.get("sumB", 0), "sumB")
        fig.update_layout(title="Capacity & Battery", xaxis_title="epoch", yaxis_title="value", height=260, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

        fig2 = go.Figure()
        _line(fig2, metrics_df["epoch"], metrics_df.get("avgT", 0), "avgT")
        fig2.update_layout(title="Average Temperature", xaxis_title="epoch", yaxis_title="T", height=260, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig2, use_container_width=True, config={"displayModeBar": False})

    with cols[1]:
        fig3 = go.Figure()
        if "MSE_s1" in metrics_df:  _line(fig3, metrics_df["epoch"], metrics_df["MSE_s1"], "MSE_s1")
        if "MSE_s1p" in metrics_df: _line(fig3, metrics_df["epoch"], metrics_df["MSE_s1p"], "MSE_s1p")
        if "MSE_mix" in metrics_df: _line(fig3, metrics_df["epoch"], metrics_df["MSE_mix"], "MSE_mix")
        fig3.update_layout(title="Decoder MSE", xaxis_title="epoch", yaxis_title="mse", height=260, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig3, use_container_width=True, config={"displayModeBar": False})

        fig4 = go.Figure()
        if "sep_low" in metrics_df:  _line(fig4, metrics_df["epoch"], metrics_df["sep_low"], "sep_low")
        if "mix_low" in metrics_df:  _line(fig4, metrics_df["epoch"], metrics_df["mix_low"], "mix_low")
        if "sep_high" in metrics_df: _line(fig4, metrics_df["epoch"], metrics_df["sep_high"], "sep_high")
        if "mix_high" in metrics_df: _line(fig4, metrics_df["epoch"], metrics_df["mix_high"], "mix_high")
        fig4.update_layout(title="Gating Indices", xaxis_title="epoch", yaxis_title="value", height=260, margin=dict(l=10,r=10,t=30,b=10))
        st.plotly_chart(fig4, use_container_width=True, config={"displayModeBar": False})

    # quick stats
    with st.expander("Summary stats", expanded=False):
        last = metrics_df.iloc[-1]
        stats = {
            "epochs": int(last["epoch"] + 1),
            "sumC_final": float(last.get("sumC", float("nan"))),
            "sumB_final": float(last.get("sumB", float("nan"))),
            "avgT_final": float(last.get("avgT", float("nan"))),
            "MSE_s1_final": float(last.get("MSE_s1", float("nan"))),
            "MSE_s1p_final": float(last.get("MSE_s1p", float("nan"))),
            "MSE_mix_final": float(last.get("MSE_mix", float("nan"))),
            "sep_low_final": float(last.get("sep_low", float("nan"))),
            "sep_high_final": float(last.get("sep_high", float("nan"))),
            "mix_low_final": float(last.get("mix_low", float("nan"))),
            "mix_high_final": float(last.get("mix_high", float("nan")))
        }
        st.json(stats)


def filters_panel() -> Dict[str, Any]:
    """Return filter selections for edge overlays."""
    c1, c2, c3 = st.columns(3)
    with c1:
        band = st.selectbox("Band", ["low", "high"], index=0)
    with c2:
        gate_mode = st.selectbox("Gate mode", ["mix", "1", "1p"], index=0)
    with c3:
        color_key = st.selectbox("Edge color", ["C", "B", "A", "E"], index=0)

    c4, c5 = st.columns(2)
    with c4:
        min_C_pct = st.slider("Min C percentile", 0.0, 100.0, 20.0, 1.0)
    with c5:
        min_B_pct = st.slider("Min B percentile", 0.0, 100.0, 20.0, 1.0)

    return {
        "band": band,
        "gate_mode": gate_mode,
        "min_C_pct": min_C_pct,
        "min_B_pct": min_B_pct,
        "color_key": color_key,
    }