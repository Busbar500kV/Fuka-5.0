"""
fuka5.analytics.metrics
-----------------------
Convenience metrics computed from shard rows or in-memory states.
This module keeps only lightweight, post-hoc helpers; the core online metrics
are already produced in substrate.updates.step_epoch.

Provided functions:
- summarize_edges_edgesnap(rows) -> dict (means/percentiles of C, B, gates, powers)
- summarize_metrics_traces(df) -> dict (final values, min/max, trends)
- topk_edges_by(df_edges, key, k) -> DataFrame
"""

from __future__ import annotations
from typing import Dict, Any, List
import numpy as np
import pandas as pd


def _percentiles(arr: np.ndarray, ps=(5, 25, 50, 75, 95)) -> Dict[str, float]:
    arr = np.asarray(arr)
    if arr.size == 0:
        return {f"p{p}": float("nan") for p in ps}
    vals = np.percentile(arr, ps)
    return {f"p{p}": float(v) for p, v in zip(ps, vals)}


def summarize_edges_edgesnap(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Aggregate a single epoch's edge rows (already dicts) into quick stats.
    """
    if not rows:
        return {}
    df = pd.DataFrame(rows)
    out: Dict[str, Any] = {}

    for col in ["C", "B", "A", "E", "P_low_s1", "P_low_s1p", "P_high_s1", "P_high_s1p"]:
        if col in df:
            arr = df[col].to_numpy()
            out[f"mean_{col}"] = float(np.mean(arr))
            out.update({f"{col}_{k}": v for k, v in _percentiles(arr).items()})

    # Simple gate summaries per band (if present)
    for band in ("low", "high"):
        cols = [f"g_{band}_mix", f"g_{band}_1", f"g_{band}_1p"]
        present = [c for c in cols if c in df]
        if present:
            arr = df[present].to_numpy()
            out[f"mean_g_{band}_mix"] = float(np.mean(df[f"g_{band}_mix"])) if f"g_{band}_mix" in df else float("nan")
            out[f"mean_g_{band}_sep"] = float(np.mean(np.abs(df.get(f"g_{band}_1", 0) - df.get(f"g_{band}_1p", 0)))) \
                                        if f"g_{band}_1" in df and f"g_{band}_1p" in df else float("nan")

    return out


def summarize_metrics_traces(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Summarize a metrics parquet DataFrame (multiple epochs).
    """
    out: Dict[str, Any] = {}
    if df.empty:
        return out
    last = df.iloc[-1].to_dict()
    for k in ["R1", "R1p", "Rmix", "MSE_s1", "MSE_s1p", "MSE_mix", "sumC", "sumB", "avgT",
              "sep_low", "mix_low", "sep_high", "mix_high", "d_resp_5k"]:
        if k in df:
            out[f"final_{k}"] = float(df[k].iloc[-1])
            out[f"min_{k}"] = float(df[k].min())
            out[f"max_{k}"] = float(df[k].max())
    out["epochs"] = int(df["epoch"].iloc[-1] + 1) if "epoch" in df and not df.empty else 0
    return out


def topk_edges_by(df_edges: pd.DataFrame, key: str, k: int = 20) -> pd.DataFrame:
    """
    Return top-k edges by a numeric key from a single-epoch edges DataFrame.
    """
    if key not in df_edges.columns:
        raise KeyError(f"Key '{key}' not found in edges dataframe")
    df = df_edges.sort_values(key, ascending=False).head(k).copy()
    return df