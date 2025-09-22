"""
fuka5.substrate.gates
---------------------
Bandwise attention gates per edge with purely local updates.

Each edge keeps, for every band `b`, a 3-vector of nonnegative entries that
softly compete under a unit-capacity constraint:

    g_b = [g_mix, g_s1, g_s1p],   with sum(g_b) <= 1.

Update rules (local, per edge & band):
- Split gates (s1, s1p) increase with their own captured power fraction
  and are discouraged from co-occupation by a competition term.
- Mix gate increases with "synergy" and decreases with "interference".

Inputs (provided by updates.py each epoch window):
- p1:  scalar   (captured power at band b attributed to source 1 on this edge)
- p1p: scalar   (captured power at band b attributed to source 1' on this edge)
- S:   scalar   synergy proxy in [0,1]   (e.g., (p1+p1p)/E clipped)
- I:   scalar   interference proxy in [0,1] (e.g., |sum M_1 * sum M_1p^*|/E clipped)
- params: {eta_g, lambda_g, mu_I, mu_C}
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np

from .edges import EdgeState


@dataclass
class GateParams:
    eta_g: float = 0.8      # learning step
    lambda_g: float = 0.03  # small decay to avoid saturation
    mu_I: float = 0.8       # interference penalty weight (for mix)
    mu_C: float = 0.5       # competition penalty between s1 and s1p

    @staticmethod
    def from_dict(d: Dict) -> "GateParams":
        return GateParams(
            eta_g=float(d.get("eta_g", 0.8)),
            lambda_g=float(d.get("lambda_g", 0.03)),
            mu_I=float(d.get("mu_I", 0.8)),
            mu_C=float(d.get("mu_C", 0.5)),
        )


def _safe_norm(x: float, denom: float) -> float:
    return float(np.clip(x / (denom + 1e-12), 0.0, 1.0))


def update_band_gate(
    gvec: np.ndarray,
    p1: float,
    p1p: float,
    S: float,
    I: float,
    params: GateParams,
) -> np.ndarray:
    """
    One-band gate update. Returns the new (clipped, renormalized) 3-vector.
    """
    g = gvec.astype(np.float32, copy=True)  # [mix, s1, s1p]
    g_mix, g_s1, g_s1p = float(g[0]), float(g[1]), float(g[2])

    # Normalized power cues (bounded to [0,1])
    total = max(1e-12, p1 + p1p)
    q1 = _safe_norm(p1, total)
    q1p = _safe_norm(p1p, total)

    # Gradient-like updates
    dg_mix = + (S - params.mu_I * I) - params.lambda_g * g_mix
    dg_s1  = + (q1) - params.mu_C * (g_s1 * g_s1p) - params.lambda_g * g_s1
    dg_s1p = + (q1p) - params.mu_C * (g_s1 * g_s1p) - params.lambda_g * g_s1p

    g_new = np.array([
        g_mix + params.eta_g * dg_mix,
        g_s1  + params.eta_g * dg_s1,
        g_s1p + params.eta_g * dg_s1p,
    ], dtype=np.float32)

    # Clamp to [0,1] and enforce soft capacity sum<=1 via renorm if needed
    g_new = np.clip(g_new, 0.0, 1.0)
    s = float(g_new.sum())
    if s > 1.0:
        g_new /= s
    return g_new


def update_edge_gates_for_all_bands(
    edge: EdgeState,
    bands: Dict[str, list],
    per_band_inputs: Dict[str, Dict[str, float]],
    params: GateParams,
) -> None:
    """
    Update gates for every band of a single edge.

    per_band_inputs[band] must include:
      - "p1":  power attributed to source 1 at this band
      - "p1p": power attributed to source 1' at this band
      - "S":   synergy proxy in [0,1]
      - "I":   interference proxy in [0,1]
    """
    for b in bands.keys():
        # Ensure gate vector exists
        gvec = edge.get_gate(b)
        # Fetch local measurements
        d = per_band_inputs.get(b, {})
        p1  = float(d.get("p1", 0.0))
        p1p = float(d.get("p1p", 0.0))
        S   = float(np.clip(d.get("S", 0.0), 0.0, 1.0))
        I   = float(np.clip(d.get("I", 0.0), 0.0, 1.0))
        # Update and set
        g_new = update_band_gate(gvec, p1, p1p, S, I, params)
        edge.set_gate(b, g_new)