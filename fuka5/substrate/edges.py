"""
fuka5.substrate.edges
---------------------
Edge state container and local feature cache.

Each undirected edge e=(u,v) keeps:
  - Geometry: u,v, distance d
  - Structure: C (capacitance), bounds Cmin..Cmax
  - Battery B (state-of-charge), Maturity A
  - Gates per band: mix, s1, s1p   (stored as dict: band -> np.array([mix, s1, s1p]))
  - Local measures cache (per epoch window): E, band powers by source, etc.

This module is intentionally lightweight; learning dynamics live in updates.py,
battery flows in battery.py, and bandwise gate rules in gates.py.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, Any
import numpy as np


@dataclass
class EdgeState:
    # topology
    eid: int
    u: int
    v: int
    d: float

    # structure
    C: float
    Cmin: float
    Cmax: float

    # energy bookkeeping
    B: float = 0.0     # battery
    A: float = 0.0     # maturity

    # gates[band] = np.array([g_mix, g_s1, g_s1p])
    gates: Dict[str, np.ndarray] = field(default_factory=dict)

    # cached measures for the last window (filled by updates.py / physics)
    E: float = 1e-12   # local energy
    P_low_s1: float = 0.0
    P_low_s1p: float = 0.0
    P_high_s1: float = 0.0
    P_high_s1p: float = 0.0
    T_u: float = 0.0   # node temperatures at endpoints (for logging)
    T_v: float = 0.0

    def ensure_band(self, band: str) -> None:
        if band not in self.gates:
            self.gates[band] = np.array([1/3, 1/3, 1/3], dtype=np.float32)

    def get_gate(self, band: str) -> np.ndarray:
        self.ensure_band(band)
        return self.gates[band]

    def set_gate(self, band: str, vec: np.ndarray) -> None:
        assert vec.shape == (3,)
        v = np.clip(vec.astype(np.float32), 0.0, 1.0)
        s = float(v.sum())
        if s > 1.0:
            v = v / s  # soft capacity
        self.gates[band] = v

    def clamp_C(self) -> None:
        self.C = float(np.clip(self.C, self.Cmin, self.Cmax))

    # ---------- Logging helpers ----------

    def band_power_dict(self) -> Dict[str, float]:
        return {
            "P_low_s1": self.P_low_s1,
            "P_low_s1p": self.P_low_s1p,
            "P_high_s1": self.P_high_s1,
            "P_high_s1p": self.P_high_s1p,
        }

    def gates_dict(self, bands: Dict[str, Any]) -> Dict[str, float]:
        out = {}
        for b in bands.keys():
            g = self.get_gate(b)
            out[f"g_{b}_mix"] = float(g[0])
            out[f"g_{b}_1"]   = float(g[1])
            out[f"g_{b}_1p"]  = float(g[2])
        return out


def initialize_edge_states(
    graph,
    Cmin: float,
    Cmax: float,
    rng: np.random.Generator | None = None,
) -> Dict[int, EdgeState]:
    """
    Create an EdgeState per graph edge, seeding C around the graph prior.
    """
    rng = rng or np.random.default_rng(0)
    states: Dict[int, EdgeState] = {}
    for e in graph.edges:
        # start around prior with small jitter, clipped to bounds
        C0 = float(np.clip(e.C_prior * (0.8 + 0.4 * rng.random()), Cmin, Cmax))
        states[e.id] = EdgeState(
            eid=e.id, u=e.u, v=e.v, d=e.d,
            C=C0, Cmin=Cmin, Cmax=Cmax,
            B=0.0, A=0.0, gates={}
        )
    return states