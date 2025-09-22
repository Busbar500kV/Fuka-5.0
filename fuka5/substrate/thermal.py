"""
fuka5.substrate.thermal
-----------------------
Node-local thermal dynamics and consolidation/leak modulation.

Equations (discrete-time per epoch window):
  Θ_i ΔT_i = ( Σ_{e→i} P^+_e ) - Λ_i (T_i - T_amb)
  χ_i(T_i) = exp( -E_b / (k * (T_i + ε)) )         # consolidation gain in (0,1]
  G_i(T_i) = G_i0 * (1 + γ_T * max(0, T_i - T_safe))  # over-temp leak multiplier

This module stores T for each node and provides helpers to compute χ and leak multipliers.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Iterable
import numpy as np


@dataclass
class ThermalParams:
    Theta: float = 1.0     # heat capacity
    Lambda: float = 0.15   # cooling rate
    T_amb: float = 0.0
    T_safe: float = 1.5
    gamma_T: float = 0.6
    E_b: float = 1.0       # barrier for consolidation curve
    k_B: float = 1.0       # Boltzmann-like constant

    @staticmethod
    def from_dict(d: Dict) -> "ThermalParams":
        return ThermalParams(
            Theta=float(d.get("Theta", 1.0)),
            Lambda=float(d.get("Lambda", 0.15)),
            T_amb=float(d.get("T_amb", 0.0)),
            T_safe=float(d.get("T_safe", 1.5)),
            gamma_T=float(d.get("gamma_T", 0.6)),
            E_b=float(d.get("E_b", 1.0)),
            k_B=float(d.get("k_B", 1.0)),
        )


class ThermalField:
    """
    Maintains node temperatures and exposes:
      - step(): integrate one epoch using incident Pplus per node
      - chi(T): consolidation gain per node
      - leak_multiplier(T): factor >= 1 when T > T_safe
    """

    def __init__(self, N_nodes: int, params: ThermalParams):
        self.N = int(N_nodes)
        self.params = params
        self.T = np.zeros(self.N, dtype=np.float32)  # initialize at ambient (0)

    def step(self, Pplus_incident: np.ndarray) -> None:
        """
        Integrate temperature for one epoch.
        Pplus_incident: shape (N,) nonnegative usable-power influx at node
        """
        assert Pplus_incident.shape == (self.N,)
        p = self.params
        dT = (Pplus_incident - p.Lambda * (self.T - p.T_amb)) / max(1e-9, p.Theta)
        self.T = (self.T + dT).astype(np.float32, copy=False)

    def chi(self) -> np.ndarray:
        """Consolidation gain χ_i(T_i) in (0,1], np.array shape (N,)."""
        p = self.params
        T_eff = self.T + 1e-3
        chi = np.exp(-p.E_b / (p.k_B * T_eff))
        return np.clip(chi, 0.0, 1.0).astype(np.float32, copy=False)

    def leak_multiplier(self) -> np.ndarray:
        """
        Return array of size (N,) with ≥1 when T > T_safe, else 1.
        Used to scale baseline node leaks.
        """
        p = self.params
        over = np.maximum(0.0, self.T - p.T_safe)
        mult = 1.0 + p.gamma_T * over
        mult[mult < 1.0] = 1.0
        return mult.astype(np.float32, copy=False)