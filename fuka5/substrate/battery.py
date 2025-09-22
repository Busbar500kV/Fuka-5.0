"""
fuka5.substrate.battery
-----------------------
Local edge "heat-bath battery" dynamics, upkeep (maintenance), and self-rehearsal spec.

Implements:
- Battery update per edge:
    B <- B + eta_h * P_pos - k_leak * B - k_struct * (|ΔC| + λ_g |Δg|)
  where P_pos is a local usable-power proxy supplied by updates.py.

- Maturity ratchet:
    A <- max(0, A + (alpha_A if B > B_thr else -beta_A))

- Adaptive floors and pruning reduction:
    Cmin_eff = Cmin * (1 + mu_A * tanh(A))
    lambdaC_eff = lambdaC / (1 + nu_A * tanh(A))

- Energy-budgeted growth factor & upkeep:
    budget = min(1, B / B0)
    upkeep = rho_upkeep * (B/(B+B1)) * chi(T_node) * (C* - C)

- Self-rehearsal amplitude:
    a_reh = alpha_reh * (B/(1+B)) * (A/(1+A))

Only scalar arithmetic here; vector assembly done by updates.py.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict
import numpy as np

from .edges import EdgeState


@dataclass
class BatteryParams:
    eta_h: float = 1.0
    k_leak: float = 0.03
    k_struct: float = 2e-10
    B0: float = 1.0
    B1: float = 0.3
    rho_upkeep: float = 0.08
    alpha_reh: float = 1e-3

    @staticmethod
    def from_dict(d: Dict) -> "BatteryParams":
        return BatteryParams(
            eta_h=float(d.get("eta_h", 1.0)),
            k_leak=float(d.get("k_leak", 0.03)),
            k_struct=float(d.get("k_struct", 2e-10)),
            B0=float(d.get("B0", 1.0)),
            B1=float(d.get("B1", 0.3)),
            rho_upkeep=float(d.get("rho_upkeep", 0.08)),
            alpha_reh=float(d.get("alpha_reh", 1e-3)),
        )


@dataclass
class MaturityParams:
    alpha_A: float = 0.06
    beta_A: float = 0.02
    B_thr: float = 0.4
    mu_A: float = 0.6     # raises Cmin
    nu_A: float = 0.6     # reduces lambdaC

    @staticmethod
    def from_dict(d: Dict) -> "MaturityParams":
        return MaturityParams(
            alpha_A=float(d.get("alpha_A", 0.06)),
            beta_A=float(d.get("beta_A", 0.02)),
            B_thr=float(d.get("B_thr", 0.4)),
            mu_A=float(d.get("mu_A", 0.6)),
            nu_A=float(d.get("nu_A", 0.6)),
        )


def battery_update(
    edge: EdgeState,
    P_pos: float,          # usable power proxy at this edge/window
    dC_struct: float,      # magnitude of structural ΔC intent (proxy)
    dgate_struct: float,   # magnitude of gate change intent (proxy)
    params: BatteryParams,
) -> None:
    """Update the battery state B using pure local signals."""
    B = edge.B
    dB = (
        params.eta_h * float(P_pos)
        - params.k_leak * B
        - params.k_struct * (abs(dC_struct) + abs(dgate_struct))
    )
    edge.B = float(max(0.0, B + dB))


def maturity_update(edge: EdgeState, params: MaturityParams) -> None:
    """Update maturity A based on whether the edge remains above charge threshold."""
    if edge.B > params.B_thr:
        edge.A = float(max(0.0, edge.A + params.alpha_A))
    else:
        edge.A = float(max(0.0, edge.A - params.beta_A))


def effective_Cmin_lambdaC(
    edge: EdgeState, Cmin_base: float, lambdaC_base: float, params: MaturityParams
) -> tuple[float, float]:
    """Return maturity-adjusted Cmin and lambdaC."""
    A = edge.A
    Cmin_eff = float(Cmin_base * (1.0 + params.mu_A * np.tanh(A)))
    lambdaC_eff = float(lambdaC_base / (1.0 + params.nu_A * np.tanh(A)))
    return Cmin_eff, lambdaC_eff


def budget_gate(edge: EdgeState, params: BatteryParams) -> float:
    """Energy-budgeted growth factor in [0,1]."""
    return float(min(1.0, edge.B / max(1e-9, params.B0)))


def upkeep_term(
    edge: EdgeState,
    C_star: float,         # local "target" (e.g., EMA of C when productive)
    chi_T: float,          # consolidation gain from node temperature
    params: BatteryParams,
) -> float:
    """Battery-funded upkeep pull toward C_star."""
    B = edge.B
    return float(params.rho_upkeep * (B / (B + params.B1 + 1e-12)) * chi_T * (C_star - edge.C))


def rehearsal_amplitude(edge: EdgeState, params: BatteryParams) -> float:
    """Tiny self-rehearsal amplitude based on B and A."""
    B = edge.B
    A = edge.A
    return float(params.alpha_reh * (B / (1.0 + B)) * (A / (1.0 + A)))