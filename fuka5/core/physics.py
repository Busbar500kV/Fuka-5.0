from __future__ import annotations
import numpy as np
from typing import Dict

# ---------------------------------------------------------------------
# Simplified phasor solver for edge energy transfer
# ---------------------------------------------------------------------

def solve_phasors(world: Dict[str, np.ndarray], graph: Dict[str, np.ndarray], dt: float = 0.05) -> Dict[str, np.ndarray]:
    """
    Compute a minimal phasor update across graph edges.

    Returns dict with:
      energy_transfer [E]
      phase_shift [E]
      power_flow [E]
    """
    rho = world["rho"]
    eps = world["eps"]
    nodes_zyx = graph["nodes_zyx"]
    edges_src = graph["edges_src"]
    edges_dst = graph["edges_dst"]
    edges_w = graph["edges_w"]

    # Gather field values
    rho_src = _gather_field(rho, nodes_zyx[edges_src])
    rho_dst = _gather_field(rho, nodes_zyx[edges_dst])
    eps_src = _gather_field(eps, nodes_zyx[edges_src])
    eps_dst = _gather_field(eps, nodes_zyx[edges_dst])

    # Phase difference proportional to local rho difference
    dphi = (rho_dst - rho_src) * np.pi
    phase_shift = np.sin(dphi).astype(np.float32)

    # Energy exchange proportional to weight * eps coupling
    energy_transfer = edges_w * (eps_src + eps_dst) * phase_shift
    power_flow = energy_transfer * dt

    return dict(
        energy_transfer=energy_transfer.astype(np.float32),
        phase_shift=phase_shift.astype(np.float32),
        power_flow=power_flow.astype(np.float32),
    )


# ---------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------

def _gather_field(arr: np.ndarray, zyx_idx: np.ndarray) -> np.ndarray:
    z = np.clip(zyx_idx[:, 0], 0, arr.shape[0] - 1)
    y = np.clip(zyx_idx[:, 1], 0, arr.shape[1] - 1)
    x = np.clip(zyx_idx[:, 2], 0, arr.shape[2] - 1)
    return arr[z, y, x].astype(np.float32)