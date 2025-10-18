from __future__ import annotations
import numpy as np
from typing import Any, Dict, List

from fuka5.core.physics import solve_phasors
from fuka5.core.world import evolve_world
from fuka5.core.sources import inject_sources


# ---------------------------------------------------------------------
# One-epoch update orchestrator
# ---------------------------------------------------------------------

def run_epoch(
    cfg: Dict[str, Any],
    world: Dict[str, np.ndarray],
    graph: Dict[str, np.ndarray],
    sources: List[Dict[str, Any]],
    *,
    epoch: int,
    t: float,
) -> Dict[str, Any]:
    """
    Execute one epoch of the simulation.
    Returns a dict with:
      updated world,
      edge_rows (list of dicts),
      metrics_row (dict)
    """
    dt = float(cfg.get("dt", 0.05))

    # 1. Apply source perturbations
    src_field = inject_sources(world, sources, t)
    world["rho"] = np.clip(world["rho"] + 0.02 * src_field, 0.0, 1.0)

    # 2. Compute phasor energy transfers on edges
    phasor = solve_phasors(world, graph, dt=dt)

    # 3. Aggregate some metrics
    total_energy = float(np.sum(np.abs(phasor["energy_transfer"])))
    avg_phase = float(np.mean(np.abs(phasor["phase_shift"])))

    # 4. Update the world for the next epoch
    world = evolve_world(world, dt=dt)

    # 5. Build edge records for writing
    edge_rows = [
        {
            "epoch": epoch,
            "src": int(s),
            "dst": int(d),
            "dist": float(dist),
            "weight": float(w),
            "energy": float(e),
            "phase": float(p),
        }
        for s, d, dist, w, e, p in zip(
            graph["edges_src"],
            graph["edges_dst"],
            graph["edges_dist"],
            graph["edges_w"],
            phasor["energy_transfer"],
            phasor["phase_shift"],
        )
    ]

    metrics_row = {
        "epoch": epoch,
        "time": t,
        "total_energy": total_energy,
        "avg_phase": avg_phase,
        "nodes": len(graph["nodes_zyx"]),
        "edges": len(graph["edges_src"]),
    }

    return {
        "world": world,
        "edge_rows": edge_rows,
        "metrics_row": metrics_row,
    }