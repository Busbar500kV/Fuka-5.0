from __future__ import annotations
import numpy as np
from typing import Any, Dict, List


# ---------------------------------------------------------------------
# Source generation utilities
# ---------------------------------------------------------------------

def make_sources(cfg: Dict[str, Any], world: Dict[str, np.ndarray]) -> List[Dict[str, Any]]:
    """
    Create a list of source descriptors distributed through the world volume.

    Each source is a dict containing:
      id, position (x,y,z), amplitude, frequency
    """
    n_sources = int(cfg.get("num_sources", 12))
    shape = world["rho"].shape

    rng = np.random.default_rng(int(cfg.get("seed", 1234)))
    positions = rng.uniform(low=0.1, high=0.9, size=(n_sources, 3))
    positions *= np.array(shape)[None, :]

    freqs = np.linspace(1.0, 3.0, n_sources) + rng.normal(0, 0.05, n_sources)
    amps = np.clip(rng.normal(1.0, 0.25, n_sources), 0.3, 2.0)

    sources: List[Dict[str, Any]] = []
    for i in range(n_sources):
        s = {
            "id": i,
            "pos": positions[i].astype(np.float32),
            "amplitude": float(amps[i]),
            "frequency": float(freqs[i]),
        }
        sources.append(s)

    return sources


# ---------------------------------------------------------------------
# Field injection (used during epoch updates)
# ---------------------------------------------------------------------

def inject_sources(world: Dict[str, np.ndarray], sources: List[Dict[str, Any]], t: float) -> np.ndarray:
    """
    Produce a perturbation field shaped by the sources for time t.
    Returns an array of same shape as world["rho"].
    """
    shape = world["rho"].shape
    field = np.zeros(shape, dtype=np.float32)
    zz, yy, xx = np.meshgrid(
        np.arange(shape[0]),
        np.arange(shape[1]),
        np.arange(shape[2]),
        indexing="ij",
    )

    for s in sources:
        px, py, pz = s["pos"]
        dist2 = (xx - px) ** 2 + (yy - py) ** 2 + (zz - pz) ** 2
        w = np.exp(-dist2 / (2 * (shape[0] * 0.05) ** 2))
        osc = s["amplitude"] * np.sin(2 * np.pi * s["frequency"] * t)
        field += (w * osc).astype(np.float32)

    return field