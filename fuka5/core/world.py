from __future__ import annotations
import numpy as np
from typing import Any, Dict

# ---------------------------------------------------------------------
# World initialization and field generation
# ---------------------------------------------------------------------

def make_world(cfg: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """
    Build the initial world fields (rho, eps, masks).

    Returns a dict of arrays keyed as:
      rho, eps, outer_mask, core_mask
    """
    shape = tuple(cfg.get("grid_shape", (64, 64, 64)))
    seed = int(cfg.get("seed", 1234))
    np.random.seed(seed)

    # Density field (rho) – start with a smoothed random field
    rho = np.random.rand(*shape).astype(np.float32)
    rho = _gaussian_blur(rho, sigma=cfg.get("rho_blur", 2.0))
    rho /= rho.max() + 1e-8

    # Permittivity-like field (eps)
    eps = 0.5 + 0.5 * np.sin(4.0 * np.pi * rho)
    eps = eps.astype(np.float32)

    # Outer and core masks
    outer_mask = rho > np.percentile(rho, 85)
    core_mask = rho < np.percentile(rho, 15)

    return dict(rho=rho, eps=eps, outer_mask=outer_mask, core_mask=core_mask)


# ---------------------------------------------------------------------
# Simple blur utility (NumPy-only)
# ---------------------------------------------------------------------

def _gaussian_blur(arr: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Apply separable Gaussian blur using FFT convolution."""
    from scipy.ndimage import gaussian_filter
    return gaussian_filter(arr, sigma=sigma, mode="wrap").astype(np.float32)


# ---------------------------------------------------------------------
# World update (one epoch)
# ---------------------------------------------------------------------

def evolve_world(world: Dict[str, np.ndarray], dt: float = 0.01) -> Dict[str, np.ndarray]:
    """
    Perform a minimal evolution step to modify rho and eps.
    The goal is to produce visible dynamic change for the UI.
    """
    rho = world["rho"]
    eps = world["eps"]

    lap = _laplacian(rho)
    drho = 0.1 * lap - 0.02 * (rho - 0.5)
    rho_next = np.clip(rho + drho * dt, 0.0, 1.0)

    eps_next = 0.5 + 0.5 * np.sin(4.0 * np.pi * rho_next)
    eps_next = eps_next.astype(np.float32)

    world.update(rho=rho_next, eps=eps_next)
    return world


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _laplacian(a: np.ndarray) -> np.ndarray:
    """Compute discrete Laplacian with periodic boundaries."""
    return (
        -6 * a
        + np.roll(a, 1, 0)
        + np.roll(a, -1, 0)
        + np.roll(a, 1, 1)
        + np.roll(a, -1, 1)
        + np.roll(a, 1, 2)
        + np.roll(a, -1, 2)
    ).astype(np.float32)