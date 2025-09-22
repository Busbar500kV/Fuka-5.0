"""
fuka5.core.world
----------------
Space–time environment defined purely by charge density rho(x,t).
Provides:
  - WorldConfig dataclass (dims, dx, rho init params, material map, morphogenesis)
  - World class:
      * build initial rho
      * compute material fields: epsilon(rho), g(rho)
      * level-sets → outer/core boolean masks
      * morphogenesis step (local, graph-Laplacian-like on voxel grid)
      * downsampling utilities for UI volume export

Notes
-----
- Everything is numpy-only; no framework dependencies.
- All parameters come from JSON config (expanded via fuka5.load_json_with_env).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Dict, Any, Optional
import numpy as np


# ---------------------------
# Utilities / nonlinearities
# ---------------------------

def _softplus(x: np.ndarray, beta: float = 1.0) -> np.ndarray:
    # numerically stable softplus
    z = np.clip(beta * x, -50.0, 50.0)
    return np.log1p(np.exp(z)) / beta

def _nonlin(name: str):
    name = (name or "softplus").lower()
    if name == "softplus":
        return _softplus
    elif name in ("relu",):
        return lambda x: np.maximum(0.0, x)
    elif name in ("sigmoid", "logistic"):
        return lambda x: 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))
    else:
        # default safe fallback
        return _softplus


# ---------------------------
# Dataclasses
# ---------------------------

@dataclass
class SpaceSpec:
    dims: Tuple[int, int, int]  # (nx, ny, nz)
    dx: float                   # voxel size

@dataclass
class RhoSpec:
    init: str                   # "gaussian_shell" | "gaussian" | "uniform_core"
    theta_outer: float          # lower threshold for substrate
    theta_core: float           # higher threshold for inner core
    sigma: float                # width used by initializers
    center: Tuple[float, float, float]

@dataclass
class MaterialSpec:
    eps_min: float
    eps_gain: float
    g_min: float
    g_gain: float
    nonlin: str = "softplus"    # name → softplus default

@dataclass
class MorphogenesisSpec:
    D: float            # diffusion coeff on rho
    eta: float          # growth from local harvest proxy (passed in externally)
    gamma_mass: float   # penalty proportional to local capacity mass (passed in)
    lam: float          # natural decay of rho

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "MorphogenesisSpec":
        # allow "lambda" key in JSON
        lam = d.get("lambda", d.get("lam", 0.0))
        return MorphogenesisSpec(D=d.get("D", 0.0),
                                 eta=d.get("eta", 0.0),
                                 gamma_mass=d.get("gamma_mass", 0.0),
                                 lam=lam)

@dataclass
class WorldConfig:
    space: SpaceSpec
    rho: RhoSpec
    material: MaterialSpec
    morph: MorphogenesisSpec


# ---------------------------
# World implementation
# ---------------------------

class World:
    """
    Holds the charge-density field rho on a regular 3D grid, plus material maps:
      - eps = eps_min + eps_gain * σ(rho)
      - g   = g_min   + g_gain   * σ(rho)
    Provides level-set masks for outer/core and a morphogenesis step:
        ∂t rho = D ∇² rho + eta * harvest - gamma_mass * mass_cost - lam * rho
    The 'harvest' and 'mass_cost' are supplied by the runner from local edge stats.
    """

    def __init__(self, cfg: WorldConfig, dtype=np.float32):
        self.cfg = cfg
        self.dtype = dtype

        nx, ny, nz = cfg.space.dims
        self.dx = float(cfg.space.dx)
        self.shape = (nx, ny, nz)

        # Build coordinate grids (centered indexing)
        xs = np.arange(nx, dtype=np.float32)
        ys = np.arange(ny, dtype=np.float32)
        zs = np.arange(nz, dtype=np.float32)
        self._X, self._Y, self._Z = np.meshgrid(xs, ys, zs, indexing="ij")

        # Initialize rho
        self.rho = self._init_rho(cfg.rho).astype(dtype, copy=False)

        # Material nonlinearity
        self._sigma = _nonlin(cfg.material.nonlin)

        # Derived fields
        self.eps = np.empty_like(self.rho)
        self.g = np.empty_like(self.rho)

        # Masks
        self.outer_mask = np.zeros_like(self.rho, dtype=bool)
        self.core_mask = np.zeros_like(self.rho, dtype=bool)

        # First compute derived fields and masks
        self._update_material_and_masks()

    # ---------- Initialization ----------

    def _init_rho(self, rs: RhoSpec) -> np.ndarray:
        cx, cy, cz = rs.center
        dx = (self._X - cx)
        dy = (self._Y - cy)
        dz = (self._Z - cz)
        r2 = dx*dx + dy*dy + dz*dz
        sigma2 = max(1e-6, rs.sigma * rs.sigma)

        mode = (rs.init or "gaussian_shell").lower()
        if mode == "gaussian_shell":
            # shell: gaussian of radius ~ sigma*sqrt(2), subtract inner bump
            base = np.exp(-0.5 * r2 / sigma2)
            inner = np.exp(-0.5 * r2 / (0.25 * sigma2))
            rho = np.clip(base - 0.7 * inner, 0.0, 1.0)
        elif mode == "gaussian":
            rho = np.exp(-0.5 * r2 / sigma2)
        elif mode == "uniform_core":
            rho = np.zeros(self.shape, dtype=np.float32)
            r = np.sqrt(r2)
            rho[r <= rs.sigma] = 1.0
            # blur edges a little
            rho = self._gaussian_blur(rho, sigma=1.0)
        else:
            # default gentle blob
            rho = np.exp(-0.5 * r2 / sigma2)

        # normalize roughly to [0,1]
        mx = float(rho.max()) or 1.0
        return (rho / mx).astype(np.float32, copy=False)

    @staticmethod
    def _gaussian_blur(vol: np.ndarray, sigma: float) -> np.ndarray:
        # Simple separable blur via FFT (sufficient for init; not used in tight loops)
        import numpy.fft as fft
        nx, ny, nz = vol.shape
        kx = np.fft.fftfreq(nx)
        ky = np.fft.fftfreq(ny)
        kz = np.fft.fftfreq(nz)
        KX, KY, KZ = np.meshgrid(kx, ky, kz, indexing="ij")
        k2 = KX**2 + KY**2 + KZ**2
        H = np.exp(-2.0 * (np.pi**2) * sigma**2 * k2).astype(np.float32)
        V = fft.fftn(vol)
        return np.real(fft.ifftn(V * H)).astype(vol.dtype, copy=False)

    # ---------- Derived fields ----------

    def _update_material_and_masks(self) -> None:
        m = self.cfg.material
        sig = self._sigma(self.rho.astype(np.float32))

        self.eps[...] = m.eps_min + m.eps_gain * sig
        self.g[...] = m.g_min + m.g_gain * sig

        # Level set masks
        th1 = float(self.cfg.rho.theta_outer)
        th2 = float(self.cfg.rho.theta_core)
        self.outer_mask[...] = (self.rho >= th1) & (self.rho < th2)
        self.core_mask[...] = (self.rho >= th2)

    # ---------- Morphogenesis ----------

    def step_morphogenesis(
        self,
        harvest_field: Optional[np.ndarray],
        mass_cost_field: Optional[np.ndarray],
        dt: float = 1.0,
    ) -> None:
        """
        Update rho using local PDE:
            ∂t rho = D ∇² rho + eta * harvest - gamma_mass * mass_cost - lam * rho

        All fields are voxelwise and same shape as rho. If harvest_field or
        mass_cost_field are None, they are treated as zeros.

        Notes:
        - Uses a 3D 6-neighbor Laplacian (second-order finite differences).
        - Clamps rho into [0, 1.5] to avoid runaway; thresholds still work.
        """
        morph = self.cfg.morph
        D = float(morph.D)
        eta = float(morph.eta)
        gamma_mass = float(morph.gamma_mass)
        lam = float(morph.lam)

        # Laplacian with Neumann boundary (replicate edges)
        rho = self.rho
        lap = self._laplacian_3d6(rho) / (self.dx * self.dx)

        h = 0.0 if harvest_field is None else harvest_field
        mc = 0.0 if mass_cost_field is None else mass_cost_field

        # Update
        drho = D * lap + eta * h - gamma_mass * mc - lam * rho
        self.rho[...] = np.clip(rho + dt * drho, 0.0, 1.5).astype(self.dtype, copy=False)

        # Refresh materials & masks
        self._update_material_and_masks()

    @staticmethod
    def _laplacian_3d6(vol: np.ndarray) -> np.ndarray:
        """6-neighbor Laplacian with replicate boundary conditions."""
        x = vol
        xp = np.pad(x, ((1,1),(1,1),(1,1)), mode="edge")
        lap = (
            xp[2:,1:-1,1:-1] + xp[:-2,1:-1,1:-1] +
            xp[1:-1,2:,1:-1] + xp[1:-1,:-2,1:-1] +
            xp[1:-1,1:-1,2:] + xp[1:-1,1:-1,:-2] -
            6.0 * xp[1:-1,1:-1,1:-1]
        )
        return lap.astype(x.dtype, copy=False)

    # ---------- Accessors & helpers ----------

    def gradient_magnitude(self) -> np.ndarray:
        """|∇rho| using central differences (replicate edges)."""
        x = self.rho
        xp = np.pad(x, ((1,1),(1,1),(1,1)), mode="edge")
        gx = (xp[2:,1:-1,1:-1] - xp[:-2,1:-1,1:-1]) / (2.0 * self.dx)
        gy = (xp[1:-1,2:,1:-1] - xp[1:-1,:-2,1:-1]) / (2.0 * self.dx)
        gz = (xp[1:-1,1:-1,2:] - xp[1:-1,1:-1,:-2]) / (2.0 * self.dx)
        return np.sqrt(gx*gx + gy*gy + gz*gz).astype(self.dtype, copy=False)

    def snapshot(self) -> Dict[str, np.ndarray]:
        """Return current volumes useful for UI export (no downsampling)."""
        return {
            "rho": self.rho.copy(),
            "eps": self.eps.copy(),
            "outer_mask": self.outer_mask.copy(),
            "core_mask": self.core_mask.copy(),
        }

    def downsample(self, factor: int = 2) -> Dict[str, np.ndarray]:
        """
        Simple average-pooling downsample for UI.
        factor must divide each dimension. For masks, use max-pool.
        """
        if factor <= 1:
            return self.snapshot()

        def _avg_pool(v: np.ndarray, f: int) -> np.ndarray:
            nx, ny, nz = v.shape
            assert nx % f == 0 and ny % f == 0 and nz % f == 0, "downsample factor must divide dims"
            v2 = v.reshape(nx//f, f, ny//f, f, nz//f, f).mean(axis=(1,3,5))
            return v2.astype(v.dtype, copy=False)

        def _max_pool_bool(v: np.ndarray, f: int) -> np.ndarray:
            nx, ny, nz = v.shape
            assert nx % f == 0 and ny % f == 0 and nz % f == 0
            v2 = v.reshape(nx//f, f, ny//f, f, nz//f, f).max(axis=(1,3,5))
            return v2

        return {
            "rho": _avg_pool(self.rho, factor),
            "eps": _avg_pool(self.eps, factor),
            "outer_mask": _max_pool_bool(self.outer_mask, factor),
            "core_mask": _max_pool_bool(self.core_mask, factor),
        }

    # ---------- Builders from JSON dicts ----------

    @staticmethod
    def from_config_dict(d: Dict[str, Any]) -> "World":
        sp = d["space"]; rs = d["rho"]; mat = d["material"]; mo = d["morphogenesis"]
        cfg = WorldConfig(
            space=SpaceSpec(dims=tuple(sp["dims"]), dx=float(sp["dx"])),
            rho=RhoSpec(
                init=str(rs.get("init","gaussian_shell")),
                theta_outer=float(rs["theta_outer"]),
                theta_core=float(rs["theta_core"]),
                sigma=float(rs["sigma"]),
                center=tuple(rs["center"]),
            ),
            material=MaterialSpec(
                eps_min=float(mat["eps_min"]),
                eps_gain=float(mat["eps_gain"]),
                g_min=float(mat["g_min"]),
                g_gain=float(mat["g_gain"]),
                nonlin=str(mat.get("nonlin","softplus")),
            ),
            morph=MorphogenesisSpec.from_dict(mo),
        )
        return World(cfg)