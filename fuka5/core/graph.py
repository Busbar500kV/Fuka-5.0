"""
fuka5.core.graph
----------------
Build a local graph (nodes + undirected edges) from the world field ρ(x,t).

Responsibilities
* Sample node positions with higher density near |∇ρ| (substrate boundaries),
  plus uniform coverage inside outer+core bands.
* Build k-NN edges with distance cap.
* Compute edge priors from material fields along the straight segment:
    C_ij_prior ~ <epsilon> * A_ij / d_ij
    G_ij_prior ~ <g>       * A_ij
  where <·> is a short quadrature average along i↔j.
* Provide lightweight containers for nodes/edges used by the substrate.

Notes
- The cross-sectional area A_ij is a tunable constant factor; we use A=1.0 by default
  and absorb scale into learning. Distances are in world 'dx' units.
- Everything is numpy; kNN is done via a naive partial sort for simplicity (dims are small).
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, List, Dict, Any
import numpy as np


@dataclass
class Node:
    id: int
    x: float
    y: float
    z: float
    # local thermal leak modifier is handled in substrate.thermal

@dataclass
class Edge:
    id: int
    u: int
    v: int
    d: float         # Euclidean distance
    C_prior: float   # prior capacitance from epsilon
    G_prior: float   # tiny prior leak from g

@dataclass
class GraphConfig:
    k_nn: int = 8
    dist_cap: float = 4.5           # max neighbor distance (in dx units)
    boundary_boost: float = 1.5     # sampling density multiplier near |∇ρ|
    samples_per_voxel: float = 0.02 # ~percentage of voxels to sample as nodes
    area_const: float = 1.0         # A in C ~ eps*A/d
    quad_steps: int = 5             # samples for path-averaging of eps,g


class Graph:
    def __init__(self, nodes: List[Node], edges: List[Edge]):
        self.nodes = nodes
        self.edges = edges
        # quick arrays for performance
        self._pos = np.array([(n.x, n.y, n.z) for n in nodes], dtype=np.float32)

    @property
    def N(self) -> int:
        return len(self.nodes)

    @property
    def E(self) -> int:
        return len(self.edges)

    def positions(self) -> np.ndarray:
        return self._pos.copy()


# ---------------------------
# Builders
# ---------------------------

def build_graph_from_world(
    rho: np.ndarray,
    eps: np.ndarray,
    g: np.ndarray,
    outer_mask: np.ndarray,
    core_mask: np.ndarray,
    dx: float,
    cfg: GraphConfig | None = None,
    rng: np.random.Generator | None = None,
) -> Graph:
    """
    Construct a graph from world volumes. The node sampler:
      - chooses a base number of samples proportional to volume
      - boosts probability where |∇ρ| is high and within outer/core masks
    """
    cfg = cfg or GraphConfig()
    rng = rng or np.random.default_rng(0)

    nx, ny, nz = rho.shape
    grad_mag = _grad_mag(rho)
    # probability grid: outer or core gets 1.0, boundary boost adds grad term
    prob = (outer_mask | core_mask).astype(np.float32)
    # normalize |∇ρ| to [0,1]
    if grad_mag.max() > 0:
        gm = grad_mag / (grad_mag.max() + 1e-9)
        prob = prob + cfg.boundary_boost * gm * (outer_mask | core_mask)

    prob = prob / (prob.sum() + 1e-9)

    # number of nodes ~ samples_per_voxel * count(substrate voxels)
    n_vox = int((outer_mask | core_mask).sum())
    n_nodes = max(16, int(cfg.samples_per_voxel * n_vox))
    idx_lin = rng.choice(prob.size, size=n_nodes, replace=False, p=prob.ravel())
    iz = idx_lin // (ny * nz)
    iy = (idx_lin - iz * ny * nz) // nz
    ix = idx_lin % nz  # careful: this is z index; re-map properly
    # fix indexing (we used ravel order (x,y,z) with ijk mapping)
    # In numpy ravel order, index -> (i,j,k) mapping:
    k = idx_lin % nz
    j = (idx_lin // nz) % ny
    i = idx_lin // (ny * nz)

    xs = i.astype(np.float32)
    ys = j.astype(np.float32)
    zs = k.astype(np.float32)

    nodes = [Node(id=i_, x=float(xs[t]), y=float(ys[t]), z=float(zs[t])) for t, i_ in enumerate(range(n_nodes))]

    # Build edges via naive kNN with distance cap
    pos = np.stack([xs, ys, zs], axis=1)
    edges: List[Edge] = []
    eid = 0
    for u in range(n_nodes):
        # distances to all others
        d2 = np.sum((pos - pos[u])**2, axis=1)
        d2[u] = np.inf
        # get k nearest within cap
        nn_idx = np.argpartition(d2, cfg.k_nn)[:cfg.k_nn]
        nn_idx = [v for v in nn_idx if v != u and d2[v] < (cfg.dist_cap * cfg.dist_cap)]
        for v in nn_idx:
            if v < u:
                continue  # add each undirected edge once
            d = float(np.sqrt(d2[v])) * dx
            Cpr, Gpr = _edge_priors(eps, g, pos[u], pos[v], dx, area_const=cfg.area_const, steps=cfg.quad_steps)
            edges.append(Edge(id=eid, u=u, v=v, d=d, C_prior=Cpr, G_prior=Gpr))
            eid += 1

    return Graph(nodes, edges)


# ---------------------------
# Internals
# ---------------------------

def _grad_mag(vol: np.ndarray) -> np.ndarray:
    x = vol
    xp = np.pad(x, ((1,1),(1,1),(1,1)), mode="edge")
    gx = (xp[2:,1:-1,1:-1] - xp[:-2,1:-1,1:-1]) * 0.5
    gy = (xp[1:-1,2:,1:-1] - xp[1:-1,:-2,1:-1]) * 0.5
    gz = (xp[1:-1,1:-1,2:] - xp[1:-1,1:-1,:-2]) * 0.5
    return np.sqrt(gx*gx + gy*gy + gz*gz).astype(np.float32, copy=False)

def _edge_priors(
    eps: np.ndarray,
    g: np.ndarray,
    pu: np.ndarray,  # (x,y,z) in voxel coords
    pv: np.ndarray,
    dx: float,
    area_const: float = 1.0,
    steps: int = 5,
) -> tuple[float, float]:
    """
    Average epsilon and g along the straight segment between nodes u and v,
    then compute simple priors:
        C_prior = <eps> * A / d
        G_prior = <g>   * A
    """
    # parameterize points between pu and pv
    ts = np.linspace(0.0, 1.0, num=max(2, int(steps)))
    pts = (1 - ts)[:, None] * pu[None, :] + ts[:, None] * pv[None, :]

    def _sample(vol: np.ndarray, p: np.ndarray) -> float:
        # trilinear with boundary clamp
        nx, ny, nz = vol.shape
        x, y, z = p
        x = float(np.clip(x, 0, nx - 1 - 1e-6))
        y = float(np.clip(y, 0, ny - 1 - 1e-6))
        z = float(np.clip(z, 0, nz - 1 - 1e-6))
        xi, yi, zi = int(x), int(y), int(z)
        xf, yf, zf = x - xi, y - yi, z - zi
        def w(a, b): return a * (1 - b) + b * a  # placeholder not used
        # eight corners
        c000 = vol[xi, yi, zi]
        c100 = vol[xi+1, yi, zi]
        c010 = vol[xi, yi+1, zi]
        c110 = vol[xi+1, yi+1, zi]
        c001 = vol[xi, yi, zi+1]
        c101 = vol[xi+1, yi, zi+1]
        c011 = vol[xi, yi+1, zi+1]
        c111 = vol[xi+1, yi+1, zi+1]
        c00 = c000*(1-xf) + c100*xf
        c01 = c001*(1-xf) + c101*xf
        c10 = c010*(1-xf) + c110*xf
        c11 = c011*(1-xf) + c111*xf
        c0 = c00*(1-yf) + c10*yf
        c1 = c01*(1-yf) + c11*yf
        return float(c0*(1-zf) + c1*zf)

    eps_vals = np.array([_sample(eps, p) for p in pts], dtype=np.float64)
    g_vals   = np.array([_sample(g,   p) for p in pts], dtype=np.float64)
    eps_mean = float(eps_vals.mean())
    g_mean   = float(g_vals.mean())

    # geometric terms
    d_vox = float(np.linalg.norm(pv - pu)) * dx
    d_vox = max(d_vox, 1e-6)
    A = area_const

    C_prior = eps_mean * A / d_vox
    G_prior = g_mean * A
    return C_prior, G_prior