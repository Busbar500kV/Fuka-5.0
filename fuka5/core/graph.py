from __future__ import annotations
from typing import Any, Dict, List, Tuple, Optional

import numpy as np

# Try SciPy KDTree for speed; fall back to brute force if unavailable
try:
    from scipy.spatial import cKDTree as _KDTree  # type: ignore
    _HAVE_KDTREE = True
except Exception:  # pragma: no cover
    _KDTree = None
    _HAVE_KDTREE = False


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------

def build_graph(cfg: Dict[str, Any], world: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    """
    Build a sparse geometric graph on a subsampled lattice of the world.

    Returns:
      {
        "nodes_xyz": float32 array [N,3] of node coordinates (z,y,x),
        "nodes_zyx": int32   array [N,3] of nearest lattice indices (z,y,x),
        "edges_src": int32   array [E],
        "edges_dst": int32   array [E],
        "edges_dist": float32 array [E],
        "edges_w": float32 array [E],   # weight for updates
      }
    """
    shape = world["rho"].shape
    stride = int(cfg.get("graph_stride", 6))
    k = int(cfg.get("graph_knn", 6))
    max_dist = float(cfg.get("graph_max_dist", max(shape) * 0.2))
    dist_decay = float(cfg.get("edge_decay", 0.12))
    eps_gain = float(cfg.get("edge_eps_gain", 0.25))  # coupling to local eps

    nodes_zyx, nodes_xyz = _sample_nodes(shape, stride)
    edges_src, edges_dst, edges_dist = _knn_edges(nodes_xyz, k=k, max_dist=max_dist)

    # Weights: exponential by distance, modulated by local epsilon around source/dst
    eps = world["eps"]
    eps_src = _gather_eps(eps, nodes_zyx[edges_src])
    eps_dst = _gather_eps(eps, nodes_zyx[edges_dst])
    eps_pair = 0.5 * (eps_src + eps_dst)

    edges_w = np.exp(-dist_decay * edges_dist).astype(np.float32)
    edges_w *= (1.0 + eps_gain * (eps_pair - 0.5)).astype(np.float32)

    return dict(
        nodes_xyz=nodes_xyz.astype(np.float32),
        nodes_zyx=nodes_zyx.astype(np.int32),
        edges_src=edges_src.astype(np.int32),
        edges_dst=edges_dst.astype(np.int32),
        edges_dist=edges_dist.astype(np.float32),
        edges_w=edges_w.astype(np.float32),
    )


# ---------------------------------------------------------------------
# Node sampling
# ---------------------------------------------------------------------

def _sample_nodes(shape: Tuple[int, int, int], stride: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Subsample a regular grid with the given stride.
    Returns (nodes_zyx int [N,3], nodes_xyz float [N,3]).
    """
    z = np.arange(0, shape[0], stride, dtype=np.int32)
    y = np.arange(0, shape[1], stride, dtype=np.int32)
    x = np.arange(0, shape[2], stride, dtype=np.int32)
    zz, yy, xx = np.meshgrid(z, y, x, indexing="ij")
    nodes_zyx = np.stack([zz.ravel(), yy.ravel(), xx.ravel()], axis=1)  # [N,3]
    nodes_xyz = nodes_zyx.astype(np.float32)  # (z,y,x) used consistently as xyz
    return nodes_zyx, nodes_xyz


# ---------------------------------------------------------------------
# Edge construction
# ---------------------------------------------------------------------

def _knn_edges(nodes_xyz: np.ndarray, k: int, max_dist: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute undirected kNN edges (source->dest) with a max distance cap.
    Returns (src_idx, dst_idx, distances). Self-edges are excluded.
    """
    if nodes_xyz.shape[0] == 0:
        return (
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.int32),
            np.empty((0,), dtype=np.float32),
        )

    if _HAVE_KDTREE and nodes_xyz.shape[0] >= 8:
        tree = _KDTree(nodes_xyz)
        # query k+1 to include self, then drop self
        dists, idxs = tree.query(nodes_xyz, k=min(k + 1, nodes_xyz.shape[0]))
        # Normalize to shapes [N,k_eff]
        if idxs.ndim == 1:
            idxs = idxs[:, None]
            dists = dists[:, None]

        # Drop self at position 0 where distance==0
        idxs_no_self = idxs[:, 1:]
        dists_no_self = dists[:, 1:]

        # Apply max distance
        mask = dists_no_self <= max_dist
        src_list, dst_list, dist_list = _mask_to_edge_lists(mask, dists_no_self, idxs_no_self)
    else:
        # Brute force (fallback)
        src_list, dst_list, dist_list = _brute_knn(nodes_xyz, k=k, max_dist=max_dist)

    # Flatten
    src = np.asarray(src_list, dtype=np.int32)
    dst = np.asarray(dst_list, dtype=np.int32)
    dist = np.asarray(dist_list, dtype=np.float32)
    return src, dst, dist


def _mask_to_edge_lists(mask: np.ndarray, dists: np.ndarray, idxs: np.ndarray) -> Tuple[List[int], List[int], List[float]]:
    src_list: List[int] = []
    dst_list: List[int] = []
    dist_list: List[float] = []
    N, K = mask.shape
    for i in range(N):
        for j in range(K):
            if mask[i, j]:
                src_list.append(i)
                dst_list.append(int(idxs[i, j]))
                dist_list.append(float(dists[i, j]))
    return src_list, dst_list, dist_list


def _brute_knn(nodes_xyz: np.ndarray, k: int, max_dist: float) -> Tuple[List[int], List[int], List[float]]:
    N = nodes_xyz.shape[0]
    src: List[int] = []
    dst: List[int] = []
    dist: List[float] = []
    # Compute full distance matrix (O(N^2)) – acceptable for small N
    diffs = nodes_xyz[:, None, :] - nodes_xyz[None, :, :]
    d2 = np.sum(diffs * diffs, axis=2)
    np.fill_diagonal(d2, np.inf)
    d = np.sqrt(d2)
    # For each node, take up to k nearest within max_dist
    for i in range(N):
        order = np.argsort(d[i])[:k]
        for j in order:
            if d[i, j] <= max_dist:
                src.append(i)
                dst.append(int(j))
                dist.append(float(d[i, j]))
    return src, dst, dist


# ---------------------------------------------------------------------
# Field sampling helpers
# ---------------------------------------------------------------------

def _gather_eps(eps: np.ndarray, zyx_idx: np.ndarray) -> np.ndarray:
    """
    Gather eps at nearest indices.
    zyx_idx: [E,3] int indices into eps[z,y,x].
    """
    z = np.clip(zyx_idx[:, 0], 0, eps.shape[0] - 1)
    y = np.clip(zyx_idx[:, 1], 0, eps.shape[1] - 1)
    x = np.clip(zyx_idx[:, 2], 0, eps.shape[2] - 1)
    return eps[z, y, x].astype(np.float32)