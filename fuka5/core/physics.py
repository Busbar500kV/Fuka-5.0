"""
fuka5.core.physics
------------------
Frequency-domain phasor solver on the adaptive graph, plus short-time synthesis
and lock-in measurements for edge-local signals.

Model
=====
For each frequency ω:
  Node KCL with admittances on edges and to sources:
    Sum_{j in N(i)} Y_ij (V_i - V_j) + Sum_{s in S(i)} Y_is (V_i - V_s) + G_i V_i = 0
  where:
    Y_ij(ω) = G_ij + j ω C_ij
    Y_is(ω) = j ω C_tap(is)          (source tap as coupling capacitor)
    V_s     = complex source voltage for that frequency (A * e^{j φ})

Rewriting:
    (Σ_j Y_ij + Σ_s Y_is + G_i) V_i  - Σ_j Y_ij V_j = Σ_s Y_is V_s

So the linear system is:
    Y(ω) · V(ω) = b(ω)

We solve for V(ω) for all ω in the sources, optionally restricting to a subset
of source names (useful to attribute band powers per-source).

Synthesis
=========
Edge time signals (for a window) are obtained by superposition over ω:
    v_e(t) = Re{ (V_u(ω) - V_v(ω)) e^{j ω t} } summed over ω
(You may choose Imag instead of Real; keep it consistent across the codebase.)

Lock-ins
========
For a signal x(t) and frequency f:
    M(f) = ⟨ x(t) · e^{-j 2π f t} ⟩_window
We provide utilities to compute window energies and lock-ins per edge.

Dependencies: numpy only.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple, Iterable, Optional
import numpy as np

from .sources import Sources, SourceDrive
from .graph import Graph, Edge


# ---------------------------
# Solver
# ---------------------------

@dataclass
class SolveResult:
    """Holds solved node phasors for a set of frequencies."""
    # map: freq_hz -> complex array of shape (N_nodes,)
    V: Dict[float, np.ndarray]


class PhasorSolver:
    """
    Build and solve Y(ω) V = b(ω) for a graph with adaptive edge C and tiny G.
    Node leaks G_i are optional (0 by default).
    Source taps are capacitive couplings given by Sources.taps.
    """

    def __init__(self, graph: Graph):
        self.g = graph
        self.N = graph.N
        self.E = graph.E

    def solve(
        self,
        freqs_hz: Iterable[float],
        C_edge: np.ndarray,      # shape (E,)
        G_edge: Optional[np.ndarray],  # shape (E,) tiny leak, or None → zeros
        node_G_leak: Optional[np.ndarray],  # shape (N,), or None → zeros
        sources: Sources,
        use_only_sources: Optional[Iterable[str]] = None,
        dtype=np.complex128,
    ) -> SolveResult:
        """
        Solve for node phasors at the requested frequencies.
        If use_only_sources is provided, only those source names inject taps (others ignored).
        """
        N = self.N
        E = self.E
        if G_edge is None:
            G_edge = np.zeros(E, dtype=float)
        if node_G_leak is None:
            node_G_leak = np.zeros(N, dtype=float)

        # prepack edges
        U = np.array([e.u for e in self.g.edges], dtype=int)
        V = np.array([e.v for e in self.g.edges], dtype=int)

        # taps map: name -> list[(node_id, Ctap)]
        taps = sources.taps
        allowed = None if use_only_sources is None else set(use_only_sources)

        solved: Dict[float, np.ndarray] = {}
        for f in freqs_hz:
            w = 2.0 * np.pi * float(f)
            j = 1j

            # Edge admittances
            Y_e = G_edge + j * w * C_edge  # shape (E,)

            # Assemble sparse-ish Y matrix in dense form (N x N) for simplicity (graphs are modest)
            Y = np.zeros((N, N), dtype=dtype)

            # Off-diagonals: -Y_e
            for k in range(E):
                u = U[k]; v = V[k]; y = Y_e[k]
                Y[u, v] -= y
                Y[v, u] -= y

            # Diagonals: sum of incident admittances + node leaks
            # Accumulate edge contributions
            diag = np.zeros(N, dtype=dtype)
            np.add.at(diag, U, Y_e)
            np.add.at(diag, V, Y_e)

            # Node leaks
            diag += node_G_leak.astype(dtype, copy=False)

            # Source taps: add Y_is on diag, add Y_is * V_s to RHS b
            b = np.zeros(N, dtype=dtype)

            for sname, lst in taps.items():
                if allowed is not None and sname not in allowed:
                    continue
                # complex source voltage at this freq
                Vs = _source_voltage_at(sources, sname, f)  # complex
                if Vs is None:
                    continue
                for (nid, Ctap) in lst:
                    yis = j * w * Ctap
                    diag[nid] += yis
                    b[nid] += yis * Vs

            # write diag into Y
            for i in range(N):
                Y[i, i] += diag[i]

            # Solve
            try:
                Vnodes = np.linalg.solve(Y, b)
            except np.linalg.LinAlgError:
                # Regularize lightly if singular
                Vnodes = np.linalg.lstsq(Y + 1e-12*np.eye(N, dtype=dtype), b, rcond=None)[0]

            solved[float(f)] = Vnodes.astype(dtype, copy=False)

        return SolveResult(V=solved)

    # ------------- Synthesis & measures -------------

    def edge_time_series(
        self,
        sol: SolveResult,
        t: np.ndarray,
        use_real_part: bool = True,
    ) -> np.ndarray:
        """
        Build edge time series v_e(t) by summing over solved frequencies.
        Returns array of shape (T, E) where E = number of edges.
        """
        T = t.shape[0]
        E = self.E
        out = np.zeros((T, E), dtype=float)
        # map edges
        U = np.array([e.u for e in self.g.edges], dtype=int)
        V = np.array([e.v for e in self.g.edges], dtype=int)

        for f, Vnodes in sol.V.items():
            w = 2.0 * np.pi * f
            # edge phasor = V_u - V_v
            edge_ph = Vnodes[U] - Vnodes[V]  # shape (E,)
            # time-domain contribution
            osc = np.exp(1j * w * t[:, None])  # (T,1)
            comp = edge_ph[None, :] * osc      # (T,E)
            if use_real_part:
                out += np.real(comp)
            else:
                out += np.imag(comp)
        return out

    @staticmethod
    def lockins(v_te: np.ndarray, freqs_hz: Iterable[float], t: np.ndarray) -> Dict[float, np.ndarray]:
        """
        Compute lock-ins for each edge signal v_e(t) at given freqs.
        v_te: shape (T, E) time series
        Returns: dict[freq] -> complex array of shape (E,) (the lock-in value)
        """
        T, E = v_te.shape
        res: Dict[float, np.ndarray] = {}
        for f in freqs_hz:
            ph = np.exp(-1j * 2.0 * np.pi * float(f) * t)[:, None]  # (T,1)
            M = (v_te * ph).mean(axis=0)  # (E,)
            res[float(f)] = M.astype(np.complex128, copy=False)
        return res

    @staticmethod
    def energy(v_te: np.ndarray) -> np.ndarray:
        """Window energy per edge: E_e = ⟨ v_e(t)^2 ⟩_t ; returns shape (E,)."""
        return (v_te ** 2).mean(axis=0) + 1e-12


# ---------------------------
# Helpers
# ---------------------------

def _source_voltage_at(sources: Sources, sname: str, f: float) -> complex | None:
    """Return complex V_s for a named source at a frequency f, or None if it has no tone at f."""
    drives = sources.drives_at(f)
    for d in drives:
        if d.name == sname:
            return d.amp_complex
    return None


# ---------------------------
# Convenience routines
# ---------------------------

def solve_all_and_synthesize(
    graph: Graph,
    C_edge: np.ndarray,
    G_edge: Optional[np.ndarray],
    node_G_leak: Optional[np.ndarray],
    sources: Sources,
    t: np.ndarray,
    subset_sources: Optional[Iterable[str]] = None,
    use_real_part: bool = True,
) -> Tuple[Dict[float, np.ndarray], np.ndarray]:
    """
    One-shot helper:
      - Solve for all source frequencies (or subset)
      - Return (Vnodes per f, edge time series v(t,e))

    Returns:
      (V_by_f, v_te)
        V_by_f: dict[f] -> (N,) complex
        v_te: (T, E) float
    """
    solver = PhasorSolver(graph)
    freqs = sources.frequencies()
    if subset_sources is not None:
        # keep only freqs that subset actually drives (optimization)
        keep = []
        allowed = set(subset_sources)
        for f in freqs:
            if any(d.name in allowed for d in sources.drives_at(f)):
                keep.append(f)
        freqs = keep

    sol = solver.solve(
        freqs_hz=freqs,
        C_edge=C_edge,
        G_edge=G_edge,
        node_G_leak=node_G_leak,
        sources=sources,
        use_only_sources=subset_sources,
    )
    v_te = solver.edge_time_series(sol, t, use_real_part=use_real_part)
    return sol.V, v_te