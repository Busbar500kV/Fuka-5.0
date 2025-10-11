from __future__ import annotations
def _normalize_freq_dict(d):
    nd = {}
    for k, v in d.items():
        try:
            kf = float(k)
        except Exception:
            try:
                kf = float(str(k))
            except Exception:
                continue
        nd[kf] = v
    return nd
def _get_lock_val(d, k):
    # tolerate 11000 / 11000.0 / "11000"
    candidates = []
    candidates.append(k)
    try: candidates.append(float(k))
    except: pass
    try: candidates.append(int(float(k)))
    except: pass
    try:
        iv = int(float(k))
        candidates.append(str(iv))
    except: pass
    try: candidates.append(str(k))
    except: pass
    seen = set()
    for key in candidates:
        if key in d and key not in seen:
            return d[key]
        seen.add(key)
    raise KeyError(k)
"""
fuka5.substrate.updates
-----------------------
One-epoch orchestration of *local* updates on the capacitor substrate.

This module ties together:
  - Physics solve (phasors) → edge time traces v_e(t)
  - Lock-ins / powers per band and per source (s1 vs s1′)
  - Edge-local gates update (split/mix), batteries, maturity
  - Thermal node update and leak multipliers
  - Capacitance growth (budgeted Hebbian) + upkeep − pruning
  - Optional self-rehearsal during OFF windows
  - LMS decoder updates and scalar reward aggregation
  - Per-edge and per-epoch outputs for shard writers

It stays framework-free (NumPy only). Higher-level scheduling (epochs, ON/OFF)
and I/O are handled in fuka5.run.sim_cli and fuka5.io.*
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import numpy as np

from ..core.physics import solve_all_and_synthesize, PhasorSolver
from ..core.sources import Sources
from ..core.graph import Graph
from .edges import EdgeState, initialize_edge_states
from .gates import GateParams, update_edge_gates_for_all_bands
from .battery import (
    BatteryParams, MaturityParams, battery_update, maturity_update,
    effective_Cmin_lambdaC, budget_gate, upkeep_term, rehearsal_amplitude
)
from .thermal import ThermalParams, ThermalField
from .decoders import DecoderBank


# ---------------------------
# Config containers
# ---------------------------

@dataclass
class CapsParams:
    Cmin: float
    Cmax: float
    etaC: float
    lambdaC: float

    @staticmethod
    def from_dict(d: Dict) -> "CapsParams":
        return CapsParams(
            Cmin=float(d["Cmin"]),
            Cmax=float(d["Cmax"]),
            etaC=float(d["etaC"]),
            lambdaC=float(d["lambdaC"]),
        )

@dataclass
class RewardsParams:
    alpha: float = 1.0
    beta: float = 2.0
    gamma_cap_cost: float = 5e9

    @staticmethod
    def from_dict(d: Dict) -> "RewardsParams":
        return RewardsParams(
            alpha=float(d.get("alpha", 1.0)),
            beta=float(d.get("beta", 2.0)),
            gamma_cap_cost=float(d.get("gamma_cap_cost", 5e9)),
        )


# ---------------------------
# Utilities
# ---------------------------

def _bands_list(bands_cfg: Dict[str, List[float]]) -> List[str]:
    return list(bands_cfg.keys())

def _band_membership(freqs: List[float], bands_cfg: Dict[str, List[float]]) -> Dict[str, List[int]]:
    """Return indices of `freqs` belonging to each named band."""
    mapping: Dict[str, List[int]] = {b: [] for b in bands_cfg.keys()}
    for idx, f in enumerate(freqs):
        for b, flist in bands_cfg.items():
            if any(abs(f - ff) < 1e-6 for ff in flist):
                mapping[b].append(idx)
    return mapping

def _sum_power_over_band(M_e: np.ndarray, idxs: List[int]) -> float:
    """Sum |M|^2 over selected frequency indices for an edge."""
    if not idxs:
        return 0.0
    sel = M_e[idxs]
    return float(np.sum(np.abs(sel)**2))

def _lockins_for_subset(graph: Graph, C_edge: np.ndarray, node_G_leak: np.ndarray,
                        sources: Sources, t: np.ndarray, subset: Optional[List[str]]) -> Dict[float, np.ndarray]:
    """
    Solve and compute lock-ins only for the subset of sources (None means all).
    Returns dict[f] -> (E,) complex lockins for each edge.
    """
    V_by_f, v_te = solve_all_and_synthesize(
        graph=graph,
        C_edge=C_edge,
        G_edge=None,
        node_G_leak=node_G_leak,
        sources=sources,
        t=t,
        subset_sources=subset,
        use_real_part=True,
    )
    lock = PhasorSolver.lockins(v_te, V_by_f.keys(), t)
    return lock


# ---------------------------
# Epoch step
# ---------------------------

def step_epoch(
    *,
    epoch: int,
    on_flag: bool,
    graph: Graph,
    edge_states: Dict[int, EdgeState],
    sources: Sources,
    bands_cfg: Dict[str, List[float]],
    time_cfg: Dict[str, float],     # keys: fs, window_sec
    caps_params: CapsParams,
    gate_params: GateParams,
    batt_params: BatteryParams,
    mat_params: MaturityParams,
    therm: ThermalField,
    decoders: DecoderBank,
    rewards_params: RewardsParams,
    rng: np.random.Generator | None = None,
) -> Tuple[List[Dict], Dict[str, float], np.ndarray]:
    """
    Run one epoch window of updates. Returns:
      - edge_rows: list of dicts suitable for writing to edges parquet
      - metrics_row: dict with per-epoch metrics
      - node_Pplus: per-node usable power proxy (for morphogenesis & thermal)
    """
    rng = rng or np.random.default_rng(0)
    fs = float(time_cfg["fs"])
    Tsec = float(time_cfg["window_sec"])
    t = np.arange(0.0, Tsec, 1.0 / fs, dtype=np.float64)
    Ecount = len(edge_states)

    # Assemble vectors aligned with graph.edges order
    edges_sorted = sorted(graph.edges, key=lambda e: e.id)
    C_edge = np.array([edge_states[e.id].C for e in edges_sorted], dtype=np.float64)

    # baseline node leaks (tiny), modulated by thermal later
    G_node0 = np.full(graph.N, 1.0 / 50e6, dtype=np.float64)

    # Thermal leak multiplier from last epoch
    leak_mult = therm.leak_multiplier().astype(np.float64, copy=False)
    node_G_leak = G_node0 * leak_mult  # (N,)

    # Frequencies present
    freqs = sources.frequencies()
    band_idxs = _band_membership(freqs, bands_cfg)

    # --- Solve physics and synthesize windows ---
    if on_flag:
        # All sources together (for LMS and total energy)
        V_all, v_te_all = solve_all_and_synthesize(
            graph, C_edge, None, node_G_leak, sources, t, subset_sources=None, use_real_part=True
        )
        # Separate contributions for s1 and s1p (for per-source band powers & gate updates)
        lock_s1  = _lockins_for_subset(graph, C_edge, node_G_leak, sources, t, subset=["s1"])
        lock_s1p = _lockins_for_subset(graph, C_edge, node_G_leak, sources, t, subset=["s1p"])
        lock_all = PhasorSolver.lockins(v_te_all, freqs, t)
        v_te = v_te_all
    else:
        # OFF: no external sources → empty drives
        # produce flat v_te; we will add tiny self-rehearsal locally
        V_all = {f: np.zeros(graph.N, np.complex128) for f in freqs}
        v_te = np.zeros((t.shape[0], graph.E), dtype=np.float64)
        lock_s1  = {f: np.zeros(graph.E, np.complex128) for f in freqs}
        lock_s1p = {f: np.zeros(graph.E, np.complex128) for f in freqs}
        lock_all = {f: np.zeros(graph.E, np.complex128) for f in freqs}

    # Self-rehearsal injection (purely local, micro amplitude) during OFF
    if not on_flag and freqs:
        for e in edges_sorted:
            st = edge_states[e.id]
            a_reh = rehearsal_amplitude(st, batt_params)
            if a_reh <= 0.0:
                continue
            # inject sinusoids at all known freqs with tiny amplitude; antisymmetric across the edge
            for f in freqs:
                v_te[:, e.id] += a_reh * np.sin(2.0 * np.pi * f * t)

        # recompute lockins after rehearsal
        lock_all = PhasorSolver.lockins(v_te, freqs, t)

    # Energies per edge (for usable-power proxy and normalization)
    E_edge = PhasorSolver.energy(v_te)  # (E,)

    # --- LMS decoders & scalar rewards ---
    # Build targets per head (sum of s1 or s1p tones)
    def _synth_target(only: Optional[List[str]]) -> np.ndarray:
        sig = np.zeros_like(t)
        if not on_flag:
            return sig
        # superpose tones of the subset
        allowed = None if only is None else set(only)
        for f in freqs:
            for d in sources.drives_at(f):
                if allowed is not None and d.name not in allowed:
                    continue
                sig += np.real(d.amp_complex * np.exp(1j * 2.0 * np.pi * f * t))
        return sig

    y_s1  = _synth_target(["s1"])
    y_s1p = _synth_target(["s1p"])
    y_mix = _synth_target(None)

    mse_s1  = decoders.update(v_te, y_s1,  head="s1")
    mse_s1p = decoders.update(v_te, y_s1p, head="s1p")
    mse_mix = decoders.update(v_te, y_mix, head="mix")

    # Capacity cost
    sumC = float(np.sum(C_edge))
    R1   = float(rewards_params.alpha * 0.0 - rewards_params.beta * mse_s1  - rewards_params.gamma_cap_cost * sumC)
    R1p  = float(rewards_params.alpha * 0.0 - rewards_params.beta * mse_s1p - rewards_params.gamma_cap_cost * sumC)
    Rmix = float(rewards_params.alpha * 0.0 - rewards_params.beta * mse_mix - rewards_params.gamma_cap_cost * sumC)
    # we’ll use Rmix as the scalar driver for C updates (can change if desired)

    # --- Per-edge local updates ---
    edge_rows: List[Dict] = []
    # Prepare band-wise lockins arrays per edge: shape (F, E), tolerant to key types/missing
    freqs_f   = [float(f) for f in freqs]
    lock_allN = _normalize_freq_dict(lock_all)
    lock_s1N  = _normalize_freq_dict(lock_s1)
    lock_s1pN = _normalize_freq_dict(lock_s1p)
    
    M_all = _safe_stack(lock_allN, "lock_all", freqs_f)   # complex (F,E)
    M_s1  = _safe_stack(lock_s1N,  "lock_s1",  freqs_f)
    M_s1p = _safe_stack(lock_s1pN, "lock_s1p", freqs_f)
    
    # Per-node usable power proxy accumulator (for thermal/morphogenesis)
    Pplus_node = np.zeros(graph.N, dtype=np.float64)
    
    # Iterate edges in solver order
    for k, e in enumerate(edges_sorted):
    st = edge_states[e.id]
    
    # Band powers attributed to each source
    # sum |M|^2 over the band's frequencies for this edge index k
    def _bp(MF, band):
            idxs = band_idxs.get(band, [])
            if not idxs:
                return 0.0
            # MF (F,E) -> slice (F_sel,) at edge k
            return _sum_power_over_band(MF[idxs, k], list(range(len(idxs))))

        P_low_s1  = _bp(M_s1,  "low");  P_low_s1p  = _bp(M_s1p, "low")
        P_high_s1 = _bp(M_s1,  "high"); P_high_s1p = _bp(M_s1p,"high")

        st.P_low_s1  = P_low_s1
        st.P_low_s1p = P_low_s1p
        st.P_high_s1 = P_high_s1
        st.P_high_s1p= P_high_s1p
        st.E = float(E_edge[k])

        # Synergy & interference proxies per band (bounded to [0,1])
        def _synergy(P1, P2, E): return float(np.clip((P1 + P2) / (E + 1e-12), 0.0, 1.0))
        # interference via cross term between complex sums over the band
        def _interf(M1_band: np.ndarray, M2_band: np.ndarray, E: float) -> float:
            z1 = np.sum(M1_band); z2 = np.sum(M2_band)
            return float(np.clip(np.abs(z1 * np.conj(z2)) / (E + 1e-12), 0.0, 1.0))

        # Build per-band inputs for gates
        per_band_inputs: Dict[str, Dict[str, float]] = {}
        for band in bands_cfg.keys():
            idxs = band_idxs.get(band, [])
            M1b = M_s1[idxs, k] if idxs else np.zeros(1, np.complex128)
            M2b = M_s1p[idxs, k] if idxs else np.zeros(1, np.complex128)
            if band == "low":
                p1, p2 = P_low_s1, P_low_s1p
            else:
                p1, p2 = P_high_s1, P_high_s1p
            S = _synergy(p1, p2, st.E)
            I = _interf(M1b, M2b, st.E)
            per_band_inputs[band] = {"p1": p1, "p1p": p2, "S": S, "I": I}

        # Gate update (local)
        update_edge_gates_for_all_bands(st, bands_cfg, per_band_inputs, gate_params)

        # Usable-power proxy for battery harvest
        P_pos = st.E + P_low_s1 + P_low_s1p + P_high_s1 + P_high_s1p

        # Structural change intents (proxies for energy cost)
        phi_e = float((P_low_s1 + P_low_s1p + P_high_s1 + P_high_s1p) / (st.E + 1e-12))
        dC_intent = caps_params.etaC * max(0.0, Rmix * phi_e)
        # Approximate gate-change magnitude as L1 of (new - old) across bands
        dgate_intent = 0.0  # conservative; energy cost dominated by C changes here

        # Battery & maturity updates
        battery_update(st, P_pos=P_pos, dC_struct=dC_intent, dgate_struct=dgate_intent, params=batt_params)
        maturity_update(st, mat_params)
        Cmin_eff, lambdaC_eff = effective_Cmin_lambdaC(st, caps_params.Cmin, caps_params.lambdaC, mat_params)

        # Node temperatures for logging (will be updated after Pplus accumulation)
        st.T_u = float(therm.T[e.u])
        st.T_v = float(therm.T[e.v])

        # Energy-budgeted growth + upkeep − pruning
        budget = budget_gate(st, batt_params)
        chi_u = float(np.clip(np.mean(therm.chi()[[e.u, e.v]]), 0.0, 1.0))
        C_star = st.C  # simple target; could be replaced with EMA of productive C
        dC = caps_params.etaC * (Rmix * phi_e) * budget \
             - lambdaC_eff * (st.C - Cmin_eff) \
             + upkeep_term(st, C_star, chi_T=chi_u, params=batt_params)
        st.C = float(np.clip(st.C + dC, st.Cmin, st.Cmax))

        # Accumulate node usable power (split equally to endpoints)
        Pplus_node[e.u] += 0.5 * P_pos
        Pplus_node[e.v] += 0.5 * P_pos

        # Prepare logging row
        row = {
            "epoch": int(epoch),
            "edge_id": f"{e.u}-{e.v}",
            "u": int(e.u), "v": int(e.v),
            "x_u": float(graph.nodes[e.u].x), "y_u": float(graph.nodes[e.u].y), "z_u": float(graph.nodes[e.u].z),
            "x_v": float(graph.nodes[e.v].x), "y_v": float(graph.nodes[e.v].y), "z_v": float(graph.nodes[e.v].z),
            "C": float(st.C), "Cmin_eff": float(Cmin_eff),
            "B": float(st.B), "A": float(st.A),
            "E": float(st.E),
            "T_u": float(st.T_u), "T_v": float(st.T_v),
            "P_low_s1": float(P_low_s1), "P_low_s1p": float(P_low_s1p),
            "P_high_s1": float(P_high_s1), "P_high_s1p": float(P_high_s1p),
        }
        # add gates per band
        row.update(st.gates_dict(bands_cfg))
        edge_rows.append(row)

    # --- Thermal update using accumulated Pplus_node ---
    therm.step(Pplus_node)

    # --- Simple separation/mix indices (for metrics) ---
    def _gate_indices(rows: List[Dict], band: str) -> Tuple[float, float]:
        # sep ~ |g1 - g1p| averaged; mix ~ g_mix averaged
        g1 = np.array([r[f"g_{band}_1"]  for r in rows], dtype=np.float64)
        g2 = np.array([r[f"g_{band}_1p"] for r in rows], dtype=np.float64)
        gmix = np.array([r[f"g_{band}_mix"] for r in rows], dtype=np.float64)
        sep = float(np.mean(np.abs(g1 - g2)))
        mix = float(np.mean(gmix))
        return sep, mix

    sep_low, mix_low   = _gate_indices(edge_rows, "low")
    sep_high, mix_high = _gate_indices(edge_rows, "high")

    metrics_row = {
        "epoch": int(epoch),
        "R1": float(R1), "R1p": float(R1p), "Rmix": float(Rmix),
        "MSE_s1": float(mse_s1), "MSE_s1p": float(mse_s1p), "MSE_mix": float(mse_mix),
        "sumC": float(sumC),
        "sumB": float(np.sum([edge_states[e.id].B for e in edges_sorted])),
        "avgT": float(np.mean(therm.T)),
        "sep_low": float(sep_low), "mix_low": float(mix_low),
        "sep_high": float(sep_high), "mix_high": float(mix_high),
        "d_resp_5k": float(sep_low),  # alias for shared-band separation metric
    }

    return edge_rows, metrics_row, Pplus_node
