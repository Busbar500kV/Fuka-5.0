import numpy as np

from fuka5.core.world import World
from fuka5.core.graph import build_graph_from_world, GraphConfig
from fuka5.core.sources import Sources
from fuka5.substrate.edges import initialize_edge_states
from fuka5.substrate.thermal import ThermalParams, ThermalField
from fuka5.substrate.decoders import DecoderBank
from fuka5.substrate.updates import (
    step_epoch, CapsParams, GateParams, BatteryParams, MaturityParams, RewardsParams
)


def tiny_world_cfg():
    return {
        "space": {"dims": [16, 16, 1], "dx": 1.0},
        "rho":   {"init": "gaussian", "theta_outer": 0.2, "theta_core": 0.5, "sigma": 5.0, "center": [8, 8, 0]},
        "material": {"eps_min": 1.0, "eps_gain": 2.0, "g_min": 1e-6, "g_gain": 2e-6, "nonlin": "softplus"},
        "morphogenesis": {"D": 0.05, "eta": 0.002, "gamma_mass": 0.0001, "lambda": 0.001}
    }


def tiny_sources_cfg():
    return {
        "sources": [
            {"name": "s1",  "loc": [6, 7, 0], "tones": [{"f": 5000, "A": 1.0, "phi_deg": 0}]},
            {"name": "s1p", "loc": [10, 8, 0], "tones": [{"f": 5000, "A": 0.9, "phi_deg": 90}]}
        ],
        "coupling": {"base_C": 1e-12, "range": 4.0}
    }


def test_smoke_local_loop_runs():
    # Build tiny world
    world = World.from_config_dict(tiny_world_cfg())
    assert world.rho.shape == (16, 16, 1)

    # Graph
    gcfg = GraphConfig(k_nn=6, dist_cap=3.5, samples_per_voxel=0.05)
    graph = build_graph_from_world(world.rho, world.eps, world.g, world.outer_mask, world.core_mask, world.dx, gcfg)
    assert graph.N > 0 and graph.E > 0

    # Sources + taps
    sources = Sources.from_config_dict(tiny_sources_cfg())
    sources.build_taps_to_graph(graph.positions(), coupling_cfg={"range": 4.0})

    # Substrate state
    caps = CapsParams(Cmin=5e-12, Cmax=2e-10, etaC=1e22, lambdaC=0.05)
    gates = GateParams(eta_g=0.5, lambda_g=0.02, mu_I=0.7, mu_C=0.4)
    batt  = BatteryParams(eta_h=1.0, k_leak=0.03, k_struct=1e-10, B0=1.0, B1=0.3, rho_upkeep=0.05, alpha_reh=5e-4)
    mat   = MaturityParams(alpha_A=0.05, beta_A=0.02, B_thr=0.2, mu_A=0.5, nu_A=0.5)
    therm = ThermalField(graph.N, ThermalParams(Theta=1.0, Lambda=0.1, T_safe=1.2, gamma_T=0.5, E_b=1.0))
    dec   = DecoderBank(n_edges=graph.E, eta_w=0.02)

    edges = initialize_edge_states(graph, Cmin=caps.Cmin, Cmax=caps.Cmax)

    bands_cfg = {"low": [5000], "high": []}
    time_cfg  = {"fs": 80000.0, "window_sec": 0.01}  # very small window for speed
    rewards   = RewardsParams(alpha=1.0, beta=1.0, gamma_cap_cost=1e9)

    # Run a few epochs with ON/OFF
    for ep in range(6):
        on_flag = ep < 3  # 0,1,2 ON; 3,4,5 OFF
        edge_rows, metrics_row, Pplus_node = step_epoch(
            epoch=ep, on_flag=on_flag, graph=graph, edge_states=edges, sources=sources,
            bands_cfg=bands_cfg, time_cfg=time_cfg, caps_params=caps, gate_params=gates,
            batt_params=batt, mat_params=mat, therm=therm, decoders=dec, rewards_params=rewards
        )

        # Basic sanity checks
        assert len(edge_rows) == graph.E
        assert "sumC" in metrics_row and "MSE_mix" in metrics_row
        assert Pplus_node.shape == (graph.N,)
        # C within [Cmin, Cmax], B nonnegative
        for e in edges.values():
            assert e.C >= e.Cmin - 1e-15 and e.C <= e.Cmax + 1e-15
            assert e.B >= -1e-12

    # Post: temperatures & chi sane
    chi = therm.chi()
    assert chi.shape == (graph.N,)
    assert np.all(chi >= 0.0) and np.all(chi <= 1.0)