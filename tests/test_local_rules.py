import numpy as np

from fuka5.substrate.edges import EdgeState
from fuka5.substrate.gates import GateParams, update_band_gate
from fuka5.substrate.battery import (
    BatteryParams, MaturityParams,
    battery_update, maturity_update,
    effective_Cmin_lambdaC, budget_gate,
    upkeep_term, rehearsal_amplitude
)


def _edge(C=1e-11, Cmin=5e-12, Cmax=2e-10):
    return EdgeState(eid=0, u=0, v=1, d=1.0, C=C, Cmin=Cmin, Cmax=Cmax)


def test_gates_increase_with_own_power_and_compete():
    gp = GateParams(eta_g=0.5, lambda_g=0.0, mu_I=0.0, mu_C=0.8)
    g0 = np.array([1/3, 1/3, 1/3], dtype=np.float32)

    # Case 1: p1 >> p1p → s1 gate should rise
    g1 = update_band_gate(g0, p1=1.0, p1p=0.0, S=0.0, I=0.0, params=gp)
    assert g1[1] > g0[1]

    # Case 2: p1p >> p1 → s1p gate should rise
    g2 = update_band_gate(g0, p1=0.0, p1p=1.0, S=0.0, I=0.0, params=gp)
    assert g2[2] > g0[2]

    # Case 3: strong interference should depress mix
    gp2 = GateParams(eta_g=0.5, lambda_g=0.0, mu_I=1.0, mu_C=0.0)
    g3 = update_band_gate(g0, p1=0.5, p1p=0.5, S=0.0, I=1.0, params=gp2)
    assert g3[0] < g0[0]

    # Capacity constraint: sum <= 1
    assert float(g1.sum()) <= 1.00001
    assert float(g2.sum()) <= 1.00001
    assert float(g3.sum()) <= 1.00001


def test_battery_rises_with_power_and_leaks_with_costs():
    st = _edge()
    bp = BatteryParams(eta_h=1.0, k_leak=0.0, k_struct=0.0)
    battery_update(st, P_pos=0.5, dC_struct=0.0, dgate_struct=0.0, params=bp)
    assert st.B > 0.0

    # Now apply leak and structural cost
    st2 = _edge()
    bp2 = BatteryParams(eta_h=1.0, k_leak=0.5, k_struct=1.0)
    battery_update(st2, P_pos=0.2, dC_struct=1e-10, dgate_struct=0.1, params=bp2)
    # Still nonnegative but could be small
    assert st2.B >= 0.0


def test_maturity_ratchet_and_effective_bounds():
    st = _edge()
    mp = MaturityParams(alpha_A=0.1, beta_A=0.05, B_thr=0.4, mu_A=0.5, nu_A=0.5)

    # Below threshold → decay A (but clipped at 0)
    st.B = 0.1
    maturity_update(st, mp)
    assert st.A == 0.0

    # Above threshold → grow A
    st.B = 0.8
    maturity_update(st, mp)
    assert st.A > 0.0

    # Effective Cmin should rise with A; lambdaC should fall
    Cmin0, lam0 = 5e-12, 0.05
    Cmin_eff, lam_eff = effective_Cmin_lambdaC(st, Cmin0, lam0, mp)
    assert Cmin_eff >= Cmin0
    assert lam_eff <= lam0


def test_budget_and_upkeep_and_rehearsal_monotonicity():
    st = _edge(C=1.2e-11)
    bp = BatteryParams(B0=1.0, B1=0.5, rho_upkeep=0.1, alpha_reh=1e-3)

    # Budget increases with B
    st.B = 0.2
    b1 = budget_gate(st, bp)
    st.B = 1.2
    b2 = budget_gate(st, bp)
    assert b2 > b1
    assert 0.0 <= b2 <= 1.0

    # Upkeep pulls towards C_star, scaled by chi(T) and battery factor
    st.B = 1.0
    chi = 0.8
    d = upkeep_term(st, C_star=2.0e-11, chi_T=chi, params=bp)
    assert d > 0.0  # pulls upward towards larger C_star

    d2 = upkeep_term(st, C_star=0.5e-11, chi_T=chi, params=bp)
    assert d2 < 0.0  # pulls downward towards smaller C_star

    # Rehearsal amplitude grows with B and A
    st.B = 0.0; st.A = 0.0
    a0 = rehearsal_amplitude(st, bp)
    st.B = 1.0; st.A = 0.0
    a1 = rehearsal_amplitude(st, bp)
    st.B = 1.0; st.A = 1.0
    a2 = rehearsal_amplitude(st, bp)
    assert a1 > a0 >= 0.0
    assert a2 > a1