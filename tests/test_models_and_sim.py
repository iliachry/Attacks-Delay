"""
Unit and integration tests for aqnet package.
Includes M/M/1 equivalence, edge case, and boundary stability tests.
"""

import sys

import numpy as np
import pytest

from aqnet import (
    simulate_feedback,
    simulate_feedforward,
    simulate_one_node_destruction,
    simulate_one_node_modification,
    simulate_tandem,
    solve_feedback_theory,
    solve_feedforward_theory,
    solve_one_node_destruction,
    solve_one_node_modification,
    solve_tandem_theory,
)
from aqnet.cli import main as cli_main


def test_one_node_destruction_theory_vs_sim():
    """Verify Case 1 analytical solution vs SimPy simulation within 8% error."""
    mu = 10.0
    lambda_n = 2.0
    p = 0.2
    T = 2.0

    theory = solve_one_node_destruction(lambda_n=lambda_n, p=p, mu=mu, T=T)
    assert np.isfinite(theory)
    assert theory > 0

    sim_results = [
        simulate_one_node_destruction(
            lambda_n=lambda_n, p=p, mu=mu, T=T, sim_duration=4000.0, warmup_period=500.0, seed=i
        )
        for i in range(5)
    ]
    sim_mean = np.mean(sim_results)
    rel_error = abs(theory - sim_mean) / theory
    assert rel_error < 0.08, f"Rel error {rel_error:.4f} too high (theory={theory}, sim={sim_mean})"


def test_one_node_modification_theory_vs_sim():
    """Verify Case 2 analytical solution vs SimPy simulation within 8% error."""
    mu = 10.0
    lambda_n = 2.0
    p = 0.2
    T = 2.0

    theory = solve_one_node_modification(lambda_n=lambda_n, p=p, mu=mu, T=T)
    assert np.isfinite(theory)
    assert theory > 0

    sim_results = [
        simulate_one_node_modification(
            lambda_n=lambda_n, p=p, mu=mu, T=T, sim_duration=4000.0, warmup_period=500.0, seed=i
        )
        for i in range(5)
    ]
    sim_mean = np.mean(sim_results)
    rel_error = abs(theory - sim_mean) / theory
    assert rel_error < 0.08, f"Rel error {rel_error:.4f} too high (theory={theory}, sim={sim_mean})"


def test_tandem_chain_theory():
    """Verify Case 3 tandem network solver monotonicity and stability boundaries."""
    del_p01 = solve_tandem_theory(p=0.05, N=3, mu=2.0, lambda_arrival=0.15, W=8.0)
    del_p02 = solve_tandem_theory(p=0.15, N=3, mu=2.0, lambda_arrival=0.15, W=8.0)
    assert np.isfinite(del_p01)
    assert np.isfinite(del_p02)
    assert del_p02 > del_p01, "Delay must increase monotonically with attack probability p"


def test_feedforward_theory():
    """Verify Case 4 feedforward network solver returns valid metrics."""
    avg_delay, lambda_0, details = solve_feedforward_theory(N=3, mu=2.0, lambda_arr=0.15, p=0.05, W=8.0)
    assert avg_delay is not None
    assert avg_delay > 0
    assert lambda_0 > 0.15
    assert isinstance(details, dict)


def test_feedback_theory():
    """Verify Case 5 feedback mesh solver convergence."""
    avg_delay, lambda_star, details = solve_feedback_theory(N=3, mu=1.0, lambda_arr=0.05, p=0.05, W=50.0)
    assert avg_delay is not None
    assert avg_delay > 0
    assert lambda_star > 0.05
    assert isinstance(details, dict)


# --- Edge Case & Boundary Tests ---


def test_mm1_equivalence_destruction():
    """With p=0, backoff=0, and large T, Case 1 must match classical M/M/1 delay E[D] = 1 / (mu - lambda)."""
    mu = 10.0
    lambda_n = 3.0
    expected_mm1 = 1.0 / (mu - lambda_n)  # 1 / 7 = 0.142857...

    theory = solve_one_node_destruction(lambda_n=lambda_n, p=0.0, mu=mu, T=100.0, backoff=0.0)
    assert np.isclose(theory, expected_mm1, rtol=1e-4)

    sim = simulate_one_node_destruction(
        lambda_n=lambda_n, p=0.0, mu=mu, T=100.0, backoff=0.0, sim_duration=10000.0, seed=42
    )
    assert abs(sim - expected_mm1) / expected_mm1 < 0.05


def test_mm1_equivalence_modification():
    """With p=0 and large T, Case 2 must match classical M/M/1 delay E[D] = 1 / (mu - lambda)."""
    mu = 10.0
    lambda_n = 3.0
    expected_mm1 = 1.0 / (mu - lambda_n)

    theory = solve_one_node_modification(lambda_n=lambda_n, p=0.0, mu=mu, T=100.0)
    assert np.isclose(theory, expected_mm1, rtol=1e-4)

    sim = simulate_one_node_modification(
        lambda_n=lambda_n, p=0.0, mu=mu, T=100.0, sim_duration=10000.0, seed=42
    )
    assert abs(sim - expected_mm1) / expected_mm1 < 0.05


def test_mm1_equivalence_tandem():
    """With p=0 and large W, N-node tandem chain delay must equal N * (1 / (mu - lambda))."""
    mu = 5.0
    lambda_arr = 1.0
    N = 3
    expected_tandem = N * (1.0 / (mu - lambda_arr))  # 3 * 0.25 = 0.75

    theory = solve_tandem_theory(p=0.0, N=N, mu=mu, lambda_arrival=lambda_arr, W=200.0)
    assert np.isclose(theory, expected_tandem, rtol=1e-3)


def test_stability_limit_arrival_exceeds_mu():
    """When arrival rate lambda >= mu, solvers should return np.inf or None indicating unstable system."""
    mu = 10.0
    # Case 1 (destruction): lambda_n >= mu
    assert solve_one_node_destruction(lambda_n=10.0, p=0.1, mu=mu) == float("inf")
    assert solve_one_node_destruction(lambda_n=12.0, p=0.1, mu=mu) == float("inf")

    # Case 2 (modification): lambda_n >= mu
    assert solve_one_node_modification(lambda_n=10.0, p=0.1, mu=mu) == float("inf")
    assert solve_one_node_modification(lambda_n=12.0, p=0.1, mu=mu) == float("inf")

    # Case 3 (tandem): lambda >= mu
    assert solve_tandem_theory(p=0.1, N=3, mu=2.0, lambda_arrival=2.0) == float("inf")

    # Case 4 (feedforward): lambda >= mu
    delay_ff, l0_ff, _ = solve_feedforward_theory(N=3, mu=2.0, lambda_arr=2.0, p=0.1)
    assert delay_ff is None and l0_ff is None

    # Case 5 (feedback): lambda high
    delay_fb, l_star_fb, _ = solve_feedback_theory(N=3, mu=1.0, lambda_arr=1.0, p=0.1)
    assert delay_fb is None and l_star_fb is None


def test_boundary_extreme_attack_prob():
    """With p -> 1.0 (100% attack rate), no packet can succeed, system becomes unstable."""
    assert solve_one_node_destruction(lambda_n=1.0, p=0.9999999, mu=10.0) == float("inf")
    assert solve_one_node_modification(lambda_n=1.0, p=1.0, mu=10.0) == float("inf")
    assert solve_tandem_theory(p=1.0, N=3, mu=2.0, lambda_arrival=0.1) == float("inf")

    delay_ff, _, _ = solve_feedforward_theory(N=3, mu=2.0, lambda_arr=0.1, p=1.0)
    assert delay_ff is None

    delay_fb, _, _ = solve_feedback_theory(N=3, mu=1.0, lambda_arr=0.05, p=1.0)
    assert delay_fb is None


def test_boundary_tiny_timeout():
    """With extremely small timeout T/W, packet timeout probability is ~1, resulting in instability."""
    assert solve_one_node_modification(lambda_n=2.0, p=0.1, mu=10.0, T=1e-5) == float("inf")
    assert solve_tandem_theory(p=0.1, N=3, mu=2.0, lambda_arrival=0.1, W=1e-5) == float("inf")

    delay_ff, _, _ = solve_feedforward_theory(N=3, mu=2.0, lambda_arr=0.1, p=0.1, W=1e-5)
    assert delay_ff is None

    delay_fb, _, _ = solve_feedback_theory(N=3, mu=1.0, lambda_arr=0.05, p=0.1, W=1e-5)
    assert delay_fb is None


def test_simulations_all_topologies():
    """Smoke test simulation functions for tandem, feedforward, and feedback networks."""
    # Tandem simulation
    d_tandem = simulate_tandem(
        p=0.05, N=2, mu=5.0, lambda_arrival=0.5, W=10.0, sim_duration=1000.0, warmup_period=100.0, seed=123
    )
    assert np.isfinite(d_tandem) and d_tandem > 0

    # Feedforward simulation wrapper
    d_ff = simulate_feedforward(
        N=2, mu=5.0, lambda_arr=0.5, p=0.05, W=10.0, sim_duration=1000.0, warmup_period=100.0, seed=123
    )
    assert np.isfinite(d_ff) and d_ff > 0

    # Feedback mesh simulation
    d_fb = simulate_feedback(
        N=3, mu=3.0, lambda_arr=0.1, p=0.05, W=20.0, sim_duration=1000.0, warmup_period=100.0, seed=123
    )
    assert np.isfinite(d_fb) and d_fb > 0


def test_cli_subcommands(monkeypatch, capsys):
    """Test CLI commands 'bench' and 'run' for various topologies."""
    # Test bench
    monkeypatch.setattr(sys, "argv", ["aqnet", "bench"])
    cli_main()
    captured = capsys.readouterr()
    assert "Multi-Topology Verification Benchmark" in captured.out

    # Test run one_node_destruction
    monkeypatch.setattr(
        sys, "argv", ["aqnet", "run", "--topology", "one_node_destruction", "--reps", "2", "--sim-duration", "500"]
    )
    cli_main()
    captured = capsys.readouterr()
    assert "one_node_destruction" in captured.out

    # Test run one_node_modification
    monkeypatch.setattr(
        sys, "argv", ["aqnet", "run", "--topology", "one_node_modification", "--reps", "2", "--sim-duration", "500"]
    )
    cli_main()
    captured = capsys.readouterr()
    assert "one_node_modification" in captured.out

    # Test run tandem
    monkeypatch.setattr(
        sys, "argv", ["aqnet", "run", "--topology", "tandem", "--nodes", "2", "--reps", "2", "--sim-duration", "500"]
    )
    cli_main()
    captured = capsys.readouterr()
    assert "tandem" in captured.out

    # Test run feedforward
    monkeypatch.setattr(
        sys, "argv", ["aqnet", "run", "--topology", "feedforward", "--nodes", "2", "--reps", "2", "--sim-duration", "500"]
    )
    cli_main()
    captured = capsys.readouterr()
    assert "feedforward" in captured.out

    # Test run feedback
    monkeypatch.setattr(
        sys, "argv", ["aqnet", "run", "--topology", "feedback", "--nodes", "2", "--mu", "3.0", "--lambda-arr", "0.1", "--W", "20.0", "--reps", "2", "--sim-duration", "500"]
    )
    cli_main()
    captured = capsys.readouterr()
    assert "feedback" in captured.out

    # Test run cascade
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aqnet",
            "run",
            "--topology",
            "cascade",
            "--lambda-arr",
            "300.0",
            "--tau",
            "0.58",
            "--batch-size",
            "4",
            "--batch-timeout-ms",
            "25.0",
            "--reps",
            "2",
            "--sim-duration",
            "2.0",
        ],
    )
    cli_main()
    captured = capsys.readouterr()
    assert "cascade" in captured.out
    assert "Theory (Closed-Form):" in captured.out

    # Test run unstable parameter (e.g. arrival rate >= mu)
    monkeypatch.setattr(
        sys, "argv", ["aqnet", "run", "--topology", "one_node_destruction", "--lambda-arr", "15.0", "--mu", "10.0", "--reps", "1", "--sim-duration", "100"]
    )
    cli_main()
    captured = capsys.readouterr()
    assert "UNSTABLE" in captured.out

    # Test no subcommand (help output + SystemExit)
    monkeypatch.setattr(sys, "argv", ["aqnet"])
    with pytest.raises(SystemExit):
        cli_main()


# --- Two-Station Confidence Cascade (aqnet.cascade) Tests ---


def test_cascade_batch_timeout_boundary_limits():
    """Verify exact closed-form limits of the (b, T) batch accumulator."""
    from aqnet.cascade import solve_batch_timeout_moments

    # 1. b = 1: zero batching delay, E[B_disp] = 1
    m_b1 = solve_batch_timeout_moments(lambda_off=150.0, b=1, T_ms=25.0)
    assert m_b1["E_B_disp"] == 1.0
    assert m_b1["E_W_batch_ms"] == 0.0

    # 2. T -> inf: E[B_disp] = b, E[W_batch] = (b - 1) / (2 * lambda_off)
    lam_off = 120.0
    b = 4
    m_inf = solve_batch_timeout_moments(lambda_off=lam_off, b=b, T_ms=float("inf"))
    expected_w_ms = ((b - 1.0) / (2.0 * lam_off)) * 1000.0
    assert np.isclose(m_inf["E_B_disp"], b)
    assert np.isclose(m_inf["E_W_batch_ms"], expected_w_ms, rtol=1e-6)

    # 3. lambda_off -> 0+: E[B_disp] -> 1, E[W_batch] -> T_ms for b >= 2
    m_zero = solve_batch_timeout_moments(lambda_off=1e-9, b=4, T_ms=25.0)
    assert np.isclose(m_zero["E_B_disp"], 1.0, rtol=1e-5)
    assert np.isclose(m_zero["E_W_batch_ms"], 25.0, rtol=1e-5)


def test_cascade_delay_theory_vs_simpy():
    """Cross-validate closed-form E[D](tau, b, T) and E[W_batch] against SimPy discrete-event simulation."""
    from aqnet.cascade import CascadeParams, simulate_cascade, solve_cascade_delay

    params = CascadeParams(lam=500.0)
    for b in (1, 4, 8):
        theory = solve_cascade_delay(tau=0.55, b=b, T_ms=25.0, params=params, q_override=0.30)
        sim = simulate_cascade(
            tau=0.55,
            b=b,
            T_ms=25.0,
            params=params,
            q_override=0.30,
            sim_duration_s=35.0,
            warmup_s=3.0,
            seed=100 + b,
        )
        rel_err_d = abs(theory["E_D_ms"] - sim["E_D_ms"]) / theory["E_D_ms"]
        assert rel_err_d < 0.025, f"b={b}: E[D] rel error {rel_err_d:.4f} (theory={theory['E_D_ms']}, sim={sim['E_D_ms']})"
        if b > 1:
            rel_err_wb = abs(theory["E_W_batch_ms"] - sim["E_W_batch_ms"]) / theory["E_W_batch_ms"]
            assert rel_err_wb < 0.03, f"b={b}: E[W_batch] rel error {rel_err_wb:.4f}"


def test_cascade_pareto_derivative_and_optimal_tau():
    """Verify analytical Pareto derivatives against central finite differences and optimal threshold solver."""
    from aqnet.cascade import (
        CascadeParams,
        cascade_accuracy,
        find_optimal_threshold_tau,
        solve_cascade_delay,
        solve_cascade_power,
        solve_pareto_derivative,
    )

    params = CascadeParams(lam=500.0)
    tau = 0.58
    eps = 1e-5
    deriv = solve_pareto_derivative(tau=tau, b=4, T_ms=25.0, params=params)

    fd_dA = (cascade_accuracy(tau + eps, params) - cascade_accuracy(tau - eps, params)) / (2.0 * eps)
    fd_dD = (
        solve_cascade_delay(tau + eps, b=4, T_ms=25.0, params=params)["E_D_ms"]
        - solve_cascade_delay(tau - eps, b=4, T_ms=25.0, params=params)["E_D_ms"]
    ) / (2.0 * eps)
    fd_dP = (
        solve_cascade_power(tau + eps, b=4, T_ms=25.0, params=params)["E_P_W"]
        - solve_cascade_power(tau - eps, b=4, T_ms=25.0, params=params)["E_P_W"]
    ) / (2.0 * eps)

    assert np.isclose(deriv["dA_dtau"], fd_dA, rtol=1e-4)
    assert np.isclose(deriv["dED_dtau_ms"], fd_dD, rtol=1e-4)
    assert np.isclose(deriv["dEP_dtau_W"], fd_dP, rtol=1e-4)

    tau_star = find_optimal_threshold_tau(mu_D=0.002, mu_P=0.0005, b=4, T_ms=25.0, params=params)
    assert 0.05 < tau_star < 0.95
    star_deriv = solve_pareto_derivative(tau=tau_star, b=4, T_ms=25.0, params=params, mu_D=0.002, mu_P=0.0005)
    assert abs(star_deriv["dU_dtau"]) < 1e-5


def test_jensen_antifragility_bound():
    """Verify Jensen convexity/antifragility theorem: b=1 is fragile (>0) whereas b in {4, 8} is antifragile (<0)."""
    from aqnet.cascade import (
        CascadeParams,
        jensen_antifragility_bound,
        solve_cascade_delay,
    )

    params = CascadeParams(lam=500.0)
    # 1. Unbatched b=1 is strictly convex / fragile in q
    jb_b1 = jensen_antifragility_bound(tau=0.55, b=1, T_ms=25.0, params=params, q_override=0.30)
    assert jb_b1["d2ED_dq2_ms"] > 0.0
    assert not jb_b1["is_delay_antifragile"]

    # 2. Micro-batched b=4 and b=8 are strictly concave / antifragile in q at q=0.30, T=25ms
    for b in (4, 8):
        jb = jensen_antifragility_bound(
            tau=0.55, b=b, T_ms=25.0, sigma_eps=0.03, delta_max=0.05, params=params, q_override=0.30
        )
        assert jb["C_batch_ms"] < 0.0
        assert jb["d2ED_dq2_ms"] < 0.0
        assert jb["d2EP_dq2_W"] < 0.0
        assert jb["is_delay_antifragile"]
        assert jb["is_power_antifragile"]
        assert jb["jensen_delay_upper_ms"] < 0.0

        # Verify against two-point symmetric perturbation E[0.5 * (D(q+eps) + D(q-eps))] - D(q)
        sigma = 0.03
        d_center = solve_cascade_delay(0.55, b=b, T_ms=25.0, params=params, q_override=0.30)["E_D_ms"]
        d_plus = solve_cascade_delay(0.55, b=b, T_ms=25.0, params=params, q_override=0.30 + sigma)["E_D_ms"]
        d_minus = solve_cascade_delay(0.55, b=b, T_ms=25.0, params=params, q_override=0.30 - sigma)["E_D_ms"]
        empirical_jensen_gap = 0.5 * (d_plus + d_minus) - d_center
        assert empirical_jensen_gap < 0.0
        assert jb["jensen_delay_lower_ms"] <= empirical_jensen_gap <= jb["jensen_delay_upper_ms"]


def test_cascade_stationary_and_nonstationary_hysteresis():
    """Verify cascade renewal-reward moments, Pareto MRS, IFT sensitivity, quartic hysteresis, SymPy, and SimPy."""
    from aqnet.models.cascade_nonstationary import (
        CascadeParams,
        apply_hysteresis_with_safety_cap,
        find_optimal_threshold_tau,
        jensen_antifragility_bound,
        simulate_nonstationary_hysteresis_simpy,
        simulate_stationary_cascade_simpy,
        solve_batch_timeout_moments,
        solve_hysteresis_band,
        solve_nonstationary_envelope,
        solve_pareto_derivative,
        solve_threshold_lambda_sensitivity,
        verify_symbolic_cascade_proofs,
    )

    c_params = CascadeParams(lam=500.0)
    tau_star = find_optimal_threshold_tau(mu_D=0.002, mu_P=0.0005, b=4, T_ms=25.0, params=c_params)
    assert np.isclose(tau_star, 0.80188, atol=1e-5)

    pareto = solve_pareto_derivative(tau_star, b=4, T_ms=25.0, params=c_params)
    assert np.isclose(pareto["q"], 0.62002, atol=1e-5)
    assert abs(pareto["dU_dtau"]) < 1e-6

    bm4 = solve_batch_timeout_moments(0.30 * 500.0, b=4, T_ms=25.0)
    assert np.isclose(bm4["E_B_disp"], 3.5877, atol=1e-4)
    assert np.isclose(bm4["E_W_batch_ms"], 9.1458, atol=1e-4)
    assert np.isclose(bm4["p_timeout"], 0.2771, atol=1e-4)

    j1 = jensen_antifragility_bound(0.55, b=1, T_ms=25.0, params=c_params, q_override=0.30)
    j4 = jensen_antifragility_bound(0.55, b=4, T_ms=25.0, params=c_params, q_override=0.30)
    assert not j1["is_delay_antifragile"]
    assert j4["is_delay_antifragile"]
    assert j4["is_power_antifragile"]

    # Non-stationary sensitivity & hysteresis deadband
    s1 = solve_threshold_lambda_sensitivity(lam=500.0, b=1, params=c_params)
    s4 = solve_threshold_lambda_sensitivity(lam=500.0, b=4, params=c_params)
    assert s1["ift_vs_fd_abs_err"] < 1e-6
    assert s4["ift_vs_fd_abs_err"] < 1e-6
    assert abs(s4["dtau_dlam_per_100fps"]) < abs(s1["dtau_dlam_per_100fps"])

    # Boundary KKT safety probes
    sb_low = solve_threshold_lambda_sensitivity(lam=8.0, b=4, params=c_params)
    sb_high = solve_threshold_lambda_sensitivity(lam=500.0, mu_D=1.0, b=4, params=c_params)
    assert np.isfinite(sb_low["dtau_dlam_per_fps"])
    assert np.isfinite(sb_high["dtau_dlam_per_fps"])

    hb4 = solve_hysteresis_band(lam=500.0, b=4, params=c_params)
    assert 0.01 < hb4["delta_tau_hyst_star"] < 0.08
    assert np.isclose(hb4["delta_tau_bm_reset_star"], hb4["delta_tau_hyst_star"] * (8.0 ** 0.25), atol=1e-12)
    assert hb4["equipartition_residual"] < 1e-12
    assert hb4["bm_equipartition_residual"] < 1e-12
    assert hb4["chattering_reduction_pct"] > 90.0

    # Pointwise ramp probe & safety cap override
    env_ramp = solve_nonstationary_envelope(
        lam_t=500.0, lam_lip=400.0, W_s=0.10, sigma_lam=0.0, b=4, params=c_params, mode="pointwise"
    )
    assert np.isclose(env_ramp["lam_sup_fps"], 540.0, atol=1e-9)
    assert np.isclose(env_ramp["lam_window_avg_sup_fps"], 520.0, atol=1e-9)

    env_edge = solve_nonstationary_envelope(
        lam_t=370.491446, lam_lip=0.0, sigma_lam=0.0, b=1, rho_max=0.85
    )
    assert env_edge["q_max_env"] < 1.0
    assert env_edge["rho_G_envelope_under_burst"] <= 0.85 + 1e-6

    tau_capped, switched = apply_hysteresis_with_safety_cap(0.80, 0.78, 0.79, 0.034)
    assert np.isclose(tau_capped, 0.79, atol=1e-12)
    assert switched is True

    # SymPy symbolic verification
    sym = verify_symbolic_cascade_proofs()
    assert sym["sympy_verified"] is True
    assert sym["theorem2_mrs_delay_residual"] == "0"
    assert sym["proposition3_ift_residual"] == "0"
    assert sym["theorem6_quartic_root_residual"] == "0"
    assert sym["theorem6_equipartition_residual"] == "0"
    assert sym["theorem6_bm_occupation_Ex2"] == "delta**2/6"
    assert sym["theorem6_bm_reset_quartic_root_residual"] == "0"

    # SimPy stationary & non-stationary checks
    sim_stat = simulate_stationary_cascade_simpy(q=0.30, lam=500.0, b=4, T_ms=25.0, duration_s=15.0, warmup_s=3.0)
    assert sim_stat["E_D_rel_err_pct"] < 2.0
    assert sim_stat["littles_law_rel_err"] < 0.01

    sim_ns = simulate_nonstationary_hysteresis_simpy(duration_s=8.0, warmup_s=2.0, b=4, params=c_params, seed=9)
    assert sim_ns["max_thinning_accept_prob"] <= 1.0
    assert sim_ns["empirical_chattering_reduction_pct"] > 80.0


# --- Saddle-Node Retry Bifurcation & Metastability Tests (aqnet v0.2.0) ---


def test_saddle_node_closed_form_wrightomega_and_all_topologies():
    """Verify Wright-omega closed-form fold vs numerical derivative & curvature across all 5 topologies."""
    from aqnet.models.bifurcation import (
        solve_saddle_node_bifurcation,
        throughput_curve_gamma,
        verify_symbolic_bifurcation_proofs,
    )

    proofs = verify_symbolic_bifurcation_proofs()
    assert proofs["all_symbolic_proofs_verified"] is True
    assert proofs["stationarity_identity_verified"] is True
    assert proofs["concavity_curvature_verified"] is True
    assert proofs["destruction_p_invariance_verified"] is True
    assert proofs["sqrt_normal_form_verified"] is True
    assert proofs["lambertw_vs_wrightomega_abs_diff"] < 1e-12

    for topo in (
        "one_node_modification",
        "one_node_destruction",
        "tandem",
        "feedforward",
        "feedback",
    ):
        sn = solve_saddle_node_bifurcation(topology=topo, p=0.25, mu=10.0, T=1.0, N=2)
        assert 0.0 < sn.Lambda_sn < sn.Lambda_max
        assert sn.lambda_crit_sn > 0.0
        assert sn.d2Gamma_dLambda2 < 0.0
        assert sn.c2_normal_form > 0.0
        assert 0.0 < sn.capacity_ratio_vs_naive < 1.0

        # Check first derivative vanishes at Lambda_sn via central differences
        eps = 1e-4
        g_plus = throughput_curve_gamma(sn.Lambda_sn + eps, topo, 0.25, 10.0, 1.0, 2)
        g_minus = throughput_curve_gamma(sn.Lambda_sn - eps, topo, 0.25, 10.0, 1.0, 2)
        g_zero = throughput_curve_gamma(sn.Lambda_sn, topo, 0.25, 10.0, 1.0, 2)
        d1_fd = (g_plus - g_minus) / (2.0 * eps)
        d2_fd = (g_plus - 2.0 * g_zero + g_minus) / (eps**2)
        assert abs(d1_fd) < 1e-6
        assert np.isclose(sn.lambda_crit_sn, g_zero, rtol=1e-9)
        assert np.isclose(sn.d2Gamma_dLambda2, d2_fd, rtol=1e-3)


def test_destruction_structural_p_invariance_vs_modification_linear_decay():
    """Verify Case 1 (Destruction) critical external arrival rate is invariant to p while Case 2 scales as (1-p)."""
    from aqnet.models.bifurcation import solve_saddle_node_bifurcation

    sn_d0 = solve_saddle_node_bifurcation("one_node_destruction", p=0.0, mu=10.0, T=1.0)
    for p_val in (0.10, 0.25, 0.50, 0.75):
        sn_dp = solve_saddle_node_bifurcation("one_node_destruction", p=p_val, mu=10.0, T=1.0)
        assert np.isclose(sn_dp.lambda_crit_sn, sn_d0.lambda_crit_sn, rtol=1e-11)
        assert np.isclose(sn_dp.Lambda_sn * (1.0 - p_val), sn_d0.Lambda_sn, rtol=1e-11)
        assert sn_dp.p_crit_for_lambda(0.5 * sn_d0.lambda_crit_sn) == 1.0
        assert sn_dp.p_crit_for_lambda(1.1 * sn_d0.lambda_crit_sn) == 0.0

    sn_m0 = solve_saddle_node_bifurcation("one_node_modification", p=0.0, mu=10.0, T=1.0)
    for p_val in (0.10, 0.25, 0.50, 0.75):
        sn_mp = solve_saddle_node_bifurcation("one_node_modification", p=p_val, mu=10.0, T=1.0)
        assert np.isclose(sn_mp.Lambda_sn, sn_m0.Lambda_sn, rtol=1e-12)
        assert np.isclose(sn_mp.lambda_crit_sn, (1.0 - p_val) * sn_m0.lambda_crit_sn, rtol=1e-12)
        lam_target = (1.0 - p_val) * sn_m0.lambda_crit_sn
        assert np.isclose(sn_mp.p_crit_for_lambda(lam_target), p_val, atol=1e-9)


def test_dual_fixed_point_branches_and_sqrt_asymptotics():
    """Verify dual fixed-point solver matches lower-branch delay solvers and square-root normal form near fold."""
    from aqnet.models.bifurcation import (
        solve_dual_fixed_points,
        solve_saddle_node_bifurcation,
        throughput_curve_gamma,
    )

    sn = solve_saddle_node_bifurcation("one_node_modification", p=0.25, mu=10.0, T=1.0)
    lam_sub = 2.0
    dual = solve_dual_fixed_points(lam_sub, "one_node_modification", p=0.25, mu=10.0, T=1.0)
    assert dual.is_stable is True
    assert 0.0 < dual.Lambda_stable < sn.Lambda_sn < dual.Lambda_unstable < sn.Lambda_max
    assert np.isclose(
        throughput_curve_gamma(dual.Lambda_stable, "one_node_modification", 0.25, 10.0, 1.0),
        lam_sub,
        rtol=1e-10,
    )
    assert np.isclose(
        throughput_curve_gamma(dual.Lambda_unstable, "one_node_modification", 0.25, 10.0, 1.0),
        lam_sub,
        rtol=1e-10,
    )
    th_mod = solve_one_node_modification(lam_sub, p=0.25, mu=10.0, T=1.0)
    assert np.isclose(dual.expected_total_time_stable, th_mod, rtol=1e-5)

    # Near-fold square-root normal-form accuracy
    lam_near = sn.lambda_crit_sn - 1e-4
    dual_near = solve_dual_fixed_points(lam_near, "one_node_modification", p=0.25, mu=10.0, T=1.0)
    rel_err_sqrt = abs(dual_near.Lambda_stable - dual_near.sqrt_asymptotic_stable) / (
        sn.Lambda_sn - dual_near.Lambda_stable
    )
    assert rel_err_sqrt < 0.01

    # Supercritical regime
    dual_super = solve_dual_fixed_points(
        sn.lambda_crit_sn + 0.5, "one_node_modification", p=0.25, mu=10.0, T=1.0
    )
    assert dual_super.is_stable is False
    assert np.isnan(dual_super.Lambda_stable)
    assert np.isinf(dual_super.expected_total_time_stable)


def test_simpy_metastable_retry_collapse_and_cli_bifurcation(monkeypatch, capsys):
    """Verify SimPy subcritical stability vs supercritical collapse and CLI 'bifurcation' subcommand."""
    from aqnet.models.bifurcation import simulate_metastable_retry_collapse_simpy

    sub = simulate_metastable_retry_collapse_simpy(
        lambda_ext=1.5, p=0.25, mu=10.0, T=2.0, retry_delay_mean=0.5, sim_time=600.0, n_batches=10, seed=42
    )
    assert sub["theory_is_stable"] is True
    assert sub["collapse_triggered"] is False
    assert sub["littles_law_rel_error"] < 0.05
    assert sub["theory_vs_sim_delay_rel_error"] < 0.05
    assert sub["theory_vs_sim_attempt_rel_error"] < 0.05

    # Metastable separatrix escape: identical subcritical lambda_ext=4.2, without burst vs with burst
    meta_no_burst = simulate_metastable_retry_collapse_simpy(
        lambda_ext=4.2,
        p=0.25,
        mu=10.0,
        T=1.0,
        burst_factor=1.0,
        max_inflight_cap=35,
        sim_time=80.0,
        n_batches=5,
        seed=7,
    )
    meta_with_burst = simulate_metastable_retry_collapse_simpy(
        lambda_ext=4.2,
        p=0.25,
        mu=10.0,
        T=1.0,
        burst_factor=4.5,
        burst_window=(10.0, 28.0),
        max_inflight_cap=35,
        sim_time=80.0,
        n_batches=5,
        seed=7,
    )
    assert meta_no_burst["theory_is_stable"] is True
    assert meta_no_burst["collapse_triggered"] is False
    assert meta_with_burst["theory_is_stable"] is True
    assert meta_with_burst["collapse_triggered"] is True
    assert np.isnan(meta_with_burst["littles_law_rel_error"])
    assert np.isnan(meta_with_burst["theory_vs_sim_delay_rel_error"])

    sup = simulate_metastable_retry_collapse_simpy(
        lambda_ext=6.5,
        p=0.25,
        mu=10.0,
        T=1.0,
        max_inflight_cap=40,
        sim_time=50.0,
        n_batches=5,
        seed=42,
    )
    assert sup["theory_is_stable"] is False
    assert sup["collapse_triggered"] is True
    assert np.isnan(sup["littles_law_rel_error"])
    assert np.isnan(sup["theory_vs_sim_delay_rel_error"])

    import json

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aqnet",
            "bifurcation",
            "--topology",
            "one_node_modification",
            "--lambda-arr",
            "2.0",
            "--p",
            "0.25",
            "--mu",
            "10.0",
            "--W",
            "1.0",
            "--json",
        ],
    )
    cli_main()
    out = capsys.readouterr().out
    parsed_sub = json.loads(out)
    assert parsed_sub["Lambda_sn"] > 7.8
    assert parsed_sub["symbolic_proofs_verified"] is True

    # Supercritical JSON output must be strictly valid RFC 8259 JSON (null instead of NaN/Infinity)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "aqnet",
            "bifurcation",
            "--topology",
            "one_node_modification",
            "--lambda-arr",
            "8.0",
            "--p",
            "0.25",
            "--mu",
            "10.0",
            "--W",
            "1.0",
            "--json",
        ],
    )
    cli_main()
    out_sup = capsys.readouterr().out
    parsed_sup = json.loads(
        out_sup,
        parse_constant=lambda c: (_ for _ in ()).throw(ValueError(f"Non-standard JSON constant {c}")),
    )
    assert parsed_sup["is_stable"] is False
    assert parsed_sup["Lambda_stable"] is None
    assert parsed_sup["expected_total_time_stable"] is None


def test_feature_catalog_parity_feat_01_to_08():
    """Verify FEAT-01..FEAT-08 Master Feature Catalog parity across web studio and Python core."""
    from pathlib import Path

    from aqnet.models.bifurcation import (
        solve_dual_fixed_points,
        solve_saddle_node_bifurcation,
        verify_symbolic_bifurcation_proofs,
    )
    from aqnet.models.cascade import find_optimal_threshold_tau, solve_hysteresis_band

    repo_root = Path(__file__).resolve().parent.parent
    html_text = (repo_root / "web" / "index.html").read_text(encoding="utf-8")
    css_text = (repo_root / "web" / "styles.css").read_text(encoding="utf-8")
    js_text = (repo_root / "web" / "app.js").read_text(encoding="utf-8")

    # FEAT-01: Five-topology switcher + cascade
    assert "FEAT-01" in html_text and "topology-select" in html_text
    # FEAT-02: Wright-omega fold & dual-branch solver
    sn_feat02 = solve_saddle_node_bifurcation("one_node_modification", p=0.25, mu=10.0, T=1.0)
    dual_feat02 = solve_dual_fixed_points(3.5, "one_node_modification", p=0.25, mu=10.0, T=1.0)
    assert sn_feat02.lambda_crit_sn > 5.0 and dual_feat02.is_stable
    # FEAT-03: Dual-Canvas Phase Portrait & Critical Capacity Manifold
    assert "bifurcation-canvas" in html_text and "renderManifoldCanvas" in js_text
    # FEAT-04: Law L10 Generative Friction Offering (Adversarial Burst Trace)
    assert "FEAT-04" in html_text and "burst-input" in html_text and "simulateBurstTrajectory" in js_text
    # FEAT-05: Two-Station Edge-Cloud Confidence Cascade & Quartic Hysteresis
    assert find_optimal_threshold_tau() > 0.5 and solve_hysteresis_band()["delta_tau_hyst_star"] > 0.0
    # FEAT-06: SymPy Proof Ledger & LaTeX / BibTeX Citation Modal
    assert "FEAT-06" in html_text and "proof-dialog" in html_text
    assert verify_symbolic_bifurcation_proofs()["all_symbolic_proofs_verified"] is True
    # FEAT-07: Chameleon 3-Theme Polarity Engine & Kanon Regularity
    for theme_id in ("academic-paper", "metatopia-light", "cyber-gold"):
        assert theme_id in html_text and theme_id in css_text
    # FEAT-08: Reproducible CLI & Quality Gate
    assert (repo_root / "src" / "aqnet" / "cli.py").is_file()


def test_cross_solver_parity_and_manuscript_table3_and_web_js():
    """Verify exact cross-solver parity across all 5 topologies, paper.tex Table 3, and web/app.js."""
    import json
    import re
    import subprocess
    from pathlib import Path

    from aqnet.models.bifurcation import (
        solve_dual_fixed_points,
        solve_saddle_node_bifurcation,
    )

    # 1. Finite-timeout p=0, B=0 equivalence between Case 1 and Case 2
    d1_p0 = solve_one_node_destruction(lambda_n=2.5, p=0.0, mu=10.0, T=1.0, backoff=0.0)
    d2_p0 = solve_one_node_modification(lambda_n=2.5, p=0.0, mu=10.0, T=1.0)
    assert np.isclose(d1_p0, d2_p0, rtol=1e-7)

    sn1_p0 = solve_saddle_node_bifurcation("one_node_destruction", p=0.0, mu=10.0, T=1.0)
    sn2_p0 = solve_saddle_node_bifurcation("one_node_modification", p=0.0, mu=10.0, T=1.0)
    assert np.isclose(sn1_p0.Lambda_sn, sn2_p0.Lambda_sn, rtol=1e-12)
    assert np.isclose(sn1_p0.lambda_crit_sn, sn2_p0.lambda_crit_sn, rtol=1e-12)

    # 2. Cross-solver delay parity across all 5 topologies against aqnet.models
    dual_c1 = solve_dual_fixed_points(2.0, "one_node_destruction", p=0.2, mu=10.0, T=2.0, backoff=0.03)
    assert np.isclose(
        dual_c1.expected_total_time_stable,
        solve_one_node_destruction(2.0, p=0.2, mu=10.0, T=2.0, backoff=0.03),
        rtol=1e-12,
        atol=1e-12,
    )

    dual_c2 = solve_dual_fixed_points(2.0, "one_node_modification", p=0.2, mu=10.0, T=2.0)
    assert np.isclose(
        dual_c2.expected_total_time_stable,
        solve_one_node_modification(2.0, p=0.2, mu=10.0, T=2.0),
        rtol=1e-12,
        atol=1e-12,
    )

    # Above-fold convergence guard (1.00001 * lambda_crit_sn must return inf, never a slow Picard residual)
    sn_c1_guard = solve_saddle_node_bifurcation("one_node_destruction", p=0.25, mu=10.0, T=1.0)
    sn_c2_guard = solve_saddle_node_bifurcation("one_node_modification", p=0.25, mu=10.0, T=1.0)
    assert np.isinf(solve_one_node_destruction(sn_c1_guard.lambda_crit_sn * 1.00001, p=0.25, mu=10.0, T=1.0))
    assert np.isinf(solve_one_node_modification(sn_c2_guard.lambda_crit_sn * 1.00001, p=0.25, mu=10.0, T=1.0))

    dual_c3 = solve_dual_fixed_points(0.15, "tandem", p=0.10, mu=2.0, T=8.0, N=3)
    assert np.isclose(
        dual_c3.expected_total_time_stable,
        solve_tandem_theory(p=0.10, N=3, mu=2.0, lambda_arrival=0.15, W=8.0),
        rtol=1e-9,
    )

    dual_c4 = solve_dual_fixed_points(0.15, "feedforward", p=0.10, mu=2.0, T=8.0, N=4)
    d_ff, lam0_ff, _ = solve_feedforward_theory(N=4, mu=2.0, lambda_arr=0.15, p=0.10, W=8.0)
    assert np.isclose(dual_c4.Lambda_stable, lam0_ff, rtol=1e-5)
    assert np.isclose(dual_c4.expected_total_time_stable, d_ff, rtol=1e-5)

    dual_c5 = solve_dual_fixed_points(0.05, "feedback", p=0.10, mu=1.0, T=50.0, N=10)
    d_fb, lam_fb, fb_details = solve_feedback_theory(N=10, mu=1.0, lambda_arr=0.05, p=0.10, W=50.0)
    assert np.isclose(dual_c5.Lambda_stable, lam_fb, rtol=1e-5)
    assert np.isclose(dual_c5.expected_total_time_stable, d_fb, rtol=1e-5)
    # Case 5 exact renewal-visit duration identity: P_succ * E[D_succ] + (1 - P_succ) * E[D_fail] == E[V] / gamma
    gamma_fb = 1.0 - lam_fb
    lhs_visit_time = fb_details["p_succ"] * fb_details["e_d_succ"] + (1.0 - fb_details["p_succ"]) * fb_details["e_d_fail"]
    rhs_visit_time = fb_details["e_v"] / gamma_fb
    assert np.isclose(lhs_visit_time, rhs_visit_time, rtol=1e-12, atol=1e-12)

    # 3. Critical fold semistable regime, tiny-lambda bracketing, small-T stability, and validation guards
    sn_mod = solve_saddle_node_bifurcation("one_node_modification", p=0.25, mu=10.0, T=1.0)
    dual_fold = solve_dual_fixed_points(sn_mod.lambda_crit_sn, "one_node_modification", p=0.25, mu=10.0, T=1.0)
    assert dual_fold.is_critical_fold is True
    assert dual_fold.is_stable is False
    assert dual_fold.stability_regime == "critical_fold_semistable"
    assert np.isfinite(dual_fold.expected_total_time_stable)

    dual_tiny = solve_dual_fixed_points(1e-4, "one_node_modification", p=0.25, mu=10.0, T=1.0)
    assert dual_tiny.is_stable is True
    assert dual_tiny.Lambda_unstable > sn_mod.Lambda_sn

    sn_small_t = solve_saddle_node_bifurcation("one_node_modification", p=0.25, mu=10.0, T=1e-6)
    assert sn_small_t.c2_normal_form > 0.0 and np.isfinite(sn_small_t.lambda_crit_sn)

    for bad_kw in (
        {"p": float("nan")},
        {"mu": float("inf")},
        {"T": -1.0},
        {"N": 2.5},
    ):
        kw = {"topology": "tandem", "p": 0.25, "mu": 10.0, "T": 1.0, "N": 2}
        kw.update(bad_kw)
        with pytest.raises(ValueError):
            solve_saddle_node_bifurcation(**kw)

    # 4. Manuscript Table 3 (tab:bifurcation_summary) digit-for-digit parity against solver
    repo_root = Path(__file__).resolve().parent.parent
    paper_tex = (repo_root / "paper.tex").read_text(encoding="utf-8")
    for topo, row_prefix in (
        ("one_node_destruction", "Case 1: Destruction"),
        ("one_node_modification", "Case 2: Modification"),
        ("tandem", "Case 3: Tandem ($N=2$)"),
        ("feedforward", "Case 4: Feedforward ($N=2$)"),
        ("feedback", "Case 5: Feedback ($N=2$)"),
    ):
        sn = solve_saddle_node_bifurcation(topo, p=0.25, mu=10.0, T=1.0, N=2)
        expected_cols = [
            f"{sn.Lambda_sn:.4f}",
            f"{sn.lambda_crit_sn:.4f}",
            f"{sn.rho_sn:.4f}",
            f"{sn.c2_normal_form:.4f}",
            f"{sn.naive_capacity:.4f}",
            f"{sn.capacity_ratio_vs_naive * 100.0:.2f}\\%",
        ]
        pattern = re.escape(row_prefix) + r".*?" + r"\s*&\s*".join(re.escape(c) for c in expected_cols)
        assert re.search(pattern, paper_tex), f"Table 3 mismatch for {topo}: expected {expected_cols}"

    # 5. Node.js cross-language parity with web/app.js across all 5 topologies + exact Cascade & Hysteresis
    from aqnet.cascade import (
        CascadeParams,
        find_optimal_threshold_tau,
        solve_batch_timeout_moments,
    )
    from aqnet.models.cascade_nonstationary import (
        solve_cascade_delay as solve_cascade_delay_ns,
    )
    from aqnet.models.cascade_nonstationary import (
        solve_hysteresis_band,
    )

    js_probe = """
    const studio = require("./web/app.js");
    const topos = ["one_node_destruction", "one_node_modification", "tandem", "feedforward", "feedback"];
    const out = { topos: {}, batch: [] };
    for (const t of topos) {
      const sn = studio.solveSaddleNodeBifurcation(t, 0.25, 10.0, 1.0, 2);
      const dual = studio.solveDualFixedPoints(1.2, t, 0.25, 10.0, 1.0, 2);
      out.topos[t] = {
        LambdaSn: sn.LambdaSn,
        lambdaCritSn: sn.lambdaCritSn,
        c2: sn.c2NormalForm,
        LambdaStable: dual.LambdaStable,
        LambdaUnstable: dual.LambdaUnstable,
        expectedDelay: dual.expectedDelay
      };
    }
    for (const [lo, b, Tm] of [[1.0, 4, 25.0], [70.5, 4, 25.0], [150.0, 4, 25.0], [150.0, 8, 50.0]]) {
      out.batch.push(studio.solveBatchTimeoutMomentsJS(lo, b, Tm / 1000.0));
    }
    studio.state.lambdaExt = 3.5;
    studio.state.p = 0.25;
    studio.state.mu = 10.0;
    studio.state.N = 2;
    studio.state.T = 1.0;
    const casDefault = studio.solveCascadeSummaryFromState();
    studio.state.lambdaExt = 6.5;
    studio.state.N = 4;
    studio.state.T = 2.0;
    const casMod = studio.solveCascadeSummaryFromState();
    out.casDefault = casDefault;
    out.casMod = casMod;
    out.cascadeChanged = (casDefault.tauOpt !== casMod.tauOpt) &&
                         (casDefault.cBatchB !== casMod.cBatchB) &&
                         (casDefault.deltaTauHyst !== casMod.deltaTauHyst);
    console.log(JSON.stringify(out));
    """
    js_res = json.loads(subprocess.check_output(["node", "-e", js_probe], cwd=str(repo_root), text=True))
    assert js_res["cascadeChanged"] is True
    for topo, jv in js_res["topos"].items():
        py_sn = solve_saddle_node_bifurcation(topo, p=0.25, mu=10.0, T=1.0, N=2)
        py_dual = solve_dual_fixed_points(1.2, topo, p=0.25, mu=10.0, T=1.0, N=2)
        assert np.isclose(jv["LambdaSn"], py_sn.Lambda_sn, rtol=1e-5)
        assert np.isclose(jv["lambdaCritSn"], py_sn.lambda_crit_sn, rtol=1e-5)
        assert np.isclose(jv["expectedDelay"], py_dual.expected_total_time_stable, rtol=1e-5)

    for idx, (lo, b, Tm) in enumerate(((1.0, 4, 25.0), (70.5, 4, 25.0), (150.0, 4, 25.0), (150.0, 8, 50.0))):
        py_bm = solve_batch_timeout_moments(lo, b, Tm)
        js_bm = js_res["batch"][idx]
        assert np.isclose(js_bm["E_B_disp"], py_bm["E_B_disp"], rtol=1e-7)
        assert np.isclose(js_bm["E_W_batch_ms"], py_bm["E_W_batch_ms"], rtol=1e-7)

    cp_def = CascadeParams(lam=500.0)
    py_tau_def = find_optimal_threshold_tau(mu_D=0.002, mu_P=0.0005, b=4, T_ms=25.0, params=cp_def)
    py_del_def = solve_cascade_delay_ns(py_tau_def, b=4, T_ms=25.0, q_override=cp_def.q_of_tau(py_tau_def))
    py_hyst_def = solve_hysteresis_band(lam=500.0, mu_D=0.002, mu_P=0.0005, b=4, T_ms=25.0, params=cp_def)
    assert np.isclose(js_res["casDefault"]["tauOpt"], py_tau_def, rtol=1e-5)
    assert np.isclose(js_res["casDefault"]["cBatchB"], py_del_def["C_batch_ms"], rtol=1e-5)
    assert js_res["casDefault"]["cBatchB"] < 0.0
    assert np.isclose(js_res["casDefault"]["deltaTauHyst"], py_hyst_def["delta_tau_hyst_star"], rtol=1e-5)

    # Verify Station 1 M/G/1 scaling & overload check in web/app.js at lam=800 and lam=1000
    from aqnet.cascade import solve_cascade_delay as solve_cascade_delay_st1

    js_st1_probe = """
    const s = require("./web/app.js");
    const d800 = s.solveCascadeDelayJS(0.30, { lam: 800.0, b: 4, TBatchMs: 25.0 });
    const d1000 = s.solveCascadeDelayJS(0.30, { lam: 1000.0, b: 4, TBatchMs: 25.0 });
    const d1000_ovr = s.solveCascadeDelayJS(0.30, { lam: 1000.0, b: 4, TBatchMs: 25.0, dEdgeMs: 1.8605 });
    console.log(JSON.stringify({
      d800_ED: d800.E_D_ms,
      d800_stable: d800.stable,
      d1000_inf: !Number.isFinite(d1000.E_D_ms),
      d1000_stable: d1000.stable,
      d1000_dED_finite: Number.isFinite(d1000.dED_dq_ms),
      d1000_ovr_stable: d1000_ovr.stable,
      d1000_ovr_inf: !Number.isFinite(d1000_ovr.E_D_ms)
    }));
    """
    js_st1 = json.loads(subprocess.check_output(["node", "-e", js_st1_probe], cwd=str(repo_root), text=True))
    py_d800 = solve_cascade_delay_st1(0.55, b=4, T_ms=25.0, params=CascadeParams(lam=800.0), q_override=0.30)
    py_d1000 = solve_cascade_delay_st1(0.55, b=4, T_ms=25.0, params=CascadeParams(lam=1000.0), q_override=0.30)
    assert js_st1["d800_stable"] is True
    assert np.isclose(js_st1["d800_ED"], py_d800["E_D_ms"], rtol=1e-6)
    assert js_st1["d1000_stable"] is False and py_d1000["stable"] is False
    assert js_st1["d1000_inf"] is True and np.isinf(py_d1000["E_D_ms"])
    assert js_st1["d1000_dED_finite"] is True
    assert js_st1["d1000_ovr_stable"] is False and js_st1["d1000_ovr_inf"] is True

    # Verify full field mapping in _coerce_params from canonical CascadeParams to cascade_nonstationary
    from aqnet.models.cascade_nonstationary import _coerce_params

    cp_custom = CascadeParams(lam=600.0, beta_a=4.0, r_slope=0.1, t_gpu_0_ms=20.0)
    coerced = _coerce_params(cp_custom)
    assert coerced.lam == 600.0
    assert coerced.alpha_g == 4.0
    assert coerced.r_slope == 0.1
    assert coerced.t0_g_ms == 20.0
    hb_custom = solve_hysteresis_band(lam=600.0, params=cp_custom)
    hb_native = solve_hysteresis_band(lam=600.0, params=coerced)
    assert np.isclose(hb_custom["tau_star"], hb_native["tau_star"], atol=1e-12)

    # Verify 5_n_node_feedback/results_feedback.json matches solve_feedback_theory
    fb_json = json.loads((repo_root / "5_n_node_feedback" / "results_feedback.json").read_text(encoding="utf-8"))
    row_n10_p01 = next(r for r in fb_json if r["N"] == 10 and np.isclose(r["p"], 0.1))
    assert np.isclose(row_n10_p01["theory"], d_fb, rtol=1e-9)







