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
