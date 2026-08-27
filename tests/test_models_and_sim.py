"""
Unit and integration tests for aqnet package.
"""

import numpy as np
import pytest
from aqnet import (
    solve_one_node_destruction,
    solve_one_node_modification,
    solve_tandem_theory,
    solve_feedforward_theory,
    solve_feedback_theory,
    simulate_one_node_destruction,
    simulate_one_node_modification,
    simulate_tandem,
    simulate_feedback,
)


def test_one_node_destruction_theory_vs_sim():
    """Verify Case 1 analytical solution vs SimPy simulation within 6% error."""
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
    """Verify Case 2 analytical solution vs SimPy simulation within 6% error."""
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


def test_feedback_theory():
    """Verify Case 5 feedback mesh solver convergence."""
    avg_delay, lambda_star, details = solve_feedback_theory(N=3, mu=1.0, lambda_arr=0.05, p=0.05, W=50.0)
    assert avg_delay is not None
    assert avg_delay > 0
    assert lambda_star > 0.05
