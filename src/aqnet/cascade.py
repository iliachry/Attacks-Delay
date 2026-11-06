"""
Public module alias `aqnet.cascade` re-exporting all closed-form two-station confidence cascade solvers.
"""

from aqnet.models.cascade import (
    CascadeParams,
    apply_hysteresis_with_safety_cap,
    simulate_nonstationary_hysteresis_simpy,
    simulate_stationary_cascade_simpy,
    solve_hysteresis_band,
    solve_nonstationary_envelope,
    solve_threshold_lambda_sensitivity,
    verify_symbolic_cascade_proofs,
    cascade_accuracy,
    conditional_oracle_gain,
    conditional_oracle_gain_prime,
    find_optimal_threshold_tau,
    jensen_antifragility_bound,
    offload_pdf_f,
    offload_pdf_prime,
    offload_rate_q,
    solve_batch_timeout_moments,
    solve_cascade_delay,
    solve_cascade_power,
    solve_pareto_derivative,
)
from aqnet.simulation.engine import simulate_cascade

__all__ = [
    "CascadeParams",
    "apply_hysteresis_with_safety_cap",
    "simulate_nonstationary_hysteresis_simpy",
    "simulate_stationary_cascade_simpy",
    "solve_hysteresis_band",
    "solve_nonstationary_envelope",
    "solve_threshold_lambda_sensitivity",
    "verify_symbolic_cascade_proofs",
    "cascade_accuracy",
    "conditional_oracle_gain",
    "conditional_oracle_gain_prime",
    "find_optimal_threshold_tau",
    "jensen_antifragility_bound",
    "offload_pdf_f",
    "offload_pdf_prime",
    "offload_rate_q",
    "simulate_cascade",
    "solve_batch_timeout_moments",
    "solve_cascade_delay",
    "solve_cascade_power",
    "solve_pareto_derivative",
]
