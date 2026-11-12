"""
Analytical models for adversarial queueing networks.
"""

from aqnet.models.bifurcation import (
    DualBranchFixedPointResult,
    SaddleNodeBifurcationResult,
    simulate_metastable_retry_collapse_simpy,
    solve_dual_fixed_points,
    solve_saddle_node_bifurcation,
    success_probability_p_succ,
    throughput_curve_gamma,
    upper_attempt_rate_bound,
    verify_symbolic_bifurcation_proofs,
)
from aqnet.models.cascade import (
    CascadeParams,
    apply_hysteresis_with_safety_cap,
    cascade_accuracy,
    conditional_oracle_gain,
    conditional_oracle_gain_prime,
    find_optimal_threshold_tau,
    jensen_antifragility_bound,
    offload_pdf_f,
    offload_pdf_prime,
    offload_rate_q,
    simulate_nonstationary_hysteresis_simpy,
    simulate_stationary_cascade_simpy,
    solve_batch_timeout_moments,
    solve_cascade_delay,
    solve_cascade_power,
    solve_hysteresis_band,
    solve_nonstationary_envelope,
    solve_pareto_derivative,
    solve_threshold_lambda_sensitivity,
    verify_symbolic_cascade_proofs,
)
from aqnet.models.feedback import solve_feedback_theory
from aqnet.models.feedforward import solve_feedforward_theory
from aqnet.models.one_node import (
    solve_one_node_destruction,
    solve_one_node_modification,
)
from aqnet.models.tandem import solve_tandem_theory

__all__ = [
    "CascadeParams",
    "DualBranchFixedPointResult",
    "SaddleNodeBifurcationResult",
    "apply_hysteresis_with_safety_cap",
    "cascade_accuracy",
    "conditional_oracle_gain",
    "conditional_oracle_gain_prime",
    "find_optimal_threshold_tau",
    "jensen_antifragility_bound",
    "offload_pdf_f",
    "offload_pdf_prime",
    "offload_rate_q",
    "simulate_metastable_retry_collapse_simpy",
    "simulate_nonstationary_hysteresis_simpy",
    "simulate_stationary_cascade_simpy",
    "solve_batch_timeout_moments",
    "solve_cascade_delay",
    "solve_cascade_power",
    "solve_dual_fixed_points",
    "solve_feedback_theory",
    "solve_feedforward_theory",
    "solve_hysteresis_band",
    "solve_nonstationary_envelope",
    "solve_one_node_destruction",
    "solve_one_node_modification",
    "solve_pareto_derivative",
    "solve_saddle_node_bifurcation",
    "solve_tandem_theory",
    "solve_threshold_lambda_sensitivity",
    "success_probability_p_succ",
    "throughput_curve_gamma",
    "upper_attempt_rate_bound",
    "verify_symbolic_bifurcation_proofs",
    "verify_symbolic_cascade_proofs",
]
