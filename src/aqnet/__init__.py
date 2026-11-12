"""
aqnet: Mathematical Modeling & Discrete-Event Simulation of Multi-Node Queueing Networks Under Adversarial Attacks.
"""

__version__ = "0.2.0"
__author__ = "Ilias Chrysovergis"

from aqnet import cascade
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
from aqnet.simulation.engine import (
    simulate_cascade,
    simulate_feedback,
    simulate_feedforward,
    simulate_one_node_destruction,
    simulate_one_node_modification,
    simulate_tandem,
)

__all__ = [
    "CascadeParams",
    "DualBranchFixedPointResult",
    "SaddleNodeBifurcationResult",
    "__version__",
    "apply_hysteresis_with_safety_cap",
    "cascade",
    "cascade_accuracy",
    "conditional_oracle_gain",
    "conditional_oracle_gain_prime",
    "find_optimal_threshold_tau",
    "jensen_antifragility_bound",
    "offload_pdf_f",
    "offload_pdf_prime",
    "offload_rate_q",
    "simulate_cascade",
    "simulate_feedback",
    "simulate_feedforward",
    "simulate_metastable_retry_collapse_simpy",
    "simulate_nonstationary_hysteresis_simpy",
    "simulate_one_node_destruction",
    "simulate_one_node_modification",
    "simulate_stationary_cascade_simpy",
    "simulate_tandem",
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
