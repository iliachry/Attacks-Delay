"""
Analytical models for adversarial queueing networks.
"""

from aqnet.models.cascade import (
    CascadeParams,
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
from aqnet.models.feedback import solve_feedback_theory
from aqnet.models.feedforward import solve_feedforward_theory
from aqnet.models.one_node import (
    solve_one_node_destruction,
    solve_one_node_modification,
)
from aqnet.models.tandem import solve_tandem_theory

__all__ = [
    "CascadeParams",
    "cascade_accuracy",
    "conditional_oracle_gain",
    "conditional_oracle_gain_prime",
    "find_optimal_threshold_tau",
    "jensen_antifragility_bound",
    "offload_pdf_f",
    "offload_pdf_prime",
    "offload_rate_q",
    "solve_batch_timeout_moments",
    "solve_cascade_delay",
    "solve_cascade_power",
    "solve_pareto_derivative",
    "solve_feedback_theory",
    "solve_feedforward_theory",
    "solve_one_node_destruction",
    "solve_one_node_modification",
    "solve_tandem_theory",
]
