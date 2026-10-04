"""
aqnet: Mathematical Modeling & Discrete-Event Simulation of Multi-Node Queueing Networks Under Adversarial Attacks.
"""

__version__ = "0.1.0"
__author__ = "Ilias Chrysovergis"

from aqnet.models.feedback import solve_feedback_theory
from aqnet.models.feedforward import solve_feedforward_theory
from aqnet.models.one_node import (
    solve_one_node_destruction,
    solve_one_node_modification,
)
from aqnet.models.tandem import solve_tandem_theory
from aqnet.simulation.engine import (
    simulate_feedback,
    simulate_feedforward,
    simulate_one_node_destruction,
    simulate_one_node_modification,
    simulate_tandem,
)

__all__ = [
    "__version__",
    "simulate_feedback",
    "simulate_feedforward",
    "simulate_one_node_destruction",
    "simulate_one_node_modification",
    "simulate_tandem",
    "solve_feedback_theory",
    "solve_feedforward_theory",
    "solve_one_node_destruction",
    "solve_one_node_modification",
    "solve_tandem_theory",
]
