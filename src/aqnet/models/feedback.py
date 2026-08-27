"""
Analytical solver for N-Node Symmetric Feedback Mesh Topologies under Adversarial Attacks (Case 5).
"""

from typing import Optional, Tuple
import numpy as np
from scipy.special import gammainc


def get_erlang_cdf(k: int, rate: float, w: float) -> float:
    """Computes Erlang-k CDF at threshold w."""
    if k <= 0:
        return 1.0
    return float(gammainc(k, rate * w))


def solve_feedback_theory(
    N: int = 5,
    mu: float = 1.0,
    lambda_arr: float = 0.05,
    p: float = 0.1,
    W: float = 50.0,
    max_k: int = 200,
    max_iterations: int = 500,
    damping: float = 0.5,
    tolerance: float = 1e-7,
) -> Tuple[Optional[float], Optional[float], Optional[dict]]:
    """
    Solves expected sojourn time in an N-Node symmetric feedback mesh network under adversarial attacks.

    Parameters
    ----------
    N : int, default=5
        Number of mesh nodes.
    mu : float, default=1.0
        Service rate at each node.
    lambda_arr : float, default=0.05
        External arrival rate per node.
    p : float, default=0.1
        Per-node attack probability.
    W : float, default=50.0
        Timeout threshold.
    max_k : int, default=200
        Maximum path visit truncation horizon.
    max_iterations : int, default=500
        Maximum fixed-point iterations.
    damping : float, default=0.5
        Damping factor for fixed-point updates.
    tolerance : float, default=1e-7
        Convergence tolerance.

    Returns
    -------
    Tuple[Optional[float], Optional[float], Optional[dict]]
        (avg_sojourn_time, lambda_star, details_dict). Returns (None, None, None) if unstable.
    """

    def get_metrics(lambda_star_val: float):
        gamma_rate = mu - lambda_star_val
        if gamma_rate <= 1e-6:
            return 0.0, 0.0, 0.0, 0.0

        p_succ = 0.0
        e_v = 0.0
        e_d_succ_num = 0.0
        e_d_fail_num = 0.0

        for k in range(1, max_k):
            term = ((1.0 - p) ** (k - 1)) * ((N / (N + 1.0)) ** (k - 1))
            f_k_minus_1 = get_erlang_cdf(k - 1, gamma_rate, W)
            f_k = get_erlang_cdf(k, gamma_rate, W)
            f_k_plus_1 = get_erlang_cdf(k + 1, gamma_rate, W)

            # Expected visits reaching the server
            e_v += term * f_k_minus_1

            # Probability of success at visit k
            p_succ_k = term * (1.0 - p) * (1.0 / (N + 1.0)) * f_k
            p_succ += p_succ_k
            e_d_succ_num += p_succ_k * (k / gamma_rate) * (f_k_plus_1 / f_k if f_k > 0 else 1.0)

            # Attack at visit k
            p_attack_k = term * p * f_k_minus_1
            e_d_fail_num += p_attack_k * (k / gamma_rate)

            # Timeout at visit k
            p_timeout_k = term * (1.0 - p) * (f_k_minus_1 - f_k)
            e_d_fail_num += p_timeout_k * (k / gamma_rate)

        return p_succ, e_v, e_d_succ_num, e_d_fail_num

    lambda_star_sol = lambda_arr * (N + 1.0)
    for _ in range(max_iterations):
        if lambda_star_sol >= mu * 0.999:
            return None, None, None

        p_succ, e_v, e_d_succ_num, e_d_fail_num = get_metrics(lambda_star_sol)
        if p_succ <= 1e-6:
            return None, None, None

        next_val = lambda_arr * (e_v / p_succ)
        if abs(next_val - lambda_star_sol) < tolerance:
            lambda_star_sol = next_val
            break
        lambda_star_sol = damping * next_val + (1.0 - damping) * lambda_star_sol
    else:
        return None, None, None

    p_succ, e_v, e_d_succ_num, e_d_fail_num = get_metrics(lambda_star_sol)
    e_attempts = 1.0 / p_succ
    e_d_succ = e_d_succ_num / p_succ if p_succ > 0 else 0.0
    e_d_fail = e_d_fail_num / (1.0 - p_succ) if p_succ < 1.0 else 0.0
    avg_sojourn_time = float((e_attempts - 1.0) * e_d_fail + e_d_succ)

    details = {
        "p_succ": p_succ,
        "e_attempts": e_attempts,
        "e_d_succ": e_d_succ,
        "e_d_fail": e_d_fail,
    }
    return avg_sojourn_time, float(lambda_star_sol), details
