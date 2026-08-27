"""
Analytical solver for N-Node Feedforward Network Topologies under Adversarial Attacks (Case 4).
"""

from typing import Optional, Tuple
import numpy as np
from scipy.special import gammainc


def solve_feedforward_theory(
    N: int = 3,
    mu: float = 2.0,
    lambda_arr: float = 0.15,
    p: float = 0.1,
    W: float = 8.0,
    max_iterations: int = 100,
    tolerance: float = 1e-6,
    damping_factor: float = 0.5,
) -> Tuple[Optional[float], Optional[float], Optional[dict]]:
    """
    Principled solver for N-Node feedforward network accounting for stage-wise retransmission delays.

    Parameters
    ----------
    N : int, default=3
        Number of pipeline nodes.
    mu : float, default=2.0
        Service rate at each node.
    lambda_arr : float, default=0.15
        External fresh arrival rate.
    p : float, default=0.1
        Per-node attack probability.
    W : float, default=8.0
        Timeout threshold.
    max_iterations : int, default=100
        Maximum fixed-point iterations.
    tolerance : float, default=1e-6
        Convergence tolerance.
    damping_factor : float, default=0.5
        Damping factor for fixed-point updates.

    Returns
    -------
    Tuple[Optional[float], Optional[float], Optional[dict]]
        (average_delay, lambda_star_0, details_dict). Returns (None, None, None) if unstable.
    """
    lambda_star_0 = lambda_arr * 1.5

    for _ in range(max_iterations):
        Lambda_star = np.array([lambda_star_0 * ((1.0 - p) ** i) for i in range(N)])

        if any(Lambda_star >= mu * 0.999):
            return None, None, None

        T = 1.0 / (mu - Lambda_star)

        mean_s = float(np.sum(T))
        var_s = float(np.sum(T**2))

        if var_s <= 0 or mean_s <= 0:
            return None, None, None

        theta = var_s / mean_s
        k_shape = (mean_s**2) / var_s

        p_no_timeout = float(gammainc(k_shape, W / theta))
        p_success_single = float(((1.0 - p) ** N) * p_no_timeout)

        if p_success_single <= 1e-6:
            return None, None, None

        new_lambda_star_0 = lambda_arr / p_success_single

        if abs(new_lambda_star_0 - lambda_star_0) < tolerance:
            e_attempts = 1.0 / p_success_single

            p_no_timeout_plus_1 = float(gammainc(k_shape + 1, W / theta))
            e_d_succ = mean_s * p_no_timeout_plus_1 / p_no_timeout if p_no_timeout > 0 else mean_s

            fail_weights = []
            fail_delays = []

            for i in range(N):
                prob = ((1.0 - p) ** i) * p
                delay = float(np.sum(T[: i + 1]))
                fail_weights.append(prob)
                fail_delays.append(delay)

            prob_to = ((1.0 - p) ** N) * (1.0 - p_no_timeout)
            if p_no_timeout < 0.9999 and (1.0 - p_no_timeout) > 0:
                e_d_timeout = mean_s * (1.0 - p_no_timeout_plus_1) / (1.0 - p_no_timeout)
            else:
                e_d_timeout = W + mean_s

            fail_weights.append(prob_to)
            fail_delays.append(e_d_timeout)

            total_fail_prob = sum(fail_weights)
            e_d_fail = (
                sum(w * d for w, d in zip(fail_weights, fail_delays)) / total_fail_prob
                if total_fail_prob > 0
                else 0.0
            )

            average_delay = float((e_attempts - 1.0) * e_d_fail + e_d_succ)

            details = {
                "lambda_star": Lambda_star,
                "node_delays": T,
                "e_attempts": e_attempts,
                "p_success_single": p_success_single,
                "e_d_succ": e_d_succ,
                "e_d_fail": e_d_fail,
            }
            return average_delay, float(lambda_star_0), details

        lambda_star_0 = damping_factor * new_lambda_star_0 + (1.0 - damping_factor) * lambda_star_0

    return None, None, None
