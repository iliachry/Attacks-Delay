"""
Analytical solvers for single-node queueing under adversarial attacks (Cases 1 & 2).
"""

import numpy as np


def solve_one_node_destruction(
    lambda_n: float,
    p: float,
    mu: float = 10.0,
    T: float = 2.0,
    backoff: float = 0.03,
    max_iter: int = 500,
    tol: float = 1e-8,
) -> float:
    """
    Fixed-point solver for single-node M/M/1 delay under pre-service destruction (Case 1).

    Parameters
    ----------
    lambda_n : float
        Normal arrival rate of original packets.
    p : float
        Attack probability (destruction prior to service).
    mu : float, default=10.0
        Mean service rate.
    T : float, default=2.0
        Timeout threshold.
    backoff : float, default=0.03
        Deterministic backoff penalty per retransmission.
    max_iter : int, default=500
        Maximum iterations for fixed-point convergence.
    tol : float, default=1e-8
        Convergence tolerance.

    Returns
    -------
    float
        Expected sojourn time E[D]. Returns np.inf if unstable.
    """
    Lambda_star = lambda_n
    for _ in range(max_iter):
        if Lambda_star >= mu * 0.999:
            return float(np.inf)

        lambda_eff = Lambda_star * (1.0 - p)
        rho_eff = lambda_eff / mu
        if rho_eff >= 1.0:
            return float(np.inf)

        # M/M/1 waiting time in queue based on effective utilization
        E_W = rho_eff / (mu * (1.0 - rho_eff))
        E_S = 1.0 / mu

        # Total sojourn time CDF for M/M/1: P(W_q + S <= T)
        gamma_rate = mu - lambda_eff
        P_complete_in_time = 1.0 - rho_eff * np.exp(-gamma_rate * T)

        # Probability of success per attempt: not attacked AND completes within T
        P_succ = (1.0 - p) * P_complete_in_time

        if P_succ <= 1e-7:
            return float(np.inf)

        E_A = 1.0 / P_succ

        # Fixed point update on total arrival attempts
        Lambda_star_new = lambda_n * E_A

        if np.isclose(Lambda_star_new, Lambda_star, atol=tol):
            break
        Lambda_star = Lambda_star_new

    # Stability condition: total attempt arrival rate must not saturate queue capacity
    if Lambda_star >= mu:
        return float(np.inf)

    # Effective arrival rate after pre-service packet destruction
    lambda_eff = Lambda_star * (1.0 - p)
    rho_eff = lambda_eff / mu
    if rho_eff >= 1.0:
        return float(np.inf)

    E_W = rho_eff / (mu * (1.0 - rho_eff))
    E_S = 1.0 / mu

    gamma_rate = mu - lambda_eff
    P_timeout_given_not_attacked = rho_eff * np.exp(-gamma_rate * T)
    P_succ = (1.0 - p) * (1.0 - P_timeout_given_not_attacked)

    if P_succ <= 1e-7:
        return float(np.inf)

    # Renewal theory sojourn time
    # E[D] = (1 / P_succ) * (E[W] + (1-p)*E[S] + p*0) + ((1-P_succ)/P_succ)*B
    E_sojourn_attempt = E_W + (1.0 - p) * E_S
    E_D = (1.0 / P_succ) * E_sojourn_attempt + ((1.0 - P_succ) / P_succ) * backoff
    return float(E_D)


def solve_one_node_modification(
    lambda_n: float,
    p: float,
    mu: float = 10.0,
    T: float = 2.0,
    max_iter: int = 200,
    tol: float = 1e-7,
) -> float:
    """
    Fixed-point solver for single-node M/M/1 delay under post-service modification (Case 2).

    Parameters
    ----------
    lambda_n : float
        Normal arrival rate of original packets.
    p : float
        Attack probability (payload corruption discovered post-service).
    mu : float, default=10.0
        Mean service rate.
    T : float, default=2.0
        Timeout threshold.
    max_iter : int, default=200
        Maximum iterations for fixed-point convergence.
    tol : float, default=1e-7
        Convergence tolerance.

    Returns
    -------
    float
        Expected sojourn time E[D]. Returns np.inf if unstable.
    """
    Lambda_star = lambda_n
    for _ in range(max_iter):
        if Lambda_star >= mu:
            return float(np.inf)

        # M/M/1 delay for one attempt
        E_S = 1.0 / (mu - Lambda_star)

        # Probability of success in one attempt: No attack and No timeout
        P_succ = (1.0 - p) * (1.0 - np.exp(-T / E_S))

        if P_succ <= 0:
            return float(np.inf)

        # Expected number of attempts
        E_A = 1.0 / P_succ

        # New effective traffic rate
        Lambda_star_new = lambda_n * E_A

        if np.isclose(Lambda_star_new, Lambda_star, atol=tol):
            break
        Lambda_star = Lambda_star_new

    if Lambda_star >= mu:
        return float(np.inf)

    # Renewal theory delay
    return float(E_A * (1.0 / (mu - Lambda_star)))
