"""
Analytical solvers for single-node queueing under adversarial attacks (Cases 1 & 2).
"""

import numpy as np
from scipy.optimize import brentq
from scipy.special import wrightomega


def solve_one_node_destruction(
    lambda_n: float,
    p: float,
    mu: float = 10.0,
    T: float = 2.0,
    backoff: float = 0.03,
    max_iter: int = 500,
    tol: float = 1e-11,
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
    tol : float, default=1e-11
        Convergence tolerance.

    Returns
    -------
    float
        Expected sojourn time E[D]. Returns np.inf if unstable.
    """
    if p >= 1.0 or p < 0.0 or lambda_n < 0.0 or lambda_n >= mu or T <= 0.0:
        return float(np.inf)
    if lambda_n == 0.0:
        p_succ_0 = (1.0 - p) * (1.0 - float(np.exp(-mu * T)))
        if p_succ_0 <= 0.0:
            return float(np.inf)
        return float((1.0 / p_succ_0) * ((1.0 - p) / mu) + ((1.0 - p_succ_0) / p_succ_0) * backoff)

    # Exact closed-form Wright-omega Saddle-Node fold check (p-invariant critical capacity)
    w_val = float( np.real(wrightomega(mu * T + 1.0)) )
    lambda_crit_sn = ((w_val - 1.0) ** 2) / (T * w_val)
    if lambda_n > lambda_crit_sn:
        return float(np.inf)

    Lambda_sn = (w_val - 1.0) / ((1.0 - p) * T)

    def _residual(lam_star: float) -> float:
        lam_eff = lam_star * (1.0 - p)
        if lam_eff >= mu:
            return -lambda_n
        p_s = (1.0 - p) * (1.0 - float(np.exp(-(mu - lam_eff) * T)))
        return lam_star * p_s - lambda_n

    if abs(lambda_n - lambda_crit_sn) <= 1e-15:
        Lambda_star = Lambda_sn
    elif _residual(Lambda_sn) < -1e-12:
        return float(np.inf)
    else:
        try:
            Lambda_star = float(brentq(_residual, 0.0, Lambda_sn, xtol=1e-14, rtol=1e-14, maxiter=max(max_iter, 200)))
        except ValueError:
            return float(np.inf)

    if abs(_residual(Lambda_star)) > max(tol, 1e-9):
        return float(np.inf)

    # Stability condition: effective server arrival rate must not saturate server capacity
    lambda_eff = Lambda_star * (1.0 - p)
    rho_eff = lambda_eff / mu
    if rho_eff >= 1.0:
        return float(np.inf)

    E_W = rho_eff / (mu * (1.0 - rho_eff))
    E_S = 1.0 / mu

    gamma_rate = mu - lambda_eff
    P_timeout_given_not_attacked = np.exp(-gamma_rate * T)
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
    tol: float = 1e-11,
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
    tol : float, default=1e-11
        Convergence tolerance.

    Returns
    -------
    float
        Expected sojourn time E[D]. Returns np.inf if unstable.
    """
    if p >= 1.0 or p < 0.0 or lambda_n < 0.0 or lambda_n >= mu or T <= 0.0:
        return float(np.inf)
    if lambda_n == 0.0:
        p_succ_0 = (1.0 - p) * (1.0 - float(np.exp(-mu * T)))
        if p_succ_0 <= 0.0:
            return float(np.inf)
        return float(1.0 / (p_succ_0 * mu))

    w_val = float(np.real(wrightomega(mu * T + 1.0)))
    lambda_crit_sn = (1.0 - p) * ((w_val - 1.0) ** 2) / (T * w_val)
    if lambda_n > lambda_crit_sn:
        return float(np.inf)

    Lambda_sn = (w_val - 1.0) / T

    def _residual(lam_star: float) -> float:
        if lam_star >= mu:
            return -lambda_n
        p_s = (1.0 - p) * (1.0 - float(np.exp(-(mu - lam_star) * T)))
        return lam_star * p_s - lambda_n

    if abs(lambda_n - lambda_crit_sn) <= 1e-15:
        Lambda_star = Lambda_sn
    elif _residual(Lambda_sn) < -1e-12:
        return float(np.inf)
    else:
        try:
            Lambda_star = float(brentq(_residual, 0.0, Lambda_sn, xtol=1e-14, rtol=1e-14, maxiter=max(max_iter, 200)))
        except ValueError:
            return float(np.inf)

    if Lambda_star >= mu or abs(_residual(Lambda_star)) > max(tol, 1e-9):
        return float(np.inf)

    P_succ = (1.0 - p) * (1.0 - float(np.exp(-(mu - Lambda_star) * T)))
    if P_succ <= 0.0:
        return float(np.inf)
    E_A = 1.0 / P_succ

    # Renewal theory delay
    return float(E_A * (1.0 / (mu - Lambda_star)))
