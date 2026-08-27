"""
Analytical solver for Tandem Chain Multi-Hop Networks under adversarial attacks (Case 3).
"""

import numpy as np
from scipy.stats import gamma


def solve_tandem_theory(
    p: float,
    N: int = 3,
    mu: float = 2.0,
    lambda_arrival: float = 0.15,
    W: float = 8.0,
    max_iter: int = 100,
    tol: float = 1e-5,
) -> float:
    """
    Solves expected end-to-end sojourn time for an N-node tandem chain under multi-hop attacks.

    Parameters
    ----------
    p : float
        Per-node attack probability.
    N : int, default=3
        Number of tandem nodes in sequence.
    mu : float, default=2.0
        Service rate at each node.
    lambda_arrival : float, default=0.15
        External original arrival rate.
    W : float, default=8.0
        End-to-end timeout window.
    max_iter : int, default=100
        Maximum fixed-point iterations.
    tol : float, default=1e-5
        Convergence tolerance.

    Returns
    -------
    float
        Expected end-to-end sojourn time. Returns np.inf if unstable.
    """
    Lambda = [lambda_arrival] * N
    P_succ_overall = 1.0

    for _ in range(max_iter):
        if any(Lambda[j] >= mu * 0.999 for j in range(N)):
            return float(np.inf)

        # Total journey time moments
        E_S_total = sum(1.0 / (mu - Lambda[j]) for j in range(N))
        Var_S_total = sum(1.0 / (mu - Lambda[j]) ** 2 for j in range(N))

        if Var_S_total <= 0:
            return float(np.inf)

        # Gamma approximation for sum of exponentials (hypoexponential)
        k_total = E_S_total**2 / Var_S_total
        theta_total = Var_S_total / E_S_total
        P_succ_overall = ((1.0 - p) ** N) * gamma.cdf(W, a=k_total, scale=theta_total)

        if P_succ_overall <= 1e-7:
            return float(np.inf)

        Lambda_new = [lambda_arrival / P_succ_overall] * N

        if np.isclose(Lambda_new[0], Lambda[0], rtol=tol):
            break
        Lambda = Lambda_new

    if any(Lambda[j] >= mu for j in range(N)) or P_succ_overall <= 1e-7:
        return float(np.inf)

    return float((1.0 / P_succ_overall) * sum(1.0 / (mu - Lambda[j]) for j in range(N)))
