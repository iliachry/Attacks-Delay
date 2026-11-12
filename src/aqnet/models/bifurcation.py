"""
Saddle-Node (Fold) Retry Bifurcation & Critical Collapse Theory for Adversarial Queueing Networks.

Provides:
1. Unimodal throughput-conservation manifold Gamma(Lambda^*; p, mu, T) across all 5 adversarial
   queueing topologies (one_node_destruction, one_node_modification, tandem, feedforward, feedback),
   with 100% mathematical parity against the corresponding solvers in `aqnet.models`.
2. Closed-form Wright-omega / Lambert-W_0 Saddle-Node fold point Lambda_SN, critical external
   arrival capacity lambda_crit^SN, critical attack probability p_crit^SN, and normal-form
   quadratic curvature c_2 for Case 1 (Destruction, with structural p-invariance of lambda_crit^SN)
   and Case 2 (Modification, with linear capacity erosion (1 - p)).
3. Dual fixed-point branch solver (locally stable lower branch Lambda_stable^* vs. unstable upper
   separatrix Lambda_unstable^* under the mean-field renewal-conservation closure) and universal
   square-root critical exponent O((lambda_crit - lambda)^{-1/2}).
4. Symbolic verification via SymPy of the core bifurcation identities.
5. Discrete-event SimPy validation of subcritical renewal convergence, metastable burst-induced
   separatrix crossing, and supercritical retry collapse with 20-batch Student-t 95% CIs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import simpy
import sympy as sp
from scipy import stats
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammainc, lambertw, wrightomega
from scipy.stats import gamma as scipy_gamma

TopologyName = Literal[
    "one_node_destruction",
    "one_node_modification",
    "tandem",
    "feedforward",
    "feedback",
]

_VALID_TOPOLOGIES = (
    "one_node_destruction",
    "one_node_modification",
    "tandem",
    "feedforward",
    "feedback",
)


def _validate_params(topology: str, p: float, mu: float, T: float = 1.0, N: int = 2) -> None:
    """Validate physical domain and finiteness of queueing parameters."""
    if topology not in _VALID_TOPOLOGIES:
        raise ValueError(f"Unsupported topology '{topology}'.")
    if isinstance(N, bool) or not isinstance(N, (int, np.integer)) or int(N) < 1:
        raise ValueError(f"Node count N must be an integer >= 1, got {N!r}.")
    if not np.isfinite(p) or not (0.0 <= float(p) < 1.0):
        raise ValueError(f"Attack probability p must be finite and satisfy 0 <= p < 1, got {p}.")
    if not np.isfinite(mu) or float(mu) <= 0.0:
        raise ValueError(f"Service rate mu must be finite and positive, got {mu}.")
    if not np.isfinite(T) or float(T) <= 0.0:
        raise ValueError(f"Timeout threshold T must be finite and positive, got {T}.")


def _feedback_visit_series(
    Lambda_star: float,
    p: float,
    mu: float,
    T: float,
    N: int,
    max_k: int = 200,
) -> tuple[float, float, float, float]:
    """
    Evaluate the Case 5 symmetric feedback mesh Erlang-k visit series at node attempt rate Lambda_star.

    Returns (p_succ_attempt, e_visits, e_d_succ, e_d_fail) matching `aqnet.models.feedback.solve_feedback_theory`.
    """
    gamma_rate = mu - Lambda_star
    if gamma_rate <= 1e-12:
        return 0.0, 1.0, float("inf"), float("inf")

    gw = gamma_rate * T
    p_succ = 0.0
    e_v = 0.0
    e_d_succ_num = 0.0
    e_d_fail_num = 0.0

    r_step = (1.0 - p) * (N / (N + 1.0))
    term = 1.0
    f_prev = 1.0
    f_curr = float(gammainc(1, gw))

    for k in range(1, max_k):
        f_next = float(gammainc(k + 1, gw))
        e_v += term * f_prev

        p_succ_k = term * (1.0 - p) * (1.0 / (N + 1.0)) * f_curr
        p_succ += p_succ_k
        e_d_succ_num += p_succ_k * (k / gamma_rate) * (f_next / f_curr if f_curr > 0.0 else 1.0)

        p_attack_k = term * p * f_prev
        e_d_att_k = (1.0 / gamma_rate) + ((k - 1.0) / gamma_rate) * (
            f_curr / f_prev if f_prev > 0.0 else 1.0
        )
        e_d_fail_num += p_attack_k * e_d_att_k

        p_timeout_k = term * (1.0 - p) * max(0.0, f_prev - f_curr)
        e_d_fail_num += p_timeout_k * (T + 1.0 / gamma_rate)

        term *= r_step
        f_prev = f_curr
        f_curr = f_next
        if term < 1e-15:
            break

    e_d_succ = e_d_succ_num / p_succ if p_succ > 0.0 else 0.0
    e_d_fail = e_d_fail_num / max(1.0 - p_succ, 1e-15) if p_succ < 1.0 else 0.0
    return float(p_succ), float(max(e_v, 1e-15)), float(e_d_succ), float(e_d_fail)


def _feedforward_stage_metrics(
    Lambda_0: float,
    p: float,
    mu: float,
    T: float,
    N: int,
) -> tuple[float, float]:
    """
    Evaluate the Case 4 N-node series feedforward network with stage-wise traffic attenuation
    Lambda_i = Lambda_0 * (1 - p)^i (matching `aqnet.models.feedforward.solve_feedforward_theory`).

    Returns (p_succ_single, expected_total_delay).
    """
    if Lambda_0 <= 0.0:
        mean_s0 = N / mu
        var_s0 = N / (mu**2)
        theta0 = var_s0 / mean_s0
        k0 = (mean_s0**2) / var_s0
        p_no_to0 = float(-np.expm1(-mu * T)) if N == 1 else float(gammainc(k0, T / theta0))
        p_succ0 = float(((1.0 - p) ** N) * p_no_to0)
        if p_succ0 <= 0.0:
            return 0.0, float("inf")
        p_no_to_p1 = float(gammainc(k0 + 1.0, T / theta0))
        e_d_succ0 = mean_s0 * p_no_to_p1 / p_no_to0
        fail_w = [((1.0 - p) ** i) * p for i in range(N)]
        fail_d = [(i + 1) / mu for i in range(N)]
        prob_to0 = ((1.0 - p) ** N) * max(0.0, 1.0 - p_no_to0)
        e_d_to0 = (
            mean_s0 * max(0.0, 1.0 - p_no_to_p1) / max(1.0 - p_no_to0, 1e-15)
            if p_no_to0 < 0.999999
            else T + mean_s0
        )
        fail_w.append(prob_to0)
        fail_d.append(e_d_to0)
        tot_fw = sum(fail_w)
        e_d_fail0 = sum(w * d for w, d in zip(fail_w, fail_d)) / tot_fw if tot_fw > 0.0 else 0.0
        e_total0 = (1.0 / p_succ0 - 1.0) * e_d_fail0 + e_d_succ0
        return p_succ0, float(e_total0)

    lam_stages = np.array([Lambda_0 * ((1.0 - p) ** i) for i in range(N)], dtype=float)
    if np.any(lam_stages >= mu):
        return 0.0, float("inf")

    stage_t = 1.0 / (mu - lam_stages)
    mean_s = float(np.sum(stage_t))
    var_s = float(np.sum(stage_t**2))
    if var_s <= 0.0 or mean_s <= 0.0:
        return 0.0, float("inf")

    theta = var_s / mean_s
    k_shape = (mean_s**2) / var_s
    p_no_timeout = float(gammainc(k_shape, T / theta))
    p_succ = float(((1.0 - p) ** N) * max(p_no_timeout, 0.0))
    if p_succ <= 1e-15:
        return 0.0, float("inf")

    p_no_timeout_plus_1 = float(gammainc(k_shape + 1.0, T / theta))
    e_d_succ = mean_s * p_no_timeout_plus_1 / p_no_timeout if p_no_timeout > 0.0 else mean_s

    fail_weights: list[float] = []
    fail_delays: list[float] = []
    for i in range(N):
        fail_weights.append(float(((1.0 - p) ** i) * p))
        fail_delays.append(float(np.sum(stage_t[: i + 1])))

    prob_to = float(((1.0 - p) ** N) * max(0.0, 1.0 - p_no_timeout))
    if p_no_timeout < 0.9999 and (1.0 - p_no_timeout) > 0.0:
        e_d_timeout = mean_s * max(0.0, 1.0 - p_no_timeout_plus_1) / (1.0 - p_no_timeout)
    else:
        e_d_timeout = T + mean_s
    fail_weights.append(prob_to)
    fail_delays.append(float(e_d_timeout))

    total_fail_prob = sum(fail_weights)
    e_d_fail = (
        sum(w * d for w, d in zip(fail_weights, fail_delays)) / total_fail_prob
        if total_fail_prob > 0.0
        else 0.0
    )
    e_attempts = 1.0 / p_succ
    avg_delay = float((e_attempts - 1.0) * e_d_fail + e_d_succ)
    return p_succ, avg_delay


@dataclass(frozen=True)
class SaddleNodeBifurcationResult:
    """Complete Saddle-Node (Fold) bifurcation characterization for an adversarial queue."""

    topology: str
    p: float
    mu: float
    T: float
    N: int
    Lambda_max: float
    Lambda_sn: float
    lambda_crit_sn: float
    rho_sn: float
    p_succ_at_sn: float
    d2Gamma_dLambda2: float
    c2_normal_form: float
    naive_capacity: float
    capacity_ratio_vs_naive: float
    closed_form_exact: bool

    def p_crit_for_lambda(self, lambda_ext: float) -> float:
        """
        Critical attack probability p_crit(lambda_ext) above which no stationary
        fixed point exists for external arrival rate lambda_ext.

        - For Case 1 (Destruction), lambda_crit_sn is strictly invariant to p in [0, 1),
          so if lambda_ext <= lambda_crit_sn the queue remains solvable for any p < 1 (p_crit = 1.0),
          whereas if lambda_ext > lambda_crit_sn no p >= 0 is stable (p_crit = 0.0).
        - For Case 2 (Modification), lambda_crit_sn(p) = (1 - p) * lambda_crit_sn(0),
          yielding the exact linear law p_crit = 1 - lambda_ext / lambda_crit_sn(0).
        - For Case 3 (Tandem), lambda_crit_sn(p) = (1 - p)^N * lambda_crit_sn(0),
          yielding p_crit = 1 - (lambda_ext / lambda_crit_sn(0))^(1/N).
        - For Cases 4 (Feedforward) and 5 (Feedback), p_crit is obtained by bisection on
          the monotonically decreasing function p -> lambda_crit_sn(p).
        """
        if not np.isfinite(lambda_ext):
            raise ValueError(f"lambda_ext must be finite, got {lambda_ext}.")
        if lambda_ext <= 0.0:
            return 1.0
        if self.topology == "one_node_destruction":
            return 1.0 if lambda_ext <= self.lambda_crit_sn else 0.0
        if self.topology == "one_node_modification":
            lam0 = self.lambda_crit_sn / max(1.0 - self.p, 1e-15)
            return float(np.clip(1.0 - lambda_ext / lam0, 0.0, 1.0))
        if self.topology == "tandem":
            lam0 = self.lambda_crit_sn / max((1.0 - self.p) ** self.N, 1e-15)
            if lambda_ext >= lam0:
                return 0.0
            return float(np.clip(1.0 - (lambda_ext / lam0) ** (1.0 / self.N), 0.0, 1.0))

        low, high = 0.0, 0.9999
        f_low = solve_saddle_node_bifurcation(self.topology, low, self.mu, self.T, self.N).lambda_crit_sn
        if lambda_ext >= f_low:
            return 0.0
        for _ in range(42):
            mid = 0.5 * (low + high)
            f_mid = solve_saddle_node_bifurcation(
                self.topology, mid, self.mu, self.T, self.N
            ).lambda_crit_sn
            if f_mid >= lambda_ext:
                low = mid
            else:
                high = mid
        return float(0.5 * (low + high))


@dataclass(frozen=True)
class DualBranchFixedPointResult:
    """Dual fixed-point solution (stable lower branch + unstable upper separatrix)."""

    topology: str
    lambda_ext: float
    p: float
    mu: float
    T: float
    N: int
    is_stable: bool
    Lambda_stable: float
    Lambda_unstable: float
    Lambda_sn: float
    lambda_crit_sn: float
    margin_to_sn: float
    separatrix_gap: float
    sqrt_asymptotic_stable: float
    sqrt_asymptotic_unstable: float
    delay_sensitivity_dLambda_dlambda: float
    expected_total_time_stable: float
    stability_regime: str = "subcritical_bistable"
    is_critical_fold: bool = False


def upper_attempt_rate_bound(topology: str, p: float, mu: float, N: int = 2) -> float:
    """
    Maximum admissible total attempt rate Lambda_max before bottleneck server utilization rho -> 1.
    """
    _validate_params(topology, p, mu, 1.0, N)
    if topology == "one_node_destruction":
        return float(mu / (1.0 - p))
    return float(mu)


def success_probability_p_succ(
    Lambda_star: float,
    topology: str,
    p: float,
    mu: float,
    T: float,
    N: int = 2,
) -> float:
    """
    Effective external throughput yield per unit of bottleneck attempt rate Lambda_star:
      Gamma(Lambda^*; p, mu, T) = Lambda^* * success_probability_p_succ(Lambda^*; p, mu, T).

    - Cases 1–4: per-attempt survival & timely completion probability P_succ(Lambda^*).
    - Case 5 (Feedback Mesh): per-visit exit yield P_succ(Lambda^*) / E[V](Lambda^*) matching Eq. (373).
    """
    _validate_params(topology, p, mu, T, N)
    if not np.isfinite(Lambda_star):
        raise ValueError(f"Lambda_star must be finite, got {Lambda_star}.")
    lam_max = upper_attempt_rate_bound(topology, p, mu, N)
    if Lambda_star >= lam_max:
        return 0.0

    lam_eval = max(0.0, float(Lambda_star))

    if topology == "one_node_destruction":
        lam_eff = (1.0 - p) * lam_eval
        p_time = float(-np.expm1(-(mu - lam_eff) * T))
        return float((1.0 - p) * max(p_time, 0.0))

    if topology == "one_node_modification":
        p_time = float(-np.expm1(-(mu - lam_eval) * T))
        return float((1.0 - p) * max(p_time, 0.0))

    if topology == "tandem":
        rate = mu - lam_eval
        p_time = float(-np.expm1(-rate * T)) if N == 1 else float(scipy_gamma.cdf(T, a=N, scale=1.0 / rate))
        return float(((1.0 - p) ** N) * max(p_time, 0.0))

    if topology == "feedforward":
        p_succ, _ = _feedforward_stage_metrics(lam_eval, p, mu, T, int(N))
        return float(p_succ)

    if topology == "feedback":
        p_succ, e_v, _, _ = _feedback_visit_series(lam_eval, p, mu, T, int(N))
        return float(p_succ / e_v)

    raise ValueError(f"Unsupported topology '{topology}'.")


def throughput_curve_gamma(
    Lambda_star: float,
    topology: str = "one_node_modification",
    p: float = 0.25,
    mu: float = 10.0,
    T: float = 1.0,
    N: int = 2,
) -> float:
    """
    Effective external throughput conservation map:
        Gamma(Lambda^*; p, mu, T) := Lambda^* * success_probability_p_succ(Lambda^*; p, mu, T).

    Under the mean-field renewal-conservation closure, a stationary fixed point at external
    arrival rate lambda exists if and only if Gamma(Lambda^*; p, mu, T) = lambda admits a
    solution in (0, Lambda_max).
    """
    _validate_params(topology, p, mu, T, N)
    if not np.isfinite(Lambda_star) or Lambda_star <= 0.0:
        return 0.0
    lam_max = upper_attempt_rate_bound(topology, p, mu, N)
    if Lambda_star >= lam_max:
        return 0.0
    return float(Lambda_star * success_probability_p_succ(Lambda_star, topology, p, mu, T, N))


def solve_saddle_node_bifurcation(
    topology: str = "one_node_modification",
    p: float = 0.25,
    mu: float = 10.0,
    T: float = 1.0,
    N: int = 2,
) -> SaddleNodeBifurcationResult:
    """
    Compute the exact Saddle-Node (Fold) retry bifurcation point (Lambda_sn, lambda_crit_sn),
    second derivative d^2 Gamma / d(Lambda^*)^2, and quadratic normal-form coefficient c_2.

    Uses the exact closed-form Wright-omega / Lambert-W_0 theorem for:
    - 'one_node_destruction' (with effective server load x = (1 - p)*Lambda^* and p-invariant lambda_crit_sn)
    - 'one_node_modification' (with linear capacity erosion (1 - p))
    and high-precision root/bounded optimization for 'tandem', 'feedforward', and 'feedback'.
    """
    _validate_params(topology, p, mu, T, N)
    lam_max = upper_attempt_rate_bound(topology, p, mu, N)

    if topology == "one_node_destruction":
        # With x = (1 - p)*Lambda^*, Gamma_dest(Lambda^*) = x * (1 - exp(-(mu - x)*T)).
        # Setting dGamma_dest/dx = 0 yields the exact Wright-omega root x_sn = (omega(mu*T + 1) - 1) / T.
        v = float(np.real(wrightomega(mu * T + 1.0)))
        x_sn = (v - 1.0) / T
        lam_sn = x_sn / (1.0 - p)
        lam_crit = ((v - 1.0) ** 2) / (T * v)
        p_succ_sn = lam_crit / lam_sn if lam_sn > 0.0 else 0.0
        d2_gamma = -((1.0 - p) ** 2) * T * (1.0 + v) / v
        c2 = -0.5 * d2_gamma
        naive_cap = float(mu)
        return SaddleNodeBifurcationResult(
            topology=topology,
            p=float(p),
            mu=float(mu),
            T=float(T),
            N=int(N),
            Lambda_max=lam_max,
            Lambda_sn=float(lam_sn),
            lambda_crit_sn=float(lam_crit),
            rho_sn=float(x_sn / mu),
            p_succ_at_sn=float(p_succ_sn),
            d2Gamma_dLambda2=float(d2_gamma),
            c2_normal_form=float(c2),
            naive_capacity=naive_cap,
            capacity_ratio_vs_naive=float(lam_crit / naive_cap),
            closed_form_exact=True,
        )

    if topology == "one_node_modification":
        v = float(np.real(wrightomega(mu * T + 1.0)))
        lam_sn = (v - 1.0) / T
        lam_crit = (1.0 - p) * ((v - 1.0) ** 2) / (T * v)
        p_succ_sn = lam_crit / lam_sn if lam_sn > 0.0 else 0.0
        d2_gamma = -(1.0 - p) * T * (1.0 + v) / v
        c2 = -0.5 * d2_gamma
        naive_cap = float(mu * (1.0 - p))
        return SaddleNodeBifurcationResult(
            topology=topology,
            p=float(p),
            mu=float(mu),
            T=float(T),
            N=int(N),
            Lambda_max=lam_max,
            Lambda_sn=float(lam_sn),
            lambda_crit_sn=float(lam_crit),
            rho_sn=float(lam_sn / mu),
            p_succ_at_sn=float(p_succ_sn),
            d2Gamma_dLambda2=float(d2_gamma),
            c2_normal_form=float(c2),
            naive_capacity=naive_cap,
            capacity_ratio_vs_naive=float(lam_crit / naive_cap),
            closed_form_exact=True,
        )

    # Multi-node topologies: Case 3 (tandem), Case 4 (feedforward), Case 5 (feedback)
    res = minimize_scalar(
        lambda x: -throughput_curve_gamma(float(x), topology, p, mu, T, N),
        bounds=(1e-10 * lam_max, lam_max * (1.0 - 1e-10)),
        method="bounded",
        options={"xatol": 1e-13},
    )
    lam_sn = float(res.x)
    lam_crit = float(-res.fun)
    p_succ_sn = lam_crit / lam_sn if lam_sn > 0.0 else 0.0

    eps = max(1e-4 * min(lam_sn, lam_max - lam_sn), 1e-8)
    g_plus = throughput_curve_gamma(lam_sn + eps, topology, p, mu, T, N)
    g_minus = throughput_curve_gamma(lam_sn - eps, topology, p, mu, T, N)
    d2_gamma = (g_plus - 2.0 * lam_crit + g_minus) / (eps**2)
    c2 = max(-0.5 * d2_gamma, 1e-12)

    if topology in ("tandem", "feedforward"):
        naive_cap = float(mu * ((1.0 - p) ** int(N)))
    else:
        naive_cap = float(mu * (1.0 - p) / (int(N) + 1.0))

    return SaddleNodeBifurcationResult(
        topology=topology,
        p=float(p),
        mu=float(mu),
        T=float(T),
        N=int(N),
        Lambda_max=lam_max,
        Lambda_sn=lam_sn,
        lambda_crit_sn=lam_crit,
        rho_sn=float(lam_sn / mu),
        p_succ_at_sn=float(p_succ_sn),
        d2Gamma_dLambda2=float(d2_gamma),
        c2_normal_form=float(c2),
        naive_capacity=naive_cap,
        capacity_ratio_vs_naive=float(lam_crit / naive_cap),
        closed_form_exact=False,
    )


def _expected_total_sojourn(
    Lambda_star: float,
    topology: str,
    p: float,
    mu: float,
    T: float,
    N: int = 2,
    backoff: float = 0.03,
) -> float:
    """
    Expected total end-to-end sojourn time E[D] at stationary bottleneck attempt rate Lambda_star,
    matching the exact renewal-reward equations of Cases 1–5 in `aqnet.models`.
    """
    p_succ = success_probability_p_succ(Lambda_star, topology, p, mu, T, N)
    if p_succ <= 1e-15:
        return float("inf")

    if topology == "one_node_destruction":
        lam_eff = (1.0 - p) * Lambda_star
        if lam_eff >= mu:
            return float("inf")
        rho_eff = lam_eff / mu
        e_w = rho_eff / (mu * (1.0 - rho_eff))
        e_s = 1.0 / mu
        e_attempt = e_w + (1.0 - p) * e_s
        return float((1.0 / p_succ) * e_attempt + ((1.0 - p_succ) / p_succ) * backoff)

    if topology == "one_node_modification":
        if Lambda_star >= mu:
            return float("inf")
        return float(1.0 / (p_succ * (mu - Lambda_star)))

    if topology == "tandem":
        if Lambda_star >= mu:
            return float("inf")
        return float(N / (p_succ * (mu - Lambda_star)))

    if topology == "feedforward":
        _, avg_delay = _feedforward_stage_metrics(Lambda_star, p, mu, T, int(N))
        return float(avg_delay)

    if topology == "feedback":
        p_succ_att, _, e_d_succ, e_d_fail = _feedback_visit_series(Lambda_star, p, mu, T, int(N))
        if p_succ_att <= 1e-15:
            return float("inf")
        e_attempts = 1.0 / p_succ_att
        return float((e_attempts - 1.0) * e_d_fail + e_d_succ)

    raise ValueError(f"Unsupported topology '{topology}'.")


def solve_dual_fixed_points(
    lambda_ext: float,
    topology: str = "one_node_modification",
    p: float = 0.25,
    mu: float = 10.0,
    T: float = 1.0,
    N: int = 2,
    backoff: float = 0.03,
) -> DualBranchFixedPointResult:
    """
    Solve the conservation equation Gamma(Lambda^*; p, mu, T) = lambda_ext for both:
      1. The locally asymptotically stable lower branch Lambda_stable^* in (0, Lambda_sn), and
      2. The locally unstable upper branch (separatrix) Lambda_unstable^* in (Lambda_sn, Lambda_max).

    Also evaluates the local normal-form square-root approximations:
      Lambda_asymp^-(lambda) = Lambda_sn - sqrt((lambda_crit_sn - lambda) / c_2),
      Lambda_asymp^+(lambda) = Lambda_sn + sqrt((lambda_crit_sn - lambda) / c_2),
    and the critical sensitivity d(Lambda_stable^*)/d(lambda_ext) = 1 / Gamma'(Lambda_stable^*).
    """
    _validate_params(topology, p, mu, T, N)
    if not np.isfinite(lambda_ext) or lambda_ext < 0.0:
        raise ValueError(f"External arrival rate lambda_ext must be finite and >= 0, got {lambda_ext}.")
    sn = solve_saddle_node_bifurcation(topology=topology, p=p, mu=mu, T=T, N=N)

    if lambda_ext == 0.0:
        p0 = max(success_probability_p_succ(0.0, topology, p, mu, T, N), 1e-15)
        return DualBranchFixedPointResult(
            topology=topology,
            lambda_ext=0.0,
            p=float(p),
            mu=float(mu),
            T=float(T),
            N=int(N),
            is_stable=True,
            Lambda_stable=0.0,
            Lambda_unstable=sn.Lambda_max,
            Lambda_sn=sn.Lambda_sn,
            lambda_crit_sn=sn.lambda_crit_sn,
            margin_to_sn=sn.lambda_crit_sn,
            separatrix_gap=sn.Lambda_max,
            sqrt_asymptotic_stable=max(0.0, sn.Lambda_sn - np.sqrt(sn.lambda_crit_sn / sn.c2_normal_form)),
            sqrt_asymptotic_unstable=min(
                sn.Lambda_max, sn.Lambda_sn + np.sqrt(sn.lambda_crit_sn / sn.c2_normal_form)
            ),
            delay_sensitivity_dLambda_dlambda=float(1.0 / p0),
            expected_total_time_stable=float(_expected_total_sojourn(0.0, topology, p, mu, T, N, backoff)),
            stability_regime="subcritical_bistable",
            is_critical_fold=False,
        )

    margin = sn.lambda_crit_sn - lambda_ext
    if margin < -1e-12:
        return DualBranchFixedPointResult(
            topology=topology,
            lambda_ext=float(lambda_ext),
            p=float(p),
            mu=float(mu),
            T=float(T),
            N=int(N),
            is_stable=False,
            Lambda_stable=float("nan"),
            Lambda_unstable=float("nan"),
            Lambda_sn=sn.Lambda_sn,
            lambda_crit_sn=sn.lambda_crit_sn,
            margin_to_sn=float(margin),
            separatrix_gap=0.0,
            sqrt_asymptotic_stable=float("nan"),
            sqrt_asymptotic_unstable=float("nan"),
            delay_sensitivity_dLambda_dlambda=float("inf"),
            expected_total_time_stable=float("inf"),
            stability_regime="supercritical_collapse",
            is_critical_fold=False,
        )

    if abs(margin) <= 1e-12:
        e_total = _expected_total_sojourn(sn.Lambda_sn, topology, p, mu, T, N, backoff)
        return DualBranchFixedPointResult(
            topology=topology,
            lambda_ext=float(lambda_ext),
            p=float(p),
            mu=float(mu),
            T=float(T),
            N=int(N),
            is_stable=False,
            Lambda_stable=sn.Lambda_sn,
            Lambda_unstable=sn.Lambda_sn,
            Lambda_sn=sn.Lambda_sn,
            lambda_crit_sn=sn.lambda_crit_sn,
            margin_to_sn=0.0,
            separatrix_gap=0.0,
            sqrt_asymptotic_stable=sn.Lambda_sn,
            sqrt_asymptotic_unstable=sn.Lambda_sn,
            delay_sensitivity_dLambda_dlambda=float("inf"),
            expected_total_time_stable=float(e_total),
            stability_regime="critical_fold_semistable",
            is_critical_fold=True,
        )

    def residual(lam_star: float) -> float:
        return throughput_curve_gamma(lam_star, topology, p, mu, T, N) - lambda_ext

    lam_stable = float(brentq(residual, 0.0, sn.Lambda_sn, xtol=1e-14, rtol=1e-14))

    # Dynamically bracket the upper root near Lambda_max even for tiny lambda_ext > 0
    upper_hi = sn.Lambda_max
    for scale in (1e-11, 1e-13, 1e-15):
        cand = sn.Lambda_max * (1.0 - scale)
        if cand > sn.Lambda_sn and residual(cand) <= 0.0:
            upper_hi = cand
            break
    lam_unstable = float(brentq(residual, sn.Lambda_sn, upper_hi, xtol=1e-14, rtol=1e-14))

    delta_sqrt = float(np.sqrt(max(margin, 0.0) / sn.c2_normal_form))
    asymp_stable = float(sn.Lambda_sn - delta_sqrt)
    asymp_unstable = float(sn.Lambda_sn + delta_sqrt)

    eps = max(1e-6 * sn.Lambda_max, 1e-8)
    x_lo = max(lam_stable - eps, 0.0)
    x_hi = min(lam_stable + eps, sn.Lambda_sn)
    dgamma_dlam = (
        throughput_curve_gamma(x_hi, topology, p, mu, T, N)
        - throughput_curve_gamma(x_lo, topology, p, mu, T, N)
    ) / max(x_hi - x_lo, 1e-15)
    sensitivity = float(1.0 / max(dgamma_dlam, 1e-15))
    e_total = float(_expected_total_sojourn(lam_stable, topology, p, mu, T, N, backoff))

    return DualBranchFixedPointResult(
        topology=topology,
        lambda_ext=float(lambda_ext),
        p=float(p),
        mu=float(mu),
        T=float(T),
        N=int(N),
        is_stable=True,
        Lambda_stable=lam_stable,
        Lambda_unstable=lam_unstable,
        Lambda_sn=sn.Lambda_sn,
        lambda_crit_sn=sn.lambda_crit_sn,
        margin_to_sn=float(margin),
        separatrix_gap=float(lam_unstable - lam_stable),
        sqrt_asymptotic_stable=asymp_stable,
        sqrt_asymptotic_unstable=asymp_unstable,
        delay_sensitivity_dLambda_dlambda=sensitivity,
        expected_total_time_stable=e_total,
        stability_regime="subcritical_bistable",
        is_critical_fold=False,
    )


def verify_symbolic_bifurcation_proofs() -> dict[str, Any]:
    """
    Symbolically prove the 4 foundational equations of the Saddle-Node Retry Bifurcation Theorem
    using SymPy:
      1. Case 2 (Modification) first-order stationarity identity:
         dGamma_mod/dLambda = 0  <=>  (1 + Lambda*T)*exp(1 + Lambda*T) = exp(mu*T + 1).
      2. Case 2 strict concavity & exact second-derivative curvature at the fold:
         Gamma_mod''(Lambda_sn) = -(1 - p)*T*(1 + v)/v < 0  where v = 1 + Lambda_sn*T.
      3. Case 1 (Destruction) structural p-invariance of the effective throughput map:
         Gamma_dest(Lambda) = x * (1 - exp(-(mu - x)*T))  with x = (1 - p)*Lambda,
         so max_{Lambda} Gamma_dest(Lambda) = max_{x in (0, mu)} G(x) is independent of p,
         and at p = 0 coincides identically with Gamma_mod.
      4. Quadratic Saddle-Node normal form and square-root sensitivity divergence:
         Delta_lambda = c_2 * (delta_Lambda)^2  =>  |d(delta_Lambda)/d(lambda)| = 1 / (2*sqrt(c_2*Delta_lambda)).
    """
    Lam, p, mu, T, v, x = sp.symbols("Lambda p mu T v x", positive=True)
    c2, dlam = sp.symbols("c_2 Delta_lambda", positive=True)

    # 1. Case 2 stationarity equation
    gamma_mod = (1 - p) * Lam * (1 - sp.exp(-(mu - Lam) * T))
    dgamma_mod = sp.diff(gamma_mod, Lam)
    stationarity_sub = sp.simplify((dgamma_mod / (1 - p)).subs(Lam, (v - 1) / T))
    expected_stationarity = 1 - v * sp.exp(v - mu * T - 1)
    stationarity_identity_residual = sp.simplify(stationarity_sub - expected_stationarity)

    # 2. Case 2 second derivative at the fold where exp(-(mu - Lam)*T) = 1 / (1 + Lam*T) = 1 / v
    d2gamma_mod = sp.diff(gamma_mod, Lam, 2)
    d2_at_fold = sp.simplify(
        d2gamma_mod.subs(sp.exp(-(mu - Lam) * T), 1 / (1 + Lam * T)).subs(Lam, (v - 1) / T)
    )
    expected_d2 = -(1 - p) * T * (1 + v) / v
    curvature_residual = sp.simplify(d2_at_fold - expected_d2)

    # 3. Case 1 structural p-invariance via effective load substitution x = (1 - p)*Lam
    #    and exact coincidence with Case 2 at p = 0
    gamma_dest = Lam * (1 - p) * (1 - sp.exp(-(mu - (1 - p) * Lam) * T))
    gamma_dest_in_x = sp.simplify(gamma_dest.subs(Lam, x / (1 - p)))
    dp_invariance = sp.simplify(sp.diff(gamma_dest_in_x, p))
    p0_equivalence_residual = sp.simplify(gamma_dest.subs(p, 0) - gamma_mod.subs(p, 0))

    # 4. Normal form square-root branch splitting and O(Delta_lambda^{-1/2}) sensitivity
    delta_lam_branch = -sp.sqrt(dlam / c2)
    normal_form_residual = sp.simplify(c2 * delta_lam_branch**2 - dlam)
    sensitivity_mag = sp.simplify(sp.diff(-delta_lam_branch, dlam))
    expected_sensitivity = 1 / (2 * sp.sqrt(c2 * dlam))
    sensitivity_residual = sp.simplify(sensitivity_mag - expected_sensitivity)

    all_passed = (
        stationarity_identity_residual == 0
        and curvature_residual == 0
        and dp_invariance == 0
        and p0_equivalence_residual == 0
        and normal_form_residual == 0
        and sensitivity_residual == 0
    )

    w0_val = float(np.real(lambertw(np.exp(11.0))))
    omega_val = float(np.real(wrightomega(11.0)))

    return {
        "stationarity_identity_verified": bool(stationarity_identity_residual == 0),
        "concavity_curvature_verified": bool(curvature_residual == 0),
        "destruction_p_invariance_verified": bool(dp_invariance == 0 and p0_equivalence_residual == 0),
        "sqrt_normal_form_verified": bool(normal_form_residual == 0 and sensitivity_residual == 0),
        "all_symbolic_proofs_verified": bool(all_passed),
        "d2Gamma_at_sn_expr": str(d2_at_fold),
        "gamma_dest_effective_load_expr": str(gamma_dest_in_x),
        "lambertw_vs_wrightomega_abs_diff": abs(w0_val - omega_val),
    }


def simulate_metastable_retry_collapse_simpy(
    lambda_ext: float = 1.5,
    p: float = 0.25,
    mu: float = 10.0,
    T: float = 1.0,
    burst_factor: float = 1.0,
    burst_duration: float = 0.0,
    burst_window: tuple[float, float] | None = None,
    retry_delay_mean: float = 0.02,
    max_inflight_cap: int = 250,
    sim_time: float = 240.0,
    n_batches: int = 20,
    seed: int = 42,
) -> dict[str, Any]:
    """
    Discrete-event SimPy simulation of the Case 2 (Modification) retry queue:
      - In the deep subcritical regime (lambda_ext << lambda_crit_sn) with no separatrix-crossing burst,
        the queue fluctuates around the mean-field lower fixed point Lambda_stable^*.
      - Because the lower branch is only metastable in a finite-backoff stochastic queue, either a
        transient adversarial burst OR a large endogenous queue excursion as lambda_ext -> lambda_crit_sn
        can push the active retry population across the separatrix barrier into runaway timeout
        amplification toward `max_inflight_cap`.
      - When `collapse_triggered` is True or `total_completed == 0`, Little's Law and stationary delay
        comparison diagnostics return `NaN` rather than a spurious zero error.
    """
    rng = np.random.default_rng(seed)
    sn = solve_saddle_node_bifurcation("one_node_modification", p=p, mu=mu, T=T)
    dual = solve_dual_fixed_points(lambda_ext, "one_node_modification", p=p, mu=mu, T=T)

    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)

    warmup = sim_time * 0.2
    batch_len = (sim_time - warmup) / n_batches
    batch_attempt_counts = np.zeros(n_batches, dtype=float)
    batch_sojourn_sums = np.zeros(n_batches, dtype=float)
    batch_completions = np.zeros(n_batches, dtype=float)

    area_inflight = 0.0
    last_event_time = 0.0
    inflight_packets = 0
    peak_inflight = 0
    dropped_packets = 0
    collapse_triggered = False

    def update_area(now: float) -> None:
        nonlocal area_inflight, last_event_time
        if now > warmup:
            t0 = max(last_event_time, warmup)
            if now > t0:
                area_inflight += inflight_packets * (now - t0)
        last_event_time = now

    def packet_lifecycle(t_enter: float) -> Any:
        nonlocal inflight_packets, peak_inflight, dropped_packets, collapse_triggered
        while True:
            peak_inflight = max(peak_inflight, inflight_packets)
            if inflight_packets > max_inflight_cap:
                collapse_triggered = True
                dropped_packets += 1
                update_area(env.now)
                inflight_packets -= 1
                return

            t_attempt_start = env.now
            if t_attempt_start >= warmup:
                b_idx = min(int((t_attempt_start - warmup) / batch_len), n_batches - 1)
                batch_attempt_counts[b_idx] += 1.0

            with server.request() as req:
                yield req
                service_time = float(rng.exponential(1.0 / mu))
                yield env.timeout(service_time)

            attempt_sojourn = env.now - t_attempt_start
            survived_attack = float(rng.random()) >= p
            timely = attempt_sojourn <= T

            if survived_attack and timely:
                now = env.now
                if now >= warmup:
                    b_idx = min(int((now - warmup) / batch_len), n_batches - 1)
                    batch_sojourn_sums[b_idx] += now - t_enter
                    batch_completions[b_idx] += 1.0
                update_area(now)
                inflight_packets -= 1
                return

            if retry_delay_mean > 0.0:
                yield env.timeout(float(rng.exponential(retry_delay_mean)))

    def arrival_generator() -> Any:
        nonlocal inflight_packets, peak_inflight
        while True:
            if burst_window is not None:
                in_burst = burst_window[0] <= env.now <= burst_window[1]
            else:
                in_burst = burst_duration > 0.0 and (warmup <= env.now <= warmup + burst_duration)
            lam_now = lambda_ext * (burst_factor if in_burst else 1.0)
            if lam_now <= 0.0:
                yield env.timeout(1.0)
                continue
            dt = float(rng.exponential(1.0 / lam_now))
            yield env.timeout(dt)
            update_area(env.now)
            inflight_packets += 1
            peak_inflight = max(peak_inflight, inflight_packets)
            env.process(packet_lifecycle(env.now))

    env.process(arrival_generator())
    env.run(until=sim_time)
    update_area(sim_time)

    obs_window = sim_time - warmup
    batch_attempt_rates = batch_attempt_counts / batch_len
    mean_attempt_rate = float(np.mean(batch_attempt_rates))
    se_attempt_rate = float(stats.sem(batch_attempt_rates)) if n_batches > 1 else 0.0
    t_crit = float(stats.t.ppf(0.975, df=max(n_batches - 1, 1)))
    ci_half_attempt = t_crit * se_attempt_rate

    total_completed = float(np.sum(batch_completions))
    mean_total_delay = (
        float(np.sum(batch_sojourn_sums) / total_completed) if total_completed > 0 else float("inf")
    )
    time_avg_inflight = float(area_inflight / obs_window)
    effective_completion_rate = total_completed / obs_window
    littles_law_lhs = time_avg_inflight
    littles_law_rhs = effective_completion_rate * mean_total_delay if total_completed > 0 else float("nan")

    if total_completed > 0 and not collapse_triggered and littles_law_rhs > 0.0:
        littles_law_rel_error = abs(littles_law_lhs - littles_law_rhs) / littles_law_rhs
    else:
        littles_law_rel_error = float("nan")

    if (
        dual.is_stable
        and np.isfinite(dual.expected_total_time_stable)
        and np.isfinite(mean_total_delay)
        and not collapse_triggered
    ):
        e_attempts = dual.Lambda_stable / max(lambda_ext, 1e-12)
        theory_total_with_retry_delay = float(
            dual.expected_total_time_stable + max(e_attempts - 1.0, 0.0) * retry_delay_mean
        )
        theory_vs_sim_delay_rel_error = float(
            abs(mean_total_delay - theory_total_with_retry_delay) / theory_total_with_retry_delay
        )
        theory_vs_sim_attempt_rel_error = float(
            abs(mean_attempt_rate - dual.Lambda_stable) / max(dual.Lambda_stable, 1e-12)
        )
    else:
        theory_total_with_retry_delay = float(dual.expected_total_time_stable)
        theory_vs_sim_delay_rel_error = float("nan")
        theory_vs_sim_attempt_rel_error = float("nan")

    return {
        "lambda_ext": float(lambda_ext),
        "p": float(p),
        "mu": float(mu),
        "T": float(T),
        "lambda_crit_sn": sn.lambda_crit_sn,
        "Lambda_sn": sn.Lambda_sn,
        "Lambda_stable_theory": dual.Lambda_stable,
        "Lambda_unstable_theory": dual.Lambda_unstable,
        "theory_is_stable": dual.is_stable,
        "collapse_triggered": bool(collapse_triggered),
        "dropped_packets": int(dropped_packets),
        "peak_inflight": int(peak_inflight),
        "sim_attempt_rate_mean": mean_attempt_rate,
        "sim_attempt_rate_ci95": (
            mean_attempt_rate - ci_half_attempt,
            mean_attempt_rate + ci_half_attempt,
        ),
        "sim_mean_total_delay": mean_total_delay,
        "theory_mean_total_delay": theory_total_with_retry_delay,
        "theory_vs_sim_delay_rel_error": theory_vs_sim_delay_rel_error,
        "theory_vs_sim_attempt_rel_error": theory_vs_sim_attempt_rel_error,
        "littles_law_L": littles_law_lhs,
        "littles_law_lambda_W": littles_law_rhs,
        "littles_law_rel_error": float(littles_law_rel_error),
    }
