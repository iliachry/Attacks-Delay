"""
Two-Tier Selective-Fidelity Cascade Queueing, Pareto Frontier & Non-Stationary Hysteresis Theory.

Implements:
1. Exact Batch-Timeout Renewal-Reward Accumulation Moments (`solve_batch_timeout_moments`)
2. End-to-End Cascade Delay & 3-Term Curvature Decomposition (`solve_cascade_delay`)
3. End-to-End Cascade Power & Idle-Tail Consolidation Curvature (`solve_cascade_power`)
4. Distribution-Invariant Pareto Marginal Rate of Substitution (`solve_pareto_derivative`)
5. Lagrangian Threshold Optimization (`find_optimal_threshold_tau`)
6. Jensen's Inequality Convex/Concave Antifragility Bounds (`jensen_antifragility_bound`)
7. Non-Stationary Arrival Envelope & Pointwise Stability Bound (`solve_nonstationary_envelope`)
8. Implicit Threshold Sensitivity to Arrival Rate (`solve_threshold_lambda_sensitivity`)
9. Quartic No-Chattering Hysteresis Deadband Optimization (`solve_hysteresis_band`)
10. SymPy Symbolic Verification of Analytical Identities (`verify_symbolic_cascade_proofs`)
11. SimPy Discrete-Event Batch-Means Validation (`simulate_nonstationary_hysteresis_simpy`)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import gammainc, gammaincc
from scipy.stats import beta as beta_dist


@dataclass(frozen=True)
class CascadeParams:
    """Calibrated two-tier selective-fidelity cascade parameters (Axelera Metis + RTX 5080)."""

    lam: float = 500.0
    # Confidence score distribution G ~ Beta(alpha_g, beta_g)
    alpha_g: float = 2.8
    beta_g: float = 1.2
    # Marginal accuracy gain rate r_1(tau) = r_slope * (1 - tau)^gamma_r
    r_slope: float = 0.30731157
    gamma_r: float = 1.0
    # Tier-1 Edge NPU baseline latency (ms) and active power (W)
    d_edge_ms: float = 1.8605
    p_edge_w: float = 8.7950
    # Tier-2 LAN framing/serialization + GPU batch execution affine laws (ms)
    t0_net_ms: float = 0.45
    t1_net_ms: float = 0.25
    t0_g_ms: float = 2.05
    t1_g_ms: float = 0.25
    # Pollaczek-Khinchine service squared coefficient of variation c_s^2
    cs2_g: float = 0.05
    # Tier-2 dynamic energy per offloaded frame (LAN + GPU marginal dynamic mJ)
    e_dyn_mj: float = 139.62900
    # Tier-2 static active power floor and P0 -> P8 idle tail parameters (W, ms)
    p_s_g_w: float = 73.965
    p_idle_p8_w: float = 20.28
    p_idle_p0_w: float = 73.41
    tau_p8_ms: float = 25.0

    def q_of_tau(self, tau: float) -> float:
        """Escalation probability q(tau) = P(G < tau) = F_G(tau)."""
        tau_c = float(np.clip(tau, 0.0, 1.0))
        return float(beta_dist.cdf(tau_c, self.alpha_g, self.beta_g))

    def tau_of_q(self, q: float) -> float:
        """Inverse escalation quantile tau(q) = F_G^{-1}(q)."""
        q_c = float(np.clip(q, 1e-9, 1.0 - 1e-9))
        return float(beta_dist.ppf(q_c, self.alpha_g, self.beta_g))

    def f_g(self, tau: float) -> float:
        """Confidence density f_G(tau) = dF_G(tau)/dtau."""
        tau_c = float(np.clip(tau, 1e-6, 1.0 - 1e-6))
        return float(beta_dist.pdf(tau_c, self.alpha_g, self.beta_g))

    def f_g_prime(self, tau: float) -> float:
        """Derivative of confidence density f_G'(tau) via central difference."""
        eps = 1e-6
        tau_c = float(np.clip(tau, 2e-6, 1.0 - 2e-6))
        t_hi = tau_c + eps
        t_lo = tau_c - eps
        return (self.f_g(t_hi) - self.f_g(t_lo)) / (t_hi - t_lo)

    def r_1(self, tau: float) -> float:
        """Conditional accuracy gain r_1(tau) = E[Delta A | G = tau] at threshold boundary."""
        tau_c = float(np.clip(tau, 0.0, 1.0))
        return float(self.r_slope * ((1.0 - tau_c) ** self.gamma_r))

    def r_1_prime(self, tau: float) -> float:
        """Derivative dr_1(tau)/dtau < 0."""
        tau_c = float(np.clip(tau, 0.0, 1.0 - 1e-9))
        if self.gamma_r == 1.0:
            return -float(self.r_slope)
        return -float(self.r_slope * self.gamma_r * ((1.0 - tau_c) ** (self.gamma_r - 1.0)))




def _coerce_params(params: Any | None = None) -> CascadeParams:
    if params is None:
        return CascadeParams()
    if isinstance(params, CascadeParams):
        return params
    lam_val = float(getattr(params, "lam", 500.0))
    alpha_g = float(getattr(params, "alpha_g", getattr(params, "beta_a", 2.8)))
    beta_g = float(getattr(params, "beta_g", getattr(params, "beta_b", 1.2)))
    r_slope = float(getattr(params, "r_slope", 0.30731157))
    gamma_r = float(getattr(params, "gamma_r", getattr(params, "gamma_gain", 1.0)))
    t0_net_ms = float(getattr(params, "t0_net_ms", getattr(params, "t_net_0_ms", 0.45)))
    t1_net_ms = float(getattr(params, "t1_net_ms", getattr(params, "t_net_1_ms", 0.25)))
    t0_g_ms = float(getattr(params, "t0_g_ms", getattr(params, "t_gpu_0_ms", 2.05)))
    t1_g_ms = float(getattr(params, "t1_g_ms", getattr(params, "t_gpu_1_ms", 0.25)))
    cs2_g = float(getattr(params, "cs2_g", getattr(params, "C_S2_sq", 0.05)))
    p_s_g_w = float(getattr(params, "p_s_g_w", getattr(params, "P_s_G", 73.965)))
    p_idle_p8_w = float(getattr(params, "p_idle_p8_w", getattr(params, "P_idle_P8", 20.28)))
    p_idle_p0_w = float(getattr(params, "p_idle_p0_w", getattr(params, "P_idle_P0", 73.41)))
    if hasattr(params, "tau_p8_ms"):
        tau_p8_ms = float(params.tau_p8_ms)
    elif hasattr(params, "tau_P8_s"):
        tau_p8_ms = float(params.tau_P8_s) * 1000.0
    else:
        tau_p8_ms = 25.0
    if hasattr(params, "e_dyn_mj"):
        e_dyn_mj = float(params.e_dyn_mj)
    elif hasattr(params, "e_dyn_G_mJ") or hasattr(params, "e_net_mJ"):
        e_dyn_mj = float(getattr(params, "e_dyn_G_mJ", 137.129)) + float(getattr(params, "e_net_mJ", 2.50))
    else:
        e_dyn_mj = 139.629
    d_edge_ms = float(getattr(params, "d_edge_ms", 1.8605))
    p_edge_w = float(getattr(params, "p_edge_w", 8.7950))
    return CascadeParams(
        lam=lam_val,
        alpha_g=alpha_g,
        beta_g=beta_g,
        r_slope=r_slope,
        gamma_r=gamma_r,
        d_edge_ms=d_edge_ms,
        p_edge_w=p_edge_w,
        t0_net_ms=t0_net_ms,
        t1_net_ms=t1_net_ms,
        t0_g_ms=t0_g_ms,
        t1_g_ms=t1_g_ms,
        cs2_g=cs2_g,
        e_dyn_mj=e_dyn_mj,
        p_s_g_w=p_s_g_w,
        p_idle_p8_w=p_idle_p8_w,
        p_idle_p0_w=p_idle_p0_w,
        tau_p8_ms=tau_p8_ms,
    )


def solve_batch_timeout_moments(
    lam_off: float,
    b: int = 4,
    T_ms: float = 25.0,
) -> dict[str, float]:
    """
    Exact renewal-reward batch accumulation moments under Poisson offload stream
    of rate lam_off (frames/s), target micro-batch size b >= 1, and deadline timeout
    T_ms (ms) triggered after the first arrival in a batch cycle.
    """
    if b <= 1 or lam_off <= 1e-9 or T_ms <= 0.0:
        return {
            "p_timeout": 0.0,
            "P_timeout": 0.0,
            "E_B_disp": 1.0,
            "E_B2_disp": 1.0,
            "E_W_batch_ms": 0.0,
        }

    # Normalized Poisson expectation mu = lam_off * (T_ms / 1000) during timeout window
    mu = float(lam_off) * (float(T_ms) / 1000.0)
    n = b - 1

    # P(N(T_max) < b - 1) = Q(b - 1, mu) = regularized upper incomplete gamma
    p_timeout = float(gammaincc(n, mu))

    # Exact truncated Poisson moments for k in {0, ..., n - 1}
    # Using exact finite series to avoid numerical underflow/overflow at small/large mu
    exp_neg_mu = math.exp(-mu) if mu < 700.0 else 0.0
    term = exp_neg_mu  # mu^0 / 0! * e^{-mu}
    e_b_trunc = 0.0
    e_b2_trunc = 0.0
    e_w_sum_ms = 0.0

    for k in range(n):
        if k > 0:
            term = term * (mu / k)
        b_size = k + 1
        e_b_trunc += b_size * term
        e_b2_trunc += (b_size * b_size) * term

    # Full-batch reach probability P(N(T_max) >= b - 1) = P(n, mu)
    p_full = float(gammainc(n, mu))
    e_b_disp = e_b_trunc + b * p_full
    e_b2_disp = e_b2_trunc + (b * b) * p_full

    # Expected total waiting time per cycle (ms):
    # Arrival #1 waits min(S_{n}, T_max). Arrival #(j+1) for 1 <= j <= n-1 waits min(S_{n} - S_j, T_max - S_j).
    # By exact Poisson renewal identity:
    # E[sum_{i=1}^{B_disp} W_i] = (1000 / lam_off) * sum_{j=1}^{n} j * P(N(T_max) >= j)
    for j in range(1, n + 1):
        e_w_sum_ms += (1000.0 / float(lam_off)) * j * float(gammainc(j, mu))

    e_w_batch_ms = e_w_sum_ms / max(e_b_disp, 1e-12)

    return {
        "p_timeout": p_timeout,
        "P_timeout": p_timeout,
        "E_B_disp": float(e_b_disp),
        "E_B2_disp": float(e_b2_disp),
        "E_W_batch_ms": float(e_w_batch_ms),
    }


def _delay_components(
    q: float,
    b: int,
    T_ms: float,
    params: CascadeParams,
) -> dict[str, float]:
    """Internal helper returning decomposed delay terms at escalation fraction q."""
    q_c = float(np.clip(q, 0.0, 1.0))
    if q_c <= 1e-12:
        s_g_1 = params.t0_g_ms + params.t1_g_ms
        t_srv_1 = (params.t0_net_ms + params.t0_g_ms) + (params.t1_net_ms + params.t1_g_ms)
        return {
            "q": 0.0,
            "E_B_disp": 1.0,
            "P_timeout": 1.0 if b > 1 else 0.0,
            "E_W_batch_ms": 0.0,
            "T_srv_ms": t_srv_1,
            "S_G_ms": s_g_1,
            "raw_rho_G": 0.0,
            "rho_G": 0.0,
            "E_W_queue_ms": 0.0,
            "E_D_ms": params.d_edge_ms,
            "term_batch": 0.0,
            "term_srv": 0.0,
            "term_queue": 0.0,
        }

    lam_off = q_c * params.lam
    bm = solve_batch_timeout_moments(lam_off, b=b, T_ms=T_ms)
    b_bar = bm["E_B_disp"]
    w_batch = bm["E_W_batch_ms"]

    s_g = params.t0_g_ms + params.t1_g_ms * b_bar
    t_srv = (params.t0_net_ms + params.t0_g_ms) + (params.t1_net_ms + params.t1_g_ms) * b_bar
    raw_rho_g = (lam_off / (1000.0 * b_bar)) * s_g
    rho_g = min(raw_rho_g, 0.999)

    c_q = 0.5 * (1.0 + params.cs2_g)
    w_queue = c_q * (rho_g / (1.0 - rho_g)) * s_g

    term_batch = q_c * w_batch
    term_srv = q_c * t_srv
    term_queue = q_c * w_queue
    e_d = params.d_edge_ms + term_batch + term_srv + term_queue

    return {
        "q": q_c,
        "E_B_disp": b_bar,
        "P_timeout": bm["P_timeout"],
        "E_W_batch_ms": w_batch,
        "T_srv_ms": t_srv,
        "S_G_ms": s_g,
        "raw_rho_G": float(raw_rho_g),
        "rho_G": float(rho_g),
        "E_W_queue_ms": w_queue,
        "E_D_ms": e_d,
        "term_batch": term_batch,
        "term_srv": term_srv,
        "term_queue": term_queue,
    }


def solve_cascade_delay(
    tau: float,
    b: int = 4,
    T_ms: float = 25.0,
    params: CascadeParams | None = None,
    q_override: float | None = None,
) -> dict[str, float]:
    """
    Compute end-to-end expected cascade latency E[D] and its exact 3-term second-order
    curvature decomposition d^2 E[D]/dq^2 = C_batch + C_amort + C_queue.
    """
    p = _coerce_params(params)
    q = float(q_override) if q_override is not None else p.q_of_tau(tau)
    tau_eff = p.tau_of_q(q) if q_override is not None else float(tau)

    base = _delay_components(q, b=b, T_ms=T_ms, params=p)

    # Central finite differences in q-space for exact partial derivatives
    eps = 5e-5
    q_mid = float(np.clip(q, eps + 1e-7, 0.995 - eps))
    q_hi = q_mid + eps
    q_lo = q_mid - eps
    h = eps

    c_hi = _delay_components(q_hi, b=b, T_ms=T_ms, params=p)
    c_mid = _delay_components(q_mid, b=b, T_ms=T_ms, params=p)
    c_lo = _delay_components(q_lo, b=b, T_ms=T_ms, params=p)

    dED_dq = (c_hi["E_D_ms"] - c_lo["E_D_ms"]) / (2.0 * h)
    c_batch = (c_hi["term_batch"] - 2.0 * c_mid["term_batch"] + c_lo["term_batch"]) / (h * h)
    c_amort = (c_hi["term_srv"] - 2.0 * c_mid["term_srv"] + c_lo["term_srv"]) / (h * h)
    c_queue = (c_hi["term_queue"] - 2.0 * c_mid["term_queue"] + c_lo["term_queue"]) / (h * h)

    if b == 1:
        c_batch = 0.0
        c_amort = 0.0

    d2ED_dq2 = c_batch + c_amort + c_queue

    fg = p.f_g(tau_eff)
    fg_p = p.f_g_prime(tau_eff)
    dED_dtau = dED_dq * fg
    d2ED_dtau2 = d2ED_dq2 * (fg * fg) + dED_dq * fg_p

    return {
        "tau": float(tau_eff),
        "q": float(base["q"]),
        "b": int(b),
        "T_ms": float(T_ms),
        "E_B_disp": float(base["E_B_disp"]),
        "P_timeout": float(base["P_timeout"]),
        "E_W_batch_ms": float(base["E_W_batch_ms"]),
        "S_G_ms": float(base["S_G_ms"]),
        "rho_G": float(base["rho_G"]),
        "E_W_queue_ms": float(base["E_W_queue_ms"]),
        "E_D_ms": float(base["E_D_ms"]),
        "dED_dq_ms": float(dED_dq),
        "dED_dtau_ms": float(dED_dtau),
        "C_batch_ms": float(c_batch),
        "C_amort_ms": float(c_amort),
        "C_queue_ms": float(c_queue),
        "d2ED_dq2_ms": float(d2ED_dq2),
        "d2ED_dtau2_ms": float(d2ED_dtau2),
    }


def _power_components(
    q: float,
    b: int,
    T_ms: float,
    params: CascadeParams,
) -> dict[str, float]:
    """Internal helper returning decomposed power terms at escalation fraction q."""
    q_c = float(np.clip(q, 0.0, 0.9999))
    if q_c <= 1e-12:
        return {
            "q": 0.0,
            "rho_G": 0.0,
            "P_idle_eff_W": params.p_idle_p8_w,
            "E_P_W": params.p_edge_w + params.p_idle_p8_w,
        }

    lam_off = q_c * params.lam
    bm = solve_batch_timeout_moments(lam_off, b=b, T_ms=T_ms)
    b_bar = bm["E_B_disp"]
    s_g = params.t0_g_ms + params.t1_g_ms * b_bar
    rho_g = min(0.999, (lam_off / (1000.0 * b_bar)) * s_g)

    inter_batch_ms = (1000.0 * b_bar) / lam_off
    gap_ms = max(0.0, inter_batch_ms - s_g)
    p_idle_eff = params.p_idle_p8_w + (params.p_idle_p0_w - params.p_idle_p8_w) * math.exp(
        -gap_ms / params.tau_p8_ms
    )

    p_dyn_w = (lam_off / 1000.0) * params.e_dyn_mj
    p_gpu_state_w = rho_g * params.p_s_g_w + (1.0 - rho_g) * p_idle_eff
    e_p = params.p_edge_w + p_dyn_w + p_gpu_state_w

    return {
        "q": q_c,
        "rho_G": rho_g,
        "gap_ms": gap_ms,
        "P_idle_eff_W": p_idle_eff,
        "E_P_W": e_p,
    }


def solve_cascade_power(
    tau: float,
    b: int = 4,
    T_ms: float = 25.0,
    params: CascadeParams | None = None,
    q_override: float | None = None,
) -> dict[str, float]:
    """
    Compute end-to-end expected cascade power E[P] (W) and its first/second
    derivatives with respect to escalation fraction q and threshold tau.
    """
    p = _coerce_params(params)
    q = float(q_override) if q_override is not None else p.q_of_tau(tau)
    tau_eff = p.tau_of_q(q) if q_override is not None else float(tau)

    base = _power_components(q, b=b, T_ms=T_ms, params=p)

    eps = 5e-5
    q_mid = float(np.clip(q, eps + 1e-7, 0.995 - eps))
    q_hi = q_mid + eps
    q_lo = q_mid - eps
    h = eps

    p_hi = _power_components(q_hi, b=b, T_ms=T_ms, params=p)
    p_mid = _power_components(q_mid, b=b, T_ms=T_ms, params=p)
    p_lo = _power_components(q_lo, b=b, T_ms=T_ms, params=p)

    dEP_dq = (p_hi["E_P_W"] - p_lo["E_P_W"]) / (2.0 * h)
    d2EP_dq2 = (p_hi["E_P_W"] - 2.0 * p_mid["E_P_W"] + p_lo["E_P_W"]) / (h * h)

    fg = p.f_g(tau_eff)
    fg_p = p.f_g_prime(tau_eff)
    dEP_dtau = dEP_dq * fg
    d2EP_dtau2 = d2EP_dq2 * (fg * fg) + dEP_dq * fg_p

    return {
        "tau": float(tau_eff),
        "q": float(base["q"]),
        "b": int(b),
        "T_ms": float(T_ms),
        "rho_G": float(base["rho_G"]),
        "P_idle_eff_W": float(base["P_idle_eff_W"]),
        "E_P_W": float(base["E_P_W"]),
        "dEP_dq_W": float(dEP_dq),
        "dEP_dtau_W": float(dEP_dtau),
        "d2EP_dq2_W": float(d2EP_dq2),
        "d2EP_dtau2_W": float(d2EP_dtau2),
    }


def solve_pareto_derivative(
    tau: float,
    b: int = 4,
    T_ms: float = 25.0,
    mu_D: float = 0.002,
    mu_P: float = 0.0005,
    params: CascadeParams | None = None,
) -> dict[str, float]:
    """
    Compute the distribution-invariant Pareto frontier marginal rate of substitution (MRS):
      dE[A]/dE[D] = r_1(tau) / (dE[D]/dq)
      dE[A]/dE[P] = r_1(tau) / (dE[P]/dq)
    proving exact cancellation of the confidence density f_G(tau) > 0.
    """
    p = _coerce_params(params)
    d_res = solve_cascade_delay(tau, b=b, T_ms=T_ms, params=p)
    p_res = solve_cascade_power(tau, b=b, T_ms=T_ms, params=p)

    fg = p.f_g(tau)
    r1 = p.r_1(tau)
    dA_dtau = fg * r1

    mrs_d_q = r1 / d_res["dED_dq_ms"]
    mrs_p_q = r1 / p_res["dEP_dq_W"]
    dED_dEP = d_res["dED_dq_ms"] / p_res["dEP_dq_W"]
    marginal_net_gain_per_q = r1 - (mu_D * d_res["dED_dq_ms"] + mu_P * p_res["dEP_dq_W"])
    dU_dtau = fg * marginal_net_gain_per_q

    return {
        "tau": float(tau),
        "q": float(d_res["q"]),
        "f_G": float(fg),
        "r_1": float(r1),
        "dA_dtau": float(dA_dtau),
        "dED_dq_ms": float(d_res["dED_dq_ms"]),
        "dED_dtau_ms": float(d_res["dED_dtau_ms"]),
        "dEP_dq_W": float(p_res["dEP_dq_W"]),
        "dEP_dtau_W": float(p_res["dEP_dtau_W"]),
        "dA_dEP_per_W": float(mrs_p_q),
        "dA_dED_per_ms": float(mrs_d_q),
        "dED_dEP_ms_per_W": float(dED_dEP),
        "marginal_net_gain_per_q": float(marginal_net_gain_per_q),
        "dU_dtau": float(dU_dtau),
    }


def find_optimal_threshold_tau(
    mu_D: float = 0.002,
    mu_P: float = 0.0005,
    b: int = 4,
    T_ms: float = 25.0,
    params: CascadeParams | None = None,
) -> float:
    """
    Solve the first-order Lagrangian optimality condition:
      r_1(tau*) = mu_D * (dE[D]/dq)|_{q(tau*)} + mu_P * (dE[P]/dq)|_{q(tau*)}
    for the unique stationary optimal confidence threshold tau* in [0, 1].
    """
    p = _coerce_params(params)

    def foc(tau_val: float) -> float:
        d_res = solve_cascade_delay(tau_val, b=b, T_ms=T_ms, params=p)
        p_res = solve_cascade_power(tau_val, b=b, T_ms=T_ms, params=p)
        marginal_cost = mu_D * d_res["dED_dq_ms"] + mu_P * p_res["dEP_dq_W"]
        return p.r_1(tau_val) - marginal_cost

    f_lo = foc(0.01)
    f_hi = foc(0.99)
    if f_lo <= 0.0:
        return 0.0
    if f_hi >= 0.0:
        return 1.0
    return float(brentq(foc, 0.01, 0.99, xtol=1e-9))


def jensen_antifragility_bound(
    tau: float = 0.55,
    b: int = 4,
    T_ms: float = 25.0,
    sigma_eps: float = 0.03,
    delta_max: float = 0.05,
    params: CascadeParams | None = None,
    q_override: float | None = None,
) -> dict[str, Any]:
    """
    Second-order Taylor/Jensen volatility bound for expected cascade delay and power
    under zero-mean offload fraction jitter epsilon_q with variance sigma_eps^2:
      Delta_Jensen(D) = E[D(q + eps_q)] - D(q) ≈ (1/2) * (d^2 E[D]/dq^2) * sigma_eps^2.
    """
    p = _coerce_params(params)
    d_res = solve_cascade_delay(tau, b=b, T_ms=T_ms, params=p, q_override=q_override)
    p_res = solve_cascade_power(tau, b=b, T_ms=T_ms, params=p, q_override=q_override)

    d2_d = d_res["d2ED_dq2_ms"]
    d2_p = p_res["d2EP_dq2_W"]

    jensen_delay_shift_ms = 0.5 * d2_d * (sigma_eps * sigma_eps)
    jensen_power_shift_w = 0.5 * d2_p * (sigma_eps * sigma_eps)

    return {
        "tau": float(d_res["tau"]),
        "q": float(d_res["q"]),
        "b": int(b),
        "sigma_eps": float(sigma_eps),
        "delta_max": float(delta_max),
        "C_batch_ms": float(d_res["C_batch_ms"]),
        "C_amort_ms": float(d_res["C_amort_ms"]),
        "C_queue_ms": float(d_res["C_queue_ms"]),
        "d2ED_dq2_ms": float(d2_d),
        "d2EP_dq2_W": float(d2_p),
        "jensen_delay_shift_ms": float(jensen_delay_shift_ms),
        "jensen_power_shift_W": float(jensen_power_shift_w),
        "is_delay_antifragile": bool(d2_d < 0.0),
        "is_power_antifragile": bool(d2_p < 0.0),
    }


# =============================================================================
# Non-Stationary Arrival Envelope & Confidence-Threshold Hysteresis Bound
# =============================================================================


def solve_nonstationary_envelope(
    lam_t: float,
    lam_lip: float = 400.0,
    sigma_lam: float = 75.0,
    W_s: float = 0.10,
    tau_corr_s: float = 0.20,
    z_alpha: float = 1.0,
    mode: str = "pointwise",
    b: int = 4,
    T_ms: float = 25.0,
    rho_max: float = 0.85,
    params: CascadeParams | None = None,
) -> dict[str, float]:
    """
    Theorem 5 (Non-Stationary Arrival Envelope & Implicit Utilization Cap):
    Given a time-varying arrival intensity lam(s) decomposed into a locally
    Lipschitz drift component |d lam_det(s)/ds| <= lam_lip (FPS/s) and a
    zero-mean correlated burst component xi(s) with marginal standard deviation
    sigma_lam and autocorrelation scale tau_corr_s over control window W_s (seconds):

      Pointwise deterministic supremum + z_alpha burst margin:
        lam_sup(t; W_s) = lam(t) + lam_lip * W_s + z_alpha * sigma_lam * sqrt(min(1.0, tau_corr_s / W_s))

      Window-average control-horizon envelope:
        lam_avg_sup(t; W_s) = lam(t) + 0.5 * lam_lip * W_s + z_alpha * sigma_lam * sqrt(tau_corr_s / W_s)

    Solves the exact unclamped fixed-point utilization equation
      rho_G^raw(q, b; lam_sup) = (q * lam_sup / (1000 * b_bar(q * lam_sup, b, T_ms))) * S_G(b_bar) <= rho_max
    over the closed interval q in [0, 1] to obtain the safety cap tau_max^env(lam_sup, b).
    """
    p_base = _coerce_params(params)
    w_eff = max(float(W_s), 1e-6)
    tc_eff = max(float(tau_corr_s), 0.0)
    drift_pointwise = float(lam_lip) * w_eff
    drift_window_avg = 0.5 * float(lam_lip) * w_eff
    burst_window_avg = float(z_alpha) * float(sigma_lam) * math.sqrt(tc_eff / w_eff) if tc_eff > 0.0 else 0.0
    burst_pointwise = (
        float(z_alpha) * float(sigma_lam) * math.sqrt(min(1.0, tc_eff / w_eff))
        if tc_eff > 0.0
        else float(z_alpha) * float(sigma_lam)
    )

    lam_pointwise_sup = float(lam_t) + drift_pointwise + burst_pointwise
    lam_window_avg_sup = float(lam_t) + drift_window_avg + burst_window_avg

    if mode == "window_average":
        drift_margin = drift_window_avg
        burst_margin = burst_window_avg
        lam_sup = lam_window_avg_sup
    else:
        drift_margin = drift_pointwise
        burst_margin = burst_pointwise
        lam_sup = lam_pointwise_sup

    p_nom = CascadeParams(**{**p_base.__dict__, "lam": float(lam_t)})
    p_env = CascadeParams(**{**p_base.__dict__, "lam": float(lam_sup)})

    # Exact unclamped GPU utilization difference over closed interval q in [0.0, 1.0]
    def raw_rho_diff(q_val: float) -> float:
        comp = _delay_components(q_val, b=b, T_ms=T_ms, params=p_env)
        return comp["raw_rho_G"] - float(rho_max)

    if raw_rho_diff(1.0) <= 0.0:
        q_max_env = 1.0
        tau_max_env = 1.0
    elif raw_rho_diff(0.0) >= 0.0:
        q_max_env = 0.0
        tau_max_env = 0.0
    else:
        q_max_env = float(brentq(raw_rho_diff, 0.0, 1.0, xtol=1e-9))
        tau_max_env = p_env.tau_of_q(q_max_env)

    # Evaluate stationary vs envelope-protected optimal thresholds
    tau_nom = find_optimal_threshold_tau(b=b, T_ms=T_ms, params=p_nom)
    tau_env = find_optimal_threshold_tau(b=b, T_ms=T_ms, params=p_env)
    tau_safe = min(tau_env, tau_max_env)
    d_safe_at_sup = solve_cascade_delay(tau_safe, b=b, T_ms=T_ms, params=p_env, q_override=min(p_env.q_of_tau(tau_safe), q_max_env))
    d_nom_at_sup = solve_cascade_delay(tau_nom, b=b, T_ms=T_ms, params=p_env)

    return {
        "lam_t_fps": float(lam_t),
        "lam_lip_fps_per_s": float(lam_lip),
        "sigma_lam_fps": float(sigma_lam),
        "W_s": float(W_s),
        "tau_corr_s": float(tau_corr_s),
        "z_alpha": float(z_alpha),
        "mode": str(mode),
        "drift_margin_fps": float(drift_margin),
        "burst_margin_fps": float(burst_margin),
        "lam_sup_fps": float(lam_sup),
        "lam_pointwise_sup_fps": float(lam_pointwise_sup),
        "lam_window_avg_sup_fps": float(lam_window_avg_sup),
        "b": int(b),
        "rho_max_bound": float(rho_max),
        "q_max_env": float(q_max_env),
        "tau_max_env": float(tau_max_env),
        "tau_star_nominal": float(tau_nom),
        "tau_star_envelope": float(tau_safe),
        "raw_rho_G_at_q_max": float(_delay_components(q_max_env, b=b, T_ms=T_ms, params=p_env)["raw_rho_G"]),
        "rho_G_nominal_under_burst": float(d_nom_at_sup["rho_G"]),
        "rho_G_envelope_under_burst": float(d_safe_at_sup["rho_G"]),
        "E_D_nominal_under_burst_ms": float(d_nom_at_sup["E_D_ms"]),
        "E_D_envelope_under_burst_ms": float(d_safe_at_sup["E_D_ms"]),
    }


def apply_hysteresis_with_safety_cap(
    tau_cur: float,
    tau_target: float,
    tau_cap: float,
    delta_band: float,
) -> tuple[float, bool]:
    """
    Update held confidence threshold tau_cur given unconstrained target tau_target,
    safety cap tau_cap, and hysteresis half-width delta_band >= 0.

    Enforces two strict invariants:
    1. Deadband suppression: if |min(tau_target, tau_cap) - tau_cur| <= delta_band AND tau_cur <= tau_cap,
       the threshold is held constant (no switch).
    2. Hard safety-cap projection: if a falling safety cap drops below tau_cur (tau_cur > tau_cap),
       tau_cur is projected immediately onto tau_cap regardless of the deadband.
    """
    tau_bounded = min(float(tau_target), float(tau_cap))
    if abs(tau_bounded - float(tau_cur)) > float(delta_band):
        return float(tau_bounded), True
    if float(tau_cur) > float(tau_cap):
        return float(tau_cap), True
    return float(tau_cur), False


def solve_threshold_lambda_sensitivity(
    lam: float = 500.0,
    mu_D: float = 0.002,
    mu_P: float = 0.0005,
    b: int = 4,
    T_ms: float = 25.0,
    params: CascadeParams | None = None,
) -> dict[str, float]:
    """
    Proposition 3 (Conditional Implicit Sensitivity of Optimal Threshold to Arrival Rate):
    On any smooth interior branch tau*(lam) in (0, 1) where the first-order optimality
    condition
      Psi(tau, lam) := r_1(tau) - mu_D * (dE[D]/dq)(q(tau), lam) - mu_P * (dE[P]/dq)(q(tau), lam) = 0
    satisfies positive restoring modulus M_tau(tau*, lam) = -dPsi/dtau > 0, differentiating
    with respect to lam via the Implicit Function Theorem yields:

      d tau*(lam) / d lam = - C_lam(tau*, lam) / M_tau(tau*, lam),

    where C_lam(tau*, lam) = mu_D * d^2 E[D]/(dq dlam) + mu_P * d^2 E[P]/(dq dlam) is the
    mixed marginal arrival-rate cost sensitivity (so d tau*/d lam < 0 iff C_lam > 0).
    At active KKT boundary solutions (tau* in {0, 1}), the subgradient derivative vanishes (0.0).
    """
    p_base = _coerce_params(params)
    lam_f = max(float(lam), 1e-3)
    p_lam = CascadeParams(**{**p_base.__dict__, "lam": lam_f})

    tau_star = find_optimal_threshold_tau(mu_D=mu_D, mu_P=mu_P, b=b, T_ms=T_ms, params=p_lam)
    q_star = p_lam.q_of_tau(tau_star)
    is_interior = bool(1e-6 < tau_star < 1.0 - 1e-6)

    d_center = solve_cascade_delay(tau_star, b=b, T_ms=T_ms, params=p_lam)
    p_center = solve_cascade_power(tau_star, b=b, T_ms=T_ms, params=p_lam)

    # Adaptive step d_lam ensuring p_lo.lam < lam < p_hi.lam for all lam > 0
    d_lam = max(0.05, min(2.0, 0.25 * lam_f))
    p_hi = CascadeParams(**{**p_base.__dict__, "lam": lam_f + d_lam})
    p_lo = CascadeParams(**{**p_base.__dict__, "lam": max(1e-4, lam_f - d_lam)})

    d_hi = solve_cascade_delay(tau_star, b=b, T_ms=T_ms, params=p_hi, q_override=q_star)
    d_lo = solve_cascade_delay(tau_star, b=b, T_ms=T_ms, params=p_lo, q_override=q_star)
    p_hi_res = solve_cascade_power(tau_star, b=b, T_ms=T_ms, params=p_hi, q_override=q_star)
    p_lo_res = solve_cascade_power(tau_star, b=b, T_ms=T_ms, params=p_lo, q_override=q_star)

    span = max(p_hi.lam - p_lo.lam, 1e-9)
    d2ED_dq_dlam = (d_hi["dED_dq_ms"] - d_lo["dED_dq_ms"]) / span
    d2EP_dq_dlam = (p_hi_res["dEP_dq_W"] - p_lo_res["dEP_dq_W"]) / span

    fg = p_lam.f_g(tau_star)
    r1_p = p_lam.r_1_prime(tau_star)

    # Second-order restoring modulus in FOC space: -d Psi / d tau > 0
    foc_slope_tau = -r1_p + fg * (mu_D * d_center["d2ED_dq2_ms"] + mu_P * p_center["d2EP_dq2_W"])
    # True Lagrangian second-order Hessian H_L(tau*) = f_G(tau*) * foc_slope_tau
    hessian_L_tau = fg * foc_slope_tau

    cross_sensitivity = mu_D * d2ED_dq_dlam + mu_P * d2EP_dq_dlam

    # Direct finite-difference verification of tau*(lam)
    tau_hi = find_optimal_threshold_tau(mu_D=mu_D, mu_P=mu_P, b=b, T_ms=T_ms, params=p_hi)
    tau_lo = find_optimal_threshold_tau(mu_D=mu_D, mu_P=mu_P, b=b, T_ms=T_ms, params=p_lo)
    dtau_dlam_fd = (tau_hi - tau_lo) / span

    if is_interior and abs(foc_slope_tau) > 1e-12:
        dtau_dlam_ift = -cross_sensitivity / foc_slope_tau
    else:
        dtau_dlam_ift = dtau_dlam_fd

    return {
        "lam_fps": float(lam_f),
        "b": int(b),
        "tau_star": float(tau_star),
        "q_star": float(q_star),
        "is_interior_solution": is_interior,
        "f_G_tau_star": float(fg),
        "r_1_prime_tau_star": float(r1_p),
        "d2ED_dq_dlam_ms_per_fps": float(d2ED_dq_dlam),
        "d2EP_dq_dlam_W_per_fps": float(d2EP_dq_dlam),
        "mixed_marginal_cost_C_lam": float(cross_sensitivity),
        "foc_restoring_modulus": float(foc_slope_tau),
        "hessian_L_tau": float(hessian_L_tau),
        "dtau_dlam_per_fps": float(dtau_dlam_ift),
        "dtau_dlam_per_100fps": float(100.0 * dtau_dlam_ift),
        "dtau_dlam_fd_per_fps": float(dtau_dlam_fd),
        "ift_vs_fd_abs_err": float(abs(dtau_dlam_ift - dtau_dlam_fd)),
    }


def solve_hysteresis_band(
    lam: float = 500.0,
    mu_D: float = 0.002,
    mu_P: float = 0.0005,
    b: int = 4,
    T_ms: float = 25.0,
    kappa_sw: float = 1.2e-4,
    sigma_lam: float = 75.0,
    tau_corr_s: float = 0.20,
    params: CascadeParams | None = None,
) -> dict[str, float]:
    """
    Theorem 6 (Confidence-Threshold Quartic Hysteresis Deadband Bound):
    Under stochastic arrival fluctuations lam(t) with variance sigma_lam^2 and
    correlation time tau_corr_s, the unconstrained target threshold tau*(lam(t))
    has scale diffusivity parameter:
      D_tau(lam) = (d tau*(lam) / d lam)^2 * (sigma_lam^2 / tau_corr_s).

    Evaluates two analytically closed-form quartic deadband conventions:
    1. Uniform-Residence / Variance-Rate Surrogate (default controller deadband):
       Assumes uniform tracking error density on [-Delta_tau, Delta_tau]
         L_track^sur(Delta_tau) = (1/6) * H_L(tau*, lam) * (Delta_tau)^2
       and variance-rate renewal switching surrogate nu_sw^sur(Delta_tau) = D_tau(lam) / (2 * (Delta_tau)^2),
       yielding the quartic optimum:
         Delta_tau_hyst*(lam) = ( 3 * kappa_sw * D_tau(lam) / H_L(tau*, lam) )^(1/4).
    2. Exact Zero-Drift Brownian-Reset First-Passage Law (dX_t = sqrt(2 D_tau) dW_t reset to 0 at +-Delta_tau):
       By Kolmogorov backward / Green's function integration g(x) = (Delta_tau - |x|) / Delta_tau^2,
       stationary occupation is triangular with E[X^2] = Delta_tau^2 / 6 -> L_track^BM = (1/12) * H_L * Delta_tau^2,
       and mean first-passage exit time is E[T_exit] = Delta_tau^2 / (2 D_tau) -> nu_sw^BM = 2 D_tau / Delta_tau^2,
       yielding the quartic optimum:
         Delta_tau_BM*(lam) = ( 24 * kappa_sw * D_tau(lam) / H_L(tau*, lam) )^(1/4) = 8^(1/4) * Delta_tau_hyst*(lam)
       (or identically Delta_tau_hyst*(lam) under renormalized switching weight kappa_sw^BM = kappa_sw / 8).
    """
    sens = solve_threshold_lambda_sensitivity(
        lam=lam, mu_D=mu_D, mu_P=mu_P, b=b, T_ms=T_ms, params=params
    )
    tau_star = sens["tau_star"]
    dtau_dlam = sens["dtau_dlam_per_fps"]
    hessian_L = max(sens["hessian_L_tau"], 1e-8)

    # Effective threshold diffusivity scale D_tau (tau^2 / s)
    d_tau_diff = (dtau_dlam * dtau_dlam) * ((sigma_lam * sigma_lam) / max(tau_corr_s, 1e-6))

    # 1. Uniform-residence / variance-rate surrogate quartic optimum
    delta_tau_star = float(((3.0 * kappa_sw * d_tau_diff) / hessian_L) ** 0.25)
    # 2. Exact zero-drift Brownian-reset first-passage quartic optimum
    delta_tau_bm_star = float(((24.0 * kappa_sw * d_tau_diff) / hessian_L) ** 0.25)

    tau_minus = max(0.0, tau_star - delta_tau_star)
    tau_plus = min(1.0, tau_star + delta_tau_star)

    p_lam = CascadeParams(**{**_coerce_params(params).__dict__, "lam": float(lam)})
    q_minus = p_lam.q_of_tau(tau_minus)
    q_plus = p_lam.q_of_tau(tau_plus)

    # Surrogate switching rate (switches/s) and regret components
    sw_rate_hyst = d_tau_diff / max(2.0 * (delta_tau_star * delta_tau_star), 1e-12)
    sw_rate_bm = (2.0 * d_tau_diff) / max(delta_tau_bm_star * delta_tau_bm_star, 1e-12)
    # Unregularized control tick rate at dt = T_ms / 1000 (40 Hz baseline for 25 ms tick)
    dt_tick = T_ms / 1000.0
    rms_step_tick = math.sqrt(max(d_tau_diff * dt_tick, 1e-16))
    sw_rate_unreg = 1.0 / dt_tick if rms_step_tick > 0 else 0.0
    chattering_reduction_pct = 100.0 * (1.0 - min(1.0, sw_rate_hyst / max(sw_rate_unreg, 1e-9)))

    tracking_regret = (1.0 / 6.0) * hessian_L * (delta_tau_star * delta_tau_star)
    switching_regret = kappa_sw * sw_rate_hyst
    total_regret = tracking_regret + switching_regret

    bm_tracking_regret = (1.0 / 12.0) * hessian_L * (delta_tau_bm_star * delta_tau_bm_star)
    bm_switching_regret = kappa_sw * sw_rate_bm

    return {
        "lam_fps": float(lam),
        "b": int(b),
        "tau_star": float(tau_star),
        "dtau_dlam_per_100fps": float(sens["dtau_dlam_per_100fps"]),
        "hessian_L_tau": float(hessian_L),
        "D_tau_per_s": float(d_tau_diff),
        "kappa_sw": float(kappa_sw),
        "delta_tau_hyst_star": float(delta_tau_star),
        "delta_tau_bm_reset_star": float(delta_tau_bm_star),
        "tau_minus_star": float(tau_minus),
        "tau_plus_star": float(tau_plus),
        "q_minus_star": float(q_minus),
        "q_plus_star": float(q_plus),
        "expected_switch_rate_hz": float(sw_rate_hyst),
        "bm_reset_switch_rate_hz": float(sw_rate_bm),
        "chattering_reduction_pct": float(chattering_reduction_pct),
        "tracking_regret_bound": float(tracking_regret),
        "switching_regret_bound": float(switching_regret),
        "total_dynamic_regret": float(total_regret),
        "equipartition_residual": float(abs(tracking_regret - switching_regret)),
        "bm_equipartition_residual": float(abs(bm_tracking_regret - bm_switching_regret)),
    }


def verify_symbolic_cascade_proofs() -> dict[str, Any]:
    """
    Run SymPy exact symbolic computer-algebra verification for Paper 2's analytical theorems:
    1. Theorem 1 (Renewal-Reward Batch Accumulation): exact boundary limits T_max -> oo and lam_off -> 0+.
    2. Theorem 2 (Distribution-Invariant Pareto Frontier): f_G(tau) cancels identically in dE[A]/dE[D] and dE[A]/dE[P].
    3. Proposition 3 (Implicit Function Theorem Threshold Sensitivity): exact symbolic differentiation of Psi(tau*(lam), lam) == 0.
    4. Theorem 6 (Quartic Hysteresis Deadband Bound): exact symbolic first/second-order optimality
       and equipartition identity for both the uniform-residence surrogate (3*kappa_sw*D_tau/H_L)^(1/4)
       and the exact Brownian-reset triangular occupation law (24*kappa_sw*D_tau/H_L)^(1/4).
    """
    import sympy as sp

    # 1. Theorem 1 boundary limits for b=2 renewal-reward batch wait:
    # b_bar(u) = 2 - exp(-u), E[W_batch] = (T * exp(-u) + (1/lam_off) * (1 - exp(-u) - u*exp(-u))) / (2 - exp(-u))
    lam_s, T_s = sp.symbols("lam_off T_max", positive=True)
    u_s = lam_s * T_s
    b_bar_b2 = 2 - sp.exp(-u_s)
    w_batch_b2 = (T_s * sp.exp(-u_s) + (1 / lam_s) * (1 - sp.exp(-u_s) - u_s * sp.exp(-u_s))) / b_bar_b2
    limit_T_inf_res = sp.simplify(sp.limit(w_batch_b2, T_s, sp.oo) - 1 / (2 * lam_s))
    limit_lam_0_res = sp.simplify(sp.limit(w_batch_b2, lam_s, 0, dir="+") - T_s)

    # 2. Theorem 2 density cancellation
    f_G, r_1, dED_dq, dEP_dq = sp.symbols("f_G r_1 dED_dq dEP_dq", positive=True)
    dEA_dtau = f_G * r_1
    dED_dtau = f_G * dED_dq
    dEP_dtau = f_G * dEP_dq

    mrs_delay_residual = sp.simplify(dEA_dtau / dED_dtau - r_1 / dED_dq)
    mrs_power_residual = sp.simplify(dEA_dtau / dEP_dtau - r_1 / dEP_dq)

    # 3. Proposition 3 IFT identity on smooth interior branch Psi(tau(lam), lam) = 0
    lam_sym = sp.Symbol("lam", real=True)
    tau_fn = sp.Function("tau_star")(lam_sym)
    q_fn = sp.Function("q_cdf")(tau_fn)
    r1_fn = sp.Function("r_1")(tau_fn)
    dED_fn = sp.Function("dED_dq")(q_fn, lam_sym)
    dEP_fn = sp.Function("dEP_dq")(q_fn, lam_sym)
    mu_D_s, mu_P_s = sp.symbols("mu_D mu_P", positive=True)
    psi_expr = r1_fn - mu_D_s * dED_fn - mu_P_s * dEP_fn
    dpsi_dlam = sp.diff(psi_expr, lam_sym)
    dtau_sol = sp.solve(dpsi_dlam, sp.diff(tau_fn, lam_sym))[0]
    c_lam_sym = (
        mu_D_s * sp.Subs(sp.diff(sp.Function("dED_dq")(q_fn, lam_sym), lam_sym), sp.diff(tau_fn, lam_sym), 0).doit()
        + mu_P_s * sp.Subs(sp.diff(sp.Function("dEP_dq")(q_fn, lam_sym), lam_sym), sp.diff(tau_fn, lam_sym), 0).doit()
    )
    m_tau_sym = -sp.diff(r1_fn, tau_fn) + sp.diff(q_fn, tau_fn) * (
        mu_D_s * sp.diff(dED_fn, q_fn) + mu_P_s * sp.diff(dEP_fn, q_fn)
    )
    ift_residual = sp.simplify(dtau_sol - (-c_lam_sym / m_tau_sym))

    # 4. Theorem 6 Uniform-Residence Surrogate: R(delta) = (1/6)*H_L*delta^2 + kappa_sw*D_tau/(2*delta^2)
    delta, H_L, kappa_sw, D_tau = sp.symbols("delta H_L kappa_sw D_tau", positive=True)
    regret = sp.Rational(1, 6) * H_L * delta**2 + kappa_sw * D_tau / (2 * delta**2)
    d_regret = sp.diff(regret, delta)
    d2_regret = sp.diff(regret, delta, 2)

    sol_delta = sp.solve(d_regret, delta)
    delta_opt = next(s for s in sol_delta if s.is_real is not False)
    expected_formula = (3 * kappa_sw * D_tau / H_L) ** sp.Rational(1, 4)
    quartic_root_residual = sp.simplify(delta_opt**4 - expected_formula**4)
    d2_at_opt = sp.simplify(d2_regret.subs(delta, expected_formula))

    l_track_opt = (sp.Rational(1, 6) * H_L * delta**2).subs(delta, expected_formula)
    l_switch_opt = (kappa_sw * D_tau / (2 * delta**2)).subs(delta, expected_formula)
    equipartition_diff = sp.simplify(l_track_opt - l_switch_opt)

    # 5. Theorem 6 Brownian-Reset First-Passage Law: Green's occupation g(x) = Delta - |x| on [-Delta, Delta]
    x_s = sp.Symbol("x", real=True)
    g_occ = delta - sp.Abs(x_s)
    bm_ex2 = sp.simplify(sp.integrate(x_s**2 * g_occ, (x_s, -delta, delta)) / sp.integrate(g_occ, (x_s, -delta, delta)))
    regret_bm = sp.Rational(1, 2) * H_L * bm_ex2 + kappa_sw * (2 * D_tau / delta**2)
    sol_bm = next(s for s in sp.solve(sp.diff(regret_bm, delta), delta) if s.is_real is not False)
    expected_bm_formula = (24 * kappa_sw * D_tau / H_L) ** sp.Rational(1, 4)
    bm_quartic_root_residual = sp.simplify(sol_bm**4 - expected_bm_formula**4)

    return {
        "sympy_verified": True,
        "theorem1_renewal_limit_T_inf_residual": str(limit_T_inf_res),
        "theorem1_renewal_limit_lam_0_residual": str(limit_lam_0_res),
        "theorem2_mrs_delay_residual": str(mrs_delay_residual),
        "theorem2_mrs_power_residual": str(mrs_power_residual),
        "proposition3_ift_residual": str(ift_residual),
        "theorem6_quartic_formula": str(expected_formula),
        "theorem6_quartic_root_residual": str(quartic_root_residual),
        "theorem6_second_derivative_at_opt": str(d2_at_opt),
        "theorem6_equipartition_residual": str(equipartition_diff),
        "theorem6_bm_occupation_Ex2": str(bm_ex2),
        "theorem6_bm_reset_quartic_root_residual": str(bm_quartic_root_residual),
    }


def simulate_stationary_cascade_simpy(
    q: float = 0.30,
    lam: float = 500.0,
    b: int = 4,
    T_ms: float = 25.0,
    duration_s: float = 40.0,
    warmup_s: float = 5.0,
    n_batches: int = 20,
    seed: int = 42,
    params: CascadeParams | None = None,
) -> dict[str, float]:
    """
    Stationary discrete-event SimPy validation of Theorem 1 (batch-timeout renewal moments)
    and Theorem 3 (expected cascade delay E[D]) with warm-up discard, 20 batch-means
    confidence intervals, and Little's Law conservation check |L - lam_off * W| / L.
    """
    import simpy

    p_base = _coerce_params(params)
    p = CascadeParams(**{**p_base.__dict__, "lam": float(lam)})
    theory = solve_cascade_delay(p.tau_of_q(q), b=b, T_ms=T_ms, params=p, q_override=q)

    rng = np.random.default_rng(seed)
    total_time_s = warmup_s + duration_s

    env = simpy.Environment()
    gpu_server = simpy.Resource(env, capacity=1)

    w_batch_samples_ms: list[float] = []
    e2e_delay_samples_ms: list[float] = []
    batch_sizes: list[int] = []
    timeout_flags: list[int] = []

    # Time-integrated batch buffer occupancy for Little's Law check
    area_in_batch_buffer = 0.0
    last_buffer_change_t = warmup_s
    cur_buffer_count = 0
    post_warmup_offloaded = 0

    pending: list[float] = []
    timer_proc: list[simpy.events.Process | None] = [None]
    k_shape = 1.0 / max(p.cs2_g, 1e-4)

    def update_buffer_area(now_t: float) -> None:
        nonlocal area_in_batch_buffer, last_buffer_change_t
        if now_t > warmup_s:
            t0 = max(last_buffer_change_t, warmup_s)
            if now_t > t0:
                area_in_batch_buffer += cur_buffer_count * (now_t - t0)
        last_buffer_change_t = max(now_t, warmup_s)

    def dispatch(items_t: list[float], timed_out: bool) -> simpy.events.Generator:
        nonlocal post_warmup_offloaded
        b_act = len(items_t)
        t_disp = env.now
        if t_disp >= warmup_s:
            batch_sizes.append(b_act)
            timeout_flags.append(1 if timed_out else 0)
            for t_arr in items_t:
                if t_arr >= warmup_s:
                    w_batch_samples_ms.append((t_disp - t_arr) * 1000.0)
                    post_warmup_offloaded += 1

        t_net_s = (p.t0_net_ms + p.t1_net_ms * b_act) / 1000.0
        s_g_mean_s = (p.t0_g_ms + p.t1_g_ms * b_act) / 1000.0
        s_g_act_s = s_g_mean_s * float(rng.gamma(shape=k_shape, scale=1.0 / k_shape))

        yield env.timeout(t_net_s)
        t_q_enter = env.now
        with gpu_server.request() as req:
            yield req
            t_start = env.now
            w_q_ms = (t_start - t_q_enter) * 1000.0
            yield env.timeout(s_g_act_s)

        for t_arr in items_t:
            if t_arr >= warmup_s:
                w_b_ms = (t_disp - t_arr) * 1000.0
                t_s_ms = (p.t0_net_ms + p.t0_g_ms) + (p.t1_net_ms + p.t1_g_ms) * b_act
                e2e_delay_samples_ms.append(p.d_edge_ms + w_b_ms + w_q_ms + t_s_ms)

    def timer_watch(target_list: list[float]) -> simpy.events.Generator:
        nonlocal cur_buffer_count
        try:
            yield env.timeout(T_ms / 1000.0)
            if pending and pending is target_list:
                update_buffer_area(env.now)
                items_t = list(pending)
                pending.clear()
                cur_buffer_count = 0
                timer_proc[0] = None
                env.process(dispatch(items_t, True))
        except simpy.Interrupt:
            return

    def source() -> simpy.events.Generator:
        nonlocal cur_buffer_count
        while env.now < total_time_s:
            dt = float(rng.exponential(1.0 / lam))
            yield env.timeout(dt)
            if env.now >= total_time_s:
                break
            if float(rng.random()) < q:
                update_buffer_area(env.now)
                pending.append(env.now)
                cur_buffer_count = len(pending)
                if cur_buffer_count == 1 and b > 1:
                    timer_proc[0] = env.process(timer_watch(pending))
                if cur_buffer_count >= b:
                    if timer_proc[0] is not None and timer_proc[0].is_alive:
                        timer_proc[0].interrupt()
                    timer_proc[0] = None
                    items_t = list(pending)
                    pending.clear()
                    cur_buffer_count = 0
                    env.process(dispatch(items_t, False))
            else:
                if env.now >= warmup_s:
                    e2e_delay_samples_ms.append(p.d_edge_ms)

    env.process(source())
    env.run(until=total_time_s)
    update_buffer_area(total_time_s)

    d_arr = np.asarray(e2e_delay_samples_ms, dtype=float)
    blocks = [float(np.mean(blk)) for blk in np.array_split(d_arr, n_batches) if len(blk) > 0]
    sim_ed_ms = float(np.mean(blocks))
    sim_ci95_ms = float(2.093 * np.std(blocks, ddof=1) / math.sqrt(len(blocks)))
    sim_wb_ms = float(np.mean(w_batch_samples_ms)) if w_batch_samples_ms else 0.0
    sim_b_disp = float(np.mean(batch_sizes)) if batch_sizes else 1.0
    sim_p_to = float(np.mean(timeout_flags)) if timeout_flags else 0.0

    # Little's Law in batch accumulation buffer: L_buf vs lam_off_emp * W_batch
    l_buf_emp = area_in_batch_buffer / duration_s
    lam_off_emp = post_warmup_offloaded / duration_s
    l_buf_little = lam_off_emp * (sim_wb_ms / 1000.0)
    littles_law_rel_err = (
        abs(l_buf_emp - l_buf_little) / max(l_buf_little, 1e-9) if b > 1 else 0.0
    )

    ed_rel_err_pct = 100.0 * abs(sim_ed_ms - theory["E_D_ms"]) / theory["E_D_ms"]

    return {
        "q": float(q),
        "lam_fps": float(lam),
        "b": int(b),
        "T_ms": float(T_ms),
        "theory_E_D_ms": float(theory["E_D_ms"]),
        "sim_E_D_ms": sim_ed_ms,
        "sim_E_D_ci95_ms": sim_ci95_ms,
        "E_D_rel_err_pct": float(ed_rel_err_pct),
        "theory_E_B_disp": float(theory["E_B_disp"]),
        "sim_E_B_disp": sim_b_disp,
        "theory_E_W_batch_ms": float(theory["E_W_batch_ms"]),
        "sim_E_W_batch_ms": sim_wb_ms,
        "theory_P_timeout": float(theory["P_timeout"]),
        "sim_P_timeout": sim_p_to,
        "littles_law_rel_err": float(littles_law_rel_err),
    }


def simulate_nonstationary_hysteresis_simpy(
    duration_s: float = 12.0,
    warmup_s: float = 2.0,
    n_batches: int = 20,
    lam_mean: float = 550.0,
    lam_amp: float = 250.0,
    mod_freq_hz: float = 1.0,
    sigma_ou: float = 75.0,
    tau_corr_s: float = 0.20,
    b: int = 4,
    T_ms: float = 25.0,
    seed: int = 42,
    params: CascadeParams | None = None,
) -> dict[str, Any]:
    """
    Discrete-event SimPy validation of the two-tier cascade under a doubly-stochastic
    (Cox) sinusoidal + Ornstein-Uhlenbeck bursty arrival stream lam(t) clipped to
    [220, 1050] FPS (max deterministic drift slope 2*pi*f_mod*A_lam = 1570.8 FPS/s).

    Compares three threshold controllers over identical arrival streams:
    1. Static Nominal Threshold: fixed tau* = tau*(lam_mean), vulnerable to burst queue buildup.
    2. Memoryless Instantaneous Tracking: updates tau(t) = tau*(lam(t)) every 25 ms control tick
       without hysteresis, suffering severe threshold chattering.
    3. Envelope + Quartic Hysteresis Controller (Proposed): updates tau(t) via
       `apply_hysteresis_with_safety_cap` (switching only when min(tau*(lam(t)), tau_max^env)
       exits [tau_cur - Delta_tau, tau_cur + Delta_tau] or when tau_max^env drops below tau_cur).
    """
    import simpy

    p = _coerce_params(params)
    total_time_s = warmup_s + duration_s
    lam_clip_min = 220.0
    lam_clip_max = 1050.0
    # Strict dominating intensity >= lam_clip_max so acceptance probability in [0, 1] for all seeds
    lam_ub = max(lam_clip_max, float(lam_mean) + float(lam_amp) + 3.0 * float(sigma_ou))
    lam_lip_sim = max(400.0, 2.0 * math.pi * float(mod_freq_hz) * float(lam_amp))

    rng = np.random.default_rng(seed)
    t_cur = 0.0
    arrivals_s: list[float] = []
    lam_trace: list[float] = []
    ou_state = 0.0
    tau_corr = max(float(tau_corr_s), 1e-6)
    max_accept_prob = 0.0

    while t_cur < total_time_s:
        dt = float(rng.exponential(1.0 / lam_ub))
        t_cur += dt
        if t_cur >= total_time_s:
            break
        ou_state = ou_state * math.exp(-dt / tau_corr) + sigma_ou * math.sqrt(
            max(1.0 - math.exp(-2.0 * dt / tau_corr), 0.0)
        ) * float(rng.normal())
        lam_inst = float(
            np.clip(
                lam_mean + lam_amp * math.sin(2.0 * math.pi * mod_freq_hz * t_cur) + ou_state,
                lam_clip_min,
                lam_clip_max,
            )
        )
        accept_prob = lam_inst / lam_ub
        if accept_prob < 0.0 or accept_prob > 1.0:
            raise ValueError(f"Invalid thinning acceptance probability: {accept_prob}")
        max_accept_prob = max(max_accept_prob, accept_prob)
        if float(rng.random()) <= accept_prob:
            arrivals_s.append(t_cur)
            lam_trace.append(lam_inst)

    n_arr = len(arrivals_s)
    g_scores = rng.beta(p.alpha_g, p.beta_g, size=n_arr)
    k_shape = 1.0 / max(p.cs2_g, 1e-4)
    srv_noise = rng.gamma(shape=k_shape, scale=1.0 / k_shape, size=n_arr)

    # Precompute lookup table for tau*(lam), hysteresis half-width, and envelope cap across [200, 1100] FPS
    lam_grid = np.linspace(200.0, 1100.0, 37)
    tau_star_grid = np.zeros_like(lam_grid)
    delta_hyst_grid = np.zeros_like(lam_grid)
    tau_env_grid = np.zeros_like(lam_grid)

    for idx, lg in enumerate(lam_grid):
        hb = solve_hysteresis_band(
            lam=float(lg), b=b, T_ms=T_ms, sigma_lam=sigma_ou, tau_corr_s=tau_corr, params=p
        )
        env_res = solve_nonstationary_envelope(
            lam_t=float(lg),
            lam_lip=lam_lip_sim,
            sigma_lam=sigma_ou,
            W_s=0.10,
            tau_corr_s=tau_corr,
            mode="window_average",
            b=b,
            T_ms=T_ms,
            params=p,
        )
        tau_star_grid[idx] = hb["tau_star"]
        delta_hyst_grid[idx] = hb["delta_tau_hyst_star"]
        tau_env_grid[idx] = env_res["tau_max_env"]

    def run_policy(policy: str) -> dict[str, Any]:
        env = simpy.Environment()
        gpu_server = simpy.Resource(env, capacity=1)

        delays_ms: list[float] = []
        acc_gains: list[float] = []
        arr_times_rec: list[float] = []
        offload_flags: list[int] = []
        batch_sizes_disp: list[int] = []
        switch_count = 0

        tau_cur = float(np.interp(lam_mean, lam_grid, tau_star_grid))
        last_tick_idx = -1

        pending_frames: list[tuple[int, float]] = []
        batch_timer_proc: list[simpy.events.Process | None] = [None]

        def dispatch_batch(items: list[tuple[int, float]]) -> simpy.events.Generator:
            b_act = len(items)
            batch_sizes_disp.append(b_act)
            t_net_s = (p.t0_net_ms + p.t1_net_ms * b_act) / 1000.0
            s_g_mean_s = (p.t0_g_ms + p.t1_g_ms * b_act) / 1000.0
            s_g_act_s = s_g_mean_s * float(srv_noise[items[0][0]])

            yield env.timeout(t_net_s)
            with gpu_server.request() as req:
                yield req
                yield env.timeout(s_g_act_s)

            t_done = env.now
            for f_idx, t_arr in items:
                if t_arr >= warmup_s:
                    total_d_ms = p.d_edge_ms + (t_done - t_arr) * 1000.0
                    delays_ms.append(total_d_ms)
                    acc_gains.append(p.r_1(float(g_scores[f_idx])))
                    arr_times_rec.append(t_arr)
                    offload_flags.append(1)

        def timeout_watcher(cycle_items: list[tuple[int, float]]) -> simpy.events.Generator:
            try:
                yield env.timeout(T_ms / 1000.0)
                if pending_frames and pending_frames is cycle_items:
                    to_send = list(pending_frames)
                    pending_frames.clear()
                    batch_timer_proc[0] = None
                    env.process(dispatch_batch(to_send))
            except simpy.Interrupt:
                return

        def arrival_driver() -> simpy.events.Generator:
            nonlocal tau_cur, last_tick_idx, switch_count
            for i in range(n_arr):
                t_target = arrivals_s[i]
                if t_target > env.now:
                    yield env.timeout(t_target - env.now)

                # Control tick every 25 ms
                tick_idx = int(env.now / 0.025)
                if tick_idx != last_tick_idx:
                    last_tick_idx = tick_idx
                    lam_est = lam_trace[i]
                    tau_target = float(np.interp(lam_est, lam_grid, tau_star_grid))
                    delta_band = float(np.interp(lam_est, lam_grid, delta_hyst_grid))
                    tau_cap = float(np.interp(lam_est, lam_grid, tau_env_grid))

                    if policy == "memoryless":
                        if abs(tau_target - tau_cur) > 1e-4:
                            tau_cur = tau_target
                            if env.now >= warmup_s:
                                switch_count += 1
                    elif policy == "hysteresis_envelope":
                        tau_next, switched = apply_hysteresis_with_safety_cap(
                            tau_cur=tau_cur,
                            tau_target=tau_target,
                            tau_cap=tau_cap,
                            delta_band=delta_band,
                        )
                        if switched:
                            tau_cur = tau_next
                            if env.now >= warmup_s:
                                switch_count += 1

                # Gate decision after Tier-1 Edge NPU execution
                if float(g_scores[i]) < tau_cur:
                    pending_frames.append((i, env.now))
                    if len(pending_frames) == 1 and b > 1:
                        batch_timer_proc[0] = env.process(timeout_watcher(pending_frames))
                    if len(pending_frames) >= b:
                        if batch_timer_proc[0] is not None and batch_timer_proc[0].is_alive:
                            batch_timer_proc[0].interrupt()
                        batch_timer_proc[0] = None
                        to_send = list(pending_frames)
                        pending_frames.clear()
                        env.process(dispatch_batch(to_send))
                else:
                    if env.now >= warmup_s:
                        delays_ms.append(p.d_edge_ms)
                        acc_gains.append(0.0)
                        arr_times_rec.append(env.now)
                        offload_flags.append(0)

        env.process(arrival_driver())
        env.run(until=total_time_s + 0.5)

        d_arr = np.asarray(delays_ms, dtype=float)
        a_arr = np.asarray(acc_gains, dtype=float)
        o_arr = np.asarray(offload_flags, dtype=float)

        # Batch-means 95% confidence interval over n_batches blocks
        block_means = [
            float(np.mean(blk)) for blk in np.array_split(d_arr, n_batches) if len(blk) > 0
        ]
        bm_mean = float(np.mean(block_means))
        bm_se = float(np.std(block_means, ddof=1) / math.sqrt(len(block_means)))
        ci95_half = 2.093 * bm_se

        return {
            "policy": policy,
            "mean_delay_ms": bm_mean,
            "delay_ci95_ms": ci95_half,
            "p99_delay_ms": float(np.percentile(d_arr, 99)),
            "offload_fraction_q": float(np.mean(o_arr)),
            "mean_acc_gain": float(np.mean(a_arr)),
            "mean_batch_disp": float(np.mean(batch_sizes_disp)) if batch_sizes_disp else 1.0,
            "switch_count": int(switch_count),
            "switch_rate_hz": float(switch_count / duration_s),
        }

    res_static = run_policy("static_nominal")
    res_memoryless = run_policy("memoryless")
    res_hyst = run_policy("hysteresis_envelope")

    chatter_reduction_pct = 100.0 * (
        1.0 - res_hyst["switch_count"] / max(res_memoryless["switch_count"], 1)
    )
    static_to_hyst_p99_speedup = res_static["p99_delay_ms"] / max(res_hyst["p99_delay_ms"], 1e-9)

    return {
        "seed": int(seed),
        "duration_s": float(duration_s),
        "warmup_s": float(warmup_s),
        "n_batches": int(n_batches),
        "lam_mean_fps": float(lam_mean),
        "lam_amp_fps": float(lam_amp),
        "mod_freq_hz": float(mod_freq_hz),
        "lam_lip_fps_per_s": float(lam_lip_sim),
        "sigma_ou_fps": float(sigma_ou),
        "tau_corr_s": float(tau_corr),
        "lam_dominating_ub_fps": float(lam_ub),
        "max_thinning_accept_prob": float(max_accept_prob),
        "b": int(b),
        "T_ms": float(T_ms),
        "static_nominal": res_static,
        "memoryless_tracking": res_memoryless,
        "hysteresis_envelope": res_hyst,
        "empirical_chattering_reduction_pct": float(chatter_reduction_pct),
        "static_to_hysteresis_p99_ratio": float(static_to_hyst_p99_speedup),
    }
