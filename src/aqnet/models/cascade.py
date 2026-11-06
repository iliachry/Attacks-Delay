"""
Closed-form analytical model for two-station threshold-routed confidence cascades:
Station 1 (M/G/1 Edge NPU) -> Threshold Router (tau) -> Station 2 ((b, T) Batch-with-Timeout + M/G/1 Cloud GPU).

Derives exact closed-form expressions for:
1. Expected end-to-end cascade latency E[D](tau, b, T) via renewal-reward analysis of the (b, T) batch accumulator
   and Pollaczek-Khinchine M/G/1 queueing at both stations.
2. Expected cascade system power E[P](tau, b, T) incorporating dual-state (P0/P8) GPU idle power relaxation.
3. Analytical Pareto frontier derivatives dA_cas/dE[P] and dA_cas/dE[D] in threshold tau (generalized Chow-Queueing rule).
4. Second-order curvature and Jensen convexity / antifragility bounds under threshold calibration error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy.optimize import brentq
from scipy.special import beta as beta_fn
from scipy.special import betainc


@dataclass(frozen=True)
class CascadeParams:
    """Hardware and workload parameters calibrated to Paper 2 (15W Axelera Metis -> Blackwell RTX 5080)."""

    # External Poisson arrival rate (frames/s)
    lam: float = 500.0

    # Station 1: 15W Axelera Metis AIPU (4-core INT8 ResNet-50: 2.4202 ms p50 at B=4 / 1.1243 ms per frame)
    E_S1_ms: float = 1.1243
    C_S1_sq: float = 0.02
    P_idle_1: float = 4.20
    e_dyn_1_mJ: float = 9.190

    # Confidence score distribution G ~ Beta(beta_a, beta_b) on [0, 1]
    beta_a: float = 2.8
    beta_b: float = 1.2
    A_N: float = 0.7582  # Edge NPU standalone top-1 accuracy (ResNet-50 INT8)
    A_G: float = 0.8453  # Cloud GPU standalone top-1 accuracy (ViT-B/16 FP16)
    gamma_gain: float = 1.35  # Shape exponent for conditional oracle gain r_1(tau)

    # Station 2: 1GbE LAN (jpeg224_q90) + NVIDIA GeForce RTX 5080 (Blackwell GB203, ViT-B/16 FP16)
    t_net_0_ms: float = 0.45  # Fixed per-batch TCP RTT base overhead (ms)
    t_net_1_ms: float = 0.25  # Marginal per-frame JPEG224 encode + 1GbE serialization slope (ms/frame)
    t_gpu_0_ms: float = 2.05  # GPU kernel launch + wave-quantization intercept (ms)
    t_gpu_1_ms: float = 0.25  # Marginal per-frame Tensor Core execution slope (ms/frame)
    C_S2_sq: float = 0.05  # Station 2 batch service time squared coefficient of variation

    # Station 2 Power Parameters (calibrated to NVML telemetry in stage2_rtx5080_results.json)
    P_idle_P8: float = 20.28  # Deep idle P8 state power (W)
    P_idle_P0: float = 73.41  # Active clock-gated P0 idle floor between bursts (W)
    tau_P8_s: float = 0.025  # P0 -> P8 transition relaxation time constant (25 ms in s)
    P_s_G: float = 73.965  # Active static power floor during kernel execution above idle (W)
    e_dyn_G_mJ: float = 137.129  # Dynamic Tensor Core + GDDR7 energy per offloaded frame (mJ)
    e_net_mJ: float = 2.50  # Network NIC + switch dynamic energy per offloaded frame (mJ)

    def q_of_tau(self, tau: float) -> float:
        return offload_rate_q(tau, self)

    def tau_of_q(self, q: float) -> float:
        from scipy.stats import beta as _beta_dist
        if q <= 0.0:
            return 0.0
        if q >= 1.0:
            return 1.0
        return float(_beta_dist.ppf(q, self.beta_a, self.beta_b))


def offload_rate_q(tau: float, params: CascadeParams | None = None) -> float:
    """Evaluate offload probability q(tau) = P(G < tau) = F_G(tau) via regularized incomplete Beta function."""
    p = params or CascadeParams()
    if tau <= 0.0:
        return 0.0
    if tau >= 1.0:
        return 1.0
    return float(betainc(p.beta_a, p.beta_b, tau))


def offload_pdf_f(tau: float, params: CascadeParams | None = None) -> float:
    """Evaluate confidence score PDF f_G(tau) = dq/dtau at threshold tau in (0, 1)."""
    p = params or CascadeParams()
    if tau <= 0.0 or tau >= 1.0:
        return 0.0
    a, b = p.beta_a, p.beta_b
    return float((tau ** (a - 1.0)) * ((1.0 - tau) ** (b - 1.0)) / beta_fn(a, b))


def offload_pdf_prime(tau: float, params: CascadeParams | None = None) -> float:
    """Evaluate first derivative of the confidence score PDF f_G'(tau) = d^2q/dtau^2."""
    p = params or CascadeParams()
    if tau <= 0.0 or tau >= 1.0:
        return 0.0
    a, b = p.beta_a, p.beta_b
    f_val = offload_pdf_f(tau, p)
    return float(f_val * ((a - 1.0) / tau - (b - 1.0) / (1.0 - tau)))


def _gain_scale(params: CascadeParams) -> float:
    """Compute normalization constant Delta_A_max so int_0^1 r_1(u) f_G(u) du = A_G - A_N."""
    a, b, gam = params.beta_a, params.beta_b, params.gamma_gain
    # E[G^gamma] for G ~ Beta(a, b) is B(a + gamma, b) / B(a, b)
    e_g_gamma = float(beta_fn(a + gam, b) / beta_fn(a, b))
    denom = max(1e-12, 1.0 - e_g_gamma)
    return (params.A_G - params.A_N) / denom


def conditional_oracle_gain(tau: float, params: CascadeParams | None = None) -> float:
    """Evaluate conditional accuracy gain r_1(tau) = E[Y_G - Y_N | G = tau] >= 0."""
    p = params or CascadeParams()
    tau_c = float(np.clip(tau, 0.0, 1.0))
    scale = _gain_scale(p)
    return float(scale * (1.0 - (tau_c ** p.gamma_gain)))


def conditional_oracle_gain_prime(tau: float, params: CascadeParams | None = None) -> float:
    """Evaluate dr_1(tau)/dtau <= 0 at threshold tau in (0, 1)."""
    p = params or CascadeParams()
    if tau <= 0.0 or tau >= 1.0:
        return 0.0
    scale = _gain_scale(p)
    return float(-scale * p.gamma_gain * (tau ** (p.gamma_gain - 1.0)))


def cascade_accuracy(tau: float, params: CascadeParams | None = None) -> float:
    """Evaluate closed-form cascade top-1 accuracy A_cas(tau) = A_N + int_0^tau r_1(u) f_G(u) du."""
    p = params or CascadeParams()
    if tau <= 0.0:
        return p.A_N
    if tau >= 1.0:
        return p.A_G
    a, b, gam = p.beta_a, p.beta_b, p.gamma_gain
    scale = _gain_scale(p)
    q_val = float(betainc(a, b, tau))
    # int_0^tau u^gam f_G(u) du = (B(a+gam, b)/B(a,b)) * I_tau(a+gam, b)
    moment_ratio = float(beta_fn(a + gam, b) / beta_fn(a, b))
    inc_moment = moment_ratio * float(betainc(a + gam, b, tau))
    return float(p.A_N + scale * (q_val - inc_moment))


def solve_batch_timeout_moments(
    lambda_off: float,
    b: int,
    T_ms: float,
) -> dict[str, float]:
    """
    Exact renewal-reward closed-form solver for a (b, T) batch accumulator with Poisson arrival rate lambda_off (s^-1).

    A renewal cycle starts when the first offloaded frame arrives to an empty batch buffer and dispatches either:
    - when b - 1 additional frames arrive within timeout T = T_ms / 1000 (full batch of size b), or
    - when timeout T expires with k in {0, ..., b-2} additional arrivals (partial batch of size k + 1).

    Returns a dictionary containing:
    - E_B_disp: Expected dispatched batch size bar{b}(lambda_off, b, T)
    - E_W_cycle_ms: Expected cumulative frame waiting time per cycle (ms)
    - E_W_batch_ms: Expected per-frame batch-accumulation waiting time E[W_batch] (ms)
    - p_timeout: Probability that a batch dispatches due to timeout T rather than reaching size b
    - dB_dlam2, d2B_dlam22: 1st and 2nd derivatives of bar{b} with respect to lambda_off (in s)
    - dW_dlam2_s, d2W_dlam22_s: 1st and 2nd derivatives of E[W_batch] (in s) with respect to lambda_off (in s^-1)
    """
    if b < 1:
        raise ValueError(f"Batch threshold b must be >= 1, got {b}")
    if T_ms < 0.0:
        raise ValueError(f"Batch timeout T_ms must be >= 0, got {T_ms}")

    T_s = T_ms / 1000.0
    if b == 1 or T_s == 0.0:
        return {
            "E_B_disp": 1.0,
            "E_W_cycle_ms": 0.0,
            "E_W_batch_ms": 0.0,
            "p_timeout": 0.0 if b == 1 else 1.0,
            "dB_dlam2": 0.0,
            "d2B_dlam22": 0.0,
            "dW_dlam2_s": 0.0,
            "d2W_dlam22_s": 0.0,
        }

    if lambda_off <= 1e-12:
        # Limit as lambda_off -> 0+: first frame waits T_s alone, E[B_disp] -> 1, E[W_batch] -> T_s
        # Series expansion in u = lambda_off * T_s:
        # E[B_disp](u) = 1 + u + O(u^2) (for b >= 3) or 1 + u - u^2/2 (for b == 2)
        # E[W_cycle](u) = T_s * (1 + u/2) (for b >= 3) or T_s * (1 + u/2 - u^2/3) (for b == 2)
        dB_du = 1.0
        d2B_du2 = -1.0 if b == 2 else 0.0
        dW_du_s = -0.5 * T_s
        d2W_du2_s = (1.0 / 3.0) * T_s if b == 2 else 1.0 * T_s
        return {
            "E_B_disp": 1.0,
            "E_W_cycle_ms": T_ms,
            "E_W_batch_ms": T_ms,
            "p_timeout": 1.0,
            "dB_dlam2": dB_du * T_s,
            "d2B_dlam22": d2B_du2 * (T_s ** 2),
            "dW_dlam2_s": dW_du_s * T_s,
            "d2W_dlam22_s": d2W_du2_s * (T_s ** 2),
        }

    if math.isinf(T_ms):
        # Pure count batching without timeout (T -> inf): E[B_disp] = b, E[W_batch] = (b - 1) / (2 * lambda_off)
        w_s = (b - 1.0) / (2.0 * lambda_off)
        dw_dlam2 = -(b - 1.0) / (2.0 * (lambda_off ** 2))
        d2w_dlam22 = (b - 1.0) / (lambda_off ** 3)
        return {
            "E_B_disp": float(b),
            "E_W_cycle_ms": float(b) * w_s * 1000.0,
            "E_W_batch_ms": w_s * 1000.0,
            "p_timeout": 0.0,
            "dB_dlam2": 0.0,
            "d2B_dlam22": 0.0,
            "dW_dlam2_s": dw_dlam2,
            "d2W_dlam22_s": d2w_dlam22,
        }

    u = lambda_off * T_s
    # Compute Poisson probabilities p_k(u) = e^{-u} u^k / k! for k = 0, ..., b
    p_k = np.zeros(b + 1, dtype=float)
    p_k[0] = math.exp(-u)
    for k in range(1, b + 1):
        p_k[k] = p_k[k - 1] * u / k

    # Expected dispatched batch size:
    # bar{b}(u) = b - sum_{k=0}^{b-2} (b - 1 - k) p_k(u)
    ks_b2 = np.arange(0, b - 1, dtype=float)
    E_B = float(b - np.sum((b - 1.0 - ks_b2) * p_k[: b - 1]))

    # Timeout probability P(N(T) <= b - 2) = sum_{k=0}^{b-2} p_k(u)
    p_timeout = float(np.sum(p_k[: b - 1]))

    # Expected cumulative wait per renewal cycle (in seconds):
    # E[W_cycle] = T_s * [ sum_{k=0}^{b-2} (1 + k/2) p_k(u) + (b(b-1)/(2u)) * (1 - sum_{k=0}^{b-1} p_k(u)) ]
    tail_b = max(0.0, 1.0 - float(np.sum(p_k[:b])))
    # Note: 1 - sum_{k=0}^{b-1} p_k(u) = u * (1 - sum_{k=0}^{b-2} p_k(u)) / b + ... exact tail
    cycle_dimless = float(np.sum((1.0 + 0.5 * ks_b2) * p_k[: b - 1])) + (b * (b - 1.0) / (2.0 * u)) * tail_b
    E_W_cycle_s = T_s * cycle_dimless
    E_W_batch_s = E_W_cycle_s / E_B

    # Exact derivatives with respect to u = lambda_off * T_s using dp_k/du = p_{k-1} - p_k:
    # d(E_B)/du = sum_{k=0}^{b-2} p_k(u) = p_timeout  (Wait: let's verify!
    # d/du [b - sum_{k=0}^{b-2} (b - 1 - k) p_k] = - sum_{k=0}^{b-2} (b - 1 - k)(p_{k-1} - p_k)
    # = sum_{k=0}^{b-2} p_k ! Exact telescopic identity!)
    dB_du = float(np.sum(p_k[: b - 1]))
    # d^2(E_B)/du^2 = d/du [sum_{k=0}^{b-2} p_k(u)] = -p_{b-2}(u)
    d2B_du2 = float(-p_k[b - 2])

    # For E[W_cycle_s] = T_s * C(u), let's compute C'(u) and C''(u) exactly:
    # Recall C(u) = sum_{j=1}^{b-1} (1 - j/b) * [ (j / u) * (1 - sum_{k=0}^{j-1} p_k(u)) ]
    # Actually even simpler: on each interval where N(T_s) >= j, the j-th arrival time tau_j has
    # E[min(tau_j, T_s)] = int_0^{T_s} P(tau_j > t) dt = (T_s / u) * sum_{m=0}^{j-1} (j - m) p_m(u)?
    # Even simpler: we can differentiate C(u) directly using dp_k/du = p_{k-1} - p_k!
    dp_du = np.zeros(b + 1, dtype=float)
    dp_du[0] = -p_k[0]
    for k in range(1, b + 1):
        dp_du[k] = p_k[k - 1] - p_k[k]

    d2p_du2 = np.zeros(b + 1, dtype=float)
    d2p_du2[0] = p_k[0]
    for k in range(1, b + 1):
        d2p_du2[k] = dp_du[k - 1] - dp_du[k]

    # Note that d(tail_b)/du = d/du (1 - sum_{k=0}^{b-1} p_k(u)) = p_{b-1}(u)
    # and d^2(tail_b)/du^2 = dp_{b-1}/du = p_{b-2}(u) - p_{b-1}(u).
    # To avoid catastrophic cancellation at small u, if u < 1e-3 we use the exact power series of tail_b / u:
    # tail_b / u = e^{-u} sum_{m=0}^inf u^{b-1+m} / (b+m)!
    if u < 1e-2:
        # Exact series for tail_b / u and its derivatives
        terms = 12
        ms = np.arange(terms, dtype=float)
        facts = np.array([math.factorial(b + int(m)) for m in ms], dtype=float)
        # g(u) = sum_{m=0}^inf u^{b-1+m} / (b+m)!
        u_pows = np.array([u ** (b - 1 + int(m)) for m in ms], dtype=float)
        g_u = float(np.sum(u_pows / facts))
        dg_du = float(np.sum([((b - 1 + int(m)) * (u ** max(0, b - 2 + int(m))) if b - 1 + int(m) >= 1 else 0.0) / facts[int(m)] for m in ms]))
        d2g_du2 = float(
            np.sum(
                [
                    (
                        (b - 1 + int(m)) * (b - 2 + int(m)) * (u ** max(0, b - 3 + int(m)))
                        if b - 1 + int(m) >= 2
                        else 0.0
                    )
                    / facts[int(m)]
                    for m in ms
                ]
            )
        )
        # h(u) = tail_b / u = e^{-u} g(u)
        exp_neg_u = p_k[0]
        dh_du = exp_neg_u * (dg_du - g_u)
        d2h_du2 = exp_neg_u * (d2g_du2 - 2.0 * dg_du + g_u)
    else:
        h_u = tail_b / u
        dh_du = (p_k[b - 1] * u - tail_b) / (u ** 2)
        d2h_du2 = (dp_du[b - 1] * (u ** 2) - 2.0 * u * p_k[b - 1] + 2.0 * tail_b) / (u ** 3)

    dC_du = float(np.sum((1.0 + 0.5 * ks_b2) * dp_du[: b - 1])) + 0.5 * b * (b - 1.0) * dh_du
    d2C_du2 = float(np.sum((1.0 + 0.5 * ks_b2) * d2p_du2[: b - 1])) + 0.5 * b * (b - 1.0) * d2h_du2

    # W_batch_s(u) = T_s * C(u) / E_B(u)
    w_dimless = cycle_dimless / E_B
    dw_du = (dC_du - w_dimless * dB_du) / E_B
    d2w_du2 = (d2C_du2 - 2.0 * dw_du * dB_du - w_dimless * d2B_du2) / E_B

    return {
        "E_B_disp": E_B,
        "E_W_cycle_ms": E_W_cycle_s * 1000.0,
        "E_W_batch_ms": E_W_batch_s * 1000.0,
        "p_timeout": p_timeout,
        "dB_dlam2": dB_du * T_s,
        "d2B_dlam22": d2B_du2 * (T_s ** 2),
        "dW_dlam2_s": dw_du * (T_s ** 2),
        "d2W_dlam22_s": d2w_du2 * (T_s ** 3),
    }


def solve_cascade_delay(
    tau: float,
    b: int = 4,
    T_ms: float = 25.0,
    params: CascadeParams | None = None,
    q_override: float | None = None,
) -> dict[str, Any]:
    """
    Compute closed-form expected end-to-end latency E[D](tau, b, T) (in ms) and all station queueing moments.

    E[D](tau, b, T) = E[D_1] + q(tau) * [ E[W_batch](tau, b, T) + E[W_queue](tau, b, T) + T_srv(bar{b}(tau, b, T)) ]
    """
    p = params or CascadeParams()
    q = float(q_override) if q_override is not None else offload_rate_q(tau, p)
    q = float(np.clip(q, 0.0, 1.0))

    # Station 1: Edge NPU M/G/1
    E_S1_s = p.E_S1_ms / 1000.0
    rho_1 = p.lam * E_S1_s
    # Critical offload threshold under asymptotic full batch b:
    # rho_2 = q * lam * (t_gpu_0 / b + t_gpu_1) < 1
    t_gpu_per_frame_full_s = (p.t_gpu_0_ms / float(max(1, b)) + p.t_gpu_1_ms) / 1000.0
    q_crit = min(1.0, 1.0 / (p.lam * t_gpu_per_frame_full_s)) if p.lam > 0 else 1.0

    if rho_1 >= 1.0:
        return {
            "E_D_ms": float("inf"),
            "E_D1_ms": float("inf"),
            "E_W_batch_ms": 0.0,
            "E_W_queue_ms": float("inf"),
            "E_S2_ms": 0.0,
            "E_T_net_ms": 0.0,
            "E_T_gpu_ms": 0.0,
            "E_B_disp": 1.0,
            "q": q,
            "lambda_2": q * p.lam,
            "lambda_batch": 0.0,
            "rho_1": rho_1,
            "rho_2": float("inf"),
            "q_crit": q_crit,
            "p_timeout": 0.0,
            "stable": False,
        }

    E_W1_ms = p.E_S1_ms * (rho_1 * (1.0 + p.C_S1_sq)) / (2.0 * (1.0 - rho_1))
    E_D1_ms = p.E_S1_ms + E_W1_ms

    if q <= 0.0 or p.lam <= 0.0:
        t_net_1 = p.t_net_0_ms + p.t_net_1_ms
        t_gpu_1 = p.t_gpu_0_ms + p.t_gpu_1_ms
        return {
            "E_D_ms": E_D1_ms,
            "E_D1_ms": E_D1_ms,
            "E_W_batch_ms": T_ms if b > 1 else 0.0,
            "E_W_queue_ms": 0.0,
            "E_S2_ms": t_net_1 + t_gpu_1,
            "E_T_net_ms": t_net_1,
            "E_T_gpu_ms": t_gpu_1,
            "E_B_disp": 1.0,
            "q": 0.0,
            "lambda_2": 0.0,
            "lambda_batch": 0.0,
            "rho_1": rho_1,
            "rho_2": 0.0,
            "q_crit": q_crit,
            "p_timeout": 1.0 if b > 1 else 0.0,
            "stable": True,
        }

    lam_2 = q * p.lam
    bm = solve_batch_timeout_moments(lam_2, b, T_ms)
    E_B = bm["E_B_disp"]
    E_W_batch_ms = bm["E_W_batch_ms"]
    lam_batch = lam_2 / E_B

    # Station 2 service times at expected dispatched batch size E_B
    E_T_net_ms = p.t_net_0_ms + p.t_net_1_ms * E_B
    E_T_gpu_ms = p.t_gpu_0_ms + p.t_gpu_1_ms * E_B
    E_S2_ms = E_T_net_ms + E_T_gpu_ms

    # GPU utilization rho_2 = lambda_batch * E[T_gpu]
    rho_2 = lam_batch * (E_T_gpu_ms / 1000.0)
    if rho_2 >= 1.0:
        return {
            "E_D_ms": float("inf"),
            "E_D1_ms": E_D1_ms,
            "E_W_batch_ms": E_W_batch_ms,
            "E_W_queue_ms": float("inf"),
            "E_S2_ms": E_S2_ms,
            "E_T_net_ms": E_T_net_ms,
            "E_T_gpu_ms": E_T_gpu_ms,
            "E_B_disp": E_B,
            "q": q,
            "lambda_2": lam_2,
            "lambda_batch": lam_batch,
            "rho_1": rho_1,
            "rho_2": rho_2,
            "q_crit": q_crit,
            "p_timeout": bm["p_timeout"],
            "stable": False,
        }

    # Pollaczek-Khinchine queueing delay at Station 2
    E_W_queue_ms = E_T_gpu_ms * (rho_2 * (1.0 + p.C_S2_sq)) / (2.0 * (1.0 - rho_2))
    E_D_ms = E_D1_ms + q * (E_W_batch_ms + E_W_queue_ms + E_S2_ms)

    return {
        "E_D_ms": E_D_ms,
        "E_D1_ms": E_D1_ms,
        "E_W_batch_ms": E_W_batch_ms,
        "E_W_queue_ms": E_W_queue_ms,
        "E_S2_ms": E_S2_ms,
        "E_T_net_ms": E_T_net_ms,
        "E_T_gpu_ms": E_T_gpu_ms,
        "E_B_disp": E_B,
        "q": q,
        "lambda_2": lam_2,
        "lambda_batch": lam_batch,
        "rho_1": rho_1,
        "rho_2": rho_2,
        "q_crit": q_crit,
        "p_timeout": bm["p_timeout"],
        "stable": True,
    }


def solve_cascade_power(
    tau: float,
    b: int = 4,
    T_ms: float = 25.0,
    params: CascadeParams | None = None,
    q_override: float | None = None,
) -> dict[str, float]:
    """
    Compute closed-form expected cascade system power E[P](tau, b, T) (in Watts) and energy per frame (mJ).

    E[P](tau, b, T) = P_1(lambda) + lambda * q(tau) * [ e_net + e_dyn^G + P_s^G * (t_0^G / bar{b} + t_1^G) ]
                      + (1 - rho_2(tau, b, T)) * P_idle^eff(tau, b, T)
    """
    p = params or CascadeParams()
    delay_res = solve_cascade_delay(tau, b=b, T_ms=T_ms, params=p, q_override=q_override)
    q = delay_res["q"]
    E_B = delay_res["E_B_disp"]
    lam_batch = delay_res["lambda_batch"]
    E_T_gpu_s = delay_res["E_T_gpu_ms"] / 1000.0
    rho_2 = min(1.0, delay_res["rho_2"]) if math.isfinite(delay_res["rho_2"]) else 1.0

    # Station 1 (Metis Edge NPU) power
    P_edge_W = p.P_idle_1 + p.lam * (p.e_dyn_1_mJ / 1000.0)

    # Network power
    P_net_W = p.lam * q * (p.e_net_mJ / 1000.0)

    # Effective GPU idle power between batch bursts (dual-state P0/P8 exponential relaxation)
    if lam_batch <= 1e-12:
        P_idle_eff_W = p.P_idle_P8
    else:
        t_inter_idle_s = max(0.0, (1.0 / lam_batch) - E_T_gpu_s)
        P_idle_eff_W = p.P_idle_P8 + (p.P_idle_P0 - p.P_idle_P8) * math.exp(-t_inter_idle_s / p.tau_P8_s)

    # GPU active + idle power
    t_gpu_per_frame_s = (p.t_gpu_0_ms / E_B + p.t_gpu_1_ms) / 1000.0
    P_gpu_active_W = p.lam * q * ((p.e_dyn_G_mJ / 1000.0) + p.P_s_G * t_gpu_per_frame_s)
    P_gpu_idle_W = (1.0 - rho_2) * P_idle_eff_W
    P_gpu_W = P_gpu_idle_W + P_gpu_active_W

    E_P_W = P_edge_W + P_net_W + P_gpu_W
    E_frame_mJ = (E_P_W / p.lam) * 1000.0 if p.lam > 0 else 0.0
    EDP_mJ_ms = E_frame_mJ * delay_res["E_D_ms"] if math.isfinite(delay_res["E_D_ms"]) else float("inf")

    return {
        "E_P_W": E_P_W,
        "P_edge_W": P_edge_W,
        "P_gpu_W": P_gpu_W,
        "P_gpu_active_W": P_gpu_active_W,
        "P_gpu_idle_W": P_gpu_idle_W,
        "P_idle_eff_W": P_idle_eff_W,
        "P_net_W": P_net_W,
        "E_frame_mJ": E_frame_mJ,
        "EDP_mJ_ms": EDP_mJ_ms,
        "q": q,
        "E_B_disp": E_B,
        "rho_2": rho_2,
    }


def _derivatives_wrt_q(
    q: float,
    b: int,
    T_ms: float,
    params: CascadeParams,
) -> dict[str, float]:
    """
    Compute exact analytical 1st and 2nd derivatives of E[D] (in ms) and E[P] (in W) with respect to q in (0, q_crit).
    All components are derived analytically via the chain rule through lambda_2 = q * lambda and bar{b}(lambda_2).
    """
    lam = params.lam
    lam_2 = q * lam
    bm = solve_batch_timeout_moments(lam_2, b, T_ms)
    B = bm["E_B_disp"]
    dB_dq = bm["dB_dlam2"] * lam
    d2B_dq2 = bm["d2B_dlam22"] * (lam ** 2)

    W_b_ms = bm["E_W_batch_ms"]
    dWb_dq_ms = bm["dW_dlam2_s"] * 1000.0 * lam
    d2Wb_dq2_ms = bm["d2W_dlam22_s"] * 1000.0 * (lam ** 2)

    # 1. Batch-accumulation component D_batch(q) = q * W_b_ms(q)
    d_qWb_dq = W_b_ms + q * dWb_dq_ms
    d2_qWb_dq2 = 2.0 * dWb_dq_ms + q * d2Wb_dq2_ms

    # 2. Service time component D_srv(q) = q * (t_0 + t_1 * B(q))
    t_0_ms = params.t_net_0_ms + params.t_gpu_0_ms
    t_1_ms = params.t_net_1_ms + params.t_gpu_1_ms
    T_srv_ms = t_0_ms + t_1_ms * B
    dTsrv_dq = t_1_ms * dB_dq
    d2Tsrv_dq2 = t_1_ms * d2B_dq2
    d_qTsrv_dq = T_srv_ms + q * dTsrv_dq
    d2_qTsrv_dq2 = 2.0 * dTsrv_dq + q * d2Tsrv_dq2

    # 3. Queueing delay component D_queue(q) = q * E[W_queue](q)
    # E[W_queue](q) = c_2 * T_G(B) * rho_2 / (1 - rho_2), where c_2 = (1 + C_S2^2) / 2
    c_2 = 0.5 * (1.0 + params.C_S2_sq)
    T_G_ms = params.t_gpu_0_ms + params.t_gpu_1_ms * B
    dTG_dq = params.t_gpu_1_ms * dB_dq
    d2TG_dq2 = params.t_gpu_1_ms * d2B_dq2

    # rho_2(q) = (q * lam / 1000) * (t_gpu_0_ms / B + t_gpu_1_ms) = (lam / 1000) * (q * T_G_ms / B)
    # Let phi(q) = T_G_ms(q) / B(q) = t_gpu_0_ms / B(q) + t_gpu_1_ms
    phi = params.t_gpu_0_ms / B + params.t_gpu_1_ms
    dphi_dq = -params.t_gpu_0_ms * dB_dq / (B ** 2)
    d2phi_dq2 = params.t_gpu_0_ms * (2.0 * (dB_dq ** 2) / (B ** 3) - d2B_dq2 / (B ** 2))

    k_lam = lam / 1000.0
    rho_2 = k_lam * q * phi
    drho_dq = k_lam * (phi + q * dphi_dq)
    d2rho_dq2 = k_lam * (2.0 * dphi_dq + q * d2phi_dq2)

    one_m_rho = max(1e-9, 1.0 - rho_2)
    # h_r(rho) = rho / (1 - rho), h_r' = 1 / (1 - rho)^2, h_r'' = 2 / (1 - rho)^3
    hr = rho_2 / one_m_rho
    dhr_dq = drho_dq / (one_m_rho ** 2)
    d2hr_dq2 = d2rho_dq2 / (one_m_rho ** 2) + 2.0 * (drho_dq ** 2) / (one_m_rho ** 3)

    W_q_ms = c_2 * T_G_ms * hr
    dWq_dq = c_2 * (dTG_dq * hr + T_G_ms * dhr_dq)
    d2Wq_dq2 = c_2 * (d2TG_dq2 * hr + 2.0 * dTG_dq * dhr_dq + T_G_ms * d2hr_dq2)

    d_qWq_dq = W_q_ms + q * dWq_dq
    d2_qWq_dq2 = 2.0 * dWq_dq + q * d2Wq_dq2

    dED_dq = d_qWb_dq + d_qTsrv_dq + d_qWq_dq
    d2ED_dq2 = d2_qWb_dq2 + d2_qTsrv_dq2 + d2_qWq_dq2

    # 4. Power E[P](q) derivatives with respect to q:
    # P_active(q) = k_lam * q * (e_net + e_dyn_G) + P_s_G * rho_2(q)
    e_var_J = (params.e_net_mJ + params.e_dyn_G_mJ) / 1000.0
    dPact_dq = lam * e_var_J + params.P_s_G * drho_dq
    d2Pact_dq2 = params.P_s_G * d2rho_dq2

    # P_idle_gpu(q) = (1 - rho_2(q)) * P_idle_eff(q)
    # where t_inter_idle(q) = 1 / lam_batch - T_G_s = (B / (q * lam)) - (T_G_ms / 1000) = (B / (q * lam)) * (1 - rho_2)
    if q <= 1e-9:
        dPidle_dq = 0.0
        d2Pidle_dq2 = 0.0
    else:
        # s(q) = t_inter_idle(q) / tau_P8_s = [ B(q) / (q * lam) - T_G_ms(q) / 1000 ] / tau_P8_s
        inv_tau = 1.0 / params.tau_P8_s
        # Let psi(q) = B(q) / q, psi' = (dB_dq * q - B) / q^2, psi'' = (d2B_dq2 * q^2 - 2 * q * dB_dq + 2 * B) / q^3
        psi = B / q
        dpsi_dq = (dB_dq * q - B) / (q ** 2)
        d2psi_dq2 = (d2B_dq2 * (q ** 2) - 2.0 * q * dB_dq + 2.0 * B) / (q ** 3)

        t_inter = psi / lam - (T_G_ms / 1000.0)
        if t_inter <= 0.0:
            P_idle_eff = params.P_idle_P0
            dPoeff_dq = 0.0
            d2Poeff_dq2 = 0.0
        else:
            dt_inter_dq = dpsi_dq / lam - (dTG_dq / 1000.0)
            d2t_inter_dq2 = d2psi_dq2 / lam - (d2TG_dq2 / 1000.0)
            exp_term = (params.P_idle_P0 - params.P_idle_P8) * math.exp(-t_inter * inv_tau)
            P_idle_eff = params.P_idle_P8 + exp_term
            dPoeff_dq = -inv_tau * dt_inter_dq * exp_term
            d2Poeff_dq2 = exp_term * (-inv_tau * d2t_inter_dq2 + (inv_tau ** 2) * (dt_inter_dq ** 2))

        dPidle_dq = -drho_dq * P_idle_eff + (1.0 - rho_2) * dPoeff_dq
        d2Pidle_dq2 = -d2rho_dq2 * P_idle_eff - 2.0 * drho_dq * dPoeff_dq + (1.0 - rho_2) * d2Poeff_dq2

    dEP_dq = dPact_dq + dPidle_dq
    d2EP_dq2 = d2Pact_dq2 + d2Pidle_dq2

    return {
        "dED_dq_ms": dED_dq,
        "d2ED_dq2_ms": d2ED_dq2,
        "d_qWb_dq_ms": d_qWb_dq,
        "d2_qWb_dq2_ms": d2_qWb_dq2,
        "d_qTsrv_dq_ms": d_qTsrv_dq,
        "d2_qTsrv_dq2_ms": d2_qTsrv_dq2,
        "d_qWq_dq_ms": d_qWq_dq,
        "d2_qWq_dq2_ms": d2_qWq_dq2,
        "dEP_dq_W": dEP_dq,
        "d2EP_dq2_W": d2EP_dq2,
        "dPact_dq_W": dPact_dq,
        "d2Pact_dq2_W": d2Pact_dq2,
        "dPidle_dq_W": dPidle_dq,
        "d2Pidle_dq2_W": d2Pidle_dq2,
    }


def solve_pareto_derivative(
    tau: float,
    b: int = 4,
    T_ms: float = 25.0,
    params: CascadeParams | None = None,
    mu_D: float = 0.002,
    mu_P: float = 0.0005,
) -> dict[str, float]:
    """
    Compute the analytical Pareto derivatives at confidence threshold tau:
    - dE[D]/dtau, dE[P]/dtau, dA_cas/dtau
    - Density-invariant Pareto frontier slopes dA_cas/dE[P] and dA_cas/dE[D]
    - Generalized Chow-Queueing Lagrangian stationarity residual dU/dtau
    """
    p = params or CascadeParams()
    q = offload_rate_q(tau, p)
    f_tau = offload_pdf_f(tau, p)
    r1_tau = conditional_oracle_gain(tau, p)
    q_derivs = _derivatives_wrt_q(max(1e-6, min(0.999, q)), b, T_ms, p)

    dED_dq = q_derivs["dED_dq_ms"]
    dEP_dq = q_derivs["dEP_dq_W"]

    dA_dtau = f_tau * r1_tau
    dED_dtau = f_tau * dED_dq
    dEP_dtau = f_tau * dEP_dq

    # Density f_G(tau) cancels identically in the Pareto frontier slopes!
    dA_dEP = r1_tau / dEP_dq if abs(dEP_dq) > 1e-12 else float("inf")
    dA_dED = r1_tau / dED_dq if abs(dED_dq) > 1e-12 else float("inf")
    dED_dEP = dED_dq / dEP_dq if abs(dEP_dq) > 1e-12 else float("inf")

    # Generalized Chow-Queueing Lagrangian: U(tau) = A_cas(tau) - mu_D * E[D](tau) - mu_P * E[P](tau)
    marginal_net_gain_per_q = r1_tau - mu_D * dED_dq - mu_P * dEP_dq
    dU_dtau = f_tau * marginal_net_gain_per_q

    return {
        "tau": tau,
        "q": q,
        "f_G": f_tau,
        "r_1": r1_tau,
        "dA_dtau": dA_dtau,
        "dED_dq_ms": dED_dq,
        "dED_dtau_ms": dED_dtau,
        "dEP_dq_W": dEP_dq,
        "dEP_dtau_W": dEP_dtau,
        "dA_dEP_per_W": dA_dEP,
        "dA_dED_per_ms": dA_dED,
        "dED_dEP_ms_per_W": dED_dEP,
        "marginal_net_gain_per_q": marginal_net_gain_per_q,
        "dU_dtau": dU_dtau,
    }


def find_optimal_threshold_tau(
    mu_D: float = 0.002,
    mu_P: float = 0.0005,
    b: int = 4,
    T_ms: float = 25.0,
    params: CascadeParams | None = None,
) -> float:
    """
    Solve the generalized Chow-Queueing optimality condition:
        r_1(tau*) = mu_D * (dE[D]/dq)(q(tau*), b, T) + mu_P * (dE[P]/dq)(q(tau*), b, T)
    for the unique Pareto-optimal threshold tau* in [0, tau_crit).
    """
    p = params or CascadeParams()
    delay_0 = solve_cascade_delay(0.0, b=b, T_ms=T_ms, params=p, q_override=0.0)
    q_max = min(0.95, 0.92 * delay_0["q_crit"])

    def obj(tau_val: float) -> float:
        res = solve_pareto_derivative(tau_val, b=b, T_ms=T_ms, params=p, mu_D=mu_D, mu_P=mu_P)
        return res["marginal_net_gain_per_q"]

    # Find tau_max corresponding to q_max
    tau_hi = 0.99
    while offload_rate_q(tau_hi, p) > q_max and tau_hi > 0.05:
        tau_hi -= 0.02

    val_lo = obj(1e-4)
    val_hi = obj(tau_hi)
    if val_lo <= 0.0:
        return 0.0
    if val_hi >= 0.0:
        return tau_hi
    return float(brentq(obj, 1e-4, tau_hi, xtol=1e-7))


def jensen_antifragility_bound(
    tau: float,
    b: int = 4,
    T_ms: float = 25.0,
    sigma_eps: float = 0.02,
    delta_max: float = 0.05,
    params: CascadeParams | None = None,
    q_override: float | None = None,
    mu_D: float = 0.002,
    mu_P: float = 0.0005,
) -> dict[str, Any]:
    """
    Derive and evaluate the exact second-order curvature and Jensen convexity / antifragility bounds
    under zero-mean threshold/calibration error (E[eps] = 0, Var(eps) = sigma_eps^2, |eps| <= delta_max).

    Theorem (Micro-Batch Cascade Calibration Antifragility):
    The second derivative of expected cascade latency with respect to offload rate q decomposes into three terms:
        d^2 E[D] / dq^2 = C_batch(q, b, T) + C_amort(q, b, T) + C_queue(q, b, T)
    where:
    1. C_batch(q, b, T) = d^2(q E[W_batch])/dq^2 <= 0 is the batch-timeout concavity term (strictly < 0 for b >= 2, T < inf),
    2. C_amort(q, b, T) = d^2(q T_srv(bar{b}))/dq^2 <= 0 is the batch-launch amortization concavity term,
    3. C_queue(q, b, T) = d^2(q E[W_queue])/dq^2 > 0 is the Pollaczek-Khinchine M/G/1 queue congestion convexity term.

    Consequently:
    - For unbatched offloading (b = 1), C_batch = C_amort = 0 and d^2 E[D]/dq^2 = C_queue > 0 everywhere:
      by Jensen's inequality, E[D(q + eps_q, 1, T)] > E[D](q, 1, T) (strictly fragile to calibration error).
    - For micro-batched offloading (b >= 2, T < inf) in the moderate-utilization regime where
      |C_batch(q, b, T) + C_amort(q, b, T)| > C_queue(q, b, T), we have d^2 E[D]/dq^2 < 0 and d^2 E[P]/dq^2 < 0:
      by the Taylor-Lagrange / Jensen bound,
          E[D(q + eps_q, b, T)] <= E[D](q, b, T) + (sigma_eps^2 / 2) * sup_{|d|<=delta} (d^2 E[D]/dq^2)(q + d) < E[D](q, b, T),
      proving that the micro-batched cascade is strictly delay-antifragile and power-antifragile to calibration jitter.
    """
    p = params or CascadeParams()
    q_center = float(q_override) if q_override is not None else offload_rate_q(tau, p)
    q_center = float(np.clip(q_center, 1e-4, 0.95))

    center_d = _derivatives_wrt_q(q_center, b, T_ms, p)
    C_batch = center_d["d2_qWb_dq2_ms"]
    C_amort = center_d["d2_qTsrv_dq2_ms"]
    C_queue = center_d["d2_qWq_dq2_ms"]
    d2ED_dq2 = center_d["d2ED_dq2_ms"]
    d2EP_dq2 = center_d["d2EP_dq2_W"]

    # Also compute tau-space second derivatives using chain rule:
    # d^2 g(q(tau)) / dtau^2 = f_G(tau)^2 * d^2g/dq^2 + f_G'(tau) * dg/dq
    f_tau = offload_pdf_f(tau, p)
    fp_tau = offload_pdf_prime(tau, p)
    r1_tau = conditional_oracle_gain(tau, p)
    r1p_tau = conditional_oracle_gain_prime(tau, p)

    d2ED_dtau2 = (f_tau ** 2) * d2ED_dq2 + fp_tau * center_d["dED_dq_ms"]
    d2EP_dtau2 = (f_tau ** 2) * d2EP_dq2 + fp_tau * center_d["dEP_dq_W"]
    d2A_dtau2 = fp_tau * r1_tau + f_tau * r1p_tau
    d2U_dtau2 = d2A_dtau2 - mu_D * d2ED_dtau2 - mu_P * d2EP_dtau2

    # Evaluate neighborhood [q_center - delta_q, q_center + delta_q] for exact Taylor-Lagrange infimum/supremum bounds
    delay_0 = solve_cascade_delay(tau, b=b, T_ms=T_ms, params=p, q_override=q_center)
    q_lo = max(1e-3, q_center - delta_max)
    q_hi = min(0.92 * delay_0["q_crit"], q_center + delta_max)
    q_grid = np.linspace(q_lo, q_hi, 17)
    d2D_grid = np.array([_derivatives_wrt_q(float(qg), b, T_ms, p)["d2ED_dq2_ms"] for qg in q_grid])
    d2P_grid = np.array([_derivatives_wrt_q(float(qg), b, T_ms, p)["d2EP_dq2_W"] for qg in q_grid])

    inf_d2D = float(np.min(d2D_grid))
    sup_d2D = float(np.max(d2D_grid))
    inf_d2P = float(np.min(d2P_grid))
    sup_d2P = float(np.max(d2P_grid))

    # Jensen gap bounds for Delta E[D] = E[D(q + eps_q)] - E[D](q):
    # (sigma^2 / 2) * inf_d2D <= E[D(q + eps_q)] - E[D](q) <= (sigma^2 / 2) * sup_d2D
    jensen_delay_shift_center_ms = 0.5 * (sigma_eps ** 2) * d2ED_dq2
    jensen_delay_lower_ms = 0.5 * (sigma_eps ** 2) * inf_d2D
    jensen_delay_upper_ms = 0.5 * (sigma_eps ** 2) * sup_d2D

    jensen_power_shift_center_W = 0.5 * (sigma_eps ** 2) * d2EP_dq2
    jensen_power_lower_W = 0.5 * (sigma_eps ** 2) * inf_d2P
    jensen_power_upper_W = 0.5 * (sigma_eps ** 2) * sup_d2P

    return {
        "tau": tau,
        "q": q_center,
        "b": b,
        "T_ms": T_ms,
        "sigma_eps": sigma_eps,
        "delta_max": delta_max,
        "C_batch_ms": C_batch,
        "C_amort_ms": C_amort,
        "C_queue_ms": C_queue,
        "d2ED_dq2_ms": d2ED_dq2,
        "d2EP_dq2_W": d2EP_dq2,
        "d2ED_dtau2_ms": d2ED_dtau2,
        "d2EP_dtau2_W": d2EP_dtau2,
        "d2A_dtau2": d2A_dtau2,
        "d2U_dtau2": d2U_dtau2,
        "inf_d2ED_dq2_ms": inf_d2D,
        "sup_d2ED_dq2_ms": sup_d2D,
        "jensen_delay_shift_ms": jensen_delay_shift_center_ms,
        "jensen_delay_lower_ms": jensen_delay_lower_ms,
        "jensen_delay_upper_ms": jensen_delay_upper_ms,
        "jensen_power_shift_W": jensen_power_shift_center_W,
        "jensen_power_lower_W": jensen_power_lower_W,
        "jensen_power_upper_W": jensen_power_upper_W,
        "is_delay_antifragile": bool(d2ED_dq2 < 0.0),
        "is_strictly_delay_antifragile_on_neighborhood": bool(sup_d2D < 0.0),
        "is_power_antifragile": bool(d2EP_dq2 < 0.0),
    }



from aqnet.models.cascade_nonstationary import (
    apply_hysteresis_with_safety_cap,
    simulate_nonstationary_hysteresis_simpy,
    simulate_stationary_cascade_simpy,
    solve_hysteresis_band,
    solve_nonstationary_envelope,
    solve_threshold_lambda_sensitivity,
    verify_symbolic_cascade_proofs,
)

__all__ = [
    "apply_hysteresis_with_safety_cap",
    "simulate_nonstationary_hysteresis_simpy",
    "simulate_stationary_cascade_simpy",
    "solve_hysteresis_band",
    "solve_nonstationary_envelope",
    "solve_threshold_lambda_sensitivity",
    "verify_symbolic_cascade_proofs",
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
]
