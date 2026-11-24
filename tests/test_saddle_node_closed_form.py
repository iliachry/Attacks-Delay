"""
Closed-form saddle-node results checked against independent numerical root-finding.

Each closed form in ``aqnet.models.bifurcation`` (Wright-omega / Lambert-W_0 fold, critical
capacity, curvature, dual branches, critical attack probability) is recomputed here with
plain SciPy root-finding on ``throughput_curve_gamma`` only, so a regression in either the
closed form or the conservation map shows up as a mismatch.
"""

import doctest
import itertools

import numpy as np
import pytest
from scipy.optimize import brentq, minimize_scalar

import aqnet
from aqnet.models import bifurcation

GRID = list(itertools.product([0.0, 0.1, 0.25, 0.6], [2.0, 10.0, 50.0], [0.2, 1.0, 3.0]))
SINGLE_NODE = ("one_node_destruction", "one_node_modification")


def _numerical_fold(topology, p, mu, T, N=2):
    """Fold by root-finding on a central-difference Gamma'(Lambda)."""
    lam_max = aqnet.upper_attempt_rate_bound(topology, p, mu, N)

    def gamma(x):
        return aqnet.throughput_curve_gamma(x, topology, p, mu, T, N)

    def dgamma(x):
        h = 1e-7 * lam_max
        return (gamma(x + h) - gamma(x - h)) / (2.0 * h)

    lo, hi = 1e-6 * lam_max, lam_max * (1.0 - 1e-6)
    assert dgamma(lo) > 0.0 > dgamma(hi)
    lam_sn = brentq(dgamma, lo, hi, xtol=1e-13)
    return lam_sn, gamma(lam_sn)


# --------------------------------------------------------------------------------------
# fold_parameter_v: Wright-omega vs Lambert-W_0 vs the defining equation
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("mu,T", [(0.5, 0.5), (2.0, 1.0), (10.0, 1.0), (40.0, 5.0), (100.0, 3.0)])
def test_fold_parameter_wrightomega_equals_lambertw_and_solves_definition(mu, T):
    v_omega = aqnet.fold_parameter_v(mu, T)
    v_w0 = aqnet.fold_parameter_v(mu, T, method="lambertw")
    assert v_omega == pytest.approx(v_w0, rel=1e-12)
    # v exp(v) = exp(mu T + 1)  <=>  v + log(v) = mu T + 1
    assert v_omega + np.log(v_omega) == pytest.approx(mu * T + 1.0, rel=1e-13)
    # independent brentq on the defining equation
    v_root = brentq(lambda v: v + np.log(v) - (mu * T + 1.0), 1.0, mu * T + 2.0, xtol=1e-14)
    assert v_omega == pytest.approx(v_root, rel=1e-12)
    assert v_omega > 1.0


def test_fold_parameter_large_mu_t_and_errors():
    # Wright-omega stays finite where exp(mu T + 1) overflows
    v = aqnet.fold_parameter_v(1000.0, 1.0)
    assert np.isfinite(v) and v + np.log(v) == pytest.approx(1001.0, rel=1e-13)
    with pytest.raises(ValueError, match="overflows"):
        aqnet.fold_parameter_v(1000.0, 1.0, method="lambertw")
    with pytest.raises(ValueError, match="Unknown method"):
        aqnet.fold_parameter_v(10.0, 1.0, method="newton")  # type: ignore[arg-type]
    for bad_mu, bad_t in [(0.0, 1.0), (-1.0, 1.0), (10.0, 0.0), (float("nan"), 1.0), (10.0, float("inf"))]:
        with pytest.raises(ValueError):
            aqnet.fold_parameter_v(bad_mu, bad_t)


# --------------------------------------------------------------------------------------
# Cases 1 & 2: closed-form fold vs numerical root of Gamma'
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("topology", SINGLE_NODE)
@pytest.mark.parametrize("p,mu,T", GRID)
def test_closed_form_fold_matches_numerical_root(topology, p, mu, T):
    sn = aqnet.solve_saddle_node_bifurcation(topology, p=p, mu=mu, T=T)
    assert sn.closed_form_exact
    lam_sn_num, lam_crit_num = _numerical_fold(topology, p, mu, T)
    assert sn.Lambda_sn == pytest.approx(lam_sn_num, rel=1e-6)
    assert sn.lambda_crit_sn == pytest.approx(lam_crit_num, rel=1e-10)

    # bounded maximisation as a second, derivative-free check
    opt = minimize_scalar(
        lambda x: -aqnet.throughput_curve_gamma(x, topology, p, mu, T),
        bounds=(0.0, sn.Lambda_max),
        method="bounded",
        options={"xatol": 1e-12},
    )
    assert sn.lambda_crit_sn == pytest.approx(-opt.fun, rel=1e-10)
    assert sn.lambda_crit_sn >= -opt.fun * (1.0 - 1e-12)  # closed form is the true maximum

    # explicit formulas
    v = aqnet.fold_parameter_v(mu, T)
    expected_crit = ((v - 1.0) ** 2) / (T * v)
    if topology == "one_node_modification":
        expected_crit *= 1.0 - p
    assert sn.lambda_crit_sn == pytest.approx(expected_crit, rel=1e-13)
    assert sn.p_succ_at_sn == pytest.approx(sn.lambda_crit_sn / sn.Lambda_sn, rel=1e-13)
    assert 0.0 < sn.rho_sn < 1.0


@pytest.mark.parametrize("topology", SINGLE_NODE)
@pytest.mark.parametrize("p,mu,T", GRID)
def test_closed_form_curvature_matches_finite_difference(topology, p, mu, T):
    sn = aqnet.solve_saddle_node_bifurcation(topology, p=p, mu=mu, T=T)
    h = 1e-4 * min(sn.Lambda_sn, sn.Lambda_max - sn.Lambda_sn)

    def gamma(x):
        return aqnet.throughput_curve_gamma(x, topology, p, mu, T)

    d2_fd = (gamma(sn.Lambda_sn + h) - 2.0 * gamma(sn.Lambda_sn) + gamma(sn.Lambda_sn - h)) / h**2
    assert sn.d2Gamma_dLambda2 < 0.0
    assert sn.d2Gamma_dLambda2 == pytest.approx(d2_fd, rel=1e-4)
    assert sn.c2_normal_form == pytest.approx(-0.5 * sn.d2Gamma_dLambda2, rel=1e-15)


@pytest.mark.parametrize("mu,T", [(2.0, 1.0), (10.0, 1.0), (50.0, 0.2)])
def test_destruction_critical_capacity_is_p_invariant_numerically(mu, T):
    crits = [_numerical_fold("one_node_destruction", p, mu, T)[1] for p in (0.0, 0.2, 0.5, 0.8)]
    assert np.ptp(crits) / np.mean(crits) < 1e-9
    # and at p = 0 destruction and modification coincide
    mod0 = aqnet.solve_saddle_node_bifurcation("one_node_modification", p=0.0, mu=mu, T=T)
    assert crits[0] == pytest.approx(mod0.lambda_crit_sn, rel=1e-10)


@pytest.mark.parametrize("mu,T", [(2.0, 1.0), (10.0, 1.0)])
@pytest.mark.parametrize("frac", [0.2, 0.5, 0.9])
def test_modification_p_crit_matches_bisection(mu, T, frac):
    base = aqnet.solve_saddle_node_bifurcation("one_node_modification", p=0.0, mu=mu, T=T)
    lam_ext = frac * base.lambda_crit_sn
    p_closed = aqnet.solve_saddle_node_bifurcation(
        "one_node_modification", p=0.3, mu=mu, T=T
    ).p_crit_for_lambda(lam_ext)

    def excess(p):
        return _numerical_fold("one_node_modification", p, mu, T)[1] - lam_ext

    p_num = brentq(excess, 0.0, 0.999, xtol=1e-12)
    assert p_closed == pytest.approx(p_num, abs=1e-7)
    assert p_closed == pytest.approx(1.0 - frac, abs=1e-12)


# --------------------------------------------------------------------------------------
# Dual branches and square-root normal form
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("topology", SINGLE_NODE)
@pytest.mark.parametrize("frac", [0.1, 0.5, 0.9, 0.999])
def test_dual_fixed_points_match_independent_brentq(topology, frac):
    p, mu, T = 0.25, 10.0, 1.0
    sn = aqnet.solve_saddle_node_bifurcation(topology, p=p, mu=mu, T=T)
    lam_ext = frac * sn.lambda_crit_sn
    dual = aqnet.solve_dual_fixed_points(lam_ext, topology, p=p, mu=mu, T=T)

    def resid(x):
        return aqnet.throughput_curve_gamma(x, topology, p, mu, T) - lam_ext

    lo = brentq(resid, 1e-12, sn.Lambda_sn, xtol=1e-14)
    hi = brentq(resid, sn.Lambda_sn, sn.Lambda_max * (1 - 1e-12), xtol=1e-14)
    assert dual.is_stable and dual.stability_regime == "subcritical_bistable"
    assert dual.Lambda_stable == pytest.approx(lo, rel=1e-9)
    assert dual.Lambda_unstable == pytest.approx(hi, rel=1e-9)
    assert 0.0 < dual.Lambda_stable < sn.Lambda_sn < dual.Lambda_unstable < sn.Lambda_max
    assert dual.separatrix_gap == pytest.approx(hi - lo, rel=1e-8)

    # sensitivity = 1 / Gamma'(Lambda_stable)
    h = 1e-6 * sn.Lambda_max
    g_prime = (resid(lo + h) - resid(lo - h)) / (2 * h)
    assert dual.delay_sensitivity_dLambda_dlambda == pytest.approx(1.0 / g_prime, rel=1e-3)


@pytest.mark.parametrize("topology", SINGLE_NODE)
def test_sqrt_normal_form_error_vanishes_faster_than_branch_splitting(topology):
    p, mu, T = 0.25, 10.0, 1.0
    sn = aqnet.solve_saddle_node_bifurcation(topology, p=p, mu=mu, T=T)
    rel_errors = []
    for margin_frac in (1e-2, 1e-4, 1e-6):
        margin = margin_frac * sn.lambda_crit_sn
        dual = aqnet.solve_dual_fixed_points(sn.lambda_crit_sn - margin, topology, p=p, mu=mu, T=T)
        half_split = np.sqrt(margin / sn.c2_normal_form)
        err = max(
            abs(dual.Lambda_stable - dual.sqrt_asymptotic_stable),
            abs(dual.Lambda_unstable - dual.sqrt_asymptotic_unstable),
        )
        rel_errors.append(err / half_split)
        # branch splitting itself scales like sqrt(margin)
        assert dual.separatrix_gap == pytest.approx(2.0 * half_split, rel=0.05)
    # leading correction is O(sqrt(margin)) relative, so errors shrink ~10x per 100x margin
    assert rel_errors[0] > rel_errors[1] > rel_errors[2]
    assert rel_errors[2] < 1e-2


def test_dual_fixed_points_supercritical_and_fold():
    sn = aqnet.solve_saddle_node_bifurcation("one_node_modification", p=0.25, mu=10.0, T=1.0)
    over = aqnet.solve_dual_fixed_points(sn.lambda_crit_sn * 1.01, "one_node_modification", 0.25, 10.0, 1.0)
    assert not over.is_stable and over.stability_regime == "supercritical_collapse"
    assert np.isnan(over.Lambda_stable) and np.isnan(over.Lambda_unstable)
    at = aqnet.solve_dual_fixed_points(sn.lambda_crit_sn, "one_node_modification", 0.25, 10.0, 1.0)
    assert at.is_critical_fold and at.Lambda_stable == at.Lambda_unstable == sn.Lambda_sn
    with pytest.raises(ValueError):
        aqnet.solve_dual_fixed_points(-1.0)


# --------------------------------------------------------------------------------------
# Cases 3-5: numerical fold is a genuine maximum
# --------------------------------------------------------------------------------------


@pytest.mark.parametrize("topology", ["tandem", "feedforward", "feedback"])
@pytest.mark.parametrize("p,N", [(0.1, 2), (0.25, 3)])
def test_multinode_fold_is_maximum_and_matches_root_of_derivative(topology, p, N):
    mu, T = 10.0, 1.0
    sn = aqnet.solve_saddle_node_bifurcation(topology, p=p, mu=mu, T=T, N=N)
    assert not sn.closed_form_exact
    lam_sn_num, lam_crit_num = _numerical_fold(topology, p, mu, T, N)
    assert sn.Lambda_sn == pytest.approx(lam_sn_num, rel=1e-4)
    assert sn.lambda_crit_sn == pytest.approx(lam_crit_num, rel=1e-9)
    xs = np.linspace(1e-6, sn.Lambda_max * (1 - 1e-6), 400)
    gam = [aqnet.throughput_curve_gamma(x, topology, p, mu, T, N) for x in xs]
    assert max(gam) <= sn.lambda_crit_sn * (1 + 1e-12)
    assert sn.c2_normal_form > 0.0


# --------------------------------------------------------------------------------------
# Docs stay executable
# --------------------------------------------------------------------------------------


def test_bifurcation_docstring_examples():
    result = doctest.testmod(bifurcation, optionflags=doctest.ELLIPSIS)
    assert result.attempted >= 6
    assert result.failed == 0


def test_public_api_exports():
    for name in (
        "fold_parameter_v",
        "solve_saddle_node_bifurcation",
        "solve_dual_fixed_points",
        "throughput_curve_gamma",
        "success_probability_p_succ",
        "upper_attempt_rate_bound",
        "SaddleNodeBifurcationResult",
        "DualBranchFixedPointResult",
        "TopologyName",
    ):
        assert name in aqnet.__all__
        assert getattr(aqnet, name).__doc__ or name == "TopologyName"
