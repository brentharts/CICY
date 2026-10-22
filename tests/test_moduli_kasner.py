"""
Tests for pyCICY.moduli_kasner: the exponent budget for kinetic-dominated Kahler moduli.

  [1] metric       closed forms for G_ij and K_ijk against sympy; exact rationals; the dilation Killing norm 3/2
  [2] budget       sum p = 1, sum p^2 = 1 - |s|^2, |s|^2 <= 2/3 and the isotropic point
  [3] geodesics    |s| conserved; dilation rays are geodesics with lambda ~ e^(sigma u)
  [4] independent  Euler-Lagrange integration of the minisuperspace Lagrangian agrees with the algebra
  [5] curvature    Kretschmann coefficient against a symbolic Riemann tensor
  [6] registry     the entry and its ledger
  [7] dilaton      the heterotic dilaton as a product factor; alpha_GUT and Lambda_QCD against pyCICY.theories.running

Run with:  python3 tests/test_moduli_kasner.py
"""
import os
import sys
from fractions import Fraction

import numpy as np
import pytest
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyCICY import moduli_kasner as mk
from pyCICY import theories as T
from pyCICY.kasner import diagonal_curvature

QUINTIC = [[4, 5]]
BICUBIC = [[2, 3], [2, 3]]


@pytest.fixture(scope="module")
def d_quintic():
    return mk.triple_numbers(QUINTIC)


@pytest.fixture(scope="module")
def d_bicubic():
    return mk.triple_numbers(BICUBIC)


def _sympy_metric(d, t0):
    ts = sp.symbols("t0:%d" % d.shape[0], positive=True)
    V = sum(int(d[i, j, k]) * ts[i] * ts[j] * ts[k] for i in range(len(ts)) for j in range(len(ts))
            for k in range(len(ts)))
    K = -sp.log(V)
    sub = dict(zip(ts, t0))
    G = np.array(sp.hessian(K, ts).subs(sub).evalf(), dtype=float) / 2
    K3 = np.array([[[sp.diff(K, ts[a], ts[b], ts[c]).subs(sub).evalf() for c in range(len(ts))]
                    for b in range(len(ts))] for a in range(len(ts))], dtype=float)
    return G, K3


# ---------------------------------------------------------------- [1] metric

@pytest.mark.parametrize("name,t", [("d_quintic", [0.7]), ("d_bicubic", [1.3, 0.8])])
def test_closed_forms_match_sympy(name, t, request):
    d = request.getfixturevalue(name)
    G, K3 = _sympy_metric(d, t)
    assert np.allclose(mk.moduli_metric(d, t), G, atol=1e-12)
    assert np.allclose(mk.metric_third_derivative(d, t), K3, atol=1e-10)


def test_exact_metric_equals_float(d_bicubic):
    t = [Fraction(13, 10), Fraction(4, 5)]
    Ge = mk.moduli_metric_exact(d_bicubic, t)
    assert all(isinstance(x, Fraction) for row in Ge for x in row)
    assert np.allclose(np.array(Ge, dtype=float), mk.moduli_metric(d_bicubic, [1.3, 0.8]))


@pytest.mark.parametrize("t", [[0.5, 1.1], [2.0, 0.3], [1.0, 1.0]])
def test_dilation_has_norm_three_halves(d_bicubic, t):
    assert mk.speed2(d_bicubic, t, t) == pytest.approx(1.5, abs=1e-12)


def test_quintic_is_a_line(d_quintic):
    assert d_quintic.shape == (1, 1, 1) and d_quintic[0, 0, 0] == 5
    assert mk.moduli_metric(d_quintic, [2.0])[0, 0] == pytest.approx(1.5 / 4.0)


def test_outside_cone_is_refused():
    d = np.zeros((1, 1, 1), dtype=int)
    d[0, 0, 0] = 5
    with pytest.raises(ValueError):
        mk.moduli_metric(d, [-1.0])


# ---------------------------------------------------------------- [2] budget

@pytest.mark.parametrize("sp2", [0, Fraction(1, 10), Fraction(1, 3), Fraction(1, 2), Fraction(2, 3)])
@pytest.mark.parametrize("theta", [0.0, 0.7, 2.0])
def test_budget_identities(sp2, theta):
    p = mk.exponents_from_speed2(sp2, theta)
    assert sum(p) == pytest.approx(1.0, abs=1e-13)
    assert sum(x * x for x in p) == pytest.approx(1.0 - float(sp2), abs=1e-13)


def test_isotropic_point_saturates():
    assert mk.max_speed2() == Fraction(2, 3)
    assert mk.exponents_from_speed2(Fraction(2, 3)) == pytest.approx((1 / 3,) * 3)


def test_over_budget_is_refused():
    with pytest.raises(ValueError):
        mk.exponents_from_speed2(Fraction(7, 10))


def test_vacuum_is_kasner():
    p = mk.exponents_from_speed2(0, 0.3)
    assert sum(p) == pytest.approx(1) and sum(x * x for x in p) == pytest.approx(1)


def test_dilation_bound():
    assert mk.speed2_of_dilation(Fraction(2, 3)) == Fraction(2, 3)
    assert mk.dilation_sigma(2.0 / 3.0) == pytest.approx(2.0 / 3.0)


def test_dilaton_product_rule():
    assert mk.product_speed2(0.3, 0.4) == pytest.approx(0.3 + 0.08)


# ---------------------------------------------------------------- [3] geodesics

def test_speed_conserved(d_bicubic):
    g = mk.moduli_geodesic(d_bicubic, [1.0, 1.0], [0.3, -0.2], (0, 3))
    assert np.ptp(g["speed2"]) < 1e-9


def test_dilation_ray_is_a_geodesic(d_bicubic):
    sigma, u1 = 0.3, 3.0
    g = mk.moduli_geodesic(d_bicubic, [1.0, 1.0], [sigma, sigma], (0, u1))
    assert g["t"][-1] / g["t"][0] == pytest.approx([np.exp(sigma * u1)] * 2, rel=1e-8)
    assert g["speed2"][0] == pytest.approx(1.5 * sigma ** 2, rel=1e-10)


# ---------------------------------------------------------------- [4] independent route

@pytest.mark.parametrize("theta", [0.4, 1.9])
def test_einstein_scalar_agrees_with_the_budget(d_bicubic, theta):
    s = np.array([0.2, 0.1])
    s2 = float(s @ mk.moduli_metric(d_bicubic, [1, 1]) @ s)
    p = mk.exponents_from_speed2(s2, theta)
    r = mk.einstein_scalar_check(d_bicubic, [1, 1], s, p, t_end=50.0)
    assert abs(r["constraint"]) < 1e-10
    assert r["speed2"] == pytest.approx(s2, rel=1e-8)
    assert r["p"] == pytest.approx(p, abs=1e-8)       # the exponents are conserved
    assert r["p"].sum() == pytest.approx(1.0, abs=1e-8)
    assert (r["p"] ** 2).sum() == pytest.approx(1.0 - s2, abs=1e-8)


# ---------------------------------------------------------------- [5] curvature

@pytest.mark.parametrize("p", [(Fraction(1, 3), Fraction(1, 2), Fraction(1, 6)),
                               (Fraction(1, 3),) * 3,
                               (Fraction(-1, 3), Fraction(2, 3), Fraction(2, 3))])
def test_kretschmann_coefficient(p):
    t, x, y, z = sp.symbols("t x y z", positive=True)
    r = diagonal_curvature([-1] + [t ** (2 * sp.Rational(q.numerator, q.denominator)) for q in p], [t, x, y, z])
    assert sp.simplify(r["kretschmann"] * t ** 4 - sp.Rational(mk.kretschmann_coefficient(p).numerator,
                                                              mk.kretschmann_coefficient(p).denominator)) == 0


def test_vacuum_coefficient_is_minus_16_p1p2p3():
    p = (Fraction(-1, 3), Fraction(2, 3), Fraction(2, 3))
    assert mk.kretschmann_coefficient(p) == -16 * p[0] * p[1] * p[2]


# ---------------------------------------------------------------- [6] registry

def test_registry_entry(d_bicubic):
    E = T.get("moduli-kasner")(BICUBIC)
    assert issubclass(E.__class__, T.CosmologyTheory)
    text = E.describe()
    assert "exact:" in text and "declined:" in text
    assert E.budget() == Fraction(2, 3)
    assert sum(E.exponents(0.3, 1.0)) == pytest.approx(1.0)
    assert E.horizon(0.0, 0.0)
    with pytest.raises(T.NeedsFit):
        E.physical_prediction()


def test_entry_needs_a_geometry():
    with pytest.raises(ValueError):
        T.get("moduli-kasner")()


# ---------------------------------------------------------------- [7] the heterotic dilaton

def test_dilaton_metric_is_one_over_two_s_squared():
    s = sp.Symbol("s", positive=True)
    assert sp.simplify(sp.diff(-sp.log(s), s, 2) / 2 - 1 / (2 * s ** 2)) == 0
    assert mk.dilaton_speed2(1) == Fraction(1, 2)                  # in ln s the line element is (1/2) (d ln s)^2


def test_joint_dilation_norm_and_bounds():
    assert mk.joint_dilation_norm2() == Fraction(3, 2) + mk.dilaton_speed2(1)
    assert mk.joint_dilation_norm2() * mk.joint_sigma_max2() == mk.max_speed2()
    assert mk.dilaton_sigma_max2() / 2 == mk.max_speed2()           # dilaton alone: sigma^2 / 2 = 2/3


def test_dilaton_alone_saturates_into_the_isotropic_point():
    p = mk.exponents_from_speed2(mk.dilaton_speed2(float(mk.dilaton_sigma_max2()) ** 0.5))
    assert p == pytest.approx((1 / 3,) * 3, abs=1e-7)


def test_product_speed2_adds():
    assert mk.product_speed2(0.1, 0.6) == pytest.approx(0.1 + float(mk.dilaton_speed2(0.6)))


@pytest.mark.parametrize("sigma", [0.5, -0.8])
def test_einstein_scalar_with_dilaton(d_bicubic, sigma):
    s = np.array([0.2, 0.1])
    sK = float(s @ mk.moduli_metric(d_bicubic, [1, 1]) @ s)
    tot = sK + float(mk.dilaton_speed2(sigma))
    p = mk.exponents_from_speed2(tot, 0.9)
    s0 = 1.7
    r = mk.einstein_scalar_check(d_bicubic, [1, 1], s, p, t_end=50.0, dilaton=(s0, s0 * sigma))
    assert abs(r["constraint"]) < 1e-10
    assert r["speed2"] == pytest.approx(tot, rel=1e-8)
    assert r["sigma_s"] == pytest.approx(sigma, abs=1e-8)           # separately conserved: a product metric
    assert r["s"] == pytest.approx(s0 * 51.0 ** sigma, rel=1e-8)    # Re S ~ (t - t_s)^sigma, t_s = -1
    assert r["p"] == pytest.approx(p, abs=1e-8)
    assert (r["p"] ** 2).sum() == pytest.approx(1.0 - tot, abs=1e-8)


def test_alpha_gut_and_qcd_ratios_match_the_running_module():
    from pyCICY.theories.moduli import alpha_gut_from_dilaton
    from pyCICY.theories import running
    s0, sigma, F = 1.5, 0.7, 3.0
    a0, a1 = alpha_gut_from_dilaton(s0), alpha_gut_from_dilaton(s0 * F ** sigma)
    assert a1 / a0 == pytest.approx(mk.alpha_gut_ratio(sigma, F), rel=1e-12)
    ratio = running.lambda_qcd(a1)["lambda_qcd"] / running.lambda_qcd(a0)["lambda_qcd"]
    assert ratio == pytest.approx(mk.qcd_scale_ratio(s0, sigma, F), rel=1e-9)


def test_qcd_ratio_has_the_published_sensitivity():
    from pyCICY.theories.moduli import alpha_gut_from_dilaton
    from pyCICY.theories import running
    s0, h = 1.0 / (4 * np.pi / 25.0), 1e-6                          # alpha_GUT = 1/25
    dlnL = (np.log(mk.qcd_scale_ratio(s0, 1.0, np.exp(h))) - np.log(mk.qcd_scale_ratio(s0, 1.0, np.exp(-h)))) / (2 * h)
    sens = running.sensitivity(alpha_gut=alpha_gut_from_dilaton(s0))["dlnLambda_dlnalpha"]
    assert dlnL == pytest.approx(-sens, rel=1e-6)                   # alpha ~ 1/s, so d ln Lambda / d ln s = -sensitivity
    assert sens == pytest.approx(52.4, abs=0.1)


def test_racetrack_budget_is_symmetric_and_flagged():
    r = mk.racetrack_kinetic_budget(4, 5, ratio=10.0, F=2.0)
    assert r["alpha_ratio_up"] * r["alpha_ratio_down"] == pytest.approx(1.0)
    assert r["lambda_ratio_up"] < 1.0 < r["lambda_ratio_down"]
    assert r["sigma_max"] == pytest.approx(2 / np.sqrt(3))
    assert r["re_s0"] > 0 and r["alpha_gut0"] > 0


def test_stabilised_dilaton_leaves_the_kahler_budget_alone(d_bicubic):
    E = T.get("moduli-kasner")(BICUBIC)
    assert E.with_dilaton(0.5, 0.0) == dict(speed2=0.5, allowed=True)
    assert E.with_dilaton(0.5, 1.0)["allowed"] is False               # 0.5 + 1/2 > 2/3
    assert E.with_dilaton(0.0, 2 / np.sqrt(3))["allowed"] is True
    assert "alpha_ratio_up" in E.racetrack_drift(4, 5)
    assert any("racetrack" in x for x in E.declined())


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
