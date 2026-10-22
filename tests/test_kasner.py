"""Kasner vacuum cosmology, with two independent symbolic routes to its curvature.

pyCICY.kasner.sympy_curvature writes Christoffel symbols, Riemann and Ricci out in sympy from the metric;
pyCICY.kasner.einsteinpy_curvature takes the same tensors from EinsteinPy. Every curvature statement is checked on
both, the routes are checked against each other, and the closed form is the third check. EinsteinPy is installed by
default; the einsteinpy route is skipped if it is missing. All tests are fast (<~30 s).
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

sp = pytest.importorskip("sympy")

from pyCICY import kasner as qk

R = sp.Rational
EXACT_U = [R(1), R(3, 2), R(2), R(5)]
ROUTES = ["sympy", pytest.param("einsteinpy", marks=pytest.mark.skipif(
    not qk.has_einsteinpy(), reason="EinsteinPy not installed"))]


def curvature(route, p):
    return (qk.sympy_curvature if route == "sympy" else qk.einsteinpy_curvature)(p)


# ---------------------------------------------------------------- exponents

@pytest.mark.parametrize("u", [1.0, 1.5, 2.0, 5.0, 10.0, 100.0])
def test_parametrisation_satisfies_both_kasner_conditions(u):
    p = qk.kasner_exponents(u)
    s1, s2 = qk.kasner_conditions(p)
    assert s1 == pytest.approx(1.0, abs=1e-12)
    assert s2 == pytest.approx(1.0, abs=1e-12)
    assert qk.is_vacuum_kasner(p)
    assert p[0] <= 0.0 <= p[1] <= p[2]


def test_special_values_of_the_parametrisation():
    assert qk.kasner_exponents(R(1)) == (R(-1, 3), R(2, 3), R(2, 3))
    assert qk.kasner_exponents(R(2)) == (R(-2, 7), R(3, 7), R(6, 7))
    p = qk.kasner_exponents(1.0e6)                       # u -> infinity approaches flat (0, 0, 1)
    assert p == pytest.approx((0.0, 0.0, 1.0), abs=1e-5)


def test_u_below_one_is_rejected():
    with pytest.raises(ValueError):
        qk.kasner_exponents(0.5)


def test_isotropic_exponents_are_not_vacuum():
    p = (R(1, 3), R(1, 3), R(1, 3))
    assert sum(p) == 1 and not qk.is_vacuum_kasner(p)    # sphere condition fails


# ---------------------------------------------------------------- symbolic curvature

@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("u", EXACT_U)
def test_ricci_vanishes_for_every_admissible_exponent_set(u, route):
    assert curvature(route, qk.kasner_exponents(u))["ricci_is_zero"]


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("u", EXACT_U)
def test_kretschmann_is_minus_16_p1p2p3_over_t4(u, route):
    p = qk.kasner_exponents(u)
    c = curvature(route, p)
    assert sp.simplify(c["kretschmann"] - qk.kretschmann_closed_form(p, c["t"])) == 0
    assert sp.simplify(c["kretschmann"] * c["t"] ** 4).free_symbols == set()   # exactly t^-4
    assert c["kretschmann"].subs(c["t"], 1) > 0                                # p1 < 0 makes K positive


@pytest.mark.parametrize("route", ROUTES)
def test_flat_case_has_no_curvature(route):
    c = curvature(route, (0, 0, 1))
    assert c["ricci_is_zero"] and c["riemann_is_zero"]
    assert c["kretschmann"] == 0


@pytest.mark.parametrize("route", ROUTES)
@pytest.mark.parametrize("u", EXACT_U)
def test_any_other_vacuum_solution_is_genuinely_curved(u, route):
    c = curvature(route, qk.kasner_exponents(u))
    assert not c["riemann_is_zero"]                      # vacuum but curved: pure Weyl curvature


@pytest.mark.parametrize("route", ROUTES)
def test_isotropic_non_vacuum_has_nonzero_ricci(route):
    c = curvature(route, (R(1, 3), R(1, 3), R(1, 3)))
    assert not c["ricci_is_zero"]


@pytest.mark.skipif(not qk.has_einsteinpy(), reason="EinsteinPy not installed")
@pytest.mark.parametrize("p", [(R(1, 3), R(1, 3), R(1, 3)), (R(1, 2), R(1, 3), R(1, 5))])
def test_the_two_symbolic_routes_agree_even_off_the_vacuum_surface(p):
    """Not vacuum, so K has no closed form: the routes can only be checked against each other."""
    a, b = qk.sympy_curvature(p), qk.einsteinpy_curvature(p)
    assert sp.simplify(a["kretschmann"] - b["kretschmann"]) == 0
    assert a["ricci_is_zero"] == b["ricci_is_zero"] and a["riemann_is_zero"] == b["riemann_is_zero"]


def test_kasner_registry_entry_agrees_with_the_modules():
    from pyCICY import theories as T
    k = T.get("kasner-vacuum")(2)
    v = k.verify_curvature(einsteinpy=qk.has_einsteinpy())
    assert v["closed"] == v["sympy"] == R(576, 343)
    assert v["sympy_ricci_zero"]
    if qk.has_einsteinpy():
        assert v["einsteinpy"] == R(576, 343) and v["einsteinpy_ricci_zero"]


# ---------------------------------------------------------------- volume

@pytest.mark.parametrize("u", EXACT_U)
def test_volume_grows_linearly_in_t(u):
    assert qk.kasner_volume_exponent(qk.kasner_exponents(u)) == 1


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
