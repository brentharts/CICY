"""
Tests for pyCICY.sixsphere_cosmology: two matches that are exact, generic and not connections.

  [1] bkl        the closed orbits of 3 + sqrt15 and 2 + sqrt5, multipliers lambda^4 and -nu^2, D two ways
  [2] generic    every periodic continued fraction closes; D(u*) = 7 lambda^2 is a one-off
  [3] pancake    the fibre clock misses 6, and the a0 that hits it is a tuned value
  [4] orbifold   stabilisers 6, 2, 3 and chi = 0 for the hexagonal torus modulo Z/6
  [5] status     every check passes and the facts the paper quotes are as stated

Run with:  python3 tests/test_sixsphere_cosmology.py
"""
import os
import sys

import pytest
import sympy as sp

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from pyCICY import sixsphere_cosmology as C


def test_spectre_orbit():
    us, eras, mult = C.closed_orbit(3 + C.S15)
    assert len(us) == 7 and eras == [6, 1]
    assert sp.simplify(mult - C.LAMBDA2 ** 2) == 0


def test_nu_orbit():
    us, eras, mult = C.closed_orbit(C.NU)
    assert len(us) == 4 and eras == [4] and sp.simplify(mult + C.NU ** 2) == 0


def test_dimension_two_ways():
    a, b = C.dimension_two_ways()
    assert a == pytest.approx(b, abs=1e-12) and a == pytest.approx(1.3993, abs=5e-5)


@pytest.mark.parametrize("ab", [(1, 6), (2, 3), (3, 1), (1, 4), (2, 5), (4, 3)])
def test_every_period_closes(ab):
    a, b = ab
    _, eras, mult = C.closed_orbit(C.periodic_u(a, b))
    assert sorted(eras) == sorted([a, b])
    assert sp.simplify(abs(mult) - C.unit_of(a, b) ** 2) == 0


@pytest.mark.parametrize("ab", [(1, 6), (2, 3), (3, 2), (2, 5)])
def test_denominator_coincidence_is_a_one_off(ab):
    u = C.periodic_u(*ab)
    assert not sp.radsimp((1 + u + u ** 2) / C.unit_of(*ab)).is_rational


def test_fibre_clock_misses_six():
    N1 = sp.N(C.levels_per_turn(1), 30)
    assert 6.08 < N1 < 6.10 and abs(N1 - 6) > 0.08
    assert abs(sp.N(C.a0_for_levels_per_turn(6)) - 1) > 0.01


def test_orbifold():
    st, orbits = C.stabilisers()
    assert sorted(st.values()) == [2, 2, 2, 3, 3, 6] and sorted(orbits.values()) == [2, 3, 6]


def test_status_all_pass_and_facts():
    s = C.status()
    bad = [k for k, v in s["checks"].items() if not v]
    assert not bad, bad
    f = s["facts"]
    assert (f["Period"], f["Eras"], f["NuPeriod"], f["NuEras"]) == (7, "6,1", 4, "4")
    assert f["Dim"] == "1.3993" and f["LevelsPerTurn"] == "6.090" and f["OrbifoldChi"] == 0
    assert f["StabOrders"] == "6, 3, 2" and f["OneOffStarts"] == 4


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
