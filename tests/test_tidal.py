"""Tidal tensors (pyCICY.tidal): closed forms, vacuum identities, and finite tides at the pancake."""
import os
import sys
import warnings

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
warnings.filterwarnings("ignore")

from pyCICY import bkl as bk
from pyCICY import kasner as qk
from pyCICY import pancake as qp
from pyCICY import tidal as qt


@pytest.mark.parametrize("u", [1.0, 2.0, 5.0])
def test_kasner_tides_are_traceless_and_match_second_derivatives(u):
    p = np.array(qk.kasner_exponents(u))
    t = np.array([0.05, 0.5, 3.0])
    lam = qt.kasner_tidal(p, t)
    assert lam.sum(axis=1) == pytest.approx(0.0, abs=1e-12 * np.max(np.abs(lam)))     # vacuum: R_00 = 0
    a = t[:, None] ** p[None, :]
    add = (p * (p - 1.0))[None, :] * t[:, None] ** (p[None, :] - 2.0)
    assert qt.tidal_from_scale_factors(a, add) == pytest.approx(lam, rel=1e-12)
    assert np.all(lam[:, 0] > 0)                           # p1 < 0 axis is stretched
    assert np.all(lam[:, 1:] < 0)                                                      # the two others are squeezed


def test_kasner_tides_diverge_like_one_over_t_squared():
    p = qk.kasner_exponents(2.0)
    T = qt.tidal_strength(qt.kasner_tidal(p, np.array([1e-1, 1e-2, 1e-3])))
    assert T[1] / T[0] == pytest.approx(100.0, rel=1e-10) and T[2] / T[1] == pytest.approx(100.0, rel=1e-10)


def test_mixmaster_tides_are_traceless_and_K_differs_from_8T2_by_the_magnetic_part():
    mm = bk.Mixmaster(3.7, 1e6, 1.0, 1e-60)
    tb = float(mm.bounce_times()[0])
    t = np.geomspace(5 * tb, tb / 5, 25)
    a, ad, add = mm.scale_factors(t)
    lam = qt.tidal_from_scale_factors(a, add)
    scale = np.max(np.abs(lam), axis=1)
    assert np.all(np.abs(lam.sum(axis=1)) < 1e-6 * scale)                             # vacuum on-shell
    K = bk.bianchi9_kretschmann(a, ad, add)
    B2 = np.sum(bk._codazzi(a, ad) ** 2, axis=-1)
    assert K == pytest.approx(8.0 * qt.tidal_strength(lam) ** 2 - 8.0 * B2, rel=1e-6)
    assert np.max(B2) > 1e-3 * np.max(qt.tidal_strength(lam) ** 2)                    # B really contributes


def test_pancake_tides_stay_finite_while_the_volume_vanishes():
    b0, f0, m, lam_c = 3.0, 2.0, 0.5, 0.1
    sol = qp.pancake_solution(1.0, b0, f0, m, lam_c, t_max=4.0)
    lam = qt.pancake_tidal(sol)
    T = qt.tidal_strength(lam)
    T0 = float(qt.tidal_strength(qt.pancake_tidal_at_zero(b0, f0, m, lam_c)))
    near = np.abs(sol["t"]) < 1e-2
    assert T[near] == pytest.approx(T0, rel=2e-3)
    assert np.all(np.isfinite(T)) and T.max() < 10.0 * T0
    assert qt.pancake_tidal_at_zero(b0, f0, m, lam_c)[0] == pytest.approx(4.0 / b0 ** 2)
    # a vacuum Kasner cigar with comparable curvature scale has unbounded tides at the same epochs
    cigar = qt.tidal_strength(qt.kasner_tidal(qk.kasner_exponents(2.0), np.array([1e-2, 1e-4])))
    assert cigar[1] / cigar[0] == pytest.approx(1e4, rel=1e-9) and cigar[1] > 1e6 * T0


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-q", "-p", "no:cacheprovider"]))
