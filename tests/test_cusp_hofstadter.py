"""
Tests for pyCICY.cusp_hofstadter (Part X: electrons on the cusp fibre's
substrate, and a magnetic field as a line bundle on the cusp torus).

  [1] substrate     6 sites (C, 3 E, 2 V) at torsion points, 12 bonds, 6 kites
  [2] bands         the characteristic polynomial in A2 invariants; det B is
                    the Weyl denominator; E = 0 exactly on the mirror lines;
                    +-sqrt 18 at Gamma
  [3] field         every kite carries flux phi; periods 1 (substrate) and
                    1/2 (Hat, faces of 8 and 10 kites)
  [4] Chern         FHS on the magnetic Bloch bundle, the TKNN relation in
                    every gap, and Kitaev's real-space number on an open patch
                    agreeing with it on the wide gaps
  [5] controls      a wrong flux breaks the kite check; Kitaev in a band is
                    not an integer; the A2 form with + |Delta|^2 fails

The Hat side needs brentharts/spectre and lives in
paper/make_sixsphere_hat_hofstadter.py.

Run with:  python3 tests/test_cusp_hofstadter.py   or   python3 run_tests.py
"""

import os
import sys
from fractions import Fraction as Fr
from math import sqrt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sympy import simplify, symbols

from pyCICY import cusp_hofstadter as C

FAILURES = []


def check(name, got, want):
    ok = got == want
    print("  {:<58} {:>12} {}".format(name, str(got)[:12],
                                      "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<58} {:>12} {}".format(name, str(bool(cond)),
                                      "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


def test_substrate():
    print("\n[1] the substrate on the cusp torus")
    check("sites per domain", len(C.SITES), 6)
    check("kinds", sorted(C.SITE_KIND), ["C", "E", "E", "E", "V", "V"])
    check("bonds per domain", len(C.BONDS), 12)
    E = {s for s, k in zip(C.SITES, C.SITE_KIND) if k == "E"}
    V = {s for s, k in zip(C.SITES, C.SITE_KIND) if k == "V"}
    check_true("E sites are the 2-torsion points",
               E == {(Fr(1, 2), 0), (0, Fr(1, 2)), (Fr(1, 2), Fr(1, 2))})
    check_true("V sites are (1/3, 1/3), (2/3, 2/3)",
               V == {(Fr(1, 3), Fr(1, 3)), (Fr(2, 3), Fr(2, 3))})
    check("kites per domain", len(C.KITES), 6)


def test_bands():
    print("\n[2] the Bloch bands in A2 invariants")
    cp = C.bloch_charpoly()
    check_true("charpoly = E^6 - (12+s)E^4 - 3(s-6)E^2 - |Delta|^2",
               cp["a2_form"])
    check_true("det B = unit x (1-x)(1-y)(1-x/y)",
               cp["detB_is_unit_times_Delta"])
    E = symbols("E")
    check_true("at Gamma: E^4 (E^2 - 18)",
               simplify(cp["at_gamma"] - E ** 4 * (E ** 2 - 18)) == 0)
    check("B B^T row sums at k = 0", cp["row_sums"], [18, 18, 18])
    nl = C.nodal_lines(36)
    check("zero modes off the mirror lines", nl["off_walls"], 0)
    check("mirror-line points without a zero mode",
          nl["walls_without_zero"], 0)
    br = C.band_ranges(24)
    check_true("spectrum fills [-3 sqrt 2, 3 sqrt 2]",
               abs(br[0][0] + sqrt(18)) < 1e-9 and abs(br[5][1] - sqrt(18)) < 1e-9)


def test_field():
    print("\n[3] the field as a connection")
    for phi in (Fr(1, 7), Fr(2, 5), 0.3137):
        check_true("every kite carries flux %s" % phi,
                   all(abs(f - float(phi)) < 1e-12 for f in C.kite_fluxes(phi)))
    fp = C.flux_periods()
    check("substrate period", fp["substrate"]["period"], Fr(1))
    check("Hat period, faces 8 and 10 kites", fp["hat"]["period"], Fr(1, 2))
    import numpy as np
    H = C.magnetic_bloch(Fr(1, 3), 0.3, 1.1)
    check_true("magnetic Bloch matrix is Hermitian, 18 x 18",
               H.shape == (18, 18) and np.allclose(H, H.conj().T))


def test_chern():
    print("\n[4] Chern numbers two ways")
    gc = C.gap_cherns(Fr(1, 3), nk=14, nkgap=14)
    check("gaps at phi = 1/3", len(gc), 2)
    check("their Chern numbers (FHS)", [g["chern_int"] for g in gc], [-1, 1])
    check_true("FHS values are integers to 1e-6",
               all(abs(g["chern"] - g["chern_int"]) < 1e-6 for g in gc))
    for phi in (Fr(1, 3), Fr(1, 4), Fr(1, 6)):
        g = C.gap_cherns(phi, nk=12, nkgap=12)
        check_true("TKNN: rho - 6 phi sigma integer in every gap, phi = %s" % phi,
                   g and all(x["tknn_integer"] for x in g))
    pos, b = C.finite_patch(24.0)
    H = C.finite_hamiltonian(pos, b, Fr(1, 3))
    ks = [round(C.kitaev_chern(H, pos, sum(g["window"]) / 2, radius=12.0))
          for g in gc]
    check("Kitaev on an open patch, phi = 1/3", ks, [-1, 1])
    g4 = [g for g in C.gap_cherns(Fr(1, 4), nk=12, nkgap=12)
          if g["window"][1] - g["window"][0] > 0.3]
    H4 = C.finite_hamiltonian(pos, b, Fr(1, 4))
    k4 = [round(C.kitaev_chern(H4, pos, sum(g["window"]) / 2, radius=12.0))
          for g in g4]
    check("Kitaev = FHS on the wide gaps at phi = 1/4", k4,
          [g["chern_int"] for g in g4])


def test_controls():
    print("\n[5] controls")
    check_true("flux 1/7 is not flux 1/6 on a kite",
               any(abs(f - 1 / 6) > 1e-3 for f in C.kite_fluxes(Fr(1, 7))))
    pos, b = C.finite_patch(24.0)
    H = C.finite_hamiltonian(pos, b, Fr(1, 3))
    k = C.kitaev_chern(H, pos, 2.5, radius=12.0)
    check_true("Kitaev inside a band is not near an integer (%.2f)" % k,
               abs(k - round(k)) > 0.05)
    x, y, E = symbols("x y E")
    cp = C.bloch_charpoly()
    s = x + 1 / x + y + 1 / y + x / y + y / x
    D = (1 - x) * (1 - y) * (1 - x / y) * (1 - 1 / x) * (1 - 1 / y) * (1 - y / x)
    wrong = E ** 6 - (12 + s) * E ** 4 - 3 * (s - 6) * E ** 2 + D
    check_true("the A2 form with + |Delta|^2 is not the charpoly",
               simplify(cp["charpoly"] - wrong) != 0)


def main():
    for section in (test_substrate, test_bands, test_field, test_chern,
                    test_controls):
        try:
            section()
        except Exception as exc:
            print("  {:<58} {:>12} FAIL".format(section.__name__ + " raised",
                                                type(exc).__name__))
            print("      " + str(exc)[:300])
            FAILURES.append("%s raised %s" % (section.__name__,
                                              type(exc).__name__))
    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("all cusp_hofstadter tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
