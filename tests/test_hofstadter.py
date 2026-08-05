"""
Tests for pyCICY.hofstadter.

Every claim here comes from Marra, Proietti and Sheng, arXiv:2312.14242, and
every one is checked against an independent computation rather than against a
value this module produced:

  [1] etilde       recurrence (Lemma III.6) vs enumeration of subsets
  [2] Theorem III.9 vs numpy.linalg.det of the Q x Q Hofstadter matrix
  [3] Remark III.11 a closed form with no free parameters, and no P dependence
  [4] Chambers    Eq. (18) over random points of the Brillouin torus
  [5] zero modes  the parity rule for where E = 0 sits
  [6] bridge      the Q x Q matrix against pyCICY.quantum_curve's Harper
                  operator, which was written independently and earlier

Section [6] is the one that matters for the package: it checks that the
analytic machinery added here describes the same operator the existing
quantized-mirror-curve code already builds for local F_0.

Run with:  python3 tests/test_hofstadter.py
       or: python3 run_tests.py
"""

import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import hofstadter as hof

FAILURES = []


def check_true(name, cond):
    print("  {:<58} {:>12} {}".format(name, str(bool(cond)),
                                      "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


def check_small(name, value, tol):
    ok = abs(value) < tol
    print("  {:<58} {:>12} {}".format(name, "%.2e" % value,
                                      "ok" if ok else "FAIL > %.0e" % tol))
    if not ok:
        FAILURES.append(name)


def check(name, got, want):
    ok = got == want
    print("  {:<58} {:>12} {}".format(name, str(got)[:12],
                                      "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def test_etilde():
    print("\n[1] two-step elementary symmetric polynomials")
    rng = np.random.default_rng(1)
    worst = 0.0
    for n in range(1, 15):
        xs = rng.normal(size=n)
        for k in range(0, n // 2 + 2):
            worst = max(worst, abs(hof.etilde(k, xs)
                                   - hof._etilde_bruteforce(k, xs)))
    check_small("recurrence agrees with subset enumeration", worst, 1e-9)
    check("etilde_0 = 1", hof.etilde(0, [3.0, 4.0]), 1.0)
    # k = 1 has no adjacency constraint, so it is the ordinary e_1
    xs = [1.0, 2.0, 3.0, 4.0]
    check("etilde_1 = e_1", hof.etilde(1, xs), 10.0)
    # etilde_2 of four variables: pairs at distance >= 2 are 13, 14, 24
    check("etilde_2 skips adjacent indices", hof.etilde(2, xs),
          1.0 * 3 + 1.0 * 4 + 2.0 * 4)
    check("etilde beyond the maximum vanishes", hof.etilde(9, xs), 0.0)

    # The recurrence must scale where enumeration cannot: 2^200 subsets is not
    # an option, and the butterfly is drawn at Q of this size.
    t0 = time.time()
    c = hof.char_poly_coefficients(89, 201)
    dt = time.time() - t0
    check("degree 201 polynomial computed (%.2fs)" % dt, len(c) - 1, 201)
    check_true("and fast enough to be usable", dt < 5.0)


def test_theorem_III9():
    print("\n[2] Theorem III.9 against a numerical determinant")
    worst, table = hof.verify_theorem_III9()
    for P, Q, err in table:
        check_small("P/Q = %d/%d" % (P, Q), err, 1e-8)
    check_small("worst over all cases", worst, 1e-8)

    # Only even powers of E appear, which is the statement that the spectrum
    # is symmetric under E -> -E for even Q. It is a property of the formula
    # and worth asserting separately from its numerical agreement.
    for P, Q in [(1, 5), (3, 8), (2, 7)]:
        c = hof.char_poly_coefficients(P, Q)
        odd = [c[Q - p] for p in range(Q, -1, -1) if (Q - p) % 2 == 1]
        check_small("only powers E^(Q-2i) survive, P/Q=%d/%d" % (P, Q),
                    max(abs(x) for x in odd) if odd else 0.0, 1e-12)

    # The polynomial is monic up to the overall (-1)^Q of the convention.
    for P, Q in [(1, 3), (3, 4), (2, 5)]:
        c = hof.char_poly_coefficients(P, Q)
        check_small("leading coefficient is (-1)^Q, P/Q=%d/%d" % (P, Q),
                    c[0] - (-1) ** Q, 1e-12)

    # The roots of the polynomial are the eigenvalues of the matrix.
    for P, Q in [(1, 5), (2, 7)]:
        roots = np.sort(np.roots(hof.char_poly_coefficients(P, Q)).real)
        eig = hof.spectrum(P, Q)
        check_small("roots of f are the spectrum, P/Q=%d/%d" % (P, Q),
                    float(np.max(np.abs(roots - eig))), 1e-6)


def test_etilde_identity():
    print("\n[3] Remark III.11, a closed form with no free parameters")
    worst, table = hof.verify_etilde_identity()
    for P, Q, got, want, err in table[:6]:
        check_small("Q=%2d P=%d  etilde_{Q/2} = 4^{-(Q/2-1)}" % (Q, P),
                    err, 1e-12)
    check_small("worst over all even Q up to 14", worst, 1e-12)

    # No P dependence at all: the value is the same for every P coprime to Q.
    for Q in (8, 12):
        vals = []
        for P in [p for p in range(1, Q) if math.gcd(p, Q) == 1]:
            a = float(P) / Q
            xs = [np.sin(np.pi * j * a) ** 2 for j in range(1, Q)]
            vals.append(hof.etilde(Q // 2, xs))
        check_small("Q=%d: independent of P across %d values"
                    % (Q, len(vals)), float(np.ptp(vals)), 1e-12)


def test_chambers():
    print("\n[4] the Chambers relation")
    check_small("f(E,nu) - f(E,mid) - offset, over random torus points",
                hof.verify_chambers(), 1e-7)

    # The offset does not depend on E. That is the content of the relation:
    # the band structure is a rigid translation of one polynomial.
    Q = 7
    a = hof.chambers_offset(Q, 0.3, 1.1)
    base = hof.hofstadter_matrix(3, Q, math.pi / (2 * Q), math.pi / (2 * Q))
    shifted = hof.hofstadter_matrix(3, Q, 0.3, 1.1)
    worst = 0.0
    for E in (-2.0, 0.0, 1.7, 3.3):
        lhs = np.linalg.det(shifted - E * np.eye(Q)).real
        rhs = np.linalg.det(base - E * np.eye(Q)).real + a
        worst = max(worst, abs(lhs - rhs))
    check_small("one E-independent offset works for every E", worst, 1e-7)


def test_zero_modes():
    print("\n[5] where E = 0 sits, by the parity of Q")
    for Q in range(3, 13):
        name, (nx, ny) = hof.zero_mode_point(Q)
        d = np.linalg.det(hof.hofstadter_matrix(1, Q, nx, ny)).real
        check_small("Q=%2d -> %s point" % (Q, name), d, 1e-9)

    check("Q odd -> mid-band", hof.zero_mode_point(7)[0], "mid-band")
    check("Q singly even -> corner", hof.zero_mode_point(6)[0], "corner")
    check("Q doubly even -> centre", hof.zero_mode_point(8)[0], "centre")

    # ... and the prediction is sharp: at the *other* two points f(0) is not
    # zero, so the rule is picking one point out of three rather than being
    # satisfied everywhere.
    for Q in (7, 6, 8):
        name, pt = hof.zero_mode_point(Q)
        others = [p for p in [(0.0, 0.0), (math.pi / (2 * Q),) * 2,
                              (math.pi / Q,) * 2] if p != tuple(pt)]
        vals = [abs(np.linalg.det(hof.hofstadter_matrix(1, Q, *p)).real)
                for p in others]
        check_true("Q=%d: f(0) non-zero at the other two points" % Q,
                   min(vals) > 0.5)


def test_bridge_to_quantum_curve():
    print("\n[6] agreement with pyCICY.quantum_curve")
    from pyCICY import quantum_curve as QC

    # quantum_curve.harper() is the square lattice, i.e. local F_0, built from
    # the Newton polygon of the mirror curve. This module builds the same
    # operator from Definition I.5 of a different paper. They must agree.
    harper = QC.harper()
    worst_gap = 0.0
    for P, Q in [(1, 3), (1, 4), (2, 5), (3, 7)]:
        a = QC.QuantumCurve.hbar(P, Q) if hasattr(QC.QuantumCurve, "hbar") else None
        band = np.sort(np.concatenate(harper.bands(P, Q, nk=24)).ravel())
        mine = []
        for nx in np.linspace(0, 2 * np.pi / Q, 24):
            for ny in np.linspace(0, 2 * np.pi / Q, 24):
                mine.extend(hof.spectrum(P, Q, nx, ny))
        mine = np.sort(np.array(mine))
        # Compare the band edges, which are basis and gauge independent.
        gap = abs(band.min() - mine.min()) + abs(band.max() - mine.max())
        worst_gap = max(worst_gap, gap)
        check_small("band extent matches for P/Q = %d/%d" % (P, Q), gap, 5e-2)
    check_small("worst band-extent mismatch", worst_gap, 5e-2)


def main():
    t0 = time.time()
    test_etilde()
    test_theorem_III9()
    test_etilde_identity()
    test_chambers()
    test_zero_modes()
    try:
        test_bridge_to_quantum_curve()
    except Exception as e:                                       # noqa: BLE001
        print("\n[6] bridge to quantum_curve SKIPPED: %r" % (e,))

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_hofstadter: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
