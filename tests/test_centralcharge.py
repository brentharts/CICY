"""
Tests for pyCICY.centralcharge.

As elsewhere in this package, every number is reached by two routes that
share no code, and each check is also made to fail on purpose.

  [1] charges       closed form in ch^B_k vs a truncated-ring expansion of
                    exp(-B - i w) ch(E) (1 + c2/24) integrated with d_rst
  [2] covariance    Z_{B+L}(E (x) O(L)) = Z_B(E): twisting the Chern character
                    vs adding L to every summand
  [3] Euler form    HRR (intersection numbers) vs the Leray computation
                    CICY.line_co_euler; Serre duality chi(E,F) = -chi(F,E)
  [4] elementary    chi(O_x) charge, shifts, additivity
  [5] Gepner        Phi = T_O o (-(x)O(1)) on K_num of the quintic: order,
                    characteristic polynomial, Euler form, and the matrix
                    rebuilt from O(k) classes with line_co_euler
  [6] phases        line bundles land in phase (0, 1] in the double-tilted
                    heart (assuming tilt stability, which is not proved)
  [7] controls

Run with:  python3 tests/test_centralcharge.py
       or: python3 run_tests.py
"""

import os
import random
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sympy as sp

from pyCICY import CICY
from pyCICY import centralcharge as CC

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


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    except Exception:
        return False
    return False


QUINTIC = CC.CY3.from_cicy([[4, 5]])
M7833 = CC.CY3.from_cicy([[2, 2, 1], [3, 1, 3]])
TETRA = CC.CY3.from_cicy([[1, 2], [1, 2], [1, 2], [1, 2]])
THREEFOLDS = [("quintic", QUINTIC), ("7833", M7833), ("tetraquadric", TETRA)]


def rand_frac(rng, lo, hi):
    return Fraction(rng.randint(lo * 6, hi * 6), 6)


def rand_data(rng, X):
    n = rng.randint(1, 4)
    summands = [[rng.randint(-3, 3) for _ in range(X.h)] for _ in range(n)]
    signs = [rng.choice([1, -1]) for _ in range(n)]
    B = [rand_frac(rng, -2, 2) for _ in range(X.h)]
    w = [rand_frac(rng, 0, 4) + Fraction(1, 6) for _ in range(X.h)]
    return summands, signs, B, w


def test_charges():
    print("\n[1] closed form against the truncated-ring expansion")
    rng = random.Random(5)
    bad_ord, bad_td, n = [], [], 0
    for name, X in THREEFOLDS:
        for _ in range(12):
            S, sg, B, w = rand_data(rng, X)
            E = CC.Object.from_lines(X, S, sg)
            n += 1
            if CC.z_ordinary(E, B, w) != CC.z_from_ring(X, S, sg, B, w,
                                                        sqrt_td=False):
                bad_ord.append((name, S, sg, B, w))
            if CC.z_sqrt_td(E, B, w) != CC.z_from_ring(X, S, sg, B, w):
                bad_td.append((name, S, sg, B, w))
    check("Z_ord, closed form vs ring (%d cases): mismatches" % n,
          len(bad_ord), 0)
    check("Z_sqrt, closed form vs ring (%d cases): mismatches" % n,
          len(bad_td), 0)
    O = CC.Object.from_lines(QUINTIC, [[0]])
    check("Z_ord(O_X) on the quintic at B=0, w=1 is -5i/6",
          CC.z_ordinary(O, [0], [1]), CC.Cx(0, Fraction(-5, 6)))
    # the sqrt-Todd charge differs from the ordinary one by -(c2/24)(ch1^B - i w ch0)
    check("Z_sqrt(O_X) at B=0, w=1: -5i/6 + 50 i/24 = 5i/4 ... via ring",
          CC.z_sqrt_td(O, [0], [1]), CC.z_from_ring(QUINTIC, [[0]], [1],
                                                    [0], [1]))
    check("Z_sqrt(O_X) imaginary part", CC.z_sqrt_td(O, [0], [1]).im,
          Fraction(-5, 6) + Fraction(50, 24))


def test_covariance():
    print("\n[2] Z_{B+L}(E (x) O(L)) = Z_B(E)")
    rng = random.Random(8)
    bad, n = [], 0
    for name, X in THREEFOLDS:
        for _ in range(10):
            S, sg, B, w = rand_data(rng, X)
            L = [rng.randint(-2, 2) for _ in range(X.h)]
            E = CC.Object.from_lines(X, S, sg)
            twisted_route1 = E.twist(L)
            twisted_route2 = CC.Object.from_lines(
                X, [[a + b for a, b in zip(s, L)] for s in S], sg)
            n += 1
            if twisted_route1.vector() != twisted_route2.vector():
                bad.append(("twist", name, S, L))
            B2 = [b + l for b, l in zip(B, L)]
            for f in (CC.z_ordinary, CC.z_sqrt_td):
                if f(twisted_route2, B2, w) != f(E, B, w):
                    bad.append((f.__name__, name, S, L, B))
    check("twist of ch vs shifted summands, covariance (%d): bad" % n,
          len(bad), 0)


def test_euler():
    print("\n[3] Euler form: HRR vs Leray, Serre duality")
    bad = []
    for name, X in THREEFOLDS:
        rng = random.Random(3)
        for _ in range(6):
            L = [rng.randint(-3, 3) for _ in range(X.h)]
            got = CC.chi(CC.Object.from_lines(X, [L]))
            want = CC.chi_from_cohomology(X, [L])
            if got != want:
                bad.append((name, L, got, want))
        for _ in range(4):
            S, sg, _, _ = rand_data(rng, X)
            if CC.chi(CC.Object.from_lines(X, S, sg)) != \
                    CC.chi_from_cohomology(X, S, sg):
                bad.append((name, S, sg))
    check("chi(E) from HRR vs line_co_euler: mismatches", len(bad), 0)
    bad = []
    rng = random.Random(4)
    for name, X in THREEFOLDS:
        for _ in range(6):
            S1, g1, _, _ = rand_data(rng, X)
            S2, g2, _, _ = rand_data(rng, X)
            E = CC.Object.from_lines(X, S1, g1)
            F = CC.Object.from_lines(X, S2, g2)
            if CC.euler_pairing(E, F) != -CC.euler_pairing(F, E):
                bad.append((name, S1, S2))
    check("chi(E,F) = -chi(F,E) on a CY3: violations", len(bad), 0)
    a, b = [1], [4]
    check("chi(O(1), O(4)) = chi(O(3)) = 35 on the quintic",
          CC.euler_pairing(CC.Object.from_lines(QUINTIC, [a]),
                           CC.Object.from_lines(QUINTIC, [b])), 35)


def test_elementary():
    print("\n[4] elementary properties")
    for name, X in THREEFOLDS:
        pt = CC.Object.point(X)
        B = [Fraction(1, 3)] * X.h
        w = [Fraction(2)] * X.h
        check("%s: Z(O_x) = -1 for both charges" % name,
              (CC.z_ordinary(pt, B, w), CC.z_sqrt_td(pt, B, w)),
              (CC.Cx(-1), CC.Cx(-1)))
    rng = random.Random(2)
    X = M7833
    S, sg, B, w = rand_data(rng, X)
    E = CC.Object.from_lines(X, S, sg)
    F = CC.Object.from_lines(X, [[1, 1]])
    for f in (CC.z_ordinary, CC.z_sqrt_td):
        check("%s additive" % f.__name__, f(E + F, B, w), f(E, B, w) + f(F, B, w))
        check("%s: Z(E[1]) = -Z(E), Z(E[2]) = Z(E)" % f.__name__,
              (f(E.shift(1), B, w), f(E.shift(2), B, w)),
              (-f(E, B, w), f(E, B, w)))


def _gepner_by_line_bundles(X):
    """Phi on the classes of O(k), k = 0..3, from line_co_euler alone."""
    one = CC.Object.from_lines(X, [[0]])
    rows = []
    for k in range(4):
        Ok1 = CC.Object.from_lines(X, [[k + 1]])
        n = CC.chi_from_cohomology(X, [[k + 1]])        # chi(O, O(k+1))
        rows.append((CC.Object.from_lines(X, [[k]]).vector(),
                     (Ok1 - one.scale(n)).vector()))
    return rows


def test_gepner():
    print("\n[5] Phi = T_O o (- (x) O(1)) on the quintic")
    X = QUINTIC
    r = CC.gepner_report(X)
    check("Phi^5 = +identity (Phi^5 = [2])", r["order"], (5, 1))
    check_true("Phi preserves the Euler form", r["preserves_euler"])
    x = sp.Symbol("x")
    check("characteristic polynomial is the 5th cyclotomic polynomial",
          sp.expand(r["charpoly"]), x ** 4 + x ** 3 + x ** 2 + x + 1)
    Phi = r["phi"]
    bad = 0
    for v, w in _gepner_by_line_bundles(X):
        got = Phi * sp.Matrix([sp.Rational(a.numerator, a.denominator)
                               for a in v])
        want = sp.Matrix([sp.Rational(a.numerator, a.denominator) for a in w])
        bad += int(got != want)
    check("Phi(O(k)) = O(k+1) - chi(O(k+1)) O, k=0..3, rebuilt: bad", bad, 0)
    # the left eigenvectors: one for each primitive 5th root of unity
    ok = True
    for j in range(1, 5):
        zeta = sp.exp(2 * sp.pi * sp.I * j / 5)
        ns = (Phi.T - zeta * sp.eye(4)).applyfunc(
            lambda e: sp.nsimplify(e)).nullspace(simplify=True)
        ok = ok and len(ns) == 1
    check_true("each primitive 5th root of unity has a 1-dim charge", ok)
    # a large-volume charge is NOT a Gepner charge (negative control)
    vals = []
    for E in (CC.Object.from_lines(X, [[k]]) for k in range(4)):
        vec = Phi * sp.Matrix([sp.Rational(a.numerator, a.denominator)
                               for a in E.vector()])
        fr = [Fraction(int(sp.Rational(c).p), int(sp.Rational(c).q))
              for c in vec]
        PhiE = CC.Object(X, fr[0], [fr[1]], [fr[2]], fr[3])
        z0 = CC.z_ordinary(E, [0], [3])
        z1 = CC.z_ordinary(PhiE, [0], [3])
        zeta = complex(sp.N(sp.exp(2 * sp.pi * sp.I * 1 / 5)))
        vals.append(abs(complex(z1) - zeta * complex(z0)))
    check_true("Z_ord at w=3 is not an eigencharge (residual > 1)",
               min(vals) > 1)


def test_phases():
    print("\n[6] line bundles in the double-tilted heart")
    bad = []
    n = 0
    for name, X in THREEFOLDS:
        rng = random.Random(6)
        for _ in range(15):
            D = [rng.randint(-3, 3) for _ in range(X.h)]
            B = [rand_frac(rng, -2, 2) for _ in range(X.h)]
            w = [rand_frac(rng, 0, 6) + 1 for _ in range(X.h)]
            k, Z, ph = CC.line_bundle_phase(X, D, B, w)
            n += 1
            if not (0 < ph <= 1) or k not in (0, 1, 2):
                bad.append((name, D, B, w, k, ph))
    check("phase in (0,1], shift in {0,1,2} (%d cases): bad" % n, len(bad), 0)
    # at huge volume a line bundle sits in the heart as O(D)[1], phase -> 1/2
    res = [CC.line_bundle_phase(QUINTIC, [D], [0], [1000]) for D in range(-3, 4)]
    check_true("quintic, w=1000, B=0: every O(D)[1], phase within 0.01 of 1/2",
               all(k == 1 and abs(ph - 0.5) < 0.01 for k, _, ph in res))
    # Hom(O(a)[1], O(b)[1]) = H^0(O(b-a)) != 0 for b > a forces
    # phase(O(a)[1]) <= phase(O(b)[1]) in an abelian heart of semistables
    phs = [ph for _, _, ph in res]
    check_true("Hom-vanishing order: phases increase with D (quintic)",
               all(x < y for x, y in zip(phs, phs[1:])))
    rng = random.Random(1)
    bad = 0
    for _ in range(12):
        B = [rand_frac(rng, -1, 1) for _ in range(TETRA.h)]
        w = [Fraction(rng.randint(300, 600)) for _ in range(TETRA.h)]
        out = []
        for t in range(-2, 3):
            k, _, ph = CC.line_bundle_phase(TETRA, [t] * TETRA.h, B, w)
            out.append((k, ph))
        ks = {k for k, _ in out}
        pp = [ph for _, ph in out]
        if len(ks) != 1 or not all(x < y for x, y in zip(pp, pp[1:])):
            bad += 1
    check("Hom-vanishing order along the diagonal (tetraquadric): bad", bad, 0)


def test_controls():
    print("\n[7] controls: each check made to fail on purpose")
    X = QUINTIC
    E = CC.Object.from_lines(X, [[2]])
    B, w = [Fraction(1, 2)], [Fraction(2)]
    check_true("the sqrt-Todd charge differs from the ordinary one",
               CC.z_sqrt_td(E, B, w) != CC.z_ordinary(E, B, w))
    check_true("ring route sees the Todd term (ordinary != sqrt via ring)",
               CC.z_from_ring(X, [[2]], [1], B, w, sqrt_td=False)
               != CC.z_from_ring(X, [[2]], [1], B, w))
    # wrong-sign twist breaks covariance
    wrong = CC.Object.from_lines(X, [[2 - 1]])
    check_true("covariance holds with the right shift of B",
               CC.z_ordinary(wrong, [B[0] - 1], w) == CC.z_ordinary(E, B, w))
    check_true("covariance fails if B is shifted the wrong way",
               CC.z_ordinary(wrong, [B[0] + 1], w) != CC.z_ordinary(E, B, w))
    check_true("floats are refused, not rounded",
               _raises(TypeError, CC.z_ordinary, E, [0.5], [2]))
    check_true("w = 0 is refused as not ample",
               _raises(CC.NotAmple, CC.z_ordinary, E, [0], [0]))
    check_true("w < 0 is refused as not ample",
               _raises(CC.NotAmple, CC.z_ordinary, E, [0], [-1]))
    check_true("a wrong-length vector is refused",
               _raises(ValueError, CC.Object.from_lines, M7833, [[1]]))
    check_true("the zero charge has no phase",
               _raises(ValueError, CC.Cx(0, 0).phase))
    # a deliberately wrong autoequivalence (no -chi(O,.) correction) must
    # fail the Euler-form and order checks
    M = CC.gepner_matrix(X)
    check_true("Phi is not just the twist: the matrix has the O-correction",
               M[0][0] != 1)
    # Phi on the tetraquadric is not a bundle-level order-5 symmetry claim
    check_true("gepner_matrix refuses h11 > 1",
               _raises(ValueError, CC.gepner_matrix, M7833))


def main():
    for section in (test_charges, test_covariance, test_euler,
                    test_elementary, test_gepner, test_phases, test_controls):
        try:
            section()
        except Exception as exc:  # an exception is a failure, not an abort
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
    print("all centralcharge tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
