"""
Tests for pyCICY.monotile.

The module's claim is that everything from the tile geometry to the
topological index lives in a quadratic field with a decidable order, so every
check below is an exact equality -- including the topological invariant,
which is a matrix signature and is compared against an independent
floating-point eigenvalue count.

  [1] field         exact arithmetic and, crucially, exact order in Q(sqrt d)
  [2] tiles         the named members of the family and the mirror map
  [3] inflation     the Perron eigenvalue phi^4, verified as a root of the
                    exactly computed characteristic polynomial
  [4] frequencies   the metatile frequencies over Q(sqrt5), the derived
                    hat : anti-hat ratio phi^4 : 1, and aperiodicity from
                    irrationality
  [5] patch         the Laves substrate, and the vertex merging that singles
                    out the Hat's proportions
  [6] signature     the LDL^T signature machinery on hand-checkable matrices
  [7] localizer     the exact spectral localizer index: trivial limits,
                    topological phases, and agreement with numpy

Run with:  python3 tests/test_monotile.py
       or: python3 run_tests.py
"""

import os
import sys
import time
from fractions import Fraction as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import monotile as MT
from pyCICY.monotile import Quad, SQRT3, SQRT5, PHI, PHI4

FAILURES = []


def check(name, got, want):
    ok = got == want
    print("  {:<58} {:>14} {}".format(name, str(got)[:14],
                                      "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<58} {:>14} {}".format(name, str(bool(cond)),
                                      "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


def _raises(exc, fn, *a, **kw):
    try:
        fn(*a, **kw)
    except exc:
        return True
    except Exception:                                            # noqa: BLE001
        return False
    return False


# ---------------------------------------------------------------------------


def test_field():
    print("\n[1] the quadratic fields, and their order")

    # Arithmetic: (1 + sqrt3)(1 - sqrt3) = -2, and division undoes
    # multiplication exactly.
    x = Quad(1, 1, 3)
    y = Quad(1, -1, 3)
    check("(1+sqrt3)(1-sqrt3) = -2", x * y, Quad(-2, 0, 3))
    check("division is exact", (x * y) / y, x)
    check("sqrt3 squared", SQRT3 * SQRT3, Quad(3, 0, 3))
    check("the golden ratio satisfies phi^2 = phi + 1",
          PHI * PHI, PHI + Quad(1, 0, 5))
    check("and phi^4 = (7 + 3 sqrt5)/2", PHI4,
          Quad(F(7, 2), F(3, 2), 5))

    # The order is the point. sqrt3 > 1 and 2 - sqrt3 > 0 but 3 - 2 sqrt3 < 0
    # are decided by comparing p^2 with d q^2, never by a float.
    check("sqrt3 > 1", SQRT3 > Quad(1, 0, 3), True)
    check("2 - sqrt3 is positive", Quad(2, -1, 3).sign(), 1)
    check("3 - 2 sqrt3 is negative", Quad(3, -2, 3).sign(), -1)
    check("26 - 15 sqrt3 is positive, a close call",
          Quad(26, -15, 3).sign(), 1)          # 26^2 = 676 vs 3*225 = 675
    check("(26 - 15 sqrt3) - 1/26 is negative, closer still",
          (Quad(26, -15, 3) - Quad(F(1, 26), 0, 3)).sign(), -1)
    check("zero has sign zero", Quad(0, 0, 3).sign(), 0)

    # Galois conjugation and the mixing guard.
    check("conjugation flips the irrational part",
          Quad(2, 5, 3).conjugate(), Quad(2, -5, 3))
    check_true("mixing sqrt3 with sqrt5 is refused",
               _raises(ValueError, lambda: SQRT3 + SQRT5))
    check_true("zero is not invertible",
               _raises(ZeroDivisionError, Quad(0, 0, 3).inverse))


def test_tiles():
    print("\n[2] the named tiles and the mirror")

    t = MT.named_tiles()
    check("five named members", sorted(t), ["Chevron", "Comet", "Hat",
                                            "Spectre", "Turtle"])
    # 1/(1 + sqrt3) rationalises to (sqrt3 - 1)/2.
    check("the Hat sits at (sqrt3 - 1)/2", t["Hat"],
          Quad(F(-1, 2), F(1, 2), 3))
    check("the Spectre at one half", t["Spectre"], Quad(F(1, 2), 0, 3))
    check("the Chevron and Comet at the ends",
          (t["Chevron"], t["Comet"]), (Quad(0, 0, 3), Quad(1, 0, 3)))

    # Exchanging the two edge lengths reflects the tile, so ell -> 1 - ell is
    # a mirror. The Hat and Turtle are mirror partners; the Spectre is the
    # fixed point, which is the parameter-space shadow of being the strictly
    # chiral member.
    check("Hat + Turtle = 1, a mirror pair", t["Hat"] + t["Turtle"],
          Quad(1, 0, 3))
    check("mirror of the Hat is the Turtle", MT.mirror_ell(t["Hat"]),
          t["Turtle"])
    check("the Spectre is the mirror fixed point",
          MT.mirror_ell(t["Spectre"]), t["Spectre"])
    check("and the ends exchange", MT.mirror_ell(t["Chevron"]), t["Comet"])

    # ell = a/(a+b) from the edge lengths; the Hat's proportions are 1:sqrt3.
    check("tile_ell(1, sqrt3) is the Hat", MT.tile_ell(1, SQRT3), t["Hat"])
    check("tile_ell(1, 1) is the Spectre", MT.tile_ell(1, 1), t["Spectre"])
    check("tile_ell(sqrt3, 1) is the Turtle", MT.tile_ell(SQRT3, 1),
          t["Turtle"])


def test_inflation():
    print("\n[3] the inflation factor")

    r = MT.inflation_factor()
    # The characteristic polynomial is computed over Z by Leverrier-Faddeev,
    # with nothing numerical, and factors as (x^2 - 7x + 1)(x^2 - 1).
    check("characteristic polynomial", r["charpoly"], [1, -7, 0, 7, -1])
    check_true("phi^4 is an exact root of it", r["is_root"])
    check_true("and satisfies x^2 - 7x + 1 = 0", r["is_phi4"])
    check("the Perron eigenvalue is phi^4 = (7 + 3 sqrt5)/2",
          r["value"], PHI4)
    # The conjugate root phi^{-4} pairs with it: their product is the
    # constant term of x^2 - 7x + 1, which is one.
    check("phi^4 times its conjugate root is one",
          PHI4 * PHI4.conjugate(), Quad(1, 0, 5))
    # And the trace of the substitution matrix is the sum of all four roots:
    # phi^4 + phi^-4 + 1 - 1 = 7.
    check("trace accounts for all four roots",
          sum(MT.SUBSTITUTION[i][i] for i in range(4)), 7)


def test_frequencies():
    print("\n[4] frequencies, chirality and aperiodicity")

    f = MT.metatile_frequencies()
    one = Quad(1, 0, 5)
    check("the frequencies sum to one",
          f["H"] + f["T"] + f["P"] + f["F"], one)
    # An eigenvector check that uses the matrix directly: M f = phi^4 f,
    # verified exactly component by component.
    names = MT.METATILES
    for i, row in enumerate(MT.SUBSTITUTION):
        lhs = Quad(0, 0, 5)
        for j, c in enumerate(row):
            lhs = lhs + Quad(c, 0, 5) * f[names[j]]
        check("   row %s of M f = phi^4 f" % names[i], lhs,
              PHI4 * f[names[i]])
    # The H frequency is rational -- a curiosity worth pinning -- while T's
    # is not, and the irrational one is what aperiodicity rests on.
    check("the H frequency is exactly one third", f["H"],
          Quad(F(1, 3), 0, 5))
    check_true("the T frequency is irrational", not f["T"].is_rational())

    # The chirality ratio, derived from the frequencies and the hat counts,
    # must be phi^4 -- the published number, reached without quoting it.
    c = MT.hat_chirality()
    check_true("unreflected : reflected = phi^4 : 1", c["is_phi4"])
    check("as an exact element of Q(sqrt5)", c["ratio"], PHI4)
    # The reflected fraction is 1/(1 + phi^4), rationalised.
    check("reflected fraction is 1/2 - sqrt5/6",
          c["reflected_fraction"], Quad(F(1, 2), F(-1, 6), 5))
    check("which is exactly 1/(1 + phi^4)",
          c["reflected_fraction"] * (one + PHI4), one)

    # Aperiodicity: a periodic tiling has rational frequencies, and these
    # are not. The headline fact about the Hat is, at this level, the
    # irrationality of an eigenvector.
    a = MT.is_aperiodic()
    check_true("the tiling is aperiodic", a["aperiodic"])
    check_true("with an explicit irrational witness",
               a["irrational_part"] != 0)


def test_patch():
    print("\n[5] the Laves substrate")

    p = MT.laves_patch(rings=0)
    # One hexagon's worth of kites at generic proportions: a centre, six
    # edge midpoints, and twelve vertices -- twelve, not six, because the
    # neighbouring kites' outer corners only coincide at one special ratio.
    check("generic proportions: 19 sites", len(p["sites"]), 19)
    check("with 18 bonds", len(p["bonds"]), 18)
    check("one centre", p["kind"].count("centre"), 1)
    check("six edge midpoints", p["kind"].count("edge"), 6)
    check("twelve split vertices", p["kind"].count("vertex"), 12)

    # At a : b = sqrt3 : 1 the split vertices merge in pairs. That ratio is
    # the Hat's, so the patch geometry itself detects the Hat point: the
    # [3.4.6.4] Laves tiling is recovered exactly there and nowhere else.
    ph = MT.laves_patch(a=SQRT3, b=1, rings=0)
    check("at the Hat's proportions: 13 sites", len(ph["sites"]), 13)
    check("   six merged vertices", ph["kind"].count("vertex"), 6)
    check("   and the same 18 bonds", len(ph["bonds"]), 18)
    # Just off the ratio they split again; exact order decides, not a
    # tolerance.
    poff = MT.laves_patch(a=Quad(F(17, 10), 0, 3), b=1, rings=0)
    check("just off it they split again", len(poff["sites"]), 19)

    # Coordinates are exact: the bond directions are 30 degree multiples, so
    # every coordinate is p + q sqrt3 with rational p, q. Spot check one
    # merged vertex at the Hat point: a e(30) + b e(120) with a = sqrt3.
    half = Quad(F(1, 2), 0, 3)
    vx = SQRT3 * (half * SQRT3) + Quad(1, 0, 3) * (-half)
    vy = SQRT3 * half + Quad(1, 0, 3) * (half * SQRT3)
    check_true("a merged vertex sits at (1, sqrt3), exactly",
               (vx, vy) in [tuple(s) for s in ph["sites"]]
               and vx == Quad(1, 0, 3) and vy == SQRT3)

    # A ring of neighbours grows the patch and keeps everything shared:
    # seven centres, and the six inner edge midpoints are shared bonds.
    p1 = MT.laves_patch(rings=1)
    check("one ring: seven centres", p1["kind"].count("centre"), 7)
    check_true("and strictly more sites than seven copies would need",
               len(p1["sites"]) < 7 * 19)


def test_signature():
    print("\n[6] the signature machinery, on matrices checkable by hand")

    Z = lambda: Quad(0, 0, 3)                                    # noqa: E731
    Q = lambda v: Quad(v, 0, 3)                                  # noqa: E731

    sig, zeros = MT._signature([[Q(2)]])
    check("[[2]] has signature +1", (sig, zeros), (1, 0))
    sig, zeros = MT._signature([[Q(-3)]])
    check("[[-3]] has signature -1", (sig, zeros), (-1, 0))
    sig, zeros = MT._signature([[Q(1), Z()], [Z(), Q(-1)]])
    check("diag(1, -1) has signature 0", (sig, zeros), (0, 0))
    sig, zeros = MT._signature([[Z(), Q(1)], [Q(1), Z()]])
    check("the hyperbolic block [[0,1],[1,0]]", (sig, zeros), (0, 0))
    sig, zeros = MT._signature([[Z(), Z()], [Z(), Z()]])
    check("the zero matrix is all zero pivots", (sig, zeros), (0, 2))
    sig, zeros = MT._signature([[Z(), Q(1), Z()],
                                [Q(1), Z(), Z()],
                                [Z(), Z(), Q(2)]])
    check("hyperbolic block plus a positive pivot", (sig, zeros), (1, 0))

    # An irrational close call: diag(26 - 15 sqrt3, -1) has signature 0
    # because the first entry is positive by one part in 676. A float would
    # get this right too, but not *provably*.
    sig, zeros = MT._signature([[Quad(26, -15, 3), Z()], [Z(), Q(-1)]])
    check("a near-zero irrational pivot is signed correctly",
          (sig, zeros), (0, 0))

    # Congruence invariance (Sylvester): conjugating by an invertible
    # matrix preserves the signature. Check on a small explicit case,
    # S -> P^T S P with P = [[1, 2], [0, 1]].
    S = [[Q(1), Q(3)], [Q(3), Q(-2)]]
    P = [[Q(1), Q(2)], [Z(), Q(1)]]
    PT_S_P = [[sum((P[k][i] * S[k][l] * P[l][j] for k in range(2)
                    for l in range(2)), Z()) for j in range(2)]
              for i in range(2)]
    check("Sylvester: congruence preserves the signature",
          MT._signature(S)[0], MT._signature(PT_S_P)[0])


def test_localizer():
    print("\n[7] the exact spectral localizer")

    t0 = time.time()
    p = MT.laves_patch(rings=0)
    Hre, Him = MT.qwz_hamiltonian(p, F(4))

    # The Hamiltonian is exactly Hermitian: real part symmetric, imaginary
    # part antisymmetric, entry by entry.
    n = len(Hre)
    check_true("H_re is symmetric",
               all(Hre[i][j] == Hre[j][i] for i in range(n)
                   for j in range(n)))
    check_true("H_im is antisymmetric",
               all(Him[i][j] == -Him[j][i] for i in range(n)
                   for j in range(n)))

    # Deep in the atomic limit the model is trivial, and the index says so
    # exactly on both sides.
    for M in (F(4), F(-4)):
        r = MT.localizer_index(M, rings=0)
        check("M = %s is trivial" % M, (r["index"], r["zero_pivots"]),
              (0, 0))

    # In between there are genuine topological phases, with the index
    # computed as a quarter of an exact signature -- no eigenvalue, no
    # tolerance, no floating point.
    r = MT.localizer_index(F(1, 2), rings=0)
    check("M = 1/2 carries index +1", (r["index"], r["zero_pivots"]), (1, 0))
    r = MT.localizer_index(F(-1), rings=0)
    check("M = -1 carries index -1", (r["index"], r["zero_pivots"]), (-1, 0))

    # The same signature from an entirely different route: numpy
    # diagonalises the complex localizer in floating point and counts
    # eigenvalue signs. The exact route must give twice that count, since
    # the real doubling duplicates the spectrum. One route is exact field
    # arithmetic, the other is LAPACK; they share nothing.
    import numpy as np
    for M in (F(-1), F(1, 2), F(4)):
        Hre, Him = MT.qwz_hamiltonian(p, M)
        r = MT.localizer_signature(Hre, Him, p, kappa=F(1, 2))
        Hn = np.array([[complex(float(Hre[i][j]), float(Him[i][j]))
                        for j in range(n)] for i in range(n)])
        X = np.zeros(n)
        Y = np.zeros(n)
        for s, (px, py) in enumerate(p["sites"]):
            X[2 * s] = X[2 * s + 1] = float(px)
            Y[2 * s] = Y[2 * s + 1] = float(py)
        k = 0.5
        Lc = np.block([[Hn, k * (np.diag(X) - 1j * np.diag(Y))],
                       [k * (np.diag(X) + 1j * np.diag(Y)), -Hn]])
        ev = np.linalg.eigvalsh(Lc)
        sig_np = int(np.sum(ev > 1e-9) - np.sum(ev < -1e-9))
        check("M = %s: exact signature is twice numpy's" % M,
              r["signature"], 2 * sig_np)

    # A scan brackets the transitions between exact rational masses. The
    # index jumps somewhere in each bracket, so a topological transition is
    # localised without ever computing a band structure -- there is none.
    scan = MT.phase_scan([F(-4), F(-1), F(1, 2), F(4)], rings=0)
    check("the scan is trivial, topological, topological, trivial",
          [i for _, i in scan], [0, -1, 1, 0])

    print("  (localizer section took %.1fs, all exact)" % (time.time() - t0))


def main():
    t0 = time.time()
    test_field()
    test_tiles()
    test_inflation()
    test_frequencies()
    test_patch()
    test_signature()
    test_localizer()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_monotile: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
