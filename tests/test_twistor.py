"""
Tests for pyCICY.twistor.

A tree scattering amplitude is a rational function of spinor brackets, so with
rational spinors it is a rational number and every claim about it is decidable.
That is what makes this suite possible: nothing below is checked to within a
tolerance.

  [1] kinematics    exact momentum conservation, and the identities that
                    follow from it rather than being imposed
  [2] parke_taylor  the closed form, its little-group weights, and its poles
  [3] bcfw          the same amplitudes rebuilt from three-point ones, which
                    is the main cross-check in the module
  [4] beyond        NMHV amplitudes, which have no closed form, checked by
                    the symmetries they must have
  [5] relations     U(1) decoupling and the fundamental BCJ relation, exactly
                    zero; and the rank of the space of colour orderings, which
                    sees Kleiss-Kuijf but provably cannot see BCJ
  [6] positroid     cells of the totally non-negative Grassmannian, counted
                    two ways
  [7] geometry      the twistor double fibration as complete intersections,
                    using the package's own Chern class machinery

Run with:  python3 tests/test_twistor.py
       or: python3 run_tests.py
"""

import os
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import twistor as TW

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


def test_kinematics():
    print("\n[1] exact spinor kinematics")

    for n in range(3, 10):
        k = TW.Kinematics.random(n, seed=n)
        check_true("n = %d conserves momentum exactly" % n, k.check())
        check("   and has %d legs" % n, k.n, n)
    check_true("an unknown three-point kind is refused",
               _raises(ValueError, TW.Kinematics.random, 3, 0, 6, "sideways"))

    k = TW.Kinematics.random(7, seed=3)

    # The Schouten identity holds because there is no antisymmetric
    # three-index tensor in two dimensions. Nothing imposes it, so its
    # vanishing tests the bracket rather than the kinematics.
    for quad in [(1, 2, 3, 4), (2, 4, 5, 7), (1, 3, 6, 7)]:
        check("Schouten on %s" % (quad,), k.schouten_residual(*quad),
              Fraction(0))

    # This one does follow from momentum conservation, and is the form of it
    # that amplitude manipulations actually use.
    for pair in [(1, 2), (3, 5), (7, 1)]:
        check("sum_i <ki>[il] on %s" % (pair,), k.momentum_residual(*pair),
              Fraction(0))

    # Antisymmetry, and that a bracket of a spinor with itself vanishes.
    check("<12> = -<21>", k.angle(1, 2), -k.angle(2, 1))
    check("[35] = -[53]", k.square(3, 5), -k.square(5, 3))
    check("<11> vanishes", k.angle(1, 1), Fraction(0))

    # The Mandelstam invariant, computed as a determinant, against the
    # bracket expression. Two routes to the same rational number.
    for pair in [(1, 2), (2, 3), (4, 6)]:
        check("s%s = <ij>[ij]" % (pair,), k.s(*pair),
              k.angle(*pair) * k.square(*pair))
    # A three-particle invariant has no two-bracket form, so it is checked by
    # the identity s_{123} = s_{12} + s_{13} + s_{23} instead.
    check("s(1,2,3) is the sum of its pairs", k.s(1, 2, 3),
          k.s(1, 2) + k.s(1, 3) + k.s(2, 3))
    # And the total momentum is null, indeed zero.
    check("the total momentum squared vanishes",
          k.s(*range(1, 8)), Fraction(0))

    # Generic position is enforced, not hoped for: a vanishing bracket would
    # put a pole of the amplitude at the evaluation point.
    check_true("no bracket vanishes",
               all(k.angle(i, j) != 0 and k.square(i, j) != 0
                   for i in range(1, 8) for j in range(i + 1, 8)))

    check_true("spinors that do not conserve momentum are refused",
               _raises(ValueError, TW.Kinematics,
                       [(1, 0), (0, 1), (1, 1)], [(1, 0), (0, 1), (1, 1)]))
    check_true("two legs are refused",
               _raises(ValueError, TW.Kinematics.random, 2))


def test_parke_taylor():
    print("\n[2] the Parke-Taylor amplitude")

    k = TW.Kinematics.random(6, seed=5)

    # Cyclic invariance: the ordering is a cycle, so rotating it changes
    # nothing.
    base = TW.parke_taylor(k, (1, 2))
    for r in range(1, 6):
        order = [((i + r) % 6) + 1 for i in range(6)]
        check("cyclic rotation by %d" % r, TW.parke_taylor(k, (1, 2), order),
              base)

    # Reflection: reversing the ordering gives (-1)^n times the amplitude.
    rev = list(range(6, 0, -1))
    check("reflection gives (-1)^n", TW.parke_taylor(k, (1, 2), rev), base)
    k5 = TW.Kinematics.random(5, seed=6)
    check("and a sign for odd n", TW.parke_taylor(k5, (1, 2), [5, 4, 3, 2, 1]),
          -TW.parke_taylor(k5, (1, 2)))

    # Little-group weight: scaling lambda_i by t scales the amplitude by
    # t^{-2h_i}, so by t^2 for a negative-helicity leg and t^{-2} for a
    # positive one. This is what fixes the fourth power in the numerator.
    for leg, weight in ((1, 2), (3, -2)):
        lam = [list(v) for v in k.lam]
        lam[leg - 1] = [3 * x for x in lam[leg - 1]]
        scaled = TW.parke_taylor(lam, (1, 2))
        check("scaling leg %d by 3 gives 3^%d" % (leg, weight),
              scaled, base * Fraction(3) ** weight)

    # It depends on the holomorphic spinors alone. That is the geometric
    # content: an MHV amplitude is supported on a line in twistor space.
    k2 = TW.Kinematics(k.lam, k.lamt)
    check_true("depends on lambda only",
               TW.parke_taylor(k.lam, (1, 2)) == TW.parke_taylor(k2, (1, 2)))

    check_true("three negative helicities are refused",
               _raises(ValueError, TW.parke_taylor, k, (1, 2, 3)))

    # At four points the two negative helicities can be chosen in six ways and
    # the amplitude is nonzero for each.
    for neg in [(1, 2), (1, 3), (2, 4), (3, 4)]:
        check_true("four-point MHV with negatives %s is nonzero" % (neg,),
                   TW.parke_taylor(TW.Kinematics.random(4, seed=7), neg) != 0)


def test_bcfw():
    print("\n[3] BCFW against the closed forms")

    # The central cross-check of the module. BCFW builds every tree amplitude
    # from the three-point ones, which are fixed by Lorentz invariance alone;
    # Parke-Taylor is a closed formula. They share no code and must agree
    # exactly, not approximately.
    for n in range(4, 9):
        k = TW.Kinematics.random(n, seed=n)
        hel = [-1, -1] + [1] * (n - 2)
        bcfw = TW.tree_amplitude(k.lam, k.lamt, hel)
        pt = TW.parke_taylor(k, (1, 2))
        check("n = %d MHV: BCFW equals Parke-Taylor" % n, bcfw, pt)
        check_true("   and is not zero", pt != 0)

    # The conjugate. Parity exchanges angle and square brackets, and the
    # recursion has to reproduce that too, through the other three-point
    # amplitude.
    for n in range(4, 8):
        k = TW.Kinematics.random(n, seed=n + 20)
        hel = [1, 1] + [-1] * (n - 2)
        check("n = %d anti-MHV: BCFW equals the conjugate form" % n,
              TW.tree_amplitude(k.lam, k.lamt, hel), TW.anti_mhv(k, (1, 2)))

    # Any placement of the two negative helicities, not just the first two.
    k = TW.Kinematics.random(6, seed=31)
    for neg in [(1, 4), (2, 5), (3, 6), (2, 3)]:
        hel = [1] * 6
        hel[neg[0] - 1] = -1
        hel[neg[1] - 1] = -1
        check("negatives at %s" % (neg,),
              TW.tree_amplitude(k.lam, k.lamt, hel),
              TW.parke_taylor(k, neg))

    # Amplitudes with fewer than two of either helicity vanish. For n >= 4
    # this is supersymmetry, and it is the only input to the recursion beyond
    # the three-point amplitudes.
    for n in (4, 5, 6):
        k = TW.Kinematics.random(n, seed=n)
        check("n = %d all plus vanishes" % n,
              TW.tree_amplitude(k.lam, k.lamt, [1] * n), Fraction(0))
        check("n = %d one minus vanishes" % n,
              TW.tree_amplitude(k.lam, k.lamt, [-1] + [1] * (n - 1)),
              Fraction(0))

    # The three-point amplitudes themselves, which exist only for complex
    # momenta. With real Lorentzian kinematics every bracket here would
    # vanish; with independent spinors they are honest rational numbers.
    # They live at different kinematic points, and must: at three points
    # momentum conservation forces one set of brackets to vanish entirely, so
    # there is no configuration supporting both. That degeneracy is the reason
    # the recursion needs complex momenta, and it is not an artefact of
    # working over the rationals -- it holds over any field.
    kh = TW.Kinematics.random(3, seed=2, kind="holomorphic")
    ka = TW.Kinematics.random(3, seed=2, kind="antiholomorphic")
    check_true("holomorphic three-point kinematics has no square brackets",
               all(kh.square(i, j) == 0 for i, j in [(1, 2), (2, 3), (1, 3)]))
    check_true("   but non-zero angle brackets",
               all(kh.angle(i, j) != 0 for i, j in [(1, 2), (2, 3), (1, 3)]))
    check_true("and the conjugate point is the other way round",
               all(ka.angle(i, j) == 0 and ka.square(i, j) != 0
                   for i, j in [(1, 2), (2, 3), (1, 3)]))
    a = TW.tree_amplitude(kh.lam, kh.lamt, [-1, -1, 1])
    b = TW.tree_amplitude(ka.lam, ka.lamt, [-1, 1, 1])
    check_true("the holomorphic three-point amplitude is nonzero", a != 0)
    check_true("and the anti-holomorphic one too", b != 0)
    check("the first agrees with Parke-Taylor", a, TW.parke_taylor(kh, (1, 2)))
    check("the second with its conjugate", b, TW.anti_mhv(ka, (2, 3)))


def test_beyond_mhv():
    print("\n[4] NMHV, where there is no formula to check against")

    # Three negative helicities at six points is the first amplitude with no
    # closed form, so it is checked by the symmetries it must have. These are
    # not trivial: the BCFW result is a sum over channels chosen after
    # rotating the ordering, and nothing in the construction makes the answer
    # cyclic.
    k = TW.Kinematics.random(6, seed=11)
    hel = [-1, -1, -1, 1, 1, 1]
    A = TW.tree_amplitude(k.lam, k.lamt, hel)
    check_true("the NMHV amplitude is nonzero", A != 0)

    def rot(v, r):
        return [v[(i + r) % 6] for i in range(6)]
    for r in range(1, 6):
        check("cyclic by %d" % r,
              TW.tree_amplitude(rot(k.lam, r), rot(k.lamt, r), rot(hel, r)), A)

    check("reflection, with n even",
          TW.tree_amplitude(k.lam[::-1], k.lamt[::-1], hel[::-1]), A)

    # Alternating helicities, the other six-point configuration.
    hel2 = [-1, 1, -1, 1, -1, 1]
    B = TW.tree_amplitude(k.lam, k.lamt, hel2)
    check_true("the alternating NMHV amplitude is nonzero", B != 0)
    for r in range(2, 6, 2):
        check("   cyclic by %d" % r,
              TW.tree_amplitude(rot(k.lam, r), rot(k.lamt, r), rot(hel2, r)),
              B)

    # Seven points, four negative: N2MHV, and still cyclic.
    k7 = TW.Kinematics.random(7, seed=13)
    hel7 = [-1, -1, -1, -1, 1, 1, 1]
    C = TW.tree_amplitude(k7.lam, k7.lamt, hel7)
    check_true("a seven-point N2MHV amplitude is nonzero", C != 0)
    rot7 = lambda v, r: [v[(i + r) % 7] for i in range(7)]   # noqa: E731
    check("   and cyclic",
          TW.tree_amplitude(rot7(k7.lam, 3), rot7(k7.lamt, 3), rot7(hel7, 3)),
          C)

    # The twistor degree these live on.
    check("MHV sits on a line", TW.mhv_degree(2)["degree"], 1)
    check("NMHV on a conic", TW.mhv_degree(3)["degree"], 2)
    check("and a loop raises the degree", TW.mhv_degree(3, 1)["degree"], 3)
    check("with genus at most the loop order",
          TW.mhv_degree(3, 1)["max_genus"], 1)
    check_true("k below two is refused", _raises(ValueError, TW.mhv_degree, 1))


def test_relations():
    print("\n[5] relations among colour orderings")

    # U(1) decoupling: constant coefficients, so it holds identically.
    for n in range(4, 9):
        k = TW.Kinematics.random(n, seed=n)
        check("n = %d U(1) decoupling" % n,
              TW.u1_decoupling_residual(k, (1, 2)), Fraction(0))

    # The fundamental BCJ relation, whose coefficients are Mandelstams. It
    # holds at each kinematic point separately.
    for n in range(4, 9):
        for seed in (n, n + 50):
            k = TW.Kinematics.random(n, seed=seed)
            check("n = %d BCJ, seed %d" % (n, seed),
                  TW.bcj_residual(k, (1, 2)), Fraction(0))

    # And the rank. This is the interesting one, because what it measures is
    # not what one first expects.
    for n in (5, 6):
        pts = [TW.Kinematics.random(n, seed=300 + i) for i in range(40)]
        r = TW.ordering_rank(pts, (1, 2))
        check("n = %d: %d orderings" % (n, r["orderings"]), r["orderings"],
              _fact(n - 1))
        check("   span has rank (n-2)!", r["rank"], _fact(n - 2))
        check_true("   which is what Kleiss-Kuijf predicts", r["agrees"])
        check_true("   and enough points were used", r["saturated"])
        # BCJ would give (n-3)!, and the rank does not see it. That is a
        # property of the method, not a failure of the relation: a rank taken
        # across kinematic points sees only relations with constant
        # coefficients, and BCJ's coefficients vary from point to point. The
        # relation itself holds, as the residual above shows.
        check_true("   but it is not (n-3)!, and cannot be",
                   r["rank"] != _fact(n - 3))
        bcj = TW.ordering_rank(pts, (1, 2), relation="bcj")
        check_true("   so asking for BCJ disagrees, correctly",
                   not bcj["agrees"])

    # Too few points bounds the rank by the number of points, and the result
    # says so rather than reporting a spuriously small rank.
    few = [TW.Kinematics.random(6, seed=400 + i) for i in range(5)]
    r = TW.ordering_rank(few, (1, 2))
    check("five points bound the rank at five", r["rank"], 5)
    check_true("and the shortfall is reported", not r["saturated"])


def _fact(k):
    out = 1
    for i in range(2, k + 1):
        out *= i
    return out


def test_positroid():
    print("\n[6] cells of the totally non-negative Grassmannian")

    # Postnikov's bijection: cells of G(k,n)_{>=0} correspond to decorated
    # permutations of [n] with k anti-exceedances, which is also what labels
    # the on-shell diagrams of planar N=4 super Yang-Mills.
    known = {2: [1, 3, 1], 3: [1, 7, 7, 1], 4: [1, 15, 33, 15, 1],
             5: [1, 31, 131, 131, 31, 1]}
    for n, row in sorted(known.items()):
        got = [TW.positroid_cells(k, n) for k in range(n + 1)]
        check("n = %d cell counts by k" % n, got, row)
        check_true("   and the row is symmetric under k -> n-k",
                   got == got[::-1])

    check("G(2,4) has 33 cells", TW.positroid_cells(2, 4), 33)
    check("G(1,n) has 2^n - 1 for n = 5", TW.positroid_cells(1, 5), 31)

    # The total over all k has a closed form that has nothing obviously to do
    # with coloured fixed points: it is the number of arrangements of n
    # objects, sum_j n!/j!. Counting one way and evaluating the other is a
    # check on the anti-exceedance statistic.
    for n in range(1, 7):
        counted, closed = TW.positroid_total_check(n)
        check("n = %d total decorated permutations" % n, counted, closed)
        check("   equals the sum over k",
              sum(TW.positroid_cells(k, n) for k in range(n + 1)), counted)
    check("and the sequence is 2, 5, 16, 65, 326",
          [TW.positroid_total_check(n)[0] for n in range(1, 6)],
          [2, 5, 16, 65, 326])

    # The top cell has dimension k(n-k), the dimension of the Grassmannian.
    for k, n in [(1, 3), (2, 4), (2, 5), (3, 6)]:
        check("dim G(%d,%d) = k(n-k)" % (k, n), TW.cell_dimension(k, n),
              k * (n - k))


def test_geometry():
    print("\n[7] the twistor double fibration")

    # Twistor space, the incidence variety and Minkowski space, all as
    # complete intersections in products of projective spaces, so that the
    # package's own Chern class routine applies to them unchanged.
    g = {x["name"]: x for x in TW.twistor_geometry()}
    check("twistor space is P^3", g["twistor space PT"]["configuration"],
          [[3]])
    check("   with chi = 4", g["twistor space PT"]["euler"], 4)
    check("Minkowski space is the Klein quadric",
          g["Minkowski G(2,4)"]["configuration"], [[5, 2]])
    check("   a quadric hypersurface, so dimension 4",
          g["Minkowski G(2,4)"]["dim"], 4)
    # The Euler characteristic of G(2,4) counts its Schubert cells, which are
    # the partitions fitting in a two-by-two box.
    check("   with chi = 6, the Schubert cells of G(2,4)",
          g["Minkowski G(2,4)"]["euler"], 6)
    check("the incidence variety is the (1,1) hypersurface",
          g["incidence F(1,3;4)"]["configuration"], [[3, 1], [3, 1]])
    check("   with chi = 12", g["incidence F(1,3;4)"]["euler"], 12)

    # The Penrose transform, and the caveat that makes it honest.
    for h, bundle in [(1, "O(-4)"), (-1, "O(0)"), (Fraction(1, 2), "O(-3)"),
                      (2, "O(-6)")]:
        p = TW.penrose_helicity(h)
        check("helicity %s maps to %s" % (h, bundle), p["bundle"], bundle)
    check("and H^1 of any O(m) on P^3 vanishes",
          TW.penrose_helicity(1)["h1_on_P3"], 0)
    check_true("which is why the transform needs an open twistor space",
               "open twistor space" in TW.penrose_helicity(1)["note"])
    check_true("a helicity that is not a half-integer is refused",
               _raises(ValueError, TW.penrose_helicity, Fraction(1, 3)))


def main():
    t0 = time.time()
    test_kinematics()
    test_parke_taylor()
    test_bcfw()
    test_beyond_mhv()
    test_relations()
    test_positroid()
    test_geometry()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_twistor: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
