"""
Tests for pyCICY.theories.contours.

The third amplitude module, and the only one in this package whose headline
result is a *failure*. That makes it the sharpest test of the three: a wrong
residue routine will usually still produce some polynomial, but it will not
produce a system of constraints that is satisfiable at six mass levels and
impossible at the seventh.

  [1] table         the Dhat leading singularities at five mass levels,
                    derived as residues, against the published table
  [2] mirror        the integrand is symmetric under swapping the two
                    coordinates together with the two curve exponents, so the
                    mirror rows of the table are produced rather than assumed
  [3] contradiction the matching, level by level: two exponents fixed, then
                    the dimension fixed to four, then nothing left to adjust
                    and no solution. Also the near miss -- without the mirror
                    levels the same system looks satisfiable, which is why a
                    failure has to be reproduced rather than asserted
  [4] baby          the truncated integrals fail differently: three different
                    mass levels return the same value, while the tree side
                    does not. Plus independence of the truncation order,
                    since a residue below that order is already exact
  [5] contour       faces of the associahedron, the Catalan number at the
                    top, the piece count of the generalised Pochhammer
                    contour, and the cutoff that keeps it off the branch cuts
  [6] ledger        NotAnalytic for convergence and for large-kinematics
                    evaluation, NeedsIntegration for the alpha' expansion

Run with:  python3 tests/test_contours.py
       or: python3 run_tests.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sympy as sp

from pyCICY import theories as T
from pyCICY.theories import contours as C
from pyCICY.theories.surface import (SurfaceTheory, NotAnalytic,
                                     NeedsIntegration)

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


def _paper(level, column):
    return sp.sympify(C.PAPER_TABLE_1[level][column],
                      locals={"X11": C.X11, "X22": C.X22, "d": C.DIM})


# ---------------------------------------------------------------------------
# [1] the table
# ---------------------------------------------------------------------------

def test_table():
    print("\n[1] Dhat leading singularities against the published table")

    for level in sorted(C.PAPER_TABLE_1):
        got = C.dhat_leading_singularity(*level)
        check_true("level %s" % (level,),
                   sp.simplify(sp.expand(got - _paper(level, "dhat"))) == 0)

    # the residues are polynomial in the exponents, which is what makes the
    # matching a finite algebraic problem rather than an analytic one
    check_true("every residue is a polynomial in the curve exponents",
               all(C.dhat_leading_singularity(*lv).is_polynomial(C.X11, C.X22)
                   for lv in C.PAPER_TABLE_1))
    check("the zero-zero level is just 1",
          C.dhat_leading_singularity(0, 0), 1)


# ---------------------------------------------------------------------------
# [2] the mirror
# ---------------------------------------------------------------------------

def test_mirror():
    print("\n[2] the mirror rows are produced, not assumed")

    swap = {C.X11: C.X22, C.X22: C.X11}
    for a, b in [(1, 0), (2, 0), (2, 1)]:
        lhs = C.dhat_leading_singularity(b, a)
        rhs = C.dhat_leading_singularity(a, b).subs(swap, simultaneous=True)
        check_true("level (%d,%d) mirrors (%d,%d)" % (b, a, a, b),
                   sp.simplify(sp.expand(lhs - rhs)) == 0)

    check_true("the doubly-massive level is its own mirror",
               sp.simplify(C.dhat_leading_singularity(1, 1)
                           - C.dhat_leading_singularity(1, 1).subs(
                               swap, simultaneous=True)) == 0)


# ---------------------------------------------------------------------------
# [3] the contradiction
# ---------------------------------------------------------------------------

def test_contradiction():
    print("\n[3] the matching, level by level, and where it breaks")

    r = C.unitarity_constraints()
    steps = {s["level"]: s for s in r["steps"]}

    check("the first massless-massive level fixes one exponent",
          steps[(1, 0)]["fixed"], {C.X22: 0})
    check("its mirror fixes the other", steps[(0, 1)]["fixed"], {C.X11: 0})
    check_true("the second pair is then automatically satisfied",
               steps[(2, 0)]["status"] == "already satisfied"
               and steps[(0, 2)]["status"] == "already satisfied")
    check("the doubly-massive level fixes the dimension",
          steps[(1, 1)]["fixed"], {C.DIM: 4})
    check("and four is what it fixes it to", r["solution"][C.DIM], 4)

    check("the next level has no solution", steps[(2, 1)]["status"],
          "no solution")
    check("the two sides there are 2 and 45/16",
          (steps[(2, 1)]["lhs"], steps[(2, 1)]["rhs"]),
          (sp.Integer(2), sp.Rational(45, 16)))
    check_true("so the system is inconsistent", not r["consistent"])
    check("and the first failure is at level (2,1)", r["first_failure"],
          (2, 1))

    # the near miss: drop the mirror levels and the same system looks fine,
    # because only one exponent is ever determined
    solution = {}
    statuses = []
    for level in [(0, 0), (1, 0), (2, 0), (1, 1), (2, 1)]:
        eq = sp.expand(C.dhat_leading_singularity(*level).subs(solution)
                       - _paper(level, "ansatz").subs(solution))
        if eq == 0:
            statuses.append("ok")
            continue
        unknowns = sorted(eq.free_symbols & {C.X11, C.X22, C.DIM},
                          key=lambda s: s.name)
        sols = sp.solve(eq, unknowns, dict=True)
        statuses.append("ok" if sols else "fail")
        if sols:
            solution.update(sols[0])
    check_true("without the mirror levels it looks satisfiable",
               "fail" not in statuses)
    check_true("and the dimension comes out wrong when it does",
               solution.get(C.DIM) not in (None, 4))


# ---------------------------------------------------------------------------
# [4] the baby integrals
# ---------------------------------------------------------------------------

def test_baby():
    print("\n[4] the truncated integrals fail a different way")

    values = [C.baby_leading_singularity(*lv) for lv in [(1, 1), (2, 1), (1, 2)]]
    want = -C.DELTA1 - C.DELTA2
    for lv, v in zip([(1, 1), (2, 1), (1, 2)], values):
        check_true("baby level %s is -Delta1 - Delta2" % (lv,),
                   sp.simplify(v - want) == 0)
    check_true("three different levels give the same value",
               sp.simplify(values[0] - values[1]) == 0
               and sp.simplify(values[1] - values[2]) == 0)

    # and the tree side does not, which is the whole argument
    a11 = _paper((1, 1), "ansatz")
    a21 = _paper((2, 1), "ansatz")
    check_true("while the tree side gives different values at those levels",
               sp.simplify(a11 - a21) != 0)

    # truncating the eta product is not an approximation below its order
    for order in (4, 6, 10):
        check_true("residue is independent of the truncation order (%d)" % order,
                   sp.simplify(C.baby_leading_singularity(2, 2, order=order)
                               - C.baby_leading_singularity(2, 2, order=8)) == 0)

    check_true("the level-one residue fixes an exponent to zero",
               sp.simplify(C.tachyon_level_one_residue()
                           - (1 - C.X22 - C.ALPHA0)) == 0)


# ---------------------------------------------------------------------------
# [5] the contour
# ---------------------------------------------------------------------------

def test_contour():
    print("\n[5] the contour, counted")

    check("the 5-gon has five chords", C.associahedron_faces(5, 1), 5)
    check("and five vertices", C.associahedron_faces(5, 2), 5)
    check("the 6-gon has nine chords", C.associahedron_faces(6, 1), 9)

    # the top face count is the Catalan number, which was not put in
    for n in (5, 6, 7):
        cat = sp.binomial(2 * (n - 2), n - 2) / (n - 1)
        check("triangulations of the %d-gon" % n,
              C.associahedron_faces(n, n - 3), int(cat))

    p = C.pochhammer_pieces(5)
    check("sheets at five points", p["sheets"], 32)
    check("tubes at five points", p["tubes"], 80)
    check("tori at five points", p["tori"], 40)
    check("sheets are one per subset of chords",
          C.pochhammer_pieces(6)["sheets"], 2 ** 9)

    check("the cutoff at five points is log 2", C.minimal_cutoff(5), sp.log(2))
    check_true("at four points any positive cutoff will do",
               C.minimal_cutoff(4) == 0)
    check_true("and the cutoff grows with multiplicity",
               C.minimal_cutoff(9) > C.minimal_cutoff(5))


# ---------------------------------------------------------------------------
# [6] the ledger
# ---------------------------------------------------------------------------

def test_ledger():
    print("\n[6] registration, and refusals of the right kind")

    check_true("registered", "cuts-and-contours" in T.registry)
    cls = T.get("cuts-and-contours")
    check_true("and is a SurfaceTheory, not a Theory",
               issubclass(cls, SurfaceTheory) and not issubclass(cls, T.Theory))

    cc = C.CutsAndContours()
    check("cuts() at the lowest level", cc.cuts(0, 0), 1)
    check_true("leading_singularity is the same computation",
               sp.simplify(cc.leading_singularity(2, 1) - cc.cuts(2, 1)) == 0)
    check_true("the baby integrand is reachable too",
               cc.cuts(1, 1, integrand="baby") != 0)
    check_true("an unknown integrand is an error, not a guess",
               _raises(ValueError, cc.cuts, 1, 1, "nonsense"))

    check_true("convergence of the contour is a theorem",
               _raises(NotAnalytic, cc.contour_is_finite))
    check_true("evaluation at large kinematics is a precision problem",
               _raises(NotAnalytic, cc.evaluate))
    check_true("the alpha' expansion needs integration",
               _raises(NeedsIntegration, cc.alpha_prime_expansion))
    check_true("no cross-section", _raises(NeedsIntegration, cc.cross_section))

    try:
        cc.contour_is_finite()
    except NotAnalytic as e:
        check_true("NotAnalytic lists what is still checkable",
                   len(e.checkable) >= 3)

    text = cc.describe()
    check_true("describe() separates exact from declined",
               "exact:" in text and "declined:" in text)

    # all three amplitude modules registered, and none of them a Theory
    keys = ["gluon-leading-singularity", "soft-factorisation",
            "cuts-and-contours"]
    check_true("all three amplitude modules are registered",
               all(k in T.registry for k in keys))
    check_true("and none of them is a compactification",
               all(not issubclass(T.get(k), T.Theory) for k in keys))


def main():
    t0 = time.time()
    print("=" * 72)
    print("test_contours: cuts, contours, and a failure of unitarity")
    print("=" * 72)

    test_table()
    test_mirror()
    test_contradiction()
    test_baby()
    test_contour()
    test_ledger()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_contours: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
