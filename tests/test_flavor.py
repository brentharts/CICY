"""
Tests for pyCICY.flavor.

This suite tests an implementation of a specific paper, arXiv:2511.10685, so
it has two jobs that must not be confused. The first is to check that the
code implements what the paper says. The second is to record what follows
from that, including where a stated conclusion does not follow from its
stated inputs.

Both kinds of check are here and both are labelled. A test named
"paper's stated value" asserts what the paper claims; a test named
"the formula as written gives" asserts what the code computes. Where those
differ the suite passes -- it is not the code that is wrong -- and the
difference is the finding.

  [1] hypercharges   all fifteen, in exact rational arithmetic
  [2] anomaly        Tr(Y) = 0 per generation
  [3] is it a fit    ranks, and the generation-3 exception explained
  [4] geometry       MDP distortion, Gram matrix, J, TBM
  [5] estimates      theta_13 reproduces the paper; the Cabibbo does not

Run with:  python3 tests/test_flavor.py
       or: python3 run_tests.py
"""

import math
import os
import sys
import time
from fractions import Fraction

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import flavor as fl
from pyCICY import polytope as P

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


def check_close(name, got, want, tol):
    ok = abs(float(got) - float(want)) <= tol
    print("  {:<58} {:>12} {}".format(name, "%.4f" % got,
                                      "ok" if ok else "FAIL want %s" % want))
    if not ok:
        FAILURES.append(name)


def test_hypercharges():
    print("\n[1] the hypercharge functional, exactly")
    ok, table = fl.verify_hypercharges()
    check_true("all fifteen hypercharges reproduced exactly", ok)
    for g, name, got, want, good in table[:5]:
        check("gen 1 %s" % name, got, want)
    check_true("every value is an exact Fraction, not a float",
               all(isinstance(r[2], Fraction) for r in table))

    used, unused = fl.distinct_vertices()
    check("fifteen distinct vertices used", len(used), 15)
    check("one vertex of V_2 left over", unused, {(-1, 1, -1, 1)})

    # The vertices really are vertices of the 24-cell in the section 2.1
    # normalisation, which polytope.py builds independently.
    unit = {tuple(np.round(v, 6)) for v in P.twenty_four_cell("unit")}
    check_true("all assigned vertices lie on the 24-cell",
               all(tuple(np.round([x / 2.0 for x in v], 6)) in unit
                   for v in used))


def test_anomaly():
    print("\n[2] anomaly cancellation")
    check("Tr(Y) over a generation", fl.anomaly_trace(), Fraction(0))
    # Spelled out, since the cancellation is between contributions of
    # opposite sign and a zero could otherwise be vacuous.
    contribs = {k: v[0] * v[1] for k, v in fl.SM_HYPERCHARGES.items()}
    check("contributions", sorted(map(str, contribs.values())),
          sorted(["-1", "1", "-1", "2", "-1"]))
    check_true("they do not all vanish individually",
               any(c != 0 for c in contribs.values()))


def test_is_it_a_fit():
    print("\n[3] prediction or interpolation")
    for g in (1, 2, 3):
        check("gen %d: 5x5 system has full rank" % g, fl.fit_rank(g)["rank"], 5)
    check_true("so a unique h_Y exists for any targets whatever",
               all(fl.fit_rank(g)["unique"] for g in (1, 2, 3)))

    # Generation 3 is the exception: eps = 0 leaves four unknowns for five
    # equations, and it is nevertheless consistent.
    a1 = fl.epsilon_zero_analysis(1)
    a2 = fl.epsilon_zero_analysis(2)
    a3 = fl.epsilon_zero_analysis(3)
    check_true("gen 1 cannot set eps = 0", not a1["consistent"])
    check_true("gen 2 cannot set eps = 0", not a2["consistent"])
    check_true("gen 3 can", a3["consistent"])
    check("gen 3 null vector c", [int(x) for x in a3["c"]], [-1, 1, -1, -2, 1])
    check("and c . Y vanishes", a3["c_dot_Y"], 0)
    check("the solved h is the paper's h_Y^(3)",
          [str(x) for x in a3["h"]], ["1/3", "1", "1/2", "-1/6"])

    # The reason: with only the Yukawa relations imposed, c . Y collapses to
    # -4 Y_H - 2 Y_eR, which vanishes because Y_eR = -1 and Y_H = 1/2. So it
    # is a fact about the SM hypercharges, not about the 24-cell.
    import sympy as sp
    Yl, Yq, Ye, Yu, Yd, YH = sp.symbols("Yl Yq Ye Yu Yd YH")
    expr = -Yl + Yq - Ye - 2 * Yu + Yd
    reduced = sp.simplify(expr.subs({Yl: Ye + YH, Yu: Yq + YH, Yd: Yq - YH}))
    check("c . Y reduces to -4 Y_H - 2 Y_eR", str(sp.expand(reduced)),
          str(sp.expand(-4 * YH - 2 * Ye)))
    check("which vanishes at Y_eR = -1, Y_H = 1/2",
          sp.simplify(reduced.subs({Ye: -1, YH: sp.Rational(1, 2)})), 0)

    # ... and it is not rare. About one choice in thirty has the property.
    hits, total, frac = fl.epsilon_zero_census()
    check("census hits", hits, 14592)
    check("census total", total, 437760)
    check_close("fraction admitting eps = 0", 100 * frac, 3.33, 0.01)
    check_true("so gen 3 is one of a family, not a unique selection",
               hits > 1000)


def test_geometry():
    print("\n[4] the projection and tribimaximal mixing")

    # The Minimal Distortion Principle has nothing to minimise.
    d = fl.mdp_distortion()
    check_true("MDP distortion is zero (%.1e)" % d, d < 1e-12)
    check_true("four points always span at most three dimensions",
               np.linalg.matrix_rank(fl.TETRAHEDRON - fl.TETRAHEDRON.mean(0))
               <= 3)

    G = fl.gram_matrix()
    off = G[np.triu_indices(4, 1)]
    check_true("all six Gram off-diagonals are -1/3",
               np.allclose(off, -1 / 3, atol=1e-12))
    check_close("tetrahedral angle in degrees",
                math.degrees(math.acos(-1 / 3)), 109.4712, 1e-3)

    w = np.linalg.eigvalsh(fl.J_matrix())
    check_true("J has eigenvalues 1/3, 4/3, 4/3",
               np.allclose(sorted(w), [1 / 3, 4 / 3, 4 / 3], atol=1e-12))

    U = fl.U_TBM()
    check_true("U_TBM is orthogonal", np.allclose(U.T @ U, np.eye(3)))
    D = U.T @ fl.J_matrix() @ U
    check_true("U_TBM diagonalises J",
               np.allclose(D - np.diag(np.diag(D)), 0, atol=1e-12))

    t12, t13, t23 = fl.tbm_angles()
    check_close("TBM theta_12", t12, 35.2644, 1e-3)
    check_close("TBM theta_23", t23, 45.0, 1e-9)
    check_close("TBM theta_13", t13, 0.0, 1e-9)
    # TBM is a starting point, not the answer: theta_12 is 2 degrees off and
    # theta_13 is 8.5 degrees off, which is what the perturbation must supply.
    check_true("TBM theta_12 differs from the measured 33.4 degrees",
               abs(t12 - 33.4) > 1.5)

    # The perturbation must be symmetric and traceless, and is checked to be.
    try:
        fl.neutrino_mass_matrix(1.0, 0.02, np.eye(3))
        check_true("a non-traceless C is rejected", False)
    except ValueError:
        check_true("a non-traceless C is rejected", True)


def test_estimates():
    print("\n[5] theta_13 and the Cabibbo angle")

    # theta_13: the paper's own numbers reproduce its own answer.
    t13 = fl.theta13_from_strain(0.017, 0.022)
    check_close("theta_13 from eps_13 = 0.017, eta = 0.022", t13, 8.5, 0.1)
    check_true("so the reactor-angle estimate is internally consistent",
               abs(t13 - 8.5) < 0.1)

    # The Cabibbo angle: the same formula with the paper's own numbers does
    # not give the paper's own answer.
    lo = fl.cabibbo_angle(0.02)
    hi = fl.cabibbo_angle(0.03)
    check_close("the formula as written, eta = 0.02", lo["tan_theta"],
                0.0327, 1e-3)
    check_close("the formula as written, eta = 0.03", hi["tan_theta"],
                0.0490, 1e-3)
    check_true("paper's stated value 0.22-0.26 is not in that range",
               not (lo["tan_theta"] <= 0.22 <= hi["tan_theta"]))
    check_close("eta required for the measured Cabibbo angle",
                lo["eta_for_quoted"], 0.138, 1e-3)
    ratio = lo["eta_for_quoted"] / 0.022
    check_true("which is %.1f times the eta the reactor angle needs" % ratio,
               ratio > 5.0)
    check_close("measured Cabibbo angle in degrees", lo["measured_degrees"],
                13.0, 0.1)


def main():
    t0 = time.time()
    test_hypercharges()
    test_anomaly()
    test_is_it_a_fit()
    test_geometry()
    test_estimates()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_flavor: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
