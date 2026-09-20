"""
Tests for pyCICY.theories.surface, .surfaceology and .gluons.

The first amplitude modules in the package, and the first constructions whose
objects are graphs and curves rather than configuration matrices. The tests
follow the same discipline as everything else here: one quantity, two routes,
no shared code -- and a refusal is required to be the right kind of refusal.

  [1] u_equations   the positive parametrisation, checked against the
                    equations that define it. u_variable knows about
                    F-polynomials and nothing about which chords of a polygon
                    interleave; crossing knows the interleaving and nothing
                    about F-polynomials. Every residual must vanish
                    identically as a rational function, not at sample points
  [2] integrand     the Koba-Nielsen exponents at five points against the
                    published form -- the check that fixes the index
                    conventions the literature states for a different range
  [3] three_point   the scaffolded three-gluon vertex derived from
                    polarisation contractions: the gauge parameters must
                    cancel on their own, the squares of the dual coordinates
                    must cancel on their own, and six monomials must remain
  [4] counting      the alternating binomial sum that makes the extensions
                    around a puncture contribute exactly 1, hence the (2 - D)
                    coefficient at any multiplicity, and the closed-curve
                    exponent rule that settles the paper's open question
  [5] lucas         the coefficients of the n-gon leading singularity from a
                    two-term recurrence and from a binomial closed form, then
                    both against the paper's table. The agreement for
                    n = 4..8 is the point: an integer sequence nobody put in
  [6] ledger        NeedsIntegration for a cut, NotAnalytic for fermion loops,
                    registration beside the compactifications, describe()

Run with:  python3 tests/test_surface.py
       or: python3 run_tests.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sympy as sp

from pyCICY import theories as T
from pyCICY.theories import gluons as G
from pyCICY.theories import surfaceology as S
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


# ---------------------------------------------------------------------------
# [1] the u-equations
# ---------------------------------------------------------------------------

def test_u_equations():
    print("\n[1] u-equations: the parametrisation against its own definition")

    check("chords of the 5-gon", len(S.chords(5)), 5)
    check("chords of the 8-gon, n(n-3)/2", len(S.chords(8)), 20)
    check("(1,3) and (2,4) interleave", S.crossing((1, 3), (2, 4)), 1)
    check("(1,3) and (1,4) share an endpoint", S.crossing((1, 3), (1, 4)), 0)
    check("(1,3) and (4,6) are disjoint", S.crossing((1, 3), (4, 6)), 0)

    for n in (4, 5, 6):
        res = S.u_equation_residuals(n)
        bad = [c for c, r in res.items() if r != 0]
        check_true("u-equations hold identically at n=%d" % n, not bad)

    # the u's are genuinely in [0,1] on the positive orthant: check exactly at
    # a rational point rather than numerically
    y = {k: sp.Rational(k + 1, 7) for k in range(3, 6)}
    us = S.u_variables(6, y)
    check_true("every u in (0,1) at a sampled rational point",
               all(0 < v < 1 for v in us.values()))


# ---------------------------------------------------------------------------
# [2] the integrand
# ---------------------------------------------------------------------------

def _exponents(expr, nvars):
    """Factor a ratio of polynomials and return {factor: exponent}."""
    num, den = sp.fraction(sp.cancel(sp.factor(expr)))
    out = {}
    for part, sign in ((num, 1), (den, -1)):
        for f, p in sp.factor_list(sp.expand(part))[1]:
            f = sp.expand(f)
            out[f] = out.get(f, 0) + sign * p
    return out


def test_integrand():
    print("\n[2] Koba-Nielsen exponents at five points")

    n = 5
    y = S.y_symbols(n)
    X = S.planar_variables(n)
    total = {}
    for c, u in S.u_variables(n, y).items():
        for f, p in _exponents(u, n).items():
            total[f] = total.get(f, 0) + p * X[c]

    y3, y4 = y[3], y[4]
    X13, X14 = X[(1, 3)], X[(1, 4)]
    X24, X25, X35 = X[(2, 4)], X[(2, 5)], X[(3, 5)]
    want = {sp.expand(y3): X13,
            sp.expand(y4): X14,
            sp.expand(1 + y3): -X13 + X14 - X24,
            sp.expand(1 + y4): -X24 + X25 - X35,
            sp.expand(1 + y4 + y3 * y4): -X14 + X24 - X25}

    check("distinct factors in the integrand", len(total), 5)
    for f, e in want.items():
        got = sp.simplify(total.get(f, sp.Integer(0)) - e)
        check_true("exponent of (%s)" % f, got == 0)

    # c_{i,j} built from planar variables must reproduce those exponents
    check_true("c_{1,4} matches the exponent of its F-polynomial",
               sp.simplify(S.nonplanar_variable(1, 4, X)
                           - (X14 - X24 + X25)) == 0)


# ---------------------------------------------------------------------------
# [3] the three-point vertex, derived
# ---------------------------------------------------------------------------

def test_three_point():
    print("\n[3] the scaffolded three-gluon vertex, derived not quoted")

    r = G.scaffolded_three_point(gauge=True)
    check_true("gauge parameters cancel on their own",
               r["gauge_dependence"] == [])
    check_true("squares of the dual coordinates cancel on their own",
               r["coordinate_dependence"] == [])
    check("monomials remaining", r["monomials"], 6)

    paper = sp.sympify(G.PAPER_THREE_POINT)
    ratio = sp.simplify(sp.cancel(r["amplitude"] / paper))
    check_true("agrees with the published vertex up to a constant",
               ratio.is_number)
    # the paper drops factors of two and restores them as 2**(vertices)
    check_true("and that constant is a power of two",
               ratio.is_number and sp.log(abs(ratio), 2).is_integer)

    # fixing the gauge by hand must not change the answer
    r0 = G.scaffolded_three_point(gauge=False)
    check_true("gauge choice does not change the result",
               sp.expand(r0["amplitude"] - r["amplitude"]) == 0)


# ---------------------------------------------------------------------------
# [4] counting around a puncture
# ---------------------------------------------------------------------------

def test_counting():
    print("\n[4] extensions, closed curves, and the (2 - D) coefficient")

    for n in range(1, 13):
        if G.extension_balance(n) != 1:
            FAILURES.append("extension_balance(%d)" % n)
    check("extension balance is 1 for n = 1..12",
          sorted({G.extension_balance(n) for n in range(1, 13)}), [1])

    Dsym = G.D
    check("closed curve homotopic to a boundary",
          sp.expand(G.closed_curve_exponent(True, Dsym) - (1 - Dsym)), 0)
    check("any other closed curve",
          sp.expand(G.closed_curve_exponent(False, Dsym) + Dsym), 0)

    for n in (2, 3, 5, 9):
        got = sp.expand(G.ngon_puncture_coefficient(n, Dsym) - (2 - Dsym))
        check_true("all-puncture coefficient is 2 - D at n=%d" % n, got == 0)

    # the bubble is not a separate result, it is n = 2
    check_true("the bubble's (2 - D) is the n=2 case",
               sp.expand(G.ngon_puncture_coefficient(2, Dsym)
                         - G.ngon_leading_singularity(2, Dsym).coeff(
                             sp.Symbol("Y"), 2)) == 0)

    # a numerical dimension flows through unchanged
    check("at D = 4 the coefficient is -2",
          sp.expand(G.ngon_puncture_coefficient(4, 4)), -2)


# ---------------------------------------------------------------------------
# [5] the Lucas coefficients
# ---------------------------------------------------------------------------

def test_lucas():
    print("\n[5] Lucas coefficients: recurrence, closed form, and the paper")

    for n in range(2, 21):
        if G.lucas_coefficients(n) != G.lucas_coefficients_closed_form(n):
            FAILURES.append("lucas routes disagree at n=%d" % n)
    check_true("recurrence and binomial closed form agree, n = 2..20",
               all(G.lucas_coefficients(n)
                   == G.lucas_coefficients_closed_form(n)
                   for n in range(2, 21)))

    # the recurrence itself, as an identity in the polynomial ring
    x = sp.Symbol("x")
    ok = all(sp.expand(G.lucas_polynomial(n, x)
                       - x * G.lucas_polynomial(n - 1, x)
                       + G.lucas_polynomial(n - 2, x)) == 0
             for n in range(2, 15))
    check_true("L_n = x L_{n-1} - L_{n-2} holds as a polynomial identity", ok)

    comp = G.paper_comparison()
    for n in range(4, 9):
        check_true("n-gon matches the published table at n=%d" % n,
                   comp[n]["agrees"])
    # the two documented exceptions
    check("triangle differs by an overall factor on the X part",
          comp[3]["x_part_ratio"], 2)
    check_true("bubble has no X-dependent part in the table",
               comp[2]["x_part_ratio"] == 0)

    # and a prediction past the end of the table
    n = 9
    ls = G.ngon_leading_singularity(n)
    Y = sp.Symbol("Y")
    check("nine-gon X-dependent monomials, one per Lucas coefficient",
          len([t for t in sp.Add.make_args(ls) if t.has(sp.Symbol("X"))]),
          len(G.lucas_coefficients(n)))
    check_true("and its Y**9 coefficient is still 2 - D",
               sp.expand(ls.coeff(Y, n) - (2 - G.D)) == 0)


# ---------------------------------------------------------------------------
# [6] the ledger
# ---------------------------------------------------------------------------

def test_ledger():
    print("\n[6] registration, and refusals of the right kind")

    check_true("registered beside the compactifications",
               "gluon-leading-singularity" in T.registry)
    cls = T.get("gluon-leading-singularity")
    check_true("get() returns the class", cls is G.GluonLeadingSingularity)
    check_true("but is not a Theory",
               not issubclass(cls, T.Theory) and issubclass(cls, SurfaceTheory))

    g = G.GluonLeadingSingularity()
    check_true("leading_singularity(three_point=True) returns a polynomial",
               g.leading_singularity(three_point=True).free_symbols != set())
    check_true("cuts() needs an integration this package does not do",
               _raises(NeedsIntegration, g.cuts))
    check_true("fermion loops decline rather than guess",
               _raises(NotAnalytic, g.fermion_loop))
    check_true("no cross-section", _raises(NeedsIntegration, g.cross_section))
    check_true("soft limits belong to a different construction",
               _raises(NotImplementedError, g.soft_limit))

    # the exceptions carry their specifications
    try:
        g.cuts()
    except NeedsIntegration as e:
        check_true("NeedsIntegration names what is missing", len(e.missing) >= 2)
    try:
        g.fermion_loop()
    except NotAnalytic as e:
        check_true("NotAnalytic names what is still checkable",
                   len(e.checkable) >= 1)

    text = g.describe()
    check_true("describe() separates exact from declined",
               "exact:" in text and "declined:" in text)
    check_true("and does not promise a cross-section",
               "no cross-section here" in text)


def main():
    t0 = time.time()
    print("=" * 72)
    print("test_surface: amplitudes on surfaces")
    print("=" * 72)

    test_u_equations()
    test_integrand()
    test_three_point()
    test_counting()
    test_lucas()
    test_ledger()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_surface: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
