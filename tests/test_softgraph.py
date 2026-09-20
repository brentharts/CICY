"""
Tests for pyCICY.theories.softgraph.

The second amplitude module, and the one where the package's usual discipline
is easiest to apply: every polynomial here can be built by linear algebra or
by enumeration, and the two must agree.

  [1] symanzik      U from the determinant of the reduced Laplacian and from
                    spanning trees; F from the adjugate and from spanning
                    2-forests. Then both against the published one-loop
                    closed forms, which is what fixes the sign conventions
                    rather than asserting them
  [2] worldline     inverting a jet block gives beta_{min(v,w)}. Not a change
                    of notation but a matrix identity, checked for jets of
                    length 1 through 5
  [3] rays          the divergent scalings, found by enumeration over a
                    bounded box rather than read off the paper: the one-loop
                    vertex has exactly one, the planar ladder exactly two, and
                    the non-planar ladder does not have the planar's partial
                    scaling at all
  [4] factorisation U -> U_H U_S and F -> U_H F_S + U_S F_H under a partial
                    soft scaling, and under the full one a product of two
                    one-loop factors
  [5] topology      the headline: planar and non-planar ladders are different
                    graphs with different Symanzik polynomials, and in
                    worldline variables their soft integrands are identical.
                    The whole difference has moved into the domain
  [6] ledger        NotAnalytic for the finiteness theorem, NeedsIntegration
                    for the resummed anomalous dimension, registration

Run with:  python3 tests/test_softgraph.py
       or: python3 run_tests.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sympy as sp

from pyCICY import theories as T
from pyCICY.theories import softgraph as SG
from pyCICY.theories.surface import (SurfaceTheory, NotAnalytic,
                                     NeedsIntegration)

FAILURES = []

M = sp.Symbol("m", positive=True)
S = sp.Symbol("s")
R_PARTIAL = {"a11": 0, "a12": 1, "a21": 0, "a22": 1, "g1": 0, "g2": 2}
R_FULL = {"a11": 1, "a12": 1, "a21": 1, "a22": 1, "g1": 2, "g2": 2}


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


def _rays(found):
    """Search output as a set of sorted integer tuples, for comparison."""
    return {tuple(int(r[k]) for k in sorted(r)) for r, _ in found}


# ---------------------------------------------------------------------------
# [1] the Symanzik polynomials
# ---------------------------------------------------------------------------

def test_symanzik():
    print("\n[1] Symanzik polynomials: determinant against enumeration")

    graphs = [SG.one_loop_vertex(), SG.planar_ladder(), SG.nonplanar_ladder()]
    for g in graphs:
        check_true("%s: U two ways" % g.name,
                   sp.expand(g.U() - g.U_spanning_trees()) == 0)
        check_true("%s: F two ways" % g.name,
                   sp.expand(g.F() - g.F_forests()) == 0)

    check("one-loop vertex has one loop", graphs[0].loops, 1)
    check("both ladders have two loops",
          (graphs[1].loops, graphs[2].loops), (2, 2))

    # against the published closed forms (2.2) and (2.4)
    g = graphs[0]
    a11, a21, g1 = g.alpha["a11"], g.alpha["a21"], g.alpha["g1"]
    check_true("U is the sum of the three parameters",
               sp.expand(g.U() - (a11 + a21 + g1)) == 0)
    check_true("F collapses to s a11 a21 - m^2 (a11 + a21)^2",
               sp.expand(g.F() - SG.one_loop_F(a11, a21)) == 0)
    check_true("F has no photon parameter in it, but U does",
               (not g.F().has(g1)) and g.U().has(g1))

    # the ladders are genuinely different graphs
    check_true("planar and non-planar have different U",
               sp.expand(graphs[1].U() - graphs[2].U()) != 0)


# ---------------------------------------------------------------------------
# [2] worldline variables
# ---------------------------------------------------------------------------

def test_worldline():
    print("\n[2] inverting a jet block produces the worldline distances")

    for length in range(1, 6):
        inv, beta = SG.worldline_inverse(length)
        want = sp.Matrix(length, length, lambda i, j: beta[min(i, j)])
        check_true("jet of length %d: J^-1 = beta_min(v,w)" % length,
                   sp.simplify(inv - want) == sp.zeros(length, length))

    # beta is a partial sum, which is what makes it a distance
    _, beta = SG.worldline_inverse(4)
    a = [sp.Symbol("alpha_%d" % (k + 1), positive=True) for k in range(4)]
    check_true("beta_k = alpha_1 + ... + alpha_k",
               all(sp.expand(beta[k] - sum(a[:k + 1])) == 0 for k in range(4)))


# ---------------------------------------------------------------------------
# [3] divergent rays, found rather than quoted
# ---------------------------------------------------------------------------

def test_rays():
    print("\n[3] divergent rays by enumeration over a bounded box")

    one = SG.one_loop_vertex()
    found, box = SG.search_soft_rays(one, bound=2)
    check("one-loop vertex: rays found", len(found), 1)
    check("and it is the published scaling", _rays(found), {(1, 1, 2)})
    check_true("the search reports its own box", "0..2" in box)

    pl, npl = SG.planar_ladder(), SG.nonplanar_ladder()
    found, _ = SG.search_soft_rays(pl, bound=2)
    check("planar ladder: rays found", len(found), 2)
    check("and they are the published pair", _rays(found),
          {(0, 1, 0, 1, 0, 2), (1, 1, 1, 1, 2, 2)})

    # the non-planar ladder does not have the partial scaling at all, which is
    # why it contributes only a single pole
    check("planar, partial scaling", SG.ray_divergence(pl, R_PARTIAL), "log")
    check("planar, full scaling", SG.ray_divergence(pl, R_FULL), "log")
    check("non-planar, full scaling", SG.ray_divergence(npl, R_FULL), "log")
    check("non-planar, partial scaling is not even soft",
          SG.ray_divergence(npl, R_PARTIAL), "not-soft")

    # the counting itself: log means the exponent vanishes
    check("tropical function vanishes on a log ray",
          SG.tropical_function(pl, R_FULL, 4), 0)
    check_true("and does not vanish away from four dimensions",
               SG.tropical_function(pl, R_FULL, 6) != 0)


# ---------------------------------------------------------------------------
# [4] hard-soft factorisation
# ---------------------------------------------------------------------------

def test_factorisation():
    print("\n[4] the integrand factorises along each ray")

    pl = SG.planar_ladder()
    a = pl.alpha
    U_H = a["a11"] + a["a21"] + a["g1"]
    U_S = a["g2"]

    out = SG.soft_integrand(pl, R_PARTIAL, worldline=False)
    check_true("partial scaling: U -> U_H U_S",
               sp.expand(out["U"] - U_H * U_S) == 0)
    F_H = SG.one_loop_F(a["a11"], a["a21"])
    F_S = SG.one_loop_F(a["a12"], a["a22"])
    check_true("partial scaling: F -> U_H F_S + U_S F_H",
               sp.expand(out["F"] - (U_H * F_S + U_S * F_H)) == 0)
    check_true("F and U scale together, as a soft ray requires",
               out["F_degree"] == out["U_degree"])

    out = SG.soft_integrand(pl, R_FULL)
    b11, b12, b21, b22 = sp.symbols("beta11 beta12 beta21 beta22",
                                    positive=True)
    g1, g2 = a["g1"], a["g2"]
    check_true("full scaling: U -> g1 g2, the two soft loops alone",
               sp.expand(out["U"] - g1 * g2) == 0)
    check_true("full scaling: F is two one-loop factors in worldline distances",
               sp.expand(out["F"] - (g2 * SG.one_loop_F(b11, b21)
                                     + g1 * SG.one_loop_F(b12, b22))) == 0)


# ---------------------------------------------------------------------------
# [5] two topologies, one integrand
# ---------------------------------------------------------------------------

def test_topology():
    print("\n[5] planar and non-planar: same integrand, different domain")

    pl, npl = SG.planar_ladder(), SG.nonplanar_ladder()

    raw_pl = SG.soft_integrand(pl, R_FULL, worldline=False)
    raw_npl = SG.soft_integrand(npl, R_FULL, worldline=False)
    check_true("in Schwinger parameters the two F differ",
               sp.expand(raw_pl["F"] - raw_npl["F"]) != 0)

    wl_pl = SG.soft_integrand(pl, R_FULL)
    wl_npl = SG.soft_integrand(npl, R_FULL)
    check_true("in worldline variables the two U agree",
               sp.expand(wl_pl["U"] - wl_npl["U"]) == 0)
    check_true("and the two F agree identically",
               sp.expand(wl_pl["F"] - wl_npl["F"]) == 0)

    check_true("the difference has moved into the domain",
               wl_pl["domain"] != wl_npl["domain"])
    check_true("both order the first jet the same way",
               "beta11 < beta12" in wl_pl["domain"]
               and "beta11 < beta12" in wl_npl["domain"])
    check_true("and order the second jet oppositely",
               "beta21 < beta22" in wl_pl["domain"]
               and "beta22 < beta21" in wl_npl["domain"])


# ---------------------------------------------------------------------------
# [6] the ledger
# ---------------------------------------------------------------------------

def test_ledger():
    print("\n[6] registration, and refusals of the right kind")

    check_true("registered", "soft-factorisation" in T.registry)
    cls = T.get("soft-factorisation")
    check_true("and is a SurfaceTheory, not a Theory",
               issubclass(cls, SurfaceTheory) and not issubclass(cls, T.Theory))

    sf = SG.SoftFactorisation()
    out = sf.soft_limit(ray=R_FULL)
    check("soft_limit reports the divergence", out["divergence"], "log")
    check_true("and returns the leading U", out["U_leading"] != 0)

    check_true("finiteness of the remainder is a theorem, not a number",
               _raises(NotAnalytic, sf.remainder_is_finite))
    check_true("the soft anomalous dimension needs integration",
               _raises(NeedsIntegration, sf.soft_anomalous_dimension))
    check_true("leading singularities belong to the other module",
               _raises(NotImplementedError, sf.leading_singularity))
    check_true("no cross-section", _raises(NeedsIntegration, sf.cross_section))

    try:
        sf.remainder_is_finite()
    except NotAnalytic as e:
        check_true("NotAnalytic still says what can be checked",
                   len(e.checkable) >= 2)
    try:
        sf.soft_anomalous_dimension()
    except NeedsIntegration as e:
        check_true("NeedsIntegration names the missing integrations",
                   len(e.missing) >= 3)

    text = sf.describe()
    check_true("describe() separates exact from declined",
               "exact:" in text and "declined:" in text)


def main():
    t0 = time.time()
    print("=" * 72)
    print("test_softgraph: soft factorisation in Schwinger space")
    print("=" * 72)

    test_symanzik()
    test_worldline()
    test_rays()
    test_factorisation()
    test_topology()
    test_ledger()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_softgraph: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
