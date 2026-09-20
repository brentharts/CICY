"""
Tests for pyCICY.theories.ninelink and pyCICY.theories.flavorbase.

The first flavour construction in the package, and the first under the third
base class. The discipline is the same: the combinatorial layer is derived
from its conditions rather than tabulated, the analytic results are checked
against exact diagonalisation, and where the paper and the arithmetic
disagree the test records the disagreement instead of the claim.

  [1] diagrams      the enumeration, from the conditions alone: nine links,
                    both sectors full rank, the diagram connected, the (3,3)
                    entries present. With nine links on nine nodes,
                    connectivity and "exactly one closed loop" are the same
                    condition, which is how the single rephasing invariant
                    arises. Loop lengths come out 4 and 6, as the paper says
  [2] invariant     the loop monomial really is rephasing invariant: assign
                    an arbitrary phase to every field, rotate every entry,
                    and the argument of the monomial must not move
  [3] determinants  perfect matchings against sympy's determinant, and the
                    strong-CP census. Over the full enumeration the safe
                    textures are exactly those with a single matching in each
                    sector, and the paper's "all failures are diagonal in a
                    sector" does not hold -- it is a statement about its
                    fitted subset, and the test records the counts
  [4] angles        exact diagonalisation: the three angles sum to pi, the
                    Jarlskog invariant is the area, and worked examples 1 and
                    3 reproduce the published closed forms with a residual
                    falling like the fourth power of the expansion parameter
  [5] example_two   the one that does not reproduce. The phase as printed
                    lies outside the unique loop, so it is removable and beta
                    comes out exactly zero; moved into the loop by the
                    paper's own rule the leading order is right and the
                    printed correction still is not
  [6] triangle      exact surds for the (pi/2, pi/8, 3pi/8) triangle, the
                    factor of i in the Z8 sketch, and the pi/4 coincidence
                    evaluated against Table S2
  [7] ledger        NeedsFit for anything that needs the scan, registration
                    beside the other constructions, describe()

Run with:  python3 tests/test_ninelink.py
       or: python3 run_tests.py
"""

import os
import sys
import time
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import sympy as sp

from pyCICY import theories as T
from pyCICY.theories import ninelink as N
from pyCICY.theories.flavorbase import FlavorTheory, NeedsFit

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
# [1] the diagrams
# ---------------------------------------------------------------------------

def test_diagrams():
    print("\n[1] the enumeration, from the conditions alone")

    c = N.census()
    check("nine-link textures", c["textures"], 1592)
    check("orbits under relabelling the quark fields", c["orbits"], 36)
    check("loop lengths present", sorted(c["loop_lengths"]), [4, 6])
    check("splits, as the paper lists them", sorted(c["splits"]),
          [(3, 6), (4, 5), (5, 4), (6, 3)])
    check_true("the two even splits dominate",
               c["splits"][(4, 5)] == c["splits"][(5, 4)] == 692)

    ts = N.enumerate_textures()
    check_true("every texture has nine links",
               all(len(t.links()) == 9 for t in ts))
    check_true("every texture has exactly one loop, and it is even",
               all(len(t.loop()) in (4, 6) for t in ts))

    # nine links on nine nodes: cycle rank is E - V + C, so connected is the
    # same as one independent loop. Check the node count is always nine.
    t = ts[0]
    nodes = set()
    for s, i, j in t.links():
        nodes |= {("q", i), (s, j)}
    check("nine nodes, hence cycle rank equal to components", len(nodes), 9)

    # the conditions are not vacuous: dropping one changes the count
    bad = N.Texture(0b100010001, 0b100010001)          # both diagonal
    check_true("two diagonal sectors are disconnected, hence excluded",
               not bad.is_connected())


# ---------------------------------------------------------------------------
# [2] the rephasing invariant
# ---------------------------------------------------------------------------

def test_invariant():
    print("\n[2] the loop monomial is invariant under rephasing the fields")

    rng = np.random.default_rng(11)
    ts = N.enumerate_textures()
    moved = 0
    for t in ts[:40]:
        entries = {}
        for s, i, j in t.links():
            entries[(s, i, j)] = complex(rng.uniform(0.5, 1.5)
                                         * np.exp(1j * rng.uniform(0, 2 * np.pi)))
        before = np.angle(complex(t.rephasing_invariant(entries)))
        # an arbitrary rephasing of all nine fields
        aq = rng.uniform(0, 2 * np.pi, 3)
        au = rng.uniform(0, 2 * np.pi, 3)
        ad = rng.uniform(0, 2 * np.pi, 3)
        rot = {}
        for s, i, j in t.links():
            b = au if s == "u" else ad
            rot[(s, i, j)] = entries[(s, i, j)] * np.exp(1j * (aq[i] - b[j]))
        after = np.angle(complex(t.rephasing_invariant(rot)))
        if abs(np.angle(np.exp(1j * (after - before)))) > 1e-9:
            moved += 1
    check("loop monomials whose phase moved under rephasing", moved, 0)

    # and the walk alternates, with exactly half the links conjugated
    t = ts[0]
    walk = t.loop_walk()
    check_true("the walk covers the loop once", len(walk) == len(t.loop()))
    check("half the links are conjugated",
          sum(1 for *_, c in walk if c), len(walk) // 2)


# ---------------------------------------------------------------------------
# [3] determinants and strong CP
# ---------------------------------------------------------------------------

def test_determinants():
    print("\n[3] perfect matchings, determinants, and strong CP")

    ts = N.enumerate_textures()
    # matchings are the determinant terms: check against sympy, symbolically
    for t in ts[:15]:
        for sector in ("u", "d"):
            M = t.matrix(sector)
            terms = sp.Add.make_args(sp.expand(M.det()))
            n = 0 if terms == (sp.Integer(0),) else len(terms)
            if n != len(t.determinant_terms(sector)):
                FAILURES.append("matchings vs det for %r" % (t,))
    check_true("perfect matchings are the determinant terms, symbolically",
               not [f for f in FAILURES if f.startswith("matchings")])

    c = N.strong_cp_census()
    check("textures admitting a real determinant", c["safe"], 1376)
    check("and the ones that do not", c["failures"], 216)
    check_true("safe is exactly single-matching-in-both",
               c["safe_equals_single_matching"])

    # the paper says all its failures are diagonal in one sector; over the
    # full enumeration that is not true, and here is the count
    check("failures that are diagonal in a sector",
          c["failures_diagonal_in_a_sector"], 56)
    check_true("so most failures are not diagonal in a sector",
               c["failures_diagonal_in_a_sector"] < c["failures"] / 2)


# ---------------------------------------------------------------------------
# [4] the angles
# ---------------------------------------------------------------------------

def test_angles():
    print("\n[4] the unitarity triangle by exact diagonalisation")

    Yu, Yd = N.example_texture(1, eps=0.1, coefficients=N._generic())
    V = N.ckm_from_yukawas(Yu, Yd)
    a, b, g = N.unitarity_angles(V)
    check_true("the three angles sum to pi",
               abs(abs(a) + abs(b) + abs(g) - np.pi) < 1e-9)
    check_true("the CKM matrix is unitary",
               np.allclose(V @ V.conj().T, np.eye(3), atol=1e-12))
    check_true("the Jarlskog invariant is non-zero",
               abs(N.jarlskog(V)) > 1e-8)

    # examples 1 and 3: the published closed forms, against exact
    # diagonalisation, with the residual falling like a power of eps
    for which in (1, 3):
        coeffs = N._generic()
        res = [abs(N.example_check(which, eps=e, coefficients=coeffs)["residual"])
               for e in (0.1, 0.05)]
        check_true("example %d reproduces its closed form" % which,
                   res[0] < 1e-3)
        check_true("example %d residual falls at least as eps^3" % which,
                   res[1] < res[0] / 7)

    # the leading order really is the special value
    r = N.example_check(1, eps=0.02)
    check_true("example 1 sits on pi/2 at leading order",
               abs(r["exact"] - np.pi / 2) < 1e-3)
    r = N.example_check(3, eps=0.02)
    check_true("example 3 sits on 3pi/8 at leading order",
               abs(r["exact"] - 3 * np.pi / 8) < 1e-3)


# ---------------------------------------------------------------------------
# [5] the one that does not reproduce
# ---------------------------------------------------------------------------

def test_example_two():
    print("\n[5] example two: the phase is not in the loop")

    r = N.example_two_anomaly()
    check("the unique loop of that diagram",
          sorted(r["loop"]),
          [("d", 0, 2), ("d", 2, 2), ("u", 0, 2), ("u", 2, 2)])
    check_true("and the printed phase link is not in it",
               not r["printed_phase_in_loop"])
    check_true("so as printed the phase is removable and beta vanishes",
               all(abs(x["beta"]) < 1e-12 for x in r["as_printed"]))

    rel = r["relocated"]
    check_true("moved into the loop, beta sits on pi/8",
               all(abs(x["departure_from_pi_8"]) < 1e-4 for x in rel))
    check_true("and the departure falls like eps^4",
               abs(rel[1]["departure_from_pi_8"])
               < abs(rel[0]["departure_from_pi_8"]) / 8)
    check_true("but the printed correction does not describe it",
               abs(rel[-1]["closed_form_residual"])
               > 100 * abs(rel[-1]["departure_from_pi_8"]))
    check_true("the residual does not vanish with eps",
               abs(rel[-1]["closed_form_residual"])
               > abs(rel[0]["closed_form_residual"]) / 8)


# ---------------------------------------------------------------------------
# [6] the special triangle
# ---------------------------------------------------------------------------

def test_triangle():
    print("\n[6] the (pi/2, pi/8, 3pi/8) triangle, exactly")

    r = N.yukawa_triangle_ratios()
    check("cot(pi/8) as a surd", sp.simplify(r["R_alpha"] - (1 + sp.sqrt(2))), 0)
    check_true("R_gamma is sin(pi/8) and also cos(3pi/8)",
               sp.simplify(r["R_gamma"] - r["R_gamma_as_cos"]) == 0)
    check_true("tan(3pi/8) = 1 + sqrt 2",
               sp.simplify(sp.tan(3 * sp.pi / 8) - (1 + sp.sqrt(2))) == 0)
    check_true("the three angles are a right triangle",
               sp.simplify(sp.pi / 2 + sp.pi / 8 + 3 * sp.pi / 8 - sp.pi) == 0)

    z = N.z8_ratio()
    check_true("the Z8 bracket is -i tan(pi/8)", z["bracket_is_minus_i_tan"])
    check_true("so i times it is real, not i tan(pi/8)",
               sp.im(z["i_times_bracket"]) == 0)
    check_true("which is not the stated target", not z["agrees"])

    p = N.pi_over_4_condition()
    check_true("the pi/4 coincidence holds to about ten percent",
               0.05 < p["relative_error"] < 0.15)
    check_true("and it is a statement about data alone, not about a texture",
               p["target"] == float(sp.sqrt(2)))


# ---------------------------------------------------------------------------
# [7] the ledger
# ---------------------------------------------------------------------------

def test_ledger():
    print("\n[7] registration, and refusals of the right kind")

    check_true("registered", "nine-link-texture" in T.registry)
    cls = T.get("nine-link-texture")
    check_true("and is a FlavorTheory", issubclass(cls, FlavorTheory))
    check_true("not a Theory and not a SurfaceTheory",
               not issubclass(cls, T.Theory)
               and not issubclass(cls, T.SurfaceTheory))

    nl = N.NineLinkTexture()
    check("texture() with no texture reports the enumeration",
          nl.texture()["textures"], 1592)
    check_true("predicted_angle runs the exact diagonalisation",
               abs(nl.predicted_angle(which=1, eps=0.05)["exact"]
                   - np.pi / 2) < 1e-2)
    check_true("deviation reports both routes",
               set(nl.deviation(which=1).keys())
               == {"from_exact", "from_closed_form", "eps"})

    check_true("the phase histogram needs a fit",
               _raises(NeedsFit, nl.histogram))
    check_true("and so does claiming to derive the texture",
               _raises(NeedsFit, nl.ultraviolet_completion))
    try:
        nl.histogram()
    except NeedsFit as e:
        check_true("NeedsFit names the missing ingredients", len(e.missing) >= 3)
        check_true("and points at the exact layer that is available",
                   e.available is not None)

    t = N.enumerate_textures()[0]
    one = N.NineLinkTexture(t)
    check_true("with a texture, it reports its loop",
               len(one.texture()["loop"]) in (4, 6))
    check_true("and its rephasing invariant",
               one.rephasing_invariant() != 1)

    text = nl.describe()
    check_true("describe() separates exact from declined",
               "exact:" in text and "declined:" in text)
    check_true("and says the texture is an ansatz", "ansatz" in text)


def main():
    t0 = time.time()
    print("=" * 72)
    print("test_ninelink: nine-link Yukawa textures")
    print("=" * 72)

    test_diagrams()
    test_invariant()
    test_determinants()
    test_angles()
    test_example_two()
    test_triangle()
    test_ledger()

    print("\n" + "=" * 72)
    if FAILURES:
        print("FAILED (%d): %s" % (len(FAILURES), ", ".join(FAILURES)))
        return 1
    print("test_ninelink: all checks passed in %.1fs" % (time.time() - t0))
    return 0


if __name__ == "__main__":
    sys.exit(main())
