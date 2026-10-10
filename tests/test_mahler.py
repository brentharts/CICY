"""
Tests for the Mahler volume product on reflexive polytopes
(pyCICY.polytope.volume_product and its helpers).

  [1] polygons    all sixteen: V(P) V(P*) against the bounds 27 and 32, with
                  V recovered independently from Pick's theorem and the old
                  toric.py (boundary points; the twelve theorem b + b* = 12)
  [2] extremal    simplices sit on (n+1)^(n+1) and cross-polytopes, cubes and
                  products of them on 4^n n!, n = 2..6; the quintic pair is
                  625 * 5; the hexagon and the 24-cell sit strictly above
  [3] two routes  the determinant volume against the Ehrhart h*-vector, whose
                  palindromy and sum are further checks; lattice independence
  [4] random      several hundred reflexive 3- and 4-polytopes, some symmetric:
                  the inequality holds on all of them, and every non-symmetric
                  equality case is a simplex
  [5] controls    each check made to fail on purpose

This is a check of a theorem, not a proof of it: no finite list establishes an
inequality. What it can do is make the exact statement concrete on the
package's own polytopes, and fail loudly on an arithmetic mistake.

Run with:  python3 tests/test_mahler.py
       or: python3 run_tests.py
"""

import itertools
import math
import os
import random
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import polytope as P
from pyCICY import toric

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


def poly(name):
    return np.array(toric.polygon(name), dtype=np.int64)


def product_polytope(A, B):
    """The product A x B (vertices are pairs of vertices)."""
    return np.array([list(a) + list(b) for a in A.tolist() for b in B.tolist()],
                    dtype=np.int64)


def test_polygons():
    print("\n[1] the sixteen reflexive polygons")
    names = list(toric.NAMED)
    check("there are sixteen", len(names), 16)
    rows = {}
    for nm in names:
        verts = toric.polygon(nm)
        r = P.volume_product(poly(nm), cross_check=True)
        # an independent route, from the older two-dimensional code: with one
        # interior lattice point Pick's theorem gives n! vol = 2 A = b
        b = len(toric.boundary_points(verts))
        bd = len(toric.boundary_points(toric.dual(verts)))
        rows[nm] = (r, b, bd)
    check_true("V(P) = boundary points (Pick), for every polygon",
               all(r["V"] == b for r, b, _ in rows.values()))
    check_true("V(P*) = boundary points of the dual (Pick)",
               all(r["V_dual"] == bd for r, _, bd in rows.values()))
    check_true("the twelve theorem: b + b* = 12 for every polygon",
               all(b + bd == 12 for _, b, bd in rows.values()))
    check_true("the inequality holds for all sixteen",
               all(r["holds"] for r, _, _ in rows.values()))
    check_true("symmetric flag agrees with toric.is_centrally_symmetric",
               all(r["symmetric"] == toric.is_centrally_symmetric(toric.polygon(n))
                   for n, (r, _, _) in rows.items()))
    check("centrally symmetric polygons",
          sorted(n for n, (r, _, _) in rows.items() if r["symmetric"]),
          ["B3", "F0", "Q8b"])
    check("product 27 (the simplex bound): P2 and its dual",
          sorted(n for n, (r, _, _) in rows.items() if r["product"] == 27),
          ["P2", "T9"])
    check("equality in the symmetric bound 32: F0 and its dual",
          sorted(n for n, (r, _, _) in rows.items() if r["equality"]
                 and r["symmetric"]), ["F0", "Q8b"])
    check("the general equality cases are exactly P2 and T9",
          sorted(n for n, (r, _, _) in rows.items()
                 if r["product"] == r["bound_general"]), ["P2", "T9"])
    check("T4 reaches 32 but is not symmetric: the bound used is 27",
          (rows["T4"][0]["product"], rows["T4"][0]["bound"],
           rows["T4"][0]["equality"]), (32, 27, False))
    check("the hexagon B3 is strictly above 32", rows["B3"][0]["product"], 36)
    check("minimum product over the sixteen",
          min(r["product"] for r, _, _ in rows.values()), 27)


def test_extremal():
    print("\n[2] extremal polytopes, n = 2..6")
    bad = []
    for n in range(2, 7):
        r = P.volume_product(P.simplex(n))
        if (r["V"], r["V_dual"], r["product"]) != (n + 1, (n + 1) ** n,
                                                   (n + 1) ** (n + 1)):
            bad.append(("simplex", n, r))
        if not r["equality"] or r["symmetric"] != False:
            bad.append(("simplex flags", n, r))
        c = P.volume_product(P.cross_polytope(n))
        if (c["V"], c["V_dual"], c["product"]) != (2 ** n, 2 ** n * math.factorial(n),
                                                   4 ** n * math.factorial(n)):
            bad.append(("cross", n, c))
        if not (c["equality"] and c["symmetric"]):
            bad.append(("cross flags", n, c))
    check("simplices and cross-polytopes on their bounds: mismatches", bad, [])
    q = P.volume_product(P.simplex(4))
    check("the quintic pair: V(P^4) = 5 and V(its dual) = 5^4",
          (q["V"], q["V_dual"], q["product"]), (5, 625, 3125))
    # Hanner polytopes: products and free sums of intervals and each other
    seg = P.cross_polytope(1)
    d2 = P.cross_polytope(2)
    prism = product_polytope(seg, d2)
    sq_x_d2 = product_polytope(P.cross_polytope(2), P.cross_polytope(2))
    sq3 = product_polytope(product_polytope(seg, seg), seg)
    for label, V, n in [("segment x diamond (3D)", prism, 3),
                        ("diamond x diamond (4D)", sq_x_d2, 4),
                        ("cube (3D, as a product)", sq3, 3)]:
        r = P.volume_product(V)
        check("%s: Hanner, so on 4^n n!" % label,
              (r["product"], r["equality"], r["symmetric"]),
              (4 ** n * math.factorial(n), True, True))
    # strictly above
    b3 = P.volume_product(poly("B3"))
    cell = P.volume_product(P.twenty_four_cell())
    check("the 24-cell: V = V* = 96, product 9216 > 6144",
          (cell["V"], cell["V_dual"], cell["product"], cell["equality"]),
          (96, 96, 9216, False))
    check_true("the 24-cell is above the symmetric bound, by a factor 3/2",
               cell["product"] * 2 == cell["bound"] * 3)
    check_true("the hexagon is above the symmetric bound",
               b3["product"] > b3["bound"])


def test_routes():
    print("\n[3] the determinant route against the Ehrhart route")
    cases = [("P2", poly("P2")), ("B3", poly("B3")), ("T9", poly("T9")),
             ("simplex(3)", P.simplex(3)), ("cross(3)", P.cross_polytope(3)),
             ("simplex(4)", P.simplex(4)), ("cross(4)", P.cross_polytope(4)),
             ("24-cell", P.twenty_four_cell())]
    bad, pal = [], []
    for label, V in cases:
        v_det = P.normalized_volume(V)
        v_ehr, hstar = P.ehrhart_normalized_volume(V)
        if v_det != v_ehr:
            bad.append(label)
        if hstar != hstar[::-1]:
            pal.append(label)
    check("determinant volume = Ehrhart h*(1): disagreements", bad, [])
    check("h* is palindromic for every reflexive case: failures", pal, [])
    check("24-cell h* vector", P.ehrhart_normalized_volume(P.twenty_four_cell())[1],
          [1, 20, 54, 20, 1])
    check("quintic polytope h* vector", P.ehrhart_normalized_volume(P.simplex(4))[1],
          [1, 1, 1, 1, 1])
    for label, V in cases:
        r = P.volume_product(V, cross_check=True)   # raises if the routes differ
    check_true("volume_product(cross_check=True) ran on all eight", True)
    # the product does not depend on the lattice
    a = P.volume_product(P.twenty_four_cell())
    b = P.volume_product(P.twenty_four_cell(lattice_basis=False))
    check("24-cell in D4 basis vs as D4 roots in Z^4: product",
          (a["product"], b["product"]), (9216, 9216))
    check("... and the second needed the generated lattice",
          (a["lattice"], b["lattice"]), ("ambient", "generated"))


def _random_reflexive(n, box, sizes, trials, seed, symmetric):
    rng = random.Random(seed)
    pts = [p for p in itertools.product(range(-box, box + 1), repeat=n) if any(p)]
    out = []
    for _ in range(trials):
        S = rng.sample(pts, rng.choice(sizes))
        if symmetric:
            S = S + [tuple(-x for x in p) for p in S]
        V = np.array(sorted(set(S)), dtype=np.int64)
        try:
            D = P.polar(V, exact=True)      # integral dual with 0 inside <=> reflexive
        except Exception:
            continue
        out.append(V)
        out.append(D)                       # the dual of a reflexive polytope is too
    return out


def test_random():
    print("\n[4] random reflexive polytopes")
    for n, box, sizes, trials, sym, cross, take in [
            (3, 1, range(4, 11), 4000, False, True, 60),
            (3, 1, range(2, 6), 4000, True, True, 60),
            (4, 1, range(5, 12), 3000, False, False, 60),
            (4, 1, range(3, 7), 3000, True, False, 60)]:
        found = _random_reflexive(n, box, sizes, trials, 11, sym)
        res = []
        for V in found[:take]:
            r = P.volume_product(V, cross_check=cross)
            r["nverts"] = len(P._hull(V).vertices)
            res.append(r)
        tag = "n=%d %s" % (n, "symmetric" if sym else "general")
        check_true("%s: enough samples (%d found)" % (tag, len(found)),
                   len(found) >= take)
        check_true("%s: the inequality holds on all %d" % (tag, len(res)),
                   all(r["holds"] for r in res))
        check_true("%s: the sampled polytopes have the intended symmetry"
                   % tag, all(r["symmetric"] == sym for r in res)
                   if sym else True)
        if not sym:
            eq = [r for r in res if r["equality"]]
            check_true("%s: every equality case is a simplex (%d found)"
                       % (tag, len(eq)),
                       all(r["nverts"] == n + 1 for r in eq))
        else:
            check_true("%s: Hanner-type equality cases occur (%d)"
                       % (tag, sum(r["equality"] for r in res)),
                       any(r["equality"] for r in res))


def test_controls():
    print("\n[5] controls: each check made to fail on purpose")
    real_nv = P.normalized_volume
    try:
        P.normalized_volume = lambda V: real_nv(V) - 1
        refused = _raises(ArithmeticError, P.volume_product, P.simplex(3),
                          cross_check=True)
    finally:
        P.normalized_volume = real_nv
    check_true("a volume off by one is caught by the Ehrhart route", refused)

    try:
        P.normalized_volume = lambda V: real_nv(V) // 2
        r = P.volume_product(P.simplex(3))
    finally:
        P.normalized_volume = real_nv
    check_true("halved volumes make `holds` False: the flag can fail",
               r["holds"] is False and r["product"] == 64)

    real_sym = P.is_centrally_symmetric
    try:
        P.is_centrally_symmetric = lambda V: True
        r = P.volume_product(poly("P2"))
    finally:
        P.is_centrally_symmetric = real_sym
    check_true("treating P2 as symmetric would wrongly break the bound",
               r["holds"] is False and r["product"] == 27 and r["bound"] == 32)

    q = P.volume_product(P.simplex(4))
    check_true("a wrong bound (n+1)^n is not met with equality by the simplex",
               q["product"] != 5 ** 4)
    check_true("the polygon T4 is not on the symmetric bound it numerically hits",
               P.volume_product(poly("T4"))["equality"] is False)

    bad_polygon = np.array([[-2, -1], [-1, -3], [2, 3], [1, 3], [0, 1]])
    check_true("a polygon with six interior points is refused",
               _raises(ValueError, P.volume_product, bad_polygon))
    check_true("a triangle with the origin on its boundary is refused",
               _raises(ValueError, P.normalized_volume,
                       np.array([[1, 0], [2, 0], [0, 1]])))
    check_true("non-integral vertices are refused, not rounded",
               _raises(ValueError, P.normalized_volume,
                       np.array([[0.5, 0], [-1, 0.5], [0, -1]])))
    check("the bounds themselves: n=2", (P.mahler_bound(2), P.mahler_bound(2, True)),
          (27, 32))
    check("the bounds themselves: n=4", (P.mahler_bound(4), P.mahler_bound(4, True)),
          (3125, 6144))


def main():
    for section in (test_polygons, test_extremal, test_routes, test_random,
                    test_controls):
        try:
            section()
        except Exception as exc:  # an exception is a failure, not an abort
            print("  {:<58} {:>12} FAIL".format(section.__name__ + " raised",
                                                type(exc).__name__))
            print("      " + str(exc)[:200])
            FAILURES.append("%s raised %s" % (section.__name__,
                                              type(exc).__name__))
    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("all mahler tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
