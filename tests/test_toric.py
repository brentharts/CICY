"""
Tests for pyCICY.toric.

The module tabulates the sixteen reflexive polygons, which are the toric
diagrams of the local Calabi-Yau geometries appearing in Sugimoto,
"Calabi-Yau geometry and electrons on 2d lattices", arXiv:1701.01561. A
table is only as good as its provenance, so the point of this suite is that
essentially nothing in it is taken on trust:

  [1] every named polygon really is reflexive;
  [2] the twelve theorem holds for all of them;
  [3] they are pairwise inequivalent under GL(2,Z);
  [4] a brute-force enumeration finds exactly these sixteen and no more;
  [5] the five smooth cases are *detected*, not tabulated, and are the
      expected del Pezzo surfaces with the expected degrees;
  [6] duality is an involution and the named dual pairs really are dual;
  [7] the hopping dictionary round-trips, and reproduces the square and
      triangular lattices for F_0 and B_3;
  [8] the anticanonical curves of P^2 and F_0 are Calabi-Yau one-folds
      according to the existing pyCICY machinery.

The enumeration in [4] takes about half a minute; pass --quick to skip it.

Run with:  python3 tests/test_toric.py
       or: python3 run_tests.py  (runs every suite)
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import toric as T
from pyCICY import transitions as Tr

FAILURES = []
QUICK = "--quick" in sys.argv


def check(name, got, want):
    ok = got == want
    print("  {:<58} {:>10} {}".format(name, str(got)[:10],
                                      "ok" if ok else "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<58} {:>10} {}".format(name, str(bool(cond)),
                                      "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


# ------------------------------------------------------------------ [1] [2] [3]

print("\n[1] every named polygon is reflexive")
check("count of named polygons", len(T.NAMED), 16)
for nm in sorted(T.NAMED):
    check_true("%s reflexive" % nm, T.is_reflexive(T.polygon(nm)))

print("\n[2] the twelve theorem")
for nm in sorted(T.NAMED):
    check("%s #bdry(P) + #bdry(P*)" % nm, T.twelve(T.polygon(nm))[2], 12)

print("\n[3] pairwise inequivalent under GL(2,Z)")
names = sorted(T.NAMED)
collisions = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
              if T.equivalent(T.polygon(a), T.polygon(b))]
check("colliding pairs", collisions, [])
for nm in ("P2", "B3", "T9"):
    check_true("%s equivalent to itself" % nm,
               T.equivalent(T.polygon(nm), T.polygon(nm)))

# --------------------------------------------------------------------- [4]

print("\n[4] brute-force enumeration agrees with the table")
if QUICK:
    print("  (skipped, --quick)")
else:
    rep = T.verify_named()
    check("enumerated", rep["n_enumerated"], 16)
    check("named", rep["n_named"], 16)
    check("missing from table", rep["missing_from_named"], [])
    check("duplicate names", rep["duplicate_names"], [])
    check_true("verify_named reports ok", rep["ok"])

# --------------------------------------------------------------------- [5]

print("\n[5] smoothness is detected, not tabulated")
smooth = sorted(nm for nm in T.NAMED if T.is_smooth(T.polygon(nm)))
check("smooth polygons", smooth, ["B3", "F0", "F1", "P2", "dP2"])
check("how many are smooth", len(smooth), 5)
for nm, deg in (("P2", 9), ("F0", 8), ("F1", 8), ("dP2", 7), ("B3", 6)):
    check("degree K^2 of %s" % nm, T.degree(T.polygon(nm)), deg)
for nm in sorted(T.NAMED):
    v = T.polygon(nm)
    check_true("%s: smooth iff #vertices == #boundary" % nm,
               T.is_smooth(v) == (len(T.convex_hull(v)) == len(T.boundary_points(v))))
    check_true("%s: degree == #bdry(P*)" % nm,
               T.degree(v) == T.twelve(v)[1])

# --------------------------------------------------------------------- [6]

print("\n[6] duality")
for nm in sorted(T.NAMED):
    v = T.polygon(nm)
    check_true("%s: P** equivalent to P" % nm, T.equivalent(T.dual(T.dual(v)), v))
check_true("T9 is dual to P2", T.equivalent(T.polygon("T9"), T.dual(T.polygon("P2"))))
check_true("Q8b is dual to F0", T.equivalent(T.polygon("Q8b"), T.dual(T.polygon("F0"))))
DUALS = {"P2": "T9", "F0": "Q8b", "F1": "Q8a", "dP2": "P7", "B3": "B3",
         "T4": "T8", "T6": "T6", "T8": "T4", "T9": "P2", "Q5": "Q7",
         "Q6": "Q6", "Q7": "Q5", "Q8a": "F1", "Q8b": "F0", "P6": "P6",
         "P7": "dP2"}
for nm in sorted(T.NAMED):
    check("dual of %s" % nm, T.dual_name(nm), DUALS[nm])
check("self-dual polygons",
      sorted(nm for nm in T.NAMED if T.dual_name(nm) == nm),
      ["B3", "P6", "Q6", "T6"])
check_true("duality is an involution on names",
           all(T.dual_name(T.dual_name(nm)) == nm for nm in T.NAMED))

print("\n[6b] normal form is idempotent and GL(2,Z) invariant")
for nm in sorted(T.NAMED):
    v = T.polygon(nm)
    nf = T.normal_form(v)
    check_true("%s: normal_form idempotent" % nm, T.normal_form(nf) == nf)
    check_true("%s: normal_form equivalent to input" % nm, T.equivalent(nf, v))
skew = [(x + y, y) for x, y in T.polygon("B3")]        # a GL(2,Z) image
check("normal form is a class function", T.normal_form(skew),
      T.normal_form(T.polygon("B3")))

# --------------------------------------------------------------------- [7]

print("\n[7] the hopping dictionary")
check("F0 hoppings are the square lattice", T.hoppings(T.polygon("F0")),
      [(-1, 0), (0, -1), (0, 1), (1, 0)])
check("B3 hoppings are the triangular lattice", T.hoppings(T.polygon("B3")),
      [(-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0)])
check("B3 has six nearest neighbours", len(T.hoppings(T.polygon("B3"))), 6)
check_true("origin excluded by default", (0, 0) not in T.hoppings(T.polygon("P2")))
check_true("origin included on request",
           (0, 0) in T.hoppings(T.polygon("P2"), include_origin=True))
for nm in sorted(T.NAMED):
    v = T.convex_hull(T.polygon(nm))
    check_true("%s: hoppings round-trip to the polygon" % nm,
               T.from_hoppings(T.hoppings(v)) == v)

print("\n[7b] bipartiteness is not central symmetry")
bip = sorted(nm for nm in T.NAMED if T.is_bipartite(T.polygon(nm)))
check("bipartite polygons", bip, ["F0", "T4"])
check_true("B3 centrally symmetric but not bipartite",
           T.is_centrally_symmetric(T.polygon("B3"))
           and not T.is_bipartite(T.polygon("B3")))
check_true("T4 bipartite but not centrally symmetric",
           T.is_bipartite(T.polygon("T4"))
           and not T.is_centrally_symmetric(T.polygon("T4")))
for nm in bip:
    f = T.bipartite_functional(T.polygon(nm))
    check_true("%s: functional is odd on every hop" % nm,
               all((f[0] * m + f[1] * n) % 2 == 1
                   for m, n in T.hoppings(T.polygon(nm))))

# --------------------------------------------------------------------- [8]

print("\n[8] the CICY bridge")
check("anticanonical curve of P^2", T.anticanonical_cicy("P2"), [[2, 3]])
check("anticanonical curve of F_0", T.anticanonical_cicy("F0"), [[1, 2], [1, 2]])
check("B3 has no product-of-projective-spaces ambient",
      T.anticanonical_cicy("B3"), None)
check("exactly two polygons carry a CICY",
      sum(1 for nm in T.NAMED if T.anticanonical_cicy(nm) is not None), 2)
for nm in ("P2", "F0"):
    cfg = T.anticanonical_cicy(nm)
    check_true("%s: anticanonical config is Calabi-Yau" % nm,
               Tr.is_calabi_yau(cfg))
    check("%s: anticanonical config has dimension 1" % nm,
          Tr.dimensions(cfg)[2], 1)

print("\n[9] rejection of bad input")
check_true("non-reflexive polygon rejected",
           not T.is_reflexive([(2, 0), (0, 2), (-2, -2)]))
check_true("polygon with no interior point rejected",
           not T.is_reflexive([(0, 0), (1, 0), (0, 1)]))
try:
    T.polygon("nonsense")
    check_true("unknown name raises", False)
except KeyError:
    check_true("unknown name raises", True)
check("alias P112 resolves to T4", T.polygon("P112"), T.polygon("T4"))
check("alias dP3 resolves to B3", T.polygon("dP3"), T.polygon("B3"))


print("\n" + "-" * 72)
if FAILURES:
    print("FAILED (%d):" % len(FAILURES))
    for f in FAILURES:
        print("  " + f)
    sys.exit(1)
print("test_toric: all checks passed")
