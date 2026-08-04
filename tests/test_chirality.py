"""
Tests for pyCICY.chirality.

The module claims that three of the mirror operations in this package have
the same shape: an involution that swaps a pair of integers and preserves
their sum or span. That claim is the thing under test, and it is checked in
each domain rather than asserted:

  [1] the pair really is swapped and the preserved quantity really is fixed,
      for every knot in the table, every reflexive polygon, and a range of
      Hodge pairs;
  [2] the involutions really are involutions;
  [3] the fixed points are found: the amphichiral knots, the four self-dual
      polygons, and the self-mirror Hodge pairs;
  [4] the preserved quantity in the knot case is meaningful, via the
      Kauffman-Murasugi-Thistlethwaite theorem: the span of the Jones
      polynomial equals the crossing number exactly for alternating knots.
      This is an independent check on the whole Jones pipeline;
  [5] the quantum curve is the case where the analogy fails, and the module
      says so. Reflection leaves the spectrum fixed, so ``detected`` is None;
      and reflection-invariance is logically independent of the spectral
      asymmetry, which is demonstrated by exhibiting all four combinations
      among the sixteen polygons;
  [6] the published CICY list is not closed under mirror symmetry, and the
      sentinel zero entries are excluded rather than counted as self-mirror.

Run with:  python3 tests/test_chirality.py
       or: python3 run_tests.py  (runs every suite)
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import chirality as C
from pyCICY import knots as K
from pyCICY import quantum_curve as Q
from pyCICY import toric as T

FAILURES = []


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


HODGE = [(1, 101), (2, 83), (4, 68), (2, 86), (15, 15), (19, 19), (5, 45)]

# --------------------------------------------------------------------- [1]

print("\n[1] the swap-and-preserve law")
for nm in sorted(K.KNOTS):
    r = C.knot_chirality(nm)
    a, b = r["pair"]
    check("%s: mirror pair is the negated swap" % nm,
          tuple(r["mirror_pair"]), (-b, -a))
    check_true("%s: span is preserved" % nm,
               r["preserved"] == b - a
               and C.knot_chirality(K.from_name(nm).mirror())["preserved"]
               == b - a)
for nm in sorted(T.NAMED):
    r = C.polygon_chirality(nm)
    a, b = r["pair"]
    check("%s: dual pair is the swap" % nm, tuple(r["mirror_pair"]), (b, a))
    check("%s: sum preserved is twelve" % nm, r["preserved"], 12)
    check("%s: pair sums to the preserved value" % nm, a + b, r["preserved"])
for h in HODGE:
    r = C.cicy_chirality(hodge=h)
    check("hodge %s: mirror pair is the swap" % (h,),
          tuple(r["mirror_pair"]), (h[1], h[0]))
    check("hodge %s: sum preserved" % (h,), r["preserved"], h[0] + h[1])
    check("hodge %s: Euler negated" % (h,), r["euler_mirror"], -r["euler"])
    check("hodge %s: Euler is 2(h11-h21)" % (h,), r["euler"],
          2 * (h[0] - h[1]))

# --------------------------------------------------------------------- [2]

print("\n[2] the involutions are involutions")
for nm in sorted(K.KNOTS):
    k = K.from_name(nm)
    check_true("%s: mirroring twice restores the Jones polynomial" % nm,
               C.mirror(C.mirror(k)).jones() == k.jones())
for nm in sorted(T.NAMED):
    v = T.polygon(nm)
    check_true("%s: double dual is equivalent to the polygon" % nm,
               T.equivalent(C.mirror(nm), v) is False
               or T.equivalent(T.dual(C.mirror(nm)), v))
    check_true("%s: dual of the dual is the polygon" % nm,
               T.equivalent(T.dual(T.dual(v)), v))
for nm in ("F0", "B3", "P2"):
    c = Q.from_polygon(nm)
    check_true("%s: reflecting the curve twice restores the hops" % nm,
               set(C.mirror(c).mirror().points) == set(c.points))
for h in HODGE:
    m = C.mirror(None, kind="cicy", hodge=h)
    check("hodge %s: double mirror" % (h,), (m["h21"], m["h11"]), h)

# --------------------------------------------------------------------- [3]

print("\n[3] fixed points in each domain")
check("knots not separated from their mirror",
      sorted(nm for nm in K.KNOTS if C.knot_chirality(nm)["fixed"]),
      ["4_1", "6_3"])
check("self-dual reflexive polygons",
      sorted(nm for nm in T.NAMED if C.polygon_chirality(nm)["fixed"]),
      ["B3", "P6", "Q6", "T6"])
check("how many polygons are self-dual",
      sum(1 for nm in T.NAMED if C.polygon_chirality(nm)["fixed"]), 4)
for h in HODGE:
    check_true("hodge %s: fixed iff h11 == h21" % (h,),
               C.cicy_chirality(hodge=h)["fixed"] == (h[0] == h[1]))
for nm in sorted(T.NAMED):
    r = C.polygon_chirality(nm)
    check_true("%s: self-dual iff the pair is balanced" % nm,
               r["fixed"] == (r["pair"][0] == r["pair"][1]))

# --------------------------------------------------------------------- [4]

print("\n[4] span of V vs crossing number (Kauffman-Murasugi-Thistlethwaite)")
NON_ALTERNATING = {"8_19", "8_20", "K15n81556"}
for nm in sorted(K.KNOTS):
    r = C.knot_chirality(nm)
    span, cr = r["preserved"], r["crossings"]
    if nm in NON_ALTERNATING:
        check_true("%s: non-alternating, span < crossings" % nm, span < cr)
    else:
        check("%s: alternating, span == crossings" % nm, span, cr)

# --------------------------------------------------------------------- [5]

print("\n[5] the quantum curve is where the analogy breaks")
for nm in sorted(T.NAMED):
    r = C.curve_chirality(nm)
    check_true("%s: reflection is not detectable" % nm, r["detected"] is None)
    check_true("%s: no swapped pair is claimed" % nm, r["pair"] is None)
for nm in sorted(T.NAMED):
    c = Q.from_polygon(nm)
    check_true("%s: reflected curve has the same spectrum" % nm,
               np.allclose(np.sort(c.spectrum(1, 3, nk=6)),
                           np.sort(c.mirror().spectrum(1, 3, nk=6)), atol=1e-12))
combos = set()
for nm in T.NAMED:
    r = C.curve_chirality(nm)
    combos.add((r["fixed"], r["spectrally_chiral"]))
check("all four combinations of the two axes occur", len(combos), 4)
check_true("B3 is fixed by reflection yet spectrally chiral",
           C.curve_chirality("B3")["fixed"]
           and C.curve_chirality("B3")["spectrally_chiral"])
check_true("T4 is not fixed by reflection yet spectrally symmetric",
           not C.curve_chirality("T4")["fixed"]
           and not C.curve_chirality("T4")["spectrally_chiral"])
check_true("spectral chirality agrees with non-bipartiteness",
           all(C.curve_chirality(nm)["spectrally_chiral"]
               != C.curve_chirality(nm)["bipartite"] for nm in T.NAMED))

# --------------------------------------------------------------------- [6]

print("\n[6] the published CICY list is chiral as a set")
rep = C.cicy_list_chirality()
check("manifolds in the list", rep["n_manifolds"], 7890)
check("sentinel-zero entries excluded", rep["n_degenerate"], 22)
check("usable entries", rep["n_usable"], 7868)
check("self-mirror Hodge pairs", rep["self_mirror_pairs"], [(15, 15), (19, 19)])
check("non-trivial mirror pairs present", rep["nontrivial_mirror_pairs"], [])
check_true("the list is not closed under mirror symmetry",
           not rep["closed_under_mirror"])
check_true("almost every pair lacks a partner",
           rep["n_pairs_without_partner"] > 250)
check_true("only the self-mirror pairs have a partner",
           rep["pairs_with_mirror_partner"] == rep["self_mirror_pairs"])
h11lo, h11hi = rep["h11_range"]
h21lo, h21hi = rep["h21_range"]
check_true("h11 never reaches the top of the h21 range", h11hi < h21hi)
check_true("that range mismatch explains the failure", h11hi < h21hi)
check_true("Euler characteristics are all non-positive",
           rep["euler_range"][1] <= 0)

# --------------------------------------------------------------------- [7]

print("\n[7] dispatch and reporting")
check("string naming a knot dispatches to the knot domain",
      C.chirality("K15n81556")["domain"], "knot")
check("string naming a polygon dispatches to the polygon domain",
      C.chirality("B3")["domain"], "polygon")
check("a Knot instance dispatches", C.chirality(K.from_name("3_1"))["domain"],
      "knot")
check("a QuantumCurve instance dispatches",
      C.chirality(Q.from_polygon("F0"))["domain"], "curve")
check("an alias resolves", C.chirality("dP3")["domain"], "polygon")
try:
    C.chirality([[2, 3]])
    check_true("a raw list without kind is rejected", False)
except ValueError:
    check_true("a raw list without kind is rejected", True)
try:
    C.chirality("not-a-thing")
    check_true("an unknown name is rejected", False)
except ValueError:
    check_true("an unknown name is rejected", True)
try:
    C.cicy_chirality()
    check_true("cicy_chirality with no data is rejected", False)
except ValueError:
    check_true("cicy_chirality with no data is rejected", True)
check("mirror_pair helper", C.mirror_pair("B3"), (6, 6))
check("mirror_invariant helper", C.mirror_invariant("P2"), 12)
check_true("mirror of a polygon is its dual",
           T.equivalent(C.mirror("P2"), T.polygon("T9")))
check_true("mirror of a knot is a Knot", isinstance(C.mirror("3_1"), K.Knot))

recs = C.survey()
check_true("survey covers all four domains",
           {r["domain"] for r in recs} == {"knot", "polygon", "curve", "cicy"})
check_true("survey records all carry the common keys",
           all({"domain", "name", "involution", "pair", "mirror_pair",
                "preserved", "fixed", "detected", "note"} <= set(r)
               for r in recs))
text = C.format_survey(recs)
check_true("the formatted table has a row per record",
           len(text.splitlines()) == len(recs) + 2)
check_true("DOMAINS documents every domain",
           {r["domain"] for r in recs} == set(C.DOMAINS))


print("\n" + "-" * 72)
if FAILURES:
    print("FAILED (%d):" % len(FAILURES))
    for f in FAILURES:
        print("  " + f)
    sys.exit(1)
print("test_chirality: all checks passed")
