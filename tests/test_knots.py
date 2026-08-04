"""
Tests for pyCICY.knots.

The module computes the Jones polynomial from a planar diagram, with no
SnapPy, Sage or spherogram anywhere in the path. The reason for it is the
observation of Wang and Zhang, "A remark on the counterexample to the
unknotting number conjecture", arXiv:2507.14265, that the two diagrams of
K15n81556 used by Brittenham and Hermiller, arXiv:2506.24088, represent a
chiral knot and its mirror image, detected by the Jones polynomial. That
computation is check [6] below.

Getting a Jones polynomial right is mostly a matter of conventions, so the
validation here is deliberately of two kinds. The convention-blind checks
cannot be satisfied by a consistently mirrored or misnormalised
implementation:

  [1] V(1) = 1, V(-1) = +-det, V(exp(2 pi i / 3)) = 1 and |V(i)| = 1 hold for
      every knot, and the determinants agree with the classical table;
  [2] mirroring is exactly t -> 1/t, and is an involution;
  [3] the Jones polynomial is multiplicative under connected sum.

The convention-fixing checks then pin the normalisation down:

  [4] the closure of the positive braid sigma_1^3 has V = -t^-4 + t^-3 + t^-1;
  [5] the braid constructor and the stored table agree for 3_1, 5_1, 7_1 and
      8_19, which are the torus knots T(2,3), T(2,5), T(2,7) and T(3,4);

and then:

  [6] K15n81556 is chiral, reproducing Wang and Zhang;
  [7] crossing changes behave (an involution, and one change unknots the
      trefoil), and the search over them is honest about being an upper bound
      in a fixed diagram.

Run with:  python3 tests/test_knots.py
       or: python3 run_tests.py  (runs every suite)
"""

import cmath
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import knots as K

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


ALL = sorted(K.KNOTS)

# Classical determinants, independent of chirality and of our normalisation.
DETERMINANTS = {
    "3_1": 3, "4_1": 5, "5_1": 5, "5_2": 7, "6_1": 9, "6_2": 11, "6_3": 13,
    "7_1": 7, "7_2": 11, "7_3": 13, "7_4": 15, "7_5": 17, "7_6": 19,
    "7_7": 21, "8_19": 3, "8_20": 9, "K15n81556": 39,
}

# ------------------------------------------------------------------- [1]

print("\n[1] convention-blind identities")
omega = cmath.exp(2j * cmath.pi / 3)
for nm in ALL:
    k = K.from_name(nm)
    v = k.jones()
    check_true("%s: single component" % nm, k.n_components() == 1)
    check_true("%s: V(1) = 1" % nm, abs(v.evaluate(1) - 1) < 1e-9)
    check("%s: determinant" % nm, k.determinant(), DETERMINANTS[nm])
    check_true("%s: |V(-1)| = det" % nm,
               abs(abs(v.evaluate(-1)) - DETERMINANTS[nm]) < 1e-6)
    check_true("%s: V(cube root of unity) = 1" % nm,
               abs(v.evaluate(omega) - 1) < 1e-9)
    check_true("%s: |V(i)| = 1" % nm, abs(abs(v.evaluate(1j)) - 1) < 1e-9)
check("unknot V", str(K.unknot().jones()), "1")
check("unknot determinant", K.unknot().determinant(), 1)

# ------------------------------------------------------------------- [2]

print("\n[2] mirroring is t -> 1/t and is an involution")
for nm in ALL:
    k = K.from_name(nm)
    check_true("%s: V(mirror) = V(1/t)" % nm,
               k.mirror().jones() == k.jones().invert_variable())
    check_true("%s: mirror is an involution" % nm,
               k.mirror().mirror().jones() == k.jones())
    check("%s: mirror flips the writhe" % nm, k.mirror().writhe(), -k.writhe())
    check_true("%s: mirror preserves the determinant" % nm,
               k.mirror().determinant() == k.determinant())
amphi = sorted(nm for nm in ALL if not K.from_name(nm).is_chiral())
check("not detected as chiral", amphi, ["4_1", "6_3"])

# ------------------------------------------------------------------- [3]

print("\n[3] Jones is multiplicative under connected sum")
PAIRS = [("3_1", "3_1"), ("3_1", "4_1"), ("4_1", "5_2"), ("5_1", "6_2"),
         ("7_1", "7_1"), ("6_3", "8_19")]
for a, b in PAIRS:
    ka, kb = K.from_name(a), K.from_name(b)
    s = ka.connected_sum(kb)
    check("%s # %s: one component" % (a, b), s.n_components(), 1)
    check("%s # %s: crossings add" % (a, b), len(s), len(ka) + len(kb))
    check_true("%s # %s: V multiplies" % (a, b),
               s.jones() == ka.jones() * kb.jones())
    check_true("%s # %s: determinant multiplies" % (a, b),
               s.determinant() == ka.determinant() * kb.determinant())
k = K.from_name("3_1")
check_true("connected sum with the unknot is trivial",
           k.connected_sum(K.unknot()).jones() == k.jones())
check_true("connected sum is commutative up to Jones",
           K.from_name("3_1").connected_sum(K.from_name("4_1")).jones()
           == K.from_name("4_1").connected_sum(K.from_name("3_1")).jones())

# ------------------------------------------------------------------- [4] [5]

print("\n[4] the normalisation is pinned by the positive trefoil")
check("closure of sigma_1^3", str(K.from_braid([1, 1, 1]).jones()),
      "- t^-4 + t^-3 + t^-1")
check("closure of sigma_1 is the unknot",
      str(K.from_braid([1]).jones()), "1")
check("empty braid is the unknot", str(K.from_braid([], 2).jones()), "1")

print("\n[5] braid closures agree with the stored table")
for (p, q), nm in (((2, 3), "3_1"), ((2, 5), "5_1"), ((2, 7), "7_1"),
                   ((3, 4), "8_19")):
    t = K.torus_knot(p, q)
    check_true("T(%d,%d) == %s" % (p, q, nm), t.jones() == K.from_name(nm).jones())
    check("T(%d,%d) crossing count" % (p, q), len(t), (p - 1) * q)
for nm in ALL:
    check_true("%s: table writhe is non-negative" % nm,
               K.from_name(nm).writhe() >= 0)
    check_true("%s: signs inferred agree with the diagram" % nm,
               K.infer_signs(K.KNOTS[nm]) == K.from_name(nm).signs)
try:
    K.torus_knot(2, 4)
    check_true("T(2,4) rejected as a link", False)
except ValueError:
    check_true("T(2,4) rejected as a link", True)

# ------------------------------------------------------------------- [6]

print("\n[6] Wang and Zhang: K15n81556 is chiral")
rep = K.chirality_report("K15n81556")
check("crossings", rep["crossings"], 15)
check("determinant", rep["determinant"], 39)
check_true("Jones is not palindromic", not rep["palindromic"])
check_true("chiral", rep["chiral"])
check_true("mirror Jones is the inverse-variable Jones", rep["mirror_is_inverse"])
check_true("K and its mirror have different Jones polynomials",
           rep["jones"] != rep["jones_mirror"])
check_true("but the same determinant",
           K.from_name("K15n81556").determinant()
           == K.from_name("K15n81556").mirror().determinant())

print("\n[6b] the Brittenham-Hermiller connected sum")
add = K.additivity_report()
check("7_1 # m7_1 crossings", add["sum_crossings"], 14)
check("7_1 # m7_1 is a knot", add["sum_components"], 1)
check_true("Jones multiplicative on the sum", add["jones_multiplicative"])
check_true("K # mK has palindromic Jones",
           add["jones_sum"].is_palindromic())
check("quoted u(7_1)", add["u_left_quoted"], 3)
check("naive sum of unknotting numbers", add["u_sum_naive"], 6)
check_true("the quoted bound is strictly smaller",
           add["u_sum_upper_bound_BH"] < add["u_sum_naive"])
check("u(7_1) matches the Milnor conjecture value (p-1)(q-1)/2",
      K.UNKNOTTING["7_1"], (2 - 1) * (7 - 1) // 2)
check("u(8_19) matches T(3,4)", K.UNKNOTTING["8_19"],
      (3 - 1) * (4 - 1) // 2)

# ------------------------------------------------------------------- [7]

print("\n[7] crossing changes")
t = K.from_name("3_1")
check_true("one crossing change unknots the trefoil",
           t.crossing_change(0).jones_is_trivial())
for i in range(len(t)):
    check_true("trefoil crossing %d: change is an involution" % i,
               t.crossing_change(i).crossing_change(i).pd == t.pd)
    check_true("trefoil crossing %d: change flips the sign" % i,
               t.crossing_change(i).signs[i] == -t.signs[i])
    check_true("trefoil crossing %d: still a knot" % i,
               t.crossing_change(i).n_components() == 1)
check_true("changing every crossing is mirroring",
           t.crossing_changes(range(len(t))).jones() == t.mirror().jones())

res = K.unknotting_search(K.from_name("3_1"), max_changes=1)
check("search unknots the trefoil in one change", res["changes"], 1)
res = K.unknotting_search(K.from_name("5_1"), max_changes=2)
check("search unknots 5_1 in two changes", res["changes"], 2)
res = K.unknotting_search(K.from_name("4_1"), max_changes=0)
check("search finds nothing with no changes allowed", res["found"], None)
check_true("search reports exhaustion honestly", res["exhausted"])
res = K.unknotting_search(K.from_name("7_1"), max_changes=1, limit=3)
check_true("search respects the limit", res["tried"] <= 3)

print("\n[8] rejection of bad input")
for bad, why in (([(1, 2, 3)], "short crossing"),
                 ([(1, 2, 3, 4)], "non-consecutive labels")):
    try:
        K.Knot(bad)
        check_true("rejects %s" % why, False)
    except ValueError:
        check_true("rejects %s" % why, True)
try:
    K.Knot([(0, 1, 2, 3)], signs=[1, 1])
    check_true("rejects a sign-count mismatch", False)
except ValueError:
    check_true("rejects a sign-count mismatch", True)
try:
    K.from_name("nonsense")
    check_true("rejects an unknown name", False)
except KeyError:
    check_true("rejects an unknown name", True)
try:
    K.from_braid([0])
    check_true("rejects generator zero", False)
except ValueError:
    check_true("rejects generator zero", True)

print("\n[9] the Laurent helper")
p = K.Laurent({0: 1, 2: -3})
q = K.Laurent({-1: 2})
check("product", str(p * q), "2t^-1 - 6t^1")
check("sum cancels", str(p + (-p)), "0")
check("variable inversion", str(p.invert_variable()), "- 3t^-2 + 1")
check_true("inversion is an involution",
           p.invert_variable().invert_variable() == p)
check_true("palindromic detection", K.Laurent({-1: 1, 1: 1}).is_palindromic())
check_true("non-palindromic detection",
           not K.Laurent({-1: 1, 1: 2}).is_palindromic())
check("evaluate", p.evaluate(2), 1 - 12)
check("degrees", p.degrees(), (0, 2))


print("\n" + "-" * 72)
if FAILURES:
    print("FAILED (%d):" % len(FAILURES))
    for f in FAILURES:
        print("  " + f)
    sys.exit(1)
print("test_knots: all checks passed")
