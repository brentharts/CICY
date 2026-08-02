"""
Tests for pyCICY.transitions.

Two references are checked against directly:

  CIPro  -- Anderson, Constantin, Gray, He, Lee, Lukas, arXiv:2606.27588,
            section 2.4 (normal forms and equivalence of configurations).
  Conifolds -- Anderson, Gray, Patil, Scanlon, arXiv:2512.18124, section 1.1
            and section 5.2 (P^1 splits, node counts, ineffective splits).

Section [4] is the strongest check in this file: the number of nodes N is
computed two completely independent ways -- by ambient intersection theory
with no cohomology at all, and from chi(X_R) = chi(X_D) + 2N using the full
Leray spectral sequence -- and the two must agree on every split.

Run with:  python3 tests/test_transitions.py
       or: python3 run_tests.py  (runs every suite)
"""

import os
import sys

# Prefer the source tree over any installed copy of pyCICY.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import transitions as T

FAILURES = []


def check(name, got, want):
    ok = got == want
    print("  {:<56} {:>12} {}".format(name, str(got), "ok" if ok else
                                      "FAIL want " + str(want)))
    if not ok:
        FAILURES.append(name)


def check_true(name, cond):
    print("  {:<56} {:>12} {}".format(name, str(bool(cond)),
                                      "ok" if cond else "FAIL"))
    if not cond:
        FAILURES.append(name)


# --------------------------------------------------------------------------
print("\n[1] Splits and contractions reproduce the published configurations")
# arXiv:2512.18124 eq. (1.11): the quintic splits to [[P^1|1 1],[P^4|1 4]]
check("split([[4,5]], 0, [1])", T.split([[4, 5]], 0, [1]),
      [[1, 1, 1], [4, 1, 4]])
check("contract back to the quintic", T.contract([[1, 1, 1], [4, 1, 4]]),
      [[4, 5]])
# arXiv:2512.18124 eqs. (5.11) -> (5.13)
check("split([[1,2],[3,4]], 0, [1,4])", T.split([[1, 2], [3, 4]], 0, [1, 4]),
      [[1, 1, 1], [1, 1, 1], [3, 4, 0]])
check("contract (5.13) back to (5.11)",
      T.contract([[1, 1, 1], [1, 1, 1], [3, 4, 0]]), [[1, 2], [3, 4]])

check_true("splits preserve the Calabi-Yau condition",
           all(T.is_calabi_yau(c) for _, _, c in T.splits([[4, 5]])))
check_true("splits preserve dim X = 3",
           all(T.dimensions(c)[2] == 3 for _, _, c in T.splits([[4, 5]])))

for bad, desc in [((0, [9]), "partition above the degree"),
                  ((0, [-1]), "negative partition"),
                  ((3, [1]), "column out of range"),
                  ((0, [1, 1]), "wrong partition length")]:
    try:
        T.split([[4, 5]], *bad)
        check_true("%s rejected" % desc, False)
    except ValueError:
        check_true("%s rejected" % desc, True)

try:
    T.contract([[4, 5]])
    check_true("contracting a non-split rejected", False)
except ValueError:
    check_true("contracting a non-split rejected", True)

# --------------------------------------------------------------------------
print("\n[2] Normal forms and equivalence (CIPro section 2.4)")
# The two dP2 configurations of CIPro differ by a relabelling.
dp2_a = [[2, 1, 1], [1, 1, 0], [1, 0, 1]]
dp2_b = [[1, 0, 1], [1, 1, 0], [2, 1, 1]]
check_true("dP2 forms share a normal form",
           T.normal_form(dp2_a)[0] == T.normal_form(dp2_b)[0])
check_true("dP2 forms are equivalent", T.equivalent(dp2_a, dp2_b))
check_true("quintic is not equivalent to its split",
           not T.equivalent([[4, 5]], [[1, 1, 1], [4, 1, 4]]))
check_true("different shapes are not equivalent",
           not T.equivalent([[4, 5]], [[2, 3], [2, 3]]))

# A configuration is always equivalent to a relabelling of itself.
base = [[1, 1, 1], [1, 2, 0], [3, 1, 3]]
shuffled = [base[2], base[0], base[1]]
check_true("row permutation is detected as equivalent",
           T.equivalent(base, shuffled))
col_swapped = [[r[0], r[2], r[1]] for r in base]
check_true("column permutation is detected as equivalent",
           T.equivalent(base, col_swapped))
check_true("normal form is idempotent",
           T.normal_form(base)[0] == T.normal_form(shuffled)[0])

# The (1,4) and (4,1) splits of the quintic differ only by naming the two
# new equations, so they must collapse to one class.
keys = {T.canonical_key(c) for _, _, c in T.splits([[4, 5]])}
check("distinct quintic splits up to relabelling", len(keys), 2)

# --------------------------------------------------------------------------
print("\n[3] Conifold transition data (arXiv:2512.18124 section 1.1)")
t = T.transition([[4, 5]], [[1, 1, 1], [4, 1, 4]])
check("deformation h^{1,1}", t["deformation"]["h11"], 1.0)
check("deformation h^{2,1}", t["deformation"]["h21"], 101.0)
check("resolution h^{1,1}", t["resolution"]["h11"], 2.0)
check("resolution h^{2,1}", t["resolution"]["h21"], 86.0)
check("nodes N (paper quotes 16)", t["nodes"], 16)
check("h^{1,1} shift (eq. 1.6)", t["h11_shift"], 1.0)
check("chi shift (eq. 1.7) == 2N", t["euler_shift"], 2 * t["nodes"])
check_true("effective", not t["ineffective"])
check_true("consistent", t["consistent"])

# --------------------------------------------------------------------------
print("\n[4] Node count: intersection theory vs cohomology, independently")
# nodes_expected uses only ambient intersection numbers; transition() uses
# the full Leray spectral sequence via chi. They must agree everywhere.
check("nodes_expected quintic (1,4) = 1*1*4*4",
      T.nodes_expected([[4, 5]], 0, [1]), 16)
check("nodes_expected quintic (2,3) = 2*2*3*3",
      T.nodes_expected([[4, 5]], 0, [2]), 36)

bases = [[[4, 5]], [[1, 2], [3, 4]], [[2, 3], [2, 3]],
         [[1, 2], [1, 2], [1, 2], [1, 2]], [[3, 4], [1, 2]], [[1, 1], [3, 4]]]
agree = 0
disagree = []
seen = set()
for b in bases:
    for col, part, c in T.splits(b):
        key = T.canonical_key(c)
        if key in seen:
            continue
        seen.add(key)
        n_geom = T.nodes_expected(b, col, part)
        n_chi = T.transition(b, c)["nodes"]
        if n_geom == n_chi:
            agree += 1
        else:
            disagree.append((b, part, n_geom, n_chi))
check("splits where both routes agree", agree, agree + len(disagree))
check("disagreements", len(disagree), 0)
check_true("a useful number of splits was checked", agree >= 20)

# --------------------------------------------------------------------------
print("\n[5] Ineffective splits (arXiv:2512.18124 section 5.2)")
# (5.11) and (5.13) are isomorphic geometries: the nodal locus is empty.
t2 = T.transition([[1, 2], [3, 4]], [[1, 1, 1], [1, 1, 1], [3, 4, 0]])
check("nodes N", t2["nodes"], 0)
check_true("flagged ineffective", t2["ineffective"])
check("h^{1,1} shift is zero", t2["h11_shift"], 0.0)
check("h^{2,1} shift is zero", t2["h21_shift"], 0.0)
check("chi shift is zero", t2["euler_shift"], 0)
check_true("consistent (an ineffective split must not add a Kahler class)",
           t2["consistent"])
check("nodes_expected agrees", T.nodes_expected([[1, 2], [3, 4]], 0, [1, 4]), 0)
# The paper notes the resolution side is a different, non-favourable
# description of the same geometry.
check_true("deformation is favourable", t2["deformation"]["favourable"])
check_true("resolution description is not favourable",
           not t2["resolution"]["favourable"])

# --------------------------------------------------------------------------
print("\n[6] Configuration validation")
info = T.check_configuration([[4, 5]])
check("quintic dim X", info["dim_X"], 3)
check("quintic dim ambient", info["dim_ambient"], 4)
check_true("quintic is Calabi-Yau", info["calabi_yau"])
check_true("quintic has no warnings", info["warnings"] == [])

info = T.check_configuration([[4, 4]])
check_true("non-CY degree is flagged", not info["calabi_yau"])
check_true("non-CY warning emitted", any("Calabi-Yau" in w
                                         for w in info["warnings"]))

info = T.check_configuration([[1, 1, 1], [4, 1, 4]])
check("split has a contractible row", info["contractible_rows"], [0])

for bad, desc in [([[0, 1]], "zero-dimensional factor"),
                  ([[4, -1]], "negative degree"),
                  ([[4]], "no degree column")]:
    try:
        T.check_configuration(bad)
        check_true("%s rejected" % desc, False)
    except ValueError:
        check_true("%s rejected" % desc, True)

# --------------------------------------------------------------------------
print("\n[7] Ambient intersection numbers")
# Note: chi = 2(h11 - h21) is always even for a Calabi-Yau threefold, so the
# odd-Euler-shift guard in transition() is unreachable for valid CY3 input.
# It is kept as a defensive check and deliberately not asserted here.
check("intersection number in P^4 of 1,1,4,4",
      T._ambient_intersection([4], [(1,), (1,), (4,), (4,)]), 16)
check("intersection number in P^4 of 2,2,3,3",
      T._ambient_intersection([4], [(2,), (2,), (3,), (3,)]), 36)
check("intersection number in P^1xP^3 of (1,0),(0,1)x2,(1,1)",
      T._ambient_intersection([1, 3], [(1, 0), (0, 1), (0, 1), (1, 1)]), 1)
# J_i^{n_i+1} = 0: two divisors both pulled back from the same P^1 factor
# cannot meet in a point of P^1 x P^1, so the product vanishes.
check("repeated class from one P^1 factor vanishes",
      T._ambient_intersection([1, 1], [(1, 0), (1, 0)]), 0)
check("P^1xP^1 of (1,0),(0,1) is one point",
      T._ambient_intersection([1, 1], [(1, 0), (0, 1)]), 1)
try:
    T._ambient_intersection([4], [(1,), (1,)])
    check_true("wrong number of classes rejected", False)
except ValueError:
    check_true("wrong number of classes rejected", True)

# --------------------------------------------------------------------------
print("\n[8] Second Chern character across a conifold transition")
# arXiv:2512.18124 eq. (1.24), with D_1, D_2 the ambient hyperplane classes
# of [[P^1|1 1],[P^4|1 4]]. Index 0 here is the P^1, index 1 the P^4.
check("ch2 of the quintic", T.class_str(T.chern_character_2([[4, 5]])),
      "-10 D0^2")
t = T.transition_ch2([[4, 5]], 0, [1])
check("re-embedded deformation (eq. 1.29)", t["deformation_conf"],
      [[1, 1, 0], [4, 0, 5]])
check("ch2(T X_D)", T.class_str(t["ch2_deformation"]), "-10 D1^2")
check("ch2(T X_R)", T.class_str(t["ch2_resolution"]), "-5 D0 D1 - 6 D1^2")
check("exceptional class [P^1_s]", T.class_str(t["exceptional"]),
      "-5 D0 D1 + 4 D1^2")
check("bridging curve [C_D]", T.class_str(t["c_deformation"]), "4 D1^2")
check("bridging curve [C_R]", T.class_str(t["c_resolution"]), "5 D0 D1")
check_true("[C_R] = [C_D] - [P^1_s]  (eq. 1.3)", t["bridging_matches"])

# The genuine test: the adjunction formula against pyCICY's separately
# implemented second Chern class, using ch_2 = -c_2 for a Calabi-Yau.
check_true("ch2 agrees with -c2, resolution",
           t["independent_check"]["resolution"]["agrees"])
check_true("ch2 agrees with -c2, deformation",
           t["independent_check"]["deformation"]["agrees"])

agree = disagree = 0
for conf in [[[4, 5]], [[2, 3], [2, 3]], [[1, 2], [1, 2], [1, 2], [1, 2]],
             [[1, 1, 1], [4, 1, 4]], [[1, 1, 1], [4, 2, 3]],
             [[3, 4], [1, 2]], [[2, 3], [1, 2], [1, 2]],
             [[1, 1, 1], [2, 1, 2], [2, 1, 2]]]:
    if T.chern_character_2(conf) == T.chern_character_2_from_c2(conf):
        agree += 1
    else:
        disagree += 1
check("configurations where ch2 == -c2", agree, agree + disagree)
check("disagreements", disagree, 0)

# The identity holds across many splits, but see the note below.
bad = 0
for base in [[[4, 5]], [[2, 3], [2, 3]], [[3, 4], [1, 2]]]:
    for col, part, _ in T.splits(base):
        if not T.transition_ch2(base, col, part)["bridging_matches"]:
            bad += 1
check("splits failing the bridging identity", bad, 0)

# NOTE: bridging_matches is an algebraic identity given the adjunction
# formula, so the check above cannot fail by construction. It is retained as
# a guard against a coding error in the class arithmetic or the re-embedding,
# not as independent evidence. The substantive check is ch2 == -c2 above.

# The exceptional class is a formal difference of curve classes and does NOT
# vanish for an ineffective split, even though the node count does. The class
# therefore cannot be used to detect ineffectiveness; the intersection number
# of nodes_expected can.
ineffective_seen = 0
for base in [[[3, 4], [1, 2]], [[1, 2], [3, 4]]]:
    for col, part, _ in T.splits(base):
        if T.nodes_expected(base, col, part) == 0:
            ineffective_seen += 1
            cls = T.transition_ch2(base, col, part)["exceptional"]
            check_true("N = 0 but [P^1_s] != 0 for %s %s"
                       % (base, list(part)), len(cls) > 0)
check_true("at least one ineffective split was examined", ineffective_seen > 0)

for bad_args, desc in [((9, [1]), "column out of range"),
                       ((0, [1, 1]), "wrong partition length")]:
    try:
        T.transition_ch2([[4, 5]], *bad_args)
        check_true("%s rejected" % desc, False)
    except ValueError:
        check_true("%s rejected" % desc, True)

# --------------------------------------------------------------------------
print("\n" + "=" * 72)
if FAILURES:
    print("FAILED ({}): {}".format(len(FAILURES), ", ".join(FAILURES)))
    sys.exit(1)
print("ALL TESTS PASSED on Python {}".format(sys.version.split()[0]))
