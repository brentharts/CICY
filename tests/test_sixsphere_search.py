"""
Tests for pyCICY.sixsphere_search (Part VII: which lattice data can build a
complex six-sphere).

  [1] survey        finite orders in SL2(Z); every hyperbolic (m1, m2, inf)
                    with m_j in {2,3,4,6} and its verdict; only (3, 4) left
  [2] gcd           |p| over all admissible twists, exhaustively mod m1 m2
  [3] action        the conjugation action derived symbolically; T2's data
                    reduced to residue classes
  [4] funnel        stage counts in a box, and stability as the box grows
  [5] classes       four classes, distinct by invariant, memberships
                    verified, exactly one reaching |p| = 1
  [6] paper         the explicit conjugator onto the paper's matrices
  [7] controls      the invariant is conjugation-invariant; a perturbed T1
                    is rejected; a non-coprime pair cannot reach |p| = 1;
                    the corner equation can fail to be integral

Connections to other theories:

  [8] monotile      the Laves kite lattice is the Hasse diagram of the cusp
                    fan, exactly in Q(sqrt3); the per-domain dictionary
  [9] chirality     the lattice data is achiral; V is not self-dual over Z
  [10] arithmetic   square classes against the Hat and the Spectre; the two
                    coincidences settled; Euclidean triangle groups; the
                    Hofstadter link
  [11] controls     at a = b the Laves sites are not the A2 centroids; the
                    sign of a square class matters; inverting only T1 breaks
                    the unipotent cusp; the stored constants recompute

Run with:  python3 tests/test_sixsphere_search.py   (about 30 s)
       or: python3 run_tests.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sympy import Matrix

from pyCICY import sixsphere as S
from pyCICY import sixsphere_search as X

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


def test_survey():
    print("\n[1] survey of triangle groups")
    fo = X.sl2_finite_orders()
    check("finite orders in SL2(Z)", fo["orders"], [1, 2, 3, 4, 6])
    check("the only involution", fo["order_2_elements"], [[[-1, 0], [0, -1]]])
    sv = X.triangle_group_survey()
    verdict = {r["m"]: r["verdict"] for r in sv["rows"]}
    check("hyperbolic pairs surveyed", len(sv["rows"]), 9)
    for m in ((2, 3), (2, 4), (2, 6), (4, 4)):
        check("%s" % (m,), verdict[m], "no unipotent cusp")
    for m in ((3, 3), (6, 6)):
        check("%s" % (m,), verdict[m], "cusp only I_n* (sign)")
    check("(3, 6)", verdict[(3, 6)], "|p| >= 3 (gcd)")
    check("(4, 6)", verdict[(4, 6)], "|p| >= 2 (gcd)")
    check("the only candidate", sv["candidates"], [(3, 4)])
    rows = {r["m"]: r for r in sv["rows"]}
    check("(3, 4) cusp: I_1", rows[(3, 4)]["cusp_n"], [1])
    check("(3, 6) cusp: I_2", rows[(3, 6)]["cusp_n"], [2])


def test_gcd():
    print("\n[2] the gcd obstruction")
    for m1, m2, g in ((3, 4, 1), (3, 6, 3), (4, 6, 2), (2, 3, 1)):
        o = X.gcd_obstruction(m1, m2)
        check("(%d,%d): minimal positive |p|" % (m1, m2),
              o["min_positive_|p|"], g)
        check_true("(%d,%d): every reachable |p| divisible by %d"
                   % (m1, m2, g), o["all_divisible_by_gcd"])
    check("(3,4): reachable residues", X.gcd_obstruction(3, 4)
          ["reachable_residues"], [1, 5])


def test_action():
    print("\n[3] the conjugation action")
    ca = X.conjugation_action()
    check_true("r' = r + x(M - I)", ca["dr = x(M - I)"])
    check_true("c' = c + (I - M)y", ca["dc = (I - M)y"])
    check_true("the middle block is unchanged", ca["M unchanged"])
    M2 = np.array([[0, -1], [1, 0]])
    check("row residue classes of M2 - I", len(X.residue_reps(M2, "row")),
          abs(round(np.linalg.det(M2 - np.eye(2)))))
    check("column residue classes of I - M2", len(X.residue_reps(M2, "col")),
          2)


def test_funnel():
    print("\n[4] the funnel")
    fn = X.funnel(5)
    c = fn["counts"]
    check("box", fn["box"], 5)
    check("M1 choices with M2 fixed", len(fn["M1_choices"]), 2)
    check("stage counts", [c[k] for k in ("tried", "orders", "cusp",
                                          "coinvariants_Z",
                                          "closure_ker_gamma",
                                          "unimodular_B0")],
          [87846, 49446, 272, 84, 84, 32])
    st = X.stability((3, 4))
    check("final count at boxes 3, 4, 5",
          [st[3]["unimodular_B0"], st[4]["unimodular_B0"],
           c["unimodular_B0"]], [32, 32, 32])
    check_true("while the cusp stage keeps growing",
               st[3]["cusp"] < st[4]["cusp"] < c["cusp"])
    check_true("the closure stage cut nothing (observed, not proved)",
               c["closure_ker_gamma"] == c["coinvariants_Z"])
    return fn


def test_classes(fn):
    print("\n[5] classification")
    cls = X.classify(fn["survivors"])
    check("classes", len(cls), 4)
    check("invariants", [sorted(c["invariant"]) for c in cls],
          [[0, 6], [1, 5], [2, 4], [3]])
    check("sizes", [len(c["members"]) for c in cls], [8, 8, 8, 8])
    check_true("pairwise distinct invariants (proof of distinctness)",
               len({c["invariant"] for c in cls}) == 4)
    check_true("every membership verified by its conjugator",
               all(c["verified"] for c in cls))
    check("classes reaching |p| = 1", sum(c["reaches_|p|=1"] for c in cls), 1)
    return cls


def test_paper(cls):
    print("\n[6] the paper's matrices")
    pc = X.paper_conjugator(cls)
    check_true("unique class reaching |p| = 1", pc["unique_class"])
    check("paper's invariant", pc["paper_invariant"], [1, 5])
    check_true("explicit conjugator verified", pc["verified"])
    check("its determinant", abs(pc["det"]), 1)
    check("the conjugator", pc["P"].tolist(),
          [[1, 3, -3, -2], [0, 1, 0, 0], [0, 0, 1, -1], [0, 0, 0, -1]])


def test_controls(cls):
    print("\n[7] controls")
    # the invariant is a conjugation invariant: conjugate a member by a
    # random unimodular matrix and recompute
    P = Matrix([[1, 2, 0, 1], [0, 1, 0, 0], [0, 3, 1, 0], [0, 0, 0, 1]])
    T1, T2 = cls[1]["rep"]
    T1c = np.array((P * Matrix(T1.tolist()) * P.inv()).tolist(),
                   dtype=np.int64)
    T2c = np.array((P * Matrix(T2.tolist()) * P.inv()).tolist(),
                   dtype=np.int64)
    check("invariant unchanged by an arbitrary conjugation",
          sorted(X.p_residues(X.admissible_ells(T1c, T2c))),
          sorted(cls[1]["invariant"]))
    bad = np.array(S.T1.tolist(), dtype=np.int64)
    bad[0, 2] = -5
    check_true("perturbed T1 (-5 for -6) has no finite order 3",
               X._order(bad) is None)
    check_true("(4, 6) cannot reach |p| = 1",
               1 not in X.gcd_obstruction(4, 6)["reachable_residues"])
    M1 = np.array([[-1, 1], [-1, 0]])
    # the corner of T^3 is 3 s + f; for r = (-1, 0), c = (-1, -1) f = 2,
    # so no integer s gives T^3 = I (found by search, not guessed)
    check("corner not integral for r = (-1, 0), c = (-1, -1)",
          X.solve_corner(np.array([-1, 0]), M1, np.array([-1, -1]), 3) is None,
          True)
    check("...while the paper's own T1 data is integral",
          X.solve_corner(np.array([0, -6]), M1, np.array([1, 1]), 3) is None,
          False)
    check_true("status_search passes", X.status_search()["all_pass"])


def test_monotile():
    print("\n[8] the monotile substrate and the cusp fibre")
    lv = X.laves_is_cusp_hasse()
    check("Laves patch sites", lv["sites"], 61)
    check_true("every site is an A2 cell of the matching kind",
               lv["sites_are_A2_cells_of_the_right_kind"])
    check("incidences = bonds", (lv["incidences"], lv["bonds"]), (102, 102))
    check_true("bonds are exactly the cover relations",
               lv["bonds_are_exactly_the_incidences"])
    for k, (laves, w) in X.laves_cusp_dictionary().items():
        check(k[:58], w, laves)


def test_chirality():
    print("\n[9] chirality and duality")
    m = X.lattice_mirror()
    check_true("(T1, T2) ~ (T1^-1, T2^-1), verified", m["verified"])
    check("mirror conjugator determinant", m["det"], 1)
    check("chirality record: fixed", m["record"]["fixed"], True)
    check("chirality record: domain", m["record"]["domain"],
          "complex six-sphere")
    d = X.dual_lattice()
    from sympy import symbols
    check_true("Q0 intertwines V and V* over Q", d["Q0_intertwines"])
    check("det Q0", d["det_Q0"], 36)
    check("integral intertwiners: rank", d["intertwiner_rank"], 2)
    check("their determinant form", d["det_form"], 36 * symbols("s") ** 4)
    check("so V is not self-dual over Z", d["self_dual_over_Z"], False)


def test_arithmetic():
    print("\n[10] arithmetic, coincidences, links")
    fc = X.field_comparison()
    check("S6 period classes (Q(zeta12))",
          fc["classes"]["S6 periods Q(sqrt-3, i)"], [-3, -1, 1, 3])
    check("shared with the monotile geometry", fc["periods & geometry"],
          [1, 3])
    check("shared with the Hat inflation", fc["periods & Hat"], [1])
    check("shared with the Spectre inflation", fc["periods & Spectre"], [1])
    check("S6 lattice and Spectre inflation", fc["lattice & Spectre"], [1, 6])
    check_true("lambda^2 = 4 + sqrt15", fc["lambda^2 = 4 + sqrt15"])
    co = X.coincidences()
    check_true("the sqrt6 is shared, with unrelated origins",
               co["sqrt6_shared"] and "-6" in co["sqrt6_origin_S6"])
    eu = X.euclidean_triangle_groups()
    check("Euclidean triangle groups", eu["euclidean"],
          [(2, 3, 6), (2, 4, 4), (3, 3, 3)])
    check("any containing both 3 and 4", eu["any_with_3_and_4"], False)
    hl = X.hofstadter_link()
    check_true("B3 hoppings are the hexagon rays", hl["equal_to_hexagon_rays"])
    check("the twelve theorem on the hexagon", hl["twelve"], (6, 6, 12))
    check_true("status_connections passes", X.status_connections()["all_pass"])


def test_controls_connections():
    print("\n[11] connection controls")
    from pyCICY import monotile as Mo
    from pyCICY.monotile import Quad
    P = Mo.laves_patch(a=Quad(1, 0, 3), b=Quad(1, 0, 3), rings=1)
    kinds, _, key = X._cells_A2_in_plane(4, Quad(0, 1, 3))
    miss = sum(1 for s_ in P["sites"] if key(s_) not in kinds)
    check_true("at a = b (the Spectre's proportions) the metric match fails",
               miss > 0)
    from pyCICY.theories.ninelink import square_classes
    check_true("signs matter: Q(sqrt-3, i) is not Q(sqrt3)",
               sorted(square_classes([-3, -1])) != sorted(square_classes([3])))
    half = S.T1.inv() * S.T2
    check_true("inverting only T1 breaks the unipotent cusp",
               (half.inv() - Matrix.eye(4)) ** 2 != Matrix.zeros(4, 4))
    for k, v in X.stored_constants_agree().items():
        check_true("stored constant recomputes: " + k, v)


def main():
    test_survey()
    test_gcd()
    test_action()
    fn = test_funnel()
    cls = test_classes(fn)
    test_paper(cls)
    test_controls(cls)
    test_monotile()
    test_chirality()
    test_arithmetic()
    test_controls_connections()
    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("all sixsphere_search tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
