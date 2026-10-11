"""
Tests for pyCICY.sixsphere_links: what the six-sphere's finite layer shares
with the polytope, K3-lattice and central-charge modules.

  [1] B3 / dP6      V(B3) by triangulation, by Ehrhart counts, and K^2 by
                    Noether (12 - rays), by blow-up (9 - 3), and by the
                    toric self-intersections (all -1); Mahler data
  [2] Kodaira       the Euler count reproduces the paper's two existing
                    examples before it is used for (3,4,infinity)
  [3] root lattices E8 roots by reflection closure (240); E7 = alpha-perp
                    (126 roots, disc 2), E6 = A2-perp (72, disc 3), E6 + A1;
                    the same embeddings decided over Q by k3lattice
  [4] guard         centralcharge refuses a space with H^2 = 0
  [5] controls

Run with:  python3 tests/test_sixsphere_links.py   or   python3 run_tests.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import sixsphere_links as L

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


def test_b3():
    print("\n[1] the degree-six del Pezzo surface")
    f = L.b3_facts()
    check("V(B3), triangulation = Ehrhart", (f["volume"], f["volume_ehrhart"]),
          (6, 6))
    check("K^2 four ways: V, Noether, blow-up, boundary points",
          (f["volume"], f["K2_noether"], f["K2_blowup"], f["boundary_points"]),
          (6, 6, 6, 6))
    check("all six toric divisors are (-1)-curves", f["self_intersections"],
          [-1] * 6)
    check("sum of self-intersections = 12 - 3n", f["sum_self_intersections"],
          12 - 3 * f["rays"])
    check("V(B3) = V(dual) = 6, product 36", (f["dual_volume"], f["product"]),
          (6, 36))
    check_true("B3 is centrally symmetric: bound 32, 36 > 32, no equality",
               f["symmetric"] and f["bound_symmetric"] == 32
               and f["product"] > 32 and not f["equality"])


def test_kodaira():
    print("\n[2] Kodaira types from Euler numbers")
    # the paper's two existing examples must come out of the same code
    a = L.kodaira_facts(3, 6, 2)["allowed"]
    check_true("(3,6,inf), cusp I2: IV* + II + I2 (the paper's example)",
               ("IV*", "II", "I2", 12) in a)
    b = L.kodaira_facts(4, 6, 1)["allowed"]
    check_true("(4,6,inf), cusp I1: III* + II + I1 (the paper's example)",
               ("III*", "II", "I1", 12) in b)
    c = L.kodaira_facts(3, 4, 1)
    check("(3,4,inf), cusp I1: exactly one triple allowed",
          c["allowed"], [("IV*", "III", "I1", 12)])
    check("the four candidate Euler totals", sorted(t[3] for t in c["all"]),
          [8, 12, 14, 18])
    check("a K3 (total 24) is impossible for (3,4) and I1",
          [t for t in c["all"] if t[3] == 24], [])


def test_roots():
    print("\n[3] root lattices inside E8")
    f = L.root_lattice_facts()
    check("E8 has 240 roots", f["n_roots_E8"], 240)
    check("E7 = alpha-perp: 126 roots, rank 7, disc 2",
          (f["n_roots_E7"], f["rank_E7"], f["disc_E7"]), (126, 7, 2))
    check("E6 = A2-perp: 72 roots, rank 6, disc 3",
          (f["n_roots_E6"], f["rank_E6"], f["disc_E6"]), (72, 6, 3))
    check_true("A1 = <alpha> is orthogonal to E6", f["A1_orthogonal_to_E6"])
    check("E6 + A1: rank 7, disc 6",
          (f["rank_E6_plus_A1"], f["disc_E6_plus_A1"]), (7, 6))
    check("over Q: E7 and E6+A1 embed in E8 (k3lattice)",
          (f["Q_embeds_E7"], f["Q_embeds_E6A1"]), (True, True))
    check("Mordell-Weil rank 1 for the three fibre configurations",
          (f["mw_rank_IVstar_III_I1"], f["mw_rank_IVstar_II_I2"],
           f["mw_rank_IIIstar_II_I1"]), (1, 1, 1))


def test_guard():
    print("\n[4] the central-charge guard")
    g = L.central_charge_guard()
    check_true("centralcharge refuses H^2 = 0 (no class with w^3 > 0)",
               g["refused"] and "not all zero" in g["reason"])


def test_controls():
    print("\n[5] controls: each check made to fail on purpose")
    f = L.root_lattice_facts()
    check("E7 + <3> does not embed in E8 over Q (wrong discriminant)",
          f["Q_embeds_E7_plus_3"], False)
    d = L.kodaira_facts(3, 4, 5)["allowed"]
    check_true("a different cusp (I5) admits IV + III + I5, not IV* + III",
               ("IV", "III", "I5", 12) in d
               and ("IV*", "III", "I5", 12) not in d)
    check_true("no third rational type slips in for I1",
               len(L.kodaira_facts(3, 4, 1)["allowed"]) == 1)
    check_true("E7 has no extra roots: 126, not 120 or 132",
               f["n_roots_E7"] not in (120, 132))


def main():
    for section in (test_b3, test_kodaira, test_roots, test_guard,
                    test_controls):
        try:
            section()
        except Exception as exc:
            print("  {:<58} {:>12} FAIL".format(section.__name__ + " raised",
                                                type(exc).__name__))
            print("      " + str(exc)[:300])
            FAILURES.append("%s raised %s" % (section.__name__,
                                              type(exc).__name__))
    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("all sixsphere_links tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
