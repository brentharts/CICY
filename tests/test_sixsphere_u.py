"""
Tests for pyCICY.sixsphere_u (Part XIV-c: u-variables from F-polynomials, the
punctured disk, and the torus).

  [1] finite type   A2, A3: the tropical regression gives a unique integer
                    solution, the u-equations hold exactly, C = 0 exactly on
                    compatible pairs
  [2] hexagon       A3 reproduces the disk module's u-variables
  [3] puncture      D4 (16 variables, 50 clusters): tagged arcs; the plain arc
                    P_i flips its tag and moves to i+1 in u
  [4] torus         the same ansatz has no solution for the initial arcs when
                    nu = 0 off the g-fan, and is not unique on the fan
  [5] controls      the wrong sign of g breaks finite type

Run with:  python3 tests/test_sixsphere_u.py   or   python3 run_tests.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import sixsphere_u as U

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


def test_finite():
    print("\n[1] finite type")
    for name, B, nv, nc, cv in (("A2", U.B_A2, 5, 5, [0, 1]), ("A3", U.B_A3, 9, 14, [0, 1])):
        r = U.finite_type_report(B)
        check(name + " variables, clusters", (r["variables"], r["clusters"]), (nv, nc))
        check_true(name + " solution unique (rank = unknowns)", r["unique"])
        check_true(name + " every variable solved with integers", r["all_solved"])
        check_true(name + " exponents of F are +-1, at most 4 F factors",
                   r["exponents_pm1"] and r["max_F_factors"] <= 4)
        check_true(name + " u-equations exact at 12 rational points", r["u_equations_exact"])
        check_true(name + " C symmetric", r["C_symmetric"])
        check_true(name + " C = 0 exactly on compatible pairs", r["C_zero_iff_compatible"])
        check(name + " values of C", r["C_values"], cv)


def test_hexagon():
    print("\n[2] the hexagon of the disk module")
    h = U.hexagon_match()
    check_true("u-values equal the disk module's as multisets", h["match"])
    check("variables", h["variables"], 9)


def test_puncture():
    print("\n[3] the punctured square, type D4")
    r = U.punctured_square()
    rep = r["report"]
    check("variables, clusters", (rep["variables"], rep["clusters"]), (16, 50))
    check_true("unique, integral, u-equations exact", rep["unique"] and rep["all_solved"] and rep["u_equations_exact"])
    check("values of C (compatibility degrees)", rep["C_values"], [0, 1, 2])
    check_true("C = 0 exactly on compatible pairs", rep["C_zero_iff_compatible"])
    check("plain, notched, chords", (r["plain"], r["notched"], r["chords"]), (4, 4, 8))
    check("chord shapes", r["chord_shapes"], [1, 1, 1, 1, 2, 2, 2, 2])
    check_true("u(P_i) = F(chord i+2,i+3) / F(N_{i+1})", r["plain_flips_tag_and_shifts"])
    check_true("u(N_j) = y_{j+1} F(chord j+2,j+3) / F(N_j)", r["notched_pattern"])


def test_torus():
    print("\n[4] the once-punctured torus")
    for d, nv, unsolved, unk in ((3, 24, 18, 24), (4, 48, 21, 48)):
        t = U.torus_ansatz(depth=d, grid=3)
        check("depth %d variables" % d, t["variables"], nv)
        check_true("depth %d: initial arcs have no solution off the fan" % d, t["initial_unsolved"])
        check("depth %d: variables without a solution off the fan" % d, t["off_fan_unsolved"], unsolved)
        check_true("depth %d: fit on the fan alone is not unique" % d, t["fan_only_rank"] < t["fan_only_unknowns"])


def test_controls():
    print("\n[5] control")
    sol = U.solve_u(U.B_A2, grid=2, sign=+1)
    check_true("g instead of -g: some variable of A2 has no solution",
               any(v is None for v in sol["E"].values()))


def main():
    for t in (test_finite, test_hexagon, test_puncture, test_torus, test_controls):
        t()
    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("all sixsphere_u tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
