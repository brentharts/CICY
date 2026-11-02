"""
Tests for pyCICY.sixsphere_surface (Part XIV: the cusp fibre and curves on
surfaces).

  [1] quiver     the FST quiver of Delta(W), read off the A_2 triangles, is the
                 Markov quiver (6 arrows, |b_ij| = 2)
  [2] mutation   Markov constant preserved, every cluster variable a Laurent
                 polynomial with positive coefficients, Markov triples
  [3] fans       6 cones through a ray of the Mumford fan; clusters through one
                 variable form a line (2d+1 within d mutations)
  [4] Lucas      LS_n = (2-D)Y^n - power sum of the roots of a^2 - X a + XY, and
                 tr of the powers of the T1, T2, T1T2 blocks = L_n(-1, 0, 2)
  [5] Hat        tr(M^n) = L_4n + 1 + (-1)^n, char poly (x-1)(x+1)(x^2-7x+1)
  [6] loops      the dual fatgraph of Delta(W): genus 1, one face, two loops,
                 no external legs (two routes; planar control)
  [7] theta      U of the theta graph by two routes, Newton polytope a triangle
  [8] principal  F-polynomials, g-vectors and slopes of every cluster variable
                 (separation formula, unimodular g-bases, Farey compatibility)

Run with:  python3 tests/test_sixsphere_surface.py   or   python3 run_tests.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import sympy as sp

from pyCICY import sixsphere_surface as S

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


def test_quiver():
    print("\n[1] the quiver of Delta(W)")
    check("side cycles", S.triangle_side_cycles(),
          {"lower": ("a", "b", "c"), "upper": ("c", "a", "b")})
    B = S.dual_quiver()
    check("exchange matrix", B.tolist(), [[0, 2, -2], [-2, 0, 2], [2, -2, 0]])
    check_true("Markov quiver", S.is_markov(B))
    cyc = S.triangle_side_cycles()
    flipped = S.quiver_from_cycles([cyc["lower"], cyc["upper"][::-1]])
    check_true("control: reversing one triangle's orientation is not Markov",
               not S.is_markov(flipped))


def test_mutation():
    print("\n[2] mutation, Laurentness, Markov triples")
    m = S.markov_checks(depth=4)
    check("arrows", m["arrows"], 6)
    check_true("Markov constant preserved on every seed", m["constant_preserved"])
    check_true("every cluster variable Laurent with positive coefficients",
               m["laurent_positive"])
    check_true("x^2+y^2+z^2 = 3xyz on every triple", m["markov_triples_ok"])
    check("first triples", m["markov_triples"][:4],
          [(1, 1, 1), (1, 1, 2), (1, 2, 5), (1, 5, 13)])
    # control: a wrong exchange matrix must break the Markov constant
    bad = sp.Matrix([[0, 1, -1], [-1, 0, 1], [1, -1, 0]])
    s = S.mutate(tuple(S.X0), 0, bad)
    check_true("control: single arrows do not preserve the constant",
               sp.cancel(S.markov_constant(s) - S.markov_constant(tuple(S.X0))) != 0)


def test_fans():
    print("\n[3] hexagon versus line")
    check("maximal cones through a ray of the Mumford fan", S.fan_link_size(), 6)
    for d in (3, 4, 5):
        c = S.cluster_link(depth=d)
        check("clusters through x0 within %d mutations" % d, c["through_x0"], 2 * d + 1)


def test_lucas():
    print("\n[4] Lucas as traces of powers")
    L = S.lucas_trace_checks(nmax=12)
    check_true("LS_n = (2-D)Y^n - (a+^n + a-^n), n = 2..12", L["ngon_power_sum"])
    check_true("middle blocks of T1, T2", L["blocks"])
    check_true("block product is [[1,1],[0,1]]", L["block_product"])
    for k, v in L["traces"].items():
        check_true("tr(%s^n) = L_n(x), n <= 12" % k, v)
    A = sp.Matrix([[-1, 1], [-1, 0]])
    check_true("control: L_n(+1) (period 6) is not tr of the order-3 block's powers",
               [(A ** n).trace() for n in range(1, 13)]
               != [S.G.lucas_polynomial(n, sp.Integer(1)) for n in range(1, 13)])


def test_hat():
    print("\n[5] the Hat inflation is the same identity at x = 7")
    h = S.hat_traces(nmax=8)
    check_true("tr(M^n) = L_4n + 1 + (-1)^n, n <= 8", h["trace_is_L4n"])
    check("L_4", h["L4"], 7)
    x = sp.Symbol("x")
    check_true("char poly (x-1)(x+1)(x^2-7x+1)",
               sp.expand(h["charpoly"] - (x - 1) * (x + 1) * (x ** 2 - 7 * x + 1)) == 0)


def test_loops():
    print("\n[6] loop order of the dual fatgraph")
    r = S.loop_order()
    check("vertices, edges", (r["V"], r["E"]), (2, 3))
    check("boundary cycles (faces)", r["F"], 1)
    check("genus", r["genus"], 1)
    check("loops from the graph E - V + 1", r["loops_graph"], 2)
    check("loops from the surface 2g + F - 1", r["loops_surface"], 2)
    check("external legs", r["external_legs"], 0)
    check("control: opposite cyclic order at Q gives 3 faces", r["control_faces"], 3)
    check("control: genus 0 (planar theta)", r["control_genus"], 0)


def test_theta():
    print("\n[7] the theta graph")
    th = S.theta_symanzik()
    check_true("U: matrix-tree = spanning trees", th["routes_agree"])
    check_true("U = ab + bc + ca", th["equals"])
    check("spanning trees", th["trees"], 3)
    check("Newton polytope vertices", th["newton"], [(0, 1, 1), (1, 0, 1), (1, 1, 0)])
    check("loops", th["loops"], 2)


def test_principal():
    print("\n[8] F-polynomials, g-vectors, slopes")
    r = S.principal_checks(depth=4)
    for k in ("slopes_consistent", "slopes_injective", "g_homogeneous",
              "F_constant_term_one", "F_positive", "F_max_monomial",
              "separation_formula", "g_bases_unimodular", "compatible_det_one",
              "exchange_det_two"):
        check_true(k, r[k])
    check("variables", r["variables"], 48)
    # control: the wrong grading is not homogeneous
    B = S.dual_quiver()
    Bt = B.col_join(sp.eye(3))
    e, _, _ = S._mutate_pc(tuple(S.X0), tuple(S._ARC_SLOPE[a] for a in S.ARCS), Bt, 0)[0:3]
    new = S._mutate_pc(tuple(S.X0), tuple(S._ARC_SLOPE[a] for a in S.ARCS), Bt, 0)[0][0]
    check_true("control: grading with -B is not homogeneous", not S._g_and_F(new, -B)[2])
    check_true("control: grading with B is homogeneous", S._g_and_F(new, B)[2])
    check("arcs crossing (1,0) once, |r| <= 10 and <= 20", S.crossings_unbounded(10), 10)
    check("same, |r| <= 20", S.crossings_unbounded(20), 20)


def main():
    for t in (test_quiver, test_mutation, test_fans, test_lucas, test_hat,
              test_loops, test_theta, test_principal):
        t()
    print()
    if FAILURES:
        print("FAILED: %d" % len(FAILURES))
        for f in FAILURES:
            print("  - " + f)
        return 1
    print("all sixsphere_surface tests passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
