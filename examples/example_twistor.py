#!/usr/bin/env python3
"""
Twistor theory and scattering amplitudes, exactly over the rationals.

    python3 examples/twistor.py
    python3 examples/twistor.py --n 7
    python3 examples/twistor.py --only bcfw

Witten's point is that perturbative gauge theory, a swamp in Feynman
diagrams, is simple in twistor space. The point for a package built on exact
arithmetic is narrower and more useful: a tree amplitude is a rational
function of spinor brackets, so with rational spinors it is a rational
number, and every claim about it is decidable. Nothing below is checked to
within a tolerance.

    1  kinematics    momentum conservation as a linear condition, imposed by
                     construction, and the identities that follow from it
    2  parke-taylor  one line replacing a factorially growing number of
                     Feynman diagrams, and the little-group weights that fix
                     its form
    3  bcfw          the same amplitudes rebuilt from three-point ones, and
                     the three-point degeneracy that forces complex momenta
    4  nmhv          amplitudes with no closed form, checked by symmetry
    5  relations     what a rank computation can and cannot see
    6  positroid     cells of the positive Grassmannian, counted two ways
    7  geometry      the twistor double fibration as complete intersections

Section 5 is the one with a moral. The colour orderings of a tree amplitude
satisfy two families of relations, and a rank computation over many kinematic
points detects exactly one of them --- not because the other is false, but
because its coefficients are momentum-dependent and no single linear relation
holds across the matrix. The method's blind spot is as much a result as its
reach.
"""

import argparse
import sys
from fractions import Fraction

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from pyCICY import twistor as TW


def banner(text):
    print("\n" + text)
    print("-" * len(text))


def kinematics(n):
    banner("1. Exact kinematics")

    print("""\
A massless momentum is a rank-one bispinor p = lambda lambda-tilde, and
momentum conservation is sum_i lambda_i lambda-tilde_i = 0. With the two
spinors independent -- complexified, or split signature -- that is linear:
each column of lambda-tilde must lie in the kernel of the two-by-n matrix of
lambdas. So a configuration is constructed, not solved for, and it conserves
momentum exactly rather than to fifteen digits.
""")
    k = TW.Kinematics.random(n, seed=n)
    print("  n = %d, spinors over Q:" % n)
    for i in range(1, min(n, 4) + 1):
        print("    leg %d   lambda = %-14s  lambda-tilde = %s"
              % (i, k.lam[i - 1], k.lamt[i - 1]))
    if n > 4:
        print("    ...")
    print("\n  momentum conservation      %s" % k.check())
    print("  Schouten <12><34>+...      %s" % k.schouten_residual(1, 2, 3, 4))
    print("  sum_i <1i>[i2]             %s" % k.momentum_residual(1, 2))
    print("  s(1,2) as a determinant    %s" % k.s(1, 2))
    print("  s(1,2) as <12>[12]         %s" % (k.angle(1, 2) * k.square(1, 2)))
    print("  total momentum squared     %s" % k.s(*range(1, n + 1)))
    print("""
The Schouten identity is not imposed anywhere: it holds because there is no
antisymmetric three-index tensor in two dimensions. That it comes out zero
tests the bracket rather than the kinematics.""")


def parke_taylor(n):
    banner("2. Parke-Taylor")

    k = TW.Kinematics.random(n, seed=n)
    A = TW.parke_taylor(k, (1, 2))
    print("""\
For exactly two negative helicities the colour-ordered tree amplitude is

    A = <ij>^4 / (<12><23> ... <n1>) ,

one line, replacing a number of Feynman diagrams that grows faster than
exponentially in n. It depends on the holomorphic spinors alone, which is the
geometric statement that an MHV amplitude is supported on a degree-one curve
in twistor space: a line.
""")
    print("  n = %d, negatives (1,2):  A = %s" % (n, A))
    print()
    print("  cyclic invariance:")
    for r in (1, 2):
        order = [((i + r) % n) + 1 for i in range(n)]
        print("    rotate by %d -> %s" % (r, TW.parke_taylor(k, (1, 2), order)))
    rev = list(range(n, 0, -1))
    print("  reflection A(n..1) = (-1)^n A:  %s"
          % (TW.parke_taylor(k, (1, 2), rev) == (-1) ** n * A))
    print()
    print("  little-group weights (scale lambda_i by 3, expect 3^{-2h_i}):")
    for leg, h in ((1, -1), (3, 1)):
        lam = [list(v) for v in k.lam]
        lam[leg - 1] = [3 * x for x in lam[leg - 1]]
        got = TW.parke_taylor(lam, (1, 2)) / A
        print("    leg %d, helicity %+d ->  %-8s  expected 3^%d"
              % (leg, h, got, -2 * h))
    print("""
The fourth power in the numerator is fixed by those weights: the denominator
carries weight -2 on every leg, so the numerator must supply +4 on each of the
two negative-helicity legs and nothing elsewhere.""")


def bcfw(nmax):
    banner("3. BCFW, and why three points need complex momenta")

    print("""\
BCFW deforms two external momenta by a complex parameter, uses the vanishing
of the deformed amplitude at large deformation, and reconstructs it from its
poles. Each residue factorises into two smaller amplitudes, so everything
follows from the three-point amplitudes, which Lorentz invariance alone
fixes. Parke-Taylor is a closed formula. They share no code.
""")
    print("  %-4s %-26s %-26s %s" % ("n", "BCFW", "Parke-Taylor", "equal"))
    for n in range(4, nmax + 1):
        k = TW.Kinematics.random(n, seed=n)
        b = TW.tree_amplitude(k.lam, k.lamt, [-1, -1] + [1] * (n - 2))
        p = TW.parke_taylor(k, (1, 2))
        print("  %-4d %-26s %-26s %s" % (n, b, p, b == p))
    print()
    print("  and the conjugate, with n-2 negative helicities:")
    for n in range(4, min(nmax, 7) + 1):
        k = TW.Kinematics.random(n, seed=n + 20)
        b = TW.tree_amplitude(k.lam, k.lamt, [1, 1] + [-1] * (n - 2))
        a = TW.anti_mhv(k, (1, 2))
        print("    n = %d  equal: %s" % (n, b == a))

    print("""
The three-point degeneracy is worth seeing directly. At n = 3 the kernel that
momentum conservation leaves is one-dimensional, so both columns of
lambda-tilde are proportional to a single vector and every square bracket
vanishes identically. There is no configuration with both sets of brackets
non-zero, over any field. The two three-point amplitudes therefore live at
different kinematic points:
""")
    kh = TW.Kinematics.random(3, seed=2, kind="holomorphic")
    ka = TW.Kinematics.random(3, seed=2, kind="antiholomorphic")
    print("    holomorphic point:      <12> = %-8s  [12] = %s"
          % (kh.angle(1, 2), kh.square(1, 2)))
    print("    anti-holomorphic point: <12> = %-8s  [12] = %s"
          % (ka.angle(1, 2), ka.square(1, 2)))
    print("    A(1-,2-,3+) = %s" % TW.tree_amplitude(kh.lam, kh.lamt,
                                                     [-1, -1, 1]))
    print("    A(1-,2+,3+) = %s" % TW.tree_amplitude(ka.lam, ka.lamt,
                                                     [-1, 1, 1]))
    print("""
That is why the recursion needs complex momenta, and it is not an artefact of
working over Q. Real Lorentzian three-point kinematics has every bracket
vanishing, so there would be nothing to recurse from.""")


def nmhv():
    banner("4. Beyond MHV, where there is no formula")

    print("""\
Three negative helicities at six points is the first amplitude with no closed
form. There is nothing to compare it against, so it is checked by the
symmetries it must have -- and these are not free: the BCFW result is a sum
over channels chosen after rotating the ordering, and nothing in the
construction makes the answer cyclic.
""")
    k = TW.Kinematics.random(6, seed=11)
    for label, hel in [("(-,-,-,+,+,+)", [-1, -1, -1, 1, 1, 1]),
                       ("(-,+,-,+,-,+)", [-1, 1, -1, 1, -1, 1])]:
        A = TW.tree_amplitude(k.lam, k.lamt, hel)

        def rot(v, r):
            return [v[(i + r) % 6] for i in range(6)]
        cyc = all(TW.tree_amplitude(rot(k.lam, r), rot(k.lamt, r),
                                    rot(hel, r)) == A for r in range(6))
        ref = TW.tree_amplitude(k.lam[::-1], k.lamt[::-1], hel[::-1]) == A
        print("  %-16s A = %-22s cyclic %s   reflection %s"
              % (label, A, cyc, ref))

    print("\n  the twistor curve each helicity sector localises on:")
    for kk in (2, 3, 4, 5):
        d = TW.mhv_degree(kk)
        print("    %-8s k = %d  ->  degree %d curve" % (d["label"], kk,
                                                        d["degree"]))
    print("    and a loop raises the degree by one, with genus up to the")
    print("    loop order:  %s" % TW.mhv_degree(3, 1))


def relations():
    banner("5. Relations, and what a rank can see")

    print("""\
The (n-1)! colour orderings are not independent. Two families of relations cut
them down, and they behave very differently under a rank computation.
""")
    print("  residuals, which must be exactly zero:")
    print("  %-6s %-20s %s" % ("n", "U(1) decoupling", "fundamental BCJ"))
    for n in range(4, 9):
        k = TW.Kinematics.random(n, seed=n)
        print("  %-6d %-20s %s" % (n, TW.u1_decoupling_residual(k, (1, 2)),
                                   TW.bcj_residual(k, (1, 2))))

    print("""
Both hold exactly. Now take the rank of the matrix of all orderings evaluated
at many kinematic points:
""")
    print("  %-4s %-12s %-8s %-10s %-10s" % ("n", "orderings", "rank",
                                             "(n-2)!", "(n-3)!"))
    for n in (5, 6):
        pts = [TW.Kinematics.random(n, seed=300 + i) for i in range(40)]
        r = TW.ordering_rank(pts, (1, 2))
        f = lambda m: __import__("math").factorial(m)      # noqa: E731
        print("  %-4d %-12d %-8d %-10d %-10d"
              % (n, r["orderings"], r["rank"], f(n - 2), f(n - 3)))
    print("""
The rank is (n-2)!, the Kleiss-Kuijf count, and it is not (n-3)!. That is a
property of the method rather than a failure of BCJ. A rank taken across
kinematic points can only detect relations whose coefficients are constant;
Kleiss-Kuijf coefficients are plus or minus one, so those relations hold at
every point at once and the rank collapses. BCJ coefficients are built from
Mandelstam invariants and vary from point to point, so no single linear
relation holds across the matrix and the further reduction is invisible --
even though, as the residuals above show, the relation itself holds at each
point separately.

Reporting (n-3)! here would require using the relations as input, which would
make the check circular. Reporting (n-2)! and saying why is the honest
result.""")


def positroid():
    banner("6. Cells of the positive Grassmannian")

    print("""\
Postnikov's bijection: the cells of the totally non-negative Grassmannian
G(k,n) correspond to decorated permutations of [n] -- permutations with each
fixed point coloured -- having k anti-exceedances. The same objects label the
on-shell diagrams of planar N=4 super Yang-Mills, which is why the
combinatorics belongs beside the amplitudes.
""")
    print("  %-6s %-40s %-8s %s" % ("n", "cells by k", "total", "sum n!/j!"))
    for n in range(1, 7):
        row = [TW.positroid_cells(k, n) for k in range(n + 1)]
        counted, closed = TW.positroid_total_check(n)
        print("  %-6d %-40s %-8d %d" % (n, row, counted, closed))
    print("""
The rows are symmetric under k -> n-k, as duality of Grassmannians requires,
and G(2,4) has 33 cells. The total has a closed form with nothing obviously to
do with coloured fixed points -- the number of arrangements of n objects,
sum_j n!/j! -- so counting one way and evaluating the other checks the
anti-exceedance statistic.

Top cell dimensions, which are the dimensions of the Grassmannians:""")
    for k, n in [(1, 3), (2, 4), (2, 5), (3, 6)]:
        print("    dim G(%d,%d) = %d" % (k, n, TW.cell_dimension(k, n)))


def geometry():
    banner("7. The twistor double fibration")

    print("""\
Twistor space is P^3. Compactified complexified Minkowski space is the
Grassmannian of two-planes in C^4, which the Plucker embedding realises as a
single quadric in P^5 -- the Klein quadric. That has a configuration matrix,
so the machinery this package already has for complete intersections applies
to it unchanged, and the Euler characteristics come from the same Chern class
routine used for orientifold fixed loci.
""")
    print("  %-22s %-18s %-6s %s" % ("space", "configuration", "dim", "chi"))
    for g in TW.twistor_geometry():
        print("  %-22s %-18s %-6d %d" % (g["name"], g["configuration"],
                                         g["dim"], g["euler"]))
    print("""
The 6 is the number of Schubert cells of G(2,4), indexed by the partitions
fitting in a two-by-two box, and the 12 is the cell count of the flag
manifold. Neither was put in.

The Penrose transform sends a free massless field of helicity h to a class in
H^1 of twistor space with values in O(-2h-2):
""")
    for h in (Fraction(-1), Fraction(-1, 2), Fraction(0), Fraction(1, 2),
              Fraction(1), Fraction(2)):
        p = TW.penrose_helicity(h)
        print("    helicity %-5s ->  %s" % (h, p["bundle"]))
    print("""
with the caveat the module records: H^1(P^3, O(m)) vanishes for every m by
Bott, so the transform cannot be taken on compact twistor space. It lives on
P^3 minus a line, and that open space is not something this package's
cohomology covers -- which is worth saying rather than leaving implied.""")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--n", type=int, default=6, help="number of legs")
    p.add_argument("--only", default=None,
                   choices=["kinematics", "pt", "bcfw", "nmhv", "relations",
                            "positroid", "geometry"])
    a = p.parse_args()

    sections = {"kinematics": lambda: kinematics(a.n),
                "pt": lambda: parke_taylor(a.n),
                "bcfw": lambda: bcfw(max(a.n, 6)),
                "nmhv": nmhv, "relations": relations,
                "positroid": positroid, "geometry": geometry}
    order = ["kinematics", "pt", "bcfw", "nmhv", "relations", "positroid",
             "geometry"]
    for key in ([a.only] if a.only else order):
        sections[key]()
    print()


if __name__ == "__main__":
    main()
