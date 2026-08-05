#!/usr/bin/env python3
"""
The 24-cell, from reflexive polytope to flavour physics.

Ali, arXiv:2511.10685, proposes the 24-cell as a quantum of spacetime encoding
Standard Model structure, and cites Braun's construction of Calabi-Yau
threefolds from it. Both halves of that are checkable, and this script checks
them: pyCICY.polytope supplies the geometry, pyCICY.flavor the physics.

The script is organised as three acts.

*The machinery*, calibrated where the answer is known. pyCICY.toric already
handles reflexive polygons; polytope.py generalises it and must reproduce it
in two dimensions on all sixteen. Batyrev's Hodge numbers are calibrated on
the quintic, where the answer is (1, 101).

*The 24-cell*, which turns out to need care about which lattice is meant. It
is not reflexive in Z^4. It is reflexive in the lattice its own vertices
generate, and there it gives a threefold with h^11 = h^21 = 20.

*The flavour claims*, which divide cleanly. The hypercharge functional
reproduces all fifteen Standard Model hypercharges exactly. The tetrahedral
projection gives tribimaximal mixing exactly. Three other things do not
follow from their stated inputs, and the script shows the arithmetic rather
than asserting the conclusion either way.

References
----------
    Ali, arXiv:2511.10685
    Braun, JHEP 05 (2012) 101, arXiv:1102.4880
    Batyrev, J. Alg. Geom. 3 (1994) 493
    Coxeter, Regular Polytopes (1973)

Usage
-----
    python3 examples/twentyfour_cell.py
    python3 examples/twentyfour_cell.py --act geometry
    python3 examples/twentyfour_cell.py --no-census      (skips ~13s)
"""

import argparse
import itertools
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import flavor as F
from pyCICY import polytope as P
from pyCICY import toric


def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


# ---------------------------------------------------------------------------

def machinery():
    banner("Act I: the machinery, calibrated where the answer is known")

    print("""\
pyCICY.toric is two-dimensional throughout, and none of it generalises: its
dual assumes an ordered vertex cycle and its lattice points are scan-converted
as a polygon. polytope.py lifts all of that to any dimension, which means it
has a ready-made oracle in two dimensions and no excuse for not using it.
""")
    polys = toric.enumerate_reflexive()
    exact = sum(1 for v in polys
                if sorted(map(tuple, np.array(toric.dual(np.array(v, int)), int).tolist()))
                == sorted(map(tuple, P.polar(np.array(v, int), convention="toric").tolist())))
    print("  reflexive polygons rederived by toric:      %d" % len(polys))
    print("  on which polar() reproduces toric.dual:     %d" % exact)

    twelve = sum(1 for v in polys
                 if (len(P.lattice_points(np.array(v, int)))
                     - len(P.interior_lattice_points(np.array(v, int)))
                     + len(P.lattice_points(P.polar(np.array(v, int), convention="toric")))
                     - len(P.interior_lattice_points(P.polar(np.array(v, int), convention="toric")))) == 12)
    print("  on which the twelve theorem still holds:    %d" % twelve)

    print("""
There is a sign convention to be careful about. toric.dual uses
P* = {y : <x,y> <= 1}; Batyrev's definition is <x,y> >= -1. The two differ by
y -> -y and nothing else, so Hodge numbers are unaffected -- every quantity
they use is a lattice point count -- but printed vertex lists are not. Hence
the convention argument rather than a silent choice.
""")

    banner("Batyrev, calibrated on the quintic")
    print("""\
A reflexive 4-polytope Delta gives a Calabi-Yau threefold. Which of Delta and
its polar to pass is an easy thing to get backwards, and getting it backwards
returns the mirror -- silently, unless one already knows the answer.
""")
    fan = P.simplex(4)
    newton = P.polar(fan)
    h = P.batyrev_hodge(newton)
    hm = P.batyrev_hodge(fan)
    print("  Delta  = polar of the P^4 fan simplex, l(Delta) = %d = C(9,4)"
          % len(P.lattice_points(newton)))
    print("     one lattice point per quintic monomial in five variables")
    print("  -> (h11, h21) = (%d, %d),  chi = %d      the quintic"
          % (h["h11"], h["h21"], h["euler"]))
    print("  Delta  = the fan simplex itself, l(Delta) = %d"
          % len(P.lattice_points(fan)))
    print("  -> (h11, h21) = (%d, %d),  chi = %d      the mirror quintic"
          % (hm["h11"], hm["h21"], hm["euler"]))

    k3 = P.batyrev_hodge(P.cross_polytope(3))
    print("\n  a reflexive 3-polytope gives a K3 instead: h11 = %d, chi = %d"
          % (k3["h11"], k3["euler"]))
    print("""
Worth stating because the paper describes the 24-cell as "the Newton polytope
defining a smooth K3 hypersurface". A reflexive three-polytope gives a K3; the
24-cell is four-dimensional and gives a threefold, which is what Braun's title
says.""")


# ---------------------------------------------------------------------------

def geometry():
    banner("Act II: the 24-cell, and which lattice is meant")

    raw = P.d4_roots()
    print("""\
Ali's section 3.1 writes the 24-cell as the 24 vectors +-e_i +- e_j, the D_4
root system, of squared norm 2. Section 2.1 writes it as {+-e_i} together with
the sixteen (1/2)(+-1,+-1,+-1,+-1), of squared norm 1. A remark in section 2.1
says the long roots are "not vertices of the 24-cell".

They are polar duals of one another.
""")
    d = P.polar(raw, exact=False)
    unit = P.twenty_four_cell("unit")
    same = (set(map(tuple, np.round(d, 6).tolist()))
            == set(map(tuple, np.round(unit, 6).tolist())))
    print("  polar of the section 3.1 set equals the section 2.1 set: %s" % same)
    print("  sample dual vertices: %s" % np.round(d[:3], 3).tolist())
    print("""
And they are half-integral, so the 24-cell as written is NOT reflexive with
respect to Z^4. Reflexivity is a property of a polytope *and a lattice*, and
the natural lattice for a root polytope is the root lattice.
""")
    r0 = P.is_reflexive(raw, use_generated_lattice=False)
    r = P.is_reflexive(raw)
    print("  reflexive in Z^4 ................ %s" % r0["reflexive"])
    print("  reflexive in the lattice it generates ... %s" % r["reflexive"])
    print("  that lattice is D_4 = {x in Z^4 : sum x_i even}, of index %d"
          % r["index"])

    V = P.twenty_four_cell()
    print("\n  f-vector in that basis: %s" % P.f_vector(V))
    print("  Euler relation 24 - 96 + 96 - 24 = %d" % (24 - 96 + 96 - 24))
    print("  lattice points of Delta: %d   interior: %d (the origin)"
          % (len(P.lattice_points(V)), len(P.interior_lattice_points(V))))

    banner("The Calabi-Yau threefold")
    h = P.batyrev_hodge(V, verbose=True)
    print("""\
Delta and Delta* each have 25 lattice points -- their 24 vertices and the
origin -- so no proper face has an interior lattice point and both Batyrev
correction sums vanish identically. Nothing is left but the leading counts.
""")
    print("  l(Delta) = %d,  l(Delta*) = %d" % (h["l_delta"], h["l_delta_star"]))
    print("  facet corrections    %s" % (h["facet_correction"],))
    print("  codim-2 corrections  %s" % (h["codim2_correction"],))
    print("\n  h^{1,1} = %d,  h^{2,1} = %d,  chi = %d"
          % (h["h11"], h["h21"], h["euler"]))
    print("""
Self-duality forces h11 = h21 with no arithmetic at all, so two independent
arguments land in the same place. Braun's Hodge numbers (1,1) are those of a
free quotient of this cover; enumerating free quotients is the boundary
pyCICY.symmetries draws on the CICY side, and it is not crossed here.""")

    banner("The tetrahedron count")
    tets = P.equilateral_subsets(raw, 4, edge_sq=4)
    allt = P.equilateral_subsets(raw, 4)
    print("""\
Section 3.1 asks for four vertices with ||v_i - v_j||^2 = 4 throughout and
states there are 576 of them. Brute force over all %d four-element subsets:
""" % math.comb(24, 4))
    print("  with ||v_i - v_j||^2 = 4 .................. %d" % len(tets))
    print("  equilateral at any edge length at all ..... %d" % len(allt))
    print("""
So 576 is not recoverable by relaxing the edge length either. The 48 is
structural: in this normalisation ||v_i - v_j||^2 = 4 is exactly orthogonality,
the twelve diagonals fall into three mutually orthogonal frames of four, and
each frame carries 2^4 sign choices. 3 x 16 = 48.
""")
    V4 = raw
    orth = all(int(V4[a] @ V4[b]) == 0
               for t in tets for a, b in itertools.combinations(t, 2))
    print("  the condition really is pairwise orthogonality: %s" % orth)
    example = {(1, 1, 0, 0), (1, -1, 0, 0), (0, 0, 1, 1), (0, 0, 1, -1)}
    found = any(set(map(tuple, V4[list(t)].tolist())) == example for t in tets)
    print("  the paper's own example tetrahedron is among the 48: %s" % found)


# ---------------------------------------------------------------------------

def flavour(census=True):
    banner("Act III: what the flavour construction does and does not give")

    print("What is exact:\n")
    ok, table = F.verify_hypercharges()
    print("  all fifteen Standard Model hypercharges, in exact rationals: %s" % ok)
    print("  {:>4} {:>5} {:>10} {:>10}".format("gen", "field", "computed", "target"))
    for g, name, got, want, good in table:
        if g == 1:
            print("  {:>4} {:>5} {:>10} {:>10}".format(g, name, str(got), str(want)))
    print("  ... and likewise for generations 2 and 3")
    used, unused = F.distinct_vertices()
    print("\n  distinct vertices used: %d of the 16 in V_2; unused: %s"
          % (len(used), unused.pop()))
    print("  Tr(Y) per generation: %s   (the U(1)_Y anomaly cancels)"
          % F.anomaly_trace())

    G = F.gram_matrix()
    print("\n  Gram matrix of the projected tetrahedron, off-diagonals: %s"
          % np.round(G[np.triu_indices(4, 1)], 12).tolist())
    print("  J eigenvalues: %s"
          % np.round(np.linalg.eigvalsh(F.J_matrix()), 10).tolist())
    t12, t13, t23 = F.tbm_angles()
    print("  tribimaximal angles: th12 = %.4f, th13 = %.4f, th23 = %.4f degrees"
          % (t12, t13, t23))

    banner("What does not follow (1): the hypercharges are fitted")
    print("""\
The functional has four components of h_Y plus an offset eps, so five free
parameters against five target hypercharges. If that 5 x 5 system has full
rank then a unique solution exists for ANY targets whatever, and nothing is
being derived.
""")
    for g in (1, 2, 3):
        print("  generation %d: rank %d of 5" % (g, F.fit_rank(g)["rank"]))
    print("\n  So generations 1 and 2 are interpolations.")

    banner("What does not follow (2): except generation 3, and here is why")
    a3 = F.epsilon_zero_analysis(3)
    print("""\
Generation 3 sets eps = 0, leaving four unknowns for five equations. That is
generically inconsistent -- and it is consistent here, which is a real
property of the vertex choice.

The five vertices carry exactly one linear dependency. Consistency is exactly
the vanishing of the same combination of the targets.
""")
    print("  null vector c (order lL, qL, eR, uR, dR): %s"
          % [int(x) for x in a3["c"]])
    print("  c . Y = %s" % a3["c_dot_Y"])
    print("  solved h_Y^(3) = %s   (the paper's value exactly)"
          % [str(x) for x in a3["h"]])
    for g in (1, 2):
        print("  generation %d: c . Y = %s, so eps = 0 is impossible"
              % (g, F.epsilon_zero_analysis(g)["c_dot_Y"]))
    print("""
Substituting only the Yukawa-invariance relations Y_L = Y_eR + Y_H,
Y_uR = Y_q + Y_H, Y_dR = Y_q - Y_H turns c . Y into

    -4 Y_H - 2 Y_eR ,

which vanishes because Y_eR = -1 and Y_H = 1/2. The coincidence is a fact
about the Standard Model hypercharges, not about the 24-cell.""")
    if census:
        hits, total, frac = F.epsilon_zero_census()
        print("""
And it is not rare. Over every five-subset of V_2 and every assignment of the
five species to it:
""")
        print("  admitting an eps = 0 solution: %d of %d  (%.2f%%)"
              % (hits, total, 100 * frac))
        print("  about one choice in thirty, so this is a family, not a "
              "selection principle.")

    banner("What does not follow (3): the MDP has nothing to minimise")
    d = F.mdp_distortion()
    print("""\
The paper defines D(Pi) = sum |  ||Pi(v_i)-Pi(v_j)|| - ||v_i-v_j||  | and takes
eta to be the residual distortion of the optimal projection, then uses
eta ~ 0.02 to drive both theta_13 and the Cabibbo angle.

But four points span an affine subspace of dimension at most three, so the
orthogonal projection onto their hull is an isometry.
""")
    print("  affine rank of the tetrahedron: %d"
          % np.linalg.matrix_rank(F.TETRAHEDRON - F.TETRAHEDRON.mean(0)))
    print("  D(Pi) at that projection: %.2e" % d)
    print("""
The tetrahedron is embedded exactly. Whatever eta is, it is not the distortion
of this map.""")

    banner("The two estimates behave oppositely")
    t13 = F.theta13_from_strain(0.017, 0.022)
    print("  theta_13 from eps_13 = 0.017, eta = 0.022:  %.2f degrees"
          % t13)
    print("     the paper quotes 8.5 -- the arithmetic closes.\n")
    lo = F.cabibbo_angle(0.02)
    hi = F.cabibbo_angle(0.03)
    print("  Cabibbo, tan(th_C) ~ sqrt(2/3) * eta * 2, as written:")
    print("     eta = 0.02  ->  %.4f  (%.2f degrees)"
          % (lo["tan_theta"], lo["degrees"]))
    print("     eta = 0.03  ->  %.4f  (%.2f degrees)"
          % (hi["tan_theta"], hi["degrees"]))
    print("     the paper quotes 0.22-0.26, i.e. 12.6-15 degrees.")
    print("\n  measured: sin(th_C) = %.4f, th_C = %.2f degrees"
          % (lo["measured_sin_theta_C"], lo["measured_degrees"]))
    print("  eta that would be needed: %.3f, which is %.1f times the %.3f"
          % (lo["eta_for_quoted"], lo["eta_for_quoted"] / 0.022, 0.022))
    print("  the reactor angle requires.")
    print("""
Since eta is described as a universal distortion shared by the quark and
lepton sectors, the two determinations are in tension by that factor. There is
no v2 of the paper. pyCICY.flavor evaluates the formula as written and returns
eta_for_quoted alongside it, so the gap stays visible instead of being
absorbed into a parameter the paper never states.""")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--act", choices=["machinery", "geometry", "flavour", "all"],
                    default="all")
    ap.add_argument("--no-census", action="store_true",
                    help="skip the exhaustive eps=0 scan (about 13 seconds)")
    args = ap.parse_args(argv)

    if args.act in ("machinery", "all"):
        machinery()
    if args.act in ("geometry", "all"):
        geometry()
    if args.act in ("flavour", "all"):
        flavour(census=not args.no_census)
    return 0


if __name__ == "__main__":
    sys.exit(main())
