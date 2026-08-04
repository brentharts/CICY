#!/usr/bin/env python3
"""
Closing the loop: A-polynomials, colored Jones, and the AJ conjecture.

This package computes Jones polynomials of knots on one side
(pyCICY.knots) and quantizes the mirror curves of local Calabi-Yau
threefolds on the other (pyCICY.quantum_curve). The object joining them is a
plane curve in (C*)^2 and its quantization. On the knot side that curve is
the A-polynomial A(M, L), which cuts out the SL(2,C) character variety of
the knot complement.

The AJ conjecture of Garoufalidis says the colored Jones polynomials obey a
q-difference equation

    A-hat(Q, L; q) J_N = 0,   L J_N = J_{N+1},  Q J_N = q^N J_N,  L Q = q Q L,

and that setting q = 1, Q = M^2 recovers the classical A-polynomial. The
relation L Q = q Q L is the same Weyl algebra pyCICY.quantum_curve uses,
with q = e^{i hbar}: both ends of the package quantize a plane curve by the
same rule.

This script:

  1. computes colored Jones polynomials for torus knots from the
     Rosso-Jones formula and checks J_2 against the Kauffman-bracket
     computation in pyCICY.knots, which shares no code with it;
  2. reads boundary slopes off the Newton polygons of A-polynomials;
  3. searches for the q-difference recursion of the trefoil and takes its
     classical limit, recovering the A-polynomial;
  4. hands an A-polynomial's Newton polygon to the quantum-curve machinery,
     and is careful about what that does and does not mean.

References
----------
    Garoufalidis, "On the characteristic and deformation varieties of a knot"
    Cooper, Culler, Gillet, Long, Shalen, Invent. Math. 118 (1994) 47
    Hikami and Lovejoy, arXiv:1409.6243 (the colored Jones formula used here)
    Borot and Eynard, arXiv:1205.2261 (the figure-eight A-polynomial)

Usage
-----
    python3 examples/aj_conjecture.py
    python3 examples/aj_conjecture.py --skip-recursion    # the slow part
    python3 examples/aj_conjecture.py --colors 8
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import apolynomial as AP
from pyCICY import knots as K
from pyCICY import toric as T

TORUS = [((2, 3), "3_1"), ((2, 5), "5_1"), ((2, 7), "7_1"), ((3, 4), "8_19")]


def colored_jones(ncolors):
    print("=" * 76)
    print("Colored Jones from Rosso-Jones, checked against the Kauffman bracket")
    print("=" * 76)
    for (s, t), name in TORUS:
        cj = AP.colored_jones_torus(s, t, 2)
        ref = K.from_name(name).jones()
        print("  T(%d,%d) = %-5s   J_2 == knots.jones(): %s"
              % (s, t, name, cj == ref))
    print()
    print("  Two entirely separate computations: a sum over 2N-1")
    print("  representation-theoretic terms against a sum over 2^n Kauffman")
    print("  states. They agree coefficient for coefficient, including for")
    print("  the non-alternating 8_19 = T(3,4).")
    print()
    print("  Higher colours of the trefoil:")
    for N in range(1, ncolors + 1):
        print("    J_%-2d = %s" % (N, AP.colored_jones_torus(2, 3, N)))


def polygons():
    print()
    print("=" * 76)
    print("A-polynomials, Newton polygons and boundary slopes")
    print("=" * 76)
    print("  {:<6} {:<34} {:>16} {:>12}".format(
        "knot", "A(L, M)", "boundary slopes", "reflexive"))
    print("  " + "-" * 72)
    for name in ("3_1", "5_1", "7_1", "8_19", "4_1"):
        A = AP.apolynomial(name)
        poly = AP.newton_polygon(A)
        refl = (T.is_reflexive(poly) if len(poly) > 2 else False)
        print("  {:<6} {:<34} {:>16} {:>12}".format(
            name, str(AP.to_sympy(A))[:34],
            ", ".join(str(s) for s in AP.boundary_slopes(A)),
            "yes" if refl else "no"))
    print()
    print("  The edge slopes of the Newton polygon are boundary slopes of")
    print("  incompressible surfaces in the knot complement (Cooper, Culler,")
    print("  Gillet, Long and Shalen). They come out right: pq for each torus")
    print("  knot, and +-4 for the figure-eight.")


def recursion():
    print()
    print("=" * 76)
    print("The q-difference recursion of the trefoil, and its classical limit")
    print("=" * 76)
    print("  searching for annihilating operators (about twenty seconds)...")
    t0 = time.time()
    rep = AP.verify_aj()
    print("  done in %.1fs" % (time.time() - t0))
    print()
    if not rep["found"]:
        print("  no operator found within these bounds")
        print("  " + rep["note"])
        return
    print("  L-degree admitting a solution   %d" % rep["L_degree"])
    print("  nullspace dimension             %d" % rep["nullspace_dim"])
    print("  gcd of classical limits         %s" % rep["gcd_of_classical_limits"])
    print("  known A-polynomial              %s" % rep["a_polynomial"])
    print("  divides the classical limit     %s" % rep["a_polynomial_divides"])
    print("  leftover factor                 %s" % rep["extra_factor"])
    print()
    print("  So the classical limit contains exactly the trefoil's")
    print("  A-polynomial: the geometric factor 1 + L M^6, with 6 = pq, and")
    print("  the abelian factor L - 1 from the reducible representations.")
    print()
    print("  Two caveats, both real. The leftover factor is expected: the")
    print("  classical limit of *an* annihilating operator contains the")
    print("  A-polynomial but need not equal it, and we take any operator in")
    print("  the nullspace rather than the minimal one. And the L-degree is")
    print("  the smallest within the chosen search bounds; widen the Q- or")
    print("  q-ranges and it can move. Nothing here proves minimality.")


def quantize():
    print()
    print("=" * 76)
    print("Handing a knot to the quantum-curve machinery")
    print("=" * 76)
    for name in ("3_1", "4_1"):
        A = AP.apolynomial(name)
        curve = AP.to_quantum_curve(A, name=name)
        print("  %s: %d hops %s" % (name, len(curve.points), curve.points))
        E = curve.spectrum(1, 3, nk=8)
        print("      spectrum at hbar/2pi = 1/3: [%+.4f, %+.4f]"
              % (E.min(), E.max()))
    print()
    print("  The AJ operators satisfy L Q = q Q L, which is the Weyl algebra")
    print("  quantum_curve quantizes with q = e^{i hbar}, so the same rule")
    print("  applies to both. But a toric diagram is a *reflexive* polygon")
    print("  and an A-polynomial's Newton polygon generally is not, as the")
    print("  table above shows. These operators are therefore not the")
    print("  quantized mirror curve of any local Calabi-Yau, and their")
    print("  spectra are not a Hofstadter problem for one. What the two ends")
    print("  of this package genuinely share is the quantization rule.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--colors", type=int, default=6,
                    help="how many colours of the trefoil to print")
    ap.add_argument("--skip-recursion", action="store_true",
                    help="skip the nullspace search, which is the slow part")
    args = ap.parse_args()

    colored_jones(args.colors)
    polygons()
    if not args.skip_recursion:
        recursion()
    quantize()


if __name__ == "__main__":
    main()
