#!/usr/bin/env python3
"""
The characteristic polynomial of the Hofstadter model.

pyCICY.quantum_curve quantizes the mirror curve of a local Calabi-Yau and
diagonalises the resulting difference operator. That gives eigenvalues, and
the butterfly is a scatter plot of them. What it never had was the
characteristic polynomial as an object.

Marra, Proietti and Sheng supply one. Their Theorem III.9 writes

    f(E) = det( H-hat_{P/Q} - E )

at the mid-band point as a sum over two-step elementary symmetric polynomials
evaluated at sin^2(j pi alpha). This script walks the results, checking each
against a numerical determinant of the same matrix, and finishes by comparing
the operator against the one quantum_curve builds from the toric diagram of
local F_0 -- two papers, two constructions, one lattice model.

The two-step polynomials are the reason any of this is computable at large Q.
Defined as a sum over subsets with no two adjacent indices, they look
exponential; the recurrence of Lemma III.6 makes them quadratic. The script
times both.

References
----------
    Marra, Proietti, Sheng, arXiv:2312.14242 (J. Math. Phys. 65, 072102)
    Hatsuda, Katsura, Tachikawa, New J. Phys. 18 (2016) 103023
    Hatsuda, Sugimoto, Xu, arXiv:1701.01561 (the basis of quantum_curve)

Usage
-----
    python3 examples/hofstadter_duality.py
    python3 examples/hofstadter_duality.py --flux 5/11
    python3 examples/hofstadter_duality.py --big 401
"""

import argparse
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import hofstadter as H


def banner(text):
    print("\n" + "=" * 78)
    print(text)
    print("=" * 78)


def theorem(P, Q):
    banner("Theorem III.9, against a numerical determinant")
    print("""\
The formula says the entire dependence on the flux enters through the two-step
symmetric functions of the Q-1 numbers sin^2(j pi alpha). Nothing about the
matrix makes that obvious, so it is worth checking rather than trusting.
""")
    worst, table = H.verify_theorem_III9()
    print("  {:<12} {:>18}".format("flux P/Q", "|det - formula|"))
    for p, q, err in table:
        print("  {:<12} {:>18.2e}".format("%d/%d" % (p, q), err))
    print("\n  worst over all cases: %.2e" % worst)

    print("\nThe polynomial for P/Q = %d/%d, coefficients highest degree first:" % (P, Q))
    c = H.char_poly_coefficients(P, Q)
    print("   ", np.array2string(np.round(c, 8), precision=6, max_line_width=72))
    print("""
Only powers E^(Q-2i) survive. That is the statement that the spectrum is
symmetric under E -> -E when Q is even, and it is visible in the zeros above
rather than needing a separate argument.""")

    roots = np.sort(np.roots(c).real)
    eig = H.spectrum(P, Q)
    print("\n  roots of f      %s" % np.array2string(roots, precision=5))
    print("  eigenvalues     %s" % np.array2string(eig, precision=5))
    print("  agreement       %.2e" % float(np.max(np.abs(roots - eig))))


def identity():
    banner("Remark III.11: a closed form with no free parameters")
    print("""\
For even Q, the top two-step symmetric function of those sines evaluates to a
pure power of four, with no dependence on P at all:

    etilde_{Q/2}( sin^2(pi a), ..., sin^2((Q-1) pi a) ) = 4^{-(Q/2 - 1)} .
""")
    worst, table = H.verify_etilde_identity()
    print("  {:>3} {:>3} {:>18} {:>18}".format("Q", "P", "computed", "4^-(Q/2-1)"))
    for p, q, got, want, err in table:
        print("  {:>3} {:>3} {:>18.12g} {:>18.12g}".format(q, p, got, want))
    print("\n  worst error: %.2e" % worst)
    print("""
It follows from f(0) = 4(-1)^{Q/2}, so it checks the constant term of Theorem
III.9 by a route that does not involve the rest of the polynomial.""")


def brillouin(P, Q):
    banner("The Chambers relation, and where the zero mode sits")
    print("""\
All of the dependence on the two momenta is one additive constant, independent
of E:

    f(E, nx, ny) = f(E, pi/2Q, pi/2Q) + 2 (-1)^{Q-1} ( cos Q nx + cos Q ny ) .

That is why the mid-band point is the natural place to state the formula: the
band structure is a rigid translation of a single polynomial.
""")
    print("  worst deviation over random torus points: %.2e" % H.verify_chambers())

    print("""
The parity of Q then decides where E = 0 lives. The characteristic polynomial
has only powers E^{Q-2i}, so its parity is that of Q, and f(0) picks out the
constant term.
""")
    print("  {:>3}  {:<10} {:>14} {:>14} {:>14}".format(
        "Q", "predicted", "centre", "mid-band", "corner"))
    for q in range(3, 13):
        name, _ = H.zero_mode_point(q)
        vals = []
        for pt in [(0.0, 0.0), (math.pi / (2 * q),) * 2, (math.pi / q,) * 2]:
            vals.append(np.linalg.det(H.hofstadter_matrix(1, q, *pt)).real)
        print("  {:>3}  {:<10} {:>14.2e} {:>14.2e} {:>14.2e}".format(
            q, name, *vals))
    print("""
The prediction is sharp: at the other two points f(0) is of order one, not
small. For even Q the zero is doubly degenerate and the dispersion around it
is linear -- the Dirac cones of Wen and Zee.""")


def scaling(big):
    banner("Why the recurrence matters")
    print("""\
etilde_k sums over k-subsets with no two adjacent indices. Read as a
definition that is exponential in the number of variables; read through Lemma
III.6 it is a two-term recurrence and quadratic.
""")
    xs = np.random.default_rng(0).normal(size=18)
    t0 = time.time()
    for k in range(0, 10):
        H._etilde_bruteforce(k, xs)
    t_brute = time.time() - t0
    t0 = time.time()
    for k in range(0, 10):
        H.etilde(k, xs)
    t_rec = time.time() - t0
    print("  18 variables, all k:  enumeration %.4fs   recurrence %.4fs"
          % (t_brute, t_rec))

    t0 = time.time()
    c = H.char_poly_coefficients(int(big) // 2 + 1, int(big))
    dt = time.time() - t0
    print("  degree %d polynomial:  %.3fs by recurrence" % (len(c) - 1, dt))
    print("  the same by enumeration would need 2^%d subsets." % (int(big) - 1))


def bridge():
    banner("The same operator, from the other side of the package")
    print("""\
pyCICY.quantum_curve.harper() is the square lattice, which is the mirror curve
of local F_0 quantized by the rule of Hatsuda, Sugimoto and Xu. This module
builds the finite-dimensional Hofstadter Hamiltonian from Definition I.5 of a
different paper. They are the same lattice model and must agree.
""")
    from pyCICY import quantum_curve as QC

    harper = QC.harper()
    print("  {:<10} {:>16} {:>16} {:>12}".format(
        "flux", "quantum_curve", "hofstadter", "difference"))
    for P, Q in [(1, 3), (1, 4), (2, 5), (3, 7)]:
        band = np.concatenate(harper.bands(P, Q, nk=24)).ravel()
        mine = []
        for nx in np.linspace(0, 2 * np.pi / Q, 24):
            for ny in np.linspace(0, 2 * np.pi / Q, 24):
                mine.extend(H.spectrum(P, Q, nx, ny))
        mine = np.array(mine)
        a = (band.min(), band.max())
        b = (mine.min(), mine.max())
        print("  {:<10} {:>16} {:>16} {:>12.2e}".format(
            "%d/%d" % (P, Q),
            "[%.4f, %.4f]" % a, "[%.4f, %.4f]" % b,
            abs(a[0] - b[0]) + abs(a[1] - b[1])))
    print("""
The band extents agree. The analytic machinery here describes the operator the
geometric machinery already had, which is the only reason it belongs in this
package rather than in a note of its own.""")


def duality(P, Q):
    banner("Modular duality, and what is not known")
    print("""\
The butterfly is self-similar under two moves on the flux, alpha -> alpha + 1
and alpha -> 1/alpha. The second exchanges P and Q and relates two spectra of
different sizes, through

    (-1)^Q f_{P/Q}(E) = (-1)^P f_{Q/P}(Etilde)

at the mid-band point. The map E -> Etilde is not known in closed form. The
paper says so, and calls understanding it the problem the formula is meant to
serve; this package does not claim otherwise.
""")
    d = H.spectral_duality_check(P, Q)
    print("  (-1)^Q f_{%d/%d} evaluated at E = %s" % (P, Q, d["energies"]))
    print("     ", np.round(d["lhs_values"], 6).tolist())
    print("""
For (P, Q) = (2, 3) the paper writes the pair out explicitly as

    E^3 - 6E - 2cosh(3x)  =  Etilde^2 - 4 - 2cosh(3x) ,

whose x-independent parts give the relation. One caution, recorded rather than
resolved: the sign in Theorem II.2 as stated and the sign of that worked
example do not appear to agree, so neither is baked into a test here.""")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--flux", default="3/7",
                    help="flux P/Q to show in detail (default 3/7)")
    ap.add_argument("--big", type=int, default=201,
                    help="degree for the scaling demonstration")
    args = ap.parse_args(argv)

    P, Q = (int(x) for x in args.flux.split("/"))
    if math.gcd(P, Q) != 1:
        raise SystemExit("flux must be in lowest terms")

    theorem(P, Q)
    identity()
    brillouin(P, Q)
    scaling(args.big)
    duality(2, 3)
    bridge()
    return 0


if __name__ == "__main__":
    sys.exit(main())
