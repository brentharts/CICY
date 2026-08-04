#!/usr/bin/env python3
"""
Hyperbolic lattices and automorphic Bloch theory.

On a hyperbolic {p,q} lattice the ordinary Bloch theorem fails, because the
translation group is Fuchsian and not abelian. Maciejko and Rayan,
"Automorphic Bloch theorems for hyperbolic lattices", PNAS 119(9)
e2116869119 (2022), replace it: periodic boundary conditions compactify the
lattice onto a genus-g Riemann surface and eigenstates transform under a
unitary representation of the Fuchsian group. One-dimensional
representations give a Brillouin zone that is the Jacobian torus T^{2g};
higher-dimensional irreducible representations give sectors no torus
captures.

This script walks the construction for the {4g,4g} family:

  * the geometry, with the circumradius derived numerically rather than
    quoted, since the three standard length formulas are easy to permute;
  * the Fuchsian generators, their relator, and the genus confirmed
    independently by Gauss-Bonnet;
  * why finite patches will not do -- the boundary fraction of a hyperbolic
    flake tends to (p-2)/(p-1), not to zero;
  * the abelian sector, which collapses to a 2g-dimensional cosine band;
  * higher-dimensional sectors built from clock and shift matrices.

Usage
-----
    python3 examples/hyperbolic_bloch.py
    python3 examples/hyperbolic_bloch.py --genus 3
    python3 examples/hyperbolic_bloch.py --plot /tmp
"""

import argparse
import math
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import hyperbolic as HY


def geometry():
    print("=" * 74)
    print("Geometry of the regular {p,q} tessellations")
    print("=" * 74)
    print("  {:<9} {:>10} {:>10} {:>10} {:>10} {:>10}".format(
        "tiling", "R (solved)", "R (formula)", "inradius", "edge", "area"))
    print("  " + "-" * 66)
    for p, q in [(7, 3), (3, 7), (5, 4), (8, 8), (12, 12), (16, 16)]:
        print("  {:<9} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f}".format(
            "{%d,%d}" % (p, q), HY.solve_circumradius(p, q),
            HY.circumradius(p, q), HY.inradius(p, q),
            HY.edge_length(p, q), HY.cell_area(p, q)))
    print()
    print("  The second column is obtained by demanding the cell's interior")
    print("  angle be 2 pi / q and solving numerically; the third is the")
    print("  closed form cosh R = cot(pi/p) cot(pi/q). They agree. The other")
    print("  two lengths are cosh r = cos(pi/q)/sin(pi/p) and")
    print("  cosh(l/2) = cos(pi/p)/sin(pi/q); it is easy to swap them.")


def group(g):
    p = 4 * g
    print()
    print("=" * 74)
    print("The Fuchsian group of {%d,%d}, and the genus" % (p, p))
    print("=" * 74)
    gens = HY.generators(p)
    print("  %d side-pairing translations, each of length 2r = %.6f"
          % (len(gens), 2 * HY.inradius(p, p)))
    pairs = all(np.allclose(gens[k] @ gens[(k + p // 2) % p], np.eye(2),
                            atol=1e-8) for k in range(p))
    print("  generator k is the inverse of generator k + %d: %s" % (p // 2, pairs))
    print("  relator  g0 g1^-1 g2 g3^-1 ... = 1  holds to %.1e"
          % HY.relator_residual(p))
    print()
    print("  Note this is the relator of the regular %d-gon with opposite" % p)
    print("  sides identified, and not the canonical surface word")
    print("  prod_i [a_i, b_i] = 1. Both present the same group, but in")
    print("  different generators; the canonical word evaluated on these")
    print("  generators is nowhere near the identity.")
    print()
    print("  cell area          %.6f" % HY.cell_area(p, p))
    print("  4 pi (g - 1)       %.6f" % (4 * math.pi * (g - 1)))
    print("  2 pi |chi|         %.6f" % (2 * math.pi * abs(2 - 2 * g)))
    print("  => genus %d, confirmed by Gauss-Bonnet independently of the group"
          % HY.genus(p))


def flakes(g, maxdepth=4):
    p = 4 * g
    print()
    print("=" * 74)
    print("Why a finite patch will not do")
    print("=" * 74)
    print("  {:>6} {:>9} {:>12} {:>12}".format(
        "depth", "cells", "boundary", "spectrum"))
    print("  " + "-" * 42)
    for d in range(1, maxdepth + 1):
        cells = len(HY.flake(p, p, d))
        frac = HY.boundary_fraction(p, p, d)
        E = HY.flake_spectrum(p, p, d)
        print("  {:>6} {:>9} {:>12.4f} {:>12}".format(
            d, cells, frac, "%+.3f .. %+.3f" % (E.min(), E.max())))
    print()
    print("  The boundary fraction tends to (p-2)/(p-1) = %.4f, not to zero."
          % HY.boundary_fraction_limit(p))
    print("  Cells grow by a factor of p-1 = %d per ring, so a fixed share of"
          % (p - 1))
    print("  them always sits on the rim however far out one goes. That is")
    print("  the quantitative reason hyperbolic band theory needs periodic")
    print("  boundary conditions and automorphic functions, rather than a")
    print("  large-flake extrapolation.")


def sectors(g, dims, samples):
    p = 4 * g
    print()
    print("=" * 74)
    print("Representation sectors")
    print("=" * 74)
    print("  A one-dimensional representation sends each generator to a")
    print("  phase, so the Bloch Hamiltonian collapses to a number:")
    print("      E(k) = 2 t sum_{j=1}^{%d} cos k_j,   k in T^{%d}"
          % (2 * g, 2 * g))
    print("  the band structure of a %d-dimensional hypercubic lattice."
          % (2 * g))
    print()
    print("  For N > 1 the generators become N x N unitaries obeying the same")
    print("  relator. Clock and shift matrices work because their commutator")
    print("  is a scalar, [X^a, Z^b] = omega^{-ab} 1, so the relator can be")
    print("  solved in closed form.")
    print()
    rep = HY.compare_sectors(g=g, dims=dims, depth=2, samples=samples)
    print("  {:<22} {:>10} {:>10} {:>10} {:>9}".format(
        "sector", "min", "max", "mean", "samples"))
    print("  " + "-" * 64)
    for row in rep["rows"]:
        print("  {:<22} {:>10.4f} {:>10.4f} {:>10.4f} {:>9}".format(
            row["sector"], row["min"], row["max"], row["mean"],
            row["n_samples"]))
    print()
    print("  These do not agree, and are not meant to. The abelian row is")
    print("  only the one-dimensional part of the representation theory; the")
    print("  flake is a finite patch whose boundary never becomes small; and")
    print("  the N-dimensional rows are particular representations, not a")
    print("  complete set. Getting the full spectrum needs the normal")
    print("  subgroups of the Fuchsian group, which is a computational group")
    print("  theory problem this module does not attempt.")


def plot(outdir, g):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; pip install pyCICY[viz]")
        return
    from pyCICY import viz

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.2))
    viz.plot_hyperbolic_flake(4 * g, depth=2, ax=axes[0])
    depths = range(1, 6)
    axes[1].plot(list(depths),
                 [HY.boundary_fraction(4 * g, 4 * g, d) for d in depths],
                 "o-", label="measured")
    axes[1].axhline(HY.boundary_fraction_limit(4 * g), color="tab:red",
                    ls="--", label="$(p-2)/(p-1)$")
    axes[1].set_xlabel("flake depth")
    axes[1].set_ylabel("boundary fraction")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False)
    axes[1].set_title("the boundary never becomes negligible", fontsize=10)
    fig.tight_layout()
    path = os.path.join(outdir, "hyperbolic_genus%d.pdf" % g)
    fig.savefig(path)
    print("\nwrote " + path)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--genus", type=int, default=2,
                    help="genus g, using the {4g,4g} tessellation")
    ap.add_argument("--dims", type=int, nargs="*", default=[1, 2, 3, 4],
                    help="representation dimensions to compare")
    ap.add_argument("--samples", type=int, default=800)
    ap.add_argument("--depth", type=int, default=4, help="deepest flake")
    ap.add_argument("--plot", metavar="DIR", default=None)
    args = ap.parse_args()

    if args.genus < 2:
        ap.error("genus must be at least 2")

    geometry()
    group(args.genus)
    flakes(args.genus, args.depth)
    sectors(args.genus, tuple(args.dims), args.samples)
    if args.plot:
        plot(args.plot, args.genus)


if __name__ == "__main__":
    main()
