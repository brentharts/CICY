#!/usr/bin/env python3
"""
Chirality and the failure of additivity of the unknotting number.

Brittenham and Hermiller, "Unknotting number is not additive under connected
sum", arXiv:2506.24088, settled Kirby problem 1.69(B) negatively with

    u(7_1 # m7_1) <= 5 < 6 = u(7_1) + u(m7_1),

where 7_1 is the (2,7) torus knot and m7_1 its mirror. Their argument passes
through the fifteen-crossing census knot K15n81556. Wang and Zhang,
"A remark on the counterexample to the unknotting number conjecture",
arXiv:2507.14265, then noticed that the two diagrams of K15n81556 appearing
in that argument do not represent the same knot but a chiral knot and its
mirror image, and that the Jones polynomial shows this.

This script does three things. It reproduces the Wang and Zhang chirality
computation. It surveys chirality across the built-in knot table. And it
runs the crossing-change search on 7_1 # m7_1 to show, concretely, why the
Brittenham-Hermiller bound is not something brute force finds: the search is
confined to one diagram, and the economical route is not visible there.

Usage
-----
    python3 examples/knot_chirality.py
    python3 examples/knot_chirality.py --search 3      # slow, see below
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import knots as K


def wang_zhang():
    print("=" * 72)
    print("Wang and Zhang, arXiv:2507.14265: K15n81556 is chiral")
    print("=" * 72)
    r = K.chirality_report("K15n81556")
    print("  crossings          {}".format(r["crossings"]))
    print("  writhe             {:+d}".format(r["writhe"]))
    print("  determinant        {}".format(r["determinant"]))
    print("  V(K)               {}".format(r["jones"]))
    print("  V(mirror K)        {}".format(r["jones_mirror"]))
    print("  V(mirror) = V(1/t) {}".format(r["mirror_is_inverse"]))
    print("  palindromic        {}".format(r["palindromic"]))
    print("  => chiral          {}".format(r["chiral"]))
    print()
    print("  The two polynomials differ, so the knot and its mirror are not")
    print("  the same knot. Note that the determinant does not see this: it")
    print("  is 39 for both. Chirality needs an invariant that is not")
    print("  symmetric under t -> 1/t.")


def survey():
    print()
    print("=" * 72)
    print("Chirality across the table")
    print("=" * 72)
    print("  {:<11} {:>3} {:>7} {:>5} {:>7}  {}".format(
        "knot", "cr", "writhe", "det", "chiral", "u (quoted)"))
    print("  " + "-" * 68)
    for nm in sorted(K.KNOTS):
        k = K.from_name(nm)
        u = K.UNKNOTTING.get(nm)
        print("  {:<11} {:>3} {:>+7d} {:>5} {:>7}  {}".format(
            nm, len(k), k.writhe(), k.determinant(),
            "yes" if k.is_chiral() else "-",
            u if u is not None else "<= 2 (BH)"))
    undetected = [nm for nm in K.KNOTS if not K.from_name(nm).is_chiral()]
    print()
    print("  Not detected as chiral: {}".format(", ".join(sorted(undetected))))
    print("  The Jones test is sufficient but not necessary, so this means")
    print("  'not detected', not 'amphichiral'. 4_1 and 6_3 happen to be")
    print("  genuinely amphichiral.")


def additivity(max_changes):
    print()
    print("=" * 72)
    print("Brittenham and Hermiller, arXiv:2506.24088: additivity fails")
    print("=" * 72)
    a = K.additivity_report()
    print("  7_1 # m7_1 has {} crossings and {} component".format(
        a["sum_crossings"], a["sum_components"]))
    print("  V(7_1)       {}".format(a["jones_left"]))
    print("  V(m7_1)      {}".format(a["jones_right"]))
    print("  V of the sum multiplies: {}".format(a["jones_multiplicative"]))
    print()
    print("  u(7_1) + u(m7_1) = {} (quoted, Kronheimer-Mrowka)".format(
        a["u_sum_naive"]))
    print("  u(7_1 # m7_1)   <= {} (quoted, {})".format(
        a["u_sum_upper_bound_BH"], a["reference"]))
    print("  so the unknotting number is not additive.")

    if max_changes is None:
        print()
        print("  Pass --search N to run the crossing-change search on the")
        print("  fourteen-crossing diagram. It costs C(14,N) * 2^14 bracket")
        print("  evaluations and will not find their bound; see below.")
        return

    total = K.from_name("7_1").connected_sum(K.from_name("7_1").mirror())
    print()
    print("  searching the standard 14-crossing diagram, up to {} changes..."
          .format(max_changes))
    t0 = time.time()
    res = K.unknotting_search(total, max_changes=max_changes)
    print("  tried {} subsets in {:.1f}s -> {}".format(
        res["tried"], time.time() - t0,
        "found {}".format(res["found"]) if res["found"] is not None
        else "nothing"))
    print()
    print("  A negative result here is the expected one. The search works in")
    print("  a single fixed diagram, while the unknotting number is a minimum")
    print("  over all diagrams, and the whole difficulty of the Brittenham-")
    print("  Hermiller result is that the five-change route is invisible in")
    print("  the obvious picture. That is why they needed SnapPy and a route")
    print("  through K15n81556, and why Wang and Zhang's correction to the")
    print("  chirality of that intermediate knot mattered.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--search", type=int, default=None, metavar="N",
                    help="run the crossing-change search up to N changes")
    args = ap.parse_args()

    wang_zhang()
    survey()
    additivity(args.search)


if __name__ == "__main__":
    main()
