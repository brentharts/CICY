#!/usr/bin/env python3
"""
One mirror operation, four kinds of object.

Three of the mirror operations in this package have the same shape. Each is
an involution, each swaps a pair of integer invariants, and each preserves
the sum or span of that pair:

    knot                  mirror image     degrees of V(t)     span of V
    reflexive polygon     polar duality    (#dP, #dP*)         12
    Calabi-Yau threefold  mirror symmetry  (h^{1,1}, h^{2,1})  h^11 + h^21

The fourth, the quantized mirror curve, is included because it is the case
where the analogy fails: reflecting its Newton polygon leaves the spectrum
exactly unchanged, so no spectral invariant can see that involution at all.
What the spectrum does see is bipartiteness, and that is a different
property, as this script demonstrates by exhibiting all four combinations of
the two among the sixteen reflexive polygons.

The script closes with the question the package is in a position to answer
directly: is the published list of CICY threefolds closed under mirror
symmetry? It is not, and the reason is visible in the Hodge data.

References
----------
    Wang and Zhang, arXiv:2507.14265 (chirality of K15n81556)
    Brittenham and Hermiller, arXiv:2506.24088 (unknotting additivity)
    Sugimoto, arXiv:1701.01561 (local Calabi-Yau and 2d lattices)
    Candelas, Dale, Lutken, Schimmrigk, Nucl. Phys. B298 (1988) 493

Usage
-----
    python3 examples/chirality_zoo.py
    python3 examples/chirality_zoo.py --domain knot
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pyCICY import chirality as C
from pyCICY import knots as K
from pyCICY import toric as T


def law():
    print("=" * 78)
    print("The swap-and-preserve law")
    print("=" * 78)
    for key in ("knot", "polygon", "cicy", "curve"):
        d = C.DOMAINS[key]
        print("  {}".format(key))
        print("    involution   {}".format(d["involution"]))
        print("    swapped pair {}".format(d["pair"] or "-- none --"))
        print("    preserved    {}".format(d["preserved"]))
        print("    detector     {}".format(d["detector"]))
        print()


def table(domain):
    print()
    print("=" * 78)
    print("Objects and their mirrors")
    print("=" * 78)
    records = C.survey()
    if domain:
        records = [r for r in records if r["domain"] == domain]
    print(C.format_survey(records))
    fixed = [r["name"] for r in records if r["fixed"]]
    print()
    print("  fixed by their involution: {}".format(
        ", ".join(str(f) for f in fixed) or "none"))


def knot_note():
    print()
    print("-" * 78)
    print("Why the preserved quantity is meaningful, on the knot side")
    print("-" * 78)
    print("  The span of the Jones polynomial is not an accident of the")
    print("  normalisation. By the Kauffman-Murasugi-Thistlethwaite theorem it")
    print("  equals the crossing number exactly for alternating knots:")
    print()
    non_alt = {"8_19", "8_20", "K15n81556"}
    for nm in sorted(K.KNOTS):
        r = C.knot_chirality(nm)
        tag = "non-alternating" if nm in non_alt else "alternating"
        rel = "==" if r["preserved"] == r["crossings"] else "< "
        print("    {:<11} span {:>3} {} {:>3} crossings   {}".format(
            nm, r["preserved"], rel, r["crossings"], tag))


def curve_note():
    print()
    print("-" * 78)
    print("Where the analogy breaks: quantized mirror curves")
    print("-" * 78)
    print("  Reflecting the Newton polygon leaves the spectrum exactly where")
    print("  it was, so chirality() reports detected=None rather than False.")
    print("  Bipartiteness is what the spectrum actually sees, and the two")
    print("  are logically independent -- all four combinations occur:")
    print()
    print("    {:<7} {:>18} {:>18}".format(
        "polygon", "fixed by reflection", "spectrally chiral"))
    print("    " + "-" * 45)
    combos = {}
    for nm in T.NAMED:
        r = C.curve_chirality(nm)
        combos.setdefault((r["fixed"], r["spectrally_chiral"]), []).append(nm)
        print("    {:<7} {:>18} {:>18}".format(
            nm, "yes" if r["fixed"] else "-",
            "yes" if r["spectrally_chiral"] else "-"))
    print()
    for (fixed, chiral), names in sorted(combos.items()):
        print("    fixed={:<5} chiral={:<5}  {}".format(
            str(fixed), str(chiral), ", ".join(names)))


def cicy_note():
    print()
    print("=" * 78)
    print("Is the CICY threefold list closed under mirror symmetry?")
    print("=" * 78)
    r = C.cicy_list_chirality()
    print("  manifolds in the list        {}".format(r["n_manifolds"]))
    print("  entries with no Hodge data   {} (sentinel zeros, products)".format(
        r["n_degenerate"]))
    print("  distinct (h11, h21) pairs    {}".format(r["n_pairs"]))
    print("  pairs whose mirror is present {}".format(r["n_pairs_with_partner"]))
    print("  pairs with no mirror partner  {}".format(
        r["n_pairs_without_partner"]))
    print("  self-mirror pairs             {}".format(r["self_mirror_pairs"]))
    print("  non-trivial mirror pairs      {}".format(
        r["nontrivial_mirror_pairs"] or "none"))
    print()
    print("  h11 ranges over {}, h21 over {}, Euler over {}.".format(
        r["h11_range"], r["h21_range"], r["euler_range"]))
    print()
    print("  So the answer is no, and emphatically. The only pairs with a")
    print("  partner are the ones that are their own partner. A mirror would")
    print("  need h11 as large as 101, and the construction never produces")
    print("  h11 above 19, which is the same statement as every Euler")
    print("  characteristic in the list being non-positive. The mirror of a")
    print("  CICY is essentially never a CICY, which is why chirality.mirror")
    print("  returns Hodge data here and not a configuration matrix.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--domain", choices=["knot", "polygon", "curve", "cicy"],
                    default=None, help="restrict the object table")
    args = ap.parse_args()

    law()
    table(args.domain)
    if args.domain in (None, "knot"):
        knot_note()
    if args.domain in (None, "curve"):
        curve_note()
    if args.domain in (None, "cicy"):
        cicy_note()


if __name__ == "__main__":
    main()
