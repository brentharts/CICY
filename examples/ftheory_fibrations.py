#!/usr/bin/env python3
"""
Which CICY threefolds fibre in elliptic curves, and which of those are
F-theory backgrounds.

    python3 examples/ftheory_fibrations.py
    python3 examples/ftheory_fibrations.py --limit 500
    python3 examples/ftheory_fibrations.py --conf '[[2,3],[2,3]]'
    python3 examples/ftheory_fibrations.py --no-list

F-theory needs an elliptically fibred Calabi-Yau. The CICY list is 7890
Calabi-Yau threefolds, so the obvious question is how many of them are usable,
and this script answers it by the criterion of Anderson, Gao, Gray and Lee: a
fibration is *obvious* when the configuration matrix can be reordered into

    [ F  0 ]
    [ C  B ]

with the top rows zero outside the left columns. Then the top block is a
complete intersection in its own ambient factors, of dimension one, hence a
Calabi-Yau one-fold, hence an elliptic curve, and the manifold fibres over the
bottom block.

The interesting part of the answer is the gap between "fibres in elliptic
curves" and "is an F-theory background", and the script is organised around
making that gap visible:

    1  one manifold      the (3,3) hypersurface in P^2 x P^2, worked through
    2  the sweep         how much of the list is obviously fibred
    3  the gap           why being fibred is not enough

A genus-one fibration has elliptic curves as fibres but need not have a
section, and F-theory in the naive sense wants one: the section is where the
axio-dilaton is read off and where the Weierstrass form comes from. The
manifold in section 1 fibres over P^2 in plane cubics with no marked point,
and its Hodge numbers are (2, 83). The Weierstrass model over the same base
has (2, 272). Same base, same fibre type, different manifolds -- and only the
second is what an F-theory model means by an elliptic fibration over P^2.
"""

import argparse
import ast
import json
import os
import sys
import time
from collections import Counter

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from pyCICY import CICY
from pyCICY.theories import ftheory as FT

DEFAULT_LIST = os.path.join(__file__.rsplit("/", 2)[0], "data", "cicylist.json")


def banner(text):
    print("\n" + text)
    print("-" * len(text))


# ---------------------------------------------------------------------------


def one_manifold(conf):
    banner("1. One manifold, worked through")

    print("  configuration %s" % (conf,))
    fs = FT.obvious_fibrations(conf)
    if not fs:
        print("  no obvious genus-one fibration")
    for k, f in enumerate(fs):
        print("\n  fibration %d" % (k + 1))
        print("    fibre rows %-12s columns %s" % (f["fibre_rows"],
                                                   f["fibre_cols"]))
        print("    fibre      %-24s dimension %d"
              % (f["fibre"], f["fibre_dim"]))
        print("    base       %-24s dimension %d   %s"
              % (f["base"], f["base_dim"], f["base_name"] or ""))
        rows = f["fibre"]
        print("    the fibre is Calabi-Yau: row sums %s against dimensions+1 %s"
              % ([sum(r[1:]) for r in rows], [r[0] + 1 for r in rows]))

    k3 = FT.obvious_fibrations(conf, fibre_dim=2)
    print("\n  and %d obvious K3 or abelian surface fibration%s"
          % (len(k3), "" if len(k3) == 1 else "s"))
    for f in k3[:3]:
        print("    fibre %-24s over %s" % (f["fibre"],
                                           f["base_name"] or f["base"]))

    try:
        X = CICY(list(conf))
        print("\n  Hodge numbers, from the rest of the package: "
              "h^{1,1} = %d, h^{2,1} = %d, chi = %d"
              % (int(X.h[2]), int(X.h[1]), X.euler_characteristic()))
    except Exception as e:                                       # noqa: BLE001
        print("\n  (Hodge numbers not computed: %s)" % e)


def sweep(path, limit=None):
    banner("2. The sweep")

    if not os.path.exists(path):
        print("  %s not found. Fetch the published list first:" % path)
        print("      make data")
        return None

    with open(path) as fh:
        entries = json.load(fh)["entries"]
    if limit:
        entries = entries[:limit]

    print("  scanning %d configurations for obvious fibrations\n"
          % len(entries))

    t0 = time.time()
    n_ell = n_k3 = 0
    ell_counts = Counter()
    bases = Counter()
    fibres = Counter()
    unfibred = []
    for e in entries:
        conf = e["conf"]
        fs = FT.obvious_fibrations(conf)
        if fs:
            n_ell += 1
            ell_counts[min(len(fs), 20)] += 1
            for f in fs:
                bases[f["base_name"] or "with defining equations"] += 1
                fibres[_fibre_name(f["fibre"])] += 1
        else:
            unfibred.append(e)
        if FT.obvious_fibrations(conf, fibre_dim=2):
            n_k3 += 1
    dt = time.time() - t0

    print("  %d of %d have an obvious genus-one fibration   (%.1f%%)"
          % (n_ell, len(entries), 100.0 * n_ell / len(entries)))
    print("  %d of %d have an obvious K3 fibration          (%.1f%%)"
          % (n_k3, len(entries), 100.0 * n_k3 / len(entries)))
    print("  %d have neither" % sum(
        1 for e in unfibred if not FT.obvious_fibrations(e["conf"], 2)))
    print("  (%.1f s)" % dt)

    print("\n  fibre types found, by number of fibrations:")
    named = [n for n in fibres if not n.startswith("[")]
    for name in sorted(named, key=lambda n: -fibres[n]):
        print("    %-36s %8d" % (name, fibres[name]))
    bigger = sum(c for n, c in fibres.items() if n.startswith("["))
    print("    %-36s %8d" % ("larger complete intersection curves", bigger))
    print("""
    The four named ones are the complete intersection elliptic curves that
    exist at all: a plane cubic, a (2,2) curve in P^1 x P^1, the intersection
    of two quadrics in P^3, and the (1,1,1) pair in P^1 x P^1 x P^1. Anything
    longer is one of those with further splittings folded in.""")

    print("\n  bases found:")
    for name, c in bases.most_common(8):
        print("    %-36s %8d" % (name, c))

    print("\n  a few of the manifolds with no obvious elliptic fibration:")
    for e in unfibred[:6]:
        print("    #%-5d h^{1,1}=%-3d h^{2,1}=%-3d  %s"
              % (e["num"], e["h11"], e["h21"], e["conf"]))
    if not unfibred:
        print("    none")

    if len(entries) == 7890:
        print("""
  Anderson, Gao, Gray and Lee report 7837 of the 7890 CICY threefolds
  admitting an obvious genus-one fibration. The count above is theirs, from a
  criterion implemented here out of the definition rather than from their
  tables: reorder the rows, look for a zero block, check the dimension of what
  is left.""")
    else:
        print("""
  This was a partial scan. The full list gives 7837 of 7890, which is the
  count Anderson, Gao, Gray and Lee report; run without --limit to reproduce
  it.""")
    return {"elliptic": n_ell, "k3": n_k3, "total": len(entries)}


def _fibre_name(fibre):
    """Name the four complete intersection elliptic curves.

    There are exactly four Calabi-Yau one-folds that are complete
    intersections in a product of projective spaces, and every obvious
    elliptic fibration here has one of them, or a longer configuration
    reducing to one, as its fibre.
    """
    dims = sorted(r[0] for r in fibre)
    ncol = len(fibre[0]) - 1
    if dims == [2] and ncol == 1:
        return "P^2[3], a plane cubic"
    if dims == [1, 1] and ncol == 1:
        return "P^1 x P^1[(2,2)]"
    if dims == [3] and ncol == 2:
        return "P^3[2,2]"
    if dims == [1, 1, 1] and ncol == 2:
        return "P^1 x P^1 x P^1[(1,1,1),(1,1,1)]"
    return "%s, %d equations" % (dims, ncol)


def the_gap():
    banner("3. Fibred is not the same as being an F-theory background")

    print("""\
Two things separate the manifolds found above from the Weierstrass models that
six-dimensional F-theory is built on.

The first is the section. An obvious fibration gives elliptic curves as
fibres, but nothing marks a point on them, and without a section there is no
Weierstrass form and no place to read off the axio-dilaton. Genus-one
fibrations without a section are perfectly good string backgrounds -- they
give discrete gauge symmetry rather than nothing -- but they are not the
naive dictionary.

The second is that the Weierstrass model is usually not a CICY at all. Its
fibre is a hypersurface in the weighted projective space P^{2,3,1}, which is
not a product of projective spaces, so it does not appear in the list.

The (3,3) hypersurface in P^2 x P^2 shows both at once. It fibres over P^2 in
plane cubics. So does the Weierstrass model. They are different manifolds:
""")
    X = CICY([[2, 3], [2, 3]])
    w = FT.FTheory6D("P2")
    print("    %-40s h^{1,1} = %2d   h^{2,1} = %3d   chi = %5d"
          % ("(3,3) in P^2 x P^2, a CICY",
             int(X.h[2]), int(X.h[1]), X.euler_characteristic()))
    print("    %-40s h^{1,1} = %2d   h^{2,1} = %3d   chi = %5d"
          % ("Weierstrass over P^2, not a CICY",
             w.hodge_numbers()[0], w.hodge_numbers()[1],
             w.euler_characteristic()))
    print("""
The cubic in the first has no marked point; it meets the section of the second
in three points, which is why the first is a three-section rather than a
section. The 272 is the complex structure moduli of a Weierstrass model, which
is the count of f in H^0(-4K) and g in H^0(-6K) modulo automorphisms, and the
83 is not that count.

What the F-theory side does with a base, once it has one:
""")
    for spec in ["P2", "F0", "F3", "F12"]:
        m = FT.FTheory6D(spec)
        s = m.spectrum()
        print("    over %-4s  T=%d  algebra %-8s  V=%-4d H=%-4d  (h11,h21)=%s"
              % (spec, s["T"], m.gauge_group(), s["V"], s["H"],
                 m.hodge_numbers()))
    print("""
None of that needed the threefold as a CICY, or as anything but a base surface
and the anomaly conditions. That is the sense in which F-theory sits beside
this package's machinery rather than on top of it: it shares the interface in
pyCICY.theories, and it shares the discipline about what is exact, but the
geometry it wants is not in the list.""")


# ---------------------------------------------------------------------------


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--conf", default="[[2,3],[2,3]]",
                   help="configuration matrix for section 1")
    p.add_argument("--list", default=DEFAULT_LIST,
                   help="path to the published CICY list")
    p.add_argument("--limit", type=int, default=None,
                   help="scan only the first N configurations")
    p.add_argument("--no-list", action="store_true",
                   help="skip the sweep over the published list")
    a = p.parse_args()

    one_manifold(ast.literal_eval(a.conf))
    if not a.no_list:
        sweep(a.list, a.limit)
    the_gap()
    print()


if __name__ == "__main__":
    main()
