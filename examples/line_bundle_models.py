#!/usr/bin/env python3
"""
Heterotic line bundle standard models on a CICY.

    python3 examples/line_bundle_models.py
    python3 examples/line_bundle_models.py --charge 2 --order 2 --budget 30
    python3 examples/line_bundle_models.py --conf '[[2,2,1],[3,1,3]]'

This walks the pipeline of :mod:`pyCICY.bundles` in the order the filters are
meant to be applied, which is strictly increasing cost:

    c_1(V) = 0  ->  index  ->  anomaly  ->  poly-stability  ->  cohomology

and prints what each stage removes. The point of the ordering is that the
last stage is thousands of times more expensive than the first, so a scan
that reaches it with the wrong candidates never finishes. Every stage here is
boxed by an explicit charge range and a wall clock budget; nothing runs
unbounded.

The default is the tetraquadric with |Gamma| = 2, where three generations
downstairs means ind(V) = -6 upstairs.
"""

import argparse
import ast
import sys
import time

import numpy as np

sys.path.insert(0, __file__.rsplit("/", 2)[0])

from pyCICY import CICY
from pyCICY import bundles as B


def banner(text):
    print("\n" + text)
    print("-" * len(text))


def why_the_order(X):
    banner("Why the filters run in this order")
    print("""\
Every stage is a constraint on the same integer vectors, but they do not cost
the same. On this manifold, per candidate bundle:

    c_1(V) = 0            integer addition
    index                 one tensor contraction
    anomaly               one tensor contraction
    poly-stability        ~0.03 ms as an exact sign test on subsets of the
                          summands, ~30 ms if the numerical search has to run
    cohomology            one line_co per summand, and per summand of
                          Lambda^2 V and V (x) V*: 35 line bundles for a
                          rank 5 model

The sign test is what makes the stability stage usable. At a common zero of
all the slopes every partial sum of slopes vanishes too, so if any subset of
the summands has a slope form with no negative entry, no such point exists.
That is exact, it is 2^n - 2 sign tests, and it inherits to pairs, which is
where scan() applies it -- before the sums are assembled rather than after.""")


def topology(X, charge, gens, order, budget):
    banner("Stage 1-3: c_1, index, anomaly")
    t0 = time.time()
    models = B.scan(X, rank=5, charge=charge, generations=gens,
                    symmetry_order=order, require_anomaly=True,
                    limit=200000, max_seconds=budget)
    print("charge box |k^r| <= %d, target ind(V) = -%d" % (charge,
                                                           gens * order))
    print("%d models passing c_1 = 0, the index and the anomaly   [%.1fs]"
          % (len(models), time.time() - t0))
    if models:
        print("first three:")
        for S in models[:3]:
            print("   ", S)
    return models


def stability(X, models, budget):
    banner("Stage 4: poly-stability")
    d = np.array(X.triple_intersection())

    t0 = time.time()
    survivors = [S for S in models if not B.slope_subsets_definite(d, S)]
    t_sign = time.time() - t0
    print("sign obstruction: %d -> %d   [%.2fs, %.3f ms each]"
          % (len(models), len(survivors), t_sign,
             1000 * t_sign / max(len(models), 1)))

    t0 = time.time()
    stable = []
    for S in survivors:
        if time.time() - t0 > budget:
            print("(numerical search stopped on its %ds budget)" % budget)
            break
        if B.stability_locus(X, S, tries=12)["found"]:
            stable.append(S)
    print("numerical search: %d found   [%.1fs]" % (len(stable),
                                                    time.time() - t0))
    return stable


def spectrum(X, models, order, limit=3):
    banner("Stage 5: cohomology and the SU(5) spectrum")
    if not models:
        print("nothing reached this stage.")
        return
    for S in models[:limit]:
        V = B.LineBundleSum(X, S)
        t0 = time.time()
        sp = V.su5_spectrum()
        loc = V.stability_locus(tries=40)
        print("\n" + "\n".join("    " + str(L) for L in S))
        print("  ind(V) = %-6s  (Leray route: %s)"
              % (sp["index"], V.index_from_cohomology()))
        print("  n(10)=%d  n(10-bar)=%d  n(5-bar)=%d  n(5)=%d  n(1)=%d"
              % (sp["n10"], sp["n10bar"], sp["n5bar"], sp["n5"], sp["n1"]))
        print("  generations upstairs = %d, downstairs of |Gamma|=%d = %s"
              % (sp["generations"], order,
                 sp["generations"] // order if sp["generations"] % order == 0
                 else "not integral"))
        print("  h^0(V) = %d, h^3(V) = %d  (both must vanish for a stable "
              "bundle of zero slope)" % (sp["h0"], sp["h3"]))
        print("  poly-stable at t = %s, residual %.1e"
              % (np.round(loc["t"], 4).tolist(), loc["residual"]))
        print("  index consistent with the cohomology: %s  [%.1fs]"
              % (sp["index_consistent"], time.time() - t0))


def cross_checks(X, S):
    banner("Cross-checks that share no code")
    V = B.LineBundleSum(X, S)
    h = V.cohomology()
    hd = V.dual().cohomology()
    print("  ind(V), intersection theory      %s" % V.index())
    print("  ind(V), Leray spectral sequence  %s" % V.index_from_cohomology())
    print("  ind(V), alternating sum of h^q   %s"
          % int(h[0] - h[1] + h[2] - h[3]))
    print("  Serre duality h^q(V) = h^(3-q)(V*): %s"
          % all(int(h[q]) == int(hd[3 - q]) for q in range(4)))
    print("""
The first two routes have nothing in common: one contracts triple
intersection numbers with the charges, the other runs the Leray spectral
sequence on each summand. The third computes the cohomology outright. Serre
duality is a constraint line_co has no knowledge of and must satisfy anyway.
This is the same discipline as node_validation elsewhere in the package.""")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    ap.add_argument("--conf", default="[[1,2],[1,2],[1,2],[1,2]]",
                    help="configuration matrix (default: the tetraquadric)")
    ap.add_argument("--charge", type=int, default=1,
                    help="scan box |k^r| <= charge (default 1)")
    ap.add_argument("--generations", type=int, default=3)
    ap.add_argument("--order", type=int, default=2,
                    help="|Gamma|, the order of the freely acting symmetry")
    ap.add_argument("--budget", type=float, default=20.0,
                    help="wall clock budget per stage, in seconds")
    ap.add_argument("--show", type=int, default=2,
                    help="how many spectra to compute")
    args = ap.parse_args(argv)

    conf = ast.literal_eval(args.conf)
    X = CICY(conf)
    banner("Manifold")
    print("  configuration %s" % conf)
    print("  h^{1,1} = %d, h^{2,1} = %d, chi = %d, favourable = %s"
          % (X.h[2], X.h[1], X.euler_characteristic(), X.fav))
    print("  c_2(TX) . J_r = %s" % list(map(int, X.second_chern())))

    why_the_order(X)
    models = topology(X, args.charge, args.generations, args.order,
                      args.budget)
    stable = stability(X, models, args.budget)
    spectrum(X, stable, args.order, limit=args.show)
    if stable:
        cross_checks(X, stable[0])
    else:
        print("\nNo poly-stable model in this box. That is a statement about "
              "the box, not about the manifold; try --charge 2.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
