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


def monad_act(order):
    banner("Aside: the other construction")
    print("""\
A sum of line bundles is not the only rank-5 bundle available. A monad

    0 -> V -> B -> C -> 0,   B, C sums of line bundles,

has the same Chern character arithmetic -- ch(V) = ch(B) - ch(C) -- so its
index is just as free, and scan_monads() searches them with the same
cost-ordered filters. What differs is the cohomology: it is not determined by
the degrees, because the ranks of the maps H^q(B) -> H^q(C) depend on the
morphism. cohomology_bounds() returns intervals instead of numbers.""")

    from pyCICY import CICY
    X7833 = CICY([[2, 2, 1], [3, 1, 3]])
    print("\n  Two things the degrees do settle, and both are cheap filters:")
    M = B.Monad(CICY([[4, 5]]), [[1], [1], [1]], [[0], [3]])
    try:
        M.cohomology_bounds()
        print("     (unexpected: no exception)")
    except B.NotABundle as e:
        print("     %s" % str(e).split(".")[0])
    print("""
     and imposing h^0 = h^3 = 0 forces one particular rank for
     H^2(B) -> H^2(C), which need not be attainable -- for about one monad in
     ninety it is not, and no stable bundle exists however the coefficients
     are chosen.
""")
    t0 = time.time()
    found = B.scan_monads(X7833, rank=4, nC=2, charge=3, generations=3,
                          symmetry_order=order, limit=6, max_seconds=20)
    print("  CICY 7833, rank 4, charge 3, |Gamma| = %d: %d monads  [%.1fs]"
          % (order, len(found), time.time() - t0))
    trivial = 0
    for Bc, Cc in found[:4]:
        triv = sum(1 for x in Bc if not any(x))
        trivial += 1 if triv else 0
        print("     B=%s" % Bc)
        print("     C=%s   (%d trivial summands in B)" % (Cc, triv))
    print("""
Every one of them has a trivial summand in B. That is not bad luck: O_X has all
charges zero, so every summand of C exceeds it somewhere and positivity lets it
through, and in this box there is nothing else. A V with a trivial summand has
a smaller structure group than advertised, so none of these is a model. Nothing
in scan_monads rejects them, because telling costs more than the filter is
worth -- the keep= argument is where such a predicate belongs.

Two arithmetic facts before choosing a box. On the quintic every realisable
monad index is a multiple of five, so three generations is unreachable at any
charge and rank. On the tetraquadric ind = -3 is unreachable for the same
parity reason that blocks it for sums of line bundles, while -6 is not.""")


def breaking_act(order):
    banner("Stage 6: down to the Standard Model")
    from pyCICY import breaking as B

    print("""\
Everything above is an SU(5) grand unified spectrum on the covering space, and
"three generations after quotienting by |Gamma| = %d" is so far a division
rather than a model. What turns SU(5) into SU(3) x SU(2) x U(1) is a Wilson
line on X/Gamma.""" % order)

    print("\n  SU(5) branching, from the hypercharge generator:")
    for rep in ("10", "5bar"):
        for name, dims, Y, mult in B.branching(rep):
            print("     %-5s -> %-11s Y = %-6s (%d states)"
                  % (rep, name, Y, mult))
    ok, _ = B.verify_against_flavor()
    print("  agrees with flavor.SM_HYPERCHARGES up to conjugation: %s" % ok)
    print("  Tr(Y) over 10 + 5bar: %s" % B.anomaly_trace_of_generation())

    print("""
A Wilson line in the hypercharge direction is W = diag(a,a,a,b,b) with
a^3 b^2 = 1. Its commutant is the Standard Model group when a != b, and all of
SU(5) when a = b -- in which case W is central and breaks nothing. Counting
the ones that work for Z_n gives a closed form:
""")
    print("     {:>4} {:>12} {:>16}".format("n", "enumerated", "n - gcd(n,5)"))
    for n in range(1, 11):
        print("     {:>4} {:>12} {:>16}".format(
            n, len(B.wilson_lines(n)), B.wilson_line_count(n)))
    print("""
It vanishes exactly for n = 1 and n = 5. So a Z_5 quotient divides the
generation count by five and still cannot break the GUT group, while Z_2 --
the order these tetraquadric models need -- is the smallest that can. That is
not automatic and is worth checking before trusting a scan's |Gamma|.""")

    r = B.worked_example()
    print("\n  A consistent quotient spectrum, |Gamma| = 2, W = (0,1):")
    for k, v in sorted(r["spectrum"].items(), key=str):
        print("     %-7s %-11s Y = %-6s  %d multiplets" % (k[0], k[1], k[2], v))
    print("  generations %d (expected %d), anomaly %s"
          % (r["generations"], r["expected_generations"], r["anomaly"]))
    print("""
The colour triplets are gone while the Higgs doublets survive: doublet-triplet
splitting, which here is a condition on Gamma-charges rather than a tuning.

One caveat, and it is the whole boundary of this module. The Gamma-charges
above are *chosen*, not derived. They encode an equivariant structure -- a
lift of the Gamma action to the total space of the bundle -- and different
lifts give different spectra on identical topological data. Nothing in this
package computes them, and project() takes them as an argument rather than
inventing them. What it does certify is that the consistency conditions close:
the generation count against the index, and the vanishing of the anomaly.""")


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
              "the box, not about the manifold: the search was truncated by "
              "the %gs budget, not exhausted. Raise --budget, or widen "
              "--charge beyond %d." % (args.budget, args.charge))

    # These two do not depend on a model having been found above: the monad
    # aside is about a different construction entirely, and the breaking act
    # is group theory. Running them only on success made them invisible
    # whenever the budget truncated the scan, which is most of the time.
    monad_act(args.order)
    breaking_act(args.order)
    return 0


if __name__ == "__main__":
    sys.exit(main())
