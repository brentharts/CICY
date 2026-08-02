r"""
pyCICY.additivity -- how invariants behave under composition of splits.

Motivation
----------
Brittenham and Hermiller, "Unknotting number is not additive under connected
sum", arXiv:2506.24088, exhibit knots with

    u(K_1 \# K_2) < u(K_1) + u(K_2),

settling Kirby problem 1.69(B) negatively. Their example is
u(7_1 \# \bar{7_1}) <= 5 < 6. Unknotting number is a *minimal move count*:
the fewest crossing changes taking a knot to the unknot. The surprise is
that composing two objects can make the total complexity smaller than the sum
of the parts, so the invariant of a composite is not determined by its
factors.

The split web of :mod:`pyCICY.cicylist` has the same shape of structure: a
family of objects, an elementary move (the P^1 split), and distinguished
"trivial" objects (the five single-equation seeds). It is natural to ask
whether the analogous complexity measures misbehave in the analogous way.
This module answers that, and the answer is instructive in both directions.

What is rigid
-------------
The obvious analogue of unknotting number is the **split depth**: the fewest
splits needed to reach a configuration from a seed. It cannot misbehave. A
split adds exactly one defining equation and a contraction removes exactly
one, and the seeds are precisely the configurations with one equation, so

    split depth = K - 1

where K is the number of equations, independent of the route taken. Every
contraction path has the same length, greedy contraction is automatically
optimal, and there is no room for a Bernhard-Jablan style counterexample.
:func:`split_depth` records this, and :func:`contraction_paths` verifies it
rather than assuming it.

The same rigidity applies to the total node count. Along any chain of splits
from X_D to X_R, the individual node counts sum to

    sum_i N_i = ( chi(X_R) - chi(X_D) ) / 2 ,

because the Euler characteristics telescope, and chi is an invariant of the
endpoints alone. The total is therefore forced.

What is not
-----------
The *decomposition* of that total is not forced. Two chains of splits joining
the same endpoints can pass through different intermediate node counts:
(12, 36) and (16, 32) both occur between the same pair, and both sum to 48.
:func:`decompositions` extracts these and :func:`survey` counts them.

So the contrast with the knot-theoretic situation is precise, and it runs the
other way. There, the total is not determined by the parts. Here the total is
determined and only the parts vary. In both settings the moral is the same:
a quantity that looks like it should be built up additively from elementary
steps need not be, and which of the two failure modes occurs is a fact about
the structure and not something to be assumed.
"""

import collections

from . import transitions as T
from .cicylist import SEEDS, web_nodes

__all__ = [
    "split_depth", "contraction_paths", "decompositions", "survey",
    "seed_keys",
]


def seed_keys():
    """Canonical keys of the five single-equation seed configurations."""
    return {T.canonical_key(s) for s in SEEDS}


def split_depth(conf):
    """Fewest splits from a seed, which equals K - 1.

    A split adds one equation and a contraction removes one, and the seeds
    are exactly the configurations with a single equation, so the move count
    is fixed by the shape of the matrix and no shorter route exists.

    >>> split_depth([[4, 5]])
    0
    >>> split_depth([[1, 1, 1], [4, 1, 4]])
    1
    """
    M = T._as_matrix(conf)
    return M.shape[1] - 2


def contraction_paths(conf, max_nodes=20000):
    """Lengths of every contraction path from ``conf`` down to a seed.

    Returns ``{"min", "max", "greedy", "stuck", "paths"}``. ``stuck`` counts
    branches that reach a configuration with no contractible row that is not
    a seed, which would be the analogue of a greedy unknotting sequence that
    fails to terminate at the unknot. ``min`` should equal
    :func:`split_depth` and ``max`` should equal it too; the function
    computes both so that the claim can be checked rather than trusted.
    """
    seeds = seed_keys()
    counter = [0]

    def walk(current):
        counter[0] += 1
        if counter[0] > max_nodes:
            raise ValueError("contraction search exceeded max_nodes=%d"
                             % max_nodes)
        key = T.canonical_key(current)
        if key in seeds:
            return [0]
        options = T.is_contractible(current)
        if not options:
            return []                      # stuck: not a seed, cannot contract
        lengths = []
        for row, _ in options:
            child = T.contract(current, row=row)
            lengths.extend(n + 1 for n in walk(child))
        return lengths

    lengths = walk([list(map(int, r)) for r in conf])
    greedy_chain = _greedy_contract(conf, seeds)

    return {
        "min": min(lengths) if lengths else None,
        "max": max(lengths) if lengths else None,
        "greedy": greedy_chain,
        "stuck": not lengths,
        "paths": len(lengths),
    }


def _greedy_contract(conf, seeds):
    """Length of the path taken by always contracting the first option."""
    current = [list(map(int, r)) for r in conf]
    steps = 0
    while T.canonical_key(current) not in seeds:
        options = T.is_contractible(current)
        if not options:
            return None                    # greedy got stuck
        current = T.contract(current)
        steps += 1
    return steps


def decompositions(web, key, max_paths=64):
    """Per-step node counts along every split chain from a seed to ``key``.

    Returns a set of tuples, each the sorted multiset of node counts N_i of
    the splits along one chain. All of them sum to the same value, since the
    Euler characteristics telescope, but the multisets themselves need not
    agree.
    """
    seeds = seed_keys()
    incoming = collections.defaultdict(list)
    for edge in web["edges"]:
        incoming[edge["child"]].append((edge["parent"], edge["nodes"]))

    cache = {}

    def walk(node):
        if node in seeds:
            return {()}
        if node in cache:
            return cache[node]
        out = set()
        cache[node] = out                  # guards against revisiting
        for parent, n in incoming[node]:
            for prefix in walk(parent):
                out.add(tuple(sorted(prefix + (n,))))
                if len(out) >= max_paths:
                    return out
        return out

    return walk(key)


def survey(web, max_paths=64):
    """Measure path dependence of the node count across a whole web.

    Returns counts of how many configurations admit more than one
    decomposition, how many have decompositions whose *totals* disagree
    (which should be none, since the total is fixed by chi), and the largest
    spread seen in a single step.
    """
    seeds = seed_keys()
    checked = 0
    multiple = 0
    total_mismatch = []
    spread = 0
    examples = []

    for key, rec in web["nodes"].items():
        if key in seeds:
            continue
        ms = decompositions(web, key, max_paths=max_paths)
        if not ms:
            continue
        checked += 1
        totals = {sum(m) for m in ms}
        if len(totals) > 1:
            total_mismatch.append({"conf": rec["conf"], "totals": sorted(totals)})
        if len(ms) > 1:
            multiple += 1
            flat = [x for m in ms for x in m]
            spread = max(spread, max(flat) - min(flat))
            if len(examples) < 5:
                examples.append({
                    "conf": rec["conf"],
                    "decompositions": sorted(ms)[:4],
                    "total": sorted(totals)[0],
                })

    return {
        "checked": checked,
        "multiple_decompositions": multiple,
        "total_mismatches": total_mismatch,
        "max_single_step_spread": spread,
        "examples": examples,
    }
