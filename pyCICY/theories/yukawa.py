r"""
pyCICY.theories.yukawa -- selection rules and texture, exactly.

What is computable without representatives
------------------------------------------
:mod:`pyCICY.theories.heterotic` says the holomorphic Yukawa couplings of a
line bundle model are cup products

    H^1(V) x H^1(V) x H^1(Lambda^2 V) -> H^3(Lambda^3 V) = C

and that evaluating them needs explicit cohomology representatives, which this
package does not construct. That remains true of the *values*.

It is not true of the pattern. Which couplings can be non-zero at all is fixed
by two conditions that need no representatives, and between them they are most
of the phenomenology, because a texture of zeros is what distinguishes one
model's flavour structure from another's.

**The charge rule.** For V = O(L_1) + ... + O(L_5) with sum_a L_a = 0, the
matter lives in

    10_a          <->  H^1(X, L_a)
    5-bar_{ab}    <->  H^1(X, L_a (x) L_b),   a < b
    5_{ab}        <->  H^1(X, L_a^-1 (x) L_b^-1)

and a cup product lands in H^3 of the tensor product of the participating line
bundles. That is H^3(O_X) = C only when the charges cancel exactly. So

    up-type    10_a 10_b 5_{ab}         allowed for every a < b
    down-type  10_a 5-bar_{bc} 5-bar_{de}
               allowed only when {a,b,c,d,e} are all five distinct indices

the second being the epsilon_{abcde} of SU(5) showing up as a statement about
line bundle charges. There are exactly 15 down-type patterns: five choices of
``a`` times three ways of pairing the remaining four.

**The dimension rule.** A charge-allowed coupling is still absent if any of the
three groups is zero-dimensional -- there is no field to couple. Those
dimensions are exactly what :meth:`pyCICY.CICY.line_co` computes, so this
second filter is as exact as the first, and it is the one that actually
produces texture zeros in a given model.

What is still missing
---------------------
The non-zero entries. A coupling that passes both rules has some value in C,
and finding it needs the representatives. So this module reports a *pattern* of
allowed and forbidden couplings with the multiplicities attached, and never a
number pretending to be a coupling strength.

One further selection rule is also missing, and it is worth naming because it
is the one a quotient model needs. On X/Gamma with a Wilson line, a coupling
survives only if it is Gamma-invariant, which depends on the Gamma-charges of
the individual states. :mod:`pyCICY.equivariant` computes the equivariant
*index*, not the charge of each state, and for a free action the index is
equidistributed and so says nothing about which particular states pair up. That
refinement needs the same representatives.
"""

import itertools

import numpy as np

__all__ = [
    "up_type_patterns", "down_type_patterns", "charge_allowed",
    "texture", "viable_triples",
]


def up_type_patterns(rank=5):
    """Index patterns of the up-type coupling ``10_a 10_b 5_{ab}``.

    Allowed for every ``a < b``, since the charges cancel identically. Returns
    a list of ``(a, b, (a, b))``.
    """
    return [(a, b, (a, b)) for a, b in itertools.combinations(range(rank), 2)]


def down_type_patterns(rank=5):
    r"""Index patterns of ``10_a 5-bar_{bc} 5-bar_{de}``.

    Allowed only when all five indices are distinct, so this exists as stated
    only at rank 5, where it is the epsilon tensor. Returns a list of
    ``(a, (b, c), (d, e))`` with ``(b, c) < (d, e)`` so each pattern appears
    once.
    """
    if rank != 5:
        raise ValueError(
            "the down-type coupling 10 5-bar 5-bar uses epsilon_{abcde} and is "
            "an SU(5) statement; rank %d has a different structure" % rank)
    out = []
    for a in range(5):
        rest = [x for x in range(5) if x != a]
        for bc in itertools.combinations(rest, 2):
            de = tuple(x for x in rest if x not in bc)
            if bc < de:
                out.append((a, bc, de))
    return out


def charge_allowed(charges, indices):
    """Whether the total line bundle of a coupling is trivial.

    ``indices`` is a flat list of summand indices, with repeats meaning the
    corresponding charge enters more than once. The coupling can be non-zero
    only if they sum to zero, since otherwise the cup product lands in H^3 of a
    non-trivial line bundle rather than in H^3(O_X) = C.
    """
    k = np.asarray(charges, dtype=np.int64)
    total = np.zeros(k.shape[1], dtype=np.int64)
    for i in indices:
        total = total + k[i]
    return not np.any(total)


def texture(X, charges, kind="both", SpaSM=False):
    r"""
    The pattern of allowed and forbidden Yukawa couplings.

    For each index pattern, reports whether it is allowed by the charges, the
    dimensions of the three cohomology groups involved, and whether the
    coupling is therefore present. A pattern that passes the charge rule but
    has a zero-dimensional group is a **texture zero**: it is forbidden by the
    geometry, not by the symmetry, and that distinction is the interesting one.

    Parameters
    ----------
    X : CICY or configuration matrix
    charges : list of charge vectors
        The line bundle sum, as :class:`pyCICY.bundles.LineBundleSum` takes it.
    kind : {'up', 'down', 'both'}

    Returns
    -------
    dict with ``up`` and/or ``down``, each a list of records, plus ``summary``
    counts and ``h1``, the dimensions used.

    Notes
    -----
    No coupling *strength* is returned, and none is available: that needs
    explicit cohomology representatives. What is returned is exact.
    """
    from ..bundles import _as_cicy as as_cicy

    XX = as_cicy(X)
    k = np.asarray(charges, dtype=np.int64)
    rank = k.shape[0]

    cache = {}

    def h1(vec):
        key = tuple(int(x) for x in vec)
        if key not in cache:
            cache[key] = int(np.asarray(
                XX.line_co(list(key), SpaSM=SpaSM), dtype=int)[1])
        return cache[key]

    out = {"h1": {}, "rank": rank}

    if kind in ("up", "both"):
        recs = []
        for a, b, ab in up_type_patterns(rank):
            fields = [("10_%d" % a, k[a]),
                      ("10_%d" % b, k[b]),
                      ("5_%d%d" % ab, -(k[ab[0]] + k[ab[1]]))]
            dims = [h1(v) for _, v in fields]
            allowed = charge_allowed(
                np.vstack([k, -(k[ab[0]] + k[ab[1]])[None, :]]),
                [a, b, rank])
            recs.append({"pattern": ("10_%d" % a, "10_%d" % b,
                                     "5_%d%d" % ab),
                         "charge_allowed": bool(allowed),
                         "dimensions": dims,
                         "present": bool(allowed and all(d > 0 for d in dims))})
        out["up"] = recs

    if kind in ("down", "both"):
        recs = []
        for a, bc, de in down_type_patterns(rank):
            vecs = [k[a], k[bc[0]] + k[bc[1]], k[de[0]] + k[de[1]]]
            dims = [h1(v) for v in vecs]
            total = vecs[0] + vecs[1] + vecs[2]
            allowed = not np.any(total)
            recs.append({"pattern": ("10_%d" % a, "5bar_%d%d" % bc,
                                     "5bar_%d%d" % de),
                         "charge_allowed": bool(allowed),
                         "dimensions": dims,
                         "present": bool(allowed and all(d > 0 for d in dims))})
        out["down"] = recs

    summary = {}
    for key in ("up", "down"):
        if key in out:
            recs = out[key]
            summary[key] = {
                "patterns": len(recs),
                "charge_allowed": sum(1 for r in recs if r["charge_allowed"]),
                "present": sum(1 for r in recs if r["present"]),
                "texture_zeros": sum(1 for r in recs
                                     if r["charge_allowed"]
                                     and not r["present"])}
    out["summary"] = summary
    out["h1"] = {kk: v for kk, v in cache.items()}
    return out


def viable_triples(X, charge=3, require_slope=True, limit=None, SpaSM=False):
    r"""
    Charge-conserving triples that could carry a non-vanishing coupling.

    A cheap question asked of the *manifold*, before any model is built. An
    up-type coupling ``10_a 10_b 5_{ab}`` needs

        h^1(L_a) > 0,  h^1(L_b) > 0,  h^1(L_a^-1 (x) L_b^-1) > 0

    with the three charges summing to zero. If no such triple exists inside a
    charge box, then no rank-5 model with charges in that box can have an
    up-type coupling, whatever else it satisfies -- so this is worth running
    before a scan rather than after.

    With ``require_slope`` each of the three charges must also have a slope
    that changes sign somewhere in the Kahler cone. That is
    :func:`pyCICY.bundles.slope_is_definite` inverted, and it is a *necessary*
    condition for the summand to sit in a poly-stable bundle: a summand whose
    slope is one-signed can never have vanishing slope, so no bundle
    containing it is poly-stable. Including it matters. The first triple found
    without it on the tetraquadric contains ``(0, 0, -2, 0)``, whose slope is
    definite -- and a search over 9390 models built around that pair found
    exactly zero poly-stable ones, because the seed itself was disqualified.

    Returns a list of ``(k_a, k_b, k_c)``.

    Note
    ----
    The answer depends on the box, and saying "this manifold has no viable
    triples" without saying which box would be meaningless. Both conditions
    are necessary and neither is sufficient: a model realising a viable triple
    still has to close as a rank-5 sum with the right index, anomaly, and
    *joint* poly-stability, and the per-summand slope test does not imply the
    joint one.
    """
    import itertools as _it

    from ..bundles import _as_cicy, slope_is_definite

    XX = _as_cicy(X)
    n = XX.len
    d = np.asarray(XX.triple_intersection(), dtype=float)

    cache = {}

    def h1(v):
        key = tuple(int(x) for x in v)
        if key not in cache:
            cache[key] = int(np.asarray(
                XX.line_co(list(key)), dtype=int)[1])
        return cache[key]

    pool = [np.array(v, dtype=np.int64)
            for v in _it.product(range(-charge, charge + 1), repeat=n)]
    good = [v for v in pool
            if (not require_slope or not slope_is_definite(d, v))
            and h1(v) > 0]

    out = []
    for i, a in enumerate(good):
        for b in good[i:]:
            c = -(a + b)
            if np.any(np.abs(c) > charge):
                continue
            if require_slope and slope_is_definite(d, c):
                continue
            if h1(c) == 0:
                continue
            out.append((a.tolist(), b.tolist(), c.tolist()))
            if limit is not None and len(out) >= limit:
                return out
    return out
