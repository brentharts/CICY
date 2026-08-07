r"""
pyCICY.theories.representatives -- cup products, beyond the dimension count.

What this adds
--------------
:mod:`pyCICY.theories.yukawa` computes the *texture*: which couplings are
allowed by the charges and which additionally die because a cohomology group
is zero-dimensional. Both rules use only dimensions, and both are exact.

They are not the whole story. A coupling can pass both and still vanish,
because the cup product of the actual cohomology classes is zero. Catching
that needs the classes, not their dimensions --- and for a favourable CICY the
classes are explicit enough to work with, without any metric and without any
numerics.

The Koszul picture
------------------
For ``X`` cut out of an ambient product ``A`` by ``K`` polynomials, the Koszul
resolution

    0 -> Lambda^K N^* -> ... -> N^* -> O_A -> O_X -> 0 ,
    N^* = sum_a O_A(-d_a)

gives a spectral sequence with ``E_1^{-r,q} = H^q(A, Lambda^r N^* (x) O(k))``
converging to ``H^{q-r}(X, O_X(k))``. A term of ``Lambda^r N^*`` is labelled
by a subset ``S`` of the defining polynomials and carries the Koszul basis
vector ``e_S = e_{a_1} ^ ... ^ e_{a_r}``. On a product of projective spaces
the ambient cohomology factorises by Kunneth, and each ``P^n`` contributes in
degree ``0`` or ``n`` only.

So when a class of ``H^m(X, L)`` comes from a **single** term --- one subset
``S``, one choice of which factors contribute in top degree --- it has a
completely explicit description: the Koszul index set, and the list of factors
carrying the top-degree part. :func:`koszul_origin` returns exactly that.

Two ways a cup product dies
---------------------------
The product of three such classes is non-zero only if it lands on the top
class of the top Koszul term, and that imposes two independent conditions:

1. **The Koszul indices must be disjoint.** The product carries
   ``e_{S_1} ^ e_{S_2} ^ e_{S_3}``, which vanishes if any index is repeated.
2. **The top-degree factors must be distinct.** The ambient part must reach
   ``H^{top}(A)``, so the three classes must contribute their top degrees in
   different projective factors.

Neither is visible in the dimensions. On CICY 5299 the second holds for every
charge-allowed coupling, and the first fails for exactly one of them: the
model has two identical summands ``L_3 = L_4 = O(1,0,-1)``, so ``10_3`` and
``10_4`` carry the same Koszul index ``e_1``, ``e_1 ^ e_1 = 0``, and the
coupling ``10_3 10_4 5_{34}`` vanishes despite all three groups being
one-dimensional and the texture calling it present.

That is the finer selection rule that representatives buy, and it is the same
phenomenon the heterotic literature calls a type II vanishing.

Scope
-----
Only the case where every participating class has a *unique* Koszul origin is
handled, which is where the description above is unambiguous. When a group
receives contributions from several terms the classes mix, the spectral
sequence differentials matter, and this module declines rather than guessing.
The overall normalisation of a surviving coupling is also not computed: what is
returned is whether it vanishes, which is the part that carries the physics.
"""

import itertools
from math import comb

import numpy as np

__all__ = [
    "ambient_degree_dims", "koszul_origin", "cup_product_vanishes",
    "refine_texture",
]


def ambient_degree_dims(n, m):
    """Cohomology of ``O(m)`` on ``P^n``, as ``{degree: dimension}``.

    Non-zero in degree ``0`` for ``m >= 0`` and degree ``n`` for
    ``m <= -n-1``, and zero in between -- the fact that makes the Kunneth
    bookkeeping finite.
    """
    n = int(n)
    m = int(m)
    if m >= 0:
        return {0: comb(m + n, n)}
    if m <= -n - 1:
        return {n: comb(-m - 1, n)}
    return {}


def koszul_origin(conf, k):
    r"""
    Where ``H^*(X, O_X(k))`` comes from in the Koszul resolution.

    Returns a list of records, one per contributing term, each with the
    subset ``S`` of defining polynomials, the cohomological degree in each
    ambient factor, the dimension, and the resulting degree on ``X``.

    A single record means the class is explicit: one Koszul index set and one
    choice of top-degree factors. Several records mean the group is a mixture
    and this module will not analyse its products.
    """
    conf = np.asarray(conf, dtype=int)
    dims = conf[:, 0]
    degs = conf[:, 1:].T
    K = degs.shape[0]
    k = np.asarray(k, dtype=int)

    out = []
    for r in range(K + 1):
        for S in itertools.combinations(range(K), r):
            kk = k - (sum(degs[a] for a in S) if S else 0)
            per = [ambient_degree_dims(dims[i], kk[i]) for i in range(len(dims))]
            if not all(per):
                continue
            qs = tuple(next(iter(p)) for p in per)
            dim = 1
            for p in per:
                dim *= next(iter(p.values()))
            out.append({"S": S, "degrees": qs, "dimension": int(dim),
                        "degree_on_X": int(sum(qs) - r),
                        "twisted_charge": kk.tolist()})
    return out


def _unique_origin(conf, k, degree=1, verify=True):
    """
    The single origin of ``H^degree(X, O(k))``, or ``None``.

    Uniqueness within a degree is necessary but not sufficient: the Koszul
    spectral sequence can still have differentials that cut the term down
    before it reaches ``H^*(X)``. When ``verify``, the term's dimension is
    checked against :meth:`CICY.line_co` and a disagreement returns ``None``,
    because in that case the term is not the cohomology and its Koszul label is
    not a label for a class.
    """
    recs = [r for r in koszul_origin(conf, k) if r["degree_on_X"] == degree]
    if len(recs) != 1:
        return None
    if verify:
        from ..pyCICY import CICY
        conf_l = np.asarray(conf, dtype=int).tolist()
        h = CICY(conf_l).line_co(list(np.asarray(k, dtype=int)))
        if int(h[int(degree)]) != int(recs[0]["dimension"]):
            return None
    return recs[0]


def cup_product_vanishes(conf, charges, degree=1):
    r"""
    Whether the cup product of three ``H^1`` classes vanishes.

    ``charges`` is the list of three line bundle charge vectors, whose sum
    must be zero so that the product lands in ``H^3(O_X) = C``.

    Returns a dict with ``vanishes``, the ``reason`` when it does, and the
    Koszul data used. A ``vanishes`` of ``False`` means the two obstructions
    below are absent; it is not a computation of the coupling's value.

    The obstructions, both exact:

    ``repeated Koszul index``
        the product carries ``e_{S_1} ^ e_{S_2} ^ e_{S_3}`` and any repetition
        kills it. This is what happens when a model has two identical
        summands.

    ``top degree not reached``
        the ambient part must land on ``H^{top}(A)``, so the three classes
        must contribute their top degrees in *different* projective factors.
    """
    charges = [np.asarray(c, dtype=int) for c in charges]
    if len(charges) != 3:
        raise ValueError("a Yukawa coupling is a product of three classes")
    if np.any(sum(charges)):
        raise ValueError(
            "the charges sum to %s, not zero, so the product does not land in "
            "H^3(O_X) and there is no coupling to analyse"
            % (sum(charges)).tolist())

    conf = np.asarray(conf, dtype=int)
    dims = conf[:, 0]

    origins = [_unique_origin(conf, c, degree) for c in charges]
    if any(o is None for o in origins):
        # Two quite different failures hide behind a `None`, and calling both
        # "a mixture" misdescribes one of them: no contributing term at all
        # means the group is zero, which is an exact absence, not an ambiguity.
        counts = [len([r for r in koszul_origin(conf, c)
                       if r["degree_on_X"] == degree]) for c in charges]
        i = next(j for j, o in enumerate(origins) if o is None)
        if counts[i] == 1:
            raise ValueError(
                "class %d has a unique Koszul term whose dimension disagrees "
                "with h^%d(X, O(k)), so the spectral sequence differentials "
                "act and the term is not a representative" % (i, degree))
        if counts[i] == 0:
            raise ValueError(
                "class %d lies in a zero-dimensional group (no Koszul term "
                "contributes in degree %d), so there is no coupling rather "
                "than an undecided one" % (i, degree))
        raise ValueError(
            "class %d receives %d Koszul contributions, so its representative "
            "is a mixture and this analysis does not apply" % (i, counts[i]))

    idx = [a for o in origins for a in o["S"]]
    if len(set(idx)) != len(idx):
        return {"vanishes": True,
                "reason": "repeated Koszul index: e_S1 ^ e_S2 ^ e_S3 = 0",
                "koszul_indices": [o["S"] for o in origins],
                "top_factors": [o["degrees"] for o in origins]}

    tops = []
    for o in origins:
        t = [i for i, q in enumerate(o["degrees"]) if q == dims[i] and q > 0]
        tops.append(tuple(t))
    flat = [i for t in tops for i in t]
    if len(set(flat)) != len(flat) or len(flat) != len(dims):
        return {"vanishes": True,
                "reason": "the ambient degrees do not reach H^top(A): the "
                          "classes contribute top degree in factors %s"
                          % (tops,),
                "koszul_indices": [o["S"] for o in origins],
                "top_factors": tops}

    return {"vanishes": False, "reason": None,
            "koszul_indices": [o["S"] for o in origins],
            "top_factors": tops}


def refine_texture(conf, summands, kind="up"):
    r"""
    Apply the cup-product rules to the couplings the texture calls present.

    :func:`pyCICY.theories.yukawa.texture` uses dimensions only. This tests
    each surviving pattern against :func:`cup_product_vanishes` and reports
    which of them die anyway.

    On CICY 5299 the texture reports six up-type couplings and this reduces it
    to five: ``10_3 10_4 5_{34}`` vanishes because the model has two identical
    summands. That refinement is invisible to any dimension count.

    Returns a dict with ``kept``, ``killed`` and a per-pattern record. Patterns
    whose classes lack a unique Koszul origin are reported as ``undecided``
    rather than assumed non-zero.
    """
    from . import yukawa

    conf = np.asarray(conf, dtype=int)
    k = np.asarray(summands, dtype=int)
    t = yukawa.texture(conf, summands, kind=kind)

    records = []
    kept = killed = undecided = 0
    if kind in ("up", "both"):
        for a, b in itertools.combinations(range(len(k)), 2):
            rec = [r for r in t["up"]
                   if r["pattern"] == ("10_%d" % a, "10_%d" % b,
                                       "5_%d%d" % (a, b))]
            if not rec or not rec[0]["present"]:
                continue
            trio = [k[a], k[b], -(k[a] + k[b])]
            try:
                res = cup_product_vanishes(conf, trio)
            except ValueError as e:
                undecided += 1
                records.append({"pattern": rec[0]["pattern"],
                                "status": "undecided", "reason": str(e)})
                continue
            if res["vanishes"]:
                killed += 1
            else:
                kept += 1
            records.append({"pattern": rec[0]["pattern"],
                            "status": "vanishes" if res["vanishes"] else "kept",
                            "reason": res["reason"],
                            "koszul_indices": res["koszul_indices"]})
    return {"records": records, "kept": kept, "killed": killed,
            "undecided": undecided,
            "texture_present": t["summary"].get(kind, {}).get("present"),
            "note": "the texture counts dimensions; this counts cup products, "
                    "and the difference is the type II vanishing"}
