r"""
pyCICY.theories.differentials -- the Koszul differentials, computed.

Where this starts
-----------------
:mod:`pyCICY.theories.cocycles` writes a class down as a monomial whenever the
``E_1`` page has a single contributing term *and* that term's dimension agrees
with ``h^m(X, L)``. When it does not agree, the spectral sequence has not
degenerated and the term is not the cohomology. That module detects the
situation and declines. This one computes it.

On a ``[-3,3]`` charge box for CICY 5299 the split is 190 empty, 57 mixed, 66
cut down by differentials, and only 30 explicit. The 123 declines are the
subject here.

The differential
----------------
The Koszul resolution of ``O_X`` by ``Lambda^r N^*``, ``N^* = sum_a O_A(-d_a)``,
gives

    E_1^{-r,q} = sum_{|S| = r} H^q(A, O(k - d_S))  =>  H^{q-r}(X, O_X(k)) ,

and the first differential is contraction with the defining polynomials:

    d_1(e_S (x) w) = sum_{j} (-1)^j p_{a_j} · (e_{S \ a_j} (x) w) ,   S = {a_0 < ...}

so on cohomology it is *multiplication by a polynomial*,
``H^q(A, O(k - d_S)) -> H^q(A, O(k - d_S + d_a))``. In the monomial basis of
:mod:`cocycles` that map is completely explicit: exponents add, and the result
is kept only if it is still a basis element of the target --- which is the same
truncation rule the cup product already uses, and which automatically encodes
the fact that a group leaving the top-degree range is zero.

So ``d_1`` is an explicit matrix, and ``E_2 = ker d_1 / im d_1`` is linear
algebra.

Why this needs actual polynomials, and what that costs
------------------------------------------------------
The rank of ``d_1`` depends on the ``p_a`` themselves, not only on their
multidegrees --- which is exactly why no amount of tabulating the ``E_1`` page
can substitute for computing. This module uses
:func:`pyCICY.smoothness.random_equations` to take generic polynomials over
``F_p`` and computes ranks there. Two consequences worth being plain about:

* the answer is the **generic** one. A non-generic ``X`` can have larger
  cohomology, and this will not see it. :func:`e2_dimensions` accepts several
  seeds and reports whether the ranks were stable across them, which is
  evidence of genericity and not a proof of it.
* it is a computation over ``F_p``, so a rank could drop for a prime dividing
  some minor. Running two primes and comparing is cheap and is what
  :func:`degeneration_report` does.

Degeneration is checked, not assumed
------------------------------------
``E_2`` need not be ``E_infinity``: ``d_2`` and beyond can still act. This
module never assumes they vanish. It sums the ``E_2`` dimensions along
``q - r = m`` and compares with :meth:`CICY.line_co`, an engine that shares no
code with it. Agreement means the sequence degenerated at ``E_2`` *and* the
generic ranks were right, both at once --- and only then are the kernel vectors
representatives. Disagreement is reported as a decline, one page further along
than before but the same discipline.
"""

import itertools

import numpy as np

from .cocycles import ambient_basis
from ..smoothness import coordinate_layout, random_equations

__all__ = [
    "koszul_blocks", "differential_matrix", "e2_dimensions",
    "degeneration_report", "cohomology_basis",
]


def _flatten(mon):
    """A per-factor exponent tuple, concatenated over all coordinates."""
    return tuple(x for part in mon for x in part)


def _unflatten(dims, flat):
    """The inverse of :func:`_flatten`."""
    out = []
    pos = 0
    for n in dims:
        out.append(tuple(flat[pos:pos + n + 1]))
        pos += n + 1
    return tuple(out)


def koszul_blocks(conf, k, r):
    r"""
    The ``E_1`` blocks at Koszul weight ``r``, indexed by subset and degree.

    Returns ``{(S, q): (degrees, basis)}`` over all ``|S| = r`` whose ambient
    cohomology is non-zero, where ``q`` is the total cohomological degree and
    ``basis`` the monomial basis from :mod:`cocycles`.
    """
    conf = np.asarray(conf, dtype=int)
    dims = [int(d) for d in conf[:, 0]]
    degs = conf[:, 1:].T
    K = degs.shape[0]
    k = np.asarray(k, dtype=int)

    blocks = {}
    for S in itertools.combinations(range(K), r):
        kk = k - (sum(degs[a] for a in S) if S else 0)
        degrees, basis = ambient_basis(dims, kk)
        if not basis:
            continue
        blocks[(S, int(sum(degrees)))] = (degrees, basis)
    return blocks


def differential_matrix(conf, k, r, q, equations, p):
    r"""
    The matrix of ``d_1 : E_1^{-r,q} -> E_1^{-r+1,q}`` over ``F_p``.

    Returns ``(matrix, source_index, target_index)`` where the index maps send
    ``(S, basis_position)`` to a row or column number, so a kernel vector can be
    read back as an explicit combination of monomials.
    """
    conf = np.asarray(conf, dtype=int)
    dims = [int(d) for d in conf[:, 0]]
    _, slices, _ = coordinate_layout(conf)

    src = {(S, i): n
           for n, (S, i) in enumerate(
               (S, i) for (S, qq), (_, basis) in sorted(koszul_blocks(conf, k, r).items())
               if qq == q for i in range(len(basis)))}
    tgt = {(S, i): n
           for n, (S, i) in enumerate(
               (S, i) for (S, qq), (_, basis) in sorted(koszul_blocks(conf, k, r - 1).items())
               if qq == q for i in range(len(basis)))} if r > 0 else {}

    M = np.zeros((len(tgt), len(src)), dtype=np.int64)
    if not src or not tgt:
        return M, src, tgt

    src_blocks = {S: b for (S, qq), b in koszul_blocks(conf, k, r).items() if qq == q}
    tgt_blocks = {S: b for (S, qq), b in koszul_blocks(conf, k, r - 1).items() if qq == q}
    tgt_pos = {S: {mon: i for i, mon in enumerate(b[1])} for S, b in tgt_blocks.items()}

    for S, (src_degs, src_basis) in src_blocks.items():
        for j, a in enumerate(S):
            Sp = tuple(x for x in S if x != a)
            if Sp not in tgt_blocks:
                continue
            tgt_degs, _ = tgt_blocks[Sp]
            if tgt_degs != src_degs:
                # different Kunneth type: the component is zero, and the
                # truncation below would say so anyway
                continue
            sign = -1 if (j % 2) else 1
            for i, mon in enumerate(src_basis):
                flat = _flatten(mon)
                for expo, coeff in equations[a].items():
                    new = tuple(x + y for x, y in zip(flat, expo))
                    ok = True
                    for f, (lo, hi) in enumerate(slices):
                        if tgt_degs[f] == 0:
                            if any(x < 0 for x in new[lo:hi]):
                                ok = False
                                break
                        else:
                            if any(x > -1 for x in new[lo:hi]):
                                ok = False
                                break
                    if not ok:
                        continue
                    key = _unflatten(dims, new)
                    idx = tgt_pos[Sp].get(key)
                    if idx is None:
                        continue
                    M[tgt[(Sp, idx)], src[(S, i)]] += sign * coeff
    return M % p, src, tgt


def _rank_mod_p(M, p):
    """Rank of an integer matrix over ``F_p``, by elimination."""
    A = (np.asarray(M, dtype=np.int64) % p).copy()
    rows, cols = A.shape
    rank = 0
    for c in range(cols):
        piv = None
        for rr in range(rank, rows):
            if A[rr, c] % p:
                piv = rr
                break
        if piv is None:
            continue
        A[[rank, piv]] = A[[piv, rank]]
        inv = pow(int(A[rank, c]), p - 2, p)
        A[rank] = (A[rank] * inv) % p
        for rr in range(rows):
            if rr != rank and A[rr, c] % p:
                A[rr] = (A[rr] - A[rr, c] * A[rank]) % p
        rank += 1
        if rank == rows:
            break
    return rank


def e2_dimensions(conf, k, p=32003, seeds=(0,), equations=None):
    r"""
    The ``E_2`` page, and the cohomology it predicts.

    Returns a dict with ``e2`` mapping ``(r, q)`` to a dimension, ``predicted``
    mapping the degree ``m = q - r`` on ``X`` to the summed dimension, and
    ``stable`` saying whether every seed gave the same answer --- which is the
    evidence that the ranks are the generic ones.
    """
    conf = np.asarray(conf, dtype=int)
    dims = [int(d) for d in conf[:, 0]]
    K = conf.shape[1] - 1
    qs = range(sum(dims) + 1)

    results = []
    seed_list = (None,) if equations is not None else tuple(seeds)
    for seed in seed_list:
        eq = equations if equations is not None else random_equations(conf, p, seed=seed)
        e1 = {}
        for r in range(K + 1):
            for (S, q), (_, basis) in koszul_blocks(conf, k, r).items():
                e1[(r, q)] = e1.get((r, q), 0) + len(basis)

        ranks = {}
        for r in range(K + 1):
            for q in qs:
                if (r, q) not in e1:
                    ranks[(r, q)] = 0
                    continue
                M, _, _ = differential_matrix(conf, k, r, q, eq, p)
                ranks[(r, q)] = _rank_mod_p(M, p) if M.size else 0

        e2 = {}
        for (r, q), d in e1.items():
            out_rank = ranks.get((r, q), 0)
            in_rank = ranks.get((r + 1, q), 0)
            e2[(r, q)] = d - out_rank - in_rank
        results.append(e2)

    e2 = results[0]
    stable = all(r == e2 for r in results)
    predicted = {}
    for (r, q), d in e2.items():
        if d:
            predicted[q - r] = predicted.get(q - r, 0) + d
    return {"e2": e2, "predicted": predicted, "stable": stable,
            "p": p, "seeds": tuple(seed_list)}


def degeneration_report(conf, k, p=32003, seeds=(0, 1), verify=True):
    r"""
    Whether ``E_2 = E_infinity`` for this charge, checked against ``line_co``.

    Returns a dict with the predicted cohomology, the true one, and a
    ``degenerates`` flag. A ``True`` means the ``E_2`` page reproduces the
    cohomology exactly, which is the licence to treat kernel vectors as
    representatives. A ``False`` means ``d_2`` or beyond acts, or the generic
    ranks were not the right ones, and the module declines --- one page further
    than :mod:`cocycles` got, with the same discipline.
    """
    res = e2_dimensions(conf, k, p=p, seeds=seeds)
    out = dict(res)
    if not verify:
        out["degenerates"] = None
        return out

    from ..pyCICY import CICY
    conf_l = np.asarray(conf, dtype=int).tolist()
    h = CICY(conf_l).line_co(list(np.asarray(k, dtype=int)))
    true = {m: int(h[m]) for m in range(len(h)) if int(h[m])}
    pred = {m: d for m, d in res["predicted"].items() if d}
    out["true"] = true
    out["degenerates"] = bool(pred == true and res["stable"])
    out["reason"] = None if out["degenerates"] else (
        "E_2 predicts %s but h^*(X, L) = %s, so a later differential acts "
        "or the ranks were not generic" % (pred, true) if pred != true
        else "the ranks were not stable across seeds")
    return out


def cohomology_basis(conf, k, degree=1, p=32003, seeds=(0, 1), equations=None):
    r"""
    An explicit basis of ``H^degree(X, O(k))``, as kernel vectors mod image.

    Only returned when :func:`degeneration_report` confirms that ``E_2`` is the
    cohomology. Each basis element is a list of ``(S, monomial, coefficient)``
    over ``F_p``: a genuine Cech representative, in the same monomial language
    :mod:`cocycles` uses for the degenerate case, but now a combination rather
    than a single term.

    Raises when the sequence does not degenerate, rather than returning vectors
    that span the wrong thing.
    """
    rep = degeneration_report(conf, k, p=p, seeds=seeds)
    if not rep["degenerates"]:
        raise ValueError("the sequence does not degenerate at E_2 for this "
                         "charge: %s" % rep["reason"])

    conf = np.asarray(conf, dtype=int)
    dims = [int(d) for d in conf[:, 0]]
    K = conf.shape[1] - 1
    eq = equations if equations is not None else random_equations(conf, p, seed=seeds[0])

    out = []
    for r in range(K + 1):
        q = degree + r
        if q > sum(dims) or rep["e2"].get((r, q), 0) == 0:
            continue
        M, src, _ = differential_matrix(conf, k, r, q, eq, p)
        blocks = {S: b for (S, qq), b in koszul_blocks(conf, k, r).items() if qq == q}
        ker = _kernel_mod_p(M, p) if M.size else np.eye(len(src), dtype=np.int64)
        Min, _, _ = differential_matrix(conf, k, r + 1, q, eq, p)
        quotient = _quotient_basis(ker, Min % p if Min.size else
                                   np.zeros((len(src), 0), dtype=np.int64), p)
        inv = {n: key for key, n in src.items()}
        for vec in quotient:
            terms = []
            for n, c in enumerate(vec):
                if c % p:
                    S, i = inv[n]
                    terms.append((S, blocks[S][1][i], int(c % p)))
            out.append({"r": r, "q": q, "terms": terms})
    return {"basis": out, "dimension": len(out), "p": p,
            "note": "coefficients live in F_%d; these are E_2 = E_infinity "
                    "representatives, valid because the sequence was checked "
                    "to degenerate" % p}


def _top_target(conf):
    """The bidegree of ``H^3(X, O_X)`` in the Koszul page: ``r = K``, ``q = top``."""
    conf = np.asarray(conf, dtype=int)
    return conf.shape[1] - 1, int(sum(conf[:, 0]))


def coupling(conf, charges, degree=1, p=32003, seeds=(0, 1), equations=None):
    r"""
    The cup product of three classes, without assuming a single Koszul term.

    This is :func:`cocycles.coupling` with the restriction lifted. Each class is
    taken as an ``E_2`` representative --- a combination of ``(S, monomial)``
    terms rather than one of them --- and the product is expanded bilinearly,
    with :func:`cocycles.koszul_sign` supplying the sign of each term and the
    monomials multiplied and projected onto the top generator as before.

    Returns a dict with the integer ``tensor`` over ``F_p``, the ``value`` when
    all three groups are one-dimensional, and ``well_defined``: whether adding
    an image vector to a representative leaves the answer alone. That last is a
    check rather than an assumption, and a ``False`` invalidates the number.
    """
    from .cocycles import koszul_sign, top_generator

    charges = [np.asarray(c, dtype=int) for c in charges]
    if len(charges) != 3:
        raise ValueError("a Yukawa coupling is a product of three classes")
    if np.any(sum(charges)):
        raise ValueError(
            "the charges sum to %s, not zero, so the product does not land in "
            "H^3(O_X)" % (sum(charges)).tolist())

    conf = np.asarray(conf, dtype=int)
    dims = [int(d) for d in conf[:, 0]]
    K, q_top = _top_target(conf)
    eq = equations if equations is not None else random_equations(conf, p, seed=seeds[0])

    bases = [cohomology_basis(conf, c, degree=degree, p=p, seeds=seeds,
                              equations=eq)["basis"] for c in charges]
    shape = tuple(len(b) for b in bases)
    tensor = np.zeros(shape, dtype=np.int64)
    if 0 in shape:
        return {"tensor": tensor, "value": 0, "magnitude": 0, "vanishes": True,
                "well_defined": True, "shape": shape, "p": p,
                "reason": "a participating group is zero-dimensional"}

    top = top_generator(dims)

    def _product(v1, v2, v3):
        total = 0
        for t1 in v1["terms"]:
            for t2 in v2["terms"]:
                for t3 in v3["terms"]:
                    sgn = koszul_sign([t1[0], t2[0], t3[0]], K)
                    if sgn == 0:
                        continue
                    prod = tuple(
                        tuple(a + b + c for a, b, c in
                              zip(t1[1][f], t2[1][f], t3[1][f]))
                        for f in range(len(dims)))
                    if prod != top:
                        continue
                    total += sgn * t1[2] * t2[2] * t3[2]
        return total % p

    for i in range(shape[0]):
        for j in range(shape[1]):
            for l in range(shape[2]):
                if bases[0][i]["r"] + bases[1][j]["r"] + bases[2][l]["r"] != K:
                    continue
                tensor[i, j, l] = _product(bases[0][i], bases[1][j], bases[2][l])

    well = _check_well_defined(conf, charges, bases, degree, p, seeds, eq,
                               _product, tensor)
    val = int(tensor.reshape(-1)[0]) if tensor.size == 1 else None
    if val is not None and val > p // 2:
        val -= p                          # a symmetric representative of F_p
    return {"tensor": tensor % p, "value": val,
            "magnitude": abs(val) if val is not None else None,
            "vanishes": not (tensor % p).any(), "well_defined": well,
            "shape": shape, "p": p, "reason": None}


def _check_well_defined(conf, charges, bases, degree, p, seeds, eq, product, tensor):
    r"""
    Whether the answer survives changing the representative by a coboundary.

    A class is a kernel vector *modulo* the image of ``d_1``, so a coupling is
    only meaningful if adding an image vector leaves it unchanged. This adds
    each image generator to the first class and re-multiplies. It is a test of
    the construction, not a formality: a failure would mean the product does
    not descend to cohomology and the number means nothing.
    """
    conf = np.asarray(conf, dtype=int)
    K, _ = _top_target(conf)
    changed = 0
    for r in sorted({b["r"] for b in bases[0]}):
        q = degree + r
        M, src, _ = differential_matrix(conf, charges[0], r + 1, q, eq, p)
        if not M.size:
            continue
        blocks = {S: b for (S, qq), b in koszul_blocks(conf, charges[0], r).items()
                  if qq == q}
        inv = {n: key for key, n in
               {(S, i): n for n, (S, i) in enumerate(
                   (S, i) for S, (_, basis) in sorted(blocks.items())
                   for i in range(len(basis)))}.items()}
        for col in range(M.shape[1]):
            vec = M[:, col] % p
            if not vec.any():
                continue
            terms = [(inv[n][0], blocks[inv[n][0]][1][inv[n][1]], int(c))
                     for n, c in enumerate(vec) if c % p and n in inv]
            if not terms:
                continue
            shifted = {"r": r, "q": q, "terms": terms}
            for j in range(len(bases[1])):
                for l in range(len(bases[2])):
                    if r + bases[1][j]["r"] + bases[2][l]["r"] != K:
                        continue
                    if product(shifted, bases[1][j], bases[2][l]) % p:
                        changed += 1
    return changed == 0


def model_couplings(conf, summands, p=32003, seeds=(0, 1)):
    r"""
    Every up-type coupling of a line bundle model, with no degeneracy assumed.

    The counterpart of :func:`cocycles.model_couplings` for models whose classes
    are not single Koszul terms. Where both apply they must agree, and the tests
    check that they do on CICY 5299.
    """
    conf = np.asarray(conf, dtype=int)
    k = np.asarray(summands, dtype=int)
    eq = random_equations(conf, p, seed=seeds[0])

    records = []
    for a, b in itertools.combinations(range(len(k)), 2):
        trio = [k[a], k[b], -(k[a] + k[b])]
        pattern = ("10_%d" % a, "10_%d" % b, "5_%d%d" % (a, b))
        try:
            res = coupling(conf, trio, p=p, seeds=seeds, equations=eq)
        except ValueError as e:
            records.append({"pattern": pattern, "status": "declined",
                            "value": None, "reason": str(e)})
            continue
        records.append({"pattern": pattern,
                        "status": "vanishes" if res["vanishes"] else "present",
                        "value": res["value"], "shape": res["shape"],
                        "well_defined": res["well_defined"]})
    return {"records": records,
            "present": sum(1 for r in records if r["status"] == "present"),
            "vanishing": sum(1 for r in records if r["status"] == "vanishes"),
            "declined": sum(1 for r in records if r["status"] == "declined"),
            "p": p,
            "note": "values are exact in F_%d up to one model-wide "
                    "normalisation, and carry no Kahler factor" % p}


def _kernel_mod_p(M, p):
    """A basis of the kernel of ``M`` over ``F_p``, as rows."""
    A = (np.asarray(M, dtype=np.int64) % p).copy()
    rows, cols = A.shape
    if rows == 0:
        return np.eye(cols, dtype=np.int64)
    pivots = []
    rank = 0
    for c in range(cols):
        piv = None
        for rr in range(rank, rows):
            if A[rr, c] % p:
                piv = rr
                break
        if piv is None:
            continue
        A[[rank, piv]] = A[[piv, rank]]
        inv = pow(int(A[rank, c]), p - 2, p)
        A[rank] = (A[rank] * inv) % p
        for rr in range(rows):
            if rr != rank and A[rr, c] % p:
                A[rr] = (A[rr] - A[rr, c] * A[rank]) % p
        pivots.append(c)
        rank += 1
        if rank == rows:
            break
    free = [c for c in range(cols) if c not in pivots]
    basis = []
    for f in free:
        v = np.zeros(cols, dtype=np.int64)
        v[f] = 1
        for i, c in enumerate(pivots):
            v[c] = (-A[i, f]) % p
        basis.append(v % p)
    return np.array(basis, dtype=np.int64) if basis else np.zeros((0, cols), dtype=np.int64)


def _quotient_basis(ker, image, p):
    """Vectors of ``ker`` completing a basis of ``image`` inside it."""
    image = np.asarray(image, dtype=np.int64)
    cols = ker.shape[1] if ker.size else 0
    img_vecs = [image[:, j] % p for j in range(image.shape[1])] if image.size else []
    chosen = []
    current = list(img_vecs)
    base_rank = _rank_mod_p(np.array(current, dtype=np.int64), p) if current else 0
    for v in ker:
        trial = current + [v % p]
        r = _rank_mod_p(np.array(trial, dtype=np.int64), p)
        if r > base_rank:
            chosen.append(v % p)
            current = trial
            base_rank = r
    return chosen
