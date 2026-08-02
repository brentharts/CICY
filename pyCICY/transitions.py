r"""
pyCICY.transitions -- configuration matrix normal forms, splits and conifold
transitions.

Two independent pieces of functionality live here.

Normal forms and equivalence
----------------------------
Different configuration matrices can describe the same variety; the simplest
way this happens is a relabelling of the projective factors and of the
defining equations, i.e. a row and column permutation. :func:`normal_form`
puts a configuration into a canonical representative of that orbit and
:func:`equivalent` decides equality of two configurations up to such
relabelling. This mirrors the ``CINormal`` / ``CIEquiv`` functionality of the
CIPro package, Anderson, Constantin, Gray, He, Lee and Lukas,
arXiv:2606.27588.

Splits and conifold transitions
-------------------------------
A P^1 split replaces one defining equation of multidegree q by two equations
whose degrees sum to q, at the cost of one extra P^1 factor
(Candelas, Dale, Lutken and Schimmrigk, Nucl. Phys. B298 (1988) 493). The
quintic splits as

    [P^4 | 5]   ->   [P^1 | 1 1]
                     [P^4 | 1 4]

Geometrically the two sides are joined through a nodal variety: the split is
a small resolution X_R of a nodal deformation X_D. Across such a transition

    h^{1,1}(X_R) = h^{1,1}(X_D) + 1                                     (1.6)
    chi(X_R)     = chi(X_D) + 2N                                        (1.7)

with N the number of nodes, following Anderson, Gray, Patil and Scanlon,
arXiv:2512.18124, section 1.1. :func:`transition` evaluates both sides with
the existing pyCICY cohomology machinery and reports N. For the quintic it
returns N = 16, the node count quoted in that reference.

A split with N = 0 changes no topology at all; these are the *ineffective*
splits of Candelas et al., where the would-be nodal locus is empty and the
two configurations describe isomorphic geometries.
"""

import itertools as it
from collections import defaultdict

import numpy as np

__all__ = [
    "normal_form", "equivalent", "canonical_key",
    "split", "splits", "contract", "is_contractible",
    "nodes_expected", "transition", "check_configuration",
    "dimensions", "is_calabi_yau",
]


# ------------------------------------------------------------------ helpers

def _as_matrix(conf):
    """Validate a configuration and return it as a 2D int64 array.

    A configuration is ``[[n_1, q_1^1, ..., q_1^K], ...]``: the first entry of
    each row is the dimension of the projective factor, the rest are the
    degrees of the K defining equations in that factor.
    """
    M = np.array(conf, dtype=np.int64)
    if M.ndim != 2:
        raise ValueError("configuration must be a 2D matrix, got shape %r"
                         % (M.shape,))
    if M.shape[1] < 2:
        raise ValueError("each row needs a dimension and at least one degree")
    if (M[:, 0] < 1).any():
        raise ValueError("projective space dimensions must be >= 1")
    if (M[:, 1:] < 0).any():
        raise ValueError("degrees must be non-negative")
    return M


def dimensions(conf):
    """Return ``(dim_ambient, n_equations, dim_X)`` for a configuration."""
    M = _as_matrix(conf)
    dim_a = int(M[:, 0].sum())
    k = M.shape[1] - 1
    return dim_a, k, dim_a - k


def is_calabi_yau(conf):
    """True if every row satisfies the CY condition sum_a q^r_a = n_r + 1."""
    M = _as_matrix(conf)
    return bool((M[:, 1:].sum(axis=1) == M[:, 0] + 1).all())


# --------------------------------------------------------- normal forms

def _refine(M):
    """Partition rows and columns into classes by iterated signature refinement.

    Returns ``(row_classes, col_classes)``, each a list of lists of indices.
    Rows in different classes can never be exchanged by a symmetry of the
    configuration, so the search in :func:`normal_form` only has to consider
    permutations inside a class. Without this the search would be m! * K!.
    """
    m, kk = M.shape[0], M.shape[1] - 1
    rlab = [0] * m
    clab = [0] * kk

    for _ in range(m + kk + 2):
        # A row is described by its dimension, its current label, and the
        # multiset of (degree, column label) pairs it contains; and dually.
        rsig = [(int(M[i, 0]), rlab[i],
                 tuple(sorted((int(M[i, 1 + c]), clab[c]) for c in range(kk))))
                for i in range(m)]
        csig = [(clab[c],
                 tuple(sorted((int(M[i, 1 + c]), rlab[i]) for i in range(m))))
                for c in range(kk)]

        rnew = _relabel(rsig)
        cnew = _relabel(csig)
        if rnew == rlab and cnew == clab:
            break
        rlab, clab = rnew, cnew

    return _classes(rlab), _classes(clab)


def _relabel(sigs):
    order = {s: i for i, s in enumerate(sorted(set(sigs)))}
    return [order[s] for s in sigs]


def _classes(labels):
    groups = defaultdict(list)
    for i, l in enumerate(labels):
        groups[l].append(i)
    return [groups[k] for k in sorted(groups)]


def canonical_key(conf, max_perms=200000):
    """Return the canonical (row-permutation, column-permutation) invariant.

    The key is the lexicographically smallest matrix obtainable by permuting
    rows and columns, flattened to a tuple of ints. Two configurations are
    related by relabelling exactly when their keys agree.
    """
    return normal_form(conf, max_perms=max_perms)[0]


def normal_form(conf, max_perms=200000):
    """Canonical representative of a configuration under row/column permutation.

    Parameters
    ----------
    conf : nested list
        Configuration matrix.
    max_perms : int
        Safety limit on the number of candidate permutations examined. The
        refinement step usually reduces this to a handful; a configuration
        with many interchangeable rows and columns can still blow up, and
        rather than run for an unbounded time this raises ValueError.

    Returns
    -------
    key : tuple
        Flattened canonical matrix, ``(shape, entries...)``.
    row_perm : tuple
        ``row_perm[i]`` is the row of the input placed at position i.
    col_perm : tuple
        ``col_perm[j]`` is the equation of the input placed at position j.

    Example
    -------
    The two configurations for dP2 in CIPro section 2.4 differ by a
    relabelling, so they share a normal form.

    >>> a = normal_form([[2, 1, 1], [1, 1, 0], [1, 0, 1]])[0]
    >>> b = normal_form([[1, 0, 1], [1, 1, 0], [2, 1, 1]])[0]
    >>> a == b
    True
    """
    M = _as_matrix(conf)
    m, kk = M.shape[0], M.shape[1] - 1

    row_classes, col_classes = _refine(M)

    n_row = 1
    for c in row_classes:
        n_row *= _factorial(len(c))
    n_col = 1
    for c in col_classes:
        n_col *= _factorial(len(c))
    if n_row * n_col > max_perms:
        raise ValueError(
            "configuration has too much symmetry to canonicalise within "
            "max_perms=%d (%d row x %d column candidates). Raise max_perms "
            "if you are willing to wait." % (max_perms, n_row, n_col))

    best = None
    best_perms = None
    for rperm in _class_permutations(row_classes, m):
        for cperm in _class_permutations(col_classes, kk):
            cand = M[np.ix_(rperm, [0] + [1 + c for c in cperm])]
            key = tuple(cand.ravel().tolist())
            if best is None or key < best:
                best = key
                best_perms = (tuple(rperm), tuple(cperm))

    return ((m, kk + 1) + best,) + best_perms


def _factorial(n):
    out = 1
    for i in range(2, n + 1):
        out *= i
    return out


def _class_permutations(classes, total):
    """Yield full index permutations that only permute within each class."""
    for choice in it.product(*[it.permutations(c) for c in classes]):
        perm = [i for grp in choice for i in grp]
        if len(perm) != total:
            raise AssertionError("class partition does not cover all indices")
        yield perm


def equivalent(conf1, conf2, max_perms=200000):
    """True if two configurations agree up to row and column permutation.

    >>> equivalent([[2, 1, 1], [1, 1, 0], [1, 0, 1]],
    ...            [[1, 0, 1], [1, 1, 0], [2, 1, 1]])
    True
    """
    a = _as_matrix(conf1)
    b = _as_matrix(conf2)
    if a.shape != b.shape:
        return False
    if sorted(a[:, 0].tolist()) != sorted(b[:, 0].tolist()):
        return False
    return (normal_form(a, max_perms)[0] == normal_form(b, max_perms)[0])


# ------------------------------------------------------- splits/contractions

def split(conf, column, partition):
    """Perform a P^1 split of one defining equation.

    The equation ``column`` is replaced by two equations whose degrees sum to
    the original ones, and a new P^1 factor is prepended carrying degree 1 in
    each of the two new equations.

    Parameters
    ----------
    conf : nested list
        Configuration matrix.
    column : int
        Which defining equation to split, indexed from 0 among the K degree
        columns (so ``column=0`` is the first degree column, not the
        dimension column).
    partition : sequence of int
        ``partition[i]`` is the degree of the *first* of the two new
        equations in projective factor i; the second gets the remainder.
        Must satisfy ``0 <= partition[i] <= conf[i][column + 1]``.

    Returns
    -------
    nested list
        The split configuration, with the new P^1 as its first row.

    Example
    -------
    The quintic splits to the configuration in arXiv:2512.18124 eq. (1.11).

    >>> split([[4, 5]], 0, [1])
    [[1, 1, 1], [4, 1, 4]]
    """
    M = _as_matrix(conf)
    m, kk = M.shape[0], M.shape[1] - 1
    if not (0 <= column < kk):
        raise ValueError("column %d out of range 0..%d" % (column, kk - 1))
    part = list(partition)
    if len(part) != m:
        raise ValueError("partition needs one entry per projective factor "
                         "(%d), got %d" % (m, len(part)))
    for i, p in enumerate(part):
        q = int(M[i, 1 + column])
        if not (0 <= p <= q):
            raise ValueError(
                "partition[%d] = %d must lie in 0..%d (the degree of equation "
                "%d in factor %d)" % (i, p, q, column, i))

    others = [c for c in range(kk) if c != column]
    rows = []
    # new P^1 factor: degree 1 in each of the two new equations
    rows.append([1] + [0] * len(others) + [1, 1])
    for i in range(m):
        q = int(M[i, 1 + column])
        rows.append([int(M[i, 0])] + [int(M[i, 1 + c]) for c in others]
                    + [part[i], q - part[i]])
    return rows


def splits(conf, column=None, include_trivial=False):
    """Enumerate P^1 splits of a configuration.

    Parameters
    ----------
    column : int or None
        Restrict to splitting this equation; None enumerates all.
    include_trivial : bool
        Whether to include partitions where one of the two new equations has
        degree zero in every factor. Those are never geometrically useful:
        a wholly zero equation does not cut anything out.

    Yields
    ------
    (column, partition, configuration)
    """
    M = _as_matrix(conf)
    m, kk = M.shape[0], M.shape[1] - 1
    cols = range(kk) if column is None else [column]
    for c in cols:
        degs = [int(M[i, 1 + c]) for i in range(m)]
        for part in it.product(*[range(d + 1) for d in degs]):
            if not include_trivial:
                if all(p == 0 for p in part) or all(
                        p == degs[i] for i, p in enumerate(part)):
                    continue
            yield c, tuple(part), split(M, c, part)


def is_contractible(conf):
    """Return the rows that look like the P^1 of a split.

    Such a row is a P^1 carrying degree 1 in exactly two equations and 0 in
    all others, which is precisely the shape :func:`split` introduces.
    """
    M = _as_matrix(conf)
    out = []
    for i in range(M.shape[0]):
        if int(M[i, 0]) != 1:
            continue
        degs = M[i, 1:]
        ones = np.flatnonzero(degs == 1)
        if len(ones) == 2 and int(degs.sum()) == 2:
            out.append((i, (int(ones[0]), int(ones[1]))))
    return out


def contract(conf, row=None):
    """Undo a P^1 split, the inverse of :func:`split`.

    Deletes a P^1 row of the shape produced by :func:`split` and merges the
    two equations it joined by adding their degrees.

    >>> contract([[1, 1, 1], [4, 1, 4]])
    [[4, 5]]
    """
    M = _as_matrix(conf)
    options = is_contractible(M)
    if not options:
        raise ValueError("no contractible P^1 row: this configuration is not "
                         "in the form produced by a split")
    if row is None:
        r, (c1, c2) = options[0]
    else:
        match = [o for o in options if o[0] == row]
        if not match:
            raise ValueError("row %d is not contractible; candidates are %r"
                             % (row, [o[0] for o in options]))
        r, (c1, c2) = match[0]

    kk = M.shape[1] - 1
    others = [c for c in range(kk) if c not in (c1, c2)]
    rows = []
    for i in range(M.shape[0]):
        if i == r:
            continue
        merged = int(M[i, 1 + c1]) + int(M[i, 1 + c2])
        rows.append([int(M[i, 0])] + [int(M[i, 1 + c]) for c in others]
                    + [merged])
    if not rows:
        raise ValueError("contraction would remove every projective factor")
    return rows


# --------------------------------------------------------------- transitions

def _hodge(conf, log=3):
    """Hodge and Euler data for a configuration, via the CICY machinery."""
    try:
        from .pyCICY import CICY
    except ImportError:  # running this file outside the package
        from pyCICY import CICY

    import logging
    cy_logger = logging.getLogger("pyCICY")

    def mute(record):
        return False

    cy_logger.addFilter(mute)
    try:
        M = CICY(conf, log=log)
        if M.nfold != 3:
            raise ValueError(
                "conifold transition analysis is implemented for Calabi-Yau "
                "threefolds; this configuration is a %d-fold" % M.nfold)
        return {
            "h11": float(M.h[2]),
            "h21": float(M.h[1]),
            "euler": int(M.euler_characteristic()),
            "favourable": bool(M.fav),
        }
    finally:
        cy_logger.removeFilter(mute)


def _ambient_intersection(dims, classes):
    """Intersection number of divisor classes in a product of projective spaces.

    ``dims`` are the n_i, and each element of ``classes`` is a multidegree
    ``(d_1, ..., d_m)`` standing for the divisor sum_i d_i J_i. The answer is
    the coefficient of prod_i J_i^{n_i} in the product of those classes, using
    J_i^{n_i + 1} = 0.
    """
    m = len(dims)
    if len(classes) != sum(dims):
        raise ValueError(
            "need exactly dim(A) = %d classes to get a number, got %d"
            % (sum(dims), len(classes)))

    # Polynomial in J_1..J_m truncated at degree n_i in each variable,
    # represented as {exponent tuple: coefficient}.
    poly = {(0,) * m: 1}
    for cls in classes:
        new = {}
        for expo, coeff in poly.items():
            for i in range(m):
                d = int(cls[i])
                if d == 0:
                    continue
                if expo[i] + 1 > dims[i]:
                    continue  # J_i^{n_i+1} = 0
                nxt = list(expo)
                nxt[i] += 1
                nxt = tuple(nxt)
                new[nxt] = new.get(nxt, 0) + coeff * d
        poly = new
        if not poly:
            return 0
    return int(poly.get(tuple(dims), 0))


def nodes_expected(conf, column, partition):
    """Number of nodes of the singular variety of a split, from geometry.

    This is an independent route to the N appearing in
    ``chi(X_R) = chi(X_D) + 2N``, computed without any cohomology. Writing the
    two split equations as ``x_0 f_1 + x_1 f_2`` and ``x_0 g_1 + x_1 g_2``,
    the nodes are the points where all four of f_1, f_2, g_1, g_2 vanish along
    with the remaining defining equations. Those four have multidegrees a, a,
    b, b where a is the ``partition`` and b its complement, so N is the
    intersection number of

        a, a, b, b, and the other defining equations

    in the ambient space. The count of classes matches dim(A) exactly for a
    Calabi-Yau threefold, so this is a number rather than a class.

    For the quintic split of arXiv:2512.18124 eq. (1.11), a = 1 and b = 4 in
    P^4, giving 1*1*4*4 = 16.

    >>> nodes_expected([[4, 5]], 0, [1])
    16
    >>> nodes_expected([[4, 5]], 0, [2])
    36
    """
    M = _as_matrix(conf)
    m, kk = M.shape[0], M.shape[1] - 1
    if not (0 <= column < kk):
        raise ValueError("column %d out of range 0..%d" % (column, kk - 1))
    part = [int(p) for p in partition]
    if len(part) != m:
        raise ValueError("partition needs one entry per projective factor")

    dims = [int(M[i, 0]) for i in range(m)]
    a = tuple(part)
    b = tuple(int(M[i, 1 + column]) - part[i] for i in range(m))
    classes = [a, a, b, b]
    for c in range(kk):
        if c == column:
            continue
        classes.append(tuple(int(M[i, 1 + c]) for i in range(m)))

    if len(classes) != sum(dims):
        raise ValueError(
            "this split does not give a point count: %d classes against "
            "dim(A) = %d. nodes_expected assumes a Calabi-Yau threefold."
            % (len(classes), sum(dims)))
    return _ambient_intersection(dims, classes)


def transition(conf_deformation, conf_resolution):
    """Analyse a conifold transition between two Calabi-Yau threefolds.

    Computes the Hodge numbers and Euler characteristic of both sides with
    the pyCICY cohomology machinery and extracts the number of nodes of the
    shared singular variety from

        chi(X_R) = chi(X_D) + 2N

    (arXiv:2512.18124 eq. 1.7). The companion relation
    ``h^{1,1}(X_R) = h^{1,1}(X_D) + 1`` (eq. 1.6) is reported as a check
    rather than assumed.

    Returns
    -------
    dict with keys
        deformation, resolution : dict
            Hodge data for each side.
        nodes : int
            N, the number of nodes of the singular variety.
        h11_shift, h21_shift, euler_shift : float/int
        expected_h11_shift : float
            1 for an effective transition, 0 for an ineffective split.
        h11_check : bool
            Whether the measured h^{1,1} shift matches that expectation.
        ineffective : bool
            True when N == 0. The transition is then trivial: the would-be
            nodal locus is empty and the two configurations describe
            isomorphic geometries (an ineffective split).
        consistent : bool
            h11_check and N >= 0.

    Example
    -------
    The quintic and its P^1 split, arXiv:2512.18124 section 1.1.

    >>> t = transition([[4, 5]], [[1, 1, 1], [4, 1, 4]])
    >>> t["nodes"]
    16
    """
    d = _hodge(conf_deformation)
    r = _hodge(conf_resolution)

    euler_shift = r["euler"] - d["euler"]
    if euler_shift % 2:
        raise ValueError(
            "chi(X_R) - chi(X_D) = %d is odd, so chi_R = chi_D + 2N has no "
            "integer solution; these two configurations are not related by a "
            "conifold transition" % euler_shift)
    nodes = euler_shift // 2

    h11_shift = r["h11"] - d["h11"]
    ineffective = (nodes == 0)
    # An effective transition adds one Kahler class (eq. 1.6). An ineffective
    # split adds no nodes and no topology at all, so h^{1,1} must be unchanged
    # there; checking for +1 in that case would be the wrong expectation.
    expected_h11_shift = 0.0 if ineffective else 1.0
    h11_check = (h11_shift == expected_h11_shift)

    return {
        "deformation": d,
        "resolution": r,
        "nodes": int(nodes),
        "h11_shift": h11_shift,
        "h21_shift": r["h21"] - d["h21"],
        "euler_shift": euler_shift,
        "expected_h11_shift": expected_h11_shift,
        "h11_check": bool(h11_check),
        "ineffective": bool(ineffective),
        "consistent": bool(h11_check and nodes >= 0),
    }


def check_configuration(conf):
    """Summarise a configuration: dimensions, CY condition, redundancies.

    Returns a dict; ``warnings`` lists anything that would make the
    configuration degenerate or non-Calabi-Yau.
    """
    M = _as_matrix(conf)
    dim_a, k, dim_x = dimensions(M)
    warnings = []

    if not is_calabi_yau(M):
        rows = [i for i in range(M.shape[0])
                if int(M[i, 1:].sum()) != int(M[i, 0]) + 1]
        warnings.append(
            "not Calabi-Yau: factors %r violate sum_a q^r_a = n_r + 1" % rows)
    if dim_x < 1:
        warnings.append("dim X = %d; the configuration cuts down too far"
                        % dim_x)
    zero_cols = [c for c in range(k) if int(M[:, 1 + c].sum()) == 0]
    if zero_cols:
        warnings.append("equations %r have degree zero everywhere and cut "
                        "out nothing" % zero_cols)
    zero_rows = [i for i in range(M.shape[0]) if int(M[i, 1:].sum()) == 0]
    if zero_rows:
        warnings.append("factors %r are untouched by every equation, so X is "
                        "a direct product" % zero_rows)

    return {
        "dim_ambient": dim_a,
        "n_equations": k,
        "dim_X": dim_x,
        "calabi_yau": is_calabi_yau(M),
        "contractible_rows": [r for r, _ in is_contractible(M)],
        "warnings": warnings,
    }
