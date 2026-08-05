r"""
pyCICY.polytope -- reflexive lattice polytopes in any dimension.

Why this exists
---------------
:mod:`pyCICY.toric` is two-dimensional throughout. It classifies the sixteen
reflexive polygons, checks the twelve theorem and feeds their lattice points
to :mod:`pyCICY.quantum_curve` as hopping vectors. Everything there is
correct and none of it generalises: ``dual`` assumes an ordered vertex cycle,
``lattice_points`` scan-converts a polygon, and ``twelve`` is a statement
about polygons only.

This module lifts that machinery to arbitrary dimension. The immediate reason
is that a reflexive *four*-polytope gives a Calabi-Yau threefold by the
Batyrev construction, which is the compact side of the package reached from
the toric side -- and the specific four-polytope of interest is the 24-cell,
cited in Ali, "Quantum Spacetime Imprints: The 24-Cell, Standard Model
Symmetry and its Flavor Mixing", arXiv:2511.10685, via

    V. Braun, "The 24-cell and Calabi-Yau threefolds with Hodge numbers
    (1,1)", JHEP 05 (2012) 101, arXiv:1102.4880.

Correctness is checked by overlap rather than by assertion: in dimension two
this module must reproduce :mod:`pyCICY.toric` exactly, on all sixteen
reflexive polygons, and ``tests/test_polytope.py`` requires it.

Reflexivity is a statement about a lattice, not a polytope
----------------------------------------------------------
A lattice polytope P containing the origin in its interior is *reflexive* when
its polar dual

    P* = { y : <x, y> >= -1 for all x in P }

is again a lattice polytope. Equivalently every facet of P lies on a
hyperplane <m, x> = -1 with m in the dual lattice.

Which lattice is meant is not a detail, and the 24-cell is the case that makes
that unavoidable. Written with the 24 vertices +-e_i +- e_j -- the D_4 root
system, which is how Ali's section 3.1 writes it -- the polar dual has
vertices

    {+- e_i}  and  { (1/2)(+-1, +-1, +-1, +-1) } ,

which is *not* integral in Z^4. So the 24-cell is **not** reflexive with
respect to Z^4. It is reflexive with respect to the lattice its own vertices
generate, namely D_4 = { x in Z^4 : sum x_i even }, of index 2 in Z^4, and in
a basis of that lattice both it and its dual are honest lattice polytopes
with 24 vertices each. :func:`in_generated_lattice` performs that change of
basis, and :func:`is_reflexive` reports which lattice it used.

Two remarks on Ali's paper follow from this and are worth stating precisely,
since they are checkable and the tests check them.

*The two vertex sets in that paper are the dual pair, not two descriptions of
one polytope.* Section 2.1 uses { +-e_i } together with the sixteen
(1/2)(+-1,+-1,+-1,+-1), of squared norm 1; section 3.1 uses +-e_i +- e_j, of
squared norm 2. :func:`polar` maps the second exactly onto the first. A remark
in section 2.1 states that the long roots +-e_i+-e_j are "not vertices of the
24-cell", which is true of the polytope as section 2.1 normalises it and false
of the one section 3.1 defines; both are 24-cells, and each is the other's
polar.

*The count of regular tetrahedra is 48, not 576.* Section 3.1 asks for
four vertices with ||v_i - v_j||^2 = 4 throughout, which in the norm-2
normalisation means four mutually orthogonal vertices.
:func:`equilateral_subsets` enumerates them by brute force over all 10626
four-element subsets and finds 48. The structure is transparent once seen:
the twelve diagonals of the 24-cell fall into three mutually orthogonal
frames of four, and each frame admits 2^4 independent sign choices, giving
3 * 16 = 48.

Batyrev's construction
----------------------
For a reflexive polytope Delta of dimension d, the anticanonical hypersurface
in the toric variety of the face fan of Delta* is a Calabi-Yau of dimension
d-1. For d = 4 the Hodge numbers are

    h^{1,1} = l(Delta*) - (d+1) - sum_{facets th*} l*(th*)
                                + sum_{codim 2 th*} l*(th*) l*(th)
    h^{2,1} = l(Delta)  - (d+1) - sum_{facets th}  l*(th)
                                + sum_{codim 2 th}  l*(th)  l*(th*)

with l the number of lattice points of a face, l* the number in its relative
interior, and th <-> th* the duality between faces of Delta and Delta* that
reverses dimension, dim th + dim th* = d - 1.

For the 24-cell both polytopes have exactly 25 lattice points -- their 24
vertices and the origin -- so no proper face has an interior lattice point
and both correction sums vanish. The result is

    h^{1,1} = h^{2,1} = 20 ,   chi = 0 ,

and self-duality forces h^{1,1} = h^{2,1} independently of the arithmetic.
Braun's Hodge numbers (1,1) are those of a free quotient of this threefold,
not of the threefold itself; this module computes the cover and does not
enumerate free quotients, which is the same boundary :mod:`pyCICY.symmetries`
draws on the CICY side.

Ali's paper describes the 24-cell as "the Newton polytope defining a smooth
K3 hypersurface". A reflexive *three*-polytope gives a K3; a reflexive
four-polytope such as the 24-cell gives a threefold, which is what Braun's
title says and what :func:`batyrev_hodge` returns.

Scope
-----
* No normal form and no GL(n,Z) classification. :func:`toric.normal_form`
  exists in two dimensions by brute force over a small box of matrices; the
  same approach does not survive to dimension four, where the right tool is
  PALP. :func:`equivalent` is therefore not provided rather than provided
  unreliably.
* :func:`lattice_points` scans the bounding box, which is fine for the
  polytopes here and is guarded by ``max_box``; it is not a substitute for a
  real lattice point enumerator on large polytopes.
* Smoothness of the resulting variety, and the existence of a crepant
  resolution, are not decided.
"""

import itertools
import math

import numpy as np

__all__ = [
    "Polytope", "polar", "lattice_points", "is_reflexive",
    "in_generated_lattice", "face_lattice", "f_vector", "dual_face",
    "batyrev_hodge", "twenty_four_cell", "d4_roots", "f4_roots",
    "equilateral_subsets", "cross_polytope", "simplex",
]

_TOL = 1e-7


# ---------------------------------------------------------------------------
# basic lattice-polytope operations
# ---------------------------------------------------------------------------

def _hull(V):
    from scipy.spatial import ConvexHull
    return ConvexHull(np.asarray(V, dtype=float))


def _facet_normals(V):
    """Facet hyperplanes as (m, c) with <m, x> <= c, deduplicated.

    scipy returns one equation per simplex of the triangulated hull, so a
    non-simplicial facet -- an octahedral cell of the 24-cell, for instance --
    appears several times. Deduplication is on the normalised hyperplane.
    """
    h = _hull(V)
    out = {}
    for eq in h.equations:
        n = np.array(eq[:-1], dtype=float)
        c = -float(eq[-1])
        s = np.linalg.norm(n)
        if s < _TOL:
            continue
        key = tuple(np.round(np.concatenate([n / s, [c / s]]), 7))
        out[key] = (n / s, c / s)
    return list(out.values())


def polar(V, exact=True, convention="batyrev"):
    r"""
    The polar dual P* = { y : <x, y> >= -1 for all x in P }, as vertices.

    The vertices of P* are the facet normals of P scaled so that each facet
    reads <m, x> = -1. Returns an integer array when the result is integral,
    which is exactly the condition that P be reflexive with respect to the
    ambient lattice.

    Parameters
    ----------
    V : array of lattice points
        Vertices of P. The origin must be in the interior.
    exact : bool
        If True, raise when the dual is not integral. Set False to inspect a
        non-reflexive case; the 24-cell written in Z^4 is one, and looking at
        the half-integral answer is how one discovers it needs D_4.
    convention : {'batyrev', 'toric'}
        Which sign to use. ``'batyrev'`` is the definition above,
        <x, y> >= -1. ``'toric'`` is <x, y> <= 1, which is what
        :func:`pyCICY.toric.dual` implements. The two results differ by
        y -> -y and nothing else: on the sixteen reflexive polygons this
        function reproduces ``toric.dual`` exactly after that negation, and
        exactly without it on those whose dual is centrally symmetric. Every
        quantity :func:`batyrev_hodge` uses -- lattice point counts of faces
        and of the whole polytope -- is invariant under y -> -y, so the choice
        does not affect Hodge numbers. It does affect whether a printed vertex
        list matches the rest of the package, which is why it is an argument
        rather than a silent decision.
    """
    V = np.asarray(V, dtype=float)
    out = set()
    for n, c in _facet_normals(V):
        if c <= _TOL:
            raise ValueError(
                "the origin is not in the interior: a facet has <m,x> <= %g" % c)
        out.add(tuple(np.round(n / c, 7)))
    A = np.array(sorted(out))
    if convention == "toric":
        A = -A
        A = np.array(sorted(map(tuple, A.tolist())))
    elif convention != "batyrev":
        raise ValueError("convention must be 'batyrev' or 'toric'")
    integral = bool(np.allclose(A, np.round(A), atol=1e-6))
    if exact and not integral:
        raise ValueError(
            "the polar dual is not a lattice polytope in this lattice, so the "
            "polytope is not reflexive with respect to it. Try "
            "in_generated_lattice() first: a polytope can fail to be "
            "reflexive in Z^n and be reflexive in the lattice its own "
            "vertices generate. The 24-cell is exactly that case.")
    return np.round(A).astype(np.int64) if integral else A


def lattice_points(V, max_box=2000000):
    """All lattice points of the polytope, by scanning the bounding box.

    ``max_box`` caps the number of candidates so that a careless call on a
    large polytope fails loudly instead of hanging.
    """
    V = np.asarray(V, dtype=np.int64)
    lo, hi = V.min(axis=0), V.max(axis=0)
    size = int(np.prod(hi - lo + 1))
    if size > max_box:
        raise ValueError(
            "bounding box has %d points, over the max_box of %d. This routine "
            "scans the box; it is not a lattice point enumerator for large "
            "polytopes." % (size, max_box))
    eqs = _hull(V).equations
    A, b = eqs[:, :-1], eqs[:, -1]
    pts = []
    for p in itertools.product(*[range(int(lo[i]), int(hi[i]) + 1)
                                 for i in range(V.shape[1])]):
        x = np.array(p, dtype=float)
        if np.all(A @ x + b <= _TOL):
            pts.append(p)
    return np.array(pts, dtype=np.int64)


def interior_lattice_points(V):
    """Lattice points strictly inside. For a reflexive polytope, just the origin."""
    V = np.asarray(V, dtype=np.int64)
    eqs = _hull(V).equations
    A, b = eqs[:, :-1], eqs[:, -1]
    pts = lattice_points(V)
    return np.array([p for p in pts if np.all(A @ p.astype(float) + b < -_TOL)],
                    dtype=np.int64)


def in_generated_lattice(V):
    r"""
    Rewrite the vertices in a basis of the lattice they generate.

    Returns ``(W, B)`` where ``W`` are the vertices in the new basis and ``B``
    is the basis matrix, so that ``W @ B == V``. The index of the generated
    lattice in Z^n is ``abs(det B)``.

    This is what makes the 24-cell tractable. Written as the D_4 roots
    +-e_i+-e_j it generates the sublattice D_4 of index 2 in Z^4, and it is
    reflexive there and not in Z^4. Reflexivity is a property of the pair
    (polytope, lattice) and the natural lattice for a root polytope is the
    root lattice.
    """
    import sympy as sp
    from sympy.matrices.normalforms import hermite_normal_form

    V = np.asarray(V, dtype=np.int64)
    H = hermite_normal_form(sp.Matrix(V.T.tolist()))
    B = np.array(H.T.tolist(), dtype=np.int64)
    if B.shape[0] != V.shape[1]:
        raise ValueError("the vertices do not span the ambient space")
    Binv = np.array(sp.Matrix(B.tolist()).inv().tolist(), dtype=object)
    W = []
    for v in V:
        row = [sum(sp.Integer(int(v[k])) * Binv[k][c] for k in range(len(v)))
               for c in range(B.shape[1])]
        if any(sp.Rational(x).q != 1 for x in row):
            raise ValueError("change of basis did not produce integers")
        W.append([int(x) for x in row])
    return np.array(W, dtype=np.int64), B


def is_reflexive(V, use_generated_lattice=True):
    """Whether the polytope is reflexive, and with respect to which lattice.

    Returns a dict with ``reflexive``, ``lattice`` (``'ambient'`` or
    ``'generated'``), ``index`` of the generated lattice in Z^n, and the dual
    vertices. Reporting the lattice rather than a bare boolean is the point:
    "is the 24-cell reflexive" has no answer until the lattice is named.
    """
    V = np.asarray(V, dtype=np.int64)
    try:
        D = polar(V, exact=True)
        return {"reflexive": True, "lattice": "ambient", "index": 1,
                "vertices": V, "dual": D}
    except ValueError:
        pass
    if not use_generated_lattice:
        return {"reflexive": False, "lattice": "ambient", "index": 1,
                "vertices": V, "dual": None}
    W, B = in_generated_lattice(V)
    idx = int(abs(round(np.linalg.det(B.astype(float)))))
    try:
        D = polar(W, exact=True)
    except ValueError:
        return {"reflexive": False, "lattice": "generated", "index": idx,
                "vertices": W, "dual": None}
    return {"reflexive": True, "lattice": "generated", "index": idx,
            "vertices": W, "dual": D}


# ---------------------------------------------------------------------------
# faces
# ---------------------------------------------------------------------------

def face_lattice(V):
    """Proper faces by dimension, as frozensets of vertex indices.

    Facets come from the hull; lower faces are the maximal intersections of
    higher ones, filtered by affine rank so that a coincidental intersection
    of the right size but the wrong dimension is discarded.

    Returns ``{dim: [frozenset, ...]}`` for dim = 0 .. n-1.
    """
    V = np.asarray(V, dtype=float)
    n = V.shape[1]
    groups = {}
    h = _hull(V)
    for eq, simp in zip(h.equations, h.simplices):
        nrm = np.array(eq[:-1], dtype=float)
        s = np.linalg.norm(nrm)
        key = tuple(np.round(np.concatenate([nrm / s, [eq[-1] / s]]), 7))
        groups.setdefault(key, set()).update(int(i) for i in simp)
    facets = [frozenset(s) for s in groups.values()]
    out = {n - 1: facets}
    cur = facets
    for d in range(n - 2, -1, -1):
        cand = set()
        for a, b in itertools.combinations(cur, 2):
            inter = a & b
            if len(inter) < d + 1:
                continue
            P = V[sorted(inter)]
            if np.linalg.matrix_rank(P - P.mean(axis=0), tol=1e-8) == d:
                cand.add(frozenset(inter))
        cand = [f for f in cand if f and not any(f < g for g in cand)]
        out[d] = cand
        cur = cand
    return out


def f_vector(V):
    """Numbers of faces by dimension, 0 up to n-1. For the 24-cell: 24, 96, 96, 24."""
    fl = face_lattice(V)
    return [len(fl[d]) for d in sorted(fl)]


def dual_face(V, D, idx):
    r"""
    The face of D = P* dual to the face of P spanned by ``V[idx]``.

    Defined by th* = { y in P* : <x, y> = -1 for all x in th }. Dimensions
    satisfy dim th + dim th* = n - 1, which :func:`batyrev_hodge` relies on
    and the tests check.

    Returns the indices into ``D`` of the vertices of th*.
    """
    V = np.asarray(V, dtype=float)
    D = np.asarray(D, dtype=float)
    face = V[sorted(idx)]
    out = []
    for j, y in enumerate(D):
        if np.all(np.abs(face @ y + 1.0) < 1e-6):
            out.append(j)
    return frozenset(out)


def _relative_interior_count(V, idx, pts):
    """Lattice points of ``pts`` in the relative interior of the given face."""
    V = np.asarray(V, dtype=float)
    if not idx:
        return 0
    P = V[sorted(idx)]
    c = P.mean(axis=0)
    A = P - c
    r = np.linalg.matrix_rank(A, tol=1e-8)
    if r == 0:
        return 0                      # a vertex has empty relative interior
    _, _, Wt = np.linalg.svd(A)
    B = Wt[:r]
    Q = A @ B.T
    if r == 1:
        lim = float(np.abs(Q).max())
        count = 0
        for p in pts:
            d = np.asarray(p, dtype=float) - c
            if np.linalg.norm(d - (d @ B.T) @ B) > 1e-7:
                continue
            t = float((d @ B.T)[0])
            if -lim + 1e-9 < t < lim - 1e-9:
                count += 1
        return count
    hull = _hull(Q)
    A2, b2 = hull.equations[:, :-1], hull.equations[:, -1]
    count = 0
    for p in pts:
        d = np.asarray(p, dtype=float) - c
        if np.linalg.norm(d - (d @ B.T) @ B) > 1e-7:
            continue
        if np.all(A2 @ (d @ B.T) + b2 < -1e-9):
            count += 1
    return count


# ---------------------------------------------------------------------------
# Batyrev
# ---------------------------------------------------------------------------

def batyrev_hodge(V, verbose=False):
    r"""
    Hodge numbers of the Calabi-Yau hypersurface of a reflexive polytope.

    Implemented for dimension 4, where the hypersurface is a threefold, and
    dimension 3, where it is a K3 and the answer is the constant (20, 0)
    with no computation to do. Dimension is taken from the vertices.

    Returns a dict with ``h11``, ``h21``, ``euler``, the lattice point counts,
    and the two correction sums, so that a caller can see which terms actually
    contributed rather than only the total.

    **Which polytope to pass.** ``V`` is Delta, the Newton polytope, whose
    lattice points index the monomials of the defining equation. Its polar
    Delta* is the fan polytope, whose vertices are the rays of the fan. The
    two are easy to interchange and doing so transposes the Hodge numbers, so
    the failure is silent unless one happens to know the answer. The quintic
    is the case to calibrate against: :func:`simplex` (4) is the *fan*
    polytope of P^4, with 6 lattice points, and passing it directly returns
    (101, 1) -- the mirror quintic. The quintic itself is
    ``batyrev_hodge(polar(simplex(4)))``, which returns (1, 101) from a Delta
    with 126 lattice points, one for each quintic monomial in five variables.
    That the count is C(9,4) = 126 is the check that the right polytope is in
    hand.

    For a self-dual polytope such as the 24-cell the distinction is invisible,
    which is convenient there and is exactly why it needs stating here.

    The polytope must be reflexive in the lattice of the coordinates given;
    call :func:`in_generated_lattice` first if it is not. :func:`is_reflexive`
    reports which case applies.
    """
    V = np.asarray(V, dtype=np.int64)
    d = V.shape[1]
    D = polar(V, exact=True)

    if d == 3:
        return {"h11": 20, "h21": 0, "euler": 24, "dimension": 2,
                "note": "a reflexive 3-polytope gives a K3 surface, for which "
                        "h^{1,1} = 20 and chi = 24 always"}
    if d != 4:
        raise NotImplementedError(
            "Batyrev Hodge numbers are implemented for reflexive 3- and "
            "4-polytopes; this one is %d-dimensional" % d)

    LV = lattice_points(V)
    LD = lattice_points(D)
    fV = face_lattice(V)
    fD = face_lattice(D)

    # facet corrections: sum of l* over the codimension-1 faces
    corr1_D = sum(_relative_interior_count(D, f, LD) for f in fD[3])
    corr1_V = sum(_relative_interior_count(V, f, LV) for f in fV[3])

    # codimension-2 corrections, paired through the duality of faces
    corr2_D = 0
    for f in fD[2]:
        li = _relative_interior_count(D, f, LD)
        if li == 0:
            continue
        g = dual_face(D, V, f)
        corr2_D += li * _relative_interior_count(V, g, LV)
    corr2_V = 0
    for f in fV[2]:
        li = _relative_interior_count(V, f, LV)
        if li == 0:
            continue
        g = dual_face(V, D, f)
        corr2_V += li * _relative_interior_count(D, g, LD)

    h11 = len(LD) - 5 - corr1_D + corr2_D
    h21 = len(LV) - 5 - corr1_V + corr2_V
    out = {"h11": int(h11), "h21": int(h21), "euler": int(2 * (h11 - h21)),
           "dimension": 3,
           "l_delta": int(len(LV)), "l_delta_star": int(len(LD)),
           "facet_correction": (int(corr1_V), int(corr1_D)),
           "codim2_correction": (int(corr2_V), int(corr2_D))}
    if verbose:
        out["f_vector"] = f_vector(V)
        out["f_vector_dual"] = f_vector(D)
    return out


# ---------------------------------------------------------------------------
# named polytopes
# ---------------------------------------------------------------------------

def d4_roots():
    """The 24 roots +-e_i +- e_j of D_4. These are the 24-cell of Ali's section 3.1."""
    V = []
    for i, j in itertools.combinations(range(4), 2):
        for si in (1, -1):
            for sj in (1, -1):
                v = [0, 0, 0, 0]
                v[i], v[j] = si, sj
                V.append(v)
    return np.array(V, dtype=np.int64)


def f4_roots():
    """The 48 roots of F_4: the 24 long roots +-e_i+-e_j and 24 short ones.

    The short roots are the 8 vectors +-e_i and the 16 half-integral
    (1/2)(+-1,+-1,+-1,+-1). Returned as a float array, since the short roots
    are not integral in Z^4. The two sets of 24 are each a 24-cell, and they
    are polar duals of each other up to the scaling that puts them at squared
    norms 2 and 1 -- which is the sense in which the 24-cell is self-dual.
    """
    long_ = d4_roots().astype(float)
    short = [list(s * np.eye(4)[i]) for i in range(4) for s in (1.0, -1.0)]
    short += [[0.5 * a, 0.5 * b, 0.5 * c, 0.5 * e]
              for a, b, c, e in itertools.product([1, -1], repeat=4)]
    return np.vstack([long_, np.array(short, dtype=float)])


def twenty_four_cell(normalisation="d4", lattice_basis=True):
    """The 24-cell.

    ``normalisation='d4'`` gives the 24 long roots +-e_i+-e_j, squared norm 2,
    as in section 3.1 of arXiv:2511.10685. ``'unit'`` gives the polar
    normalisation {+-e_i} together with the sixteen (1/2)(+-1,+-1,+-1,+-1), of
    squared norm 1, as in section 2.1 of the same paper; that one is returned
    as floats because it is not integral in Z^4.

    With ``lattice_basis`` the ``'d4'`` vertices are re-expressed in a basis of
    the lattice they generate, which is where the polytope is reflexive.
    """
    if normalisation == "unit":
        short = [list(s * np.eye(4)[i]) for i in range(4) for s in (1.0, -1.0)]
        short += [[0.5 * a, 0.5 * b, 0.5 * c, 0.5 * e]
                  for a, b, c, e in itertools.product([1, -1], repeat=4)]
        return np.array(short, dtype=float)
    if normalisation != "d4":
        raise ValueError("normalisation must be 'd4' or 'unit'")
    V = d4_roots()
    if lattice_basis:
        W, _ = in_generated_lattice(V)
        return W
    return V


def cross_polytope(n):
    """The n-dimensional cross-polytope, vertices +-e_i. Reflexive for all n."""
    return np.array([list(s * np.eye(n, dtype=np.int64)[i])
                     for i in range(n) for s in (1, -1)], dtype=np.int64)


def simplex(n):
    """The reflexive simplex with vertices e_1..e_n and -(e_1+...+e_n)."""
    V = np.eye(n, dtype=np.int64).tolist()
    V.append([-1] * n)
    return np.array(V, dtype=np.int64)


# ---------------------------------------------------------------------------
# combinatorics
# ---------------------------------------------------------------------------

def equilateral_subsets(V, k=4, edge_sq=None):
    r"""
    All k-subsets of the vertices with all pairwise squared distances equal.

    With ``edge_sq`` given, only that squared length is accepted. This is the
    criterion of section 3.1 of arXiv:2511.10685, which asks for four vertices
    of the 24-cell with ||v_i - v_j||^2 = 4 and states that there are 576 of
    them. Brute force over all 10626 four-element subsets gives **48**.

    The 48 is structural rather than accidental: in the norm-2 normalisation
    ||v_i - v_j||^2 = 4 is equivalent to <v_i, v_j> = 0, the twelve diagonals
    of the 24-cell fall into three mutually orthogonal frames of four, and
    each frame carries 2^4 independent sign choices, so 3 * 16 = 48.

    Returns the list of index tuples.
    """
    V = np.asarray(V, dtype=np.int64)
    out = []
    for c in itertools.combinations(range(len(V)), k):
        ds = {int(((V[a] - V[b]) ** 2).sum())
              for a, b in itertools.combinations(c, 2)}
        if len(ds) != 1:
            continue
        if edge_sq is not None and ds.pop() != edge_sq:
            continue
        out.append(c)
    return out
