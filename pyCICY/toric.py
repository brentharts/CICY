r"""
pyCICY.toric -- Newton polygons, reflexive polygons and local Calabi-Yau data.

Scope
-----
The rest of pyCICY deals with *compact* Calabi-Yau threefolds realised as
complete intersections in products of projective spaces. This module deals
with the *local* (non-compact, toric) Calabi-Yau threefolds K_S, S a toric
surface. The two settings share their combinatorial backbone -- a reflexive
lattice polytope -- and it is that shared backbone which is implemented here.
The bridge is made explicit and narrow rather than overclaimed; see
:func:`anticanonical_cicy` and the note at the end of this docstring.

Newton polygons and the lattice dictionary
------------------------------------------
A local toric Calabi-Yau K_S is specified by a two-dimensional reflexive
polygon P, the *toric diagram*: the rays of the fan of S are the vertices of
P. Mirror symmetry replaces K_S by its mirror curve

    Sigma :  sum_{(m,n) in P} c_{mn} x^m y^n = 0,     x, y in C^* ,

a plane curve whose Newton polygon is P. Quantizing Sigma by promoting
x = e^u, y = e^v with [u, v] = i hbar turns it into a difference operator.

Sugimoto, "Calabi-Yau geometry and electrons on 2d lattices",
arXiv:1701.01561 (Phys. Rev. D 95, 086004), together with the earlier
observation of Hatsuda, Katsura and Tachikawa for local F_0, identifies that
difference operator with the Hamiltonian of an electron hopping on a
two-dimensional lattice in a magnetic field:

    lattice points of P     <->  hopping vectors
    coefficients c_{mn}     <->  hopping amplitudes
    hbar = 2 pi Phi         <->  magnetic flux per unit cell

Under this dictionary local F_0, whose polygon is a square, gives the square
lattice and hence Harper's equation and Hofstadter's butterfly; local B_3,
the three-point blow-up of local P^2 (that is, dP_3), whose polygon is a
hexagon, gives the triangular lattice. :func:`hoppings` and
:func:`from_hoppings` implement the dictionary in both directions;
:mod:`pyCICY.quantum_curve` does the quantization.

The classification
------------------
There are exactly sixteen reflexive polygons up to GL(2,Z). This is not
assumed here: :func:`enumerate_reflexive` derives them by brute force and the
test suite checks that the sixteen named representatives in :data:`NAMED`
are pairwise inequivalent and exhaust the list. Five of them are smooth --
P^2, F_0, dP_1 (= F_1), dP_2 and dP_3 (= B_3) -- and these are detected
rather than tabulated, by the criterion that a reflexive polygon has a smooth
fan exactly when every edge has lattice length one, i.e. when its vertex
count equals its boundary point count.

Every reflexive polygon satisfies the "twelve theorem",

    #(boundary points of P) + #(boundary points of P*) = 12 ,

which :func:`twelve` evaluates and the tests check for all sixteen. The
degree of the corresponding surface is K^2 = #(boundary points of P*).

Relation to the compact CICY side
---------------------------------
The honest connection is this. The mirror curve of K_S, read at the level of
its Newton polygon rather than as a quantum object, is an anticanonical curve
of S. When S is itself a product of projective spaces that curve is a
complete intersection Calabi-Yau *one-fold* -- an elliptic curve -- and so is
an object the existing :class:`pyCICY.CICY` machinery already understands:

    local P^2  ->  [[2, 3]]           the plane cubic
    local F_0  ->  [[1, 2], [1, 2]]   the (2,2) curve in P^1 x P^1

:func:`anticanonical_cicy` returns these and ``None`` for the thirteen
polygons whose surface is not a product of projective spaces. That is the
whole of the overlap; the local geometries are not CICY threefolds and this
module does not pretend otherwise.
"""

import itertools as it
from fractions import Fraction

__all__ = [
    "NAMED", "LOCAL_CY",
    "convex_hull", "lattice_points", "interior_points", "boundary_points",
    "is_convex_position", "dual", "is_reflexive", "twelve",
    "normal_form", "gl2z_map", "equivalent",
    "enumerate_reflexive", "classify", "is_smooth", "degree",
    "hoppings", "from_hoppings", "vertex_hoppings",
    "is_centrally_symmetric", "reflect",
    "bipartite_functional", "is_bipartite",
    "polygon", "ALIASES", "verify_named",
    "dual_name", "anticanonical_cicy", "describe",
]


# ------------------------------------------------------------ named polygons
#
# Vertices are listed counter-clockwise. The five smooth cases carry their
# standard fan; the rest are canonical representatives produced by
# normal_form(). Names follow the local Calabi-Yau K_S they define.

NAMED = {
    # --- the five smooth toric del Pezzo surfaces (vertices == boundary) ---
    "P2":  [(-1, -1), (1, 0), (0, 1)],
    "F0":  [(-1, 0), (0, -1), (1, 0), (0, 1)],
    "F1":  [(-1, -1), (0, -1), (1, 0), (0, 1)],
    "dP2": [(-1, -1), (0, -1), (1, 0), (0, 1), (-1, 0)],
    "B3":  [(-1, 0), (0, -1), (1, -1), (1, 0), (0, 1), (-1, 1)],
    # --- the eleven singular ones, named <shape><boundary count> ---
    "T4":  [(-1, -2), (1, 0), (0, 1)],
    "T6":  [(-2, -3), (1, 0), (0, 1)],
    "T8":  [(-2, -1), (2, -1), (0, 1)],
    "T9":  [(-2, -1), (1, -1), (1, 2)],
    "Q5":  [(-1, -1), (1, -1), (0, 1), (-1, 0)],
    "Q6":  [(-1, -2), (1, 0), (0, 1), (-1, 0)],
    "Q7":  [(-1, -2), (1, 0), (0, 1), (-1, 1)],
    "Q8a": [(-2, -1), (0, -1), (1, 0), (1, 2)],
    "Q8b": [(-1, -1), (1, -1), (1, 1), (-1, 1)],
    "P6":  [(-1, -1), (1, -1), (1, 0), (0, 1), (-1, 0)],
    "P7":  [(-1, -1), (1, -1), (1, 1), (0, 1), (-1, 0)],
}

# Alternative names in common use. T4, T6 and T9 are the weighted projective
# planes P(1,1,2), P(1,2,3) and the triple of P^2; T9 is dual to P^2 and Q8b
# is dual to F_0, both of which the tests check rather than assume.
ALIASES = {
    "P112": "T4",
    "P123": "T6",
    "3P2": "T9",
    "dP1": "F1",
    "dP3": "B3",
    "P1xP1": "F0",
}

LOCAL_CY = {
    "P2":  "local P^2",
    "F0":  "local F_0 = P^1 x P^1",
    "F1":  "local F_1 = dP_1",
    "dP2": "local dP_2",
    "B3":  "local B_3 = dP_3, the three-point blow-up of local P^2",
}


def polygon(name):
    """Look up a named polygon, resolving aliases."""
    key = ALIASES.get(name, name)
    if key not in NAMED:
        raise KeyError("unknown polygon {!r}; known: {}".format(
            name, ", ".join(sorted(NAMED))))
    return [tuple(v) for v in NAMED[key]]


# --------------------------------------------------------------- basic geometry

def _cross(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull(points):
    """Vertices of the convex hull, counter-clockwise, no collinear points."""
    pts = sorted(set(map(tuple, points)))
    if len(pts) <= 2:
        return pts
    lower = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def is_convex_position(points):
    """True if every given point is a vertex of the hull."""
    pts = [tuple(p) for p in points]
    h = convex_hull(pts)
    return len(h) == len(set(pts)) and set(h) == set(pts)


def _sides(verts):
    n = len(verts)
    return [(verts[i], verts[(i + 1) % n]) for i in range(n)]


def lattice_points(verts):
    """All integer points of the closed polygon, sorted."""
    verts = [tuple(v) for v in verts]
    xs = [v[0] for v in verts]
    ys = [v[1] for v in verts]
    out = []
    for x in range(min(xs), max(xs) + 1):
        for y in range(min(ys), max(ys) + 1):
            if all(_cross(a, b, (x, y)) >= 0 for a, b in _sides(verts)):
                out.append((x, y))
    return sorted(out)


def interior_points(verts):
    """Integer points strictly inside the polygon."""
    verts = [tuple(v) for v in verts]
    return sorted(p for p in lattice_points(verts)
                  if all(_cross(a, b, p) > 0 for a, b in _sides(verts)))


def boundary_points(verts):
    """Integer points on the boundary of the polygon."""
    verts = [tuple(v) for v in verts]
    inner = set(interior_points(verts))
    return sorted(p for p in lattice_points(verts) if p not in inner)


def dual(verts):
    r"""Vertices of the polar dual P* = {y : <y,x> >= -1 for all x in P}.

    Returns ``None`` when some edge of P does not lie at lattice distance one
    from the origin, which is exactly the failure of reflexivity.
    """
    verts = [tuple(v) for v in verts]
    out = []
    for (x1, y1), (x2, y2) in _sides(verts):
        det = x1 * y2 - x2 * y1
        if det == 0:                       # edge line through the origin
            return None
        a = Fraction(y1 - y2, det)
        b = Fraction(x2 - x1, det)
        if a.denominator != 1 or b.denominator != 1:
            return None
        out.append((int(a), int(b)))
    return convex_hull(out)


def is_reflexive(verts):
    """True if the origin is the unique interior point and P* is a lattice polygon."""
    verts = [tuple(v) for v in verts]
    if not is_convex_position(verts):
        return False
    if interior_points(verts) != [(0, 0)]:
        return False
    return dual(verts) is not None


def twelve(verts):
    """(#boundary of P, #boundary of P*, their sum). The sum is always 12."""
    d = dual(verts)
    if d is None:
        raise ValueError("polygon is not reflexive")
    a = len(boundary_points(verts))
    b = len(boundary_points(d))
    return a, b, a + b


# ------------------------------------------------------------ GL(2,Z) actions

def _apply(M, points):
    (a, b), (c, d) = M
    return [(a * x + b * y, c * x + d * y) for x, y in points]


def _gl2z_small(bound=3):
    """Every integer matrix with entries in [-bound, bound] and determinant +-1."""
    rng = range(-bound, bound + 1)
    for a, b, c, d in it.product(rng, repeat=4):
        if abs(a * d - b * c) == 1:
            yield ((a, b), (c, d))


def _key(verts):
    h = convex_hull(verts)
    flat = sorted(h)
    return (max(max(abs(x), abs(y)) for x, y in h), len(h), flat)


def normal_form(verts, bound=3):
    """A canonical GL(2,Z) representative of the polygon.

    Chosen to minimise the largest coordinate and then lexicographically, so
    the output is deterministic and idempotent. This plays the same role for
    Newton polygons that :func:`pyCICY.transitions.normal_form` plays for
    configuration matrices.
    """
    verts = convex_hull([tuple(v) for v in verts])
    best, best_key = verts, _key(verts)
    for M in _gl2z_small(bound):
        img = convex_hull(_apply(M, verts))
        k = _key(img)
        if k < best_key:
            best, best_key = img, k
    return best


def gl2z_map(P, Q):
    """A matrix in GL(2,Z) carrying polygon P onto polygon Q, or ``None``.

    Any such matrix must send a lattice basis drawn from P to a lattice basis
    drawn from Q, so one basis of P is fixed and all bases of Q are tried.
    """
    P = convex_hull([tuple(v) for v in P])
    Q = convex_hull([tuple(v) for v in Q])
    if len(P) != len(Q):
        return None
    lp = [p for p in lattice_points(P) if p != (0, 0)]
    lq = [p for p in lattice_points(Q) if p != (0, 0)]
    if len(lp) != len(lq):
        return None

    def bases(pts):
        for u, v in it.permutations(pts, 2):
            if abs(u[0] * v[1] - u[1] * v[0]) == 1:
                yield u, v

    try:
        a, b = next(bases(lp))
    except StopIteration:
        return None
    det = a[0] * b[1] - a[1] * b[0]
    # inverse of [a b] as columns, times det
    inv = ((b[1], -b[0]), (-a[1], a[0]))
    for c, d in bases(lq):
        cols = ((c[0], d[0]), (c[1], d[1]))
        M = []
        ok = True
        for i in range(2):
            row = []
            for j in range(2):
                s = Fraction(cols[i][0] * inv[0][j] + cols[i][1] * inv[1][j], det)
                if s.denominator != 1:
                    ok = False
                    break
                row.append(int(s))
            if not ok:
                break
            M.append(tuple(row))
        if not ok:
            continue
        M = tuple(M)
        if abs(M[0][0] * M[1][1] - M[0][1] * M[1][0]) != 1:
            continue
        if set(_apply(M, P)) == set(Q):
            return M
    return None


def equivalent(P, Q):
    """True if the two polygons agree up to GL(2,Z)."""
    return gl2z_map(P, Q) is not None


# ------------------------------------------------------------- classification

def enumerate_reflexive(triangle_box=5, box=3):
    """Brute-force every reflexive polygon up to GL(2,Z). Returns sixteen.

    Triangles need a wider search box than the rest because the weighted
    projective planes P(1,2,3) and its relatives are elongated, so the two
    ranges are separated to keep the enumeration tractable.
    """
    big = [(x, y) for x in range(-triangle_box, triangle_box + 1)
           for y in range(-triangle_box, triangle_box + 1) if (x, y) != (0, 0)]
    small = [(x, y) for x in range(-box, box + 1)
             for y in range(-box, box + 1) if (x, y) != (0, 0)]
    found = []
    for k, pool in ((3, big), (4, small), (5, small), (6, small)):
        for sub in it.combinations(pool, k):
            xs = [p[0] for p in sub]
            ys = [p[1] for p in sub]
            if max(xs) - min(xs) > 4 or max(ys) - min(ys) > 4:
                continue
            h = convex_hull(list(sub))
            if len(h) != k or set(h) != set(sub):
                continue
            if not is_reflexive(h):
                continue
            if not any(equivalent(h, g) for g in found):
                found.append(normal_form(h))
    return found


def is_smooth(verts):
    """True if the fan over the polygon is smooth.

    For a reflexive polygon this happens exactly when every edge has lattice
    length one, that is when the vertex count equals the boundary count.
    """
    verts = convex_hull([tuple(v) for v in verts])
    return len(verts) == len(boundary_points(verts))


def degree(verts):
    """K^2 of the surface, equal to the number of boundary points of P*."""
    d = dual(verts)
    if d is None:
        raise ValueError("polygon is not reflexive")
    return len(boundary_points(d))


def is_centrally_symmetric(verts):
    s = set(convex_hull([tuple(v) for v in verts]))
    return all((-x, -y) in s for x, y in s)


def reflect(verts):
    """The polygon reflected through the origin, (m,n) -> (-m,-n).

    The Newton-polygon shadow of orientation reversal. A centrally symmetric
    polygon is fixed by it.
    """
    return convex_hull([(-x, -y) for x, y in verts])


def bipartite_functional(verts):
    r"""A linear functional odd on every hopping vector, or ``None``.

    The lattice model attached to P is bipartite exactly when some
    f in (Z/2)^2 satisfies f(v) = 1 mod 2 for every lattice point v of P
    other than the origin: the sign (-1)^{f(site)} then anticommutes with the
    Hamiltonian and forces the spectrum to be symmetric under E -> -E. There
    are only three candidates to test.

    Note that this is *not* the same as central symmetry of P, in either
    direction. B_3 is centrally symmetric and not bipartite, because its
    hopping set contains (1,0), (0,1) and (1,-1) whose parities cannot all be
    made odd; T4 = P(1,1,2) is not centrally symmetric and is bipartite.
    Central symmetry is a statement about P as a set, bipartiteness a
    statement about P modulo 2, and it is the latter that the spectrum sees.
    """
    verts = convex_hull([tuple(v) for v in verts])
    hops = hoppings(verts)
    for f in ((1, 0), (0, 1), (1, 1)):
        if all((f[0] * m + f[1] * n) % 2 == 1 for m, n in hops):
            return f
    return None


def is_bipartite(verts):
    """True when the lattice model attached to the polygon is bipartite."""
    return bipartite_functional(verts) is not None


def classify(verts):
    """A dict of the invariants of a reflexive polygon."""
    verts = convex_hull([tuple(v) for v in verts])
    if not is_reflexive(verts):
        raise ValueError("polygon is not reflexive")
    b, bd, tot = twelve(verts)
    name = None
    for nm, ref in NAMED.items():
        if equivalent(verts, ref):
            name = nm
            break
    return {
        "name": name,
        "vertices": verts,
        "n_vertices": len(verts),
        "n_lattice": len(lattice_points(verts)),
        "n_boundary": b,
        "n_boundary_dual": bd,
        "twelve": tot,
        "smooth": is_smooth(verts),
        "degree": bd,
        "centrally_symmetric": is_centrally_symmetric(verts),
        "bipartite": is_bipartite(verts),
        "dual": dual(verts),
        "local_cy": LOCAL_CY.get(name),
    }


# -------------------------------------------------------- the lattice dictionary

def hoppings(verts, include_origin=False):
    """Hopping vectors of the 2d lattice model attached to the polygon.

    These are the lattice points of P, which index the monomials of the mirror
    curve. The origin corresponds to a constant term, i.e. an on-site energy,
    and is excluded by default so that the output is a set of genuine hops.
    """
    verts = convex_hull([tuple(v) for v in verts])
    pts = lattice_points(verts)
    if not include_origin:
        pts = [p for p in pts if p != (0, 0)]
    return pts


def from_hoppings(vectors):
    """Inverse of :func:`hoppings`: the Newton polygon spanned by hopping vectors."""
    pts = [tuple(v) for v in vectors]
    if (0, 0) not in pts:
        pts = pts + [(0, 0)]
    return convex_hull(pts)


def vertex_hoppings(verts):
    """Only the vertices, i.e. nearest-neighbour hops for the standard models."""
    return convex_hull([tuple(v) for v in verts])


# ------------------------------------------------------------- the CICY bridge

_ANTICANONICAL = {
    "P2": [[2, 3]],
    "F0": [[1, 2], [1, 2]],
}


def anticanonical_cicy(name_or_polygon):
    """Configuration matrix of the anticanonical curve, when it is a CICY.

    The mirror curve of K_S is an anticanonical curve of S. If S is a product
    of projective spaces that curve is a complete intersection Calabi-Yau
    one-fold and can be handed to :class:`pyCICY.CICY`; the plane cubic for
    local P^2 and the (2,2) curve for local F_0. For the other fourteen
    reflexive polygons the surface is not such a product and ``None`` is
    returned. Nothing wider than this is claimed: the local geometries here
    are not themselves CICY threefolds.
    """
    if isinstance(name_or_polygon, str):
        name = name_or_polygon
    else:
        name = classify(name_or_polygon)["name"]
    cfg = _ANTICANONICAL.get(name)
    return [row[:] for row in cfg] if cfg else None


def dual_name(name):
    """Name of the polar dual of a named polygon.

    Polar duality is an involution on the sixteen. Four are self-dual --
    B_3, T6, Q6 and P6 -- and the rest fall into six swapped pairs, among
    them P^2 <-> T9 and F_0 <-> Q8b.
    """
    d = dual(polygon(name))
    for nm in NAMED:
        if equivalent(d, polygon(nm)):
            return nm
    return None


def describe(verts):
    """One-line human summary, in the spirit of :func:`pyCICY.viz.describe`."""
    c = classify(verts)
    tag = c["name"] or "?"
    kind = "smooth" if c["smooth"] else "singular"
    return ("{}  {}-gon  bdry={}  K^2={}  {}".format(
        tag, c["n_vertices"], c["n_boundary"], c["degree"], kind))


def verify_named():
    """Check :data:`NAMED` against the brute-force enumeration.

    Returns a dict reporting the counts and any mismatch. This is the reason
    :data:`NAMED` can be a literal table without being a leap of faith, and it
    is what ``tests/test_toric.py`` calls. It takes a few seconds, so it is
    never run at import time.
    """
    full = enumerate_reflexive()
    named = {nm: convex_hull([tuple(v) for v in vs]) for nm, vs in NAMED.items()}
    missing = [g for g in full
               if not any(equivalent(g, v) for v in named.values())]
    dupes = []
    items = sorted(named.items())
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            if equivalent(items[i][1], items[j][1]):
                dupes.append((items[i][0], items[j][0]))
    return {
        "n_enumerated": len(full),
        "n_named": len(named),
        "missing_from_named": missing,
        "duplicate_names": dupes,
        "ok": len(full) == len(named) == 16 and not missing and not dupes,
    }
