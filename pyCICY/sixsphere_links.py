r"""
pyCICY.sixsphere_links -- what the six-sphere's finite layer shares with the
rest of the package, computed.

:mod:`pyCICY.sixsphere` and :mod:`pyCICY.sixsphere_search` compute the finite
layer of Alpoge's complex structure on S^6. This module records three exact
contacts with modules written afterwards, each reached by two routes, and one
place where a neighbouring module correctly refuses.

1. **The cusp component and the Mahler module** (:func:`b3_facts`). The cusp
   fibre's component is the degree-six del Pezzo surface, the toric surface
   whose polygon is B3. Its normalised volume V(B3) (the anticanonical
   degree K^2) is computed from the polytope module both by triangulation and
   by Ehrhart counts, and K^2 is computed again from the fan by Noether's
   formula K^2 = 12 - (number of rays) and from the self-intersections of the
   toric divisors. All routes give 6, the number of kites per hexagon and of
   (-1)-curves on the component.

2. **The elliptic fibration and the K3-lattice module** (:func:`kodaira_facts`,
   :func:`root_lattice_facts`). A unipotent cusp of type I_1 together with
   local monodromies of orders 3 and 4 fixes the Kodaira types by an Euler
   number count; the fibres' root lattices are then embedded in E8 explicitly
   from the 240 roots (an integral sublattice with its discriminant) and,
   independently, decided over Q by :func:`pyCICY.k3lattice.embeds`.

3. **The central-charge module refuses** (:func:`central_charge_guard`). The
   charges of :mod:`pyCICY.centralcharge` need a class w with w^3 > 0. A
   space with H^2 = 0 has none, and S^6 has H^2 = 0 (so it is not Kaehler,
   and the module's own guard says so).

What is not claimed: that the Jacobian of the six-sphere's fibration is the
rational elliptic surface with those fibres beyond the Euler count (existence
is Miranda-Persson territory and is not decided here); anything integral for
the K3-lattice module, which decides existence over Q only (the integral
statement is the explicit root construction, on E8 only).
"""

from fractions import Fraction
from itertools import product

import numpy as np

from . import k3lattice as K
from . import polytope as P
from . import toric

__all__ = ["KODAIRA", "b3_facts", "kodaira_facts", "root_lattice_facts",
           "central_charge_guard", "e8_roots"]

# Kodaira fibre types: name -> (Euler number, order of monodromy, root lattice)
# order 0 means unipotent / infinite order (I_n); root lattice as (type, rank).
KODAIRA = {
    "II": (2, 6, None), "III": (3, 4, ("A", 1)), "IV": (4, 3, ("A", 2)),
    "I0*": (6, 2, ("D", 4)), "IV*": (8, 3, ("E", 6)),
    "III*": (9, 4, ("E", 7)), "II*": (10, 6, ("E", 8)),
}


def _i_n(n):
    return (n, 0, ("A", n - 1) if n >= 2 else None)


# ---------------------------------------------------------------------------
# 1. the degree-six del Pezzo surface
# ---------------------------------------------------------------------------

def b3_facts():
    """K^2 of the cusp component, by four routes, and the Mahler data."""
    V = np.array(toric.polygon("B3"), dtype=np.int64)
    vp = P.volume_product(V, cross_check=True)      # triangulation + Ehrhart
    rays = [tuple(int(x) for x in v) for v in V]
    # order the rays cyclically by angle
    import math
    rays.sort(key=lambda v: math.atan2(v[1], v[0]))
    n = len(rays)
    selfint = []
    for i in range(n):
        a, b, c = rays[i - 1], rays[i], rays[(i + 1) % n]
        s = (a[0] + c[0], a[1] + c[1])
        # v_{i-1} + v_{i+1} = -(D_i^2) v_i
        k = None
        for t in range(-5, 6):
            if (-t * b[0], -t * b[1]) == s:
                k = t
        selfint.append(k)
    return {
        "volume": P.normalized_volume(V),
        "volume_ehrhart": P.ehrhart_normalized_volume(V)[0],
        "boundary_points": len(toric.boundary_points(V)),
        "dual_volume": vp["V_dual"],
        "product": vp["product"],
        "symmetric": vp["symmetric"],
        "bound_symmetric": P.mahler_bound(2, True),
        "bound_general": P.mahler_bound(2),
        "equality": vp["equality"],
        "rays": n,
        "K2_noether": 12 - n,
        "K2_blowup": 9 - 3,                  # P^2 blown up at three points
        "self_intersections": selfint,
        "sum_self_intersections": sum(selfint),
    }


# ---------------------------------------------------------------------------
# 2. Kodaira types and root lattices
# ---------------------------------------------------------------------------

def kodaira_facts(m1=3, m2=4, cusp_n=1):
    """All Kodaira-type triples compatible with the local monodromy orders.

    A fibre of monodromy order 3 is IV or IV*, of order 4 III or III*, and
    the cusp is I_n. The Euler numbers of the singular fibres of an elliptic
    surface over P^1 sum to 12 chi(O); the list of (types, total) with total
    in {12, 24} is returned together with the Euler total of every triple.
    """
    by_order = {}
    for name, (e, order, _) in KODAIRA.items():
        by_order.setdefault(order, []).append(name)
    triples = []
    for a, b in product(by_order[m1], by_order[m2]):
        total = KODAIRA[a][0] + KODAIRA[b][0] + cusp_n
        triples.append((a, b, "I%d" % cusp_n, total))
    allowed = [t for t in triples if t[3] % 12 == 0]
    return {"all": triples, "allowed": allowed}


def e8_roots():
    """The 240 roots of E8 in simple-root coordinates, from the Cartan matrix
    by closure under simple reflections. Returns (roots, G)."""
    G = np.array(K.cartan_E8(), dtype=int)
    n = G.shape[0]
    roots = {tuple(int(x) for x in np.eye(n, dtype=int)[i]) for i in range(n)}
    frontier = list(roots)
    while frontier:
        new = []
        for v in frontier:
            v = np.array(v)
            for i in range(n):
                w = v.copy()
                w[i] -= int(v @ G[:, i])
                t = tuple(int(x) for x in w)
                if t not in roots:
                    roots.add(t)
                    new.append(t)
        frontier = new
    return sorted(roots), G


def _gram_of_lattice(vectors, G):
    """Gram matrix (sympy) of the Z-span of integer vectors, in a Hermite
    basis; also returns the rank."""
    import sympy as sp
    from sympy.matrices.normalforms import hermite_normal_form
    M = sp.Matrix(vectors).T           # columns are the vectors
    H = hermite_normal_form(M)          # column-style basis of the same lattice
    B = H
    Gs = sp.Matrix(G.tolist())
    gram = (B.T * Gs * B)
    return gram, B.shape[1]


def root_lattice_facts():
    """E7 and E6 + A1 inside E8, explicitly and over Q."""
    import sympy as sp
    roots, G = e8_roots()
    n = G.shape[0]
    ip = lambda u, v: int(np.array(u) @ G @ np.array(v))
    alpha = tuple(int(x) for x in np.eye(n, dtype=int)[0])
    beta = tuple(int(x) for x in np.eye(n, dtype=int)[2])      # adjacent: -1
    assert ip(alpha, beta) == -1
    e7_roots = [r for r in roots if ip(r, alpha) == 0]
    e6_roots = [r for r in roots if ip(r, alpha) == 0 and ip(r, beta) == 0]
    out = {"n_roots_E8": len(roots), "n_roots_E7": len(e7_roots),
           "n_roots_E6": len(e6_roots)}
    g7, r7 = _gram_of_lattice(e7_roots, G)
    g6, r6 = _gram_of_lattice(e6_roots, G)
    out.update({"rank_E7": r7, "disc_E7": int(g7.det()),
                "rank_E6": r6, "disc_E6": int(g6.det())})
    # E6 + A1: A1 = <alpha>, orthogonal to E6 because alpha lies in the A2
    # spanned by alpha, beta, whose orthogonal complement is E6
    out["A1_orthogonal_to_E6"] = all(ip(r, alpha) == 0 for r in e6_roots)
    out["disc_E6_plus_A1"] = out["disc_E6"] * 2
    out["rank_E6_plus_A1"] = r6 + 1
    # over Q: k3lattice decides the embedding of the Gram matrices
    E8 = K.QSpace.from_gram(G.tolist())
    q7 = K.QSpace.from_gram(g7.tolist())
    q61 = K.QSpace.from_gram(K.direct_sum(g6.tolist(), [[2]]))
    out["Q_embeds_E7"] = bool(K.embeds(q7, E8).embeds)
    out["Q_embeds_E6A1"] = bool(K.embeds(q61, E8).embeds)
    # a control: E7 + <3> has the right rank but the wrong discriminant
    wrong = K.QSpace.from_gram(K.direct_sum(g7.tolist(), [[3]]))
    out["Q_embeds_E7_plus_3"] = bool(K.embeds(wrong, E8).embeds)
    # Picard rank 10 of a rational elliptic surface: trivial lattice
    # U + root lattice, Mordell-Weil rank = 10 - 2 - rank(roots)
    out["mw_rank_IVstar_III_I1"] = 10 - 2 - (6 + 1)
    out["mw_rank_IVstar_II_I2"] = 10 - 2 - (6 + 1)
    out["mw_rank_IIIstar_II_I1"] = 10 - 2 - 7
    return out


# ---------------------------------------------------------------------------
# 3. the central-charge guard
# ---------------------------------------------------------------------------

def central_charge_guard():
    """Run centralcharge on a space with H^2 = 0 and report what happens."""
    from . import centralcharge as CC
    X = CC.CY3(np.zeros((0, 0, 0), dtype=int), [])
    try:
        CC.z_ordinary(CC.Object.point(X), [], [])
    except CC.NotAmple as exc:
        return {"refused": True, "reason": str(exc)}
    return {"refused": False, "reason": None}
