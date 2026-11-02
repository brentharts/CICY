"""
The cusp fibre and curves on surfaces (Part XIV): the dual complex of W as the
Markov triangulation of the once-punctured torus, the Lucas recurrence of the
n-gon leading singularity as a trace of powers, and two contrasts.

Companion to ``FigueiredoSurfaceology.md`` (``pyCICY.theories.surfaceology``,
``pyCICY.theories.gluons``).  Everything here is exact (sympy integers and
rational functions); each number is reached by at least two routes.

  1. ``dual_quiver``         the Fomin--Shapiro--Thurston quiver of the ideal
                             triangulation Delta(W), read off the A_2 triangles
                             of :mod:`pyCICY.sixsphere`; it is the Markov quiver
  2. ``markov_checks``       mutation, the Markov constant, Laurentness and
                             positivity of every cluster variable to a depth
  3. ``cluster_link``        clusters through one variable: a line, against the
                             hexagon (6 cones) of the Mumford fan
  4. ``lucas_trace_checks``  LS_n = (2-D) Y^n - (a_+^n + a_-^n), and the
                             monodromy blocks at x = -1, 0, 2
  5. ``hat_traces``          the Hat inflation is the same identity at x = 7

What is *not* claimed: that the once-punctured torus carries an amplitude (the
loop-order bookkeeping is not checked), or that the Lucas identity says why the
n-gon carries a companion matrix.  Both are recorded in the paper as open.
"""

import sympy as sp

from . import sixsphere as SS
from . import monotile as MT
from .theories import gluons as G

ARCS = ("a", "b", "c")      # D_a, D_b, D_c : directions e1, e2-e1, e2

# direction of a side of an A_2 triangle (in the lattice basis, up to sign) -> arc
_SIDE_ARC = {(1, 0): "a", (-1, 1): "b", (0, 1): "c"}


def _arc_of(u, v):
    d = (v[0] - u[0], v[1] - u[1])
    for s in (1, -1):
        key = (s * d[0], s * d[1])
        if key in _SIDE_ARC:
            return _SIDE_ARC[key]
    raise ValueError("not an A_2 side: %r" % (d,))


def _ccw(tri):
    """The vertices of a triangle in counter-clockwise order."""
    (x0, y0), (x1, y1), (x2, y2) = tri
    cross = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    return tri if cross > 0 else (tri[0], tri[2], tri[1])


def triangle_side_cycles():
    """Cyclic order of the arcs round each of the two triangles of Delta(W).

    Read off ``SS.a2_triangles((0,0))`` (the lower and upper triangle of the
    unit parallelogram): sides taken counter-clockwise, each mapped to the
    double curve of its direction.  Nothing here is typed in.
    """
    out = {}
    for (name, _), tri in SS.a2_triangles((0, 0)).items():
        v = _ccw(tuple(tri))
        out[name] = tuple(_arc_of(v[i], v[(i + 1) % 3]) for i in range(3))
    return out


def quiver_from_cycles(cycles):
    """FST exchange matrix from the cyclic side order of each triangle."""
    idx = {a: i for i, a in enumerate(ARCS)}
    B = sp.zeros(3, 3)
    for cyc in cycles:
        for i in range(3):
            s, t = idx[cyc[i]], idx[cyc[(i + 1) % 3]]
            B[s, t] += 1
            B[t, s] -= 1
    return B


def dual_quiver():
    """The exchange matrix of the FST quiver of Delta(W).

    One arrow s -> t for each triangle in which t follows s in the
    counter-clockwise order of sides; 3 vertices (arcs), 6 arrows.
    """
    return quiver_from_cycles(triangle_side_cycles().values())


def is_markov(B):
    """Skew-symmetric with every off-diagonal entry of absolute value 2."""
    return (B == -B.T and all(abs(B[i, j]) == 2
                              for i in range(3) for j in range(3) if i != j))


X0 = sp.symbols("x0 x1 x2", positive=True)


def mutate(seed, k, B=None):
    """Cluster mutation of a seed (tuple of 3 rational functions) at k.

    For the Markov quiver the exchange matrix is the same after every
    mutation (the two arrows cancel and re-form), so B is reused."""
    B = dual_quiver() if B is None else B
    col = [B[i, k] for i in range(3)]
    pos = sp.Mul(*[seed[i] ** max(col[i], 0) for i in range(3)])
    neg = sp.Mul(*[seed[i] ** max(-col[i], 0) for i in range(3)])
    new = list(seed)
    new[k] = sp.cancel((pos + neg) / seed[k])
    return tuple(new)


def markov_constant(seed):
    s = seed
    return sp.cancel((s[0] ** 2 + s[1] ** 2 + s[2] ** 2) / (s[0] * s[1] * s[2]))


def _exchange_graph(depth, B):
    seed0 = tuple(X0)
    key = lambda s: frozenset(sp.srepr(e) for e in s)          # noqa: E731
    seen = {key(seed0): seed0}
    layer = [seed0]
    for _ in range(depth):
        nxt = []
        for s in layer:
            for k in range(3):
                m = mutate(s, k, B)
                if key(m) not in seen:
                    seen[key(m)] = m
                    nxt.append(m)
        layer = nxt
    return seen


def _laurent_positive(expr):
    """Denominator a monomial and every numerator coefficient positive."""
    num, den = sp.fraction(sp.cancel(expr))
    if len(sp.Poly(den, *X0).terms()) != 1:
        return False
    return all(c > 0 for c in sp.Poly(num, *X0).coeffs())


def markov_checks(depth=4):
    B = dual_quiver()
    seen = _exchange_graph(depth, B)
    c0 = markov_constant(tuple(X0))
    variables = {sp.srepr(e): e for s in seen.values() for e in s}
    triples, frontier = set(), {(1, 1, 1)}
    for _ in range(depth + 1):
        nxt = set()
        for t in frontier:
            for k in range(3):
                u = list(t)
                u[k] = (t[(k + 1) % 3] ** 2 + t[(k + 2) % 3] ** 2) // t[k]
                nxt.add(tuple(sorted(u)))
        triples |= nxt
        frontier = nxt
    return {
        "B": B.tolist(),
        "markov_quiver": is_markov(B),
        "arrows": int(sum(abs(B[i, j]) for i in range(3) for j in range(i + 1, 3))),
        "seeds": len(seen),
        "constant_preserved": all(sp.cancel(markov_constant(s) - c0) == 0
                                  for s in seen.values()),
        "variables": len(variables),
        "laurent_positive": all(_laurent_positive(e) for e in variables.values()),
        "markov_triples_ok": all(a * a + b * b + c * c == 3 * a * b * c
                                 for a, b, c in triples),
        "markov_triples": sorted(triples),
    }


def fan_link_size():
    """Maximal cones of the Mumford cusp fan through the ray over (0,0)."""
    base = [(i, j) for i in (-1, 0, 1) for j in (-1, 0, 1)]
    cones = set()
    for v in base:
        for tri in SS.a2_triangles(v).values():
            if (0, 0) in tri:
                cones.add(frozenset(tri))
    return len(cones)


def cluster_link(depth=5):
    """Clusters within ``depth`` mutations that contain the variable x0."""
    seen = _exchange_graph(depth, dual_quiver())
    x0 = sp.srepr(X0[0])
    n = sum(1 for s in seen.values() if any(sp.srepr(e) == x0 for e in s))
    return {"depth": depth, "clusters": len(seen), "through_x0": n,
            "line": n == 2 * depth + 1, "hexagon": fan_link_size()}


# ---------------------------------------------------------------------------
# Lucas = traces of powers
# ---------------------------------------------------------------------------

def lucas_trace_checks(nmax=12):
    X, Y = sp.symbols("X Y")
    comp = sp.Matrix([[X, -X * Y], [1, 0]])
    ngon = True
    for n in range(2, nmax + 1):
        p = [sp.Integer(2), X]
        for _ in range(n - 1):
            p.append(sp.expand(X * p[-1] - X * Y * p[-2]))
        ls = G.ngon_leading_singularity(n, x=X, y=Y)
        ngon &= sp.expand(ls - ((2 - G.D) * Y ** n - p[n])) == 0
        ngon &= sp.expand((comp ** n).trace() - p[n]) == 0
    t1, t2 = SS.T1[1:3, 1:3], SS.T2[1:3, 1:3]
    prod = t1 * t2
    per = {}
    for name, M, xv in (("T1", t1, -1), ("T2", t2, 0), ("T1T2", prod, 2)):
        trs = [(M ** n).trace() for n in range(1, nmax + 1)]
        want = [G.lucas_polynomial(n, sp.Integer(xv)) for n in range(1, nmax + 1)]
        per[name] = (trs == want)
    return {
        "nmax": nmax,
        "ngon_power_sum": bool(ngon),
        "blocks": t1 == sp.Matrix([[-1, 1], [-1, 0]]) and t2 == sp.Matrix([[0, -1], [1, 0]]),
        "block_product": prod == sp.Matrix([[1, 1], [0, 1]]),
        "traces": per,
        "orders": (3, 4),
    }


def hat_traces(nmax=8):
    M = sp.Matrix(MT.SUBSTITUTION)
    ok = all((M ** n).trace() == sp.lucas(4 * n) + 1 + (-1) ** n
             for n in range(1, nmax + 1))
    x = sp.Symbol("x")
    return {"charpoly": sp.factor(M.charpoly(x).as_expr()),
            "trace_is_L4n": ok,
            "x": 7,
            "L4": int(sp.lucas(4))}


def _all_checks_old():
    return {"markov": markov_checks(), "link": cluster_link(),
            "lucas": lucas_trace_checks(), "hat": hat_traces()}


# ---------------------------------------------------------------------------
# Loop order: the dual fatgraph of Delta(W)
# ---------------------------------------------------------------------------

def ribbon_faces(rotations):
    """Faces of the ribbon graph whose cubic vertices carry the given cyclic
    orders of arcs; the two ends of arc s are the darts (v, s), (w, s).

    Returns the number of boundary cycles (face permutation = rotation after
    crossing the edge).  The graph here is the theta graph: vertices ``P``,
    ``Q``, edges the arcs.
    """
    verts = list(rotations)
    nxt = {(v, s): (v, rotations[v][(i + 1) % 3])
           for v in verts for i, s in enumerate(rotations[v])}

    def other(d):
        v, s = d
        return next((w, s) for w in verts if w != v)

    seen, faces = set(), 0
    for d in nxt:
        if d in seen:
            continue
        faces += 1
        while d not in seen:
            seen.add(d)
            d = nxt[other(d)]
    return faces


def loop_order():
    """Genus, faces and loops of the dual fatgraph of Delta(W), two ways.

    V = 2 (the triangles), E = 3 (the arcs), F = boundary cycles of the ribbon
    graph.  Loops from the graph, E - V + 1, and from the surface,
    2 g + F - 1 with 2 - 2 g = V - E + F.
    """
    rot = dict(triangle_side_cycles())
    rot = {"P": rot["lower"], "Q": rot["upper"]}
    V, E = 2, 3
    F = ribbon_faces(rot)
    g = (2 - V + E - F) // 2
    # control: the opposite cyclic order at Q is the planar theta graph
    ctl = {"P": rot["P"], "Q": rot["Q"][::-1]}
    Fc = ribbon_faces(ctl)
    return {"V": V, "E": E, "F": F, "genus": g,
            "loops_graph": E - V + 1, "loops_surface": 2 * g + F - 1,
            "external_legs": 0,
            "control_faces": Fc, "control_genus": (2 - V + E - Fc) // 2}


def theta_symanzik():
    """U of the theta graph by the matrix-tree theorem and by spanning trees
    (the softgraph module's two routes), and its Newton polytope."""
    from .theories import softgraph as SG
    gr = SG.Graph([("a", "P", "Q", 0), ("b", "P", "Q", 0), ("c", "P", "Q", 0)],
                  reduce_at="P", name="theta")
    a, b, c = (gr.alpha[k] for k in "abc")
    u_det = sp.expand(gr.U())
    u_trees = sp.expand(gr.U_spanning_trees())
    monos = sp.Poly(u_det, a, b, c).monoms()
    return {"U": u_det, "routes_agree": sp.expand(u_det - u_trees) == 0,
            "equals": sp.expand(u_det - (a * b + b * c + c * a)) == 0,
            "trees": len(monos), "newton": sorted(monos),
            "loops": gr.loops}


# ---------------------------------------------------------------------------
# F-polynomials and g-vectors with principal coefficients; curves as slopes
# ---------------------------------------------------------------------------

YS = sp.symbols("y0 y1 y2", positive=True)
_ARC_SLOPE = {"a": (1, 0), "b": (-1, 1), "c": (0, 1)}      # directions e1, e2-e1, e2


def _canon(v):
    return v if (v[0], v[1]) > (0, 0) and (v[0] > 0 or (v[0] == 0 and v[1] > 0)) else (-v[0], -v[1])


def _det(u, v):
    return u[0] * v[1] - u[1] * v[0]


def _mutate_matrix(Bt, k):
    """Matrix mutation of the extended 6 x 3 exchange matrix at column k."""
    out = Bt.copy()
    for i in range(Bt.rows):
        for j in range(3):
            if i == k or j == k:
                out[i, j] = -Bt[i, j]
            else:
                out[i, j] = Bt[i, j] + (abs(Bt[i, k]) * Bt[k, j] + Bt[i, k] * abs(Bt[k, j])) // 2
    return out


def _mutate_pc(seed, slopes, Bt, k):
    """Mutate (cluster variables in Z[x^+-, y], slope vectors, extended matrix)
    at k.  The frozen variables are y_0, y_1, y_2 (rows 3..5 of Bt), so the
    exchange relation is x_k x_k' = prod x~_i^[b~_ik]+ + prod x~_i^[-b~_ik]+ ."""
    full = list(seed) + list(YS)
    col = [Bt[i, k] for i in range(6)]
    pos = sp.Mul(*[full[i] ** max(col[i], 0) for i in range(6)])
    neg = sp.Mul(*[full[i] ** max(-col[i], 0) for i in range(6)])
    new = list(seed)
    new[k] = sp.expand(sp.cancel((pos + neg) / seed[k]))
    u, v = [slopes[i] for i in range(3) if i != k]
    plus, minus = _canon((u[0] + v[0], u[1] + v[1])), _canon((u[0] - v[0], u[1] - v[1]))
    ns = list(slopes)
    ns[k] = minus if _canon(slopes[k]) == plus else plus
    return tuple(new), tuple(ns), _mutate_matrix(Bt, k)


def _g_and_F(expr, B):
    """(F, g, homogeneous?) of a cluster variable X in Z[x^+-, y]."""
    gens = tuple(X0) + tuple(YS)
    poly = sp.Poly(sp.expand(expr * sp.Mul(*X0) ** 40), *gens)     # clear x-denominators
    gs = set()
    for mon in poly.monoms():
        a = [mon[i] - 40 for i in range(3)]
        c = mon[3:]
        gs.add(tuple(a[i] - sum(B[i, j] * c[j] for j in range(3)) for i in range(3)))
    F = sp.expand(expr.subs({x: 1 for x in X0}))
    return F, (next(iter(gs)) if len(gs) == 1 else None), len(gs) == 1


def principal_checks(depth=4):
    """F-polynomials, g-vectors and slopes of every cluster variable within
    ``depth`` mutations of the triangulation Delta(W)."""
    B = dual_quiver()
    s0 = tuple(X0)
    sl0 = tuple(_ARC_SLOPE[a] for a in ARCS)
    key = lambda s: frozenset(sp.srepr(e) for e in s)             # noqa: E731
    Bt0 = B.col_join(sp.eye(3))
    seeds = {key(s0): (s0, sl0)}
    layer = [(s0, sl0, Bt0)]
    exch = set()
    for _ in range(depth):
        nxt = []
        for s, sl, Bt in layer:
            for k in range(3):
                m, ms, Bm = _mutate_pc(s, sl, Bt, k)
                exch.add((frozenset([_canon(sl[k]), _canon(ms[k])])))
                if key(m) not in seeds:
                    seeds[key(m)] = (m, ms)
                    nxt.append((m, ms, Bm))
        layer = nxt
    variables = {}                                    # slope -> expression
    consistent = True
    for s, sl in seeds.values():
        for e, v in zip(s, sl):
            v = _canon(v)
            if v in variables and sp.expand(variables[v] - e) != 0:
                consistent = False
            variables.setdefault(v, e)
    injective = len({sp.srepr(e) for e in variables.values()}) == len(variables)
    F_ok = g_ok = pos_ok = ymax_ok = tilde_ok = True
    markov_ok = True
    gv = {}
    for v, e in variables.items():
        F, g, homog = _g_and_F(e, B)
        g_ok &= homog
        gv[v] = g
        P = sp.Poly(F, *YS)
        F_ok &= P.coeff_monomial(1) == 1
        pos_ok &= all(c > 0 for c in P.coeffs())
        mons = P.monoms()
        top = tuple(max(m[i] for m in mons) for i in range(3))
        ymax_ok &= top in mons
        yhat = [YS[j] * sp.Mul(*[X0[i] ** B[i, j] for i in range(3)]) for j in range(3)]
        rebuilt = sp.expand(sp.Mul(*[X0[i] ** g[i] for i in range(3)]) * F.subs(dict(zip(YS, yhat)), simultaneous=True))
        tilde_ok &= sp.expand(rebuilt - e) == 0
    unimod = all(abs(sp.Matrix([gv[_canon(v)] for v in sl]).det()) == 1
                 for _, sl in seeds.values())
    compat = all(abs(_det(_canon(u), _canon(v))) == 1
                 for _, sl in seeds.values() for u, v in
                 [(sl[0], sl[1]), (sl[0], sl[2]), (sl[1], sl[2])])
    exch_ok = all(abs(_det(*tuple(p))) == 2 for p in exch)
    return {"depth": depth, "seeds": len(seeds), "variables": len(variables),
            "slopes_consistent": consistent, "slopes_injective": injective, "g_homogeneous": g_ok,
            "F_constant_term_one": F_ok, "F_positive": pos_ok,
            "F_max_monomial": ymax_ok, "separation_formula": tilde_ok,
            "g_bases_unimodular": unimod, "compatible_det_one": compat,
            "exchange_det_two": exch_ok, "exchange_pairs": len(exch)}


def crossings_unbounded(bound=20, det=2):
    """Number of arcs (primitive (r,s), up to sign) in the box |r| <= bound
    that cross the arc (1,0) exactly det-1 times (|s| = det)."""
    from math import gcd
    return sum(1 for r in range(-bound, bound + 1) if gcd(abs(r), det) == 1)


def all_checks():
    out = _all_checks_old()
    out["loop"] = loop_order()
    out["theta"] = theta_symanzik()
    out["principal"] = principal_checks()
    out["crossing_counts"] = (crossings_unbounded(10), crossings_unbounded(20))
    return out
